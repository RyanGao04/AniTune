#!/usr/bin/env python
"""使用 iCartoonFace rec_test 官方评估代码评估模型

这个脚本:
1. 加载训练好的模型
2. 在 rectest 数据集上提取特征
3. 保存为 .bin 格式
4. 使用官方评估代码计算检索准确率
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# 添加 experiments/rec_evalution_code 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments' / 'rec_evalution_code'))

from anitune.models import ModelConfig, build_model
from anitune.utils import load_config
from icartoonface_rec_test import IcartoonFaceScore


def load_rectest_images(rectest_info_txt, rectest_image_dir):
    """加载 rectest 图像路径和边界框信息

    Args:
        rectest_info_txt: icartoonface_rectest_info.txt 路径
        rectest_image_dir: 图像目录

    Returns:
        samples: list of (img_path, bbox, class_id)
    """
    samples = []

    with open(rectest_info_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 6:
                # 格式: img_name x1 y1 x2 y2 class_id
                img_name = parts[0]
                bbox = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
                class_id = parts[5]

                img_path = Path(rectest_image_dir) / img_name
                samples.append((str(img_path), bbox, class_id))

    return samples


def extract_features(model, samples, device, image_size=224, batch_size=64):
    """提取所有样本的特征向量

    Args:
        model: 模型
        samples: list of (img_path, bbox, class_id)
        device: 设备
        image_size: 输入图像大小
        batch_size: 批次大小

    Returns:
        features: numpy array of shape (N, D)
    """
    model.eval()

    # 数据预处理（与训练时保持一致）
    transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []

    # 批处理
    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting features"):
        batch_samples = samples[i:i+batch_size]
        batch_images = []

        for img_path, bbox, _ in batch_samples:
            # 加载并裁剪图像
            try:
                img = Image.open(img_path).convert('RGB')
                # 使用边界框裁剪
                x1, y1, x2, y2 = bbox
                img = img.crop((x1, y1, x2, y2))
                # 转换
                img = transform(img)
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # 使用零向量作为占位符
                batch_images.append(torch.zeros(3, image_size, image_size))

        # 转换为 batch tensor
        batch_tensor = torch.stack(batch_images).to(device)

        # 提取特征
        with torch.no_grad():
            # 使用 forward_features 获取 embedding
            if hasattr(model, 'forward_features'):
                features = model.forward_features(batch_tensor)
            else:
                # 对于 ViT，手动提取 cls token
                x = model.patch_embed(batch_tensor)
                x = model._pos_embed(x)
                x = model.blocks(x)
                x = model.norm(x)
                features = x[:, 0]  # cls token

            # L2 归一化
            features = F.normalize(features, p=2, dim=1)
            all_features.append(features.cpu().numpy())

    # 合并所有特征
    all_features = np.vstack(all_features)
    return all_features


def save_features_as_bin(features, output_path):
    """保存特征为 .bin 格式（符合官方评估代码要求）

    Args:
        features: (N, D) numpy array
        output_path: 输出 .bin 文件路径
    """
    # 官方代码期望的格式: 所有特征按行展平
    features_flat = features.flatten()
    features_flat.astype(np.float64).tofile(output_path)
    print(f"Features saved to {output_path}")
    print(f"Shape: {features.shape} -> flattened to {features_flat.shape}")


def main():
    parser = argparse.ArgumentParser(description='iCartoonFace RecTest Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best.pt)')
    parser.add_argument('--config', type=str, default='configs/lora_vitb16_a100_balanced.yaml',
                        help='Path to config file')
    parser.add_argument('--rectest-info', type=str,
                        default='data/icartoonface_rectest_info.txt',
                        help='Path to icartoonface_rectest_info.txt')
    parser.add_argument('--rectest-dir', type=str,
                        default='data/personai_icartoonface_rec/personai_icartoonface_rectest/icartoonface_rectest',
                        help='Path to rectest image directory')
    parser.add_argument('--output-bin', type=str, default=None,
                        help='Output .bin file path (default: auto-generated)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # 自动生成输出路径
    if args.output_bin is None:
        checkpoint_name = Path(args.checkpoint).parent.name
        args.output_bin = f"evaluation_results/{checkpoint_name}_rectest.bin"
        Path(args.output_bin).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("iCartoonFace RecTest Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"RecTest Info: {args.rectest_info}")
    print(f"RecTest Dir: {args.rectest_dir}")
    print(f"Output Bin: {args.output_bin}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # 1. 加载配置
    print("\n[1/5] Loading config...")
    cfg = load_config(args.config)

    # 2. 构建模型
    print("\n[2/5] Building model...")
    model_cfg = ModelConfig(**cfg['model'])
    model = build_model(model_cfg)

    # 3. 加载检查点
    print("\n[3/5] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # 处理不同的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best val accuracy: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Best val accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(args.device)
    model.eval()

    # 4. 加载 rectest 数据
    print("\n[4/5] Loading rectest data...")
    samples = load_rectest_images(args.rectest_info, args.rectest_dir)
    print(f"Loaded {len(samples)} samples")

    # 5. 提取特征
    print("\n[5/5] Extracting features...")
    features = extract_features(model, samples, args.device, batch_size=args.batch_size)
    print(f"Extracted features shape: {features.shape}")

    # 6. 保存为 .bin 文件
    print("\nSaving features to .bin file...")
    save_features_as_bin(features, args.output_bin)

    # 7. 使用官方评估代码计算得分
    print("\n" + "=" * 80)
    print("Running official evaluation...")
    print("=" * 80)

    icartoon_score = IcartoonFaceScore(args.rectest_info, feat_size=len(samples))
    score = icartoon_score.compute_score(args.output_bin)

    print("\n" + "=" * 80)
    print(f"📊 FINAL SCORE: {score:.2f}%")
    print("=" * 80)

    # 保存结果
    result_json = args.output_bin.replace('.bin', '_result.json')
    import json
    with open(result_json, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'score': float(score),
            'num_samples': len(samples),
            'feature_dim': features.shape[1],
        }, f, indent=2)
    print(f"\nResults saved to {result_json}")


if __name__ == '__main__':
    main()
