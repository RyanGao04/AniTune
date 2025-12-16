#!/usr/bin/env python3
"""智能 RecTest 评估 - 自动检测模型配置

自动检测:
1. 模型是否使用 LoRA
2. 类别数量
3. 训练模式
"""

import argparse
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import timm

# 添加 experiments/rec_evalution_code 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments' / 'rec_evalution_code'))

from icartoonface_rec_test import IcartoonFaceScore


def detect_model_config(checkpoint_path):
    """自动检测模型配置

    Returns:
        dict: {
            'num_classes': int,
            'has_lora': bool,
            'lora_rank': int or None
        }
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # 检测类别数
    if 'head.weight' in state_dict:
        num_classes = state_dict['head.weight'].shape[0]
    else:
        num_classes = None

    # 检测 LoRA
    has_lora = any('lora' in k for k in state_dict.keys())

    # 检测 LoRA rank
    lora_rank = None
    if has_lora:
        for key in state_dict.keys():
            if 'lora_A.weight' in key:
                lora_rank = state_dict[key].shape[0]
                break

    return {
        'num_classes': num_classes,
        'has_lora': has_lora,
        'lora_rank': lora_rank
    }


def build_model_from_checkpoint(checkpoint_path, pretrained=False):
    """根据 checkpoint 自动构建模型

    Args:
        checkpoint_path: 检查点路径
        pretrained: 是否使用预训练权重（用于初始化）

    Returns:
        model: 构建好的模型
        config: 检测到的配置
    """
    from anitune.lora import apply_lora_to_attention

    # 检测配置
    config = detect_model_config(checkpoint_path)

    print(f"检测到模型配置:")
    print(f"  类别数: {config['num_classes']}")
    print(f"  LoRA: {'是' if config['has_lora'] else '否'}")
    if config['has_lora']:
        print(f"  LoRA Rank: {config['lora_rank']}")

    # 构建基础模型
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=config['num_classes']
    )

    # 如果有 LoRA，应用 LoRA
    if config['has_lora']:
        print(f"\n应用 LoRA (rank={config['lora_rank']})...")
        apply_lora_to_attention(
            model,
            rank=config['lora_rank'],
            alpha=config['lora_rank'] * 2,  # 假设 alpha = 2 * rank
            dropout=0.0
        )

    return model, config


def load_rectest_images(rectest_info_txt, rectest_image_dir):
    """加载 rectest 图像路径和边界框"""
    samples = []
    with open(rectest_info_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 6:
                img_name = parts[0]
                bbox = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
                class_id = parts[5]
                img_path = Path(rectest_image_dir) / img_name
                samples.append((str(img_path), bbox, class_id))
    return samples


def extract_features(model, samples, device, image_size=224, batch_size=64):
    """提取所有样本的特征向量"""
    model.eval()

    transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting features"):
        batch_samples = samples[i:i+batch_size]
        batch_images = []

        for img_path, bbox, _ in batch_samples:
            try:
                img = Image.open(img_path).convert('RGB')
                x1, y1, x2, y2 = bbox
                img = img.crop((x1, y1, x2, y2))
                img = transform(img)
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                batch_images.append(torch.zeros(3, image_size, image_size))

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            # 提取特征
            if hasattr(model, 'forward_features'):
                features = model.forward_features(batch_tensor)
            else:
                x = model.patch_embed(batch_tensor)
                x = model._pos_embed(x)
                x = model.blocks(x)
                x = model.norm(x)
                features = x[:, 0]  # cls token

            # L2 归一化
            features = F.normalize(features, p=2, dim=1)
            all_features.append(features.cpu().numpy())

    all_features = np.vstack(all_features)
    return all_features


def save_features_as_bin(features, output_path):
    """保存特征为 .bin 格式"""
    features_flat = features.flatten()
    features_flat.astype(np.float64).tofile(output_path)
    print(f"Features saved to {output_path}")
    print(f"Shape: {features.shape} -> flattened to {features_flat.shape}")


def main():
    parser = argparse.ArgumentParser(description='Smart iCartoonFace RecTest Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--rectest-info', type=str,
                        default='data/icartoonface_rectest_info.txt')
    parser.add_argument('--rectest-dir', type=str,
                        default='data/personai_icartoonface_rec/personai_icartoonface_rectest/icartoonface_rectest')
    parser.add_argument('--output-bin', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # 自动生成输出路径
    if args.output_bin is None:
        checkpoint_name = Path(args.checkpoint).parent.name
        args.output_bin = f"evaluation_results/rec_test/{checkpoint_name}_rectest.bin"
        Path(args.output_bin).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Smart iCartoonFace RecTest Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_bin}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # 1. 自动构建模型
    print("\n[1/4] Building model from checkpoint...")
    model, config = build_model_from_checkpoint(args.checkpoint, pretrained=False)

    # 2. 加载检查点
    print("\n[2/4] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Val accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(args.device)
    model.eval()

    # 3. 加载数据
    print("\n[3/4] Loading rectest data...")
    samples = load_rectest_images(args.rectest_info, args.rectest_dir)
    print(f"Loaded {len(samples)} samples")

    # 4. 提取特征
    print("\n[4/4] Extracting features...")
    features = extract_features(model, samples, args.device, batch_size=args.batch_size)
    print(f"Extracted features shape: {features.shape}")

    # 5. 保存
    print("\nSaving features...")
    save_features_as_bin(features, args.output_bin)

    # 6. 评估
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
    with open(result_json, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'score': float(score),
            'num_samples': len(samples),
            'feature_dim': features.shape[1],
            'config': config,
        }, f, indent=2)
    print(f"\nResults saved to {result_json}")


if __name__ == '__main__':
    main()
