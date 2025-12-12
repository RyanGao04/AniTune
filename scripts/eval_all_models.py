#!/usr/bin/env python
"""多模型评估脚本

评估所有训练好的模型并生成对比报告:
1. experiments/runs/ 下的实验模型 (full_ft, head_only, lora_only_r8)
2. runs/ 下的生产模型 (lora_vitb16_a100_balanced)

生成:
- 每个模型的详细性能指标
- 模型间的对比可视化
- 综合性能报告
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import time

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from anitune.data import DataConfig, build_dataloaders, build_eval_loader
from anitune.models import ModelConfig, build_model
from anitune.utils import load_config


def evaluate_model(model, loader, device, num_classes):
    """评估单个模型

    Returns:
        dict: 包含详细指标的字典
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    # Top-k accuracy
    top1_correct = 0
    top5_correct = 0

    inference_times = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # 测量推理时间
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            probs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            # Top-1 accuracy
            _, preds = outputs.max(1)
            top1_correct += preds.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_correct += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    avg_loss = loss_sum / total

    # 计算置信度统计
    pred_confidences = np.max(all_probs, axis=1)
    correct_mask = all_preds == all_labels

    metrics = {
        'total_samples': total,
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'avg_loss': float(avg_loss),
        'avg_inference_time': float(np.mean(inference_times)),
        'std_inference_time': float(np.std(inference_times)),
        'throughput': float(total / sum(inference_times)),  # samples/sec
        'confidence': {
            'mean': float(np.mean(pred_confidences)),
            'std': float(np.std(pred_confidences)),
            'correct_mean': float(np.mean(pred_confidences[correct_mask])),
            'incorrect_mean': float(np.mean(pred_confidences[~correct_mask])) if (~correct_mask).sum() > 0 else 0.0,
        },
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
    }

    return metrics


def compute_per_class_metrics(preds, labels, num_classes):
    """计算每个类别的详细指标"""
    per_class_stats = {}

    for cls in range(num_classes):
        cls_mask = labels == cls
        if cls_mask.sum() == 0:
            continue

        cls_preds = preds[cls_mask]
        cls_correct = (cls_preds == cls).sum()
        cls_total = cls_mask.sum()

        per_class_stats[int(cls)] = {
            'total': int(cls_total),
            'correct': int(cls_correct),
            'accuracy': float(cls_correct / cls_total),
            'error_rate': float(1 - cls_correct / cls_total),
        }

    return per_class_stats


def load_model_and_config(checkpoint_path, config_path, device):
    """加载模型和配置"""
    # 加载配置
    cfg = load_config(config_path)
    data_cfg = DataConfig(**cfg["data"])
    model_cfg = ModelConfig(**cfg["model"])

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # 检查类别数
    if 'head.weight' in state_dict:
        num_classes = state_dict['head.weight'].shape[0]
        model_cfg.num_classes = num_classes

    # 构建模型
    model = build_model(model_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    return model, data_cfg, model_cfg, cfg


def get_model_info(model_cfg, checkpoint_path):
    """获取模型信息"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # 计算参数量
    total_params = sum(p.numel() for p in state_dict.values())

    # 检测LoRA参数
    lora_params = sum(p.numel() for k, p in state_dict.items() if 'lora' in k.lower())

    # 文件大小
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    info = {
        'config': {
            'name': model_cfg.name,
            'num_classes': model_cfg.num_classes,
            'use_lora': model_cfg.use_lora,
            'lora_rank': getattr(model_cfg, 'lora_rank', None),
            'lora_alpha': getattr(model_cfg, 'lora_alpha', None),
        },
        'parameters': {
            'total': int(total_params),
            'lora': int(lora_params),
            'total_mb': float(total_params * 4 / (1024 * 1024)),  # float32
        },
        'checkpoint': {
            'path': str(checkpoint_path),
            'size_mb': float(file_size_mb),
            'val_acc': float(checkpoint.get('val_acc', 0.0)) if isinstance(checkpoint, dict) else 0.0,
        }
    }

    return info


def main():
    parser = argparse.ArgumentParser(description="评估所有训练好的模型")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default="evaluation_results", help="输出目录")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="评估数据集")
    parser.add_argument("--save-predictions", action="store_true", help="保存每个模型的预测结果")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # 定义要评估的模型
    models_to_eval = [
        {
            'name': 'LoRA-only (r=8) - Experiment',
            'checkpoint': 'experiments/runs/vit_experiment_lora_only_r8/best.pt',
            'config': 'experiments/configs/base_experiment.yaml',
            'type': 'lora_only',
        },
        {
            'name': 'Head-only - Experiment',
            'checkpoint': 'experiments/runs/vit_experiment_head_only/best.pt',
            'config': 'experiments/configs/base_experiment.yaml',
            'type': 'head_only',
        },
        {
            'name': 'Full Fine-tuning - Experiment',
            'checkpoint': 'experiments/runs/vit_experiment_full_ft/best.pt',
            'config': 'experiments/configs/base_experiment.yaml',
            'type': 'full_ft',
        },
        {
            'name': 'LoRA (r=12) - A100 Balanced',
            'checkpoint': 'runs/lora_vitb16_a100_balanced/best.pt',
            'config': 'configs/lora_vitb16_a100_balanced.yaml',
            'type': 'lora_balanced',
        },
    ]

    # 检查模型是否存在
    models_to_eval = [m for m in models_to_eval if Path(m['checkpoint']).exists()]

    if not models_to_eval:
        print("❌ No model checkpoints found!")
        return

    print(f"\n{'='*70}")
    print(f"  Evaluating {len(models_to_eval)} models on {args.split} set")
    print(f"{'='*70}\n")

    # 评估所有模型
    all_results = {}

    for i, model_spec in enumerate(models_to_eval, 1):
        print(f"\n[{i}/{len(models_to_eval)}] Evaluating: {model_spec['name']}")
        print(f"{'─'*70}")

        checkpoint_path = Path(model_spec['checkpoint'])
        config_path = Path(model_spec['config'])

        try:
            # 加载模型
            print("  Loading model...")
            model, data_cfg, model_cfg, cfg = load_model_and_config(
                checkpoint_path, config_path, device
            )

            # 获取模型信息
            model_info = get_model_info(model_cfg, checkpoint_path)
            print(f"  Model: {model_cfg.name}")
            print(f"  Classes: {model_cfg.num_classes}")
            print(f"  Parameters: {model_info['parameters']['total']:,} ({model_info['parameters']['total_mb']:.1f} MB)")
            if model_info['parameters']['lora'] > 0:
                print(f"  LoRA Parameters: {model_info['parameters']['lora']:,}")

            # 构建数据加载器
            print("  Loading data...")
            train_loader, val_loader = build_dataloaders(data_cfg)
            eval_loader = val_loader if args.split == "val" else None  # TODO: Add test loader support

            if eval_loader is None:
                print("  ⚠️  Test set not available, using validation set")
                eval_loader = val_loader

            # 评估模型
            print("  Evaluating...")
            metrics = evaluate_model(model, eval_loader, device, model_cfg.num_classes)

            # 计算每类指标
            print("  Computing per-class metrics...")
            preds = np.array(metrics['predictions'])
            labels = np.array(metrics['labels'])
            per_class = compute_per_class_metrics(preds, labels, model_cfg.num_classes)

            # 汇总结果
            results = {
                'name': model_spec['name'],
                'type': model_spec['type'],
                'info': model_info,
                'metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'labels', 'probabilities']},
                'per_class_summary': {
                    'num_classes': len(per_class),
                    'avg_accuracy': float(np.mean([v['accuracy'] for v in per_class.values()])),
                    'std_accuracy': float(np.std([v['accuracy'] for v in per_class.values()])),
                    'min_accuracy': float(min([v['accuracy'] for v in per_class.values()])),
                    'max_accuracy': float(max([v['accuracy'] for v in per_class.values()])),
                },
            }

            # 保存详细的每类统计
            per_class_path = output_dir / f"{model_spec['type']}_per_class.json"
            with open(per_class_path, 'w') as f:
                json.dump(per_class, f, indent=2)

            # 可选：保存预测结果
            if args.save_predictions:
                pred_path = output_dir / f"{model_spec['type']}_predictions.npz"
                np.savez_compressed(
                    pred_path,
                    predictions=preds,
                    labels=labels,
                    probabilities=np.array(metrics['probabilities'])
                )
                print(f"  ✅ Saved predictions to {pred_path}")

            all_results[model_spec['type']] = results

            # 打印关键指标
            print(f"\n  📊 Results:")
            print(f"     Top-1 Accuracy: {metrics['top1_accuracy']*100:.2f}%")
            print(f"     Top-5 Accuracy: {metrics['top5_accuracy']*100:.2f}%")
            print(f"     Avg Loss: {metrics['avg_loss']:.4f}")
            print(f"     Throughput: {metrics['throughput']:.1f} samples/sec")
            print(f"     Avg Confidence: {metrics['confidence']['mean']:.3f}")
            print(f"  ✅ Evaluation complete")

            # 清理显存
            del model, eval_loader, train_loader
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ Error evaluating {model_spec['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存综合结果
    results_path = output_dir / f"all_models_results_{args.split}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}\n")

    # 按Top-1准确率排序
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1]['metrics']['top1_accuracy'],
        reverse=True
    )

    print(f"{'Model':<40} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'Params':>15}")
    print(f"{'─'*70}")
    for model_type, result in sorted_models:
        name = result['name']
        top1 = result['metrics']['top1_accuracy'] * 100
        top5 = result['metrics']['top5_accuracy'] * 100
        params = result['info']['parameters']['total']

        print(f"{name:<40} {top1:>11.2f}% {top5:>11.2f}% {params:>14,}")

    print(f"\n{'='*70}")
    print(f"✅ All results saved to: {output_dir}")
    print(f"   Main results: {results_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
