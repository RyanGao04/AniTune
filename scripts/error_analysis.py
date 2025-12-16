#!/usr/bin/env python
"""错误分析和可视化脚本

分析模型预测错误的样本，生成：
1. 混淆矩阵
2. 最容易混淆的类别对
3. 错误样本可视化
4. 按类别的准确率分析
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from anitune.data import DataConfig, build_dataloaders
from anitune.models import ModelConfig, build_model
from anitune.utils import load_config


def analyze_errors(model, loader, device, model_num_classes):
    """分析预测错误和正确样本
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        model_num_classes: 模型的类别数（可能大于数据集的类别数）
    
    Returns:
        preds, labels, probs, error_samples, correct_samples
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    error_samples = []
    correct_samples = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Analyzing")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 记录错误和正确样本
            for i in range(len(labels)):
                sample_info = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item(),
                    'confidence': probs[i, preds[i]].item(),
                    'true_prob': probs[i, labels[i]].item(),
                }
                
                if preds[i] != labels[i]:
                    error_samples.append(sample_info)
                else:
                    correct_samples.append(sample_info)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), error_samples, correct_samples


def compute_confusion_matrix(preds, labels, num_classes):
    """计算混淆矩阵
    
    Args:
        preds: 预测标签
        labels: 真实标签
        num_classes: 使用的类别数（应该是模型的类别数，覆盖所有可能的预测）
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for pred, label in zip(preds, labels):
        # 确保索引在范围内
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            conf_matrix[label, pred] += 1
        else:
            print(f"Warning: Skipping out-of-range prediction (label={label}, pred={pred})")
    
    return conf_matrix


def plot_confusion_matrix_top_k(conf_matrix, k=50, save_path=None):
    """绘制 top-k 类别的混淆矩阵"""
    # 选择样本数最多的 k 个类别
    class_counts = conf_matrix.sum(axis=1)
    # 只考虑有样本的类别
    valid_classes = np.where(class_counts > 0)[0]
    valid_counts = class_counts[valid_classes]
    
    # 按样本数排序，取 top-k
    top_k_indices = np.argsort(valid_counts)[-min(k, len(valid_counts)):][::-1]
    top_k_classes = valid_classes[top_k_indices]
    
    # 提取子矩阵
    sub_matrix = conf_matrix[np.ix_(top_k_classes, top_k_classes)]
    
    # 归一化（按行）
    row_sums = sub_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    normalized_matrix = sub_matrix.astype(float) / row_sums
    
    # 绘图
    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_matrix, cmap='YlOrRd', vmin=0, vmax=1,
                cbar_kws={'label': 'Normalized Frequency'})
    plt.title(f'Confusion Matrix (Top {len(top_k_classes)} Classes by Sample Count)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()


def find_most_confused_pairs(conf_matrix, top_n=20):
    """找出最容易混淆的类别对"""
    num_classes = conf_matrix.shape[0]
    confused_pairs = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j] > 0:
                confused_pairs.append((i, j, conf_matrix[i, j]))
    
    # 按混淆次数排序
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    return confused_pairs[:top_n]


def plot_confused_pairs(confused_pairs, save_path=None):
    """绘制最容易混淆的类别对"""
    if not confused_pairs:
        print("No confused pairs to plot")
        return
    
    pairs = [f"{true} → {pred}" for true, pred, _ in confused_pairs]
    counts = [count for _, _, count in confused_pairs]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(pairs)), counts, color='coral')
    plt.yticks(range(len(pairs)), pairs, fontsize=8)
    plt.xlabel('Number of Misclassifications')
    plt.title('Top Confused Class Pairs')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confused pairs to {save_path}")
    else:
        plt.show()
    plt.close()


def compute_per_class_accuracy(conf_matrix):
    """计算每个类别的准确率"""
    per_class_acc = []
    for i in range(conf_matrix.shape[0]):
        total = conf_matrix[i].sum()
        if total > 0:
            acc = conf_matrix[i, i] / total
            per_class_acc.append((i, acc, total))
    return per_class_acc


def plot_per_class_accuracy(per_class_acc, save_path=None):
    """绘制每个类别的准确率分布"""
    if not per_class_acc:
        print("No per-class accuracy data to plot")
        return
    
    accuracies = [acc for _, acc, _ in per_class_acc]
    counts = [count for _, _, count in per_class_acc]
    
    # 按样本数量分组
    bins = [0, 10, 30, 50, 100, float('inf')]
    bin_labels = ['<10', '10-30', '30-50', '50-100', '≥100']
    bin_accs = {label: [] for label in bin_labels}
    
    for acc, count in zip(accuracies, counts):
        for i in range(len(bins) - 1):
            if bins[i] <= count < bins[i+1]:
                bin_accs[bin_labels[i]].append(acc)
                break
    
    # 绘制箱线图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：准确率直方图
    ax1.hist(accuracies, bins=20, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.3f}')
    ax1.set_xlabel('Per-Class Accuracy')
    ax1.set_ylabel('Number of Classes')
    ax1.set_title('Distribution of Per-Class Accuracy')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：按样本数量分组的箱线图
    data_to_plot = [bin_accs[label] for label in bin_labels if bin_accs[label]]
    labels_to_plot = [label for label in bin_labels if bin_accs[label]]
    
    if data_to_plot:
        ax2.boxplot(data_to_plot, labels=labels_to_plot)
        ax2.set_xlabel('Sample Count per Class')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs. Sample Count')
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class accuracy to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_samples(samples, loader, data_root, save_path, title, num_samples=20, sort_by='confidence', reverse=True):
    """通用的样本可视化函数
    
    Args:
        samples: 样本列表
        loader: 数据加载器
        data_root: 数据根目录
        save_path: 保存路径
        title: 图表标题
        num_samples: 显示样本数
        sort_by: 排序字段
        reverse: 是否降序
    """
    # 从 loader 获取图像路径（需要 ManifestDataset）
    dataset = loader.dataset
    if not hasattr(dataset, 'entries'):
        print(f"Dataset doesn't support path retrieval for {title}")
        return
    
    # 排序并选择样本
    samples_sorted = sorted(samples, key=lambda x: x[sort_by], reverse=reverse)
    selected_samples = samples_sorted[:num_samples]
    
    # 计算网格大小
    n_cols = 5
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, sample in enumerate(selected_samples):
        # 计算全局索引
        global_idx = sample['batch_idx'] * loader.batch_size + sample['sample_idx']
        
        if global_idx >= len(dataset.entries):
            continue
        
        rel_path, _ = dataset.entries[global_idx]
        img_path = data_root / rel_path
        
        if not img_path.exists():
            continue
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 显示
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            # 根据是否正确预测设置标题颜色
            is_correct = sample['true_label'] == sample['pred_label']
            title_color = 'green' if is_correct else 'red'
            
            axes[idx].set_title(
                f"True: {sample['true_label']}\n"
                f"Pred: {sample['pred_label']}\n"
                f"Conf: {sample['confidence']:.3f}",
                fontsize=8,
                color=title_color
            )
        except Exception as e:
            axes[idx].axis('off')
            continue
    
    # 隐藏多余的子图
    for idx in range(len(selected_samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {title} to {save_path}")
    plt.close()


def visualize_class_samples(error_samples, correct_samples, loader, data_root, save_dir, 
                            worst_classes, best_classes, num_samples_per_class=10):
    """可视化特定类别的样本（错误最多和最好的类别）"""
    dataset = loader.dataset
    if not hasattr(dataset, 'entries'):
        print("Dataset doesn't support path retrieval")
        return
    
    # 为每个最差类别可视化错误样本
    for i, (cls_id, acc, count) in enumerate(worst_classes[:3]):  # Top 3 worst classes
        class_errors = [s for s in error_samples if s['true_label'] == cls_id]
        
        if len(class_errors) >= 5:  # 至少5个错误样本才可视化
            visualize_samples(
                class_errors, loader, data_root,
                save_dir / f'worst_class_{cls_id}_errors.png',
                f'Worst Class {cls_id} - Errors (Acc: {acc:.2%}, Count: {count})',
                num_samples=min(num_samples_per_class, len(class_errors))
            )
    
    # 为每个最好类别可视化正确样本
    for i, (cls_id, acc, count) in enumerate(best_classes[:3]):  # Top 3 best classes
        class_correct = [s for s in correct_samples if s['true_label'] == cls_id]
        
        if len(class_correct) >= 5:  # 至少5个正确样本才可视化
            visualize_samples(
                class_correct, loader, data_root,
                save_dir / f'best_class_{cls_id}_correct.png',
                f'Best Class {cls_id} - Correct Predictions (Acc: {acc:.2%}, Count: {count})',
                num_samples=min(num_samples_per_class, len(class_correct))
            )


def save_error_statistics(error_samples, per_class_acc, confused_pairs, save_path, 
                          total_samples, correct_samples):
    """保存错误统计信息"""
    if not per_class_acc:
        print("Warning: No per-class accuracy data")
        per_class_acc = [(0, 0.0, 0)]
    
    stats = {
        'total_samples': int(total_samples),
        'correct_predictions': int(correct_samples),
        'total_errors': len(error_samples),
        'overall_accuracy': float(correct_samples / total_samples) if total_samples > 0 else 0.0,
        'avg_error_confidence': float(np.mean([e['confidence'] for e in error_samples])) if error_samples else 0.0,
        'median_error_confidence': float(np.median([e['confidence'] for e in error_samples])) if error_samples else 0.0,
        'per_class_accuracy': {
            'mean': float(np.mean([acc for _, acc, _ in per_class_acc])),
            'median': float(np.median([acc for _, acc, _ in per_class_acc])),
            'std': float(np.std([acc for _, acc, _ in per_class_acc])),
            'min': float(min([acc for _, acc, _ in per_class_acc])),
            'max': float(max([acc for _, acc, _ in per_class_acc])),
        },
        'worst_classes': [
            {'class_id': int(cls), 'accuracy': float(acc), 'count': int(count)}
            for cls, acc, count in sorted(per_class_acc, key=lambda x: x[1])[:20]
        ],
        'best_classes': [
            {'class_id': int(cls), 'accuracy': float(acc), 'count': int(count)}
            for cls, acc, count in sorted(per_class_acc, key=lambda x: x[1], reverse=True)[:20]
        ],
        'most_confused_pairs': [
            {'true_class': int(true), 'pred_class': int(pred), 'count': int(count)}
            for true, pred, count in confused_pairs[:20]
        ],
    }
    
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved error statistics to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Error analysis and visualization")
    parser.add_argument("--config", type=Path, required=True, help="Config file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--output-dir", type=Path, default="error_analysis", help="Output directory")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Which split to analyze")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    cfg = load_config(args.config)
    data_cfg = DataConfig(**cfg["data"])
    data_cfg.root = args.data_root
    model_cfg = ModelConfig(**cfg["model"])
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Loading data and model...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 构建数据加载器
    print("Loading data...")
    loaders = build_dataloaders(data_cfg)
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    # 选择评估集
    eval_loader = val_loader if args.split == "val" else loaders[2] if len(loaders) >= 3 else val_loader
    
    # 获取数据集的类别数
    if hasattr(train_loader.dataset, 'classes'):
        dataset_num_classes = len(train_loader.dataset.classes)
    elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'classes'):
        dataset_num_classes = len(train_loader.dataset.dataset.classes)
    else:
        dataset_num_classes = model_cfg.num_classes
    
    print(f"  Dataset has {dataset_num_classes} classes")
    
    # 加载 checkpoint 并检查其类别数
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    
    # 检查 checkpoint 中的类别数
    model_num_classes = dataset_num_classes
    if 'head.weight' in state_dict:
        ckpt_num_classes = state_dict['head.weight'].shape[0]
        print(f"  Checkpoint has {ckpt_num_classes} classes")
        
        if ckpt_num_classes != dataset_num_classes:
            print(f"  ⚠️  Class count mismatch!")
            print(f"     Checkpoint: {ckpt_num_classes} classes")
            print(f"     Dataset: {dataset_num_classes} classes")
            print(f"  Using checkpoint's class count for model")
            model_num_classes = ckpt_num_classes
    
    # 使用 checkpoint 的类别数构建模型
    model_cfg.num_classes = model_num_classes
    model = build_model(model_cfg)
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device(args.device)
    model.to(device)
    
    print(f"✅ Model loaded successfully")
    print(f"   Model output classes: {model_num_classes}")
    print(f"   Dataset classes: {dataset_num_classes}")
    
    # 分析错误
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Analyzing errors on {args.split} set...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    preds, labels, probs, error_samples, correct_samples = analyze_errors(model, eval_loader, device, model_num_classes)
    
    # 计算混淆矩阵（使用模型的类别数）
    print("Computing confusion matrix...")
    conf_matrix = compute_confusion_matrix(preds, labels, model_num_classes)
    
    # 计算每个类别的准确率
    per_class_acc = compute_per_class_accuracy(conf_matrix)
    
    # 找出最容易混淆的类别对
    confused_pairs = find_most_confused_pairs(conf_matrix, top_n=20)
    
    # 统计信息
    total_samples = len(labels)
    num_correct = (preds == labels).sum()
    accuracy = num_correct / total_samples
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Correct predictions: {num_correct}")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"   Total errors: {len(error_samples)}")
    
    # 更详细的统计
    if error_samples:
        error_confs = [e['confidence'] for e in error_samples]
        print(f"\n   Error statistics:")
        print(f"     Average error confidence: {np.mean(error_confs):.3f}")
        print(f"     Median error confidence: {np.median(error_confs):.3f}")
        print(f"     Min error confidence: {np.min(error_confs):.3f}")
        print(f"     Max error confidence: {np.max(error_confs):.3f}")
    
    if correct_samples:
        correct_confs = [c['confidence'] for c in correct_samples]
        print(f"\n   Correct prediction statistics:")
        print(f"     Average confidence: {np.mean(correct_confs):.3f}")
        print(f"     Median confidence: {np.median(correct_confs):.3f}")
        print(f"     Min confidence: {np.min(correct_confs):.3f}")
        print(f"     Max confidence: {np.max(correct_confs):.3f}")
    
    # 生成可视化
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Generating visualizations...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 1. 混淆矩阵 (top-50 类别)
    plot_confusion_matrix_top_k(conf_matrix, k=50, 
                                save_path=output_dir / f'confusion_matrix_top50_{args.split}.png')
    
    # 2. 最容易混淆的类别对
    plot_confused_pairs(confused_pairs, 
                       save_path=output_dir / f'confused_pairs_{args.split}.png')
    
    # 3. 每个类别的准确率分布
    plot_per_class_accuracy(per_class_acc, 
                           save_path=output_dir / f'per_class_accuracy_{args.split}.png')
    
    # 4. 样本可视化
    if hasattr(eval_loader.dataset, 'entries'):
        print("\n  4.1 High-confidence errors...")
        visualize_samples(
            error_samples, eval_loader, args.data_root,
            output_dir / 'high_confidence_errors.png',
            'High-Confidence Prediction Errors (Top 20)',
            num_samples=20, sort_by='confidence', reverse=True
        )
        
        print("  4.2 Low-confidence errors...")
        visualize_samples(
            error_samples, eval_loader, args.data_root,
            output_dir / 'low_confidence_errors.png',
            'Low-Confidence Prediction Errors (Bottom 20)',
            num_samples=20, sort_by='confidence', reverse=False
        )
        
        print("  4.3 High-confidence correct predictions...")
        visualize_samples(
            correct_samples, eval_loader, args.data_root,
            output_dir / 'high_confidence_correct.png',
            'High-Confidence Correct Predictions (Top 20)',
            num_samples=20, sort_by='confidence', reverse=True
        )
        
        print("  4.4 Low-confidence correct predictions...")
        visualize_samples(
            correct_samples, eval_loader, args.data_root,
            output_dir / 'low_confidence_correct.png',
            'Low-Confidence Correct Predictions (Bottom 20)',
            num_samples=20, sort_by='confidence', reverse=False
        )
        
        # 为最差和最好的类别生成可视化
        if per_class_acc:
            print("  4.5 Worst and best classes...")
            worst_classes = sorted(per_class_acc, key=lambda x: x[1])[:5]
            best_classes = sorted(per_class_acc, key=lambda x: x[1], reverse=True)[:5]
            
            visualize_class_samples(
                error_samples, correct_samples, eval_loader, args.data_root,
                output_dir, worst_classes, best_classes, num_samples_per_class=10
            )
    else:
        print("Skipping sample visualization (dataset doesn't support path retrieval)")
    
    # 5. 保存统计信息
    save_error_statistics(error_samples, per_class_acc, confused_pairs,
                         output_dir / f'error_statistics_{args.split}.json',
                         total_samples, num_correct)
    
    # 打印总结
    print("\n" + "="*70)
    print("ERROR ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total samples: {total_samples}")
    print(f"Total errors: {len(error_samples)}")
    print(f"Overall accuracy: {accuracy * 100:.2f}%")
    if error_samples:
        print(f"Average error confidence: {np.mean([e['confidence'] for e in error_samples]):.3f}")
    print(f"\nPer-class accuracy statistics:")
    if per_class_acc:
        print(f"  Mean: {np.mean([acc for _, acc, _ in per_class_acc]):.3f}")
        print(f"  Std:  {np.std([acc for _, acc, _ in per_class_acc]):.3f}")
        print(f"  Min:  {min([acc for _, acc, _ in per_class_acc]):.3f}")
        print(f"  Max:  {max([acc for _, acc, _ in per_class_acc]):.3f}")
    print(f"\nMost confused pairs (top 10):")
    for i, (true, pred, count) in enumerate(confused_pairs[:10], 1):
        print(f"  {i}. Class {true} → {pred}: {count} errors")
    print("\n" + "="*70)
    print(f"All results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
