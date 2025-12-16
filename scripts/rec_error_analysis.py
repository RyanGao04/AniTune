#!/usr/bin/env python
"""基于 rec_evaluation_code 评估方式的错误分析

这个脚本针对 iCartoonFace 比赛的评估方式进行错误分析:
1. 提取模型特征向量
2. 使用余弦距离进行匹配
3. 分析检索错误和可视化
4. 生成详细的错误报告
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
from sklearn.metrics.pairwise import pairwise_distances

from anitune.data import DataConfig, build_dataloaders
from anitune.models import ModelConfig, build_model
from anitune.utils import load_config


def extract_features(model, loader, device):
    """提取所有样本的特征向量

    Returns:
        features: numpy array of shape (N, D)
        labels: numpy array of shape (N,)
        img_paths: list of image paths
    """
    model.eval()

    all_features = []
    all_labels = []
    all_img_paths = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Extracting features")):
            images = images.to(device)

            # 提取特征（从最后一层之前）
            # 对于 ViT，我们使用 cls token 的输出
            if hasattr(model, 'forward_features'):
                features = model.forward_features(images)
            else:
                # 如果没有 forward_features，使用倒数第二层
                # 移除最后的分类层
                with torch.no_grad():
                    # 获取 forward_head 之前的特征
                    if hasattr(model, 'blocks'):
                        # ViT architecture
                        x = model.patch_embed(images)
                        x = model._pos_embed(x)
                        x = model.blocks(x)
                        x = model.norm(x)
                        features = x[:, 0]  # cls token
                    else:
                        # 通用方法：使用全连接层之前的特征
                        features = model.forward_features(images)

            # L2 normalize
            features = F.normalize(features, p=2, dim=1)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

            # 获取图像路径（如果可用）
            dataset = loader.dataset
            for i in range(len(labels)):
                global_idx = batch_idx * loader.batch_size + i
                if hasattr(dataset, 'entries'):
                    img_path, _ = dataset.entries[global_idx]
                    all_img_paths.append(img_path)
                else:
                    all_img_paths.append(f"sample_{global_idx}")

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    return all_features, all_labels, all_img_paths


def compute_retrieval_accuracy(features, labels, top_k=[1, 5, 10, 20]):
    """计算检索准确率（基于余弦距离）

    Args:
        features: (N, D) 特征向量
        labels: (N,) 标签
        top_k: list of k values for top-k accuracy

    Returns:
        metrics: dict with retrieval metrics
        distance_matrix: (N, N) 距离矩阵
        rank_matrix: (N, N) 排序矩阵
    """
    print("Computing pairwise distances...")
    # 计算余弦距离
    distance = pairwise_distances(features, features, metric='cosine', n_jobs=-1)

    # 排序（每行按距离从小到大排序）
    rank_matrix = np.argsort(distance, axis=1)

    # 计算 top-k 准确率
    metrics = {}
    n_samples = len(labels)

    for k in top_k:
        correct = 0
        for i in range(n_samples):
            # 获取最近的 k+1 个样本（包括自己）
            top_k_indices = rank_matrix[i, :k+1]
            # 移除自己
            top_k_indices = top_k_indices[top_k_indices != i][:k]
            # 检查是否有同类
            top_k_labels = labels[top_k_indices]
            if np.any(top_k_labels == labels[i]):
                correct += 1

        metrics[f'top{k}_accuracy'] = correct / n_samples

    # 计算 Mean Average Precision (mAP)
    print("Computing mAP...")
    aps = []
    for i in tqdm(range(n_samples), desc="Computing AP"):
        # 获取同类样本的索引（排除自己）
        same_class_mask = (labels == labels[i]) & (np.arange(n_samples) != i)
        num_same_class = same_class_mask.sum()

        if num_same_class == 0:
            continue

        # 获取检索排序
        retrieved_indices = rank_matrix[i, 1:]  # 排除自己
        retrieved_labels = labels[retrieved_indices]

        # 计算 Average Precision
        precisions = []
        num_correct = 0
        for rank, idx in enumerate(retrieved_indices, 1):
            if labels[idx] == labels[i]:
                num_correct += 1
                precision = num_correct / rank
                precisions.append(precision)

        if precisions:
            ap = np.mean(precisions)
            aps.append(ap)

    metrics['mAP'] = np.mean(aps) if aps else 0.0

    # 计算每个查询的最近邻准确率
    nearest_neighbor_correct = 0
    for i in range(n_samples):
        nn_idx = rank_matrix[i, 1]  # 最近邻（排除自己）
        if labels[nn_idx] == labels[i]:
            nearest_neighbor_correct += 1

    metrics['nearest_neighbor_accuracy'] = nearest_neighbor_correct / n_samples

    return metrics, distance, rank_matrix


def analyze_retrieval_errors(labels, distance, rank_matrix, top_k=10):
    """分析检索错误

    Returns:
        error_samples: list of dicts with error information
        correct_samples: list of dicts with correct retrieval information
    """
    error_samples = []
    correct_samples = []

    n_samples = len(labels)

    for query_idx in range(n_samples):
        # 获取 top-k 检索结果（排除自己）
        top_k_indices = rank_matrix[query_idx, 1:top_k+1]
        top_k_labels = labels[top_k_indices]
        top_k_distances = distance[query_idx, top_k_indices]

        query_label = labels[query_idx]

        # 检查是否有同类
        has_same_class = np.any(top_k_labels == query_label)

        # 第一个检索结果
        first_retrieved_idx = top_k_indices[0]
        first_retrieved_label = top_k_labels[0]
        first_distance = top_k_distances[0]

        sample_info = {
            'query_idx': query_idx,
            'query_label': int(query_label),
            'retrieved_idx': int(first_retrieved_idx),
            'retrieved_label': int(first_retrieved_label),
            'distance': float(first_distance),
            'top_k_labels': top_k_labels.tolist(),
            'top_k_distances': top_k_distances.tolist(),
        }

        if first_retrieved_label != query_label:
            error_samples.append(sample_info)
        else:
            correct_samples.append(sample_info)

    return error_samples, correct_samples


def plot_retrieval_metrics(metrics, save_path):
    """绘制检索指标"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Top-K 准确率曲线
    top_k_keys = [k for k in metrics.keys() if k.startswith('top')]
    k_values = [int(k.replace('top', '').replace('_accuracy', '')) for k in top_k_keys]
    accuracies = [metrics[k] * 100 for k in top_k_keys]

    ax1.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax1.fill_between(k_values, accuracies, alpha=0.3, color='#3498db')
    ax1.set_xlabel('K (Top-K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Retrieval Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Top-K Retrieval Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)

    for k, acc in zip(k_values, accuracies):
        ax1.text(k, acc + 1, f'{acc:.1f}%', ha='center', fontsize=9)

    # 右图：整体指标
    overall_metrics = {
        'Nearest\nNeighbor': metrics.get('nearest_neighbor_accuracy', 0) * 100,
        'Top-5': metrics.get('top5_accuracy', 0) * 100,
        'Top-10': metrics.get('top10_accuracy', 0) * 100,
        'mAP': metrics.get('mAP', 0) * 100,
    }

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax2.bar(range(len(overall_metrics)), list(overall_metrics.values()),
                   color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(overall_metrics)))
    ax2.set_xticklabels(list(overall_metrics.keys()))
    ax2.set_ylabel('Accuracy / Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Retrieval Metrics', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved retrieval metrics to {save_path}")
    plt.close()


def plot_distance_distribution(error_samples, correct_samples, save_path):
    """绘制距离分布对比"""
    error_distances = [s['distance'] for s in error_samples]
    correct_distances = [s['distance'] for s in correct_samples]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：直方图
    bins = np.linspace(0, 2, 40)
    ax1.hist(correct_distances, bins=bins, alpha=0.6, label='Correct Retrieval',
            color='green', edgecolor='black')
    ax1.hist(error_distances, bins=bins, alpha=0.6, label='Incorrect Retrieval',
            color='red', edgecolor='black')
    ax1.set_xlabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distance Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 右图：箱线图
    data_to_plot = [correct_distances, error_distances]
    labels = ['Correct', 'Incorrect']
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    colors = ['green', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax2.set_title('Distance Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 添加统计信息
    stats_text = f"Correct: μ={np.mean(correct_distances):.3f}, σ={np.std(correct_distances):.3f}\n"
    stats_text += f"Incorrect: μ={np.mean(error_distances):.3f}, σ={np.std(error_distances):.3f}"
    ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved distance distribution to {save_path}")
    plt.close()


def plot_confusion_pairs(error_samples, save_path, top_n=20):
    """绘制最容易混淆的类别对"""
    pair_counts = Counter()

    for sample in error_samples:
        pair = (sample['query_label'], sample['retrieved_label'])
        pair_counts[pair] += 1

    # 获取 top-n
    top_pairs = pair_counts.most_common(top_n)

    if not top_pairs:
        print("No confusion pairs to plot")
        return

    pairs = [f"{q} → {r}" for (q, r), _ in top_pairs]
    counts = [count for _, count in top_pairs]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(pairs)), counts, color='coral', edgecolor='black')
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel('Number of Retrieval Errors', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Confused Class Pairs (Query → Retrieved)',
                fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{int(width)}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion pairs to {save_path}")
    plt.close()


def visualize_retrieval_examples(samples, img_paths, data_root, rank_matrix, labels,
                                 save_path, title, num_examples=5, top_k=5):
    """可视化检索示例（查询图片 + top-k 检索结果）"""
    n_cols = top_k + 1  # 1 query + k retrieved
    n_rows = num_examples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    samples_to_show = samples[:num_examples]

    for row_idx, sample in enumerate(samples_to_show):
        query_idx = sample['query_idx']
        query_label = sample['query_label']

        # 显示查询图片
        query_path = data_root / img_paths[query_idx]
        if query_path.exists():
            try:
                img = Image.open(query_path).convert('RGB')
                axes[row_idx, 0].imshow(img)
                axes[row_idx, 0].set_title(f'QUERY\nClass: {query_label}',
                                          fontsize=10, fontweight='bold', color='blue')
            except:
                pass
        axes[row_idx, 0].axis('off')

        # 显示 top-k 检索结果
        top_k_indices = rank_matrix[query_idx, 1:top_k+1]

        for k_idx, retrieved_idx in enumerate(top_k_indices):
            col_idx = k_idx + 1
            retrieved_label = labels[retrieved_idx]
            is_correct = retrieved_label == query_label

            retrieved_path = data_root / img_paths[retrieved_idx]
            if retrieved_path.exists():
                try:
                    img = Image.open(retrieved_path).convert('RGB')
                    axes[row_idx, col_idx].imshow(img)

                    color = 'green' if is_correct else 'red'
                    axes[row_idx, col_idx].set_title(
                        f'#{k_idx+1}\nClass: {retrieved_label}',
                        fontsize=9, color=color, fontweight='bold'
                    )
                except:
                    pass
            axes[row_idx, col_idx].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved retrieval examples to {save_path}")
    plt.close()


def save_error_statistics(metrics, error_samples, correct_samples, save_path):
    """保存错误统计信息"""
    stats = {
        'retrieval_metrics': {k: float(v) for k, v in metrics.items()},
        'total_queries': len(error_samples) + len(correct_samples),
        'total_errors': len(error_samples),
        'error_rate': len(error_samples) / (len(error_samples) + len(correct_samples)),
        'distance_statistics': {
            'correct_retrievals': {
                'mean': float(np.mean([s['distance'] for s in correct_samples])) if correct_samples else 0.0,
                'std': float(np.std([s['distance'] for s in correct_samples])) if correct_samples else 0.0,
                'min': float(np.min([s['distance'] for s in correct_samples])) if correct_samples else 0.0,
                'max': float(np.max([s['distance'] for s in correct_samples])) if correct_samples else 0.0,
            },
            'incorrect_retrievals': {
                'mean': float(np.mean([s['distance'] for s in error_samples])) if error_samples else 0.0,
                'std': float(np.std([s['distance'] for s in error_samples])) if error_samples else 0.0,
                'min': float(np.min([s['distance'] for s in error_samples])) if error_samples else 0.0,
                'max': float(np.max([s['distance'] for s in error_samples])) if error_samples else 0.0,
            },
        },
    }

    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved error statistics to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="基于检索的错误分析（rec_evaluation_code方式）")
    parser.add_argument("--config", type=Path, required=True, help="Config file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--output-dir", type=Path, default="rec_error_analysis", help="Output directory")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Which split to analyze")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval analysis")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  RETRIEVAL-BASED ERROR ANALYSIS (rec_evaluation_code 方式)")
    print("="*70)
    print()

    # 加载配置和模型
    print("Loading model and data...")
    cfg = load_config(args.config)
    data_cfg = DataConfig(**cfg["data"])
    data_cfg.root = args.data_root
    model_cfg = ModelConfig(**cfg["model"])

    # 构建数据加载器
    train_loader, val_loader = build_dataloaders(data_cfg)
    eval_loader = val_loader if args.split == "val" else None

    if eval_loader is None:
        print("⚠️  Test loader not available, using validation set")
        eval_loader = val_loader

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    if 'head.weight' in state_dict:
        model_cfg.num_classes = state_dict['head.weight'].shape[0]

    model = build_model(model_cfg)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device)
    model.to(device)

    print("✅ Model loaded successfully")
    print()

    # 提取特征
    print("="*70)
    print("  Extracting features...")
    print("="*70)
    features, labels, img_paths = extract_features(model, eval_loader, device)
    print(f"✅ Extracted features: {features.shape}")
    print()

    # 计算检索指标
    print("="*70)
    print("  Computing retrieval metrics...")
    print("="*70)
    metrics, distance, rank_matrix = compute_retrieval_accuracy(
        features, labels, top_k=[1, 5, 10, 20]
    )

    print("\n📊 Retrieval Metrics:")
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value*100:.2f}%")
    print()

    # 分析错误
    print("="*70)
    print("  Analyzing retrieval errors...")
    print("="*70)
    error_samples, correct_samples = analyze_retrieval_errors(
        labels, distance, rank_matrix, top_k=args.top_k
    )

    print(f"Total queries: {len(labels)}")
    print(f"Correct retrievals: {len(correct_samples)}")
    print(f"Incorrect retrievals: {len(error_samples)}")
    print(f"Error rate: {len(error_samples) / len(labels) * 100:.2f}%")
    print()

    # 生成可视化
    print("="*70)
    print("  Generating visualizations...")
    print("="*70)

    # 1. 检索指标图
    plot_retrieval_metrics(metrics, output_dir / f'retrieval_metrics_{args.split}.png')

    # 2. 距离分布
    plot_distance_distribution(error_samples, correct_samples,
                              output_dir / f'distance_distribution_{args.split}.png')

    # 3. 混淆对
    plot_confusion_pairs(error_samples, output_dir / f'confusion_pairs_{args.split}.png')

    # 4. 可视化检索示例
    if hasattr(eval_loader.dataset, 'entries'):
        # 错误示例
        print("  Visualizing error examples...")
        # 按距离排序，显示高置信度错误
        error_samples_sorted = sorted(error_samples, key=lambda x: x['distance'])
        visualize_retrieval_examples(
            error_samples_sorted[:5], img_paths, args.data_root,
            rank_matrix, labels,
            output_dir / 'retrieval_errors_low_distance.png',
            'Retrieval Errors (Low Distance = High Confidence)',
            num_examples=5, top_k=5
        )

        # 正确示例
        print("  Visualizing correct examples...")
        correct_samples_sorted = sorted(correct_samples, key=lambda x: x['distance'])
        visualize_retrieval_examples(
            correct_samples_sorted[:5], img_paths, args.data_root,
            rank_matrix, labels,
            output_dir / 'retrieval_correct_low_distance.png',
            'Correct Retrievals (Low Distance)',
            num_examples=5, top_k=5
        )

    # 5. 保存统计信息
    save_error_statistics(metrics, error_samples, correct_samples,
                         output_dir / f'retrieval_statistics_{args.split}.json')

    # 保存特征向量（可选）
    features_path = output_dir / f'features_{args.split}.npz'
    np.savez_compressed(
        features_path,
        features=features,
        labels=labels,
        img_paths=img_paths
    )
    print(f"✅ Saved features to {features_path}")

    print()
    print("="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print(f"All results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
