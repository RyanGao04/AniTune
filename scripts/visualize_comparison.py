#!/usr/bin/env python
"""模型对比可视化脚本

基于 eval_all_models.py 的评估结果生成可视化对比:
1. 准确率对比柱状图
2. 参数量 vs 性能散点图
3. 置信度分布对比
4. 每类准确率分布对比
5. Top-5 vs Top-1 准确率对比
6. 推理速度对比
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_results(results_path):
    """加载评估结果"""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(results, save_path):
    """绘制准确率对比柱状图"""
    models = []
    top1_accs = []
    top5_accs = []

    for model_type, result in results.items():
        models.append(result['name'])
        top1_accs.append(result['metrics']['top1_accuracy'] * 100)
        top5_accs.append(result['metrics']['top5_accuracy'] * 100)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, top1_accs, width, label='Top-1 Accuracy', color='#3498db')
    bars2 = ax.bar(x + width/2, top5_accs, width, label='Top-5 Accuracy', color='#2ecc71')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved accuracy comparison to {save_path}")
    plt.close()


def plot_params_vs_performance(results, save_path):
    """绘制参数量 vs 性能散点图"""
    models = []
    params = []
    accs = []
    colors = []

    color_map = {
        'lora_only': '#3498db',
        'lora_balanced': '#2ecc71',
        'head_only': '#e74c3c',
        'full_ft': '#f39c12',
    }

    for model_type, result in results.items():
        models.append(result['name'])
        params.append(result['info']['parameters']['total'] / 1e6)  # Convert to millions
        accs.append(result['metrics']['top1_accuracy'] * 100)
        colors.append(color_map.get(model_type, '#95a5a6'))

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制散点
    scatter = ax.scatter(params, accs, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

    # 添加标签
    for i, model in enumerate(models):
        ax.annotate(model,
                   (params[i], accs[i]),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('Total Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Efficiency Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加效率等高线（虚线）
    ax.axhline(y=max(accs), color='green', linestyle='--', alpha=0.3, label=f'Best Accuracy: {max(accs):.2f}%')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved params vs performance to {save_path}")
    plt.close()


def plot_confidence_comparison(results, save_path):
    """绘制置信度分布对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (model_type, result) in enumerate(results.items()):
        if idx >= 4:
            break

        ax = axes[idx]
        conf = result['metrics']['confidence']

        # 创建置信度对比柱状图
        categories = ['Overall\nMean', 'Correct\nMean', 'Incorrect\nMean']
        values = [conf['mean'], conf['correct_mean'], conf.get('incorrect_mean', 0)]
        colors_bar = ['#3498db', '#2ecc71', '#e74c3c']

        bars = ax.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylim([0, 1])
        ax.set_ylabel('Confidence', fontsize=10, fontweight='bold')
        ax.set_title(f"{result['name']}\n(Acc: {result['metrics']['top1_accuracy']*100:.2f}%)",
                    fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(results), 4):
        axes[idx].axis('off')

    plt.suptitle('Prediction Confidence Comparison', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confidence comparison to {save_path}")
    plt.close()


def plot_per_class_accuracy_distribution(results, per_class_data, save_path):
    """绘制每类准确率分布对比"""
    fig, ax = plt.subplots(figsize=(12, 7))

    data_to_plot = []
    labels = []

    for model_type, result in results.items():
        # 加载每类统计
        per_class_file = per_class_data.get(model_type)
        if per_class_file and per_class_file.exists():
            with open(per_class_file, 'r') as f:
                per_class = json.load(f)

            accs = [v['accuracy'] for v in per_class.values()]
            data_to_plot.append(accs)
            labels.append(result['name'])

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

        # 设置颜色
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Per-Class Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Accuracy Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved per-class accuracy distribution to {save_path}")
        plt.close()
    else:
        print("⚠️  No per-class data available for distribution plot")


def plot_top5_vs_top1(results, save_path):
    """绘制 Top-5 vs Top-1 准确率对比"""
    models = []
    top1_accs = []
    top5_accs = []
    gaps = []

    for model_type, result in results.items():
        models.append(result['name'])
        top1 = result['metrics']['top1_accuracy'] * 100
        top5 = result['metrics']['top5_accuracy'] * 100
        top1_accs.append(top1)
        top5_accs.append(top5)
        gaps.append(top5 - top1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：Top-1 vs Top-5 散点图
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(models)]
    ax1.scatter(top1_accs, top5_accs, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

    # 添加对角线
    min_val = min(min(top1_accs), min(top5_accs))
    max_val = max(max(top1_accs), max(top5_accs))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Top-1 = Top-5')

    for i, model in enumerate(models):
        ax1.annotate(model, (top1_accs[i], top5_accs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax1.set_xlabel('Top-1 Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Top-5 Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Top-1 vs Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：Gap (Top-5 - Top-1)
    bars = ax2.bar(range(len(models)), gaps, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.set_ylabel('Accuracy Gap (Top-5 - Top-1) %', fontsize=11, fontweight='bold')
    ax2.set_title('Top-5 vs Top-1 Gap', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Top-5 vs Top-1 comparison to {save_path}")
    plt.close()


def plot_throughput_comparison(results, save_path):
    """绘制推理速度对比"""
    models = []
    throughputs = []
    latencies = []

    for model_type, result in results.items():
        models.append(result['name'])
        throughputs.append(result['metrics']['throughput'])
        latencies.append(result['metrics']['avg_inference_time'] * 1000)  # Convert to ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：吞吐量
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(models)]
    bars1 = ax1.bar(range(len(models)), throughputs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.set_ylabel('Throughput (samples/sec)', fontsize=11, fontweight='bold')
    ax1.set_title('Inference Throughput Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 右图：延迟
    bars2 = ax2.bar(range(len(models)), latencies, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.set_ylabel('Average Latency (ms/batch)', fontsize=11, fontweight='bold')
    ax2.set_title('Inference Latency Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved throughput comparison to {save_path}")
    plt.close()


def create_summary_report(results, save_path):
    """创建汇总报告（文本格式）"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("                    MODEL COMPARISON SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        # 1. 整体性能排名
        f.write("1. OVERALL PERFORMANCE RANKING (by Top-1 Accuracy)\n")
        f.write("-"*80 + "\n")

        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]['metrics']['top1_accuracy'],
            reverse=True
        )

        for rank, (model_type, result) in enumerate(sorted_models, 1):
            f.write(f"\n  #{rank}. {result['name']}\n")
            f.write(f"      Type: {result['type']}\n")
            f.write(f"      Top-1 Accuracy: {result['metrics']['top1_accuracy']*100:.2f}%\n")
            f.write(f"      Top-5 Accuracy: {result['metrics']['top5_accuracy']*100:.2f}%\n")
            f.write(f"      Total Parameters: {result['info']['parameters']['total']:,}\n")
            if result['info']['parameters']['lora'] > 0:
                f.write(f"      LoRA Parameters: {result['info']['parameters']['lora']:,}\n")
            f.write(f"      Checkpoint Size: {result['info']['checkpoint']['size_mb']:.1f} MB\n")
            f.write(f"      Throughput: {result['metrics']['throughput']:.1f} samples/sec\n")

        # 2. 参数效率分析
        f.write("\n\n2. PARAMETER EFFICIENCY ANALYSIS\n")
        f.write("-"*80 + "\n")

        for model_type, result in sorted_models:
            params = result['info']['parameters']['total']
            acc = result['metrics']['top1_accuracy'] * 100
            efficiency = acc / (params / 1e6)  # Accuracy per million parameters

            f.write(f"\n  {result['name']}:\n")
            f.write(f"      Accuracy/M params: {efficiency:.4f}\n")

        # 3. 置信度分析
        f.write("\n\n3. PREDICTION CONFIDENCE ANALYSIS\n")
        f.write("-"*80 + "\n")

        for model_type, result in results.items():
            conf = result['metrics']['confidence']
            f.write(f"\n  {result['name']}:\n")
            f.write(f"      Overall Mean: {conf['mean']:.3f} ± {conf['std']:.3f}\n")
            f.write(f"      Correct Predictions: {conf['correct_mean']:.3f}\n")
            f.write(f"      Incorrect Predictions: {conf.get('incorrect_mean', 0):.3f}\n")

        # 4. 每类准确率统计
        f.write("\n\n4. PER-CLASS ACCURACY STATISTICS\n")
        f.write("-"*80 + "\n")

        for model_type, result in results.items():
            pca = result['per_class_summary']
            f.write(f"\n  {result['name']}:\n")
            f.write(f"      Num Classes: {pca['num_classes']}\n")
            f.write(f"      Avg Accuracy: {pca['avg_accuracy']:.3f} ± {pca['std_accuracy']:.3f}\n")
            f.write(f"      Min Accuracy: {pca['min_accuracy']:.3f}\n")
            f.write(f"      Max Accuracy: {pca['max_accuracy']:.3f}\n")

        # 5. 推荐
        f.write("\n\n5. RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")

        best_acc = sorted_models[0]
        f.write(f"\n  Best Overall Performance: {best_acc[1]['name']}\n")
        f.write(f"      - Highest accuracy: {best_acc[1]['metrics']['top1_accuracy']*100:.2f}%\n")

        # 找到参数效率最高的
        lora_models = [
            (model_type, result) for model_type, result in results.items()
            if 'lora' in model_type.lower()
        ]
        if lora_models:
            best_lora = max(lora_models, key=lambda x: x[1]['metrics']['top1_accuracy'])
            f.write(f"\n  Best Parameter-Efficient Model: {best_lora[1]['name']}\n")
            f.write(f"      - Accuracy: {best_lora[1]['metrics']['top1_accuracy']*100:.2f}%\n")
            f.write(f"      - Parameters: {best_lora[1]['info']['parameters']['total']:,}\n")

        # 找到速度最快的
        fastest = max(results.items(), key=lambda x: x[1]['metrics']['throughput'])
        f.write(f"\n  Fastest Inference: {fastest[1]['name']}\n")
        f.write(f"      - Throughput: {fastest[1]['metrics']['throughput']:.1f} samples/sec\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✅ Saved summary report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="生成模型对比可视化")
    parser.add_argument("--results", type=Path, required=True, help="评估结果JSON文件")
    parser.add_argument("--per-class-dir", type=Path, help="每类统计数据目录")
    parser.add_argument("--output-dir", type=Path, default="comparison_plots", help="输出目录")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载结果
    print(f"\n{'='*70}")
    print(f"  Loading results from {args.results}")
    print(f"{'='*70}\n")

    results = load_results(args.results)

    if not results:
        print("❌ No results found!")
        return

    print(f"Found {len(results)} models to compare\n")

    # 准备每类数据文件映射
    per_class_data = {}
    if args.per_class_dir:
        per_class_dir = Path(args.per_class_dir)
        if per_class_dir.exists():
            for model_type in results.keys():
                per_class_file = per_class_dir / f"{model_type}_per_class.json"
                if per_class_file.exists():
                    per_class_data[model_type] = per_class_file

    # 生成所有可视化
    print("Generating visualizations...\n")

    plot_accuracy_comparison(results, output_dir / "1_accuracy_comparison.png")
    plot_params_vs_performance(results, output_dir / "2_params_vs_performance.png")
    plot_confidence_comparison(results, output_dir / "3_confidence_comparison.png")
    plot_top5_vs_top1(results, output_dir / "4_top5_vs_top1.png")
    plot_throughput_comparison(results, output_dir / "5_throughput_comparison.png")

    if per_class_data:
        plot_per_class_accuracy_distribution(results, per_class_data,
                                            output_dir / "6_per_class_distribution.png")

    # 生成汇总报告
    create_summary_report(results, output_dir / "summary_report.txt")

    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
