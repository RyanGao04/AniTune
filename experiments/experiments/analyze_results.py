#!/usr/bin/env python
"""分析实验结果

读取所有实验的 checkpoint，生成对比表格和可视化
"""

import json
from pathlib import Path
import torch


def analyze_checkpoint(ckpt_path):
    """分析 checkpoint 文件

    Returns:
        dict: {
            'val_acc': float,
            'trainable_params': int,
            'total_params': int,
        }
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # 获取准确率
        val_acc = ckpt.get('val_acc', None)

        # 统计参数
        state_dict = ckpt.get('model', ckpt)
        total_params = sum(p.numel() for p in state_dict.values())

        # 估算可训练参数（通过模式识别）
        trainable_params = 0
        for key, param in state_dict.items():
            # LoRA 参数
            if 'lora_' in key:
                trainable_params += param.numel()
            # 分类头
            elif any(k in key for k in ['head', 'fc', 'classifier']):
                trainable_params += param.numel()

        return {
            'val_acc': val_acc,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_params / total_params * 100 if total_params > 0 else 0,
        }

    except Exception as e:
        print(f"  ⚠️  读取失败: {e}")
        return None


def main():
    runs_dir = Path("experiments/runs")

    if not runs_dir.exists():
        print(f"❌ 找不到实验目录: {runs_dir}")
        return

    print("\n" + "="*80)
    print("实验结果分析")
    print("="*80 + "\n")

    # 收集所有实验
    experiments = []

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        best_ckpt = exp_dir / "best.pt"

        print(f"分析: {exp_name}")

        if not best_ckpt.exists():
            print(f"  ⚠️  找不到 best.pt")
            continue

        result = analyze_checkpoint(best_ckpt)

        if result:
            experiments.append({
                'name': exp_name,
                **result
            })

            print(f"  ✓ Val Acc: {result['val_acc']:.4f}")
            print(f"  ✓ Trainable: {result['trainable_params']:,} ({result['trainable_ratio']:.2f}%)")
        print()

    # 生成对比表格
    if not experiments:
        print("❌ 没有找到有效的实验结果")
        return

    print("\n" + "="*80)
    print("对比表格")
    print("="*80 + "\n")

    # Markdown 表格
    print("| Experiment | Val Acc | Trainable Params | Ratio |")
    print("|------------|---------|-----------------|-------|")

    for exp in sorted(experiments, key=lambda x: x['val_acc'], reverse=True):
        print(f"| {exp['name']:<40} | {exp['val_acc']:.4f} | "
              f"{exp['trainable_params']:>12,} | {exp['trainable_ratio']:>5.2f}% |")

    # LaTeX 表格
    print("\n\nLaTeX 表格:")
    print("\\begin{tabular}{lrrr}")
    print("\\hline")
    print("Method & Val Acc & Trainable Params & Ratio \\\\")
    print("\\hline")

    for exp in sorted(experiments, key=lambda x: x['val_acc'], reverse=True):
        # 简化实验名称
        name = exp['name'].replace('vit_experiment_', '')
        print(f"{name} & {exp['val_acc']:.1%} & "
              f"{exp['trainable_params']/1e3:.0f}K & {exp['trainable_ratio']:.2f}\\% \\\\")

    print("\\hline")
    print("\\end{tabular}")

    # 保存 JSON
    results_file = runs_dir / "results_summary.json"
    with open(results_file, 'w') as f:
        json.dump(experiments, f, indent=2)

    print(f"\n\n✓ 结果已保存到: {results_file}")

    # 统计信息
    print("\n" + "="*80)
    print("统计信息")
    print("="*80 + "\n")

    best_exp = max(experiments, key=lambda x: x['val_acc'])
    most_efficient = min(experiments, key=lambda x: x['trainable_ratio'])

    print(f"最高准确率: {best_exp['name']} ({best_exp['val_acc']:.4f})")
    print(f"最参数高效: {most_efficient['name']} ({most_efficient['trainable_ratio']:.2f}%)")

    # LoRA-only 实验对比
    lora_exps = [e for e in experiments if 'lora_only' in e['name']]
    if lora_exps:
        print(f"\nLoRA-only 实验 (共{len(lora_exps)}个):")
        for exp in sorted(lora_exps, key=lambda x: x['trainable_params']):
            print(f"  {exp['name']}: {exp['val_acc']:.4f} "
                  f"({exp['trainable_params']:,} params)")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
