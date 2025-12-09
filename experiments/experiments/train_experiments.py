#!/usr/bin/env python
"""实验对比脚本

运行4种训练模式的对比实验：
1. Head-only fine-tuning
2. Full fine-tuning (no LoRA)
3. LoRA-only adaptation (frozen backbone) - 推荐
4. LoRA + Full (实验对比)

额外支持：Rank ablation实验
"""

import argparse
from pathlib import Path
import sys

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anitune.data import DataConfig, build_dataloaders
from anitune.train_loop import OptimConfig, run_train
from anitune.utils import set_seed, load_config

# 使用修复后的models模块
from anitune.models import ModelConfig, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="ViT 微调实验对比")
    parser.add_argument("--config", type=Path, required=True, help="基础配置文件")
    parser.add_argument("--data-root", type=Path, help="数据根目录")

    # 训练模式选择
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["head_only", "full_ft", "lora_only", "lora_full"],
        help="训练模式"
    )

    # LoRA 参数（仅在 lora_only/lora_full 模式下使用）
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # 其他参数
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, help="数据加载器worker数量")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="使用 Weights & Biases 记录")
    parser.add_argument("--wandb-project", default="AniTune-Experiments", help="Wandb项目名")
    parser.add_argument("--wandb-run-name", default=None, help="Wandb运行名称")

    # 实验名称
    parser.add_argument("--exp-name", type=str, help="实验名称（覆盖自动生成）")

    return parser.parse_args()


def get_experiment_name(args, base_run_name):
    """生成实验名称"""
    if args.exp_name:
        return args.exp_name

    # 自动生成名称
    if args.mode == "lora_only" or args.mode == "lora_full":
        return f"{base_run_name}_{args.mode}_r{args.lora_rank}"
    else:
        return f"{base_run_name}_{args.mode}"


def main():
    args = parse_args()

    # 加载基础配置
    cfg = load_config(args.config)

    # 数据配置
    data_cfg = DataConfig(**cfg["data"])
    if args.data_root:
        data_cfg.root = args.data_root
    if args.num_workers is not None:
        data_cfg.num_workers = args.num_workers

    # 模型配置
    model_cfg = ModelConfig(**cfg["model"])

    # 设置训练模式
    model_cfg.train_mode = args.mode

    # LoRA 参数
    if args.mode in ["lora_only", "lora_full"]:
        model_cfg.use_lora = True
        model_cfg.lora_rank = args.lora_rank
        model_cfg.lora_alpha = args.lora_alpha
        model_cfg.lora_dropout = args.lora_dropout
    else:
        model_cfg.use_lora = False

    # 优化器配置
    optim_cfg = OptimConfig(**cfg["optim"])

    # 设置随机种子
    set_seed(cfg.get("seed", 42))

    # 构建数据加载器
    print("\n加载数据...")
    train_loader, val_loader = build_dataloaders(data_cfg)

    # 获取类别数
    train_ds = train_loader.dataset
    base_ds = getattr(train_ds, "dataset", train_ds)
    if hasattr(base_ds, "classes"):
        num_classes = len(base_ds.classes)
    else:
        raise SystemExit("❌ 无法推断类别数")

    model_cfg.num_classes = num_classes
    print(f"✓ 数据集类别数: {num_classes}")

    # 构建模型
    model = build_model(model_cfg)

    # 移至设备
    device = torch.device(args.device)
    model.to(device)
    print(f"✓ 模型已移至: {device}")

    # 设置保存目录
    exp_name = get_experiment_name(args, cfg.get("run_name", "experiment"))
    save_dir = Path(cfg.get("save_dir", "runs")) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 保存目录: {save_dir}")

    # Wandb 初始化
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            raise SystemExit("❌ wandb 未安装。运行: pip install wandb")

        wandb_config = {
            **cfg,
            "num_classes": num_classes,
            "train_mode": args.mode,
            "lora_rank": args.lora_rank if args.mode in ["lora_only", "lora_full"] else None,
            "lora_alpha": args.lora_alpha if args.mode in ["lora_only", "lora_full"] else None,
        }

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or exp_name,
            config=wandb_config,
            tags=[args.mode],
        )
        print(f"✓ Wandb 已初始化")

    # 日志函数
    def log_fn(metrics):
        if wandb_run:
            wandb_run.log(metrics)

    # 开始训练
    print(f"\n{'='*60}")
    print(f"开始训练: {exp_name}")
    print(f"{'='*60}\n")

    metrics = run_train(
        model,
        train_loader,
        val_loader,
        device,
        optim_cfg,
        save_dir,
        log_fn=log_fn
    )

    # 打印结果
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"{'='*60}")
    print(f"最佳验证准确率: {metrics['best_val_acc']:.4f}")
    print(f"模型保存至: {save_dir / 'best.pt'}")
    print(f"{'='*60}\n")

    # 更新 Wandb summary
    if wandb_run:
        wandb_run.summary["best_val_acc"] = metrics["best_val_acc"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
