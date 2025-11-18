#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from anitune.data import DataConfig, build_dataloaders
from anitune.models import ModelConfig, build_model, freeze_backbone, enable_full_finetune
from anitune.train_loop import OptimConfig, run_train
from anitune.utils import set_seed, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT/LoRA on iCartoonFace-style data")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--data-root", type=Path, help="Override data.root from config")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA; full model trains")
    parser.add_argument("--head-only", action="store_true", help="Freeze backbone and train head only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="AniTune", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", default=None, help="Weights & Biases run name")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = DataConfig(**cfg["data"])
    if args.data_root:
        data_cfg.root = args.data_root
    model_cfg = ModelConfig(**cfg["model"])
    optim_cfg = OptimConfig(**cfg["optim"])

    if args.no_lora:
        model_cfg.use_lora = False

    set_seed(cfg.get("seed", 42))

    train_loader, val_loader = build_dataloaders(data_cfg)
    num_classes = len(train_loader.dataset.dataset.classes)
    model_cfg.num_classes = num_classes

    model = build_model(model_cfg)
    if args.head_only:
        freeze_backbone(model, unfreeze_head=True)
    elif not model_cfg.use_lora:
        enable_full_finetune(model)

    device = torch.device(args.device)
    model.to(device)

    save_dir = Path(cfg.get("save_dir", "runs")) / cfg.get("run_name", "experiment")
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            raise SystemExit("wandb not installed. Install via `pip install wandb` or disable --wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or cfg.get("run_name", "experiment"),
            config={**cfg, "num_classes": num_classes},
        )

    def log_fn(metrics):
        if wandb_run:
            wandb_run.log(metrics)

    metrics = run_train(model, train_loader, val_loader, device, optim_cfg, save_dir, log_fn=log_fn)
    print(f"Best val acc: {metrics['best_val_acc']:.4f}")
    if wandb_run:
        wandb_run.summary["best_val_acc"] = metrics["best_val_acc"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
