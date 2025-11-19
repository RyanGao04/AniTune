#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from anitune.data import DataConfig, build_dataloaders
from anitune.models import ModelConfig, build_model
from anitune.train_loop import evaluate
from anitune.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViT/LoRA on validation or test set")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Which split to evaluate (default: val)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = DataConfig(**cfg["data"])
    if args.data_root:
        data_cfg.root = args.data_root
    model_cfg = ModelConfig(**cfg["model"])

    loaders = build_dataloaders(data_cfg)
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    # Get number of classes from train dataset
    train_ds = train_loader.dataset
    base_ds = getattr(train_ds, "dataset", train_ds)
    if hasattr(base_ds, "classes"):
        num_classes = len(base_ds.classes)
    else:
        raise SystemExit("Unable to infer number of classes from dataset.")
    model_cfg.num_classes = num_classes

    # Select the appropriate loader
    if args.split == "test":
        if len(loaders) < 3:
            raise SystemExit("Test set not found. Please run prepare_icartoonface.py with test split enabled.")
        eval_loader = loaders[2]
        split_name = "test"
    else:
        eval_loader = val_loader
        split_name = "validation"

    model = build_model(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, eval_loader, criterion, device, amp=cfg.get("optim", {}).get("amp", True))
    print(f"{split_name.capitalize()} set metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
