#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from anitune.data import DataConfig, build_dataloaders
from anitune.models import ModelConfig, build_model
from anitune.train_loop import evaluate
from anitune.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViT/LoRA on validation set")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = DataConfig(**cfg["data"])
    if args.data_root:
        data_cfg.root = args.data_root
    model_cfg = ModelConfig(**cfg["model"])

    train_loader, val_loader = build_dataloaders(data_cfg)
    num_classes = len(train_loader.dataset.dataset.classes)
    model_cfg.num_classes = num_classes

    model = build_model(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, val_loader, criterion, device, amp=cfg.get("optim", {}).get("amp", True))
    print(metrics)


if __name__ == "__main__":
    main()
