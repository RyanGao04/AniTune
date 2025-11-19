#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from anitune.data import DataConfig, build_dataloaders, build_eval_loader
from anitune.models import ModelConfig, build_model
from anitune.train_loop import evaluate
from anitune.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViT/LoRA on validation set")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="Use validation split or a held-out test split (e.g., rectest).",
    )
    parser.add_argument("--test-root", type=Path, help="Root directory for the test/rectest split.")
    parser.add_argument(
        "--test-manifest",
        type=Path,
        help="Optional manifest for the test split (defaults to manifest_dir/test.txt if available).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = DataConfig(**cfg["data"])
    if args.data_root:
        data_cfg.root = args.data_root
    model_cfg = ModelConfig(**cfg["model"])

    train_loader, val_loader = build_dataloaders(data_cfg)
    train_ds = train_loader.dataset
    base_ds = getattr(train_ds, "dataset", train_ds)
    if hasattr(base_ds, "classes"):
        num_classes = len(base_ds.classes)
    else:
        raise SystemExit("Unable to infer number of classes from dataset.")
    model_cfg.num_classes = num_classes

    model = build_model(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device)
    model.to(device)

    if args.eval_split == "test" or args.test_root or args.test_manifest:
        eval_loader = build_eval_loader(
            data_cfg,
            root=args.test_root or data_cfg.root,
            manifest=args.test_manifest,
        )
    else:
        eval_loader = val_loader

    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, eval_loader, criterion, device, amp=cfg.get("optim", {}).get("amp", True))
    print(metrics)


if __name__ == "__main__":
    main()
