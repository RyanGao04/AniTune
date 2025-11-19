#!/usr/bin/env python
"""Prepare iCartoonFace recognition data: stats + train/val/test split manifests.

Creates train/val/test manifest files (relative paths and label ids) without copying images.
Use the manifests via DataConfig.manifest_dir.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def gather_identities(source: Path):
    return sorted([d for d in source.iterdir() if d.is_dir()])


def main():
    parser = argparse.ArgumentParser(description="Prepare iCartoonFace manifests")
    parser.add_argument("--source", type=Path, required=True, help="Path to icartoonface_rectrain directory (class folders)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for manifests/stats")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Training ratio per identity (default: 0.6)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per identity (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test ratio per identity (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate ratios sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.6f}")

    rng = random.Random(args.seed)
    identities = gather_identities(args.source)
    manifest_dir = args.output / "splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_lines, val_lines, test_lines = [], [], []
    hist = Counter()
    skipped = 0

    for label_idx, cls_dir in enumerate(identities):
        images = sorted(p for p in cls_dir.glob("*.jpg"))
        if not images:
            skipped += 1
            continue
        rng.shuffle(images)
        n = len(images)
        hist[n] += 1

        # Calculate split points based on ratios
        if n == 0:
            continue
        elif n == 1:
            # Only one image: put in training set
            train_imgs = images
            val_imgs = []
            test_imgs = []
        elif n == 2:
            # Two images: train and val
            train_imgs = images[:1]
            val_imgs = images[1:2]
            test_imgs = []
        else:
            # Three or more images: use ratios
            train_count = max(1, int(n * args.train_ratio))
            val_count = max(1, int(n * args.val_ratio))
            # Ensure we have at least one for each split and don't exceed total
            train_count = min(train_count, n - 2)  # Leave at least 2 for val+test
            val_count = min(val_count, n - train_count - 1)  # Leave at least 1 for test
            test_count = n - train_count - val_count

            # Split images
            train_imgs = images[:train_count]
            val_imgs = images[train_count : train_count + val_count]
            test_imgs = images[train_count + val_count :]

        train_lines.extend([f"{img.relative_to(args.source)} {label_idx}\n" for img in train_imgs])
        val_lines.extend([f"{img.relative_to(args.source)} {label_idx}\n" for img in val_imgs])
        test_lines.extend([f"{img.relative_to(args.source)} {label_idx}\n" for img in test_imgs])

    (manifest_dir / "train.txt").write_text("".join(train_lines))
    (manifest_dir / "val.txt").write_text("".join(val_lines))
    (manifest_dir / "test.txt").write_text("".join(test_lines))

    stats = {
        "num_identities": len(identities),
        "num_images": len(train_lines) + len(val_lines) + len(test_lines),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "histogram_top": hist.most_common(10),
        "min_images_per_id": min(hist) if hist else 0,
        "max_images_per_id": max(hist) if hist else 0,
        "skipped_empty_ids": skipped,
        "train_images": len(train_lines),
        "val_images": len(val_lines),
        "test_images": len(test_lines),
    }
    (manifest_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print(f"Wrote manifests to {manifest_dir}")


if __name__ == "__main__":
    main()
