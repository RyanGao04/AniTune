#!/usr/bin/env python
"""Prepare iCartoonFace recognition data: stats + train/val split manifests.

Creates train/val manifest files (relative paths and label ids) without copying images.
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
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio per identity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ids", type=int, default=None, help="Optional: limit number of identities for a smoke split")
    parser.add_argument("--max-per-id", type=int, default=None, help="Optional: cap images per identity (after shuffling)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    identities = gather_identities(args.source)
    if args.max_ids:
        identities = identities[: args.max_ids]
    manifest_dir = args.output / "splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_lines, val_lines = [], []
    hist = Counter()
    skipped = 0

    for label_idx, cls_dir in enumerate(identities):
        images = sorted(p for p in cls_dir.glob("*.jpg"))
        if not images:
            skipped += 1
            continue
        rng.shuffle(images)
        if args.max_per_id:
            images = images[: args.max_per_id]
        n = len(images)
        hist[n] += 1
        val_count = max(1, int(n * args.val_ratio)) if n > 1 else 0
        val_count = min(val_count, n - 1) if n > 1 else val_count
        val_imgs = images[:val_count]
        train_imgs = images[val_count:]
        if not train_imgs:
            train_imgs = val_imgs
            val_imgs = []
        train_lines.extend([f"{img.relative_to(args.source)} {label_idx}\n" for img in train_imgs])
        val_lines.extend([f"{img.relative_to(args.source)} {label_idx}\n" for img in val_imgs])

    (manifest_dir / "train.txt").write_text("".join(train_lines))
    (manifest_dir / "val.txt").write_text("".join(val_lines))

    stats = {
        "num_identities": len(identities),
        "num_images": len(train_lines) + len(val_lines),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "histogram_top": hist.most_common(10),
        "min_images_per_id": min(hist) if hist else 0,
        "max_images_per_id": max(hist) if hist else 0,
        "skipped_empty_ids": skipped,
        "train_images": len(train_lines),
        "val_images": len(val_lines),
    }
    (manifest_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print(f"Wrote manifests to {manifest_dir}")


if __name__ == "__main__":
    main()
