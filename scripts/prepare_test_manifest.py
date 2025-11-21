#!/usr/bin/env python
"""Prepare test set manifest for iCartoonFace rectest split.

Similar to prepare_icartoonface.py but for the test/rectest directory.
Creates a test.txt manifest file.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def gather_identities(source: Path):
    """Get all identity directories."""
    return sorted([d for d in source.iterdir() if d.is_dir()])


def main():
    parser = argparse.ArgumentParser(description="Prepare iCartoonFace test set manifest")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to icartoonface_rectest directory (test class folders)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for test manifest (should match train manifest dir)",
    )
    args = parser.parse_args()

    identities = gather_identities(args.source)
    manifest_dir = args.output / "splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    test_lines = []
    hist = Counter()
    skipped = 0

    print(f"Processing {len(identities)} identities from test set...")
    
    for label_idx, cls_dir in enumerate(identities):
        images = sorted(p for p in cls_dir.glob("*.jpg"))
        if not images:
            skipped += 1
            continue
        
        n = len(images)
        hist[n] += 1
        
        # Add all test images
        test_lines.extend([f"{img.relative_to(args.source)} {label_idx}\n" for img in images])

    # Write test manifest
    test_manifest = manifest_dir / "test.txt"
    test_manifest.write_text("".join(test_lines))

    stats = {
        "num_identities": len(identities),
        "num_images": len(test_lines),
        "histogram_top": hist.most_common(10),
        "min_images_per_id": min(hist) if hist else 0,
        "max_images_per_id": max(hist) if hist else 0,
        "skipped_empty_ids": skipped,
    }
    
    # Write stats
    (manifest_dir / "test_stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print(f"\nWrote test manifest to {test_manifest}")
    print(f"Total test images: {len(test_lines)}")


if __name__ == "__main__":
    main()

