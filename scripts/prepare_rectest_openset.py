#!/usr/bin/env python
"""处理 iCartoonFace Recognition 测试集（开放集格式）

官方测试集格式：filename x1 y1 x2 y2 label_id
- label_id >= 0: 训练集中存在的类别
- label_id == -1: 不属于任何训练类别（开放集样本）

此脚本生成：
1. 裁剪后的人脸图片（使用边界框）
2. closed-set 测试清单（仅包含训练集类别）
3. open-set 测试清单（包含所有样本）
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def parse_label_file(label_file: Path):
    """解析标签文件

    Returns:
        list of dict: [{
            'filename': str,
            'bbox': (x1, y1, x2, y2),
            'label_id': int
        }, ...]
    """
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Warning: 跳过格式错误的行: {line.strip()}")
                continue

            filename, x1, y1, x2, y2, label_id = parts
            samples.append({
                'filename': filename,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'label_id': int(label_id)
            })

    return samples


def crop_and_save(image_path: Path, bbox: tuple, output_path: Path, margin: float = 0.1):
    """裁剪人脸区域并保存

    Args:
        image_path: 原始图片路径
        bbox: (x1, y1, x2, y2) 边界框
        output_path: 输出路径
        margin: 边界框扩展比例（默认扩展10%）
    """
    img = Image.open(image_path)
    x1, y1, x2, y2 = bbox

    # 扩展边界框
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - w * margin))
    y1 = max(0, int(y1 - h * margin))
    x2 = min(img.width, int(x2 + w * margin))
    y2 = min(img.height, int(y2 + h * margin))

    # 裁剪
    face = img.crop((x1, y1, x2, y2))

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    face.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="处理 iCartoonFace Recognition 测试集")
    parser.add_argument(
        "--label-file",
        type=Path,
        required=True,
        help="测试集标签文件路径（格式：filename x1 y1 x2 y2 label_id）"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="测试集图片目录"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录（裁剪后的图片和清单）"
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="是否裁剪人脸区域（如果不裁剪，直接使用原图）"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="裁剪时的边界框扩展比例（默认0.1）"
    )
    args = parser.parse_args()

    # 解析标签文件
    print("解析标签文件...")
    samples = parse_label_file(args.label_file)
    print(f"✓ 找到 {len(samples)} 个测试样本")

    # 统计
    label_counter = Counter(s['label_id'] for s in samples)
    closed_set_count = sum(1 for s in samples if s['label_id'] >= 0)
    open_set_count = sum(1 for s in samples if s['label_id'] == -1)

    print(f"\n统计信息：")
    print(f"  Closed-set 样本（训练集类别）: {closed_set_count}")
    print(f"  Open-set 样本（未知类别）: {open_set_count}")
    print(f"  唯一类别数（不含-1）: {len([k for k in label_counter.keys() if k >= 0])}")

    # 创建输出目录
    output_image_dir = args.output_dir / "images"
    output_manifest_dir = args.output_dir / "splits"
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    # 处理图片
    closed_set_lines = []
    open_set_lines = []

    print(f"\n{'裁剪' if args.crop else '复制'}图片...")
    for sample in tqdm(samples):
        filename = sample['filename']
        bbox = sample['bbox']
        label_id = sample['label_id']

        # 输入图片路径
        input_path = args.image_dir / filename
        if not input_path.exists():
            print(f"Warning: 找不到图片 {input_path}")
            continue

        # 输出图片路径
        if args.crop:
            # 裁剪后保存到 images/ 目录
            output_path = output_image_dir / filename
            crop_and_save(input_path, bbox, output_path, args.margin)
            relative_path = f"images/{filename}"
        else:
            # 不裁剪，使用原始路径
            relative_path = filename

        # 生成清单行
        manifest_line = f"{relative_path} {label_id}\n"

        # Closed-set: 仅包含训练集类别
        if label_id >= 0:
            closed_set_lines.append(manifest_line)

        # Open-set: 包含所有样本
        open_set_lines.append(manifest_line)

    # 保存清单
    print("\n保存测试清单...")

    # Closed-set 清单
    closed_set_manifest = output_manifest_dir / "test_closedset.txt"
    closed_set_manifest.write_text("".join(closed_set_lines))
    print(f"✓ Closed-set 清单: {closed_set_manifest} ({len(closed_set_lines)} 样本)")

    # Open-set 清单
    open_set_manifest = output_manifest_dir / "test_openset.txt"
    open_set_manifest.write_text("".join(open_set_lines))
    print(f"✓ Open-set 清单: {open_set_manifest} ({len(open_set_lines)} 样本)")

    # 保存统计信息
    stats = {
        "total_samples": len(samples),
        "closed_set_samples": closed_set_count,
        "open_set_samples": open_set_count,
        "unique_classes": len([k for k in label_counter.keys() if k >= 0]),
        "label_distribution": dict(label_counter.most_common(20)),
        "cropped": args.crop,
        "margin": args.margin if args.crop else None,
    }

    stats_file = output_manifest_dir / "test_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2))
    print(f"✓ 统计信息: {stats_file}")

    print("\n完成！")
    print(f"\n使用方法：")
    print(f"  # Closed-set 评估（仅训练集类别）")
    print(f"  PYTHONPATH=src python scripts/eval.py \\")
    print(f"    --config configs/lora_vitb16.yaml \\")
    print(f"    --checkpoint runs/lora_vitb16/best.pt \\")
    print(f"    --data-root {args.output_dir / 'images' if args.crop else args.image_dir} \\")
    print(f"    --test-manifest {closed_set_manifest}")
    print()
    print(f"  # Open-set 评估（包含未知类别，需要修改评估脚本）")
    print(f"  # 使用 {open_set_manifest}")


if __name__ == "__main__":
    main()
