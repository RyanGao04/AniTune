#!/bin/bash
# 在验证集上运行错误分析

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AniTune 错误分析 - 验证集"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 配置
CONFIG="configs/lora_vitb16_a100_balanced.yaml"
CHECKPOINT="best.pt"  # 使用完整 checkpoint
# 数据根目录：包含角色文件夹的目录（配置文件中的 manifest_dir 会自动使用 val.txt）
DATA_ROOT="data/personai_icartoonface_rec/personai_icartoonface_rectrain/icartoonface_rectrain"
OUTPUT_DIR="error_analysis_val"
SPLIT="val"

# 检查文件
if [ ! -f "$CONFIG" ]; then
    echo "❌ 配置文件不存在: $CONFIG"
    exit 1
fi

# 如果 best.pt 不在当前目录，尝试从 runs 目录查找
if [ ! -f "$CHECKPOINT" ]; then
    # 尝试查找 runs 目录下的 best.pt
    if [ -f "runs/lora_vitb16_a100_balanced/best.pt" ]; then
        CHECKPOINT="runs/lora_vitb16_a100_balanced/best.pt"
        echo "✓ 找到 checkpoint: $CHECKPOINT"
    else
        echo "❌ Checkpoint 不存在: $CHECKPOINT"
        echo "💡 提示：请确保 best.pt 已上传到服务器"
        exit 1
    fi
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 数据目录不存在: $DATA_ROOT"
    echo "💡 提示：确保训练集已下载"
    exit 1
fi

echo "📋 配置信息:"
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data root:  $DATA_ROOT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Split:      $SPLIT"
echo ""

# 激活虚拟环境
source .venv/bin/activate

echo "🚀 开始错误分析..."
echo ""

# 运行错误分析
PYTHONPATH=src python scripts/error_analysis.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ 错误分析完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 生成的文件:"
echo "  🔥 混淆矩阵:         $OUTPUT_DIR/confusion_matrix_top50_${SPLIT}.png"
echo "  📊 混淆类别对:       $OUTPUT_DIR/confused_pairs_${SPLIT}.png"
echo "  📈 每类准确率:       $OUTPUT_DIR/per_class_accuracy_${SPLIT}.png"
echo "  🖼️  错误样本可视化:   $OUTPUT_DIR/error_samples_visualization.png"
echo "  📝 统计报告:         $OUTPUT_DIR/error_statistics_${SPLIT}.json"
echo ""
echo "💡 查看结果:"
echo "  ls -lh $OUTPUT_DIR/"
echo ""
echo "📖 详细文档:"
echo "  cat ERROR_ANALYSIS.md"
echo ""

