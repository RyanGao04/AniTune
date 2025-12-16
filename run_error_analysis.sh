#!/bin/bash
# 快速运行错误分析的脚本

set -e  # 出错时退出

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AniTune 错误分析"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 配置
CONFIG="configs/lora_vitb16_a100_balanced.yaml"
CHECKPOINT="best_lora_only.pt"  # 使用 LoRA-only checkpoint
DATA_ROOT="data/personai_icartoonface_rec/personai_icartoonface_rectest/icartoonface_rectest"
OUTPUT_DIR="error_analysis_test"
SPLIT="test"  # 使用测试集
LORA_ONLY="--lora-only"  # 使用 LoRA-only 模式

# 检查文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "❌ 配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint 不存在: $CHECKPOINT"
    echo "💡 提示：请先下载训练好的模型"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 数据目录不存在: $DATA_ROOT"
    exit 1
fi

echo "📋 配置信息:"
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data root:  $DATA_ROOT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Split:      $SPLIT"
echo ""

read -p "是否继续？(y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 开始错误分析..."
echo ""

# 运行错误分析
PYTHONPATH=src python scripts/error_analysis.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  $LORA_ONLY

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
echo "  open $OUTPUT_DIR/"
echo ""
echo "📖 详细文档:"
echo "  cat ERROR_ANALYSIS.md"
echo ""

