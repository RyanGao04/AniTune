#!/bin/bash
# 运行单个实验的便捷脚本
#
# 用法:
#   ./run_single_experiment.sh head_only
#   ./run_single_experiment.sh full_ft
#   ./run_single_experiment.sh lora_only 8
#   ./run_single_experiment.sh lora_only 16

MODE=$1
RANK=${2:-8}  # 默认 rank=8

if [ -z "$MODE" ]; then
    echo "用法: $0 <mode> [rank]"
    echo ""
    echo "模式选择:"
    echo "  head_only  - 仅训练分类头"
    echo "  full_ft    - 全量微调（无LoRA）"
    echo "  lora_only  - LoRA-only（推荐）"
    echo "  lora_full  - LoRA + 全量（实验对比）"
    echo ""
    echo "示例:"
    echo "  $0 head_only"
    echo "  $0 lora_only 8"
    echo "  $0 full_ft"
    exit 1
fi

CONFIG="experiments/configs/base_experiment.yaml"
DATA_ROOT="data/personai_icartoonface_rectrain/icartoonface_rectrain"
DEVICE="cuda"

echo "=========================================="
echo "运行实验: $MODE"
if [ "$MODE" = "lora_only" ] || [ "$MODE" = "lora_full" ]; then
    echo "LoRA Rank: $RANK"
fi
echo "=========================================="
echo ""

PYTHONPATH=src python experiments/train_experiments.py \
    --config $CONFIG \
    --data-root $DATA_ROOT \
    --mode $MODE \
    --lora-rank $RANK \
    --lora-alpha $((RANK * 2)) \
    --device $DEVICE \
    --wandb \
    --wandb-project AniTune-Experiments

echo ""
echo "=========================================="
echo "实验完成!"
echo "=========================================="
