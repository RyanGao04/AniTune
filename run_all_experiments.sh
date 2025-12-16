#!/bin/bash
# 运行所有对比实验
#
# 实验列表：
# 1. Head-only fine-tuning (baseline)
# 2. Full fine-tuning without LoRA
# 3. LoRA-only adaptation (推荐，参数高效)
# 4. Rank ablation: r ∈ {4, 8, 16, 24, 32}

set -e  # 遇到错误立即退出

# 配置
CONFIG="experiments/configs/base_experiment.yaml"
DATA_ROOT="data/personai_icartoonface_rectrain/icartoonface_rectrain"
DEVICE="cuda"
WANDB_PROJECT="AniTune-Experiments"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ViT Fine-tuning 实验对比${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查数据
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}错误: 找不到数据目录 $DATA_ROOT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 数据目录: $DATA_ROOT${NC}"
echo -e "${GREEN}✓ 配置文件: $CONFIG${NC}"
echo ""

# 询问是否使用 Wandb
read -p "是否使用 Weights & Biases 记录实验? (y/n): " use_wandb
if [ "$use_wandb" = "y" ] || [ "$use_wandb" = "Y" ]; then
    WANDB_FLAG="--wandb --wandb-project $WANDB_PROJECT"
    echo -e "${GREEN}✓ 将使用 Wandb 记录实验${NC}"
else
    WANDB_FLAG=""
    echo -e "${GREEN}✓ 不使用 Wandb${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验 1: Head-only Fine-tuning${NC}"
echo -e "${BLUE}========================================${NC}"
PYTHONPATH=src python experiments/train_experiments.py \
    --config $CONFIG \
    --data-root $DATA_ROOT \
    --mode head_only \
    --device $DEVICE \
    $WANDB_FLAG

echo ""
echo -e "${GREEN}✓ 实验 1 完成${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验 2: Full Fine-tuning (no LoRA)${NC}"
echo -e "${BLUE}========================================${NC}"
PYTHONPATH=src python experiments/train_experiments.py \
    --config $CONFIG \
    --data-root $DATA_ROOT \
    --mode full_ft \
    --device $DEVICE \
    $WANDB_FLAG

echo ""
echo -e "${GREEN}✓ 实验 2 完成${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验 3: LoRA-only (r=8, 推荐)${NC}"
echo -e "${BLUE}========================================${NC}"
PYTHONPATH=src python experiments/train_experiments.py \
    --config $CONFIG \
    --data-root $DATA_ROOT \
    --mode lora_only \
    --lora-rank 8 \
    --lora-alpha 16 \
    --device $DEVICE \
    $WANDB_FLAG

echo ""
echo -e "${GREEN}✓ 实验 3 完成${NC}"
echo ""

# Rank Ablation 实验
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验 4: Rank Ablation${NC}"
echo -e "${BLUE}========================================${NC}"

for RANK in 4 16 24 32; do
    echo ""
    echo -e "${BLUE}--- Rank = $RANK ---${NC}"

    PYTHONPATH=src python experiments/train_experiments.py \
        --config $CONFIG \
        --data-root $DATA_ROOT \
        --mode lora_only \
        --lora-rank $RANK \
        --lora-alpha $((RANK * 2)) \
        --device $DEVICE \
        $WANDB_FLAG

    echo -e "${GREEN}✓ Rank $RANK 完成${NC}"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果保存在: experiments/runs/"
echo ""
echo "查看结果:"
echo "  ls -lh experiments/runs/"
echo ""
if [ "$use_wandb" = "y" ] || [ "$use_wandb" = "Y" ]; then
    echo "在线查看: https://wandb.ai"
fi
