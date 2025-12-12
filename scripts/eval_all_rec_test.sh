#!/bin/bash
# 批量评估所有实验模型的 RecTest 结果

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "批量 RecTest 评估"
echo "=========================================="
echo ""

# 配置
RECTEST_INFO="data/icartoonface_rectest_info.txt"
RECTEST_DIR="data/personai_icartoonface_rec/personai_icartoonface_rectest/icartoonface_rectest"
BASE_CONFIG="experiments/configs/base_experiment.yaml"
BATCH_SIZE=128
DEVICE="cpu"

# 创建输出目录
mkdir -p evaluation_results/rec_test

# 定义模型列表
declare -a MODELS=(
    "experiments/runs/vit_experiment_head_only/best.pt:head_only"
    "experiments/runs/vit_experiment_lora_only_r8/best.pt:lora_only_r8"
    "experiments/runs/vit_experiment_full_ft/best.pt:full_ft"
)

# 评估每个模型
for model_info in "${MODELS[@]}"; do
    IFS=':' read -r checkpoint model_name <<< "$model_info"

    if [ ! -f "$checkpoint" ]; then
        echo "⚠️  跳过 $model_name: 文件不存在 ($checkpoint)"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "评估: $model_name"
    echo "检查点: $checkpoint"
    echo "=========================================="

    output_bin="evaluation_results/rec_test/${model_name}_rectest.bin"

    # 运行评估（使用智能版本）
    PYTHONPATH=src python3 scripts/eval_rec_test_smart.py \
        --checkpoint "$checkpoint" \
        --rectest-info "$RECTEST_INFO" \
        --rectest-dir "$RECTEST_DIR" \
        --output-bin "$output_bin" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE"

    echo "✓ 完成: $model_name"
done

echo ""
echo "=========================================="
echo "所有评估完成！"
echo "=========================================="
echo ""
echo "结果文件:"
ls -lh evaluation_results/rec_test/*_result.json

echo ""
echo "汇总结果:"
echo "=========================================="
for result_file in evaluation_results/rec_test/*_result.json; do
    if [ -f "$result_file" ]; then
        model_name=$(basename "$result_file" _rectest_result.json)
        score=$(python3 -c "import json; print(json.load(open('$result_file'))['score'])")
        printf "%-25s: %.2f%%\n" "$model_name" "$score"
    fi
done
echo "=========================================="
