#!/bin/bash
# 显示所有A100优化配置的对比

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "          AniTune A100配置对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

configs=(
    "lora_vitb16_a100_balanced:🌟 平衡版（推荐）"
    "lora_vitb16_a100_fast:⚡ 快速版"
    "lora_vitb16_a100_highperf:🎯 高性能版"
    "lora_vitb16_a100_maxspeed:🏃 极速版"
    "lora_vitb16:📦 原始配置"
)

echo "配置文件                          Batch  Rank  Epochs  Workers  学习率"
echo "────────────────────────────────  ─────  ────  ──────  ───────  ──────"

for config_info in "${configs[@]}"; do
    IFS=':' read -r config_name display_name <<< "$config_info"
    config_file="configs/${config_name}.yaml"
    
    if [ -f "$config_file" ]; then
        batch_size=$(grep "batch_size:" "$config_file" | awk '{print $2}')
        lora_rank=$(grep "lora_rank:" "$config_file" | awk '{print $2}')
        epochs=$(grep "epochs:" "$config_file" | awk '{print $2}')
        num_workers=$(grep "num_workers:" "$config_file" | awk '{print $2}')
        lr=$(grep "lr:" "$config_file" | head -1 | awk '{print $2}')
        
        printf "%-30s    %-5s  %-4s  %-6s  %-7s  %s\n" \
            "$display_name" "$batch_size" "$lora_rank" "$epochs" "$num_workers" "$lr"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 使用示例："
echo ""
echo "   PYTHONPATH=src python scripts/train.py \\"
echo "     --config configs/lora_vitb16_a100_balanced.yaml \\"
echo "     --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain"
echo ""
echo "📖 详细说明："
echo "   - 配置指南: cat CONFIG_GUIDE_CN.md"
echo "   - 快速开始: cat QUICK_START_A100.md"
echo ""

