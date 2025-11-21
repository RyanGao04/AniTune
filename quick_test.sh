#!/bin/bash
# 快速测试脚本 - 验证环境和运行LoRA测试

echo "🧪 运行AniTune快速测试..."
echo ""

cd /workspace/AniTune
source .venv/bin/activate

# 运行单元测试
echo "步骤1: 运行LoRA单元测试..."
python -m pytest tests/test_lora.py -v

echo ""
echo "✅ 测试完成！如果所有测试通过，你的环境已经配置好了。"
echo ""
echo "接下来的步骤："
echo "1. 下载iCartoonFace数据集（见 SETUP_CN.md）"
echo "2. 运行数据准备脚本"
echo "3. 开始训练！"

