#!/bin/bash
# AniTune 在A100服务器上的快速设置脚本

set -e  # 遇到错误立即退出

echo "==================================="
echo "  AniTune 环境设置 (A100优化版)"
echo "==================================="

# 进入项目目录
cd /workspace/AniTune

# 检查Python版本
echo ""
echo "步骤1: 检查Python版本..."
python3 --version

# 创建虚拟环境
echo ""
echo "步骤2: 创建Python虚拟环境..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ 虚拟环境创建成功"
else
    echo "✓ 虚拟环境已存在"
fi

# 激活虚拟环境
echo ""
echo "步骤3: 激活虚拟环境..."
source .venv/bin/activate

# 升级pip
echo ""
echo "步骤4: 升级pip..."
pip install --upgrade pip

# 安装PyTorch（针对CUDA 11.8，A100推荐）
echo ""
echo "步骤5: 安装PyTorch (CUDA 11.8版本)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
echo ""
echo "步骤6: 安装其他依赖..."
pip install -r requirements.txt

# 安装anitune包（可编辑模式）
echo ""
echo "步骤7: 安装anitune包..."
pip install -e .

# 验证安装
echo ""
echo "步骤8: 验证安装..."
python3 -c "
import torch
import torchvision
import timm
print('=' * 50)
print('✓ PyTorch版本:', torch.__version__)
print('✓ TorchVision版本:', torchvision.__version__)
print('✓ TIMM版本:', timm.__version__)
print('✓ CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA版本:', torch.version.cuda)
    print('✓ GPU数量:', torch.cuda.device_count())
    print('✓ GPU名称:', torch.cuda.get_device_name(0))
    print('✓ GPU显存:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('=' * 50)
"

# 创建必要的目录
echo ""
echo "步骤9: 创建必要的目录..."
mkdir -p data runs

echo ""
echo "==================================="
echo "  ✓ 环境设置完成！"
echo "==================================="
echo ""
echo "接下来的步骤："
echo "1. 激活环境: source .venv/bin/activate"
echo "2. 下载数据集到 data/ 目录"
echo "3. 运行数据准备脚本"
echo "4. 开始训练！"
echo ""
echo "快速测试命令："
echo "  python3 -m pytest tests/  # 运行单元测试"
echo ""

