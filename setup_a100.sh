#!/bin/bash
# AniTune A100 虚拟机环境快速设置脚本（无虚拟环境版本）
# 适用于已配置 CUDA 的 A100 服务器/虚拟机

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  AniTune 环境设置 (A100 虚拟机版)"
echo "=========================================="
echo ""

# 检查当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "项目目录: $SCRIPT_DIR"
echo ""

# ========================================
# 步骤 1: 检查 Python 和 CUDA
# ========================================
echo "步骤 1: 检查环境..."
echo "----------------------------------------"

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python: $PYTHON_VERSION"

# 检查 CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo "✓ CUDA: $CUDA_VERSION"
else
    echo "⚠️  警告: 未找到 nvcc (CUDA 编译器)"
    echo "  如果已安装 CUDA，请确保在 PATH 中"
fi

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  警告: 未找到 nvidia-smi"
fi

echo ""

# ========================================
# 步骤 2: 升级 pip
# ========================================
echo "步骤 2: 升级 pip..."
echo "----------------------------------------"
python3 -m pip install --upgrade pip
echo "✓ pip 已升级"
echo ""

# ========================================
# 步骤 3: 安装 PyTorch (CUDA 版本)
# ========================================
echo "步骤 3: 安装 PyTorch..."
echo "----------------------------------------"

# 检测 CUDA 版本并选择对应的 PyTorch
# 优先级: CUDA 12.1 > 11.8 > CPU
if [[ -n "$CUDA_VERSION" ]]; then
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        echo "检测到 CUDA 12.x, 安装 PyTorch (CUDA 12.1 版本)..."
        python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        echo "检测到 CUDA 11.x, 安装 PyTorch (CUDA 11.8 版本)..."
        python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠️  未知的 CUDA 版本 $CUDA_VERSION，安装默认 PyTorch..."
        python3 -m pip install torch torchvision
    fi
else
    echo "未检测到 CUDA，安装 CPU 版本的 PyTorch..."
    python3 -m pip install torch torchvision
fi

echo "✓ PyTorch 安装完成"
echo ""

# ========================================
# 步骤 4: 安装项目依赖
# ========================================
echo "步骤 4: 安装项目依赖..."
echo "----------------------------------------"

# 检查 requirements.txt 是否存在
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  警告: requirements.txt 不存在，创建基础版本..."
    cat > requirements.txt << 'EOF'
# Core dependencies
pyyaml>=6.0
timm>=0.9.0
tqdm>=4.65.0
wandb>=0.15.0

# Analysis and visualization
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0

# Utilities
numpy>=1.24.0
EOF
fi

python3 -m pip install -r requirements.txt

echo "✓ 项目依赖安装完成"
echo ""

# ========================================
# 步骤 5: 安装 anitune 包（可编辑模式）
# ========================================
echo "步骤 5: 安装 anitune 包..."
echo "----------------------------------------"

# 检查 setup.py 是否存在
if [ -f "setup.py" ]; then
    python3 -m pip install -e .
    echo "✓ anitune 包已安装（可编辑模式）"
else
    echo "⚠️  警告: setup.py 不存在，跳过包安装"
    echo "  将使用 PYTHONPATH 方式运行"
fi

echo ""

# ========================================
# 步骤 6: 验证安装
# ========================================
echo "步骤 6: 验证安装..."
echo "----------------------------------------"

python3 -c "
import sys
import torch
import torchvision
import timm
import yaml
import wandb

print('=' * 60)
print('环境验证')
print('=' * 60)
print()

# Python 信息
print('Python 版本:', sys.version.split()[0])
print()

# PyTorch 信息
print('PyTorch 版本:', torch.__version__)
print('TorchVision 版本:', torchvision.__version__)
print('TIMM 版本:', timm.__version__)
print()

# CUDA 信息
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA 版本:', torch.version.cuda)
    print('cuDNN 版本:', torch.backends.cudnn.version())
    print()
    print('GPU 信息:')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}')
        print(f'    显存: {props.total_memory / 1024**3:.1f} GB')
        print(f'    计算能力: {props.major}.{props.minor}')
else:
    print('⚠️  CUDA 不可用，将使用 CPU 训练（非常慢）')

print()
print('=' * 60)
print('✓ 环境验证完成')
print('=' * 60)
" || {
    echo "❌ 验证失败！请检查安装"
    exit 1
}

echo ""

# ========================================
# 步骤 7: 创建必要的目录
# ========================================
echo "步骤 7: 创建必要的目录..."
echo "----------------------------------------"

mkdir -p data
mkdir -p runs
mkdir -p experiments/runs
mkdir -p logs

echo "✓ 目录结构:"
echo "  data/              - 数据集存放目录"
echo "  runs/              - 训练输出目录"
echo "  experiments/runs/  - 实验结果目录"
echo "  logs/              - 日志目录"
echo ""

# ========================================
# 步骤 8: 测试模型构建
# ========================================
echo "步骤 8: 测试模型构建（快速验证）..."
echo "----------------------------------------"

PYTHONPATH=src python3 -c "
from anitune.models import ModelConfig, build_model

print('测试 LoRA-only 模式...')
cfg = ModelConfig(
    name='vit_base_patch16_224',
    num_classes=100,
    pretrained=False,  # 不下载权重，仅测试结构
    train_mode='lora_only',
    lora_rank=8,
)

model = build_model(cfg)
print('✓ 模型构建成功！')
" 2>&1 | grep -E "✓|LoRA|参数|训练模式" || {
    echo "⚠️  模型测试失败，但可以继续"
}

echo ""

# ========================================
# 完成
# ========================================
echo "=========================================="
echo "  ✓ A100 环境设置完成！"
echo "=========================================="
echo ""
echo "📋 下一步操作："
echo ""
echo "1️⃣  准备数据集"
echo "   bash check_data.sh  # 检查数据"
echo ""
echo "2️⃣  生成数据清单（如果还没有）"
echo "   PYTHONPATH=src python scripts/prepare_icartoonface.py \\"
echo "     --source data/personai_icartoonface_rectrain/icartoonface_rectrain \\"
echo "     --output data/icartoonface \\"
echo "     --val-ratio 0.1 --seed 42"
echo ""
echo "3️⃣  运行实验"
echo "   # 单个实验"
echo "   ./experiments/run_single_experiment.sh lora_only 8"
echo ""
echo "   # 完整对比实验"
echo "   ./experiments/run_all_experiments.sh"
echo ""
echo "4️⃣  查看结果"
echo "   python experiments/analyze_results.py"
echo ""
echo "📚 文档："
echo "   cat experiments/INDEX.md  # 文档导航"
echo "   cat FINAL_SUMMARY.md      # 完整总结"
echo ""
echo "🔧 有用的命令："
echo "   nvidia-smi            # 查看 GPU 状态"
echo "   htop                  # 查看 CPU/内存"
echo "   wandb login           # 登录 Wandb（可选）"
echo ""
