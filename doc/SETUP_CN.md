# AniTune 设置指南（中文）

## 项目简介

**AniTune** 是一个使用LoRA技术对Vision Transformer进行参数高效微调的项目，专注于动漫人脸识别任务。

### 核心特点
- 🎯 **任务**: 5013个动漫角色的人脸识别
- 🧠 **模型**: Vision Transformer Base (ViT-B/16)
- ⚡ **技术**: LoRA (Low-Rank Adaptation) - 只训练2%的参数
- 📊 **数据集**: iCartoonFace (大规模动漫人脸数据集)
- 🚀 **优化**: 支持混合精度训练，完美适配A100

## 快速开始（A100服务器）

### 方法1：使用自动设置脚本（推荐）

```bash
cd /workspace/AniTune
chmod +x setup_a100.sh
./setup_a100.sh
```

脚本会自动完成：
- ✓ 创建Python虚拟环境
- ✓ 安装PyTorch (CUDA版本)
- ✓ 安装所有依赖包
- ✓ 验证GPU可用性
- ✓ 创建必要的目录

### 方法2：手动设置

#### 1. 创建虚拟环境

```bash
cd /workspace/AniTune
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. 安装PyTorch（CUDA版本）

```bash
# 对于CUDA 11.8 (推荐)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 对于CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 3. 安装其他依赖

```bash
pip install -r requirements.txt
pip install -e .  # 以开发模式安装anitune包
```

#### 4. 验证安装

```bash
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

应该看到输出类似：
```
CUDA可用: True
GPU: NVIDIA A100-SXM4-80GB
```

## 数据准备

### 1. 下载iCartoonFace数据集

**选项A: 爱奇艺网盘（国内推荐）**
- 链接: https://fft.cloud.iqiyi.com/s/bUbdw5A
- 密码: 5Kv2M1

**选项B: Google Drive**
- 链接: https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW

### 2. 解压数据集

下载 `personai_icartoonface_rectrain.zip` 后解压到项目目录：

```bash
cd /workspace/AniTune
unzip personai_icartoonface_rectrain.zip -d data/
```

预期目录结构：
```
data/personai_icartoonface_rectrain/icartoonface_rectrain/
├── 00001/
│   ├── 00001_001.jpg
│   ├── 00001_002.jpg
│   └── ...
├── 00002/
│   └── ...
└── ... (共5013个角色目录)
```

### 3. 生成训练/验证集划分

```bash
source .venv/bin/activate  # 如果还没激活环境
python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 \
  --seed 42
```

这会在 `data/icartoonface/splits/` 下生成 `train.txt` 和 `val.txt` 清单文件。

## 训练模型

### 基础训练（LoRA）

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

### 针对A100的优化配置

修改 `configs/lora_vitb16.yaml` 中的参数：

```yaml
data:
  batch_size: 128        # A100可以支持更大的batch (原始64)
  num_workers: 16        # 增加数据加载速度 (原始8)

optim:
  lr: 2.0e-4
  epochs: 10
  amp: true              # 混合精度训练已启用
```

### 训练参数说明

| 参数 | 说明 |
|------|------|
| `--wandb` | 启用Weights & Biases日志 |
| `--wandb-project AniTune` | 指定W&B项目名称 |
| `--no-lora` | 关闭LoRA，进行全量微调（需更多显存） |
| `--head-only` | 只训练分类头，冻结backbone |
| `--num-workers 0` | CPU单线程数据加载（调试用） |

### 使用W&B追踪训练（推荐）

```bash
# 首次使用需要登录
wandb login

# 训练时启用W&B
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb \
  --wandb-project AniTune
```

### 训练输出

训练过程中会：
- 在 `runs/lora_vitb16/` 保存检查点
- 保存 `best.pt` (验证集最佳) 和 `last.pt` (最后一轮)
- 输出训练日志和指标

## 评估模型

### 在验证集上评估

```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16.yaml \
  --checkpoint runs/lora_vitb16/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

### 在测试集上评估

如果有测试集 `personai_icartoonface_rectest`：

```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16.yaml \
  --checkpoint runs/lora_vitb16/best.pt \
  --eval-split test \
  --test-root data/personai_icartoonface_rectest/icartoonface_rectest
```

## 项目结构详解

```
AniTune/
├── configs/                    # 配置文件
│   └── lora_vitb16.yaml       # LoRA ViT-B/16配置
├── scripts/                    # 可执行脚本
│   ├── train.py               # 训练入口
│   ├── eval.py                # 评估入口
│   └── prepare_icartoonface.py # 数据准备
├── src/anitune/               # 核心库代码
│   ├── __init__.py
│   ├── models.py              # 模型定义（ViT + 分类头）
│   ├── lora.py                # LoRA实现和注入逻辑
│   ├── data.py                # 数据集和数据加载器
│   ├── train_loop.py          # 训练循环和验证
│   └── utils.py               # 辅助函数
├── tests/                     # 单元测试
│   └── test_lora.py           # LoRA测试
├── runs/                      # 训练输出（自动生成）
├── data/                      # 数据集目录（需要下载）
├── environment.yml            # Conda环境配置
├── requirements.txt           # Python依赖
├── README.md                  # 英文README
└── SETUP_CN.md               # 本文件（中文设置指南）
```

## 核心概念解释

### LoRA (Low-Rank Adaptation)

LoRA是一种参数高效的微调方法：
- 🔒 **冻结原始模型权重** - 预训练的ViT权重保持不变
- ➕ **添加低秩矩阵** - 在attention层注入可训练的低秩分解矩阵
- 💾 **大幅减少训练参数** - 只训练约2%的参数（~2M vs 86M）
- 🎯 **保持性能** - 性能接近全量微调

配置参数：
- `lora_rank: 8` - 秩（越大参数越多，性能越好）
- `lora_alpha: 16` - 缩放因子（通常是rank的2倍）
- `lora_dropout: 0.05` - Dropout率

### Vision Transformer (ViT)

- **ViT-B/16** 表示：
  - Base大小（~86M参数）
  - Patch size 16×16
  - 输入图像被分割成 196 个patches (224÷16 = 14, 14²=196)

- **工作流程**:
  1. 图像 → Patch嵌入
  2. 添加位置编码
  3. 通过12层Transformer编码器
  4. 分类头预测5013个角色之一

## 常见问题

### Q: 训练需要多长时间？

在A100上：
- **LoRA训练**: ~1-2小时/10 epochs (batch_size=128)
- **全量微调**: ~4-6小时/10 epochs (batch_size=64)

### Q: 显存使用情况？

- **LoRA (batch_size=128)**: ~20-25GB
- **LoRA (batch_size=256)**: ~40-45GB
- **全量微调 (batch_size=64)**: ~25-30GB

### Q: 如何减少显存使用？

1. 减小batch size: `--batch-size 32`
2. 减小图像尺寸（修改config中的 `img_size`）
3. 使用梯度累积（需要修改代码）

### Q: 如何提高训练速度？

1. 增加 `num_workers` (如16或32)
2. 使用更大的batch size
3. 启用混合精度训练（默认已启用）
4. 确保数据在本地SSD而非网络存储

### Q: 离线训练怎么办？

如果无法访问互联网下载预训练权重，脚本会自动从随机初始化开始训练：
```
[INFO] Offline mode detected, skipping pretrained weights
```

### Q: 如何尝试不同的配置？

复制配置文件并修改：
```bash
cp configs/lora_vitb16.yaml configs/my_config.yaml
# 编辑 my_config.yaml
PYTHONPATH=src python scripts/train.py --config configs/my_config.yaml --data-root ...
```

## 性能基准

| 配置 | Top-1准确率 | 训练参数量 | 训练时间(A100) |
|------|------------|-----------|---------------|
| ViT-B/16 (Head Only) | ~75% | ~2M | 40分钟 |
| ViT-B/16 + LoRA(r=8) | ~85-90% | ~4M | 1-2小时 |
| ViT-B/16 (Full FT) | ~92% | ~88M | 4-6小时 |

*实际性能可能因超参数和数据集划分而异*

## 进阶用法

### 实验不同的LoRA秩

```bash
# 秩4（更少参数）
# 修改config: lora_rank: 4, lora_alpha: 8

# 秩16（更多参数）
# 修改config: lora_rank: 16, lora_alpha: 32
```

### 可视化学习曲线

如果使用W&B，可以在网页界面查看：
- 训练/验证损失
- Top-1/Top-5准确率
- 学习率变化
- GPU利用率

### 运行单元测试

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## 许可证

本项目遵循LICENSE文件中的许可证条款。

## 参考资料

- **LoRA论文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **ViT论文**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **iCartoonFace数据集**: [iCartoonFace: A Large-scale Dataset for Cartoon Face Recognition](https://github.com/luxiangju-PersonAI/iCartoonFace)

---

如有问题，请查看 `AGENTS.md` 或提交issue。祝训练顺利！🚀

