# AniTune A100配置指南

## 📊 配置对比表

我为你的A100-40GB创建了4个优化配置，根据不同需求选择：

| 配置文件 | 场景 | Batch Size | LoRA Rank | 预计时间 | 显存占用 | 推荐度 |
|---------|------|-----------|-----------|---------|---------|--------|
| **lora_vitb16_a100_balanced.yaml** 🌟 | 平衡版 | 144 | 12 | ~1小时 | 25-28GB | ⭐⭐⭐⭐⭐ |
| **lora_vitb16_a100_fast.yaml** | 快速版 | 160 | 8 | ~45-60分钟 | 28-32GB | ⭐⭐⭐⭐ |
| **lora_vitb16_a100_highperf.yaml** | 高性能版 | 128 | 16 | ~1.5-2小时 | 30-35GB | ⭐⭐⭐⭐ |
| **lora_vitb16_a100_maxspeed.yaml** | 极速版 | 192 | 8 | ~30-40分钟 | 35-38GB | ⭐⭐⭐ |
| **lora_vitb16.yaml** | 原始配置 | 64 | 8 | ~2-3小时 | 18-20GB | ⭐⭐ |

## 🎯 推荐选择

### 1️⃣ 首次使用（推荐）：`lora_vitb16_a100_balanced.yaml`

**最佳平衡点，适合大多数场景**

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune
```

**优势：**
- ✓ 速度和准确率的最佳平衡
- ✓ 显存占用适中（留有安全余量）
- ✓ LoRA rank=12提供更好的表达能力
- ✓ 约1小时完成训练

### 2️⃣ 快速实验：`lora_vitb16_a100_fast.yaml`

**适合快速迭代和调试**

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_fast.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

**优势：**
- ✓ 45-60分钟完成训练
- ✓ 大batch size (160) 加快收敛
- ✓ 适合快速验证想法

### 3️⃣ 追求最佳性能：`lora_vitb16_a100_highperf.yaml`

**愿意花时间换取更高准确率**

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_highperf.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune
```

**优势：**
- ✓ LoRA rank=16 提供最强表达能力
- ✓ 15 epochs更充分训练
- ✓ 预期达到最高准确率（~88-92%）

### 4️⃣ 极限速度：`lora_vitb16_a100_maxspeed.yaml`

**快速原型和debug用**

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_maxspeed.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

**注意：**
- ⚠️ batch_size=192接近显存上限
- ⚠️ 如果OOM（内存不足），降低batch_size到160或144
- ✓ 30-40分钟完成训练

## 🔧 配置参数详解

### Batch Size（批次大小）
- **作用**：每次前向传播处理的样本数量
- **影响**：
  - 越大 → 训练越快，显存占用越高
  - 越小 → 训练越慢，梯度估计更准确
- **A100推荐范围**：128-192
- **原始配置**：64（未充分利用GPU）

### LoRA Rank（秩）
- **作用**：低秩分解矩阵的维度，控制LoRA的容量
- **影响**：
  - rank=8：~2M参数，速度快，性能略低
  - rank=12：~3M参数，平衡点
  - rank=16：~4M参数，性能高，略慢
- **推荐**：首次使用rank=12

### LoRA Alpha（缩放因子）
- **作用**：控制LoRA更新的缩放系数
- **推荐**：alpha = 2 × rank（标准设置）

### Learning Rate（学习率）
- **规则**：batch size增大时，学习率也应相应提高
- **原始**：lr=2e-4（batch_size=64）
- **调整后**：
  - batch_size=128 → lr=2.0-2.2e-4
  - batch_size=160 → lr=2.5e-4
  - batch_size=192 → lr=3.0e-4

### Num Workers（数据加载线程）
- **作用**：并行加载数据的进程数
- **推荐**：16-20（A100服务器通常有足够CPU）
- **原始**：8（可能造成GPU等待数据）

## 📈 预期性能

### 准确率预期（Top-1 on validation）

| 配置 | 预期准确率 | 训练参数量 |
|------|-----------|-----------|
| 原始 (rank=8) | 85-88% | ~2M |
| 平衡版 (rank=12) | 87-90% | ~3M |
| 高性能版 (rank=16) | 88-92% | ~4M |
| 全量微调 | 92-95% | ~88M |

### 训练曲线特点
- **前3 epochs**：快速下降期，loss从~8降到~3
- **4-7 epochs**：平稳提升期，准确率持续上升
- **8-10 epochs**：收敛期，性能趋于稳定
- **建议**：使用`--wandb`实时监控曲线

## 🚀 快速开始命令

### 一键训练（平衡版 - 推荐）

```bash
cd /workspace/AniTune
source .venv/bin/activate

PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb \
  --wandb-project AniTune \
  --wandb-run-name "a100-balanced-$(date +%m%d-%H%M)"
```

### 多个实验对比

```bash
# 实验1：快速版
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_fast.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune --wandb-run-name "fast"

# 实验2：平衡版
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune --wandb-run-name "balanced"

# 实验3：高性能版
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_highperf.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune --wandb-run-name "highperf"
```

## ⚠️ 故障排除

### 问题1：OOM (Out of Memory)

**错误信息**：`RuntimeError: CUDA out of memory`

**解决方法**：
1. 减小batch size（在config中修改`batch_size`）
2. 减小LoRA rank
3. 减少num_workers（可能是CPU内存不足）

```bash
# 临时覆盖batch size
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --batch-size 96  # 从144降到96
```

### 问题2：数据加载慢

**症状**：GPU利用率低（<50%），训练很慢

**解决方法**：
1. 增加`num_workers`
2. 检查数据是否在本地SSD（而非网络存储）
3. 使用SSD而非HDD存储数据集

### 问题3：训练不稳定/Loss震荡

**解决方法**：
1. 降低学习率（减半试试）
2. 减小batch size
3. 增加weight decay

## 🎨 自定义配置

如果想创建自己的配置：

```bash
# 复制一个现有配置
cp configs/lora_vitb16_a100_balanced.yaml configs/my_custom.yaml

# 编辑配置文件
nano configs/my_custom.yaml

# 使用自定义配置训练
PYTHONPATH=src python scripts/train.py \
  --config configs/my_custom.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

### 配置调优建议

**如果想要更快：**
- ↑ batch_size (但注意显存)
- ↑ num_workers
- ↑ learning_rate (与batch_size成比例)

**如果想要更准确：**
- ↑ lora_rank (8→12→16)
- ↑ epochs
- ↓ lora_dropout (小心过拟合)

**如果显存不足：**
- ↓ batch_size
- ↓ lora_rank
- ↓ img_size (224→192)

## 📊 监控训练

### 使用Weights & Biases（推荐）

```bash
# 首次使用需要登录
wandb login

# 训练时会自动上传指标到W&B云端
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune
```

然后在浏览器访问 https://wandb.ai 查看：
- 实时训练曲线
- GPU/CPU利用率
- 学习率变化
- 对比多个实验

### 命令行监控

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 查看训练日志
tail -f runs/lora_vitb16_a100_balanced/train.log
```

## 🎯 总结

**我的推荐优先级：**

1. 🥇 **lora_vitb16_a100_balanced.yaml** - 首选，平衡性能和速度
2. 🥈 **lora_vitb16_a100_fast.yaml** - 快速迭代实验
3. 🥉 **lora_vitb16_a100_highperf.yaml** - 最终模型训练

**开始训练吧！** 🚀

```bash
# 最简单的开始命令
cd /workspace/AniTune
source .venv/bin/activate
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

