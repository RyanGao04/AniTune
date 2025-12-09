# 🚀 A100快速开始指南

## 一键训练命令

下载好数据集后，选择一个配置直接开始训练：

### 🌟 推荐：平衡版（首选）

```bash
cd /workspace/AniTune
source .venv/bin/activate

PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune
```

**特点**：batch_size=144, rank=12, ~1小时完成

---

### ⚡ 快速版

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_fast.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

**特点**：batch_size=160, rank=8, ~45-60分钟完成

---

### 🎯 高性能版

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_highperf.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune
```

**特点**：batch_size=128, rank=16, 15 epochs, ~1.5-2小时，最高准确率

---

### 🏃 极速版

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_maxspeed.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

**特点**：batch_size=192, rank=8, ~30-40分钟完成

---

## 配置对比

| 配置 | 速度 | 准确率 | 显存 | 推荐 |
|-----|------|-------|------|------|
| 平衡版 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 25-28GB | ✅ 首选 |
| 快速版 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 28-32GB | 快速实验 |
| 高性能版 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 30-35GB | 最终模型 |
| 极速版 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 35-38GB | Debug用 |

## 完整流程

```bash
# 1. 进入项目并激活环境
cd /workspace/AniTune
source .venv/bin/activate

# 2. 准备数据（假设已下载到data/目录）
python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 --seed 42

# 3. 开始训练（平衡版）
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune

# 4. 评估模型
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

## 常用参数

在训练命令中可以添加这些参数：

```bash
--wandb                    # 启用W&B日志
--wandb-project AniTune    # W&B项目名
--num-workers 16           # 数据加载线程
--no-lora                  # 全量微调（需要更多显存）
--head-only                # 只训练分类头
```

## 监控训练

### 方法1：实时GPU监控
```bash
watch -n 1 nvidia-smi
```

### 方法2：W&B网页监控
训练时添加 `--wandb --wandb-project AniTune`，然后访问 https://wandb.ai

### 方法3：查看日志
```bash
tail -f runs/lora_vitb16_a100_balanced/train.log
```

## 故障排除

### OOM错误？
```bash
# 降低batch size
--batch-size 96
```

### 训练太慢？
```bash
# 检查数据加载
--num-workers 20
```

## 更多信息

- 详细配置说明：`CONFIG_GUIDE_CN.md`
- 完整设置指南：`SETUP_CN.md`
- 英文README：`README.md`

