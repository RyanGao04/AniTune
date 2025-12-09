# 🗂️ 训练输出文件说明

## 📁 保存的文件位置

当你运行训练命令：

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune
```

会在以下位置保存文件：

```
/workspace/AniTune/
├── runs/                                    # 训练输出目录
│   └── lora_vitb16_a100_balanced/          # 实验名称（来自config中的run_name）
│       ├── best.pt                          # ✅ 最佳模型（验证集准确率最高）
│       └── last.pt                          # ✅ 最后一轮的模型
│
├── wandb/                                   # W&B本地日志（如果启用）
│   └── run-20241121_123456-abc123/
│       ├── files/
│       └── logs/
│
└── data/icartoonface/splits/                # 数据集清单（prepare时生成）
    ├── train.txt
    ├── val.txt
    └── stats.json
```

## 📊 保存的内容详解

### 1. 模型Checkpoints

#### `runs/lora_vitb16_a100_balanced/best.pt`

**最重要的文件！** 包含：

```python
{
    'model': state_dict,      # 模型权重
    'val_acc': 0.8823        # 验证集准确率
}
```

**何时保存：**
- 每个epoch验证后，如果验证准确率**超过**之前的最佳值
- 自动覆盖，只保留最好的

**如何使用：**
```bash
# 评估模型
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain

# 在测试集上评估
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --eval-split test \
  --test-root data/personai_icartoonface_rectest/icartoonface_rectest
```

**文件大小：**
- LoRA模型: ~350-400MB（包含ViT backbone + LoRA weights）
- 全量Fine-tune: ~340MB

#### `runs/lora_vitb16_a100_balanced/last.pt`

**最后一轮的模型**

```python
state_dict  # 只包含模型权重
```

**何时保存：**
- 训练结束后（第12个epoch后）
- 只保存一次

**用途：**
- 继续训练
- 比较last vs best的性能
- 调试和分析

### 2. Weights & Biases日志（如果启用）

#### 在线Dashboard

访问 https://wandb.ai/your-username/AniTune

**记录的指标（每个epoch）：**

| 指标 | 说明 | 示例值 |
|------|------|--------|
| `train_loss` | 训练损失 | 1.2345 |
| `train_acc` | 训练准确率 | 0.7856 |
| `val_loss` | 验证损失 | 1.0234 |
| `val_acc` | 验证准确率 | 0.8123 |
| `epoch` | 当前轮数 | 5 |

**额外信息：**
- ✅ 完整的配置参数（YAML内容）
- ✅ 系统信息（GPU型号、驱动版本等）
- ✅ 代码版本（git commit）
- ✅ 训练时间和持续时间
- ✅ 硬件使用率（GPU/CPU/内存）

#### 本地W&B文件

```
wandb/
└── run-20241121_123456-abc123/
    ├── files/
    │   ├── config.yaml           # 完整配置
    │   ├── requirements.txt      # Python依赖
    │   ├── wandb-metadata.json   # 元数据
    │   └── wandb-summary.json    # 最终指标总结
    └── logs/
        └── debug.log             # 调试日志
```

### 3. 终端输出

#### 训练开始时

```
Total params: 86.57M | Trainable params: 3.85M
Epoch 1/12 | train_loss=2.5432 acc=0.4532 | val_loss=2.1234 acc=0.5234
```

**显示：**
- 总参数量
- 可训练参数量（LoRA只训练约4M）
- 每个epoch的训练和验证指标

#### 训练过程中（实时进度）

```
train: 100%|████████████| 2479/2479 [15:23<00:00, 2.68it/s, loss=1.234, acc=0.823]
```

**显示：**
- 实时进度条
- 处理速度（it/s）
- 实时损失和准确率

#### 训练结束时

```
Epoch 12/12 | train_loss=0.3456 acc=0.9123 | val_loss=0.5234 acc=0.8823
Best val acc: 0.8823
```

**显示：**
- 最终指标
- 最佳验证准确率

### 4. 配置文件（输入）

你使用的配置文件也会被记录：

```yaml
# configs/lora_vitb16_a100_balanced.yaml
seed: 42
run_name: lora_vitb16_a100_balanced  # ← 决定保存目录名
save_dir: runs                       # ← 保存的根目录

data:
  root: data/...
  batch_size: 144
  ...

model:
  name: vit_base_patch16_224
  use_lora: true
  lora_rank: 12
  ...

optim:
  lr: 2.2e-4
  epochs: 12
  ...
```

## 📈 查看训练结果

### 方法1：查看终端输出（实时）

训练过程中直接在终端看到：
```
Epoch 5/12 | train_loss=0.8234 acc=0.7456 | val_loss=0.9123 acc=0.7234
```

### 方法2：查看W&B Dashboard（推荐）

```bash
# 训练时添加 --wandb
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/... \
  --wandb --wandb-project AniTune
```

然后访问: https://wandb.ai/your-username/AniTune

**可以看到：**
- 📊 实时训练曲线
- 📈 准确率变化图
- 💻 GPU使用率
- ⏱️ 训练速度
- 🔍 超参数对比（多个实验）

### 方法3：加载checkpoint查看

```python
import torch

# 加载最佳模型
checkpoint = torch.load('runs/lora_vitb16_a100_balanced/best.pt')
print(f"最佳验证准确率: {checkpoint['val_acc']:.4f}")

# 查看模型参数
state_dict = checkpoint['model']
print(f"保存的参数数量: {len(state_dict)}")
```

### 方法4：评估模型

```bash
# 在验证集上评估
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain

# 输出:
# {'loss': 0.5234, 'acc': 0.8823}
```

## 🔍 文件大小参考

| 文件 | 大小 | 说明 |
|------|------|------|
| `best.pt` | ~350MB | LoRA模型 |
| `last.pt` | ~340MB | LoRA模型 |
| `wandb/` | ~10-50MB | W&B本地日志 |
| **总计** | ~700MB | 每次训练 |

**空间规划：**
```
数据集:         ~5GB
训练输出:       ~1-2GB（多次实验）
Python环境:     ~2-3GB
──────────────────────
总需求:         ~8-10GB
你的可用空间:   186GB ✅
```

## 💡 常见操作

### 继续训练（从checkpoint）

```python
# 加载last.pt继续训练
checkpoint = torch.load('runs/.../last.pt')
model.load_state_dict(checkpoint)
# 继续训练...
```

### 对比多个实验

```bash
# 实验1：rank=8
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_fast.yaml \
  --data-root ... \
  --wandb --wandb-run-name "rank8"

# 实验2：rank=12
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root ... \
  --wandb --wandb-run-name "rank12"

# 实验3：rank=16
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_highperf.yaml \
  --data-root ... \
  --wandb --wandb-run-name "rank16"
```

在W&B中可以并排对比三个实验的曲线！

### 备份重要模型

```bash
# 备份最佳模型
cp runs/lora_vitb16_a100_balanced/best.pt \
   backups/best_20241121_acc8823.pt

# 或上传到云存储
# aws s3 cp runs/.../best.pt s3://bucket/models/
# gsutil cp runs/.../best.pt gs://bucket/models/
```

### 清理旧的实验

```bash
# 删除不需要的实验
rm -rf runs/old_experiment/

# 只保留best.pt，删除last.pt节省空间
rm runs/*/last.pt

# 清理W&B本地缓存
rm -rf wandb/
```

## ⚙️ 自定义保存位置

### 修改保存目录

编辑配置文件：

```yaml
# configs/lora_vitb16_a100_balanced.yaml
run_name: my_experiment_20241121  # ← 修改实验名称
save_dir: experiments             # ← 修改保存根目录
```

保存到：`experiments/my_experiment_20241121/`

### 使用时间戳

```bash
# 自动添加时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root ... \
  --wandb --wandb-run-name "balanced_${TIMESTAMP}"
```

## 🚨 注意事项

### 1. 不会自动保存的内容

❌ **训练集的预测结果** - 太大，不保存
❌ **中间epoch的checkpoints** - 只保存best和last
❌ **优化器状态** - 除非你修改代码添加
❌ **学习率调度器状态** - 当前没有使用

### 2. 保存频率

- ✅ **每个epoch后**: 检查是否需要更新best.pt
- ✅ **训练结束时**: 保存last.pt
- ✅ **实时W&B日志**: 每个epoch后上传

### 3. 磁盘空间管理

```bash
# 查看runs目录大小
du -sh runs/

# 查看每个实验的大小
du -sh runs/*/

# 定期清理
# 保留最近3个实验，删除其他
ls -t runs/ | tail -n +4 | xargs -I {} rm -rf runs/{}
```

## 📊 输出示例

### 完整训练输出

```
Total params: 86.57M | Trainable params: 3.85M

Epoch 1/12 | train_loss=2.5432 acc=0.4532 | val_loss=2.1234 acc=0.5234
Epoch 2/12 | train_loss=1.8923 acc=0.6123 | val_loss=1.7456 acc=0.6345
Epoch 3/12 | train_loss=1.4567 acc=0.7012 | val_loss=1.4123 acc=0.7123
Epoch 4/12 | train_loss=1.1234 acc=0.7678 | val_loss=1.1567 acc=0.7678
Epoch 5/12 | train_loss=0.9123 acc=0.8123 | val_loss=0.9876 acc=0.8012
Epoch 6/12 | train_loss=0.7456 acc=0.8456 | val_loss=0.8567 acc=0.8234
Epoch 7/12 | train_loss=0.6123 acc=0.8678 | val_loss=0.7456 acc=0.8456
Epoch 8/12 | train_loss=0.5234 acc=0.8812 | val_loss=0.6789 acc=0.8567
Epoch 9/12 | train_loss=0.4567 acc=0.8934 | val_loss=0.6234 acc=0.8678
Epoch 10/12 | train_loss=0.4012 acc=0.9012 | val_loss=0.5789 acc=0.8756
Epoch 11/12 | train_loss=0.3678 acc=0.9089 | val_loss=0.5456 acc=0.8789
Epoch 12/12 | train_loss=0.3456 acc=0.9123 | val_loss=0.5234 acc=0.8823

Best val acc: 0.8823
```

### 文件列表

```bash
$ ls -lh runs/lora_vitb16_a100_balanced/
total 700M
-rw-r--r-- 1 user group 353M Nov 21 14:32 best.pt
-rw-r--r-- 1 user group 340M Nov 21 15:45 last.pt
```

## ✅ 总结

**运行一次训练会保存：**

1. ✅ **best.pt** - 最重要！最佳模型
2. ✅ **last.pt** - 最后一轮模型
3. ✅ **W&B在线日志** - 完整的训练历史和可视化（如果启用）
4. ✅ **终端输出** - 实时指标（建议重定向到文件保存）

**推荐操作：**
```bash
# 训练时启用W&B + 重定向输出
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune \
  2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

这样你会有完整的记录！🎉

