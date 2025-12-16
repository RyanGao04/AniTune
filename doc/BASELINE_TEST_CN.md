# 🧪 Base模型Baseline测试指南

## 为什么要测试base模型？

在fine-tuning之前测试预训练的base模型非常重要：

1. **建立baseline**：了解起点，对比fine-tuning的提升效果
2. **验证数据加载**：确保数据集正确加载
3. **理解任务难度**：动漫人脸识别 vs ImageNet预训练的差距

## 📊 测试集标签格式说明

### 格式
每行一个样本，格式为：
```
相对路径 标签ID
```

### 示例
```
00001/00001_001.jpg 0
00001/00001_002.jpg 0
00002/00002_001.jpg 1
00003/00003_001.jpg 2
...
```

- **相对路径**：相对于数据根目录（如 `icartoonface_rectrain/`）
- **标签ID**：整数，0到5012（共5013个角色）
- **每个ID**：代表一个动漫角色的身份

### 数据集结构

```
data/
├── personai_icartoonface_rectrain/      # 训练+验证集
│   └── icartoonface_rectrain/
│       ├── 00001/                       # 角色ID = 0
│       │   ├── 00001_001.jpg
│       │   ├── 00001_002.jpg
│       │   └── ...
│       ├── 00002/                       # 角色ID = 1
│       │   └── ...
│       └── ... (5013个文件夹)
│
└── personai_icartoonface_rectest/       # 测试集（可选）
    └── icartoonface_rectest/
        ├── 00001/
        └── ...
```

## 🚀 测试Base模型的步骤

### 步骤1：准备训练集的清单文件（如果还没做）

```bash
cd /workspace/AniTune
source .venv/bin/activate

python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 \
  --seed 42
```

这会生成：
- `data/icartoonface/splits/train.txt`
- `data/icartoonface/splits/val.txt`
- `data/icartoonface/splits/stats.json`

### 步骤2：测试预训练base模型

**在验证集上测试：**

```bash
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val
```

**快速测试（只评估前1000个样本）：**

```bash
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --num-samples 1000
```

### 步骤3（可选）：准备并测试独立测试集

如果你下载了 `personai_icartoonface_rectest` 测试集：

**3.1 准备测试集清单：**

```bash
python scripts/prepare_test_manifest.py \
  --source data/personai_icartoonface_rectest/icartoonface_rectest \
  --output data/icartoonface
```

这会生成：
- `data/icartoonface/splits/test.txt`
- `data/icartoonface/splits/test_stats.json`

**3.2 在测试集上评估fine-tuned模型：**

```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --eval-split test \
  --test-root data/personai_icartoonface_rectest/icartoonface_rectest
```

## 📈 预期结果

### Base模型（未fine-tune）

| 方法 | 预期准确率 | 说明 |
|-----|-----------|------|
| 随机猜测 | ~0.02% (1/5013) | 随机baseline |
| 预训练ViT（随机head） | ~0.1-1% | ImageNet特征+随机分类头 |

**为什么这么低？**
- ImageNet预训练的ViT从未见过动漫风格的人脸
- 分类头是随机初始化的，完全没训练
- 这是**正常的**！这就是为什么需要fine-tuning

### Fine-tuned模型（预期提升）

| 方法 | 预期准确率 | 训练时间 |
|-----|-----------|---------|
| Head-only (冻结backbone) | 75-80% | 20-30分钟 |
| LoRA (rank=8) | 85-88% | 45-60分钟 |
| LoRA (rank=12) | 87-90% | 1小时 |
| LoRA (rank=16) | 88-92% | 1.5-2小时 |
| 全量fine-tune | 92-95% | 4-6小时 |

## 📝 输出示例

运行 `test_base_model.py` 后，你会看到类似这样的输出：

```
Number of classes: 5013

============================================================
Random Guessing Baseline:
  Top-1 Accuracy: 0.0200%
============================================================

============================================================
Pretrained ViT with Random Classification Head:
============================================================
Evaluating model predictions...
100%|███████████████████| 156/156 [00:45<00:00,  3.42it/s]
  Samples evaluated: 10000
  Top-1 Accuracy: 0.1234%
  Top-5 Accuracy: 0.5678%

Note: Low accuracy is expected since the classification head
      is randomly initialized (not trained on anime faces).

============================================================
Summary:
============================================================
Dataset: 5013 classes, 10000 images (val split)
Random baseline: 0.0200%
Pretrained ViT (untrained head): 0.1234%

This baseline will improve significantly after fine-tuning!
Expected after LoRA fine-tuning: 85-90% (10 epochs)
```

## 🎯 使用场景

### 场景1：快速验证数据加载

```bash
# 只测试100个样本，快速检查
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --num-samples 100
```

### 场景2：完整baseline评估

```bash
# 在完整验证集上评估
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val
```

### 场景3：对比训练集和验证集

```bash
# 训练集
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split train

# 验证集
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val
```

## 📊 完整实验流程

### 1. 测试base模型baseline

```bash
cd /workspace/AniTune
source .venv/bin/activate

# 准备数据
python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 --seed 42

# 测试baseline
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val
```

### 2. Fine-tune模型

```bash
# 使用A100优化配置训练
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --wandb --wandb-project AniTune
```

### 3. 评估fine-tuned模型

```bash
# 在验证集上评估
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

### 4. 对比结果

| 阶段 | Top-1准确率 | 提升 |
|-----|-----------|------|
| Base模型 | ~0.1% | - |
| Fine-tuned | ~88% | +87.9% |

**结论**：Fine-tuning将准确率从接近0提升到88%，证明了LoRA的有效性！

## 💡 注意事项

1. **Base模型准确率很低是正常的**
   - ImageNet预训练不包含动漫人脸
   - 这不是bug，是预期行为

2. **测试集和训练集的标签必须对齐**
   - 如果使用独立测试集，确保标签ID映射一致
   - 使用相同的identity顺序

3. **快速测试选项**
   - 使用 `--num-samples` 限制评估样本数
   - 适合快速验证代码是否正常工作

4. **显存使用**
   - Base模型测试只需要推理，显存占用很少（~2-3GB）
   - 可以使用较大的batch size

## 🔍 故障排除

### 问题1：找不到清单文件

**错误**：`FileNotFoundError: data/icartoonface/splits/train.txt`

**解决**：先运行数据准备脚本
```bash
python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 --seed 42
```

### 问题2：类别数量不匹配

**错误**：`RuntimeError: size mismatch`

**原因**：配置文件中的 `num_classes` 与实际数据集不匹配

**解决**：脚本会自动检测类别数，无需手动设置

### 问题3：运行太慢

**解决**：使用 `--num-samples` 限制样本数
```bash
PYTHONPATH=src python scripts/test_base_model.py \
  --config configs/lora_vitb16.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --num-samples 500  # 只测试500个样本
```

## 📚 相关文档

- 完整设置指南：`SETUP_CN.md`
- A100配置指南：`CONFIG_GUIDE_CN.md`
- 快速开始：`QUICK_START_A100.md`
- 原始README：`README.md`

---

**总结**：测试base模型是建立baseline的重要步骤。预期准确率会很低（~0.1%），这是正常的。Fine-tuning后应该能达到85-90%的准确率！

