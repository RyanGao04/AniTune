# 🔄 使用 LoRA-only Checkpoint 进行错误分析

## 📋 概述

`best_lora_only.pt` 是一个轻量级的 checkpoint，只包含：
- LoRA 适配器权重
- 分类头权重
- **不包含** base 模型权重（从 timm 自动加载）

**优势**：
- 文件小（~25MB vs ~345MB）
- 上传速度快
- 适合快速测试和错误分析

## 🚀 快速开始

### 1. 确保已上传 LoRA-only checkpoint

```bash
# 检查文件是否存在
ls -lh best_lora_only.pt

# 应该看到约 25MB 的文件
```

### 2. 运行错误分析（验证集）

```bash
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --output-dir error_analysis_val \
  --lora-only
```

### 3. 运行错误分析（测试集）

首先确保已下载测试集（见 `TEST_SET_GUIDE.md`），然后：

```bash
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/personai_icartoonface_rectest/icartoonface_rectest \
  --split test \
  --output-dir error_analysis_test \
  --lora-only
```

## 🔍 工作原理

### LoRA-only 模式的工作流程

1. **构建模型**：
   - 从 timm 加载预训练的 base 模型（ViT-B/16）
   - 应用 LoRA 适配器（根据配置文件）

2. **加载权重**：
   - Base 模型权重：来自 timm（已预训练）
   - LoRA 权重：从 `best_lora_only.pt` 加载
   - 分类头权重：从 `best_lora_only.pt` 加载

3. **评估**：
   - 在指定数据集上运行推理
   - 收集错误样本和统计信息

### 关键参数

- `--lora-only`: 告诉脚本这是 LoRA-only checkpoint
- `--config`: 必须与训练时使用的配置一致（LoRA 参数）
- `--checkpoint`: LoRA-only checkpoint 文件路径

## ⚠️ 重要提示

### 1. 配置文件必须匹配

确保配置文件中的 LoRA 参数与训练时一致：

```yaml
model:
  use_lora: true
  lora_rank: 12        # 必须匹配
  lora_alpha: 24       # 必须匹配
  lora_dropout: 0.08   # 必须匹配
```

### 2. Base 模型会自动加载

- Base 模型权重从 timm 自动加载（`pretrained=True`）
- 不需要手动下载或指定 base 模型权重

### 3. 类别数量

- 确保配置文件中的 `num_classes` 与数据集匹配
- 脚本会自动从数据集获取类别数

## 📊 输出结果

运行后会生成：

```
error_analysis_test/
├── confusion_matrix_top50_test.png      # 混淆矩阵（Top-50类别）
├── confused_pairs_test.png              # 最容易混淆的类别对
├── per_class_accuracy_test.png          # 每类准确率分析
├── error_samples_visualization.png      # 高置信度错误样本
└── error_statistics_test.json           # 详细统计（JSON）
```

## 🔄 与完整 Checkpoint 对比

| 特性 | LoRA-only | 完整 Checkpoint |
|------|-----------|----------------|
| 文件大小 | ~25MB | ~345MB |
| 上传时间 | ~5秒 | ~1-2分钟 |
| Base 模型 | 从 timm 加载 | 包含在文件中 |
| LoRA 权重 | ✅ 包含 | ✅ 包含 |
| 分类头 | ✅ 包含 | ✅ 包含 |
| 使用场景 | 快速测试、错误分析 | 完整部署、继续训练 |

## 🐛 常见问题

### Q1: 加载失败，提示 missing keys

**原因**：这是正常的！LoRA-only checkpoint 不包含 base 模型权重。

**解决**：确保使用 `--lora-only` 参数，脚本会正确处理。

### Q2: 准确率与训练时不一致

**检查**：
1. 配置文件中的 LoRA 参数是否匹配
2. 数据集路径是否正确
3. 数据预处理是否一致

### Q3: 找不到 base 模型

**原因**：timm 无法下载预训练权重（网络问题）。

**解决**：
```bash
# 手动下载预训练权重
python -c "import timm; timm.create_model('vit_base_patch16_224', pretrained=True)"
```

### Q4: 测试集类别数不匹配

**解决**：
- 脚本会自动从数据集获取类别数
- 确保测试集与训练集使用相同的类别映射

## 📝 完整示例

```bash
# 1. 检查 LoRA-only checkpoint
ls -lh best_lora_only.pt

# 2. 在验证集上运行错误分析
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --output-dir error_analysis_val \
  --lora-only

# 3. 下载测试集（如果还没有）
./download_testset.sh

# 4. 在测试集上运行错误分析
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/personai_icartoonface_rectest/icartoonface_rectest \
  --split test \
  --output-dir error_analysis_test \
  --lora-only

# 5. 查看结果
ls -lh error_analysis_test/
```

## 🔗 相关文档

- `TEST_SET_GUIDE.md` - 测试集下载指南
- `ERROR_ANALYSIS.md` - 错误分析详细文档
- `ERROR_ANALYSIS_QUICKSTART.md` - 快速入门

---

**提示**：LoRA-only checkpoint 非常适合快速迭代和错误分析，但如需继续训练或完整部署，请使用完整 checkpoint。

