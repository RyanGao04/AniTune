# 扩展错误分析功能说明

## 新增可视化内容

运行错误分析后，会生成以下文件：

### 1. 原有可视化（3个）

- `confusion_matrix_top50_val.png` - 混淆矩阵（Top-50类别）
- `confused_pairs_val.png` - 最容易混淆的类别对
- `per_class_accuracy_val.png` - 每类准确率分析

### 2. 新增样本可视化（4个）

#### 错误样本对比
- `high_confidence_errors.png` - **高置信度错误样本**（模型很自信但预测错误）
  - 这些可能是：标注错误、极其相似的角色、或模型的系统性偏差
  
- `low_confidence_errors.png` - **低置信度错误样本**（模型不确定且预测错误）
  - 这些可能是：图像质量差、遮挡严重、角色不清晰

#### 正确预测样本
- `high_confidence_correct.png` - **高置信度正确预测**（模型很自信且正确）
  - 这些是模型最擅长识别的样本，特征明显
  
- `low_confidence_correct.png` - **低置信度正确预测**（模型不确定但仍正确）
  - 这些样本虽然困难，但模型勉强答对了

### 3. 类别级别可视化（最多6个）

#### 最差类别（准确率最低的3个类别）
- `worst_class_{ID}_errors.png` - 显示该类别的错误样本
  - 帮助理解为什么这个类别难以识别
  - 可能发现：样本质量问题、标注错误、类内变化大

#### 最好类别（准确率最高的3个类别）
- `best_class_{ID}_correct.png` - 显示该类别的正确预测样本
  - 帮助理解模型为什么能识别好这个类别
  - 可能发现：特征明显、样本质量高、类间区分度大

## 新增统计信息

### 错误样本统计
```
Error statistics:
  Average error confidence: 0.X
  Median error confidence: 0.X
  Min error confidence: 0.X
  Max error confidence: 0.X
```

### 正确预测统计
```
Correct prediction statistics:
  Average confidence: 0.X
  Median confidence: 0.X
  Min confidence: 0.X
  Max confidence: 0.X
```

## 分析价值

### 1. 高置信度错误 vs 低置信度错误
- **高置信度错误**更严重 → 模型有系统性偏差
- **低置信度错误**相对正常 → 模型知道自己不确定

### 2. 正确预测的置信度分布
- **平均置信度高** → 模型有信心，泛化能力好
- **平均置信度低** → 模型犹豫不决，可能需要更多训练

### 3. 最差类别分析
- 识别需要改进的类别
- 可能需要：
  - 增加训练数据
  - 数据增强
  - 检查标注质量
  - 考虑合并相似类别

### 4. 最好类别分析
- 了解模型的优势
- 作为基准参考其他类别

## 使用建议

### Step 1: 查看整体统计
查看 `error_statistics_val.json` 了解：
- 总体准确率
- 错误数量和分布
- 类别准确率统计

### Step 2: 分析混淆矩阵
查看 `confusion_matrix_top50_val.png` 和 `confused_pairs_val.png`：
- 哪些类别容易混淆？
- 是否有系统性混淆模式？

### Step 3: 检查高置信度错误
查看 `high_confidence_errors.png`：
- 这些是最值得关注的错误
- 可能是标注问题或模型偏差
- 需要仔细分析原因

### Step 4: 对比正确和错误样本
对比 `high_confidence_correct.png` 和 `high_confidence_errors.png`：
- 正确的样本有什么特点？
- 错误的样本缺少什么？
- 两者有什么明显差异？

### Step 5: 分析特定类别
查看最差类别的可视化：
- 为什么这个类别难以识别？
- 样本质量如何？
- 是否需要特殊处理？

## 实际应用示例

### 场景1：发现标注错误
如果 `high_confidence_errors.png` 中的很多样本看起来预测是对的，但标签显示错误 → 可能是标注问题

### 场景2：识别困难类别
如果某些类别在 `worst_class_X_errors.png` 中的样本质量都很差 → 可能需要更多高质量数据

### 场景3：评估模型信心
对比错误和正确样本的置信度：
- 如果错误样本平均置信度接近正确样本 → 模型过度自信
- 如果差距很大 → 模型有良好的不确定性估计

### 场景4：数据增强策略
查看 `low_confidence_correct.png`：
- 这些勉强正确的样本有什么特点？
- 可以针对性地进行数据增强

## 生成的文件总览

运行后会生成 **最多13个文件**：

```
error_analysis_val/
├── confusion_matrix_top50_val.png          (1) 混淆矩阵
├── confused_pairs_val.png                  (2) 混淆类别对
├── per_class_accuracy_val.png              (3) 每类准确率
├── high_confidence_errors.png              (4) 高置信度错误
├── low_confidence_errors.png               (5) 低置信度错误
├── high_confidence_correct.png             (6) 高置信度正确
├── low_confidence_correct.png              (7) 低置信度正确
├── worst_class_X_errors.png                (8-10) 最差3个类别
├── best_class_X_correct.png                (11-13) 最好3个类别
└── error_statistics_val.json               详细统计
```

## 运行命令

```bash
./run_error_analysis_val.sh
```

或手动运行：
```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rec/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --output-dir error_analysis_val
```

## 预计运行时间

- 分析阶段：~1-2分钟（取决于GPU和数据量）
- 可视化生成：~30秒
- 总时间：~2-3分钟

---

**提示**：生成的可视化图片可以直接用于论文、报告或演示！

