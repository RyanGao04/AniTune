# 错误分析和可视化指南

# 在测试集上运行
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/personai_icartoonface_rec/personai_icartoonface_rectest/icartoonface_rectest \
  --split test \
  --output-dir error_analysis_test \
  --lora-only

## 功能概述

`error_analysis.py` 脚本提供全面的模型错误分析，包括：

1. **混淆矩阵**：Top-50 类别的归一化混淆矩阵
2. **混淆类别对**：最容易混淆的类别对（bar chart）
3. **每类准确率分析**：准确率分布和与样本数量的关系
4. **错误样本可视化**：展示高置信度预测错误的样本
5. **统计报告**：详细的错误统计JSON文件

## 快速开始

### 基本用法

```bash
# 在验证集上运行错误分析
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output-dir error_analysis_results
```

### 在测试集上运行

```bash
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split test \
  --output-dir error_analysis_test
```

## 输出文件

运行后会在输出目录生成以下文件：

```
error_analysis_results/
├── confusion_matrix_top50_val.png          # 混淆矩阵热力图
├── confused_pairs_val.png                  # 最容易混淆的类别对
├── per_class_accuracy_val.png              # 每类准确率分布
├── error_samples_visualization.png         # 错误样本可视化
└── error_statistics_val.json               # 详细统计JSON
```

## 生成的可视化说明

### 1. 混淆矩阵 (Confusion Matrix)

**文件**: `confusion_matrix_top50_val.png`

显示样本数最多的 50 个类别的混淆矩阵（按行归一化）。

- **X轴**: 预测标签
- **Y轴**: 真实标签
- **颜色**: 归一化频率（0-1）
- **对角线**: 正确预测
- **非对角线**: 混淆情况

**解读**：
- 对角线越亮 → 该类别准确率越高
- 非对角线亮点 → 表示容易混淆的类别对
- 整体越"对角线化" → 模型性能越好

### 2. 最容易混淆的类别对 (Confused Pairs)

**文件**: `confused_pairs_val.png`

横向条形图显示 Top-20 最容易混淆的类别对。

- **格式**: "True Class → Predicted Class"
- **长度**: 错误预测的数量

**解读**：
- 条形越长 → 该类别对混淆次数越多
- 可能原因：
  - 视觉相似（同一角色的不同造型）
  - 数据标注错误
  - 类别定义模糊

### 3. 每类准确率分析 (Per-Class Accuracy)

**文件**: `per_class_accuracy_val.png`

包含两个子图：

**左图：准确率直方图**
- 显示所有类别的准确率分布
- 红色虚线：平均准确率

**右图：准确率 vs 样本数量**
- 箱线图显示不同样本数量组的准确率分布
- 组别：<10, 10-30, 30-50, 50-100, ≥100

**解读**：
- 如果右图呈上升趋势 → 更多数据 = 更高准确率（数据不足问题）
- 如果右图平坦 → 数据量不是主要瓶颈（可能是模型容量或特征问题）

### 4. 错误样本可视化 (Error Samples)

**文件**: `error_samples_visualization.png`

展示 20 个高置信度预测错误的样本（4×5 网格）。

每个样本显示：
- **原始图像**
- **True**: 真实标签
- **Pred**: 预测标签
- **Conf**: 预测置信度

**解读**：
- 高置信度错误往往是：
  - 真实混淆（相似角色）
  - 数据标注错误
  - 遮挡/低质量图像

### 5. 统计报告 (Error Statistics JSON)

**文件**: `error_statistics_val.json`

包含详细的统计信息：

```json
{
  "total_errors": 1234,
  "avg_error_confidence": 0.567,
  "per_class_accuracy": {
    "mean": 0.912,
    "std": 0.089,
    "min": 0.234,
    "max": 1.000
  },
  "worst_classes": [...],
  "best_classes": [...],
  "most_confused_pairs": [...]
}
```

## 分析示例

### 发现：长尾分布问题

如果 `per_class_accuracy` 图显示：
- 样本数 <10 的类别：准确率 ~70%
- 样本数 ≥100 的类别：准确率 ~95%

**解决方案**：
1. 数据增强（针对尾部类别）
2. 类别平衡采样
3. Focal Loss / 类别权重

### 发现：特定类别混淆

如果 `confused_pairs` 显示：
- Class 42 → Class 97: 89 errors
- Class 97 → Class 42: 76 errors

**解决方案**：
1. 检查数据标注是否有误
2. 分析两个类别的视觉差异
3. 考虑合并类别或使用层次化分类

### 发现：高置信度错误

如果错误样本中多数置信度 >0.8：

**可能原因**：
1. 模型过拟合训练集的噪声标签
2. 训练-测试分布不一致
3. 特定视角/姿态的泛化能力不足

**解决方案**：
1. 标签平滑 (Label Smoothing)
2. 更强的正则化
3. 集成学习

## 高级用法

### 分析特定类别

修改脚本以只分析特定类别的错误：

```python
# 在 analyze_errors 函数中添加过滤
target_classes = [42, 97, 123]  # 你关心的类别
if labels[i].item() in target_classes and preds[i] != labels[i]:
    error_samples.append(...)
```

### 导出错误样本用于标注检查

```python
# 保存错误样本路径到文件
with open('errors_to_review.txt', 'w') as f:
    for error in error_samples:
        f.write(f"{img_path}\t{true_label}\t{pred_label}\n")
```

### 生成混淆矩阵的 LaTeX 表格

```python
import pandas as pd

# 选择 top-10 混淆对
top_pairs = confused_pairs[:10]
df = pd.DataFrame(top_pairs, columns=['True', 'Pred', 'Count'])
print(df.to_latex(index=False))
```

## 与 Progress Report 集成

生成的图表可以直接用于报告：

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{error_analysis_results/confusion_matrix_top50_val.png}
    \caption{Confusion matrix for top-50 classes on validation set.}
    \label{fig:confusion-matrix}
\end{figure}
```

## 性能提示

- **GPU 推荐**：分析需要运行整个验证集，使用 GPU 加速
- **时间估计**：~2-3 分钟 (A100, 17K 验证样本)
- **内存需求**：~8GB GPU 内存

## 常见问题

**Q: 为什么混淆矩阵只显示 50 个类别？**  
A: 5013 个类别的完整矩阵太大无法可视化。可以修改 `k=50` 参数。

**Q: 如何查看所有类别的准确率？**  
A: 查看 `error_statistics_val.json` 中的 `worst_classes` 和 `best_classes`。

**Q: 错误样本可视化为什么是空白？**  
A: 确保使用 ManifestDataset（而非 ImageFolder）且图像路径正确。

## 下一步

基于错误分析结果，可以：

1. **改进数据**：修正标注、增加困难样本
2. **改进模型**：调整架构、增加正则化
3. **改进训练**：使用类别权重、Focal Loss
4. **后处理**：针对易混淆类别设计规则

---

**创建时间**: 2025-11-22  
**适用于**: AniTune LoRA-ViT 模型  
**依赖**: PyTorch, matplotlib, seaborn, numpy

