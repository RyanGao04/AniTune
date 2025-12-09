# 使用验证集进行错误分析

## 为什么使用验证集？

当前测试集的问题：
- ❌ 所有图片在一个文件夹，没有按角色ID分类
- ❌ 只有边界框坐标文件（Detection格式）
- ❌ 文件名中的数字是序列号，不是角色ID
- ❌ 没有标签信息，无法进行错误分析

验证集的优势：
- ✅ 有正确的标签（角色ID）
- ✅ 按角色ID分文件夹组织
- ✅ 可以完整地做错误分析
- ✅ 数据质量与训练集一致

## 快速运行

```bash
cd /workspace/AniTune
./run_error_analysis_val.sh
```

## 手动运行

```bash
source .venv/bin/activate

# 使用完整 checkpoint (best.pt)
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rec/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --output-dir error_analysis_val
```

或者如果 `best.pt` 在当前目录：
```bash
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best.pt \
  --data-root data/personai_icartoonface_rec/personai_icartoonface_rectrain/icartoonface_rectrain \
  --split val \
  --output-dir error_analysis_val
```

**注意**：
- `--data-root` 应该指向包含角色文件夹的目录（如 `icartoonface_rectrain/`）
- 配置文件中的 `manifest_dir: data/icartoonface/splits` 会自动使用 `val.txt`
- 数据根目录路径：`data/personai_icartoonface_rec/personai_icartoonface_rectrain/icartoonface_rectrain`
- manifest 文件中的路径是相对于这个数据根目录的

## 输出结果

运行后会生成：

```
error_analysis_val/
├── confusion_matrix_top50_val.png      # 混淆矩阵（Top-50类别）
├── confused_pairs_val.png              # 最容易混淆的类别对
├── per_class_accuracy_val.png          # 每类准确率分析
├── error_samples_visualization.png     # 高置信度错误样本
└── error_statistics_val.json           # 详细统计（JSON）
```

## 结果说明

### 1. 混淆矩阵
显示哪些类别容易被互相混淆

### 2. 混淆类别对
Top-20 最容易混淆的类别对，帮助识别：
- 视觉相似的角色
- 可能的标注错误
- 需要改进的类别

### 3. 每类准确率
分析不同样本数量的类别的准确率：
- 是否存在长尾问题
- 哪些类别最难识别

### 4. 错误样本可视化
展示 20 个高置信度预测错误的样本

### 5. 统计报告（JSON）
详细的数值统计，可用于报告

## 与测试集的关系

验证集的分析结果可以很好地反映模型性能：
- 验证集从训练集分割而来
- 数据分布与训练集一致
- 如果验证集表现好，模型泛化能力就好

## 如果必须使用测试集

### 选项1：仅做推理（不做错误分析）

如果只需要预测结果，不需要分析错误：

```bash
PYTHONPATH=src python scripts/predict_test.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --test-dir data/personai_icartoonface_rec/personai_icartoonface_rectest/icartoonface_rectest \
  --output predictions.json \
  --lora-only
```

### 选项2：下载正确的测试集

需要按角色ID分文件夹的测试集：
- 从爱奇艺网盘或 Google Drive 下载
- 确保是 Recognition 格式（不是 Detection 格式）
- 文件结构应与训练集相同

### 选项3：使用官方标签文件

如果有官方的测试集标签文件（包含角色ID），可以创建 manifest 文件。

## 常见问题

### Q: 验证集的结果可靠吗？
A: 是的。验证集是从训练集中随机分割的（10%），能够很好地反映模型的泛化能力。

### Q: 验证集有多少样本？
A: 约 38,000-40,000 张图片（取决于分割比例）。

### Q: 为什么测试集没有标签？
A: 您下载的是 Detection 测试集（用于人脸检测任务），不是 Recognition 测试集（用于人脸识别任务）。

## 下一步

运行分析后：
1. 查看混淆矩阵，找出容易混淆的类别
2. 分析错误样本，识别问题模式
3. 检查是否有长尾问题（少样本类别准确率低）
4. 根据分析结果改进模型或数据

---

**推荐**：先在验证集上完成错误分析，确认模型性能后，再考虑测试集的使用。

