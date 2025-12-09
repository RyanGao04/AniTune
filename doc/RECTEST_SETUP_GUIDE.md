# 📊 Recognition 测试集设置指南

## 🎯 测试集特点

iCartoonFace Recognition 测试集是**开放集**（Open-set）格式：
- ✅ 包含训练集中存在的角色（Closed-set）
- ✅ 包含训练时未见过的角色（Open-set，label_id = -1）
- ✅ 提供人脸边界框坐标

## 📥 步骤1：下载标签文件

### 方法1：爱奇艺网盘（推荐，国内可访问）

1. **访问链接**：
   ```
   https://fft.cloud.iqiyi.com/s/bUbdw5A
   ```

2. **输入密码**：
   ```
   X6fgYZ
   ```

3. **下载文件**：
   - 找到 "iCartoonFace recognition test dataset label"
   - 无需注册，直接下载（浏览器会自动下载，忽略 FFT Client 登录界面）

4. **保存位置**：
   ```bash
   # 将下载的标签文件重命名并移动到：
   mv ~/Downloads/icartoonface_rectest_label.txt data/
   ```

### 方法2：Google Drive（需要翻墙）

如果无法访问，跳过此方法。

## 📂 步骤2：验证数据

### 检查标签文件格式

```bash
# 查看标签文件前几行
head -5 data/icartoonface_rectest_label.txt
```

**应该看到类似**：
```
personai_icartoonface_rectest_0000000.jpg 101 131 258 327 42
personai_icartoonface_rectest_0000001.jpg 75  17  251 205 42
personai_icartoonface_rectest_0000002.jpg 120 80  300 280 -1
...
```

**格式说明**：
```
文件名                                    x1  y1  x2  y2  label_id
personai_icartoonface_rectest_0000000.jpg 101 131 258 327 42
                                          └─────────┘ └──┘
                                           边界框坐标   标签
```

- **label_id >= 0**: 训练集中存在的角色
- **label_id == -1**: 不属于训练集的角色（开放集）

### 检查测试集图片

```bash
# 确认测试集图片位置（根据你的截图）
ls data/personai_icartoonface_rectest/icartoonface_rectest/ | head -5
```

应该看到：
```
personai_icartoonface_rectest_0000000.jpg
personai_icartoonface_rectest_0000001.jpg
personai_icartoonface_rectest_0000002.jpg
...
```

## 🔧 步骤3：处理测试集

### 选项A：不裁剪（直接使用原图）

```bash
PYTHONPATH=src python scripts/prepare_rectest_openset.py \
  --label-file data/icartoonface_rectest_label.txt \
  --image-dir data/personai_icartoonface_rectest/icartoonface_rectest \
  --output-dir data/icartoonface_rectest_processed
```

这会生成：
```
data/icartoonface_rectest_processed/
├── splits/
│   ├── test_closedset.txt    # 仅训练集类别
│   ├── test_openset.txt       # 包含所有样本（含未知类别）
│   └── test_stats.json        # 统计信息
```

### 选项B：裁剪人脸区域（推荐）

```bash
PYTHONPATH=src python scripts/prepare_rectest_openset.py \
  --label-file data/icartoonface_rectest_label.txt \
  --image-dir data/personai_icartoonface_rectest/icartoonface_rectest \
  --output-dir data/icartoonface_rectest_processed \
  --crop \
  --margin 0.1
```

这会额外生成裁剪后的图片：
```
data/icartoonface_rectest_processed/
├── images/                     # 裁剪后的人脸图片
│   ├── personai_icartoonface_rectest_0000000.jpg
│   └── ...
└── splits/
    ├── test_closedset.txt
    ├── test_openset.txt
    └── test_stats.json
```

**参数说明**：
- `--crop`: 启用裁剪
- `--margin 0.1`: 边界框扩展10%（避免裁剪太紧）

## 📊 步骤4：评估模型

### Closed-set 评估（仅训练集类别）

这是标准的识别任务评估：

```bash
# 使用不裁剪的版本
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16.yaml \
  --checkpoint runs/lora_vitb16/best.pt \
  --data-root data/personai_icartoonface_rectest/icartoonface_rectest \
  --test-manifest data/icartoonface_rectest_processed/splits/test_closedset.txt
```

或者使用裁剪的版本：
```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16.yaml \
  --checkpoint runs/lora_vitb16/best.pt \
  --data-root data/icartoonface_rectest_processed/images \
  --test-manifest data/icartoonface_rectest_processed/splits/test_closedset.txt
```

### Open-set 评估（包含未知类别）

Open-set 需要不同的评估指标（检测未知类别的能力），当前的 `eval.py` 不支持。

如果需要进行 open-set 评估，需要修改评估脚本添加：
- Unknown class detection
- AUROC, AUPR 等指标
- 置信度阈值设置

## 🔍 查看统计信息

```bash
cat data/icartoonface_rectest_processed/splits/test_stats.json
```

会显示：
```json
{
  "total_samples": 17500,
  "closed_set_samples": 15000,
  "open_set_samples": 2500,
  "unique_classes": 5013,
  "label_distribution": {...},
  ...
}
```

## 🎯 常见问题

### Q1: 标签文件下载不了？

**A**:
- 使用浏览器直接下载，不要用 FFT Client
- 如果浏览器弹出 FFT Client 登录窗口，点击取消，然后右键链接 → "另存为"
- 或者使用命令行工具（如果有直链）

### Q2: 图片和标签对不上？

**A**: 检查图片路径：
```bash
# 标签文件中的文件名
head -1 data/icartoonface_rectest_label.txt | awk '{print $1}'

# 实际图片路径
ls data/personai_icartoonface_rectest/icartoonface_rectest/ | head -1
```

确保文件名匹配。

### Q3: 要不要裁剪人脸？

**A**:
- **裁剪**：更接近训练时的数据分布（训练集是裁剪好的）
- **不裁剪**：保留更多上下文信息，但可能与训练分布不一致

建议**先裁剪试试**，如果效果不好再用原图。

### Q4: Open-set 样本怎么评估？

**A**:
当前的 `eval.py` 只计算分类准确率，无法处理 label_id = -1 的样本。

有两种方案：
1. **仅评估 closed-set**（使用 `test_closedset.txt`）
2. **修改评估脚本**支持 open-set（需要添加未知类别检测逻辑）

### Q5: 为什么测试集格式和训练集不一样？

**A**:
- 训练集：按文件夹组织（方便训练）
- 测试集：平铺 + 标签文件（支持 open-set 评估，包含边界框信息）

这是数据集设计的选择，需要用脚本转换格式。

## 📝 完整流程示例

```bash
# 1. 下载标签文件
# 访问 https://fft.cloud.iqiyi.com/s/bUbdw5A
# 密码：X6fgYZ
# 保存为 data/icartoonface_rectest_label.txt

# 2. 验证数据
head -5 data/icartoonface_rectest_label.txt
ls data/personai_icartoonface_rectest/icartoonface_rectest/ | head -5

# 3. 处理测试集（裁剪版本）
PYTHONPATH=src python scripts/prepare_rectest_openset.py \
  --label-file data/icartoonface_rectest_label.txt \
  --image-dir data/personai_icartoonface_rectest/icartoonface_rectest \
  --output-dir data/icartoonface_rectest_processed \
  --crop \
  --margin 0.1

# 4. 查看统计
cat data/icartoonface_rectest_processed/splits/test_stats.json

# 5. 评估模型（Closed-set）
PYTHONPATH=src python scripts/eval.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/icartoonface_rectest_processed/images \
  --test-manifest data/icartoonface_rectest_processed/splits/test_closedset.txt

# 6. 错误分析（可选）
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/icartoonface_rectest_processed/images \
  --test-manifest data/icartoonface_rectest_processed/splits/test_closedset.txt \
  --output-dir error_analysis_test
```

## 🚀 下一步

测试集准备好后，你可以：
1. ✅ 在 closed-set 上评估模型性能
2. ✅ 进行错误分析
3. ✅ 比较不同配置的效果
4. 📝 （可选）修改评估脚本支持 open-set 评估

---

**提示**：如果在下载或处理过程中遇到问题，可以先用验证集继续实验，等测试集准备好后再做最终评估。
