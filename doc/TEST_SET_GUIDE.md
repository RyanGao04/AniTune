# 📊 测试集下载和使用指南

## 🎯 下载测试集

### 方法1：使用脚本（推荐）

```bash
cd /workspace/AniTune
./download_testset.sh
```

脚本会引导您完成下载过程。

### 方法2：手动下载

#### 从Google Drive下载

1. **访问链接**：
   ```
   https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW
   ```

2. **找到文件**：
   - 文件名：`personai_icartoonface_rectest.zip`
   - ⚠️ **注意**：确保是 `rectest`（Recognition），不是 `dettest`（Detection）

3. **获取文件ID并下载**：
   ```bash
   cd /workspace/AniTune/data
   source ../.venv/bin/activate
   
   # 使用文件ID下载（替换 FILE_ID）
   gdown --id FILE_ID --output personai_icartoonface_rectest.zip
   ```

#### 从爱奇艺网盘下载（国内推荐）

1. **访问链接**：
   ```
   https://fft.cloud.iqiyi.com/s/bUbdw5A
   ```

2. **输入密码**：
   ```
   5Kv2M1
   ```

3. **找到并下载**：
   - `personai_icartoonface_rectest.zip`

4. **上传到服务器**：
   ```bash
   # 在本地电脑上
   scp personai_icartoonface_rectest.zip user@server:/workspace/AniTune/data/
   ```

## 📂 解压和验证

### 解压

```bash
cd /workspace/AniTune/data
unzip personai_icartoonface_rectest.zip
```

### 验证目录结构

解压后应该看到：

```
data/
└── personai_icartoonface_rectest/
    └── icartoonface_rectest/
        ├── 00001/
        │   ├── 00001_001.jpg
        │   └── ...
        ├── 00002/
        └── ...（多个角色文件夹）
```

### 检查数据

```bash
# 统计文件夹数量
ls -d data/personai_icartoonface_rectest/icartoonface_rectest/*/ | wc -l

# 统计图片数量
find data/personai_icartoonface_rectest/icartoonface_rectest -name "*.jpg" | wc -l

# 查看第一个文件夹
ls data/personai_icartoonface_rectest/icartoonface_rectest/00001/ | head -5
```

## 🔍 使用测试集进行错误分析

### 使用完整checkpoint

```bash
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint runs/lora_vitb16_a100_balanced/best.pt \
  --data-root data/personai_icartoonface_rectest/icartoonface_rectest \
  --split test \
  --output-dir error_analysis_test
```

### 使用LoRA-only checkpoint（轻量级）

如果您上传了 `best_lora_only.pt`：

```bash
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/personai_icartoonface_rectest/icartoonface_rectest \
  --split test \
  --output-dir error_analysis_test \
  --lora-only
```

**说明**：
- `--lora-only`: 告诉脚本这是LoRA-only checkpoint，base模型权重会从timm加载
- `--split test`: 使用test数据集
- `--data-root`: 指向测试集的根目录

### 使用快速脚本

更新 `run_error_analysis.sh` 以支持测试集：

```bash
# 编辑脚本，修改以下变量：
SPLIT="test"  # 改为 test
DATA_ROOT="data/personai_icartoonface_rectest/icartoonface_rectest"
CHECKPOINT="best_lora_only.pt"  # 如果使用LoRA-only

# 然后运行
./run_error_analysis.sh
```

## 📊 输出结果

运行后会生成以下文件：

```
error_analysis_test/
├── confusion_matrix_top50_test.png      # 混淆矩阵
├── confused_pairs_test.png              # 混淆类别对
├── per_class_accuracy_test.png          # 每类准确率
├── error_samples_visualization.png      # 错误样本可视化
└── error_statistics_test.json           # 详细统计
```

## ⚠️ 重要提示

### 1. 确保下载的是Recognition数据集

✅ **正确**：`personai_icartoonface_rectest.zip`  
❌ **错误**：`personai_icartoonface_dettest.zip`（这是Detection数据集）

### 2. 目录结构

测试集的目录结构应该和训练集类似：
- 按角色ID组织文件夹
- 每个文件夹包含该角色的多张图片
- 不是单个images文件夹

### 3. 类别数量

测试集可能包含的类别数量与训练集不同。确保：
- 模型能处理测试集中的所有类别
- 或者只分析测试集中与训练集重叠的类别

## 🐛 常见问题

### Q1: 找不到测试集文件

**解决**：
```bash
# 检查是否下载
ls -lh data/personai_icartoonface_rectest.zip

# 如果不存在，重新下载
./download_testset.sh
```

### Q2: 解压后目录结构不对

**检查**：
```bash
ls -la data/personai_icartoonface_rectest/
```

**应该看到**：
```
icartoonface_rectest/
```

**如果看到**：
- 直接是数字文件夹（00001, 00002...）→ 需要创建 `icartoonface_rectest` 目录并移动
- 或者重新解压到正确位置

### Q3: 运行错误分析时找不到数据

**检查路径**：
```bash
# 确保路径正确
ls data/personai_icartoonface_rectest/icartoonface_rectest/ | head -5

# 如果路径不对，调整 --data-root 参数
```

### Q4: 模型加载失败（使用LoRA-only时）

**确保**：
1. 配置文件中的模型设置与训练时一致
2. 使用 `--lora-only` 标志
3. base模型会从timm自动加载

## 📝 完整示例

```bash
# 1. 下载测试集
cd /workspace/AniTune
./download_testset.sh

# 2. 验证数据
ls -d data/personai_icartoonface_rectest/icartoonface_rectest/*/ | wc -l

# 3. 运行错误分析（使用LoRA-only checkpoint）
PYTHONPATH=src python scripts/error_analysis.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --checkpoint best_lora_only.pt \
  --data-root data/personai_icartoonface_rectest/icartoonface_rectest \
  --split test \
  --output-dir error_analysis_test \
  --lora-only

# 4. 查看结果
ls -lh error_analysis_test/
```

## 🔗 相关文档

- `DOWNLOAD_GUIDE_CN.md` - 完整下载指南
- `DATA_FORMAT_CN.md` - 数据格式说明
- `ERROR_ANALYSIS.md` - 错误分析详细文档
- `ERROR_ANALYSIS_QUICKSTART.md` - 快速入门

---

**提示**：测试集用于最终模型评估，建议在验证集上先完成调试和优化。

