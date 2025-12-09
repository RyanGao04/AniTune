# 📊 iCartoonFace数据集格式说明

## ⚠️ 重要：两种不同的任务和数据格式

iCartoonFace包含**两个完全不同**的任务，使用不同的数据格式：

### 1. Detection（检测）- ❌ 不是本项目需要的

**任务**：在图片中找到人脸的位置  
**数据集**：`personai_icartoonface_dettest` / `personai_icartoonface_dettrain`  
**标签格式**：边界框坐标

```
图片文件名                                x1   y1   x2   y2
personai_icartoonface_rectest_0000000.jpg  101  131  258  327
personai_icartoonface_rectest_0000001.jpg  75   17   251  205
```

**含义：**
- `x1, y1`: 人脸边界框左上角坐标
- `x2, y2`: 人脸边界框右下角坐标
- 用于训练检测模型（找到人脸在哪里）

### 2. Recognition（识别）- ✅ 本项目需要的

**任务**：识别这是哪个动漫角色  
**数据集**：`personai_icartoonface_rectrain` / `personai_icartoonface_rectest`  
**目录结构**：按角色ID组织的文件夹

```
personai_icartoonface_rectrain/icartoonface_rectrain/
├── 00001/          # 角色ID 0
│   ├── 00001_001.jpg
│   ├── 00001_002.jpg
│   └── 00001_003.jpg
├── 00002/          # 角色ID 1
│   ├── 00002_001.jpg
│   └── 00002_002.jpg
├── 00003/          # 角色ID 2
│   └── ...
└── ...（共5013个角色文件夹）
```

**标签格式**（由prepare脚本生成）：

```
相对路径                    标签ID
00001/00001_001.jpg         0
00001/00001_002.jpg         0
00002/00002_001.jpg         1
00003/00003_001.jpg         2
```

## 🎯 本项目（AniTune）使用的数据

**任务**：动漫人脸识别（Face Recognition）  
**需要下载**：Recognition数据集

### 下载链接（正确的数据集）

**训练集：personai_icartoonface_rectrain**
- 爱奇艺网盘: https://fft.cloud.iqiyi.com/s/bUbdw5A (密码: 5Kv2M1)
- Google Drive: https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW

**测试集：personai_icartoonface_rectest（可选）**
- 爱奇艺网盘: https://fft.cloud.iqiyi.com/s/bUbdw5A (密码: 5Kv2M1)
- Google Drive: https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW

### 正确的目录结构

解压后应该是这样的：

```
/workspace/AniTune/data/
├── personai_icartoonface_rectrain/
│   └── icartoonface_rectrain/      # 注意这里有两层目录
│       ├── 00001/
│       │   ├── 00001_001.jpg
│       │   ├── 00001_002.jpg
│       │   └── ...
│       ├── 00002/
│       │   └── ...
│       └── ...（共5013个文件夹）
│
└── personai_icartoonface_rectest/   # 可选的测试集
    └── icartoonface_rectest/
        ├── 00001/
        └── ...
```

## 🔍 如何检查你下载的数据是否正确

### 方法1：检查目录结构

```bash
cd /workspace/AniTune/data
ls -la personai_icartoonface_rectrain/icartoonface_rectrain/ | head -20
```

**应该看到：**
- 一堆以数字命名的文件夹（00001, 00002, ...）
- 共5013个文件夹

**如果看到：**
- `personai_icartoonface_rectest_0000000.jpg` 这样的文件
- 说明你下载的是**Detection数据集**，不对！

### 方法2：检查一个角色文件夹

```bash
ls -la data/personai_icartoonface_rectrain/icartoonface_rectrain/00001/
```

**应该看到：**
```
00001_001.jpg
00001_002.jpg
00001_003.jpg
...
```

同一个文件夹内的所有图片都是同一个角色的不同照片。

### 方法3：统计文件夹数量

```bash
ls -d data/personai_icartoonface_rectrain/icartoonface_rectrain/*/ | wc -l
```

**应该输出：** `5013`（表示5013个角色）

## 🚨 常见错误

### 错误1：下载了Detection数据集

**症状：**
- 看到带有坐标的txt文件
- 图片文件名像：`personai_icartoonface_dettest_0000000.jpg`
- 没有按角色ID分文件夹

**解决：**
- 重新下载 **Recognition** 数据集
- 文件名应该包含 `rectrain` 或 `rectest`（rec = recognition）

### 错误2：目录层级不对

**症状：**
- 找不到 `icartoonface_rectrain` 目录
- 5013个文件夹直接在根目录

**解决：**
```bash
# 检查你的目录结构
ls -la data/personai_icartoonface_rectrain/

# 应该看到一个 icartoonface_rectrain/ 子目录
# 如果没有，可能需要重新解压或移动文件
```

### 错误3：解压不完整

**症状：**
- 文件夹数量少于5013个
- 某些文件夹是空的

**解决：**
- 重新下载数据集（可能下载时损坏）
- 确保有足够的磁盘空间

## ✅ 验证数据集的完整脚本

创建一个验证脚本：

```bash
#!/bin/bash
# 保存为 check_data.sh

echo "检查iCartoonFace Recognition数据集..."
echo ""

DATA_ROOT="data/personai_icartoonface_rectrain/icartoonface_rectrain"

if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 错误：找不到目录 $DATA_ROOT"
    echo "   请确保数据集已正确解压"
    exit 1
fi

echo "✓ 找到数据根目录: $DATA_ROOT"

# 统计文件夹数量
NUM_DIRS=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✓ 角色文件夹数量: $NUM_DIRS"

if [ "$NUM_DIRS" -ne 5013 ]; then
    echo "⚠️  警告：预期5013个文件夹，实际找到 $NUM_DIRS 个"
else
    echo "✓ 文件夹数量正确！"
fi

# 统计总图片数量
NUM_IMAGES=$(find "$DATA_ROOT" -name "*.jpg" | wc -l)
echo "✓ 总图片数量: $NUM_IMAGES"

# 检查前几个文件夹
echo ""
echo "前5个角色文件夹："
ls -d "$DATA_ROOT"/*/ | head -5

# 检查第一个文件夹的内容
FIRST_DIR=$(ls -d "$DATA_ROOT"/*/ | head -1)
echo ""
echo "第一个文件夹的图片数量:"
ls "$FIRST_DIR"*.jpg 2>/dev/null | wc -l

echo ""
echo "数据集检查完成！"
```

运行检查：

```bash
chmod +x check_data.sh
./check_data.sh
```

## 📥 正确的下载和解压流程

### 步骤1：下载数据集

从爱奇艺网盘或Google Drive下载：
- **文件名应该是**：`personai_icartoonface_rectrain.zip` 或类似
- **大小**：约几GB（5013个角色的图片）

### 步骤2：解压到正确位置

```bash
cd /workspace/AniTune

# 创建data目录
mkdir -p data

# 解压（假设zip文件在当前目录）
unzip personai_icartoonface_rectrain.zip -d data/

# 检查结构
ls -la data/personai_icartoonface_rectrain/icartoonface_rectrain/
```

### 步骤3：验证数据

```bash
# 应该看到5013个文件夹
ls -d data/personai_icartoonface_rectrain/icartoonface_rectrain/*/ | wc -l

# 应该输出：5013
```

### 步骤4：生成训练清单

```bash
source .venv/bin/activate

python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 \
  --seed 42
```

**成功标志：**
```json
{
  "num_identities": 5013,
  "num_images": 389678,  // 约这个数量
  "train_images": 350710,
  "val_images": 38968,
  ...
}
```

## 🤔 如何区分两种数据集

| 特征 | Detection数据集 | Recognition数据集 |
|------|---------------|-----------------|
| 文件名关键词 | `det` | `rec` |
| 目录结构 | 所有图片在一起 | 按角色ID分文件夹 |
| 标签文件 | 边界框坐标txt | 文件夹名即标签 |
| 文件夹数量 | 1个images文件夹 | 5013个角色文件夹 |
| 图片命名 | `..._dettest_000000.jpg` | `00001_001.jpg` |
| 用途 | 人脸检测 | 人脸识别 ✓ |

## 💡 总结

### 你看到的数据（边界框坐标）：
```
personai_icartoonface_rectest_0000000.jpg  101  131  258  327
```
- 这是 **Detection任务** 的格式
- 用于训练人脸检测模型
- **不是本项目需要的**

### 本项目需要的数据（Recognition）：
```
data/personai_icartoonface_rectrain/icartoonface_rectrain/
├── 00001/  ← 角色1的所有照片
├── 00002/  ← 角色2的所有照片
└── ...
```
- 这是 **Recognition任务** 的格式
- 用于训练人脸识别模型
- **这才是正确的**

### 下一步：

1. ✅ 确认你下载了 **Recognition** 数据集（包含rectrain/rectest关键词）
2. ✅ 验证目录结构（5013个角色文件夹）
3. ✅ 运行 `prepare_icartoonface.py` 生成训练清单
4. ✅ 开始训练！

---

**还有疑问？** 运行上面的 `check_data.sh` 脚本来验证你的数据集！

