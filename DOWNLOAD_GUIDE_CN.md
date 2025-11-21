# 📥 iCartoonFace数据集下载指南

## 🎯 你需要下载什么

**任务**：动漫人脸识别（Face Recognition）  
**数据集名称**：iCartoonFace Recognition Split

## 📦 下载链接

### 训练集（必需）
**personai_icartoonface_rectrain**

- **爱奇艺网盘**（国内推荐，速度快）:
  - 链接: https://fft.cloud.iqiyi.com/s/bUbdw5A
  - 密码: `5Kv2M1`
  - 文件大小: 约4-6GB

- **Google Drive**（国际）:
  - 链接: https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW
  - 找到并下载: `personai_icartoonface_rectrain.zip`

### 测试集（可选）
**personai_icartoonface_rectest**

- 从同样的链接下载
- 文件名: `personai_icartoonface_rectest.zip`
- 用于最终模型评估

## ⚠️ 重要：不要下载这些

❌ **personai_icartoonface_dettrain** - 这是Detection（检测）数据集
❌ **personai_icartoonface_dettest** - 这也是Detection数据集
❌ 任何包含"det"关键词的文件

## 📋 下载步骤

### 使用爱奇艺网盘（推荐）

1. **访问链接**
   ```
   https://fft.cloud.iqiyi.com/s/bUbdw5A
   ```

2. **输入密码**
   ```
   5Kv2M1
   ```

3. **找到正确的文件**
   - 找到 `personai_icartoonface_rectrain.zip`
   - **不是** `personai_icartoonface_dettrain.zip`

4. **下载到本地**
   - 下载到你的电脑
   - 然后上传到服务器，或者直接在服务器上用wget下载

### 使用Google Drive

1. **访问链接**
   ```
   https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW
   ```

2. **找到文件**
   - `personai_icartoonface_rectrain.zip`

3. **下载**
   - 点击下载（可能需要Google账号）

## 🚀 在服务器上直接下载（推荐）

如果你有下载链接的直接URL，可以在服务器上用wget：

```bash
cd /workspace/AniTune/data

# 方法1：wget（如果有直链）
# wget -O personai_icartoonface_rectrain.zip "直链URL"

# 方法2：使用rclone（如果配置了）
# rclone copy ...

# 方法3：使用gdown（Google Drive）
pip install gdown
# gdown --id FILE_ID --output personai_icartoonface_rectrain.zip
```

## 📂 解压和验证

### 1. 上传到服务器

如果是在本地下载的，传到服务器：

```bash
# 在本地电脑上
scp personai_icartoonface_rectrain.zip user@server:/workspace/AniTune/data/
```

### 2. 解压

```bash
cd /workspace/AniTune/data

# 解压
unzip personai_icartoonface_rectrain.zip

# 解压后应该看到
ls -la personai_icartoonface_rectrain/
```

### 3. 验证结构

运行检查脚本：

```bash
cd /workspace/AniTune
./check_data.sh
```

**应该看到：**
```
✓ 角色文件夹数量: 5013
✓ 总图片数量: 300000-400000
✓ 文件夹包含图片
✓ 数据集类型正确（Recognition）
```

## 🔍 如何确认下载正确

### 检查点1：文件名
✅ 正确：`personai_icartoonface_rectrain.zip`  
❌ 错误：`personai_icartoonface_dettrain.zip`（这是detection）

### 检查点2：文件大小
✅ 正确：4-6 GB左右  
❌ 可疑：如果只有几MB，可能不完整

### 检查点3：解压后的结构

**正确的结构：**
```
personai_icartoonface_rectrain/
└── icartoonface_rectrain/
    ├── 00001/  或  00000/
    │   ├── 00001_001.jpg  ← 有实际的图片文件
    │   ├── 00001_002.jpg
    │   └── ...
    ├── 00002/
    │   └── ...
    └── ...（5013个文件夹）
```

**错误的标志：**
- ❌ 文件夹是空的（只有目录结构）
- ❌ 有txt文件包含边界框坐标
- ❌ 图片文件名像 `..._dettest_000000.jpg`

### 检查点4：随机检查一个文件夹

```bash
# 查看第一个文件夹
ls -la data/personai_icartoonface_rectrain/icartoonface_rectrain/*/  | head -30

# 应该看到很多 .jpg 文件
```

## 🐛 常见问题

### Q1: 文件夹存在但是空的

**症状：**
```
✓ 角色文件夹数量: 5013
❌ 总图片数量: 0
```

**原因：**
- 下载不完整
- 只下载了目录结构
- 解压失败

**解决：**
1. 删除现有的空目录
2. 重新下载完整的zip文件
3. 重新解压

```bash
cd /workspace/AniTune/data
rm -rf personai_icartoonface_rectrain
# 重新下载和解压
```

### Q2: 看到边界框坐标的txt文件

**症状：**
```
personai_icartoonface_rectest_0000000.jpg  101  131  258  327
```

**原因：**
- 下载了Detection数据集而非Recognition数据集

**解决：**
- 重新下载，确保文件名包含 `rectrain` 不是 `dettrain`

### Q3: 文件夹命名格式不对

**症状：**
```
personai_icartoonface_rectrain_00000/  ← 太长
personai_icartoonface_rectrain_00001/
```

**预期：**
```
00000/  或  00001/  ← 简短的数字
00001/      00002/
```

**解决：**
如果文件夹有图片，只是命名长了，可以批量重命名：

```bash
cd data/personai_icartoonface_rectrain/icartoonface_rectrain

# 批量重命名（小心操作！先备份）
for dir in personai_icartoonface_rectrain_*; do
    new_name=$(echo "$dir" | sed 's/personai_icartoonface_rectrain_//')
    mv "$dir" "$new_name"
done
```

但如果文件夹是空的，最好重新下载完整数据。

### Q4: 下载速度太慢

**建议：**
1. 优先使用爱奇艺网盘（国内服务器速度快）
2. 使用下载工具（IDM、aria2等）
3. 在服务器上直接wget下载（如果有直链）
4. 晚上或非高峰时段下载

### Q5: Google Drive提示需要权限

**解决：**
- 确保使用Google账号登录
- 使用gdown工具绕过浏览器限制
- 或使用爱奇艺网盘替代

## ✅ 验证清单

下载完成后，确认以下所有项：

- [ ] 文件名正确：包含 `rectrain` 不是 `dettrain`
- [ ] 文件大小合理：4-6GB
- [ ] 解压成功：没有错误
- [ ] 目录结构正确：
  ```
  data/personai_icartoonface_rectrain/icartoonface_rectrain/
  ```
- [ ] 有5013个文件夹
- [ ] 每个文件夹包含.jpg图片（不是空的）
- [ ] 总图片数量：30-40万张
- [ ] 运行 `./check_data.sh` 全部通过

## 📚 完整流程示例

```bash
# 1. 进入数据目录
cd /workspace/AniTune/data

# 2. 下载（假设已有zip文件）
# ... 使用网盘或wget下载 ...

# 3. 解压
unzip personai_icartoonface_rectrain.zip

# 4. 检查结构
ls -la personai_icartoonface_rectrain/icartoonface_rectrain/ | head -20

# 5. 运行验证
cd /workspace/AniTune
./check_data.sh

# 6. 如果一切正常，生成训练清单
source .venv/bin/activate
python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 --seed 42

# 7. 开始训练！
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

## 🆘 需要帮助？

如果还有问题，检查：
- `./check_data.sh` - 数据验证脚本
- `DATA_FORMAT_CN.md` - 数据格式详细说明
- `README.md` - 原始英文文档

---

**总结：确保下载的是 personai_icartoonface_**rec**train（Recognition），不是 **det**train（Detection）！**

