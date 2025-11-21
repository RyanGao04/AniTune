# 📥 从Google Drive下载iCartoonFace数据集

## ✅ gdown已安装

```bash
✓ gdown 5.2.0 已安装
✓ 环境: /workspace/AniTune/.venv
```

## 🎯 Google Drive链接

**官方分享文件夹：**
https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW

## 🚀 下载方法

### 方法1：一键下载文件夹（最简单）⭐⭐⭐⭐⭐

```bash
cd /workspace/AniTune/data
source ../.venv/bin/activate

# 下载整个文件夹
gdown --folder https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW \
  --remaining-ok
```

**说明：**
- `--folder`: 下载整个文件夹
- `--remaining-ok`: 忽略已存在的文件，继续下载其他文件

**预计时间：**
- 取决于网络速度
- 5GB大约需要10-30分钟（100Mbps网络）

### 方法2：下载特定文件（需要文件ID）⭐⭐⭐⭐

**步骤：**

1. **在浏览器中打开链接：**
   ```
   https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW
   ```

2. **找到文件：**
   - 找到 `personai_icartoonface_rectrain.zip` 文件
   - 或 `personai_icartoonface_rectrain` 文件夹

3. **获取文件ID：**
   - 右键点击文件 → 获取链接
   - 链接格式: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - 复制其中的 `FILE_ID` 部分

4. **使用gdown下载：**
   ```bash
   cd /workspace/AniTune/data
   source ../.venv/bin/activate
   
   # 使用文件ID下载
   gdown --id FILE_ID --output personai_icartoonface_rectrain.zip
   
   # 或者使用完整链接（推荐）
   gdown "https://drive.google.com/uc?id=FILE_ID"
   ```

**示例：**
```bash
# 假设FILE_ID是 1ABC2DEF3GHI
gdown --id 1ABC2DEF3GHI --output personai_icartoonface_rectrain.zip

# 或
gdown "https://drive.google.com/uc?id=1ABC2DEF3GHI"
```

### 方法3：使用screen后台下载⭐⭐⭐⭐

适合长时间下载，避免SSH断开导致中断：

```bash
# 1. 启动screen会话
screen -S gdrive_download

# 2. 在screen中下载
cd /workspace/AniTune/data
source ../.venv/bin/activate
gdown --folder https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW \
  --remaining-ok

# 3. 分离screen（Ctrl+A, 然后按D）
#    下载会在后台继续

# 4. 随时重新连接查看进度
screen -r gdrive_download
```

### 方法4：分批下载（如果文件夹有多个文件）

```bash
cd /workspace/AniTune/data
source ../.venv/bin/activate

# 只下载zip文件（使用正则匹配）
gdown --folder https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW \
  --remaining-ok \
  --output-document personai_icartoonface_rectrain.zip
```

## 📋 下载后的步骤

### 1. 检查下载的文件

```bash
cd /workspace/AniTune/data

# 列出下载的文件
ls -lh

# 应该看到 personai_icartoonface_rectrain.zip 或类似文件
# 大小应该是 4-6GB
```

### 2. 解压文件

```bash
cd /workspace/AniTune/data

# 解压zip文件
unzip personai_icartoonface_rectrain.zip

# 如果文件很大，显示进度
unzip -q personai_icartoonface_rectrain.zip &
# 在另一个终端查看进度
watch -n 5 'du -sh personai_icartoonface_rectrain'
```

### 3. 验证数据集

```bash
cd /workspace/AniTune
./check_data.sh
```

**应该看到：**
```
✓ 找到数据根目录
✓ 角色文件夹数量: 5013
✓ 总图片数量: 300000-400000
✓ 文件夹命名格式正确
✓ 文件夹包含图片
✓ 数据集类型正确（Recognition）
```

### 4. （可选）删除zip文件节省空间

```bash
cd /workspace/AniTune/data
rm personai_icartoonface_rectrain.zip

# 这会释放 5GB 空间
```

## 🔧 高级选项

### 显示下载进度

```bash
# gdown会自动显示进度条
# 输出示例:
# Downloading...
# From: https://drive.google.com/...
# To: /workspace/AniTune/data/personai_icartoonface_rectrain.zip
# 100%|████████████████████| 5.23G/5.23G [15:23<00:00, 5.66MB/s]
```

### 断点续传

```bash
# gdown支持断点续传
# 如果下载中断，重新运行相同的命令即可继续

cd /workspace/AniTune/data
source ../.venv/bin/activate

# 重新运行，会从断点继续
gdown --id FILE_ID --output personai_icartoonface_rectrain.zip
```

### 限速下载

```bash
# 使用trickle限速（如果安装了）
sudo apt-get install trickle
trickle -d 10240 gdown --id FILE_ID  # 限制下载速度为10MB/s
```

### 并行下载多个文件

如果文件夹中有多个文件需要下载：

```bash
cd /workspace/AniTune/data
source ../.venv/bin/activate

# 下载整个文件夹，gdown会自动处理
gdown --folder https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW \
  --remaining-ok \
  --quiet  # 减少输出
```

## ⚠️ 常见问题

### 问题1：下载速度很慢

**解决方案：**
```bash
# 1. 尝试使用代理（如果有）
export HTTP_PROXY=http://proxy-server:port
export HTTPS_PROXY=http://proxy-server:port
gdown --id FILE_ID

# 2. 换个时间段下载（晚上通常更快）

# 3. 使用screen后台下载，让它慢慢下
screen -S download
gdown --folder ...
# Ctrl+A, D 分离
```

### 问题2：提示需要权限

**错误信息：**
```
Permission denied / Access denied
```

**解决方案：**
```bash
# 1. 确保文件夹是公开分享的
# 2. 如果需要登录，使用--fuzzy选项
gdown --fuzzy "https://drive.google.com/..."

# 3. 或使用cookies认证
# 在浏览器登录Google Drive后，导出cookies
gdown --id FILE_ID --output file.zip --cookies cookies.txt
```

### 问题3：下载的是HTML而不是文件

**症状：**
```bash
file personai_icartoonface_rectrain.zip
# 输出: HTML document
```

**原因：** 文件太大，Google Drive返回了病毒扫描页面

**解决方案：**
```bash
# 使用--fuzzy选项绕过病毒扫描确认页面
gdown --fuzzy "https://drive.google.com/file/d/FILE_ID/view"

# 或使用--no-check-certificate
gdown --no-check-certificate --id FILE_ID
```

### 问题4：gdown版本太旧

**更新gdown：**
```bash
pip install --upgrade gdown

# 或安装最新开发版
pip install --upgrade git+https://github.com/wkentaro/gdown.git
```

### 问题5：下载中断了

**解决方案：**
```bash
# gdown支持断点续传，直接重新运行
cd /workspace/AniTune/data
source ../.venv/bin/activate

# 重新运行相同的命令
gdown --id FILE_ID --output personai_icartoonface_rectrain.zip
```

## 🎯 推荐流程（完整）

### 新手推荐（最简单）

```bash
# === 一键下载脚本 ===

cd /workspace/AniTune/data
source ../.venv/bin/activate

# 下载
echo "开始下载..."
gdown --folder https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW \
  --remaining-ok

# 查找下载的zip文件
ZIP_FILE=$(find . -name "*.zip" -name "*rectrain*" | head -1)

if [ -z "$ZIP_FILE" ]; then
    echo "未找到zip文件，检查下载的文件："
    ls -lh
else
    echo "找到文件: $ZIP_FILE"
    echo "开始解压..."
    unzip "$ZIP_FILE"
    
    echo "验证数据..."
    cd /workspace/AniTune
    ./check_data.sh
fi
```

### 高级用户（使用screen）

```bash
# 1. 启动screen
screen -S gdrive

# 2. 下载
cd /workspace/AniTune/data
source ../.venv/bin/activate
gdown --folder https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW \
  --remaining-ok && \
unzip personai_icartoonface_rectrain.zip && \
cd /workspace/AniTune && \
./check_data.sh

# 3. 分离screen（Ctrl+A, D）

# 4. 稍后重新连接
screen -r gdrive
```

## 📊 预期时间

| 网络速度 | 下载5GB需要 | 解压需要 | 总时间 |
|---------|------------|---------|--------|
| 1 Gbps | ~1分钟 | 2-3分钟 | ~5分钟 |
| 100 Mbps | ~7分钟 | 2-3分钟 | ~10分钟 |
| 10 Mbps | ~70分钟 | 2-3分钟 | ~75分钟 |

## ✅ 验证清单

下载完成后，确认：

- [ ] 文件大小正确（4-6GB）
- [ ] 解压成功无错误
- [ ] 目录结构正确：
  ```
  data/personai_icartoonface_rectrain/icartoonface_rectrain/
  ```
- [ ] 有5013个角色文件夹
- [ ] 文件夹不是空的（包含.jpg图片）
- [ ] 总图片数量：30-40万张
- [ ] `./check_data.sh` 全部通过 ✅

## 🔄 与其他方法对比

| 方法 | 速度 | 难度 | 推荐度 |
|------|------|------|--------|
| **Google Drive (gdown)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 爱奇艺网盘 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 本地上传 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 云存储中转 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

**Google Drive + gdown的优势：**
- ✅ 命令行操作，简单直接
- ✅ 支持断点续传
- ✅ 国际访问速度稳定
- ✅ 无需浏览器
- ✅ 适合服务器环境

## 🆘 需要帮助？

如果遇到问题：

1. 检查gdown版本：`gdown --version`（应该≥4.0）
2. 更新gdown：`pip install --upgrade gdown`
3. 查看详细错误：添加 `-v` 参数查看详细日志
4. 查看其他下载方法：`cat DOWNLOAD_GUIDE_CN.md`

---

**总结：使用gdown从Google Drive下载是最简单的方法之一！一行命令搞定！** 🚀

