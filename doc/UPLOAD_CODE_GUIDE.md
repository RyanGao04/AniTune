# 代码上传指南 📤

## 🚀 快速使用

### 一键上传代码

```bash
./upload_code.sh
```

就这么简单！脚本会自动：
- ✅ 使用最新服务器配置（`remote_config.sh`）
- ✅ 排除所有大文件和不需要的内容
- ✅ 显示上传进度
- ✅ 只上传代码、配置、文档等小文件

---

## 📦 会上传什么？

### ✅ 包含的内容
- `src/` - 源代码
- `scripts/` - 所有脚本
- `configs/` - 配置文件
- `tests/` - 测试代码
- `*.md`, `*.tex` - 文档
- `requirements.txt`, `environment.yml` - 依赖
- `*.sh`, `*.py` - 工具脚本（除模型相关）

### ❌ 排除的内容
- `data/` - 数据集（太大，单独上传）
- `runs/` - 模型checkpoints（使用 `upload_checkpoint.sh`）
- `wandb/` - 训练日志（从服务器下载）
- `*.pt`, `*.pth`, `*.ckpt` - 所有模型文件
- `.venv/` - Python虚拟环境
- `__pycache__/`, `*.pyc` - Python缓存
- `.git/` - Git仓库

---

## 📊 上传大小估计

| 内容 | 大小 | 上传时间 |
|------|------|---------|
| 源代码 | ~2-5 MB | <1 分钟 |
| 配置文件 | ~50 KB | 几秒 |
| 文档 | ~5 MB | <1 分钟 |
| **总计** | **~10 MB** | **<2 分钟** |

vs 完整上传：~350 MB+，需要 1 小时+

---

## 🎯 使用场景

### 场景 1：首次部署到新服务器

```bash
# 1. 上传代码（2分钟）
./upload_code.sh

# 2. 在服务器上设置环境
ssh -p 40698 root@216.129.245.165
cd /workspace/AniTune
conda env create -f environment.yml
# 或 pip install -r requirements.txt

# 3. 单独上传数据（如果需要）
# 或者从 GDrive 下载数据
```

### 场景 2：更新代码

```bash
# 修改代码后，一键上传
./upload_code.sh

# 在服务器上立即使用最新代码
```

### 场景 3：只上传特定修改

如果只修改了部分文件，rsync 会自动增量更新：

```bash
./upload_code.sh
# rsync 只会传输修改过的文件，非常快！
```

---

## 🔧 高级用法

### 只上传特定目录

```bash
# 只上传源代码
rsync -avz --progress \
  -e "ssh -p 40698" \
  src/ \
  root@216.129.245.165:/workspace/AniTune/src/

# 只上传配置
rsync -avz --progress \
  -e "ssh -p 40698" \
  configs/ \
  root@216.129.245.165:/workspace/AniTune/configs/
```

### 查看会传输什么（不实际传输）

```bash
rsync -avz --dry-run \
  -e "ssh -p 40698" \
  --exclude 'data/' \
  --exclude 'runs/' \
  --exclude '*.pt' \
  /Users/tdu/Documents/GitHub/AniTune/ \
  root@216.129.245.165:/workspace/AniTune/
```

### 使用 tmux 防止中断

```bash
# 启动 tmux
tmux new -s upload-code

# 在 tmux 中运行
./upload_code.sh

# 分离 (Ctrl+B, D)
# 即使关闭终端，上传继续进行

# 重新连接查看
tmux attach -t upload-code
```

---

## ⚙️ 服务器配置

脚本使用 `remote_config.sh` 中的配置：

- **Host**: `216.129.245.165`
- **Port**: `40698`
- **路径**: `/workspace/AniTune`

如需修改，编辑 `remote_config.sh` 即可。

---

## 🚨 常见问题

### Q: 上传很慢？

**检查网络连接**：
```bash
ping 216.129.245.165
```

**使用压缩传输**（已默认启用 `-z`）：
rsync 会自动压缩传输。

### Q: 上传中断了？

**rsync 支持断点续传**：
重新运行 `./upload_code.sh`，会自动从中断处继续。

**使用 `--partial` 选项**（已包含）：
保留部分传输的文件。

### Q: 想排除更多文件？

编辑 `upload_code.sh`，添加更多 `--exclude` 选项：
```bash
--exclude 'your_pattern/' \
```

### Q: 想包含特定大文件？

临时注释掉相应的 `--exclude` 行，或使用：
```bash
rsync -avz --progress \
  --include 'specific_file.pt' \
  --exclude '*.pt' \
  ...
```

---

## 📋 完整工作流示例

```bash
# 1. 本地开发完成
git add .
git commit -m "Update code"

# 2. 上传代码到服务器（2分钟）
./upload_code.sh

# 3. 如果需要，上传轻量级 checkpoint（5分钟）
python extract_lora_weights.py
rsync -avz --progress -e "ssh -p 40698" \
  best_lora_only.pt \
  root@216.129.245.165:/workspace/AniTune/

# 4. SSH 到服务器
ssh -p 40698 root@216.129.245.165

# 5. 在服务器上运行
cd /workspace/AniTune
PYTHONPATH=src python scripts/train.py --config configs/lora_vitb16_a100_balanced.yaml
```

---

## 🎯 推荐流程

1. **首次设置**：
   - 上传代码：`./upload_code.sh`
   - 在服务器上安装依赖：`conda env create -f environment.yml`
   - 从 GDrive 下载数据：`./download_from_gdrive.sh`

2. **日常更新**：
   - 修改代码
   - 上传：`./upload_code.sh`
   - 立即在服务器上使用

3. **训练模型**：
   - 上传代码
   - 运行训练
   - 下载结果：`./download_results.sh`

---

**脚本位置**：`./upload_code.sh`  
**配置文件**：`remote_config.sh`  
**估计时间**：1-2 分钟

