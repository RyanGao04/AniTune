# 快速上传 Checkpoint 到新服务器 🚀

## 问题
- `best.pt` 文件：**345MB**，直接上传需要 **1 小时+**
- 新服务器：`216.129.245.165:40698`

## 🎯 最快解决方案（2步搞定）

### 步骤 1：安装本地环境（一次性，2分钟）

```bash
cd /Users/tdu/Documents/GitHub/AniTune
./setup_local_mac.sh
# 选择 1 (conda) 或 2 (venv)
```

### 步骤 2：提取并上传（5分钟）

```bash
# 激活环境
conda activate anitune-local
# 或 source .venv_local/bin/activate

# 提取 LoRA 权重（345MB → 25MB）
python extract_lora_weights.py

# 上传（只需 1-2 分钟！）
rsync -avz --progress \
  -e "ssh -p 40698" \
  best_lora_only.pt \
  root@216.129.245.165:/workspace/AniTune/
```

完成！

---

## 🔧 如果环境安装失败

### 方案 A：直接在服务器上提取

不在本地安装环境，直接在服务器上操作：

```bash
# 1. 先上传完整 checkpoint（开始后可以断开，用 tmux）
ssh -p 40698 root@216.129.245.165

# 在服务器上创建 tmux 会话
tmux new -s upload

# 从本地上传（在本地终端运行）
rsync -avz --progress --partial \
  -e "ssh -p 40698" \
  runs/lora_vitb16_a100_balanced/best.pt \
  root@216.129.245.165:/workspace/AniTune/runs/lora_vitb16_a100_balanced/

# 可以关闭终端，稍后重连查看进度
ssh -p 40698 root@216.129.245.165
tmux attach -s upload
```

### 方案 B：使用压缩传输

不依赖 PyTorch，只用系统工具：

```bash
cd /Users/tdu/Documents/GitHub/AniTune

# 1. 压缩（2-3分钟）
gzip -c runs/lora_vitb16_a100_balanced/best.pt > best.pt.gz

# 2. 查看压缩后大小
ls -lh best.pt.gz
# 预计：150-200MB

# 3. 上传压缩文件（20-30分钟）
rsync -avz --progress --partial \
  -e "ssh -p 40698" \
  best.pt.gz \
  root@216.129.245.165:/workspace/AniTune/

# 4. 服务器上解压
ssh -p 40698 root@216.129.245.165 \
  "cd /workspace/AniTune && gunzip best.pt.gz && mkdir -p runs/lora_vitb16_a100_balanced && mv best.pt runs/lora_vitb16_a100_balanced/"

# 5. 清理本地
rm best.pt.gz
```

---

## 📊 速度对比

| 方案 | 文件大小 | 上传时间 | 需要本地环境 |
|------|---------|---------|------------|
| 直接上传 | 345 MB | 69 分钟 | ❌ |
| 压缩上传 | 180 MB | 36 分钟 | ❌ |
| **提取 LoRA** | **25 MB** | **5 分钟** | ✅ |

---

## 🚀 立即开始（推荐）

### 如果你的 PyTorch 正常：
```bash
conda activate anitune-local
python extract_lora_weights.py
# 然后上传 best_lora_only.pt
```

### 如果 PyTorch 有问题（快速方案）：
```bash
# 使用压缩上传（不需要 Python）
gzip -c runs/lora_vitb16_a100_balanced/best.pt > best.pt.gz
rsync -avz --progress -e "ssh -p 40698" best.pt.gz root@216.129.245.165:/workspace/AniTune/
ssh -p 40698 root@216.129.245.165 "cd /workspace/AniTune && gunzip best.pt.gz && mkdir -p runs/lora_vitb16_a100_balanced && mv best.pt runs/lora_vitb16_a100_balanced/"
rm best.pt.gz
```

---

## 💡 推荐流程

```bash
# 1. 快速设置本地环境（一次性）
./setup_local_mac.sh

# 2. 激活环境
conda activate anitune-local

# 3. 提取 LoRA 权重
python extract_lora_weights.py

# 4. 上传
rsync -avz --progress -e "ssh -p 40698" best_lora_only.pt root@216.129.245.165:/workspace/AniTune/

# 5. 完成！
```

---

## ⚠️ 常见问题

**Q: conda activate 报错？**
```bash
# 初始化 conda
conda init zsh
# 重启终端或
source ~/.zshrc
```

**Q: 不想安装 conda？**
使用压缩上传方案（方案 B），不需要 Python 环境。

**Q: 上传中断了怎么办？**
使用 rsync 的 `--partial` 选项支持断点续传，重新运行命令即可从中断处继续。

---

**当前服务器配置已更新到 `remote_config.sh`：**
- Host: 216.129.245.165
- Port: 40698

