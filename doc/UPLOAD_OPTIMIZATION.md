# Checkpoint 上传优化指南

## 问题
`best.pt` 文件大小：**345MB**，直接上传很慢。

## 🚀 优化方案（从快到慢）

### 方案 1：rsync 压缩传输（推荐）⭐

**优点**：
- 支持断点续传（网络中断可恢复）
- 自动压缩（节省 30-50% 带宽）
- 显示实时进度
- 增量传输（如果文件已存在，只传输差异部分）

**使用**：
```bash
./upload_checkpoint.sh
# 选择 1
```

**手动命令**：
```bash
rsync -avz --progress --partial \
  -e "ssh -p 40698" \
  runs/lora_vitb16_a100_balanced/best.pt \
  root@216.129.245.165:/workspace/AniTune/runs/lora_vitb16_a100_balanced/best.pt
```

**速度提升**：比 scp 快 30-50%

---

### 方案 2：先压缩再传输（最快）⭐⭐⭐

PyTorch checkpoint 可以进一步压缩！

**步骤**：

```bash
# 1. 压缩文件（约 1-2 分钟）
cd /Users/tdu/Documents/GitHub/AniTune
gzip -c runs/lora_vitb16_a100_balanced/best.pt > best.pt.gz

# 查看压缩后大小
ls -lh best.pt.gz
# 预计：150-200MB（节省 40-50%）

# 2. 上传压缩文件
rsync -avz --progress --partial \
  -e "ssh -p 40698" \
  best.pt.gz \
  root@216.129.245.165:/workspace/AniTune/

# 3. 在服务器上解压
ssh -p 40698 root@216.129.245.165 \
  "cd /workspace/AniTune && gunzip best.pt.gz && mkdir -p runs/lora_vitb16_a100_balanced && mv best.pt runs/lora_vitb16_a100_balanced/"

# 4. 删除本地压缩文件
rm best.pt.gz
```

**速度提升**：比直接传输快 50-60%

---

### 方案 3：分片上传（网络不稳定时）

如果网络经常中断，分片上传更稳定：

```bash
# 1. 分片（每片 50MB）
cd /Users/tdu/Documents/GitHub/AniTune
split -b 50M runs/lora_vitb16_a100_balanced/best.pt best.pt.part_

# 2. 上传所有分片（支持并行）
for part in best.pt.part_*; do
  rsync -avz --progress \
    -e "ssh -p 40698" \
    "$part" \
    root@216.129.245.165:/workspace/AniTune/
done

# 3. 在服务器上合并
ssh -p 40698 root@216.129.245.165 \
  "cd /workspace/AniTune && cat best.pt.part_* > runs/lora_vitb16_a100_balanced/best.pt && rm best.pt.part_*"

# 4. 清理本地分片
rm best.pt.part_*
```

---

### 方案 4：使用 screen/tmux（长时间传输）

避免网络中断导致传输失败：

```bash
# 1. 启动 tmux/screen 会话
tmux new -s upload

# 2. 在会话中运行上传
rsync -avz --progress --partial \
  -e "ssh -p 40698" \
  runs/lora_vitb16_a100_balanced/best.pt \
  root@216.129.245.165:/workspace/AniTune/runs/lora_vitb16_a100_balanced/best.pt

# 3. 分离会话（Ctrl+B, D）
# 即使关闭终端，上传继续进行

# 4. 重新连接查看进度
tmux attach -t upload
```

---

### 方案 5：只上传 LoRA 权重（终极优化）⭐⭐⭐⭐

**原理**：best.pt 包含完整模型（86M base + 4M LoRA + head），但 base 权重可以从 timm 重新加载！

**只保存 LoRA 权重**：

```python
# 创建脚本：extract_lora_only.py
import torch

checkpoint = torch.load('runs/lora_vitb16_a100_balanced/best.pt', map_location='cpu')
state_dict = checkpoint.get('model', checkpoint)

# 只保留 LoRA 和 head 权重
lora_state_dict = {}
for k, v in state_dict.items():
    if 'lora' in k or 'head' in k:
        lora_state_dict[k] = v

# 保存
torch.save({
    'model': lora_state_dict,
    'val_acc': checkpoint.get('val_acc', None)
}, 'best_lora_only.pt')

# 检查大小
import os
original_size = os.path.getsize('runs/lora_vitb16_a100_balanced/best.pt')
lora_size = os.path.getsize('best_lora_only.pt')
print(f"Original: {original_size / 1024**2:.1f} MB")
print(f"LoRA only: {lora_size / 1024**2:.1f} MB")
print(f"Reduction: {(1 - lora_size/original_size) * 100:.1f}%")
```

**运行**：
```bash
cd /Users/tdu/Documents/GitHub/AniTune
PYTHONPATH=src python extract_lora_only.py

# 上传（只有 ~20-30MB！）
rsync -avz --progress \
  -e "ssh -p 40698" \
  best_lora_only.pt \
  root@216.129.245.165:/workspace/AniTune/
```

**在服务器上加载**：
```python
# 先加载预训练模型
model = build_model(cfg)  # 会自动下载 base 权重

# 再加载 LoRA 权重
checkpoint = torch.load('best_lora_only.pt')
model.load_state_dict(checkpoint['model'], strict=False)
```

**速度提升**：90% 减小，快 10 倍！

---

## 📊 性能对比

假设网络速度：5 MB/s

| 方案 | 文件大小 | 传输时间 | 额外时间 | 总时间 |
|------|---------|---------|---------|--------|
| 标准 scp | 345 MB | 69 分钟 | 0 | **69 分钟** |
| rsync 压缩 | 345 MB | 45 分钟 | 0 | **45 分钟** |
| 先压缩再传 | 180 MB | 36 分钟 | 2 分钟 | **38 分钟** |
| 只传 LoRA | 25 MB | 5 分钟 | 1 分钟 | **6 分钟** ⭐ |

---

## 🎯 推荐方案

### 如果你只需要推理/评估：
→ **方案 5（只传 LoRA）** - 最快！

### 如果你需要继续训练：
→ **方案 2（压缩传输）** - 平衡速度和完整性

### 如果网络不稳定：
→ **方案 4（tmux/screen + rsync）** - 支持断点续传

---

## 🛠️ 快速开始

### 推荐做法（只传 LoRA）：

```bash
cd /Users/tdu/Documents/GitHub/AniTune

# 1. 提取 LoRA 权重
cat > extract_lora.py << 'EOF'
import torch
checkpoint = torch.load('runs/lora_vitb16_a100_balanced/best.pt', map_location='cpu')
state_dict = checkpoint.get('model', checkpoint)
lora_state_dict = {k: v for k, v in state_dict.items() if 'lora' in k or 'head' in k}
torch.save({'model': lora_state_dict, 'val_acc': checkpoint.get('val_acc')}, 'best_lora_only.pt')
print(f"Saved LoRA-only checkpoint: best_lora_only.pt")
EOF

python extract_lora.py

# 2. 上传（只需 1-2 分钟）
rsync -avz --progress \
  -e "ssh -p 40698" \
  best_lora_only.pt \
  root@216.129.245.165:/workspace/AniTune/

# 3. 在服务器上解压并放到正确位置
ssh -p 40698 root@216.129.245.165 << 'EOF'
cd /workspace/AniTune
mkdir -p runs/lora_vitb16_a100_balanced
# 加载 base 模型后再加载 LoRA 即可
EOF

# 4. 清理
rm extract_lora.py best_lora_only.pt
```

---

## ⚡ 网络优化技巧

### 1. 检查网络速度
```bash
# 测试上传速度
dd if=/dev/zero bs=1M count=100 | ssh -p 40698 root@216.129.245.165 "cat > /dev/null"
```

### 2. 优化 SSH 配置
在 `~/.ssh/config` 添加：
```
Host vast-server
    HostName 216.129.245.165
    Port 40698
    User root
    Compression yes
    CompressionLevel 9
    TCPKeepAlive yes
    ServerAliveInterval 60
```

然后使用：
```bash
rsync -avz --progress vast-server:/path/to/file ./
```

### 3. 使用更快的压缩算法
```bash
# 使用 pigz（并行 gzip）
pigz -c runs/lora_vitb16_a100_balanced/best.pt > best.pt.gz
```

---

## 🔍 验证上传

上传后验证文件完整性：

```bash
# 本地 MD5
md5 runs/lora_vitb16_a100_balanced/best.pt

# 远程 MD5
ssh -p 40698 root@216.129.245.165 \
  "md5sum /workspace/AniTune/runs/lora_vitb16_a100_balanced/best.pt"

# 应该一致
```

---

**立即开始**：`./upload_checkpoint.sh` 或使用上面的推荐做法！

