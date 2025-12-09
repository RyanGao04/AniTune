# 📤 数据上传指南

## 问题：需要把所有照片都上传吗？

**答案：是的，这是必需的！而且完全合理！** ✅

## 💾 空间需求分析

### 你的服务器状态
```
总空间: 218GB
已用:   33GB  
可用:   186GB (85%空闲) ✅✅✅
```

### 项目空间需求
```
数据集:           ~5-6GB
训练输出:         ~1-2GB
Python环境:       ~2-3GB
────────────────────────
总计:             ~8-12GB
可用空间:         186GB
使用率:           < 10% ✅ 完全够用！
```

**结论：你有足够的空间，无需担心！**

## 📊 数据集规模对比

| 数据集 | 图片数量 | 大小 | 说明 |
|--------|---------|------|------|
| **iCartoonFace** | **30-40万** | **~5GB** | **本项目** ✅ |
| MNIST | 7万 | 0.05GB | 太小 |
| CIFAR-10 | 6万 | 0.16GB | 小型 |
| ImageNet | 1400万 | 150GB | 大型 |
| COCO | 20万 | 25GB | 中型 |

**iCartoonFace相对来说是中小型数据集，非常合理！**

## 🚀 推荐上传方案

### 方案1：服务器直接下载（⭐⭐⭐⭐⭐ 强烈推荐）

**为什么最好：**
- ✅ 服务器网络通常更快
- ✅ 避免本地→服务器的传输
- ✅ 省时省力

**步骤：**

```bash
# 1. SSH连接到服务器
ssh user@your-server

# 2. 进入数据目录
cd /workspace/AniTune/data

# 3. 下载数据集

# 选项A: 从爱奇艺网盘下载（推荐，国内快）
# 访问 https://fft.cloud.iqiyi.com/s/bUbdw5A
# 密码: 5Kv2M1
# 获取下载链接，然后:
wget -O personai_icartoonface_rectrain.zip "下载链接"

# 选项B: 从Google Drive下载
pip install gdown
gdown --id "FILE_ID" --output personai_icartoonface_rectrain.zip

# 4. 解压
unzip personai_icartoonface_rectrain.zip

# 5. 验证
cd /workspace/AniTune
./check_data.sh
```

**预计时间：**
- 下载（100Mbps）: 5-10分钟
- 解压: 2-5分钟
- **总计: 10-15分钟** ⚡

### 方案2：本地上传zip文件（⭐⭐⭐⭐）

**适合场景：**
- 已经在本地下载了数据集
- 本地网络上传速度可接受

**步骤：**

```bash
# 在本地电脑上执行

# 方法1: 使用scp（简单）
scp personai_icartoonface_rectrain.zip \
  user@server:/workspace/AniTune/data/

# 方法2: 使用rsync（支持断点续传，推荐）
rsync -avz --progress \
  personai_icartoonface_rectrain.zip \
  user@server:/workspace/AniTune/data/

# 参数说明:
# -a: 归档模式（保留属性）
# -v: 详细输出
# -z: 压缩传输
# --progress: 显示进度条
```

**传输时间估算：**

| 网络速度 | 传输5GB需要 |
|---------|-----------|
| 1 Gbps | ~1分钟 ⚡⚡⚡ |
| 100 Mbps | ~7分钟 ⚡⚡ |
| 10 Mbps | ~70分钟 ⚡ |
| 1 Mbps | ~12小时 🐌 |

### 方案3：使用screen/tmux后台传输（⭐⭐⭐⭐）

**适合场景：**
- 传输时间较长
- 担心网络中断或SSH断开

**步骤：**

```bash
# 1. 在服务器上启动screen会话
ssh user@server
screen -S data_upload

# 2. 在screen中执行下载/上传
cd /workspace/AniTune/data
wget -O personai_icartoonface_rectrain.zip "下载链接"

# 3. 分离screen会话（传输继续进行）
# 按 Ctrl+A, 然后按 D

# 4. 断开SSH也没关系，传输继续

# 5. 重新连接查看进度
ssh user@server
screen -r data_upload

# 6. 完成后关闭screen
exit
```

### 方案4：云存储中转（⭐⭐⭐）

**适合场景：**
- 跨国传输，直连很慢
- 有云存储账号

**步骤：**

```bash
# 1. 从原始源下载到云存储
#    (Google Drive, 阿里云OSS, AWS S3等)

# 2. 在服务器上从云存储下载
# 阿里云OSS示例
ossutil cp oss://bucket/personai_icartoonface_rectrain.zip ./

# AWS S3示例  
aws s3 cp s3://bucket/personai_icartoonface_rectrain.zip ./

# Google Drive (gdown)
gdown --id FILE_ID --output personai_icartoonface_rectrain.zip
```

## ⚡ 优化技巧

### 1. 传输zip，不要传输解压后的文件夹

```bash
# ✅ 推荐：传输zip文件（1个文件，5GB）
scp personai_icartoonface_rectrain.zip user@server:/path/

# ❌ 不推荐：传输解压后的文件夹（40万个文件）
# scp -r icartoonface_rectrain/ user@server:/path/
# 原因：建立40万个文件连接非常慢！
```

### 2. 使用压缩传输

```bash
# rsync自动压缩
rsync -avz personai_icartoonface_rectrain.zip user@server:/path/
#      ^
#      z = 压缩传输，节省带宽
```

### 3. 限速避免占满带宽

```bash
# 限制上传速度为10MB/s
rsync -avz --bwlimit=10240 \
  personai_icartoonface_rectrain.zip \
  user@server:/path/
```

### 4. 验证文件完整性

```bash
# 上传前：计算本地文件的MD5
md5sum personai_icartoonface_rectrain.zip > checksum.txt

# 上传后：在服务器上验证
md5sum personai_icartoonface_rectrain.zip
# 对比两个MD5值应该一致
```

## 🔍 传输后验证

### 1. 检查文件大小

```bash
cd /workspace/AniTune/data
ls -lh personai_icartoonface_rectrain.zip

# 应该显示 4.0G-6.0G 左右
# 如果大小不对，可能传输不完整
```

### 2. 解压测试

```bash
# 尝试解压
unzip personai_icartoonface_rectrain.zip

# 如果报错，可能是文件损坏，需要重新传输
```

### 3. 运行验证脚本

```bash
cd /workspace/AniTune
./check_data.sh
```

**期望输出：**
```
✓ 角色文件夹数量: 5013
✓ 总图片数量: 300000-400000
✓ 文件夹包含图片
✓ 数据集类型正确（Recognition）
```

## 💡 常见问题

### Q1: 传输中断了怎么办？

**使用rsync可以断点续传：**
```bash
rsync -avz --partial --progress \
  personai_icartoonface_rectrain.zip \
  user@server:/path/

# --partial: 保留部分传输的文件
# 再次运行相同命令会从断点继续
```

### Q2: 能不能只上传一部分数据？

**不推荐，会导致：**
- ❌ 某些角色没有训练数据
- ❌ 模型性能大幅下降
- ❌ 实验结果不准确
- ❌ 浪费时间（训练不出好模型）

**建议：上传完整数据集，一次性做对！**

### Q3: 上传后数据会被复制多份吗？

**不会：**
- ✅ 数据只存储在 `data/` 目录
- ✅ 训练时直接读取，不复制
- ✅ 只会生成小的清单文件（train.txt, val.txt，几MB）
- ✅ 模型检查点另外存储在 `runs/`（1-2GB）

### Q4: 训练期间可以删除原始zip文件吗？

**可以，但建议保留：**
```bash
# 解压后，可以删除zip节省空间
cd /workspace/AniTune/data
rm personai_icartoonface_rectrain.zip

# 但建议保留，以防需要重新解压
```

### Q5: 网络太慢，有没有其他办法？

**替代方案：**
1. **在服务器上直接下载**（绕过本地）
2. **使用云存储中转**（选择就近节点）
3. **分时段传输**（晚上网络空闲时）
4. **请朋友帮忙**（在网络好的地方下载后拷贝）

## 🎯 推荐流程

### 新手推荐流程

```bash
# === 在服务器上执行 ===

# 1. 进入项目
cd /workspace/AniTune/data

# 2. 下载数据集
# 访问 https://fft.cloud.iqiyi.com/s/bUbdw5A (密码: 5Kv2M1)
# 获取下载链接后：
wget -O personai_icartoonface_rectrain.zip "下载链接"

# 3. 解压
unzip personai_icartoonface_rectrain.zip

# 4. 验证
cd /workspace/AniTune
./check_data.sh

# 5. 生成训练清单
source .venv/bin/activate
python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 --seed 42

# 6. 开始训练！
PYTHONPATH=src python scripts/train.py \
  --config configs/lora_vitb16_a100_balanced.yaml \
  --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain
```

### 高级用户流程

```bash
# === 使用screen后台传输 ===

# 1. 启动screen
screen -S download

# 2. 下载并解压
cd /workspace/AniTune/data
wget -O personai_icartoonface_rectrain.zip "下载链接" && \
unzip personai_icartoonface_rectrain.zip && \
cd /workspace/AniTune && \
./check_data.sh

# 3. 分离screen（Ctrl+A, D）

# 4. 稍后重新连接查看结果
screen -r download
```

## 📊 空间使用监控

### 实时监控

```bash
# 监控磁盘使用
watch -n 5 'df -h /workspace'

# 监控特定目录大小
watch -n 5 'du -sh /workspace/AniTune/data'
```

### 清理空间

如果空间紧张（虽然你有186GB，不用担心）：

```bash
# 删除zip（解压后）
rm /workspace/AniTune/data/*.zip

# 清理旧的训练检查点
rm -rf /workspace/AniTune/runs/old_experiment/

# 清理Python缓存
find /workspace/AniTune -type d -name "__pycache__" -exec rm -rf {} +
```

## ✅ 总结

### 你的情况

| 项目 | 值 | 评估 |
|------|-----|------|
| 可用空间 | 186GB | ✅ 非常充足 |
| 数据集大小 | 5GB | ✅ 很小 |
| 使用率 | < 10% | ✅ 完全没问题 |

### 最佳方案

1. **推荐：在服务器上直接下载**
   - 最快最简单
   - 避免本地上传
   
2. **备选：本地上传zip**
   - 使用rsync支持断点续传
   - 使用screen避免中断

3. **不推荐：只上传部分数据**
   - 会严重影响训练效果

### 下一步

✅ 上传/下载完整数据集  
✅ 运行 `./check_data.sh` 验证  
✅ 生成训练清单  
✅ 开始训练！

---

**记住：30-40万张图片、5GB大小，在深度学习中很正常！你的186GB空间绰绰有余！** 🎉

