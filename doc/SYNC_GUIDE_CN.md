# 📡 远程服务器同步指南

## 🖥️ 服务器信息

**远程服务器：**
```bash
SSH: ssh -p 9870 root@34.68.208.1 -L 8080:localhost:8080
路径: /workspace/AniTune/
```

**本地路径：**
```bash
/Users/tdu/Documents/GitHub/AniTune/
```

## 📥 下载脚本使用指南

我为你创建了3个脚本，在**本地Mac**上运行：

### 1. `download_results.sh` - 下载所有训练结果 ⭐⭐⭐⭐⭐

**用途：** 下载所有训练输出（checkpoints、日志等）

**使用方法：**
```bash
# 在本地Mac终端运行
cd /Users/tdu/Documents/GitHub/AniTune
chmod +x download_results.sh
./download_results.sh
```

**会下载：**
- ✅ `runs/` - 所有训练checkpoints（best.pt, last.pt）
- ✅ `wandb/` - Weights & Biases本地日志
- ✅ `logs/` - 训练日志文件

**特点：**
- 支持断点续传（如果中断，重新运行继续下载）
- 增量同步（只下载新的或修改过的文件）
- 显示实时进度

### 2. `backup_code.sh` - 备份代码（排除数据）⭐⭐⭐⭐⭐

**用途：** 备份整个项目代码，排除大文件

**使用方法：**
```bash
# 在本地Mac终端运行
cd /Users/tdu/Documents/GitHub/AniTune
chmod +x backup_code.sh
./backup_code.sh
```

**会备份：**
- ✅ `src/` - 源代码
- ✅ `scripts/` - 训练脚本
- ✅ `configs/` - 配置文件
- ✅ `tests/` - 测试代码
- ✅ `*.md` - 文档
- ✅ `requirements.txt`, `environment.yml` - 依赖文件

**会排除：**
- ❌ `data/` - 数据集（太大，5GB+）
- ❌ `runs/` - 训练结果（用download_results.sh单独下载）
- ❌ `wandb/` - W&B日志（用download_results.sh单独下载）
- ❌ `.venv/` - Python虚拟环境
- ❌ `__pycache__/` - Python缓存

### 3. `download_specific_run.sh` - 下载特定实验 ⭐⭐⭐⭐

**用途：** 只下载某一个特定的训练实验

**使用方法：**
```bash
# 在本地Mac终端运行
cd /Users/tdu/Documents/GitHub/AniTune
chmod +x download_specific_run.sh
./download_specific_run.sh

# 然后根据提示输入实验名称，例如:
# lora_vitb16_a100_balanced
```

## 🎯 推荐工作流程

### 场景1：训练完成，下载所有结果

```bash
# 步骤1: 下载训练结果
cd /Users/tdu/Documents/GitHub/AniTune
./download_results.sh

# 步骤2: 备份最新代码（如果远程有修改）
./backup_code.sh

# 完成！现在本地有完整的训练结果和代码
```

### 场景2：只想下载一个实验

```bash
# 下载特定实验
cd /Users/tdu/Documents/GitHub/AniTune
./download_specific_run.sh
# 输入: lora_vitb16_a100_balanced
```

### 场景3：定期同步代码

```bash
# 每天或每次修改后运行
cd /Users/tdu/Documents/GitHub/AniTune
./backup_code.sh
```

### 场景4：训练进行中，想看中间结果

```bash
# 可以多次运行，rsync会增量同步
cd /Users/tdu/Documents/GitHub/AniTune
./download_results.sh
```

## 💡 实用技巧

### 1. 添加到cron定时任务（自动备份）

```bash
# 编辑crontab
crontab -e

# 添加以下行（每天凌晨2点自动备份）
0 2 * * * cd /Users/tdu/Documents/GitHub/AniTune && ./backup_code.sh > /tmp/backup.log 2>&1
```

### 2. 创建快捷命令

在 `~/.zshrc` 或 `~/.bash_profile` 添加：

```bash
# AniTune同步快捷命令
alias anitune-download='cd /Users/tdu/Documents/GitHub/AniTune && ./download_results.sh'
alias anitune-backup='cd /Users/tdu/Documents/GitHub/AniTune && ./backup_code.sh'
alias anitune-sync='cd /Users/tdu/Documents/GitHub/AniTune && ./backup_code.sh && ./download_results.sh'
```

然后直接运行：
```bash
anitune-download  # 下载结果
anitune-backup    # 备份代码
anitune-sync      # 完整同步
```

### 3. 只下载best.pt（节省时间）

如果只想要最佳模型：

```bash
cd /Users/tdu/Documents/GitHub/AniTune
mkdir -p runs/lora_vitb16_a100_balanced

rsync -avz --progress \
  -e "ssh -p 9870" \
  root@34.68.208.1:/workspace/AniTune/runs/lora_vitb16_a100_balanced/best.pt \
  runs/lora_vitb16_a100_balanced/
```

### 4. 查看远程文件大小（下载前）

```bash
ssh -p 9870 root@34.68.208.1 "du -sh /workspace/AniTune/runs/*"
```

### 5. 后台下载（适合大文件）

```bash
# 在本地后台运行
cd /Users/tdu/Documents/GitHub/AniTune
nohup ./download_results.sh > download.log 2>&1 &

# 查看进度
tail -f download.log
```

## 🔍 下载后验证

### 检查下载的文件

```bash
cd /Users/tdu/Documents/GitHub/AniTune

# 查看runs目录
ls -lh runs/

# 查看特定实验
ls -lh runs/lora_vitb16_a100_balanced/

# 查看文件大小
du -sh runs/
```

### 加载模型测试

```python
import torch

# 在本地加载下载的模型
checkpoint = torch.load('runs/lora_vitb16_a100_balanced/best.pt')
print(f"验证准确率: {checkpoint['val_acc']:.4f}")
```

## 📊 脚本对比

| 脚本 | 用途 | 大小 | 时间 | 频率 |
|------|------|------|------|------|
| `download_results.sh` | 下载训练结果 | ~1-2GB | 5-10分钟 | 训练后 |
| `backup_code.sh` | 备份代码 | ~10-50MB | 1-2分钟 | 每天/每周 |
| `download_specific_run.sh` | 下载单个实验 | ~700MB | 3-5分钟 | 按需 |

## ⚙️ 自定义配置

### 修改远程服务器信息

编辑脚本开头的配置：

```bash
# 在脚本中修改这些行
REMOTE_HOST="root@34.68.208.1"
REMOTE_PORT="9870"
REMOTE_PATH="/workspace/AniTune"
LOCAL_PATH="/Users/tdu/Documents/GitHub/AniTune"
```

### 修改排除规则

在 `backup_code.sh` 中添加更多排除：

```bash
rsync -avz --progress \
  ...
  --exclude 'your_folder/' \
  --exclude '*.log' \
  ...
```

## 🚨 故障排除

### 问题1：Permission denied

**错误：**
```
Permission denied (publickey,password).
```

**解决：**
```bash
# 确保SSH密钥配置正确
ssh -p 9870 root@34.68.208.1

# 或者添加 -i 指定密钥
rsync -avz -e "ssh -p 9870 -i ~/.ssh/your_key" ...
```

### 问题2：连接超时

**解决：**
```bash
# 检查网络连接
ping 34.68.208.1

# 检查SSH端口
nc -zv 34.68.208.1 9870
```

### 问题3：下载速度慢

**解决：**
```bash
# 1. 压缩传输（已在脚本中使用）
# 2. 使用screen在服务器端运行
# 3. 分时段下载（晚上网络更快）
# 4. 只下载需要的文件
```

### 问题4：中断后继续

```bash
# rsync支持断点续传，直接重新运行脚本即可
./download_results.sh
```

### 问题5：磁盘空间不足

```bash
# 检查本地空间
df -h /Users/tdu/Documents/GitHub/AniTune

# 清理不需要的旧实验
rm -rf runs/old_experiment/

# 只下载best.pt，不下载last.pt
rsync ... --exclude 'last.pt' ...
```

## 📚 rsync参数说明

| 参数 | 说明 |
|------|------|
| `-a` | 归档模式（保留权限、时间戳等） |
| `-v` | 详细输出 |
| `-z` | 压缩传输 |
| `--progress` | 显示进度 |
| `-e "ssh -p PORT"` | 指定SSH端口 |
| `--exclude` | 排除文件/目录 |
| `--delete` | 删除目标中多余的文件（慎用） |

## 🎓 最佳实践

### 1. 定期备份代码
```bash
# 每次远程修改代码后
./backup_code.sh
```

### 2. 训练完成后立即下载
```bash
# 训练结束后
./download_results.sh
```

### 3. 使用Git管理代码
```bash
# 备份后提交到Git
cd /Users/tdu/Documents/GitHub/AniTune
git add .
git commit -m "Backup from server"
git push
```

### 4. 保留重要实验的备份
```bash
# 重命名保存
cp -r runs/lora_vitb16_a100_balanced \
      runs/backup_20241121_best_acc8823
```

## ✅ 总结

**三个脚本的使用场景：**

1. **`download_results.sh`** 
   - 训练完成后运行
   - 下载所有训练输出
   - 推荐每次训练后使用

2. **`backup_code.sh`**
   - 定期运行（每天/每周）
   - 同步代码和配置
   - 推荐作为备份习惯

3. **`download_specific_run.sh`**
   - 按需使用
   - 只想要某个实验时
   - 节省时间和空间

**现在就可以使用！**
```bash
cd /Users/tdu/Documents/GitHub/AniTune
chmod +x *.sh
./download_results.sh  # 开始下载！
```

---

**提示：所有脚本都支持断点续传，可以随时中断和继续！** 🎉

