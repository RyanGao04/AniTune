# 🚀 A100 虚拟机设置指南

完整的 A100 环境配置和实验运行指南。

## 📋 前置条件

- ✅ Ubuntu/Linux 系统
- ✅ Python 3.8+
- ✅ CUDA 11.8 或 12.x
- ✅ A100 GPU（40GB 或 80GB）
- ✅ 网络连接（下载模型权重和依赖）

## 🎯 快速开始（3 步）

### 步骤 1：克隆代码

```bash
# 如果代码已在本地
cd /path/to/AniTune

# 或者从 Git 克隆
git clone <your-repo-url> AniTune
cd AniTune
```

### 步骤 2：运行设置脚本

```bash
bash setup_a100.sh
```

脚本会自动：
1. 检查 Python 和 CUDA
2. 升级 pip
3. 安装 PyTorch（自动检测 CUDA 版本）
4. 安装所有依赖
5. 验证环境
6. 测试模型构建

**预计时间**: 5-10 分钟

### 步骤 3：准备数据

```bash
# 检查数据集（假设已上传）
bash check_data.sh

# 生成训练清单
PYTHONPATH=src python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 \
  --seed 42
```

## 📊 运行实验

### 选项 A：单个实验（推荐先测试）

```bash
# LoRA-only (r=8) - 推荐
./experiments/run_single_experiment.sh lora_only 8
```

**预计时间**: 2-3 小时 (10 epochs)

### 选项 B：完整对比实验

```bash
# 运行所有实验（包括 rank ablation）
./experiments/run_all_experiments.sh
```

**预计时间**: 8-10 小时

### 选项 C：自定义实验

```bash
PYTHONPATH=src python experiments/train_experiments.py \
  --config experiments/configs/base_experiment.yaml \
  --mode lora_only \
  --lora-rank 16 \
  --device cuda \
  --wandb \
  --wandb-project AniTune-MyExperiments
```

## 🔧 环境详细说明

### 依赖版本

脚本会安装以下主要依赖：

| 包 | 版本 | 用途 |
|---|------|------|
| PyTorch | 2.x (CUDA 版本) | 深度学习框架 |
| timm | >=0.9.12 | 预训练模型 |
| wandb | >=0.16.0 | 实验追踪 |
| pandas | >=2.1.0 | 数据分析 |
| matplotlib | >=3.8.0 | 可视化 |

### CUDA 版本对应

脚本自动检测 CUDA 版本并安装匹配的 PyTorch：

| CUDA 版本 | PyTorch 安装命令 |
|----------|----------------|
| 12.x | `torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.x | `torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| 其他 | 默认 PyTorch |

### 手动安装（如果脚本失败）

```bash
# 升级 pip
python3 -m pip install --upgrade pip

# 安装 PyTorch (CUDA 12.1)
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
python3 -m pip install -r requirements.txt

# 安装项目
python3 -m pip install -e .
```

## 📈 监控和调试

### 查看 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
gpustat -i 1
```

### 查看训练进度

```bash
# 查看最新日志
tail -f experiments/runs/*/logs/train.log

# 或使用 Wandb（推荐）
# 访问: https://wandb.ai
```

### 性能优化

#### 如果显存不足 (OOM)

编辑配置文件减小 batch size：

```yaml
# experiments/configs/base_experiment.yaml
data:
  batch_size: 64  # 减小到 32 或更小
```

#### 如果训练太慢

检查：
1. GPU 是否被正确使用
   ```bash
   nvidia-smi  # 应该看到 Python 进程
   ```

2. DataLoader workers
   ```yaml
   # 增加 num_workers
   data:
     num_workers: 12  # 根据 CPU 核心数调整
   ```

3. 使用混合精度训练（已默认启用）
   ```yaml
   optim:
     amp: true  # 确保启用
   ```

## 🎓 实验配置建议

### A100 40GB

```yaml
# experiments/configs/base_experiment.yaml
data:
  batch_size: 128   # 或 144
  num_workers: 12

optim:
  lr: 2.0e-4
  epochs: 10
  amp: true
```

### A100 80GB

```yaml
data:
  batch_size: 256   # 可以更大
  num_workers: 16

optim:
  lr: 2.5e-4  # 稍微提高学习率
  epochs: 10
  amp: true
```

## 📊 预期结果

### 性能基准（A100 40GB）

| 模式 | Batch Size | 训练时间/epoch | 总时间 (10 epochs) | 显存占用 |
|------|-----------|---------------|-------------------|---------|
| head_only | 128 | ~7 min | ~1.2h | ~10GB |
| full_ft | 128 | ~20 min | ~3.5h | ~24GB |
| lora_only (r=8) | 128 | ~12 min | ~2.0h | ~12GB |
| lora_only (r=16) | 128 | ~13 min | ~2.2h | ~14GB |

### 准确率基准

| 模式 | 验证准确率 | 可训练参数 |
|------|----------|-----------|
| head_only | 75-80% | 0.01% |
| full_ft | 92-95% | 100% |
| lora_only (r=8) | 91-94% | 0.35% |
| lora_only (r=16) | 92-95% | 0.70% |

## 🔍 常见问题

### Q: CUDA out of memory

**A**: 减小 batch_size 或使用 gradient checkpointing

```yaml
# 减小 batch size
data:
  batch_size: 32  # 从 128 减到 32
```

### Q: nvidia-smi 显示 0% GPU 使用

**A**: 检查是否在使用 CPU

```python
import torch
print(torch.cuda.is_available())  # 应该是 True
print(torch.cuda.device_count())   # 应该 > 0
```

### Q: timm 模型下载失败

**A**: 使用代理或离线模式

```python
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 或使用已下载的模型
cfg = ModelConfig(pretrained=False)
```

### Q: Wandb 登录失败

**A**: 跳过 wandb 或使用离线模式

```bash
# 不使用 wandb
./experiments/run_single_experiment.sh lora_only 8  # 编辑脚本移除 --wandb

# 或离线模式
wandb offline
```

## 📝 数据上传

### 方法 1：使用 scp

```bash
# 本地 → 服务器
scp -r data/personai_icartoonface_rectrain user@server:/path/to/AniTune/data/
```

### 方法 2：使用 rsync（推荐）

```bash
# 支持断点续传
rsync -avz --progress \
  data/personai_icartoonface_rectrain \
  user@server:/path/to/AniTune/data/
```

### 方法 3：直接在服务器下载

```bash
# 如果数据在网盘
# 参考 doc/DOWNLOAD_GUIDE_CN.md
```

## 🎯 完整工作流程示例

```bash
# 1. 设置环境
bash setup_a100.sh

# 2. 验证环境
python3 -c "import torch; print(torch.cuda.is_available())"

# 3. 上传数据（如果需要）
# rsync -avz data/ user@server:/path/to/AniTune/data/

# 4. 准备数据清单
PYTHONPATH=src python scripts/prepare_icartoonface.py \
  --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
  --output data/icartoonface \
  --val-ratio 0.1 --seed 42

# 5. 运行实验
./experiments/run_single_experiment.sh lora_only 8

# 6. 监控训练（另一个终端）
watch -n 1 nvidia-smi

# 7. 查看结果
python experiments/analyze_results.py
```

## 📚 相关文档

- [实验框架文档](experiments/INDEX.md)
- [快速开始指南](experiments/QUICK_START.md)
- [完整总结](FINAL_SUMMARY.md)
- [代码修复说明](experiments/CHANGELOG.md)

---

**祝实验顺利！如有问题，查看上述常见问题或联系团队。** 🚀
