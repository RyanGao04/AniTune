# 🚀 快速开始指南

## 📋 前置要求

1. **数据准备完成**
   ```bash
   # 检查数据
   ls data/personai_icartoonface_rectrain/icartoonface_rectrain/ | wc -l
   # 应该输出: 5013

   # 检查 manifest
   ls data/icartoonface/splits/
   # 应该看到: train.txt, val.txt
   ```

2. **环境设置**
   ```bash
   conda activate anitune
   pip install -r requirements.txt
   ```

## ⚡ 10秒快速测试

```bash
cd /Users/tdu/Documents/GitHub/AniTune

# 测试模型参数统计（无需数据）
PYTHONPATH=src python src/anitune/models.py
```

你会看到4种模式的参数对比。

## 🎯 运行单个实验（推荐从这里开始）

### 实验1：LoRA-only (r=8) - 推荐

```bash
./experiments/run_single_experiment.sh lora_only 8
```

**预期**：
- 训练时间: ~2小时 (10 epochs)
- 可训练参数: ~300K (0.35%)
- 验证准确率: ~91-94%

### 实验2：Head-only (baseline)

```bash
./experiments/run_single_experiment.sh head_only
```

**预期**：
- 训练时间: ~1.2小时
- 可训练参数: ~10K (0.01%)
- 验证准确率: ~75-80%

### 实验3：Full Fine-tuning

```bash
./experiments/run_single_experiment.sh full_ft
```

**预期**：
- 训练时间: ~3.5小时
- 可训练参数: ~86M (100%)
- 验证准确率: ~92-95%

## 📊 运行完整对比实验

```bash
# 运行所有实验（约8-10小时）
./experiments/run_all_experiments.sh
```

包括：
1. Head-only
2. Full FT
3. LoRA r=8
4. LoRA r=4, 16, 24, 32 (rank ablation)

## 📈 查看结果

### 方法1：命令行查看

```bash
# 分析所有实验
python experiments/analyze_results.py
```

输出示例：
```
| Experiment                           | Val Acc | Trainable Params | Ratio  |
|--------------------------------------|---------|-----------------|--------|
| vit_experiment_lora_only_r16         | 0.9384  |      600,000    |  0.70% |
| vit_experiment_full_ft               | 0.9352  |   86,000,000    | 100.0% |
| vit_experiment_lora_only_r8          | 0.9312  |      300,000    |  0.35% |
```

### 方法2：Wandb 在线查看

如果使用了 `--wandb` 标志：
1. 访问 https://wandb.ai
2. 登录你的账号
3. 查看项目：AniTune-Experiments

## 🎨 自定义实验

### 修改超参数

编辑 `experiments/configs/base_experiment.yaml`：

```yaml
optim:
  lr: 3.0e-4        # 学习率
  weight_decay: 0.05
  epochs: 15        # 增加训练轮数
  batch_size: 128   # 更大的 batch size（需要更多显存）
```

### 测试不同 LoRA rank

```bash
# 快速测试多个 rank
for RANK in 4 8 12 16 20 24; do
    ./experiments/run_single_experiment.sh lora_only $RANK
done
```

### 添加新实验

复制并修改 `experiments/configs/base_experiment.yaml`：

```bash
cp experiments/configs/base_experiment.yaml \
   experiments/configs/my_experiment.yaml

# 编辑配置
vim experiments/configs/my_experiment.yaml

# 运行
PYTHONPATH=src python experiments/train_experiments.py \
    --config experiments/configs/my_experiment.yaml \
    --mode lora_only \
    --lora-rank 12 \
    --device cuda
```

## 🔧 常见问题

### Q: 训练太慢怎么办？

**A**: 减少 epochs 或使用更小的 rank：
```bash
# 修改配置文件，设置 epochs: 5
# 或使用 rank=4
./experiments/run_single_experiment.sh lora_only 4
```

### Q: 显存不足 (CUDA out of memory)

**A**: 减小 batch_size：
```yaml
# experiments/configs/base_experiment.yaml
data:
  batch_size: 32  # 从 64 减到 32
```

### Q: 如何恢复中断的训练？

**A**: 当前代码不支持 resume。建议：
1. 增加 checkpoint 保存频率
2. 或修改 `train_loop.py` 添加 resume 功能

### Q: Wandb 登录失败

**A**:
```bash
# 首次使用需要登录
wandb login

# 或者不使用 wandb
# 编辑脚本，移除 --wandb 标志
```

## 📝 论文用图表

### 生成对比表格

```bash
python experiments/analyze_results.py > results.txt

# 查看 Markdown 表格
cat results.txt

# LaTeX 表格也在输出中
```

### 导出到 CSV

```python
import json
import pandas as pd

# 读取结果
with open('experiments/runs/results_summary.json') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV
df.to_csv('results.csv', index=False)
```

## 🎯 推荐的实验顺序

### 对于初学者：

1. **先跑一个快速实验**（验证环境）
   ```bash
   ./experiments/run_single_experiment.sh lora_only 8
   ```

2. **对比 baseline**
   ```bash
   ./experiments/run_single_experiment.sh head_only
   ```

3. **如果时间充足，跑完整对比**
   ```bash
   ./experiments/run_all_experiments.sh
   ```

### 对于论文/报告：

1. **完整实验**
   ```bash
   ./experiments/run_all_experiments.sh
   ```

2. **生成表格**
   ```bash
   python experiments/analyze_results.py
   ```

3. **记录结果**
   - 截图 Wandb dashboard
   - 复制 LaTeX 表格
   - 保存 results_summary.json

## 📚 下一步

- 阅读完整文档：[experiments/README.md](README.md)
- 查看原代码问题分析：[../doc/README.md](../doc/README.md)
- 修改训练脚本添加更多功能

---

**祝实验顺利！🎉**
