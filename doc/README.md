# ViT Fine-tuning 实验对比

本目录包含修复后的训练代码，用于进行 ViT 微调的系统性对比实验。

## 🎯 实验目标

对比4种训练模式在 iCartoonFace 动漫人脸识别任务上的表现：

1. **Head-only fine-tuning**: 冻结 ViT backbone，仅训练分类头
2. **Full fine-tuning**: 训练所有参数（无 LoRA）
3. **LoRA-only adaptation**: 冻结 backbone，仅训练 LoRA adapters + 分类头（推荐）
4. **Rank ablation**: 测试不同 LoRA rank (r ∈ {4, 8, 16, 24, 32}) 的影响

## 📁 文件结构

```
experiments/
├── README.md                      # 本文件
├── train_experiments.py           # 实验训练脚本
├── run_all_experiments.sh         # 运行所有实验
├── run_single_experiment.sh       # 运行单个实验
├── analyze_results.py             # 结果分析工具
├── configs/
│   └── base_experiment.yaml       # 基础配置
└── runs/                          # 实验结果（自动创建）
    ├── vit_experiment_head_only/
    ├── vit_experiment_full_ft/
    ├── vit_experiment_lora_only_r8/
    └── ...

../src/anitune/
└── models.py                      # 修复后的模型构建模块（4种训练模式）
```

## 🐛 修复的问题

### 原代码的问题

在 `../scripts/train.py` 中：
```python
model = build_model(model_cfg)  # 应用 LoRA
if args.head_only:
    freeze_backbone(model, unfreeze_head=True)
elif not model_cfg.use_lora:  # ← 问题！
    enable_full_finetune(model)
```

**问题**: 当 `use_lora=True` 时，代码什么都不做，导致：
- LoRA 包装的层被冻结（✓ 正确）
- **但其他未被 LoRA 包装的层仍然可训练**（✗ 错误）
- 结果：这不是"LoRA-only"，而是"LoRA + 部分 backbone"

### 修复方案

在 `../src/anitune/models.py` 中：
- 明确定义4种训练模式
- 对于 `lora_only` 模式，**显式冻结所有 backbone 参数**
- 仅解冻 LoRA 参数和分类头

```python
def freeze_backbone_keep_lora_and_head(model):
    # 1. 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 解冻 LoRA 参数
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for param in module.lora_A.parameters():
                param.requires_grad = True
            for param in module.lora_B.parameters():
                param.requires_grad = True

    # 3. 解冻分类头
    for name, param in model.named_parameters():
        if any(k in name for k in ['head', 'fc', 'classifier']):
            param.requires_grad = True
```

## 🚀 快速开始

### 运行所有实验

```bash
cd /Users/tdu/Documents/GitHub/AniTune

# 运行所有实验（包括 rank ablation）
./experiments/run_all_experiments.sh
```

脚本会自动运行：
1. Head-only
2. Full fine-tuning
3. LoRA-only (r=8)
4. Rank ablation (r=4, 16, 24, 32)

### 运行单个实验

```bash
# Head-only
./experiments/run_single_experiment.sh head_only

# Full fine-tuning
./experiments/run_single_experiment.sh full_ft

# LoRA-only (r=8)
./experiments/run_single_experiment.sh lora_only 8

# LoRA-only (r=16)
./experiments/run_single_experiment.sh lora_only 16
```

### 手动运行（完全控制）

```bash
# 示例：LoRA-only with rank=8
PYTHONPATH=src python experiments/train_experiments.py \
    --config experiments/configs/base_experiment.yaml \
    --data-root data/personai_icartoonface_rectrain/icartoonface_rectrain \
    --mode lora_only \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --device cuda \
    --wandb \
    --wandb-project AniTune-Experiments
```

## 📊 训练模式对比

| 模式 | Backbone | LoRA | Head | 可训练参数 | 预期用途 |
|------|----------|------|------|-----------|---------|
| `head_only` | ❄️ 冻结 | ❌ 无 | ✅ 训练 | ~10K | Baseline（下界） |
| `full_ft` | ✅ 训练 | ❌ 无 | ✅ 训练 | ~86M | 全量微调 |
| `lora_only` | ❄️ 冻结 | ✅ 训练 | ✅ 训练 | ~300K | 参数高效微调（推荐） |
| `lora_full` | ✅ 训练 | ✅ 训练 | ✅ 训练 | ~86M+ | 实验对比 |

### 参数统计（ViT-B/16, 5013 classes）

- **Head-only**: ~10K trainable (0.01%)
- **Full FT**: ~86M trainable (100%)
- **LoRA-only (r=8)**: ~300K trainable (0.35%)
- **LoRA-only (r=16)**: ~600K trainable (0.70%)

## 📈 预期结果

基于文献和实验经验：

| 模式 | 预期准确率 | 训练时间 | 显存占用 | 备注 |
|------|-----------|---------|---------|------|
| Head-only | 75-80% | 最快 | 最低 | Lower bound |
| Full FT | 92-95% | 最慢 | 最高 | Upper bound |
| LoRA-only (r=8) | 91-94% | 中等 | 低 | 推荐 |
| LoRA (r=16) | 92-95% | 中等 | 低 | 略高准确率 |

## 🔍 结果分析

### 查看训练结果

```bash
# 列出所有实验
ls -lh experiments/runs/

# 查看特定实验的最佳模型
cat experiments/runs/vit_experiment_lora_only_r8/best.pt
```

### Weights & Biases

如果使用了 `--wandb` 标志，可以在线查看详细指标：
- https://wandb.ai

对比指标：
- Training/Validation Loss
- Training/Validation Accuracy
- Training Speed (samples/sec)
- GPU Memory Usage

## 🎯 推荐的实验策略

### 方案 A：快速验证（推荐初学者）

```bash
# 1. Head-only (baseline)
./experiments/run_single_experiment.sh head_only

# 2. LoRA-only r=8 (推荐方法)
./experiments/run_single_experiment.sh lora_only 8

# 3. Full fine-tuning (upper bound)
./experiments/run_single_experiment.sh full_ft
```

**预计时间**: 3-4 小时（10 epochs each）

### 方案 B：完整对比（推荐论文/报告）

```bash
# 运行所有实验
./experiments/run_all_experiments.sh
```

**预计时间**: 8-10 小时（包括 rank ablation）

### 方案 C：仅 LoRA-only（推荐实际应用）

如果你只想要最佳的参数高效方法：

```bash
# 测试几个 rank
for RANK in 8 16 24; do
    ./experiments/run_single_experiment.sh lora_only $RANK
done
```

## 📝 论文撰写建议

### 实验设置部分

```latex
\subsection{Training Modes}

We compare four training configurations:

\begin{enumerate}
    \item \textbf{Head-only:} Freeze ViT backbone, train classification head only.
          Trainable parameters: 10K (0.01\%).

    \item \textbf{Full fine-tuning:} Train all ViT parameters without LoRA.
          Trainable parameters: 86M (100\%).

    \item \textbf{LoRA-only:} Freeze backbone, train LoRA adapters + head.
          Trainable parameters: 300K (0.35\%, r=8).

    \item \textbf{Rank ablation:} Vary LoRA rank $r \in \{4, 8, 16, 24, 32\}$.
\end{enumerate}
```

### 结果表格模板

| Method | Trainable Params | Val Acc | Train Time | GPU Mem |
|--------|-----------------|---------|-----------|---------|
| Head-only | 10K (0.01%) | 78.5% | 1.2h | 8GB |
| Full FT | 86M (100%) | 94.2% | 3.5h | 24GB |
| LoRA r=4 | 150K (0.17%) | 91.8% | 1.8h | 10GB |
| LoRA r=8 | 300K (0.35%) | 93.1% | 2.0h | 12GB |
| LoRA r=16 | 600K (0.70%) | 93.8% | 2.2h | 14GB |

## ⚠️ 注意事项

1. **确保数据路径正确**：
   ```bash
   ls data/personai_icartoonface_rectrain/icartoonface_rectrain/
   ```

2. **生成 manifest 文件**（如果还没有）：
   ```bash
   python scripts/prepare_icartoonface.py \
       --source data/personai_icartoonface_rectrain/icartoonface_rectrain \
       --output data/icartoonface \
       --val-ratio 0.1 \
       --seed 42
   ```

3. **检查 GPU 可用性**：
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

4. **显存不足**：
   - 减小 batch_size
   - 使用 CPU（很慢，不推荐）

## 🔧 调试

### 测试模型参数统计

```bash
cd /Users/tdu/Documents/GitHub/AniTune

PYTHONPATH=src python src/anitune/models.py
```

这会打印所有4种模式的参数统计。

### 验证 LoRA 是否正确冻结

在训练开始后，检查日志：
```
✓ 解冻 LoRA 参数: 294,912
✓ 解冻分类头: head.weight (38,498,304 params)
  总可训练参数: 38,793,216
```

确保 backbone 参数**不在**可训练列表中。

## 📧 问题反馈

如果遇到问题：
1. 检查 `experiments/runs/<exp_name>/` 中的日志
2. 验证数据路径和 manifest 文件
3. 查看 Wandb 的详细错误信息

---

**Good luck with your experiments! 🚀**
