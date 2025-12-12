# 🔬 ViT Fine-tuning 实验框架

修复后的训练代码已整理到 `experiments/` 目录。

## ✨ 主要改进

### 1. 修复了关键Bug
- ✅ **LoRA-only 模式现在正确冻结 backbone**
- ✅ 明确定义4种训练模式
- ✅ 详细的参数统计

### 2. 系统化的实验对比
- 🎯 Head-only fine-tuning (baseline)
- 🎯 Full fine-tuning (upper bound)
- 🎯 LoRA-only adaptation (推荐)
- 🎯 Rank ablation (r ∈ {4, 8, 16, 24, 32})

### 3. 完整的工具链
- 一键运行所有实验
- 自动结果分析和表格生成
- Wandb 集成（可选）

## 📁 文件结构

```
experiments/
├── INDEX.md                    ← 从这里开始！
├── QUICK_START.md              ← 快速开始指南
├── README.md                   ← 完整文档
├── CHANGELOG.md                ← 修复说明
├── (模型代码在 ../src/anitune/models.py)
├── train_experiments.py        ← 训练脚本
├── run_all_experiments.sh      ← 运行所有实验
├── run_single_experiment.sh    ← 运行单个实验
├── analyze_results.py          ← 结果分析
└── configs/
    └── base_experiment.yaml    ← 基础配置
```

## 🚀 快速开始（3步）

### 步骤1：查看文档索引
```bash
cat experiments/INDEX.md
```

### 步骤2：测试环境（无需数据）
```bash
PYTHONPATH=src python src/anitune/models.py
```

### 步骤3：运行第一个实验
```bash
./experiments/run_single_experiment.sh lora_only 8
```

## 📊 关键发现

原代码的 "LoRA" 模式实际上是 **LoRA + 部分 backbone**，不是真正的参数高效微调。

| 指标 | 原代码 (错误) | 修复后 (正确) |
|------|-------------|-------------|
| Backbone 冻结 | ❌ 部分 | ✅ 完全 |
| 可训练参数 | ~10M+ | ~300K |
| 参数比例 | >10% | 0.35% |

## 🎯 推荐使用方式

### 对于初学者
1. 运行 `head_only` (baseline)
2. 运行 `lora_only` (推荐方法)
3. 对比结果

### 对于论文/报告
```bash
# 运行完整对比实验
./experiments/run_all_experiments.sh

# 生成表格
python experiments/analyze_results.py
```

## 📚 详细文档

- **[experiments/INDEX.md](experiments/INDEX.md)** - 文档导航
- **[experiments/QUICK_START.md](experiments/QUICK_START.md)** - 快速开始
- **[experiments/README.md](experiments/README.md)** - 完整文档
- **[experiments/CHANGELOG.md](experiments/CHANGELOG.md)** - 修复详情

## ⚠️ 注意

原始的 `scripts/train.py` 仍然保留，但有 bug。
**请使用 `experiments/` 目录下的修复版本。**

---

**开始实验：**
```bash
cd /Users/tdu/Documents/GitHub/AniTune
cat experiments/INDEX.md  # 阅读导航
```
