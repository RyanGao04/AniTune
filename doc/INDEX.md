# 📚 实验文档索引

欢迎使用 AniTune 实验对比框架！

## 🎯 你应该从哪里开始？

### 如果你想快速开始
→ 阅读 **[QUICK_START.md](QUICK_START.md)**
- 10秒验证环境
- 运行第一个实验
- 查看结果

### 如果你想了解完整功能
→ 阅读 **[README.md](README.md)**
- 详细的实验设计
- 所有4种训练模式
- Rank ablation 实验
- 论文撰写建议

### 如果你想知道修复了什么问题
→ 阅读 **[CHANGELOG.md](CHANGELOG.md)**
- 原代码的问题分析
- 修复方案详解
- 验证方法

## 📂 文件说明

### 核心代码
- **../src/anitune/models.py** - 修复后的模型构建（4种训练模式）
- **train_experiments.py** - 实验训练脚本
- **analyze_results.py** - 结果分析和表格生成

### 配置文件
- **configs/base_experiment.yaml** - 基础实验配置

### 便捷脚本
- **run_all_experiments.sh** - 运行所有实验（一键）
- **run_single_experiment.sh** - 运行单个实验

### 文档
- **INDEX.md** - 本文件
- **QUICK_START.md** - 快速开始
- **README.md** - 完整文档
- **CHANGELOG.md** - 修复日志

## 🚀 常用命令

```bash
# 快速测试（无需数据）
PYTHONPATH=src python -c "from anitune.models import build_model, ModelConfig; print('✓ 环境正常')"

# 运行单个实验
./experiments/run_single_experiment.sh lora_only 8

# 运行所有实验
./experiments/run_all_experiments.sh

# 分析结果
python experiments/analyze_results.py
```

## 🎓 实验模式速查

| 模式 | 命令 | 参数量 | 用途 |
|------|------|--------|------|
| Head-only | `./run_single_experiment.sh head_only` | 0.01% | Baseline |
| Full FT | `./run_single_experiment.sh full_ft` | 100% | Upper bound |
| LoRA r=8 | `./run_single_experiment.sh lora_only 8` | 0.35% | 推荐 |
| LoRA r=16 | `./run_single_experiment.sh lora_only 16` | 0.70% | 高准确率 |

## 📊 预期结果

| 模式 | 准确率 | 训练时间 | 显存 |
|------|--------|---------|------|
| Head-only | 75-80% | 1.2h | 8GB |
| Full FT | 92-95% | 3.5h | 24GB |
| LoRA r=8 | 91-94% | 2.0h | 12GB |
| LoRA r=16 | 92-95% | 2.2h | 14GB |

## ❓ 常见问题

**Q: 我应该用哪种模式？**
- 论文/报告：运行所有模式对比
- 实际应用：`lora_only` (r=8 或 16)
- 快速验证：`head_only` + `lora_only`

**Q: 需要多长时间？**
- 单个实验：1-3小时
- 所有实验：8-10小时
- 快速验证（2个实验）：3-4小时

**Q: 显存不够怎么办？**
- 减小 `batch_size`（配置文件中）
- 使用更小的 rank（如 r=4）

## 🔗 相关文档

- [原项目 README](../doc/README.md) - 项目概述
- [数据准备指南](../doc/DATA_FORMAT_CN.md) - 数据集说明
- [错误分析文档](../doc/ERROR_ANALYSIS.md) - 结果分析

---

**选择一个文档开始吧！建议从 [QUICK_START.md](QUICK_START.md) 开始 🚀**
