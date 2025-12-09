# 训练时间查看指南

## 快速查看当前训练时间

```bash
python check_training_time.py
```

或者查看特定 run：
```bash
python check_training_time.py wandb/run-20251121_045942-tlzuc518
```

## 当前训练统计 (Run: 2025-11-21)

### 时间信息
- **开始时间**: 2025-11-21 04:59:42 UTC (vast.ai A100)
- **已运行时长**: 1小时10分钟
- **已完成**: 9 / 12 epochs
- **平均每个 Epoch**: ~7分49秒
- **估计剩余时间**: ~23分钟 (3 epochs)

### 硬件配置
- **GPU**: NVIDIA A100-SXM4-40GB
- **显存**: 40 GB
- **平台**: vast.ai
- **成本**: ~$0.35-0.50/小时

### 训练配置
- **Batch size**: 64
- **Total epochs**: 12
- **Iterations per epoch**: 2,452
- **Training images**: ~157K
- **Validation images**: ~17K

## 其他查看方法

### 1. 查看 wandb 输出日志
```bash
cat wandb/latest-run/files/output.log
```

### 2. 查看进度条信息
训练时会显示类似：
```
train: 81%|█████████████▋| 1980/2452 [05:27<01:17, 6.11it/s, acc=0.984, loss=0.0621]
```
- `1980/2452`: 当前/总 iterations
- `[05:27<01:17]`: 已用时间 < 剩余时间
- `6.11it/s`: 每秒处理的 iterations

### 3. 在 Weights & Biases 查看
访问你的 wandb 项目页面：
https://wandb.ai/your-username/AniTune

可以看到：
- 实时训练曲线
- 系统资源使用（GPU/内存）
- 完整的训练历史

## 训练时间估算公式

```python
# 每个 epoch 时间
time_per_epoch = total_training_samples / batch_size / iterations_per_second

# 对于我们的设置：
# ~157,000 samples / 64 batch size / 6.11 it/s ≈ 400秒 ≈ 6.7分钟

# 总训练时间
total_time = time_per_epoch × num_epochs
# 6.7分钟 × 12 epochs ≈ 80分钟 ≈ 1.3小时
```

## 训练成本估算

基于 vast.ai A100 定价：
```
成本 = 小时数 × 每小时价格
     = 1.5小时 × $0.40/小时
     ≈ $0.60
```

💡 **提示**: 比 AWS/GCP 便宜约 80%！

## 优化建议

如果训练太慢，可以尝试：
1. **增加 batch size**: 64 → 128 (需要更多显存)
2. **减少 workers**: 16 → 8 (避免 CPU 瓶颈)
3. **启用 AMP**: 混合精度训练 (已启用)
4. **使用更快的 GPU**: A100 → H100 (如果预算允许)

如果训练太快导致过拟合：
1. **增加 epochs**: 12 → 20
2. **降低 learning rate**: 2e-4 → 1e-4
3. **添加更多 augmentation**
4. **使用 cosine scheduler**


