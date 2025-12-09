# 错误分析快速入门 🚀

## 一键运行

```bash
./run_error_analysis.sh
```

## 会生成什么？

在 `error_analysis_results/` 目录下生成 5 个文件：

### 1️⃣ 混淆矩阵 (Confusion Matrix)
`confusion_matrix_top50_val.png`

热力图显示哪些类别容易被混淆。

### 2️⃣ 最容易混淆的类别对
`confused_pairs_val.png`

柱状图显示 Top-20 混淆对，例如：
- Class 42 → Class 97: 89 次
- Class 123 → Class 456: 67 次

### 3️⃣ 每类准确率分析
`per_class_accuracy_val.png`

两个子图：
- 左：准确率直方图
- 右：准确率 vs 样本数量（箱线图）

### 4️⃣ 错误样本可视化
`error_samples_visualization.png`

4×5 网格显示 20 个高置信度错误预测的样本。

### 5️⃣ 详细统计报告
`error_statistics_val.json`

包含：
- 总错误数
- 平均错误置信度
- 最差/最好的 20 个类别
- 详细的混淆对列表

## 快速查看结果

```bash
# macOS
open error_analysis_results/

# Linux
xdg-open error_analysis_results/

# 查看 JSON 统计
cat error_analysis_results/error_statistics_val.json | jq .
```

## 常见用途

### 1. 发现数据标注错误
查看 `error_samples_visualization.png`，如果高置信度错误看起来预测是对的，可能标注有误。

### 2. 识别相似类别
查看 `confused_pairs_val.png`，找出经常混淆的类别对，考虑：
- 是否应该合并这些类别？
- 是否需要更多区分特征？

### 3. 分析长尾问题
查看 `per_class_accuracy_val.png` 右图：
- 如果样本少的类别准确率低 → 数据不足
- 如果样本多的类别也准确率低 → 类别本身难以区分

### 4. 定位问题类别
查看 `error_statistics_val.json` 的 `worst_classes`：
```json
"worst_classes": [
  {"class_id": 42, "accuracy": 0.234, "count": 15},
  {"class_id": 97, "accuracy": 0.456, "count": 28},
  ...
]
```

针对这些类别重点改进。

## 在远程服务器上运行

如果模型在远程服务器（如 vast.ai）：

```bash
# 1. SSH 到服务器
ssh -p 9870 root@34.68.208.1

# 2. 运行错误分析
cd /workspace/AniTune
./run_error_analysis.sh

# 3. 下载结果到本地
exit
scp -P 9870 -r root@34.68.208.1:/workspace/AniTune/error_analysis_results ./
```

## 添加到 Progress Report

生成的图表可以直接用于报告：

```latex
\begin{figure}[H]
    \centering
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{error_analysis_results/confusion_matrix_top50_val.png}
        \caption{Confusion matrix}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{error_analysis_results/confused_pairs_val.png}
        \caption{Most confused pairs}
    \end{subfigure}
    \caption{Error analysis on validation set (91.28\% accuracy).}
\end{figure}
```

## 需要帮助？

详细文档：`cat ERROR_ANALYSIS.md`

---

**估计运行时间**: 2-3 分钟 (A100, 17K 验证样本)  
**GPU 内存需求**: ~8GB  
**输出文件大小**: ~5MB

