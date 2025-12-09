#!/bin/bash
# 下载 iCartoonFace Recognition 测试集标签文件

echo "=================================="
echo "下载 iCartoonFace Recognition 测试集标签"
echo "=================================="
echo ""

# 爱奇艺网盘信息
echo "📥 下载源：爱奇艺网盘"
echo "🔗 链接：https://fft.cloud.iqiyi.com/s/bUbdw5A"
echo "🔑 密码：X6fgYZ"
echo ""
echo "📝 说明："
echo "   1. 访问上面的链接"
echo "   2. 输入密码：X6fgYZ"
echo "   3. 找到 Recognition 测试集标签文件"
echo "   4. 下载到本地"
echo "   5. 将文件移动到项目的 data/ 目录"
echo ""
echo "⚠️  注意："
echo "   - 确保下载的是 Recognition 测试集标签（不是 Detection）"
echo "   - 文件名可能类似：icartoonface_rectest_label.txt"
echo ""

# 检查是否已下载
if [ -f "data/icartoonface_rectest_label.txt" ]; then
    echo "✓ 找到标签文件：data/icartoonface_rectest_label.txt"
    echo ""
    echo "标签文件格式预览："
    head -5 data/icartoonface_rectest_label.txt
    echo "..."
    echo ""
    echo "总行数：$(wc -l < data/icartoonface_rectest_label.txt)"
else
    echo "❌ 未找到标签文件"
    echo ""
    echo "请按照上述步骤下载标签文件，然后将其放置到："
    echo "   $(pwd)/data/icartoonface_rectest_label.txt"
fi

echo ""
echo "=================================="
