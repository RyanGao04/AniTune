#!/bin/bash
# 从Google Drive下载iCartoonFace数据集

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  从Google Drive下载iCartoonFace Recognition数据集"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 检查gdown是否安装
if ! command -v gdown &> /dev/null; then
    echo "步骤1: 安装gdown..."
    pip install gdown
else
    echo "✓ gdown已安装"
fi

# 创建数据目录
mkdir -p data
cd data

echo ""
echo "步骤2: 从Google Drive下载数据集..."
echo ""
echo "Google Drive文件夹: "
echo "https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW"
echo ""

# 方法1: 如果知道具体文件ID
# echo "请输入Google Drive文件ID（从分享链接中获取）:"
# echo "例如: https://drive.google.com/file/d/FILE_ID/view"
# read -p "文件ID: " FILE_ID

# 使用gdown下载
# gdown --id "$FILE_ID" --output personai_icartoonface_rectrain.zip

# 方法2: 从文件夹下载（需要gdown >= 4.6.0）
echo "尝试从文件夹下载..."
echo ""
echo "注意: 如果这是一个共享文件夹，你可能需要："
echo "1. 在浏览器中打开链接"
echo "2. 找到 personai_icartoonface_rectrain.zip 文件"
echo "3. 获取该文件的直接分享链接"
echo "4. 从链接中提取FILE_ID"
echo ""
echo "文件ID格式: https://drive.google.com/file/d/FILE_ID/view"
echo ""

# 尝试使用gdown的folder功能（可能需要更高版本）
# gdown --folder https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  手动下载步骤（推荐）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "如果自动下载失败，请使用以下步骤："
echo ""
echo "1. 在浏览器打开:"
echo "   https://drive.google.com/drive/folders/1m6pAL9Wbn8B1td0hFUj9RVRrSweNKskW"
echo ""
echo "2. 找到文件: personai_icartoonface_rectrain.zip"
echo ""
echo "3. 右键点击 -> 获取链接 -> 复制链接"
echo ""
echo "4. 使用gdown下载:"
echo "   cd /workspace/AniTune/data"
echo "   gdown 'YOUR_LINK_HERE'"
echo ""
echo "   或者如果有文件ID:"
echo "   gdown --id FILE_ID --output personai_icartoonface_rectrain.zip"
echo ""
echo "5. 下载完成后，解压并验证:"
echo "   unzip personai_icartoonface_rectrain.zip"
echo "   cd /workspace/AniTune"
echo "   ./check_data.sh"
echo ""

