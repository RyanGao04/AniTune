#!/bin/bash
# 修复iCartoonFace文件夹命名格式
# 将 personai_icartoonface_rectrain_00000 重命名为 00000

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  修复文件夹命名格式"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

DATA_ROOT="data/personai_icartoonface_rectrain/icartoonface_rectrain"

if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 错误：找不到目录 $DATA_ROOT"
    exit 1
fi

cd "$DATA_ROOT"

# 检查是否需要重命名
SAMPLE_DIR=$(ls -d */ 2>/dev/null | head -1)
if [[ "$SAMPLE_DIR" != personai_icartoonface_rectrain_* ]]; then
    echo "✓ 文件夹命名格式已经正确，无需修复"
    echo "  示例: $SAMPLE_DIR"
    exit 0
fi

echo "检测到需要重命名的文件夹格式："
echo "  当前: personai_icartoonface_rectrain_00000/"
echo "  目标: 00000/"
echo ""

# 统计需要重命名的文件夹
COUNT=$(ls -d personai_icartoonface_rectrain_* 2>/dev/null | wc -l)
echo "找到 $COUNT 个需要重命名的文件夹"
echo ""

read -p "是否继续重命名？(y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消操作"
    exit 0
fi

echo ""
echo "开始重命名..."

SUCCESS=0
FAILED=0

for dir in personai_icartoonface_rectrain_*; do
    if [ -d "$dir" ]; then
        # 提取数字部分
        new_name=$(echo "$dir" | sed 's/personai_icartoonface_rectrain_//')
        
        if [ -d "$new_name" ]; then
            echo "  ⚠️  跳过 $dir (目标已存在)"
            FAILED=$((FAILED + 1))
        else
            mv "$dir" "$new_name"
            SUCCESS=$((SUCCESS + 1))
            if [ $((SUCCESS % 500)) -eq 0 ]; then
                echo "  进度: $SUCCESS/$COUNT"
            fi
        fi
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  重命名完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  成功: $SUCCESS"
echo "  失败: $FAILED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 显示前几个文件夹
echo "前5个文件夹（重命名后）："
ls -d */ | head -5

echo ""
echo "下一步：运行 ./check_data.sh 验证数据集"

