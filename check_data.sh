#!/bin/bash
# 检查iCartoonFace Recognition数据集是否正确

set +e  # 允许错误继续执行

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  iCartoonFace Recognition 数据集检查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

DATA_ROOT="data/personai_icartoonface_rectrain/icartoonface_rectrain"
HAS_ERROR=0

# 检查数据根目录
echo "步骤1: 检查数据目录..."
if [ ! -d "$DATA_ROOT" ]; then
    echo "  ❌ 错误：找不到目录 $DATA_ROOT"
    echo "     请确保数据集已下载并解压到正确位置"
    echo ""
    echo "  预期结构："
    echo "  data/"
    echo "  └── personai_icartoonface_rectrain/"
    echo "      └── icartoonface_rectrain/  ← 这里应该有5013个文件夹"
    echo "          ├── 00001/"
    echo "          ├── 00002/"
    echo "          └── ..."
    HAS_ERROR=1
else
    echo "  ✓ 找到数据根目录: $DATA_ROOT"
fi

# 统计文件夹数量
echo ""
echo "步骤2: 统计角色文件夹数量..."
if [ -d "$DATA_ROOT" ]; then
    NUM_DIRS=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  ✓ 角色文件夹数量: $NUM_DIRS"
    
    if [ "$NUM_DIRS" -ne 5013 ]; then
        echo "  ⚠️  警告：预期5013个文件夹，实际找到 $NUM_DIRS 个"
        echo "     可能原因："
        echo "     1. 数据集下载不完整"
        echo "     2. 解压出错"
        echo "     3. 下载了错误的数据集（Detection而非Recognition）"
        HAS_ERROR=1
    else
        echo "  ✓ 文件夹数量正确！(5013个角色)"
    fi
fi

# 统计总图片数量
echo ""
echo "步骤3: 统计图片数量..."
if [ -d "$DATA_ROOT" ]; then
    NUM_IMAGES=$(find "$DATA_ROOT" -name "*.jpg" 2>/dev/null | wc -l)
    echo "  ✓ 总图片数量: $NUM_IMAGES"
    
    if [ "$NUM_IMAGES" -lt 100000 ]; then
        echo "  ⚠️  警告：图片数量似乎太少了"
        echo "     预期约300,000-400,000张图片"
        HAS_ERROR=1
    elif [ "$NUM_IMAGES" -gt 500000 ]; then
        echo "  ⚠️  警告：图片数量似乎太多了"
        echo "     可能包含了其他数据"
    else
        echo "  ✓ 图片数量正常（预期范围内）"
    fi
fi

# 检查文件夹命名格式
echo ""
echo "步骤4: 检查文件夹命名格式..."
if [ -d "$DATA_ROOT" ]; then
    FIRST_DIRS=$(ls -d "$DATA_ROOT"/*/ 2>/dev/null | head -5 | xargs -n 1 basename)
    echo "  前5个文件夹:"
    echo "$FIRST_DIRS" | while read dir; do
        echo "    - $dir"
    done
    
    # 检查是否符合预期格式（00001, 00002等）
    FIRST_DIR=$(ls -d "$DATA_ROOT"/*/ 2>/dev/null | head -1 | xargs basename)
    if [[ "$FIRST_DIR" =~ ^[0-9]{5}$ ]]; then
        echo "  ✓ 文件夹命名格式正确（5位数字）"
    else
        echo "  ⚠️  警告：文件夹命名格式可能不对"
        echo "     预期格式：00001, 00002, 00003, ..."
        echo "     实际格式：$FIRST_DIR"
        HAS_ERROR=1
    fi
fi

# 检查第一个文件夹的内容
echo ""
echo "步骤5: 检查文件夹内容..."
if [ -d "$DATA_ROOT" ]; then
    FIRST_DIR_PATH=$(ls -d "$DATA_ROOT"/*/ 2>/dev/null | head -1)
    if [ -n "$FIRST_DIR_PATH" ]; then
        NUM_IMGS_IN_FIRST=$(ls "$FIRST_DIR_PATH"*.jpg 2>/dev/null | wc -l)
        echo "  第一个文件夹: $(basename "$FIRST_DIR_PATH")"
        echo "  图片数量: $NUM_IMGS_IN_FIRST"
        
        if [ "$NUM_IMGS_IN_FIRST" -gt 0 ]; then
            echo "  前3张图片:"
            ls "$FIRST_DIR_PATH"*.jpg 2>/dev/null | head -3 | xargs -n 1 basename | while read img; do
                echo "    - $img"
            done
            echo "  ✓ 文件夹包含图片"
        else
            echo "  ❌ 错误：文件夹是空的"
            HAS_ERROR=1
        fi
    fi
fi

# 检查是否误下载了Detection数据集
echo ""
echo "步骤6: 检查数据集类型..."
if [ -d "$DATA_ROOT" ]; then
    # 检查是否有detection特征的文件名
    DETECTION_FILES=$(find "$DATA_ROOT" -name "*det*" 2>/dev/null | head -1)
    if [ -n "$DETECTION_FILES" ]; then
        echo "  ⚠️  警告：发现包含'det'的文件，可能下载了Detection数据集"
        echo "     本项目需要Recognition数据集（rectrain/rectest）"
        HAS_ERROR=1
    else
        echo "  ✓ 数据集类型正确（Recognition）"
    fi
fi

# 检查是否已生成清单文件
echo ""
echo "步骤7: 检查训练清单文件..."
MANIFEST_DIR="data/icartoonface/splits"
if [ -f "$MANIFEST_DIR/train.txt" ] && [ -f "$MANIFEST_DIR/val.txt" ]; then
    echo "  ✓ 找到训练和验证集清单文件"
    TRAIN_LINES=$(wc -l < "$MANIFEST_DIR/train.txt")
    VAL_LINES=$(wc -l < "$MANIFEST_DIR/val.txt")
    echo "    - 训练集: $TRAIN_LINES 张图片"
    echo "    - 验证集: $VAL_LINES 张图片"
else
    echo "  ⚠️  未找到清单文件"
    echo "     运行以下命令生成："
    echo ""
    echo "     python scripts/prepare_icartoonface.py \\"
    echo "       --source $DATA_ROOT \\"
    echo "       --output data/icartoonface \\"
    echo "       --val-ratio 0.1 --seed 42"
fi

# 总结
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $HAS_ERROR -eq 0 ]; then
    echo "  ✅ 数据集检查通过！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "下一步："
    if [ ! -f "$MANIFEST_DIR/train.txt" ]; then
        echo "  1. 生成训练清单:"
        echo "     python scripts/prepare_icartoonface.py \\"
        echo "       --source $DATA_ROOT \\"
        echo "       --output data/icartoonface \\"
        echo "       --val-ratio 0.1 --seed 42"
        echo ""
    fi
    echo "  2. 开始训练:"
    echo "     PYTHONPATH=src python scripts/train.py \\"
    echo "       --config configs/lora_vitb16_a100_balanced.yaml \\"
    echo "       --data-root $DATA_ROOT"
else
    echo "  ❌ 发现问题，请检查上述警告和错误"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "常见问题解决："
    echo "  1. 确认下载了Recognition数据集（不是Detection）"
    echo "  2. 文件名应该包含'rectrain'或'rectest'"
    echo "  3. 解压后应该有5013个按数字命名的文件夹"
    echo "  4. 查看详细说明: cat DATA_FORMAT_CN.md"
fi

echo ""

