#!/bin/bash
# 下载特定的训练实验结果

# 远程服务器配置
REMOTE_HOST="root@34.68.208.1"
REMOTE_PORT="9870"
REMOTE_PATH="/workspace/AniTune"
LOCAL_PATH="/Users/tdu/Documents/GitHub/AniTune"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  下载特定训练实验"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 列出远程可用的实验
echo "正在获取远程实验列表..."
ssh -p ${REMOTE_PORT} ${REMOTE_HOST} "ls -la ${REMOTE_PATH}/runs/"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "请输入要下载的实验名称（例如: lora_vitb16_a100_balanced）: " RUN_NAME

if [ -z "$RUN_NAME" ]; then
    echo "❌ 未输入实验名称，退出"
    exit 1
fi

echo ""
echo "下载实验: ${RUN_NAME}"
echo ""

# 创建本地目录
mkdir -p "${LOCAL_PATH}/runs/${RUN_NAME}"

# 下载特定实验
rsync -avz --progress \
  -e "ssh -p ${REMOTE_PORT}" \
  ${REMOTE_HOST}:${REMOTE_PATH}/runs/${RUN_NAME}/ \
  ${LOCAL_PATH}/runs/${RUN_NAME}/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ 下载完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "文件保存在: ${LOCAL_PATH}/runs/${RUN_NAME}/"
echo ""
echo "查看下载的文件:"
ls -lh "${LOCAL_PATH}/runs/${RUN_NAME}/"
echo ""

