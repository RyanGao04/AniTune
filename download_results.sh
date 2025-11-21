#!/bin/bash
# 从远程服务器下载训练结果到本地

# 远程服务器配置
REMOTE_HOST="root@34.68.208.1"
REMOTE_PORT="9870"
REMOTE_PATH="/workspace/AniTune"
LOCAL_PATH="/Users/tdu/Documents/GitHub/AniTune"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  从远程服务器下载训练结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "远程服务器: ${REMOTE_HOST}:${REMOTE_PORT}"
echo "远程路径: ${REMOTE_PATH}"
echo "本地路径: ${LOCAL_PATH}"
echo ""

# 创建本地目录
mkdir -p "${LOCAL_PATH}/runs"
mkdir -p "${LOCAL_PATH}/wandb"
mkdir -p "${LOCAL_PATH}/logs"

# 1. 下载runs目录（训练checkpoints）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 下载 runs/ 目录（模型checkpoints）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rsync -avz --progress \
  -e "ssh -p ${REMOTE_PORT}" \
  ${REMOTE_HOST}:${REMOTE_PATH}/runs/ \
  ${LOCAL_PATH}/runs/

# 2. 下载wandb日志（如果存在）
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. 下载 wandb/ 目录（训练日志）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rsync -avz --progress \
  -e "ssh -p ${REMOTE_PORT}" \
  ${REMOTE_HOST}:${REMOTE_PATH}/wandb/ \
  ${LOCAL_PATH}/wandb/ 2>/dev/null || echo "wandb目录不存在，跳过"

# 3. 下载训练日志文件（如果有保存）
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 下载 logs/ 目录（训练日志文件）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rsync -avz --progress \
  -e "ssh -p ${REMOTE_PORT}" \
  ${REMOTE_HOST}:${REMOTE_PATH}/logs/ \
  ${LOCAL_PATH}/logs/ 2>/dev/null || echo "logs目录不存在，跳过"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ 下载完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "文件保存在:"
echo "  ${LOCAL_PATH}/runs/"
echo "  ${LOCAL_PATH}/wandb/"
echo "  ${LOCAL_PATH}/logs/"
echo ""
echo "查看下载的文件:"
echo "  ls -lh ${LOCAL_PATH}/runs/"
echo ""

