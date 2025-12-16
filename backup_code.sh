#!/bin/bash
# 备份远程服务器代码到本地（排除data目录）

# 远程服务器配置
REMOTE_HOST="root@34.68.208.1"
REMOTE_PORT="9870"
REMOTE_PATH="/workspace/AniTune"
LOCAL_PATH="/Users/tdu/Documents/GitHub/AniTune"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  备份远程代码到本地"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "远程服务器: ${REMOTE_HOST}:${REMOTE_PORT}"
echo "远程路径: ${REMOTE_PATH}/"
echo "本地路径: ${LOCAL_PATH}/"
echo ""
echo "排除的目录/文件:"
echo "  - data/                    (太大)"
echo "  - runs/                    (使用download_results.sh单独下载)"
echo "  - wandb/                   (使用download_results.sh单独下载)"
echo "  - .venv/                   (Python虚拟环境)"
echo "  - __pycache__/             (Python缓存)"
echo "  - *.pyc                    (Python编译文件)"
echo "  - .git/                    (Git仓库)"
echo ""

read -p "是否继续备份？(y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消备份"
    exit 0
fi

# 创建本地目录
mkdir -p "${LOCAL_PATH}"

echo ""
echo "开始备份..."
echo ""

# 使用rsync备份，排除大文件和不需要的目录
rsync -avz --progress \
  -e "ssh -p ${REMOTE_PORT}" \
  --exclude 'data/' \
  --exclude 'runs/' \
  --exclude 'wandb/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.git/' \
  --exclude '.gitignore' \
  --exclude '*.swp' \
  --exclude '*.swo' \
  --exclude '.DS_Store' \
  ${REMOTE_HOST}:${REMOTE_PATH}/ \
  ${LOCAL_PATH}/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ 备份完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "备份内容包括:"
echo "  ✅ 源代码 (src/)"
echo "  ✅ 脚本 (scripts/)"
echo "  ✅ 配置文件 (configs/)"
echo "  ✅ 测试代码 (tests/)"
echo "  ✅ 文档 (*.md)"
echo "  ✅ 依赖文件 (requirements.txt, environment.yml)"
echo ""
echo "排除的内容:"
echo "  ❌ data/ (请使用原始下载源获取)"
echo "  ❌ runs/ (使用 ./download_results.sh 单独下载)"
echo "  ❌ wandb/ (使用 ./download_results.sh 单独下载)"
echo "  ❌ .venv/ (虚拟环境，本地重新创建)"
echo ""
echo "查看备份文件:"
echo "  ls -la ${LOCAL_PATH}/"
echo ""
echo "如需下载训练结果，运行:"
echo "  ./download_results.sh"
echo ""

