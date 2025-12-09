#!/usr/bin/env python
"""自动下载 iCartoonFace Recognition 测试集标签文件

尝试多种方法下载标签文件：
1. 直接从可能的直链下载
2. 使用 requests 模拟浏览器访问
3. 提供手动下载的详细说明
"""

import os
import sys
import requests
from pathlib import Path
from urllib.parse import quote


def download_file(url, output_path, headers=None):
    """下载文件

    Args:
        url: 下载链接
        output_path: 保存路径
        headers: 请求头

    Returns:
        bool: 是否成功
    """
    try:
        print(f"尝试从 {url} 下载...")

        if headers is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # 检查是否是 HTML（可能是登录页面）
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"  ❌ 返回的是 HTML 页面，不是文件")
            return False

        # 下载文件
        total_size = int(response.headers.get('content-length', 0))
        print(f"  文件大小: {total_size / 1024 / 1024:.2f} MB" if total_size > 0 else "  文件大小: 未知")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  下载进度: {percent:.1f}%", end='', flush=True)

        print(f"\n  ✓ 下载成功: {output_path}")
        print(f"  文件大小: {output_path.stat().st_size / 1024:.2f} KB")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ❌ 下载失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 发生错误: {e}")
        return False


def verify_label_file(file_path):
    """验证标签文件格式

    Args:
        file_path: 文件路径

    Returns:
        bool: 是否是有效的标签文件
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            print("  ❌ 文件为空")
            return False

        # 检查前几行的格式
        valid_lines = 0
        for i, line in enumerate(lines[:10]):
            parts = line.strip().split()
            # 格式：filename x1 y1 x2 y2 label_id
            if len(parts) == 6:
                try:
                    # 验证坐标和标签是数字
                    x1, y1, x2, y2, label_id = map(int, parts[1:])
                    valid_lines += 1
                except ValueError:
                    pass

        if valid_lines >= 5:
            print(f"  ✓ 文件格式正确（共 {len(lines)} 行）")
            print(f"\n  前3行预览：")
            for line in lines[:3]:
                print(f"    {line.strip()}")
            return True
        else:
            print(f"  ❌ 文件格式不正确")
            return False

    except Exception as e:
        print(f"  ❌ 验证失败: {e}")
        return False


def main():
    print("=" * 60)
    print("iCartoonFace Recognition 测试集标签文件下载工具")
    print("=" * 60)
    print()

    # 输出路径
    output_dir = Path("data")
    output_file = output_dir / "icartoonface_rectest_label.txt"

    # 检查是否已存在
    if output_file.exists():
        print(f"✓ 发现已存在的标签文件: {output_file}")
        print(f"  文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        print()

        response = input("是否重新下载？(y/n): ").strip().lower()
        if response != 'y':
            print("跳过下载。")

            # 验证现有文件
            print("\n验证文件格式...")
            if verify_label_file(output_file):
                print("\n✓ 现有文件有效，可以使用！")
                return
            else:
                print("\n⚠️  现有文件可能损坏，建议重新下载。")
                response = input("是否重新下载？(y/n): ").strip().lower()
                if response != 'y':
                    return

    print("\n尝试下载标签文件...")
    print()

    # 尝试不同的下载方法
    download_methods = [
        # 方法1：GitHub 备份（如果有人上传过）
        {
            "name": "GitHub 镜像",
            "url": "https://github.com/luxiangju-PersonAI/iCartoonFace/raw/master/rectest_label.txt",
        },
        # 方法2：尝试直接访问爱奇艺文件（通常需要认证）
        {
            "name": "爱奇艺直链",
            "url": "https://fft.cloud.iqiyi.com/api/download?fid=X6fgYZ",
        },
    ]

    success = False

    for i, method in enumerate(download_methods, 1):
        print(f"方法 {i}: {method['name']}")
        if download_file(method['url'], output_file):
            # 验证下载的文件
            print("\n验证文件格式...")
            if verify_label_file(output_file):
                success = True
                break
            else:
                print("  下载的文件格式不正确，删除并尝试其他方法...")
                output_file.unlink()
        print()

    if not success:
        print("=" * 60)
        print("⚠️  自动下载失败，请手动下载")
        print("=" * 60)
        print()
        print("请按以下步骤操作：")
        print()
        print("步骤1：在浏览器中打开以下链接")
        print("  https://fft.cloud.iqiyi.com/s/bUbdw5A")
        print()
        print("步骤2：输入密码")
        print("  X6fgYZ")
        print()
        print("步骤3：找到并下载")
        print("  文件名可能是：")
        print("  - rectest_label.txt")
        print("  - icartoonface_rectest_label.txt")
        print("  - personai_icartoonface_rectest_label.txt")
        print()
        print("步骤4：将下载的文件重命名并移动到：")
        print(f"  {output_file.absolute()}")
        print()
        print("提示：")
        print("  - 如果浏览器弹出 FFT Client 登录窗口，可以：")
        print("    1. 点击取消")
        print("    2. 右键文件 → '下载' 或 '另存为'")
        print("    3. 或者直接关闭登录窗口，文件会自动下载")
        print()
        print("替代方案：")
        print("  如果爱奇艺网盘无法访问，可以尝试：")
        print("  1. 使用 VPN 访问 Google Drive")
        print("  2. 寻找数据集的其他镜像源")
        print("  3. 联系数据集作者获取标签文件")
        print()

        sys.exit(1)

    print()
    print("=" * 60)
    print("✓ 下载完成！")
    print("=" * 60)
    print()
    print(f"标签文件位置: {output_file.absolute()}")
    print()
    print("下一步：")
    print("  运行以下命令处理测试集：")
    print()
    print("  PYTHONPATH=src python scripts/prepare_rectest_openset.py \\")
    print("    --label-file data/icartoonface_rectest_label.txt \\")
    print("    --image-dir data/personai_icartoonface_rectest/icartoonface_rectest \\")
    print("    --output-dir data/icartoonface_rectest_processed \\")
    print("    --crop \\")
    print("    --margin 0.1")
    print()


if __name__ == "__main__":
    main()
