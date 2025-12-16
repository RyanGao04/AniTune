#!/usr/bin/env python3
"""查看 wandb 训练时间的工具脚本"""

import json
from pathlib import Path
from datetime import datetime, timedelta

def parse_wandb_time(run_dir):
    """从 wandb run 目录中解析训练时间"""
    run_dir = Path(run_dir)
    
    # 读取 metadata
    metadata_file = run_dir / "files" / "wandb-metadata.json"
    if not metadata_file.exists():
        print(f"❌ 找不到 {metadata_file}")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    start_time_str = metadata.get("startedAt")
    if not start_time_str:
        print("❌ 找不到开始时间")
        return
    
    # 解析开始时间
    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
    
    # 获取日志最后修改时间作为结束时间的近似
    output_log = run_dir / "files" / "output.log"
    if output_log.exists():
        import os
        end_timestamp = os.path.getmtime(output_log)
        end_time = datetime.fromtimestamp(end_timestamp, tz=start_time.tzinfo)
    else:
        end_time = None
    
    # 读取输出日志，统计完成的 epoch
    completed_epochs = 0
    if output_log.exists():
        with open(output_log) as f:
            for line in f:
                if line.startswith("Epoch") and "/" in line:
                    # 提取 Epoch X/Y
                    parts = line.split("|")[0].strip()
                    if "/" in parts:
                        epoch_info = parts.split()[1]
                        current_epoch = int(epoch_info.split("/")[0])
                        completed_epochs = max(completed_epochs, current_epoch)
    
    # 从进度条估算每个 epoch 时间
    # 从日志看到: [05:27<01:17, 6.11it/s] 表示 1980/2452 iterations
    # 总时间约 5:27 + 1:17 = 6:44 per epoch
    estimated_time_per_epoch = timedelta(minutes=6, seconds=44)
    
    print("=" * 70)
    print("📊 训练时间分析")
    print("=" * 70)
    print(f"\n🕐 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if end_time:
        print(f"🕑 最后更新: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        duration = end_time - start_time
        print(f"⏱️  总时长:   {duration}")
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"            ({int(hours)} 小时 {int(minutes)} 分钟 {int(seconds)} 秒)")
    
    print(f"\n📈 已完成 Epoch: {completed_epochs}")
    
    if completed_epochs > 0 and end_time:
        actual_time_per_epoch = duration / completed_epochs
        print(f"⏱️  平均每个 Epoch: {actual_time_per_epoch}")
        mins = int(actual_time_per_epoch.total_seconds() / 60)
        secs = int(actual_time_per_epoch.total_seconds() % 60)
        print(f"                    ({mins} 分钟 {secs} 秒)")
        
        # 估算剩余时间
        total_epochs = 12  # 从配置中读取
        remaining_epochs = total_epochs - completed_epochs
        if remaining_epochs > 0:
            estimated_remaining = actual_time_per_epoch * remaining_epochs
            print(f"\n⏳ 剩余 Epoch: {remaining_epochs}")
            print(f"⏱️  估计剩余时间: {estimated_remaining}")
            hours, remainder = divmod(estimated_remaining.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            print(f"                ({int(hours)} 小时 {int(minutes)} 分钟)")
    
    # GPU 信息
    gpu_info = metadata.get("gpu_nvidia", [])
    if gpu_info:
        gpu = gpu_info[0]
        print(f"\n🖥️  GPU: {gpu['name']}")
        print(f"💾 显存: {int(gpu['memoryTotal']) / (1024**3):.1f} GB")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # 使用最新的 run
        run_dir = "wandb/latest-run"
    
    parse_wandb_time(run_dir)

