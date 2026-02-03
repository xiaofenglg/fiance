#!/usr/bin/env python3
"""
自动恢复任务脚本

用途：
    - 电脑重启或待机后自动恢复未完成的爬虫任务
    - 可以添加到 Windows 启动项或任务计划程序中

使用方法：
    1. 直接运行: python auto_resume_task.py
    2. 添加到 Windows 启动项:
       - 按 Win+R，输入 shell:startup
       - 创建此脚本的快捷方式放入启动文件夹
    3. 使用任务计划程序:
       - 创建"计算机启动时"触发的任务
       - 操作设置为运行此脚本

日期：2026-01-17
"""

import os
import sys
import time
from datetime import datetime

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*60)
    print("  银行理财爬虫 - 自动恢复任务")
    print("="*60)
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查任务状态文件是否存在
    state_file = "task_state.json"
    if not os.path.exists(state_file):
        print("没有未完成的任务，退出")
        return

    # 读取任务状态
    import json
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except Exception as e:
        print(f"读取任务状态失败: {e}")
        return

    # 检查是否有未完成的任务
    if state.get('status') not in ['running', 'paused']:
        print("没有未完成的任务，退出")
        return

    print(f"检测到未完成的任务: {state.get('task_id')}")
    print(f"已完成银行: {state.get('banks_completed', [])}")
    print(f"待处理银行: {state.get('banks_pending', [])}")
    print()

    # 等待网络连接（可选，给系统启动一些缓冲时间）
    print("等待系统就绪...")
    time.sleep(10)

    # 导入并执行自动恢复
    try:
        from bank_crawler import auto_resume
        print("\n开始恢复任务...")
        auto_resume()
    except Exception as e:
        print(f"恢复任务失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n任务完成，窗口将在30秒后关闭...")
    time.sleep(30)


if __name__ == "__main__":
    main()
