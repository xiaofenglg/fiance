@echo off
REM 银行理财爬虫 - 自动恢复任务脚本
REM
REM 用途：电脑重启后自动恢复未完成的爬虫任务
REM
REM 设置方法（任选其一）：
REM   1. 添加到启动文件夹: 按 Win+R，输入 shell:startup，将此文件的快捷方式放入
REM   2. 任务计划程序: 创建"计算机启动时"触发的任务

cd /d "%~dp0"

echo ============================================================
echo   银行理财爬虫 - 自动恢复任务
echo ============================================================
echo.

REM 检查任务状态文件
if not exist "task_state.json" (
    echo 没有未完成的任务，退出
    timeout /t 5
    exit /b 0
)

REM 等待系统启动完成
echo 等待系统就绪...
timeout /t 15

REM 执行恢复
echo 开始检查并恢复任务...
python bank_crawler.py --resume

echo.
echo 任务执行完成
pause
