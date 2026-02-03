@echo off
echo ========================================
echo 启动Chrome调试模式 - 银行理财数据采集
echo ========================================
echo.
echo 调试端口: 9222
echo.
echo 支持的银行:
echo   - 光大理财: https://www.cebwm.com/wealth/grlc/index.html
echo   - 浦发理财: https://www.spdb-wm.com/financialProducts/
echo   - 农银理财: https://www.abcwealth.com.cn/#/product
echo   - 中邮理财: https://www.psbc-wm.com/products/index.html
echo   - 工银理财: https://wm.icbc.com.cn/netWorthDisclosure
echo.
echo ========================================

:: 启动Chrome
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%TEMP%\chrome_debug_profile"

echo.
echo Chrome已启动！
echo.
echo 下一步:
echo 1. 在Chrome中打开上述银行理财网站
echo 2. 光大理财运行: python run_ceb_crawler.py
echo    其他银行运行: python -m crawlers.manual_crawler
echo.
pause
