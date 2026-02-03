# -*- coding: utf-8 -*-
"""
检查详情页加载情况
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time


def check():
    print("连接Chrome...")
    options = Options()
    options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    driver = webdriver.Chrome(options=options)

    print(f"当前URL: {driver.current_url}")
    print(f"页面标题: {driver.title}")

    # 检查当前窗口
    print(f"\n窗口数: {len(driver.window_handles)}")
    for i, handle in enumerate(driver.window_handles):
        driver.switch_to.window(handle)
        print(f"  窗口 {i}: {driver.current_url[:80]}")
        print(f"       标题: {driver.title}")

    # 切换到最后一个窗口（可能是详情页）
    if len(driver.window_handles) > 1:
        driver.switch_to.window(driver.window_handles[-1])
        print(f"\n切换到最后一个窗口...")

    print(f"\n当前URL: {driver.current_url}")
    print(f"页面长度: {len(driver.page_source)}")

    # 等待并检查
    print("\n等待5秒让页面加载...")
    time.sleep(5)

    print(f"页面长度: {len(driver.page_source)}")

    # 保存页面
    with open('D:/AI-FINANCE/cebwm_current_page.html', 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print("页面已保存到 cebwm_current_page.html")

    # 查找净值相关内容
    print("\n查找净值相关...")

    # 查找所有包含数字的表格行
    trs = driver.find_elements(By.TAG_NAME, 'tr')
    print(f"找到 {len(trs)} 个表格行")

    for i, tr in enumerate(trs[:30]):
        text = tr.text.strip()
        if text and ('2026' in text or '2025' in text or '净值' in text):
            print(f"  行 {i}: {text[:100]}")

    # 查找包含日期的div
    divs = driver.find_elements(By.TAG_NAME, 'div')
    print(f"\n找到 {len(divs)} 个div")

    for div in divs:
        text = div.text.strip()
        if '2026-01' in text or '2025-12' in text:
            print(f"  包含日期的div: {text[:200]}")
            break

    print("\n完成")


if __name__ == "__main__":
    check()
