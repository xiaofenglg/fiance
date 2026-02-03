# -*- coding: utf-8 -*-
"""
分析光大理财详情页结构
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json


def analyze():
    print("连接Chrome...")

    options = Options()
    options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    driver = webdriver.Chrome(options=options)

    # 获取当前页面信息
    print(f"\n当前URL: {driver.current_url}")
    print(f"页面标题: {driver.title}")

    # 查找净值表格
    print("\n=== 查找净值数据 ===")

    # 查找所有表格
    tables = driver.find_elements(By.TAG_NAME, 'table')
    print(f"页面有 {len(tables)} 个表格")

    for i, table in enumerate(tables):
        rows = table.find_elements(By.TAG_NAME, 'tr')
        if len(rows) >= 5:
            # 检查是否包含净值数据
            table_text = table.text
            if '净值' in table_text or '2026-' in table_text or '2025-' in table_text:
                print(f"\n表格 {i} ({len(rows)} 行) - 可能是净值表格:")
                for j, row in enumerate(rows[:25]):
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if not cells:
                        cells = row.find_elements(By.TAG_NAME, 'th')
                    texts = [c.text.strip() for c in cells]
                    if any(texts):
                        print(f"  {texts}")

    # 保存详情页HTML
    with open('D:/AI-FINANCE/cebwm_detail_actual.html', 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print("\n详情页HTML已保存到 cebwm_detail_actual.html")

    # 分析URL模式
    url = driver.current_url
    print(f"\n详情页URL: {url}")

    # 提取URL中的参数
    if '?' in url:
        params = url.split('?')[1]
        print(f"URL参数: {params}")

    print("\n完成")
    return driver


if __name__ == "__main__":
    analyze()
