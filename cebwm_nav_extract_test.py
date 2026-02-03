# -*- coding: utf-8 -*-
"""
测试从详情页提取净值数据
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import re


def test():
    print("连接Chrome...")
    options = Options()
    options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    driver = webdriver.Chrome(options=options)

    print(f"当前URL: {driver.current_url}")

    # 如果当前是详情页，直接分析
    if 'lcxx/lccp' in driver.current_url:
        print("\n在详情页，开始分析...")
    else:
        # 导航到一个详情页
        print("\n导航到详情页...")
        driver.get("https://www.cebwm.com/wealth/lcxx/lccp14/127554501/index.html")
        time.sleep(5)

    print(f"页面标题: {driver.title}")

    # 滚动到页面底部
    print("\n滚动到底部...")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # 查找所有表格
    print("\n=== 查找表格 ===")
    tables = driver.find_elements(By.TAG_NAME, 'table')
    print(f"找到 {len(tables)} 个表格")

    for i, table in enumerate(tables):
        rows = table.find_elements(By.TAG_NAME, 'tr')
        print(f"\n表格 {i} ({len(rows)} 行):")

        # 打印前25行
        for j, row in enumerate(rows[:25]):
            cells = row.find_elements(By.TAG_NAME, 'td')
            if not cells:
                cells = row.find_elements(By.TAG_NAME, 'th')
            texts = [c.text.strip() for c in cells]
            if any(texts):
                print(f"  {j}: {texts}")

    # 直接查找包含日期的元素
    print("\n\n=== 查找日期元素 ===")
    date_els = driver.find_elements(By.XPATH, "//*[contains(text(), '2026-') or contains(text(), '2025-')]")
    print(f"找到 {len(date_els)} 个日期元素")

    for el in date_els[:30]:
        try:
            text = el.text.strip()
            parent = el.find_element(By.XPATH, "./..")
            parent_text = parent.text.strip()[:100]
            print(f"  元素: {text[:50]}")
            print(f"  父元素: {parent_text}")
            print()
        except:
            pass

    # 尝试获取页面源码中的净值数据
    print("\n=== 搜索页面源码 ===")
    page_source = driver.page_source

    # 查找日期格式的数据
    date_pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d+\.\d+)\s+(\d+\.\d+)'
    matches = re.findall(date_pattern, page_source)
    print(f"找到 {len(matches)} 个日期净值匹配")

    for match in matches[:20]:
        print(f"  日期: {match[0]}, 单位净值: {match[1]}, 累计净值: {match[2]}")

    # 查找div或其他容器
    print("\n\n=== 查找净值容器 ===")
    containers = driver.find_elements(By.XPATH, "//*[contains(text(), '日期') and contains(text(), '净值')]")
    for c in containers:
        print(f"容器: {c.tag_name}")
        print(f"文本: {c.text[:500]}")

    print("\n完成")


if __name__ == "__main__":
    test()
