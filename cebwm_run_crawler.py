# -*- coding: utf-8 -*-
"""
运行光大理财爬虫
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# 先导航到列表页
print("连接Chrome并导航到列表页...")
options = Options()
options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
driver = webdriver.Chrome(options=options)

print(f"当前URL: {driver.current_url}")

# 导航到列表页
driver.get("https://www.cebwm.com/wealth/grlc/index.html")
print("等待列表页加载...")
time.sleep(8)

print(f"当前URL: {driver.current_url}")
print(f"页面长度: {len(driver.page_source)}")

# 检查是否加载成功
if len(driver.page_source) > 5000:
    print("列表页加载成功！")

    # 运行爬虫
    from cebwm_final_crawler import CEBWMFinalCrawler

    crawler = CEBWMFinalCrawler()
    crawler.driver = driver
    crawler.main_window = driver.current_window_handle

    print("\n开始爬取...")
    products = crawler.crawl(max_products=3)

    if products:
        crawler.save_to_excel()
    else:
        print("未获取到数据")
else:
    print("列表页加载失败")
    print(f"页面内容: {driver.page_source[:500]}")
