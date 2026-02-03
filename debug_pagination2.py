# -*- coding: utf-8 -*-
"""调试分页功能 - 更多等待时间"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财列表页...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

for i in range(15):
    if len(driver.page_source) > 10000:
        break
    time.sleep(1)

print(f"页面加载完成")

js_get_products = """
var products = [];
var rows = document.querySelectorAll('table tbody tr');
for (var i = 0; i < rows.length; i++) {
    var cells = rows[i].querySelectorAll('td');
    if (cells.length >= 2) {
        var code = cells[1] ? cells[1].innerText.trim() : '';
        if (code && code.startsWith('EB')) {
            products.push(code);
        }
    }
}
return products;
"""

def wait_for_products(timeout=15):
    """等待产品列表加载"""
    for i in range(timeout):
        products = driver.execute_script(js_get_products)
        if products:
            return products
        time.sleep(1)
    return []

print("\n第1页产品:")
page1 = wait_for_products()
print(f"  {page1}")

# 测试连续翻页
for page_num in [2, 3, 4, 5]:
    print(f"\n跳转到第{page_num}页...")

    # 执行分页
    driver.execute_script(f"goPage1({page_num})")

    # 等待页面更新
    time.sleep(3)

    # 检查当前页码
    current = driver.execute_script("return document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null")
    print(f"  当前页码显示: {current}")

    # 等待产品加载
    products = wait_for_products(10)
    print(f"  产品数量: {len(products)}")
    if products:
        print(f"  产品列表: {products}")
    else:
        # 检查表格状态
        table_html = driver.execute_script("return document.querySelector('table tbody') ? document.querySelector('table tbody').innerHTML.substring(0, 500) : 'no tbody'")
        print(f"  表格内容: {table_html[:200]}...")

    # 保存页面源码大小
    print(f"  页面大小: {len(driver.page_source)} bytes")

print("\n完成")
driver.quit()
