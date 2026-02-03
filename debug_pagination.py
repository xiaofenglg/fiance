# -*- coding: utf-8 -*-
"""调试分页功能"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财列表页...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

for i in range(10):
    if len(driver.page_source) > 10000:
        break
    time.sleep(1)

print(f"页面加载完成: {len(driver.page_source)} bytes")

# 获取第一页产品
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

print("\n第1页产品:")
page1_products = driver.execute_script(js_get_products)
print(f"  {page1_products}")

# 检查分页函数
print("\n检查分页函数...")
js_check_pagination = """
return {
    goPage1_exists: typeof goPage1 === 'function',
    goPage2_exists: typeof goPage2 === 'function',
    currentPage: document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null,
    totalPage: document.querySelector('#totalpage1') ? document.querySelector('#totalpage1').innerText : null
};
"""
pagination_info = driver.execute_script(js_check_pagination)
print(f"  goPage1函数存在: {pagination_info['goPage1_exists']}")
print(f"  当前页: {pagination_info['currentPage']}")
print(f"  总页数: {pagination_info['totalPage']}")

# 尝试跳转到第2页
print("\n尝试跳转到第2页...")
print("  执行: goPage1(2)")
driver.execute_script("goPage1(2)")
time.sleep(5)

# 检查当前页
current_page = driver.execute_script("return document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null")
print(f"  执行后当前页: {current_page}")

page2_products = driver.execute_script(js_get_products)
print(f"  第2页产品: {page2_products}")

# 比较两页产品
if page1_products == page2_products:
    print("\n警告: 第1页和第2页产品相同! 分页可能没有生效")
else:
    print("\n分页正常工作")

# 尝试使用"下一页"功能
print("\n尝试使用'next'参数...")
driver.execute_script("goPage1('next')")
time.sleep(5)

current_page = driver.execute_script("return document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null")
print(f"  执行后当前页: {current_page}")

page3_products = driver.execute_script(js_get_products)
print(f"  第3页产品: {page3_products}")

# 尝试点击分页按钮
print("\n尝试点击分页输入框后回车...")
# 设置当前页为5然后触发回车
driver.execute_script("""
    var input = document.querySelector('#currentPage1');
    if (input) {
        input.value = '5';
        var event = new KeyboardEvent('keyup', {keyCode: 13, which: 13});
        document.dispatchEvent(event);
    }
""")
time.sleep(5)

current_page = driver.execute_script("return document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null")
print(f"  执行后当前页: {current_page}")

page5_products = driver.execute_script(js_get_products)
print(f"  第5页产品: {page5_products}")

print("\n完成")
driver.quit()
