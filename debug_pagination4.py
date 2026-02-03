# -*- coding: utf-8 -*-
"""调试分页 - 直接跳转到第3页"""

import undetected_chromedriver as uc
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
var table = document.querySelector('#finance_tb1');
if (table) {
    var rows = table.querySelectorAll('tbody tr, tr');
    for (var i = 0; i < rows.length; i++) {
        var cells = rows[i].querySelectorAll('td');
        if (cells.length >= 10) {
            var code = cells[1] ? cells[1].innerText.trim() : '';
            if (code && code.startsWith('EB')) {
                products.push(code);
            }
        }
    }
}
return products;
"""

def get_products():
    return driver.execute_script(js_get_products)

def get_current_page():
    return driver.execute_script("return document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null")

print("\n第1页产品:")
print(f"  页码: {get_current_page()}")
print(f"  产品: {get_products()}")

# 直接跳转到第3页（不经过第2页）
print("\n直接跳转到第3页...")
driver.execute_script("goPage1(3)")
time.sleep(5)

print(f"  页码: {get_current_page()}")
products = get_products()
print(f"  产品数: {len(products)}")
if products:
    print(f"  产品: {products}")
else:
    # 检查表格内容
    table_text = driver.execute_script("return document.querySelector('#finance_tb1') ? document.querySelector('#finance_tb1').innerText.substring(0, 500) : 'no table'")
    print(f"  表格内容: {table_text}")

# 重新加载页面，然后跳转到第50页
print("\n重新加载页面...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")
for i in range(15):
    if len(driver.page_source) > 10000:
        break
    time.sleep(1)

print(f"  第1页产品: {get_products()}")

print("\n直接跳转到第50页...")
driver.execute_script("goPage1(50)")
time.sleep(5)

print(f"  页码: {get_current_page()}")
products = get_products()
print(f"  产品数: {len(products)}")
if products:
    print(f"  产品: {products}")

# 尝试使用input value + keypress
print("\n尝试通过输入框跳转...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")
for i in range(15):
    if len(driver.page_source) > 10000:
        break
    time.sleep(1)

print(f"  第1页产品: {get_products()}")

# 使用Selenium直接操作输入框
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

page_input = driver.find_element(By.ID, "currentPage1")
page_input.clear()
page_input.send_keys("10")
page_input.send_keys(Keys.RETURN)
time.sleep(5)

print(f"  页码: {get_current_page()}")
products = get_products()
print(f"  产品数: {len(products)}")
if products:
    print(f"  产品: {products}")

print("\n完成")
driver.quit()
