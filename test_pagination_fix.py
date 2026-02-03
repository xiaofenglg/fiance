# -*- coding: utf-8 -*-
"""测试分页修复 - 验证EB和EW产品都能获取"""

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

# 更新后的产品提取代码（支持EB和EW）
js_get_products = """
var products = [];
var tables = document.querySelectorAll('table');
for (var t = 0; t < tables.length; t++) {
    var rows = tables[t].querySelectorAll('tbody tr, tr');
    for (var i = 0; i < rows.length; i++) {
        var cells = rows[i].querySelectorAll('td');
        if (cells.length >= 10) {
            var code = cells[1] ? cells[1].innerText.trim() : '';
            if (code && (code.startsWith('EB') || code.startsWith('EW'))) {
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

# 测试多页
all_products = []
pages_to_test = [1, 2, 3, 5, 10]  # 测试几个不同的页面

for page_num in pages_to_test:
    if page_num > 1:
        print(f"\n跳转到第{page_num}页...")
        driver.execute_script(f"goPage1({page_num})")
        time.sleep(4)

    current = get_current_page()
    products = get_products()

    print(f"第{page_num}页 (显示页码: {current})")
    print(f"  产品数: {len(products)}")
    if products:
        # 分类统计
        eb_count = len([p for p in products if p.startswith('EB')])
        ew_count = len([p for p in products if p.startswith('EW')])
        print(f"  EB产品: {eb_count}, EW产品: {ew_count}")
        print(f"  产品列表: {products}")
        all_products.extend(products)
    else:
        print("  无产品!")

print(f"\n总计: {len(all_products)} 个产品")
print(f"  EB产品: {len([p for p in all_products if p.startswith('EB')])}")
print(f"  EW产品: {len([p for p in all_products if p.startswith('EW')])}")

# 检查是否有重复
unique_products = set(all_products)
if len(unique_products) < len(all_products):
    print(f"警告: 有重复产品! 唯一产品数: {len(unique_products)}")
else:
    print("分页正常: 各页产品不重复")

print("\n完成")
driver.quit()
