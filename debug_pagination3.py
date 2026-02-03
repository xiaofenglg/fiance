# -*- coding: utf-8 -*-
"""调试分页 - 更精确的表格选择器"""

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

# 更精确的产品提取 - 查找包含EB代码的表格
js_get_products = """
var products = [];
// 查找所有表格
var tables = document.querySelectorAll('table');
for (var t = 0; t < tables.length; t++) {
    var rows = tables[t].querySelectorAll('tbody tr, tr');
    for (var i = 0; i < rows.length; i++) {
        var cells = rows[i].querySelectorAll('td');
        // 产品表格应该有10+列，第二列是产品代码
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

# 检查页面上所有表格的结构
print("\n分析页面表格结构...")
js_analyze_tables = """
var result = [];
var tables = document.querySelectorAll('table');
for (var t = 0; t < tables.length; t++) {
    var rows = tables[t].querySelectorAll('tr');
    var firstRowCells = rows[0] ? rows[0].querySelectorAll('td, th').length : 0;
    var hasEB = tables[t].innerHTML.includes('EB10') || tables[t].innerHTML.includes('EB1');
    result.push({
        index: t,
        rows: rows.length,
        firstRowCells: firstRowCells,
        hasEB: hasEB,
        className: tables[t].className,
        id: tables[t].id
    });
}
return result;
"""

tables_info = driver.execute_script(js_analyze_tables)
print("表格列表:")
for t in tables_info:
    print(f"  [{t['index']}] rows={t['rows']}, cols={t['firstRowCells']}, hasEB={t['hasEB']}, class='{t['className']}', id='{t['id']}'")

# 找到产品表格的选择器
js_find_product_table = """
var tables = document.querySelectorAll('table');
for (var t = 0; t < tables.length; t++) {
    var rows = tables[t].querySelectorAll('tr');
    for (var i = 0; i < rows.length; i++) {
        var cells = rows[i].querySelectorAll('td');
        if (cells.length >= 10) {
            var code = cells[1] ? cells[1].innerText.trim() : '';
            if (code && code.startsWith('EB')) {
                // 找到了产品表格
                return {
                    tableIndex: t,
                    tableClass: tables[t].className,
                    tableId: tables[t].id,
                    parentId: tables[t].parentElement ? tables[t].parentElement.id : null
                };
            }
        }
    }
}
return null;
"""

product_table = driver.execute_script(js_find_product_table)
print(f"\n产品表格: {product_table}")

def get_products():
    return driver.execute_script(js_get_products)

print("\n第1页产品:")
page1 = get_products()
print(f"  {page1}")

# 翻页测试
for page_num in [2, 3, 4, 5]:
    print(f"\n跳转到第{page_num}页...")

    driver.execute_script(f"goPage1({page_num})")
    time.sleep(3)

    # 等待更长时间
    for _ in range(10):
        products = get_products()
        if products:
            break
        time.sleep(1)

    current = driver.execute_script("return document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null")
    print(f"  页码: {current}, 产品数: {len(products)}")

    if products:
        print(f"  产品: {products}")
    else:
        # 重新分析表格
        tables_info = driver.execute_script(js_analyze_tables)
        print(f"  表格状态:")
        for t in tables_info[:5]:
            print(f"    [{t['index']}] rows={t['rows']}, hasEB={t['hasEB']}")

print("\n完成")
driver.quit()
