# -*- coding: utf-8 -*-
"""调试详情页内容"""

import undetected_chromedriver as uc
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(15):
    size = len(driver.page_source)
    print(f"{i+1}秒: {size} bytes")
    if size > 10000:
        break
    time.sleep(1)

print("\n查找产品链接...")

# 先分析链接结构
js_analyze = """
var result = [];
var tables = document.querySelectorAll('table');
for (var t = 0; t < tables.length; t++) {
    var rows = tables[t].querySelectorAll('tbody tr, tr');
    for (var i = 0; i < rows.length && i < 3; i++) {
        var cells = rows[i].querySelectorAll('td');
        if (cells.length >= 2) {
            var code = cells[1] ? cells[1].innerText.trim() : '';
            if (code && code.startsWith('EB')) {
                var link = cells[0].querySelector('a');
                var detailBtn = rows[i].querySelector('button, .detail, [onclick*="detail"], a[href*="detail"]');
                var allLinks = rows[i].querySelectorAll('a');
                var allBtns = rows[i].querySelectorAll('button');
                result.push({
                    code: code,
                    link_href: link ? link.href : null,
                    link_onclick: link ? link.getAttribute('onclick') : null,
                    link_target: link ? link.target : null,
                    all_links_count: allLinks.length,
                    all_btns_count: allBtns.length,
                    row_html: rows[i].innerHTML.substring(0, 500)
                });
            }
        }
    }
}
return result;
"""

link_info = driver.execute_script(js_analyze)
print("\n链接分析:")
for item in link_info:
    print(f"  产品: {item['code']}")
    print(f"  href: {item['link_href']}")
    print(f"  onclick: {item['link_onclick']}")
    print(f"  target: {item['link_target']}")
    print(f"  链接数: {item['all_links_count']}, 按钮数: {item['all_btns_count']}")
    print(f"  行HTML: {item['row_html'][:300]}...")
    print()

# 找到第一个产品并点击
js_click = """
var tables = document.querySelectorAll('table');
for (var t = 0; t < tables.length; t++) {
    var rows = tables[t].querySelectorAll('tbody tr, tr');
    for (var i = 0; i < rows.length; i++) {
        var cells = rows[i].querySelectorAll('td');
        if (cells.length >= 2) {
            var code = cells[1] ? cells[1].innerText.trim() : '';
            if (code && code.startsWith('EB')) {
                var link = cells[0].querySelector('a');
                if (link) {
                    console.log("Found link for: " + code);
                    link.click();
                    return code;
                }
            }
        }
    }
}
return null;
"""

main_window = driver.current_window_handle
windows_before = len(driver.window_handles)

clicked_code = driver.execute_script(js_click)
print(f"点击了产品: {clicked_code}")

# 等待新窗口
time.sleep(3)

if len(driver.window_handles) > windows_before:
    print(f"新窗口已打开 (共{len(driver.window_handles)}个)")
    new_window = [h for h in driver.window_handles if h != main_window][-1]
    driver.switch_to.window(new_window)

    print("\n等待详情页加载...")
    for i in range(20):
        size = len(driver.page_source)
        print(f"{i+1}秒: {size} bytes")
        if size > 10000:
            break
        time.sleep(1)

    # 保存页面源码
    with open("D:/AI-FINANCE/detail_page_source.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print("\n页面源码已保存到 detail_page_source.html")

    # 打印页面标题和URL
    print(f"\n标题: {driver.title}")
    print(f"URL: {driver.current_url}")

    # 查找所有表格
    tables = driver.execute_script("return document.querySelectorAll('table').length")
    print(f"\n表格数量: {tables}")

    # 查找是否有净值表格内容
    body_text = driver.execute_script("return document.body.innerText")

    # 查找日期模式
    import re
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', body_text)
    if dates:
        print(f"\n找到日期: {dates[:10]}...")
    else:
        print("\n未找到日期格式数据")

    # 查找净值数据模式
    nav_pattern = re.findall(r'(\d{4}-\d{2}-\d{2})\s+([\d.]+)\s+([\d.]+)', body_text)
    if nav_pattern:
        print(f"\n找到净值数据: {nav_pattern[:5]}")
    else:
        print("\n未找到净值数据模式")

    # 打印部分页面文本
    print("\n页面文本片段:")
    print(body_text[:2000])

    driver.close()
    driver.switch_to.window(main_window)
else:
    print("未打开新窗口")

print("\n关闭浏览器...")
driver.quit()
