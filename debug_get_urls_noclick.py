# -*- coding: utf-8 -*-
"""不点击任何链接获取产品URL"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import json
import re

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财列表页...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(10):
    size = len(driver.page_source)
    if size > 10000:
        break
    time.sleep(1)

print(f"列表页加载完成: {len(driver.page_source)} bytes")

# 方法1: 从页面源码中直接提取产品URL
print("\n方法1: 从源码提取URL...")
page_source = driver.page_source

# 查找类似 /wealth/lcxx/lccp14/{id}/index.html 的模式
# 从JS代码看，HTMLURL会被拼接到页面中
urls = re.findall(r'/wealth/lcxx/lccp\d+/\d+/index\.html', page_source)
print(f"找到 {len(urls)} 个URL模式")
for url in list(set(urls))[:5]:
    print(f"  {url}")

# 方法2: 监听页面AJAX请求
print("\n方法2: 获取AJAX数据...")
js_get_data = """
// 页面可能已经有数据在某个变量中
// 查找包含产品列表的数据结构

// 尝试从表格行中提取data属性
var result = [];
var rows = document.querySelectorAll('table tbody tr');
for (var i = 0; i < rows.length; i++) {
    var cells = rows[i].querySelectorAll('td');
    if (cells.length >= 10) {
        var code = cells[1] ? cells[1].innerText.trim() : '';
        if (code && code.startsWith('EB')) {
            // 从链接的data-analytics-click属性提取
            var detailLink = cells[10] ? cells[10].querySelector('a') : null;
            var nameLink = cells[0] ? cells[0].querySelector('a') : null;

            result.push({
                code: code,
                name: cells[0] ? cells[0].innerText.trim() : '',
                analytics: detailLink ? detailLink.getAttribute('data-analytics-click') : null
            });
        }
    }
}
return result;
"""

products = driver.execute_script(js_get_data)
print(f"从表格获取 {len(products)} 个产品")
for p in products[:3]:
    print(f"  {p['code']}: {p['name']}")

# 方法3: 找到并调用页面的数据加载函数
print("\n方法3: 调用页面数据加载函数...")

# 查找当前页数据中是否有URL
js_find_url = """
// 检查全局变量
var possibleData = [];

// 查找window下可能存储数据的变量
for (var key in window) {
    try {
        var val = window[key];
        if (val && typeof val === 'object') {
            var str = JSON.stringify(val);
            if (str && str.includes('HTMLURL') || str.includes('/wealth/lcxx/')) {
                possibleData.push({
                    key: key,
                    preview: str.substring(0, 500)
                });
            }
        }
    } catch(e) {}
}

return possibleData;
"""

possible_data = driver.execute_script(js_find_url)
if possible_data:
    print(f"找到可能包含URL的数据:")
    for d in possible_data[:5]:
        print(f"  {d['key']}: {d['preview'][:200]}...")
else:
    print("未找到包含URL的全局变量")

# 方法4: 修改window.open来捕获URL但不实际打开窗口
print("\n方法4: 捕获window.open调用...")
js_patch = """
window._capturedUrls = [];
var origOpen = window.open;
window.open = function(url, target, features) {
    window._capturedUrls.push({url: url, target: target});
    return {closed: true, close: function(){}, focus: function(){}};  // 假窗口
};
return true;
"""
driver.execute_script(js_patch)

# 现在点击一个链接，但它不会真正打开窗口
print("点击第一个产品（不会打开真实窗口）...")
link = driver.find_element(By.CSS_SELECTOR, "a.lb_title")
link.click()
time.sleep(1)

captured = driver.execute_script("return window._capturedUrls")
print(f"捕获的URL: {captured}")

if captured:
    # 我们捕获到URL了，但没有打开真正的窗口
    # 检查这样是否会影响后续导航
    detail_url = captured[0]['url'].split('?')[0] if captured else None

    if detail_url:
        print(f"\n测试直接导航到: {detail_url}")
        driver.execute_script(f"window.location.href = '{detail_url}'")

        for i in range(15):
            size = len(driver.page_source)
            print(f"{i+1}秒: {size} bytes")
            if size > 10000:
                break
            time.sleep(1)

        if len(driver.page_source) > 10000:
            print("\n成功！使用window.open补丁可以获取URL而不触发WAF")

            # 提取净值
            try:
                nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
                nav_tab.click()
                time.sleep(2)

                iframe = driver.find_element(By.ID, "fundValueframe")
                driver.switch_to.frame(iframe)
                time.sleep(2)

                js_extract = """
                var result = [];
                var tables = document.querySelectorAll('table');
                for (var t = 0; t < tables.length; t++) {
                    var rows = tables[t].querySelectorAll('tr');
                    for (var i = 0; i < rows.length; i++) {
                        var cells = rows[i].querySelectorAll('td');
                        if (cells.length >= 2) {
                            var text0 = cells[0] ? cells[0].innerText.trim() : '';
                            var text1 = cells[1] ? cells[1].innerText.trim() : '';
                            var text2 = cells[2] ? cells[2].innerText.trim() : '';
                            if (/\\d{4}-\\d{2}-\\d{2}/.test(text0)) {
                                result.push({date: text0, unit: text1, total: text2});
                            }
                        }
                    }
                }
                return result;
                """
                nav_data = driver.execute_script(js_extract)
                print(f"\n净值数据 ({len(nav_data)} 条):")
                for d in nav_data[:5]:
                    print(f"  {d['date']}: {d['unit']}")

                driver.switch_to.default_content()
            except Exception as e:
                print(f"净值提取失败: {e}")

        else:
            print("导航仍然失败 - 可能点击本身就触发了WAF")

print("\n完成")
driver.quit()
