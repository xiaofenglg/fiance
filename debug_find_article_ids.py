# -*- coding: utf-8 -*-
"""查找产品和文章ID的映射"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import re
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(10):
    size = len(driver.page_source)
    if size > 10000:
        break
    time.sleep(1)

print(f"页面加载完成: {len(driver.page_source)} bytes")

# 保存列表页源码
with open("D:/AI-FINANCE/list_page_source.html", "w", encoding="utf-8") as f:
    f.write(driver.page_source)
print("列表页源码已保存")

# 方法1: 查找页面中的所有数字ID
page_source = driver.page_source
article_ids = re.findall(r'\b(20\d{7})\b', page_source)
print(f"\n找到可能的文章ID: {set(article_ids)}")

# 方法2: 查找JavaScript中的数据
js_find_data = """
var result = {};

// 检查全局变量
result.globals = [];
for (var key in window) {
    if (key.toLowerCase().includes('product') ||
        key.toLowerCase().includes('article') ||
        key.toLowerCase().includes('data') ||
        key.toLowerCase().includes('list')) {
        try {
            var val = window[key];
            if (typeof val === 'object' && val !== null) {
                result.globals.push({key: key, type: typeof val, preview: JSON.stringify(val).substring(0, 200)});
            }
        } catch(e) {}
    }
}

// 检查数据属性
result.data_attrs = [];
var elements = document.querySelectorAll('[data-article], [data-id], [data-key]');
elements.forEach(function(el) {
    result.data_attrs.push({
        tag: el.tagName,
        data: el.dataset,
        text: el.innerText.substring(0, 50)
    });
});

// 检查表格行的所有属性
result.row_attrs = [];
var rows = document.querySelectorAll('table tbody tr');
for (var i = 0; i < rows.length && i < 3; i++) {
    var row = rows[i];
    result.row_attrs.push({
        attrs: Array.from(row.attributes).map(a => ({name: a.name, value: a.value})),
        cells: Array.from(row.querySelectorAll('td')).map(c => ({
            text: c.innerText.substring(0, 30),
            attrs: Array.from(c.attributes).map(a => ({name: a.name, value: a.value}))
        }))
    });
}

return result;
"""

data = driver.execute_script(js_find_data)
print("\n相关全局变量:")
for g in data.get('globals', [])[:10]:
    print(f"  {g['key']}: {g['preview'][:100]}...")

print("\n数据属性元素:")
for d in data.get('data_attrs', [])[:5]:
    print(f"  {d}")

print("\n表格行属性:")
for r in data.get('row_attrs', []):
    print(f"  attrs: {r['attrs']}")
    for c in r['cells'][:3]:
        print(f"    cell: {c['text']} - {c['attrs']}")

# 方法3: 监听网络请求 (检查XHR)
print("\n分析JavaScript函数...")
js_find_functions = """
// 查找可能处理点击的函数
var result = [];
var scripts = document.querySelectorAll('script');
scripts.forEach(function(s) {
    var code = s.innerHTML;
    // 查找包含 window.open 或 article 的代码
    if (code.includes('window.open') || code.includes('article') ||
        code.includes('lccp') || code.includes('href')) {
        result.push(code.substring(0, 500));
    }
});
return result;
"""

scripts = driver.execute_script(js_find_functions)
for i, s in enumerate(scripts[:5]):
    print(f"\n脚本 {i+1}: {s[:200]}...")

# 方法4: 通过网络请求日志查找
print("\n启用网络日志...")
driver.execute_script("""
window._networkLog = [];
var origXHR = window.XMLHttpRequest.prototype.open;
window.XMLHttpRequest.prototype.open = function() {
    window._networkLog.push(arguments[1]);
    return origXHR.apply(this, arguments);
};

var origFetch = window.fetch;
window.fetch = function(url) {
    window._networkLog.push(url);
    return origFetch.apply(this, arguments);
};
""")

# 点击产品链接触发请求
print("点击产品触发请求...")
js_click = """
var link = document.querySelector('a.lb_title');
if (link) link.click();
return true;
"""
driver.execute_script(js_click)
time.sleep(2)

# 获取网络日志
network_log = driver.execute_script("return window._networkLog")
print(f"网络请求: {network_log}")

# 检查新窗口
if len(driver.window_handles) > 1:
    print(f"\n打开了新窗口: {len(driver.window_handles)}")
    for h in driver.window_handles:
        driver.switch_to.window(h)
        print(f"  窗口URL: {driver.current_url}")

print("\n完成")
driver.quit()
