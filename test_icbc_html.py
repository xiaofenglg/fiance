# -*- coding: utf-8 -*-
"""检查工银理财页面HTML中的净值数据"""

import undetected_chromedriver as uc
import time
import re

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("\n打开净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")

print("等待页面加载...")
time.sleep(8)

# 获取页面HTML
html = driver.page_source

# 查找净值数据格式（如 1.0234, 0.9876）
nav_pattern = r'\d\.\d{4}'
navs = re.findall(nav_pattern, html)
print(f"\n页面中的净值格式数据 (X.XXXX): {len(navs)} 个")
for nav in navs[:20]:
    print(f"  {nav}")

# 查找product-item的完整内容
print("\n\n=== 检查product-item内容 ===")
js_get_items = """
var items = document.querySelectorAll('.product-item');
var result = [];
for (var i = 0; i < Math.min(items.length, 3); i++) {
    var item = items[i];
    result.push({
        html: item.innerHTML.substring(0, 1500),
        text: item.innerText.substring(0, 500),
        children: item.children.length
    });
}
return result;
"""
items = driver.execute_script(js_get_items)
print(f"找到 {len(items)} 个产品项")
for i, item in enumerate(items):
    print(f"\n产品 {i+1}:")
    print(f"  子元素数: {item['children']}")
    print(f"  文本内容: {item['text'][:300]}...")

# 检查是否有展开按钮或详情链接
print("\n\n=== 查找展开按钮或详情链接 ===")
js_find_buttons = """
var result = [];
var items = document.querySelectorAll('.product-item');
for (var i = 0; i < Math.min(items.length, 3); i++) {
    var item = items[i];
    // 查找按钮、链接、可点击元素
    var clickables = item.querySelectorAll('a, button, [onclick], [class*="btn"], [class*="more"], [class*="detail"]');
    for (var j = 0; j < clickables.length; j++) {
        var el = clickables[j];
        result.push({
            tag: el.tagName,
            classes: el.className,
            text: el.innerText.substring(0, 50),
            href: el.href || '',
            onclick: el.onclick ? 'has onclick' : ''
        });
    }
}
return result;
"""
buttons = driver.execute_script(js_find_buttons)
print(f"找到 {len(buttons)} 个可点击元素:")
for btn in buttons[:10]:
    print(f"  [{btn['tag']}] class='{btn['classes']}' text='{btn['text']}' href='{btn['href']}'")

# 尝试点击产品标题查看是否有展开效果
print("\n\n=== 点击产品标题 ===")
js_click_title = """
var title = document.querySelector('.product-item .p-title');
if (title) {
    // 获取点击前的兄弟元素
    var parent = title.parentElement;
    var before = parent.innerHTML.length;
    title.click();
    return {clicked: true, before: before};
}
return {clicked: false};
"""
result = driver.execute_script(js_click_title)
print(f"点击结果: {result}")

time.sleep(2)

# 检查是否有新内容
js_check_new = """
var title = document.querySelector('.product-item .p-title');
if (title) {
    var parent = title.parentElement;
    return {after: parent.innerHTML.length, text: parent.innerText.substring(0, 1000)};
}
return {};
"""
after = driver.execute_script(js_check_new)
if after:
    print(f"点击后长度: {after.get('after')}")
    print(f"内容: {after.get('text', '')[:500]}...")

# 检查页面中是否有其他数据来源
print("\n\n=== 检查页面数据源 ===")
js_check_data = """
var result = {};
// 检查是否有全局数据对象
if (window.__NUXT__) result.nuxt = Object.keys(window.__NUXT__);
if (window.__INITIAL_STATE__) result.initial = Object.keys(window.__INITIAL_STATE__);
if (window.productData) result.productData = 'exists';
if (window.navData) result.navData = 'exists';

// 检查script标签中的JSON数据
var scripts = document.querySelectorAll('script:not([src])');
for (var i = 0; i < scripts.length; i++) {
    var content = scripts[i].innerText;
    if (content.includes('product') || content.includes('networth')) {
        result['script_' + i] = content.substring(0, 200);
    }
}
return result;
"""
data_sources = driver.execute_script(js_check_data)
print(f"数据源: {data_sources}")

print("\n完成")
driver.quit()
