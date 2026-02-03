# -*- coding: utf-8 -*-
"""提取产品详情URL"""

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

# 从页面HTML中查找链接的href属性（可能在data-*属性中）
print("\n分析产品链接...")
js_analyze = """
var result = [];
var rows = document.querySelectorAll('table tbody tr');
for (var i = 0; i < rows.length && result.length < 5; i++) {
    var cells = rows[i].querySelectorAll('td');
    if (cells.length >= 10) {
        var codeCell = cells[1];
        var code = codeCell ? codeCell.innerText.trim() : '';
        if (code && code.startsWith('EB')) {
            var nameLink = cells[0].querySelector('a.lb_title');
            var detailLink = cells[10] ? cells[10].querySelector('a') : null;

            result.push({
                code: code,
                name: nameLink ? nameLink.innerText.trim() : '',
                name_link_html: nameLink ? nameLink.outerHTML : '',
                detail_link_html: detailLink ? detailLink.outerHTML : '',
                // 检查所有可能的数据属性
                name_data_href: nameLink ? nameLink.getAttribute('data-href') : null,
                detail_data_href: detailLink ? detailLink.getAttribute('data-href') : null,
                row_attrs: Array.from(rows[i].attributes).map(a => ({name: a.name, value: a.value}))
            });
        }
    }
}
return result;
"""

products = driver.execute_script(js_analyze)
for p in products:
    print(f"\n{p['code']}: {p['name']}")
    print(f"  名称链接: {p['name_link_html'][:200]}...")
    print(f"  详情链接: {p['detail_link_html'][:200] if p['detail_link_html'] else 'None'}...")

# 尝试找到页面内所有隐藏的URL或数据
print("\n\n查找页面中的URL模式...")
page_source = driver.page_source

# 查找所有类似detail页面的URL
detail_urls = re.findall(r'/wealth/lcxx/[^"\'>\s]+', page_source)
print(f"找到 {len(detail_urls)} 个URL模式:")
for url in list(set(detail_urls))[:10]:
    print(f"  {url}")

# 查找可能存储在JavaScript中的数据
print("\n查找JavaScript数据...")
js_data = """
// 尝试找到所有全局数据
var result = {};

// 查找可能包含产品数据的数组
if (window.resultList) result.resultList = window.resultList;
if (window.productList) result.productList = window.productList;

// 检查jQuery data
result.jqueryData = [];
$('table tbody tr').each(function(i) {
    if (i < 3) {
        var data = $(this).data();
        if (Object.keys(data).length > 0) {
            result.jqueryData.push(data);
        }
    }
});

// 查找链接上可能存储的onclick数据
result.linkOnclick = [];
$('a.lb_title').each(function(i) {
    if (i < 3) {
        var elem = this;
        // 获取所有事件处理器
        var events = $._data(elem, 'events');
        if (events && events.click) {
            result.linkOnclick.push({
                index: i,
                text: $(elem).text(),
                clickHandlers: events.click.length
            });
        }
    }
});

return result;
"""

try:
    data = driver.execute_script(js_data)
    print(f"jQuery数据: {json.dumps(data, indent=2, default=str)[:500]}...")
except Exception as e:
    print(f"获取数据失败: {e}")

# 直接导航到一个产品详情页测试
print("\n\n尝试构造URL直接访问...")
# 从保存的detail页面知道URL格式是 /wealth/lcxx/lccp14/{id}/index.html
# 尝试从页面找到某种ID映射

# 方法: 监听click事件获取目标URL
print("设置click事件监听...")
driver.execute_script("""
window._clickedUrls = [];
document.addEventListener('click', function(e) {
    var target = e.target;
    if (target.tagName === 'A') {
        window._clickedUrls.push({
            href: target.href,
            text: target.innerText
        });
    }
}, true);

// 同时监听window.open
var origOpen = window.open;
window.open = function(url, target, features) {
    window._clickedUrls.push({
        windowOpen: true,
        url: url,
        target: target
    });
    // 不实际打开，返回null
    return null;
};
""")

# 点击第一个产品
print("点击产品...")
first_link = driver.find_element(By.CSS_SELECTOR, "a.lb_title")
product_name = first_link.text
print(f"点击: {product_name}")

# 使用JavaScript点击
driver.execute_script("arguments[0].click();", first_link)
time.sleep(2)

# 获取捕获的URL
clicked_urls = driver.execute_script("return window._clickedUrls")
print(f"\n捕获的URL: {clicked_urls}")

# 检查是否有新窗口（并获取URL）
if len(driver.window_handles) > 1:
    main_window = driver.current_window_handle
    for handle in driver.window_handles:
        if handle != main_window:
            driver.switch_to.window(handle)
            print(f"新窗口URL: {driver.current_url}")

            # 如果URL有效，尝试在主窗口直接导航
            detail_url = driver.current_url
            driver.close()
            driver.switch_to.window(main_window)

            if 'wealth/lcxx' in detail_url:
                print(f"\n在主窗口导航到: {detail_url}")
                driver.get(detail_url)

                for i in range(15):
                    size = len(driver.page_source)
                    print(f"{i+1}秒: {size} bytes")
                    if size > 10000:
                        break
                    time.sleep(1)

                if len(driver.page_source) > 10000:
                    print("\n成功加载详情页!")
                    with open("D:/AI-FINANCE/detail_direct.html", "w", encoding="utf-8") as f:
                        f.write(driver.page_source)

print("\n完成")
driver.quit()
