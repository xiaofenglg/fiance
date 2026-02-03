# -*- coding: utf-8 -*-
"""调试 - 在同一标签页导航到详情页"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import re

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(10):
    size = len(driver.page_source)
    print(f"{i+1}秒: {size} bytes")
    if size > 10000:
        break
    time.sleep(1)

# 分析产品链接获取detail URL
print("\n分析产品链接...")
js_get_links = """
var result = [];
var links = document.querySelectorAll('a.lb_title');
for (var i = 0; i < links.length && i < 3; i++) {
    var link = links[i];
    // 获取onclick或数据属性
    result.push({
        text: link.innerText,
        href: link.href,
        onclick: link.getAttribute('onclick'),
        datas_ts: link.getAttribute('datas-ts'),
        parent_row_html: link.closest('tr') ? link.closest('tr').innerHTML.substring(0, 300) : null
    });
}
return result;
"""
links_info = driver.execute_script(js_get_links)
for info in links_info:
    print(f"  产品: {info['text']}")
    print(f"  href: {info['href']}")
    print(f"  datas-ts: {info['datas_ts']}")
    print()

# 尝试直接获取产品详情URL
# 光大理财网站可能有API或者隐藏的URL模式
print("\n尝试获取产品详情URL...")

# 方法1: 尝试通过JavaScript直接获取链接目标
js_get_target = """
var links = document.querySelectorAll('a.lb_title');
if (links.length > 0) {
    var link = links[0];
    // 模拟点击看看会打开什么
    var clickEvent = new MouseEvent('click', {
        view: window,
        bubbles: true,
        cancelable: true,
        ctrlKey: true  // 尝试ctrl+click避免导航
    });

    // 监听新标签请求
    var origOpen = window.open;
    var targetUrl = null;
    window.open = function(url) {
        targetUrl = url;
        return null;
    };

    link.dispatchEvent(clickEvent);

    window.open = origOpen;
    return targetUrl;
}
return null;
"""

# 方法2: 直接检查页面上是否有产品ID和构建URL
print("\n分析页面结构找产品ID...")
page_source = driver.page_source

# 查找可能的产品ID模式
# 从之前的分析，detail URL是: /wealth/lcxx/lccp14/{article_id}/index.html
js_find_product_data = """
var result = [];
var rows = document.querySelectorAll('table tbody tr, table tr');
for (var i = 0; i < rows.length && result.length < 3; i++) {
    var cells = rows[i].querySelectorAll('td');
    if (cells.length >= 2) {
        var code = cells[1] ? cells[1].innerText.trim() : '';
        if (code && code.startsWith('EB')) {
            var link = cells[0].querySelector('a');
            result.push({
                code: code,
                name: cells[0] ? cells[0].innerText.trim() : '',
                link_element: link ? link.outerHTML : null
            });
        }
    }
}
return result;
"""
products = driver.execute_script(js_find_product_data)
for p in products:
    print(f"  {p['code']}: {p['name']}")
    if p['link_element']:
        print(f"    链接HTML: {p['link_element'][:200]}")

# 方法3: 检查是否有onclick事件绑定
print("\n查找click事件绑定...")
js_check_event = """
var link = document.querySelector('a.lb_title');
if (link) {
    // 检查jQuery事件
    if (window.jQuery) {
        var events = jQuery._data(link, 'events');
        return JSON.stringify(events);
    }
    return 'no jquery or no events';
}
return null;
"""
events = driver.execute_script(js_check_event)
print(f"  事件: {events}")

# 方法4: 直接在同一标签页导航
print("\n尝试在同一标签页打开详情...")
# 从已保存的detail_page_source.html我们知道URL格式
# 尝试获取第一个产品的详情URL

# 通过拦截window.open获取目标URL
js_intercept = """
return new Promise((resolve) => {
    var origOpen = window.open;
    window.open = function(url, target, features) {
        window.open = origOpen;
        resolve(url);
        return null;  // 阻止实际打开
    };

    var link = document.querySelector('a.lb_title');
    if (link) {
        link.click();
        // 超时返回null
        setTimeout(() => resolve(null), 1000);
    } else {
        resolve(null);
    }
});
"""

# 使用execute_async_script来处理Promise
driver.set_script_timeout(10)
try:
    result = driver.execute_async_script("""
        var callback = arguments[arguments.length - 1];
        var origOpen = window.open;
        window.open = function(url, target, features) {
            window.open = origOpen;
            callback(url);
            return null;
        };

        var link = document.querySelector('a.lb_title');
        if (link) {
            link.click();
            setTimeout(function() { callback(null); }, 2000);
        } else {
            callback(null);
        }
    """)
    print(f"  拦截到URL: {result}")

    if result:
        print(f"\n直接导航到: {result}")
        driver.get(result)

        # 等待页面加载
        for i in range(15):
            size = len(driver.page_source)
            print(f"{i+1}秒: {size} bytes")
            if size > 10000:
                break
            time.sleep(1)

        if len(driver.page_source) > 1000:
            print("\n页面加载成功!")

            # 保存页面
            with open("D:/AI-FINANCE/detail_same_tab.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)

            # 点击产品净值标签
            try:
                nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
                nav_tab.click()
                time.sleep(3)

                # 找iframe
                iframe = driver.find_element(By.ID, "fundValueframe")
                iframe_src = iframe.get_attribute('src')
                print(f"iframe src: {iframe_src}")

                driver.switch_to.frame(iframe)
                time.sleep(3)

                body_text = driver.execute_script("return document.body.innerText")
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', body_text)
                if dates:
                    print(f"找到日期: {dates[:10]}")

                nav_data = re.findall(r'(\d{4}-\d{2}-\d{2})\s+([\d.]+)', body_text)
                if nav_data:
                    print(f"找到净值数据: {nav_data[:5]}")
                else:
                    print("iframe文本前500字符:")
                    print(body_text[:500])

            except Exception as e:
                print(f"详情页处理失败: {e}")
        else:
            print("页面加载被阻止")
except Exception as e:
    print(f"拦截失败: {e}")

print("\n完成")
driver.quit()
