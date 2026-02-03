# -*- coding: utf-8 -*-
"""使用JavaScript进行页内导航"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
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
    if size > 10000:
        break
    time.sleep(1)

print(f"页面加载完成: {len(driver.page_source)} bytes")

# 方法1: 尝试使用window.location导航
print("\n方法1: 使用window.location导航...")
detail_url = "https://www.cebwm.com/wealth/lcxx/lccp14/205687237/index.html"

driver.execute_script(f"window.location.href = '{detail_url}'")

for i in range(10):
    size = len(driver.page_source)
    print(f"{i+1}秒: {size} bytes")
    if size > 10000:
        break
    time.sleep(1)

if len(driver.page_source) > 10000:
    print("方法1成功!")
else:
    print("方法1失败，尝试方法2...")

    # 重新加载列表页
    driver.get("https://www.cebwm.com/wealth/grlc/index.html")
    for i in range(10):
        size = len(driver.page_source)
        if size > 10000:
            break
        time.sleep(1)

    # 方法2: 创建链接并模拟点击
    print("\n方法2: 创建链接并模拟点击...")
    js_create_link = f"""
    var link = document.createElement('a');
    link.href = '{detail_url}';
    link.target = '_self';
    link.id = 'test_nav_link';
    document.body.appendChild(link);
    return true;
    """
    driver.execute_script(js_create_link)

    # 用Selenium点击
    link = driver.find_element(By.ID, "test_nav_link")
    link.click()

    for i in range(10):
        time.sleep(1)
        size = len(driver.page_source)
        print(f"{i+1}秒: {size} bytes")
        if size > 10000:
            break

if len(driver.page_source) > 10000:
    print("导航成功!")
else:
    print("方法2也失败，尝试方法3...")

    # 方法3: 直接使用fetch获取页面并替换内容
    print("\n方法3: 使用fetch获取页面...")
    driver.get("https://www.cebwm.com/wealth/grlc/index.html")
    for i in range(10):
        size = len(driver.page_source)
        if size > 10000:
            break
        time.sleep(1)

    js_fetch = f"""
    return fetch('{detail_url}', {{
        method: 'GET',
        credentials: 'include',
        headers: {{
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
        }}
    }})
    .then(response => response.text())
    .then(html => {{
        return {{success: true, length: html.length, preview: html.substring(0, 500)}};
    }})
    .catch(error => {{
        return {{success: false, error: error.toString()}};
    }});
    """

    driver.set_script_timeout(30)
    try:
        result = driver.execute_async_script("""
            var callback = arguments[arguments.length - 1];
            fetch('""" + detail_url + """', {
                method: 'GET',
                credentials: 'include'
            })
            .then(response => response.text())
            .then(html => {
                callback({success: true, length: html.length, preview: html.substring(0, 1000)});
            })
            .catch(error => {
                callback({success: false, error: error.toString()});
            });
        """)
        print(f"Fetch结果: 成功={result.get('success')}, 长度={result.get('length')}")
        if result.get('preview'):
            print(f"预览: {result.get('preview')[:300]}...")
    except Exception as e:
        print(f"Fetch失败: {e}")

# 方法4: 重新打开一个新标签页导航
print("\n方法4: 打开新标签页导航...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")
for i in range(10):
    size = len(driver.page_source)
    if size > 10000:
        break
    time.sleep(1)

main_window = driver.current_window_handle

# 使用Ctrl+T打开新标签页
driver.execute_script("window.open('about:blank', '_blank');")
time.sleep(1)

if len(driver.window_handles) > 1:
    new_tab = [h for h in driver.window_handles if h != main_window][0]
    driver.switch_to.window(new_tab)

    # 在新标签页中导航
    driver.get(detail_url)

    for i in range(15):
        size = len(driver.page_source)
        print(f"新标签{i+1}秒: {size} bytes")
        if size > 10000:
            break
        time.sleep(1)

    if len(driver.page_source) > 10000:
        print("\n新标签页导航成功!")
        # 处理净值数据...
    else:
        print("新标签页也被阻止")

print("\n完成")
driver.quit()
