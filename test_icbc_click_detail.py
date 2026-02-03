# -*- coding: utf-8 -*-
"""检查工银理财点击产品后的行为 - 下载或打开详情"""

import undetected_chromedriver as uc
import time
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
# 禁用下载提示，允许自动下载
prefs = {
    "download.prompt_for_download": False,
    "download.default_directory": "D:\\AI-FINANCE\\downloads",
    "plugins.always_open_pdf_externally": True
}
options.add_experimental_option("prefs", prefs)
driver = uc.Chrome(options=options, use_subprocess=True)

# 监控网络请求
driver.execute_cdp_cmd('Network.enable', {})

print("\n打开净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")

print("等待页面加载...")
time.sleep(8)

# 查看产品项的完整HTML结构
print("\n=== 产品项HTML结构 ===")
js_get_html = """
var item = document.querySelector('.product-item');
if (item) {
    return {
        outerHTML: item.outerHTML,
        parentHTML: item.parentElement.outerHTML.substring(0, 500)
    };
}
return {};
"""
html = driver.execute_script(js_get_html)
if html.get('outerHTML'):
    print(f"产品项HTML:\n{html['outerHTML']}")

# 检查是否有链接href
print("\n\n=== 检查所有链接 ===")
js_all_links = """
var links = document.querySelectorAll('a[href]');
var result = [];
for (var i = 0; i < links.length; i++) {
    var link = links[i];
    var href = link.href;
    // 只显示可能是产品详情的链接
    if (href.includes('pdf') || href.includes('detail') || href.includes('product') ||
        href.includes('disclosure') || href.includes('networth') || href.includes('info')) {
        result.push({
            href: href,
            text: link.innerText.substring(0, 50)
        });
    }
}
return result;
"""
links = driver.execute_script(js_all_links)
print(f"找到 {len(links)} 个相关链接:")
for link in links[:10]:
    print(f"  {link['text']}: {link['href']}")

# 检查产品列表的父容器是否有事件监听
print("\n\n=== 检查事件监听 ===")
js_events = """
var container = document.querySelector('.product-list') || document.querySelector('[class*="list"]');
if (container) {
    // Vue/React通常在__vue__或_reactRootContainer中
    var hasVue = !!container.__vue__;
    var hasReact = !!container._reactRootContainer;
    return {
        hasVue: hasVue,
        hasReact: hasReact,
        className: container.className
    };
}
return {found: false};
"""
events = driver.execute_script(js_events)
print(f"事件检查: {events}")

# 模拟用户点击 - 使用ActionChains
print("\n\n=== 使用ActionChains点击第一个产品 ===")
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

try:
    items = driver.find_elements(By.CSS_SELECTOR, '.product-item')
    if items:
        print(f"找到 {len(items)} 个产品项")

        # 记录当前窗口
        original_window = driver.current_window_handle
        original_url = driver.current_url

        # 用ActionChains点击
        actions = ActionChains(driver)
        actions.move_to_element(items[0]).click().perform()
        print("ActionChains点击完成")

        time.sleep(3)

        # 检查是否有新窗口
        windows = driver.window_handles
        if len(windows) > 1:
            print(f"打开了新窗口! 共 {len(windows)} 个窗口")
            driver.switch_to.window(windows[-1])
            print(f"新窗口URL: {driver.current_url}")
            print(f"新窗口标题: {driver.title}")
            # 如果是PDF，获取内容
            if '.pdf' in driver.current_url.lower():
                print("这是PDF文件!")
        else:
            print("没有打开新窗口")
            print(f"当前URL: {driver.current_url}")
            if driver.current_url != original_url:
                print("URL已变化!")
except Exception as e:
    print(f"点击出错: {e}")

# 获取下载的文件
print("\n\n=== 检查下载 ===")
import os
download_dir = "D:\\AI-FINANCE\\downloads"
if os.path.exists(download_dir):
    files = os.listdir(download_dir)
    print(f"下载目录文件: {files}")

# 检查网页源代码中的Vue组件
print("\n\n=== 检查Vue组件 ===")
js_vue = """
var app = document.querySelector('#app');
if (app && app.__vue__) {
    var vm = app.__vue__;
    var data = vm.$data || {};
    return {
        hasData: Object.keys(data).length > 0,
        dataKeys: Object.keys(data).slice(0, 20),
        methods: Object.keys(vm).filter(k => typeof vm[k] === 'function').slice(0, 20)
    };
}
return {noVue: true};
"""
vue = driver.execute_script(js_vue)
print(f"Vue组件: {vue}")

print("\n完成")
driver.quit()
