# -*- coding: utf-8 -*-
"""
光大理财净值获取 v11 - 监控网络请求找到真实API
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json


def test():
    print("连接Chrome...")

    options = Options()
    options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    driver = webdriver.Chrome(options=options)

    # 切换到光大理财页面
    for handle in driver.window_handles:
        driver.switch_to.window(handle)
        if 'cebwm' in driver.current_url:
            print(f"已连接: {driver.current_url}")
            break

    time.sleep(1)

    # 首先确保在正确的页面
    if 'cebwm' not in driver.current_url:
        print("不在光大理财页面，尝试导航...")
        driver.get("https://www.cebwm.com/wealth/grlc/index.html")
        time.sleep(5)

    print(f"当前URL: {driver.current_url}")

    # 启用CDP网络监控
    driver.execute_cdp_cmd('Network.enable', {})

    # 获取性能日志
    print("\n=== 分析最近的网络请求 ===")

    logs = driver.execute_script("""
        var entries = performance.getEntriesByType('resource');
        var apiCalls = [];
        for (var i = 0; i < entries.length; i++) {
            var entry = entries[i];
            if (entry.name.includes('api') || entry.name.includes('eportal') ||
                entry.name.includes('product') || entry.name.includes('search') ||
                entry.name.includes('finance') || entry.name.includes('wealth')) {
                apiCalls.push({
                    name: entry.name,
                    type: entry.initiatorType
                });
            }
        }
        return apiCalls;
    """)

    print(f"找到 {len(logs)} 个相关请求:")
    for log in logs[:30]:
        print(f"  {log['type']}: {log['name'][:100]}")

    # 查找页面中已加载的产品数据
    print("\n\n=== 查找页面中的产品数据 ===")

    # 在JavaScript变量中查找产品数据
    product_data = driver.execute_script("""
        // 查找可能包含产品数据的全局变量
        var result = {};

        // 检查window对象中的可能变量
        for (var key in window) {
            try {
                var val = window[key];
                if (val && typeof val === 'object') {
                    var str = JSON.stringify(val);
                    if (str && str.includes('EB1008') || str.includes('CPCODE') ||
                        str.includes('HTMLURL') || str.includes('productDetail')) {
                        result[key] = val;
                    }
                }
            } catch(e) {}
        }

        return result;
    """)

    if product_data:
        print(f"找到包含产品数据的变量:")
        for key, val in product_data.items():
            print(f"  {key}: {str(val)[:500]}")

    # 直接从表格中提取产品信息，包括链接
    print("\n\n=== 从表格提取产品链接 ===")

    tables = driver.find_elements(By.TAG_NAME, 'table')
    print(f"找到 {len(tables)} 个表格")

    for i, table in enumerate(tables):
        rows = table.find_elements(By.TAG_NAME, 'tr')
        if len(rows) > 5:  # 产品表格应该有多行
            print(f"\n表格 {i} ({len(rows)} 行):")

            for j, row in enumerate(rows[1:4]):  # 前3个产品行
                cells = row.find_elements(By.TAG_NAME, 'td')
                if len(cells) >= 2:
                    # 获取产品名称和代码
                    name = cells[0].text.strip()
                    code = cells[1].text.strip() if len(cells) > 1 else ""

                    # 获取所有链接
                    links = row.find_elements(By.TAG_NAME, 'a')
                    print(f"\n  产品 {j+1}: {name} ({code})")
                    print(f"  链接数: {len(links)}")

                    for link in links:
                        href = link.get_attribute('href')
                        text = link.text[:20]
                        onclick = link.get_attribute('onclick')
                        data_ts = link.get_attribute('datas-ts')

                        print(f"    [{text}] href={href[:50] if href else 'None'}")
                        if data_ts:
                            print(f"      datas-ts={data_ts}")

    # 尝试直接调用页面的JavaScript函数来获取产品详情
    print("\n\n=== 尝试调用页面函数 ===")

    # 查找可能的详情函数
    funcs = driver.execute_script("""
        var funcs = [];
        for (var key in window) {
            if (typeof window[key] === 'function') {
                var str = window[key].toString();
                if (str.length < 3000 &&
                    (str.includes('detail') || str.includes('Detail') ||
                     str.includes('product') || str.includes('Product') ||
                     str.includes('showCp') || str.includes('openCp') ||
                     str.includes('lhgdialog') || str.includes('layer'))) {
                    funcs.push({name: key, code: str.substring(0, 300)});
                }
            }
        }
        return funcs;
    """)

    print(f"找到 {len(funcs)} 个可能相关的函数:")
    for func in funcs[:15]:
        print(f"\n  {func['name']}:")
        print(f"    {func['code'][:200]}...")

    print("\n完成")


if __name__ == "__main__":
    test()
