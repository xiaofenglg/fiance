# -*- coding: utf-8 -*-
"""
光大理财API分析工具

通过CDP连接Chrome，监控网络请求，找到历史净值API
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json


def find_nav_api():
    """查找净值历史API"""
    print("连接Chrome浏览器...")

    options = Options()
    options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    driver = webdriver.Chrome(options=options)

    # 切换到光大理财页面
    for handle in driver.window_handles:
        driver.switch_to.window(handle)
        if 'cebwm' in driver.current_url:
            print(f"已连接: {driver.current_url}")
            break

    # 启用网络日志
    driver.execute_cdp_cmd('Network.enable', {})

    print("\n分析页面结构...")

    # 获取页面的所有script标签
    scripts = driver.find_elements(By.TAG_NAME, 'script')
    print(f"找到 {len(scripts)} 个script标签")

    # 查找包含API相关代码的脚本
    api_hints = []
    for i, script in enumerate(scripts):
        try:
            src = script.get_attribute('src')
            if src:
                print(f"  外部脚本: {src}")
            else:
                content = script.get_attribute('innerHTML')
                if content and len(content) > 100:
                    # 查找API相关关键词
                    keywords = ['ajax', 'fetch', 'api', 'nav', '净值', 'history', 'detail', 'product']
                    for kw in keywords:
                        if kw.lower() in content.lower():
                            api_hints.append((i, kw, content[:500]))
                            break
        except:
            pass

    print(f"\n找到 {len(api_hints)} 个可能包含API的脚本")
    for idx, kw, content in api_hints[:3]:
        print(f"\n脚本 {idx} (关键词: {kw}):")
        print(content[:300] + "...")

    # 查找产品链接的点击事件
    print("\n\n分析产品行的点击事件...")

    try:
        rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')
        if rows:
            first_row = rows[0]
            # 获取行的HTML
            row_html = first_row.get_attribute('outerHTML')
            print(f"第一行HTML:\n{row_html[:1000]}")

            # 查找onclick属性
            onclick = first_row.get_attribute('onclick')
            if onclick:
                print(f"\nonclick事件: {onclick}")

            # 查找行内的所有链接
            links = first_row.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                onclick = link.get_attribute('onclick')
                text = link.text
                print(f"\n链接: {text}")
                print(f"  href: {href}")
                print(f"  onclick: {onclick}")
    except Exception as e:
        print(f"分析行错误: {e}")

    # 查找全局JavaScript函数
    print("\n\n查找页面全局函数...")

    js_functions = driver.execute_script("""
        var funcs = [];
        for (var key in window) {
            if (typeof window[key] === 'function' &&
                key.indexOf('_') !== 0 &&
                !['alert', 'confirm', 'prompt', 'open', 'close'].includes(key)) {
                var funcStr = window[key].toString();
                if (funcStr.length < 2000 &&
                    (funcStr.includes('ajax') ||
                     funcStr.includes('fetch') ||
                     funcStr.includes('product') ||
                     funcStr.includes('detail') ||
                     funcStr.includes('nav') ||
                     funcStr.includes('净值'))) {
                    funcs.push({name: key, code: funcStr.substring(0, 500)});
                }
            }
        }
        return funcs;
    """)

    print(f"找到 {len(js_functions)} 个可能相关的函数:")
    for func in js_functions[:10]:
        print(f"\n函数: {func['name']}")
        print(f"代码: {func['code'][:300]}...")

    # 尝试获取Vue/React组件数据
    print("\n\n尝试获取前端框架数据...")

    vue_data = driver.execute_script("""
        // 尝试获取Vue实例数据
        var vueApps = [];
        document.querySelectorAll('*').forEach(function(el) {
            if (el.__vue__) {
                var data = el.__vue__.$data;
                if (data) {
                    vueApps.push({
                        tag: el.tagName,
                        data: JSON.stringify(data).substring(0, 1000)
                    });
                }
            }
        });
        return vueApps;
    """)

    if vue_data:
        print(f"找到 {len(vue_data)} 个Vue组件:")
        for v in vue_data[:5]:
            print(f"\n{v['tag']}: {v['data'][:500]}...")
    else:
        print("未找到Vue组件，尝试其他框架...")

    # 查找网络请求相关配置
    print("\n\n查找API配置...")

    api_config = driver.execute_script("""
        var config = {};

        // 查找常见的API配置变量
        var configNames = ['API_BASE', 'BASE_URL', 'apiUrl', 'baseUrl', 'serverUrl', 'config'];
        for (var name of configNames) {
            if (window[name]) {
                config[name] = window[name];
            }
        }

        // 查找axios配置
        if (window.axios && window.axios.defaults) {
            config['axios.defaults.baseURL'] = window.axios.defaults.baseURL;
        }

        return config;
    """)

    print(f"API配置: {json.dumps(api_config, indent=2, ensure_ascii=False)}")

    # 监控点击产品后的网络请求
    print("\n\n准备监控网络请求...")
    print("尝试点击第一个产品并捕获网络请求...")

    # 设置请求拦截
    requests_log = []

    driver.execute_cdp_cmd('Network.enable', {})

    # 获取产品代码用于后续分析
    try:
        first_product_code = driver.find_element(By.CSS_SELECTOR, 'table tbody tr td:nth-child(2)').text
        print(f"第一个产品代码: {first_product_code}")
    except:
        first_product_code = "EB1008"

    # 尝试构造可能的API URL
    print("\n\n尝试可能的API端点...")

    possible_apis = [
        f"/api/product/{first_product_code}",
        f"/api/product/detail/{first_product_code}",
        f"/api/nav/{first_product_code}",
        f"/api/nav/history/{first_product_code}",
        f"/wealth/api/product/{first_product_code}",
        f"/wealth/grlc/api/product/{first_product_code}",
        f"/wealth/grlc/productDetail/{first_product_code}",
    ]

    print("可能的API端点:")
    for api in possible_apis:
        print(f"  https://www.cebwm.com{api}")

    # 查找页面中的所有data属性
    print("\n\n查找产品数据属性...")

    data_attrs = driver.execute_script("""
        var attrs = [];
        document.querySelectorAll('table tbody tr').forEach(function(tr, i) {
            if (i < 3) {
                var rowData = {};
                for (var attr of tr.attributes) {
                    if (attr.name.startsWith('data-')) {
                        rowData[attr.name] = attr.value;
                    }
                }
                if (Object.keys(rowData).length > 0) {
                    attrs.push(rowData);
                }

                // 检查行内的元素
                tr.querySelectorAll('*').forEach(function(el) {
                    for (var attr of el.attributes) {
                        if (attr.name.startsWith('data-') || attr.name === 'onclick') {
                            attrs.push({
                                tag: el.tagName,
                                attr: attr.name,
                                value: attr.value.substring(0, 200)
                            });
                        }
                    }
                });
            }
        });
        return attrs;
    """)

    print(f"找到 {len(data_attrs)} 个数据属性:")
    for attr in data_attrs[:20]:
        print(f"  {attr}")

    print("\n\n分析完成。请检查上述信息找到历史净值API。")

    return driver


if __name__ == "__main__":
    driver = find_nav_api()
    print("\n保持连接，你可以继续手动操作浏览器分析...")
    input("按Enter键退出...")
