# -*- coding: utf-8 -*-
"""测试光大理财网站筛选功能"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time

BASE_URL = "https://www.cebwm.com/wealth/grlc/index.html"

def test_filters():
    print("启动浏览器...")
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    driver = uc.Chrome(options=options, use_subprocess=True)

    try:
        print(f"访问: {BASE_URL}")
        driver.get(BASE_URL)
        time.sleep(5)

        # 查找筛选条件相关元素
        print("\n=== 查找筛选条件 ===")

        # 查找所有可能的筛选按钮/链接
        filters = driver.execute_script("""
            var result = [];
            // 查找筛选区域
            var filterDivs = document.querySelectorAll('.filter, .screen, .search, [class*="filter"], [class*="screen"]');
            result.push('筛选区域: ' + filterDivs.length + ' 个');

            // 查找所有链接
            var links = document.querySelectorAll('a');
            for (var i = 0; i < links.length; i++) {
                var text = links[i].innerText.trim();
                if (text.indexOf('10万') >= 0 || text.indexOf('开放') >= 0 ||
                    text.indexOf('人民币') >= 0 || text.indexOf('筛选') >= 0) {
                    result.push('链接: ' + text + ' | href: ' + (links[i].href || 'none'));
                }
            }

            // 查找所有按钮
            var buttons = document.querySelectorAll('button, input[type="button"], .btn');
            for (var i = 0; i < buttons.length; i++) {
                var text = buttons[i].innerText.trim() || buttons[i].value;
                if (text) result.push('按钮: ' + text);
            }

            // 查找下拉选择框
            var selects = document.querySelectorAll('select');
            for (var i = 0; i < selects.length; i++) {
                var options = [];
                for (var j = 0; j < selects[i].options.length; j++) {
                    options.push(selects[i].options[j].text);
                }
                result.push('下拉框: ' + options.join(', '));
            }

            return result;
        """)

        for f in filters:
            print(f"  {f}")

        # 查看页面上的筛选条件HTML
        print("\n=== 筛选条件区域HTML ===")
        filter_html = driver.execute_script("""
            var screenDiv = document.querySelector('.product_screen') ||
                           document.querySelector('[class*="screen"]') ||
                           document.querySelector('[class*="filter"]');
            if (screenDiv) {
                return screenDiv.outerHTML.substring(0, 2000);
            }
            return '未找到筛选区域';
        """)
        print(filter_html[:1000])

        # 查看是否有JavaScript筛选函数
        print("\n=== 查找筛选相关JS函数 ===")
        js_funcs = driver.execute_script("""
            var funcs = [];
            for (var key in window) {
                if (typeof window[key] === 'function' &&
                    (key.toLowerCase().indexOf('filter') >= 0 ||
                     key.toLowerCase().indexOf('screen') >= 0 ||
                     key.toLowerCase().indexOf('search') >= 0 ||
                     key.indexOf('go') >= 0)) {
                    funcs.push(key);
                }
            }
            return funcs;
        """)
        print(f"  相关函数: {js_funcs}")

        input("\n按回车键关闭浏览器...")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    test_filters()
