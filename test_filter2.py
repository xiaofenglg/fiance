# -*- coding: utf-8 -*-
"""测试光大理财网站筛选功能 - 详细版"""

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

        # 查看完整的筛选区域
        print("\n=== 查看筛选区域详情 ===")
        filter_info = driver.execute_script("""
            var result = [];
            // 查找所有 dt/dd 结构（常见的筛选布局）
            var dts = document.querySelectorAll('dt');
            for (var i = 0; i < dts.length; i++) {
                var dt = dts[i];
                var dd = dt.nextElementSibling;
                if (dd && dd.tagName === 'DD') {
                    var labels = [];
                    var items = dd.querySelectorAll('a, span, label');
                    for (var j = 0; j < Math.min(items.length, 10); j++) {
                        var text = items[j].innerText.trim();
                        var onclick = items[j].getAttribute('onclick') || '';
                        if (text) labels.push(text + (onclick ? ' [' + onclick.substring(0,50) + ']' : ''));
                    }
                    result.push(dt.innerText.trim() + ': ' + labels.join(' | '));
                }
            }

            // 查找筛选区域的所有链接
            var screenDiv = document.querySelector('.screen_list, .filter_list, [class*="screen"]');
            if (screenDiv) {
                var links = screenDiv.querySelectorAll('a');
                result.push('--- 筛选链接 ---');
                for (var i = 0; i < links.length; i++) {
                    var text = links[i].innerText.trim();
                    var onclick = links[i].getAttribute('onclick') || links[i].getAttribute('data-value') || '';
                    if (text && text.length < 30) {
                        result.push(text + ' -> ' + onclick.substring(0, 80));
                    }
                }
            }

            return result;
        """)

        for f in filter_info:
            print(f"  {f}")

        # 尝试获取筛选条件的值
        print("\n=== 尝试获取筛选参数 ===")
        params = driver.execute_script("""
            // 查看页面中的筛选参数变量
            var params = {};
            if (typeof screenData !== 'undefined') params.screenData = screenData;
            if (typeof filterParams !== 'undefined') params.filterParams = filterParams;
            if (typeof searchParams !== 'undefined') params.searchParams = searchParams;

            // 查看表单
            var forms = document.querySelectorAll('form');
            params.forms = [];
            for (var i = 0; i < forms.length; i++) {
                var inputs = forms[i].querySelectorAll('input, select');
                var formData = {};
                for (var j = 0; j < inputs.length; j++) {
                    formData[inputs[j].name || inputs[j].id] = inputs[j].value;
                }
                params.forms.push(formData);
            }

            return JSON.stringify(params, null, 2);
        """)
        print(params)

        # 点击"开放式"筛选
        print("\n=== 点击'开放式'筛选 ===")
        result = driver.execute_script("""
            var links = document.querySelectorAll('a');
            for (var i = 0; i < links.length; i++) {
                if (links[i].innerText.trim() === '开放式') {
                    links[i].click();
                    return '点击成功';
                }
            }
            return '未找到开放式链接';
        """)
        print(f"  {result}")
        time.sleep(3)

        # 查看筛选后的产品数量
        count = driver.execute_script("""
            var countElem = document.querySelector('.total_count1, .product_total_num, [class*="total"]');
            return countElem ? countElem.innerText : '未找到';
        """)
        print(f"  筛选后产品数量: {count}")

        # 查看当前页面URL
        print(f"  当前URL: {driver.current_url}")

        # 查看是否有更多筛选条件（起购金额）
        print("\n=== 查找起购金额筛选 ===")
        amount_filters = driver.execute_script("""
            var result = [];
            var allLinks = document.querySelectorAll('a');
            for (var i = 0; i < allLinks.length; i++) {
                var text = allLinks[i].innerText.trim();
                if (text.indexOf('万') >= 0 || text.indexOf('元') >= 0) {
                    result.push(text + ' | onclick: ' + (allLinks[i].getAttribute('onclick') || 'none').substring(0, 50));
                }
            }
            return result;
        """)
        for f in amount_filters:
            print(f"  {f}")

        # 点击"10万以上"
        print("\n=== 点击'10万以上'筛选 ===")
        result = driver.execute_script("""
            var links = document.querySelectorAll('a');
            for (var i = 0; i < links.length; i++) {
                var text = links[i].innerText.trim();
                if (text.indexOf('10万') >= 0) {
                    links[i].click();
                    return '点击: ' + text;
                }
            }
            return '未找到10万相关链接';
        """)
        print(f"  {result}")
        time.sleep(3)

        # 查看筛选后的产品数量
        count = driver.execute_script("""
            var countElem = document.querySelector('.total_count1, .product_total_num, [class*="total"]');
            return countElem ? countElem.innerText : '未找到';
        """)
        print(f"  筛选后产品数量: {count}")

        # 获取总页数
        pages = driver.execute_script("""
            var pageElem = document.querySelector('#totalpage1, [class*="totalpage"]');
            return pageElem ? pageElem.innerText : '未找到';
        """)
        print(f"  总页数: {pages}")

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
