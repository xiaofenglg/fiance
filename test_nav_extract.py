# -*- coding: utf-8 -*-
"""测试净值提取功能"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

BASE_URL = "https://www.cebwm.com/wealth/grlc/index.html"
TEST_URL = "https://www.cebwm.com/wealth/lcxx/lccp14/191764706/index.html"

def test_nav_extract():
    print("启动浏览器...")
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    driver = uc.Chrome(options=options, use_subprocess=True)

    try:
        # 1. 先访问列表页建立会话
        print(f"访问列表页: {BASE_URL}")
        driver.get(BASE_URL)
        time.sleep(5)
        print(f"页面标题: {driver.title}")
        print(f"页面长度: {len(driver.page_source)}")

        # 2. 使用 JS 导航到详情页
        print(f"\nJS导航到详情页: {TEST_URL}")
        driver.execute_script(f"window.location.href = '{TEST_URL}'")
        time.sleep(5)
        print(f"当前URL: {driver.current_url}")
        print(f"页面长度: {len(driver.page_source)}")

        # 3. 查找页面元素
        print("\n查找页面元素...")

        # 查找标签页
        tabs = driver.find_elements(By.CSS_SELECTOR, ".a1, .a2, .a3")
        print(f"找到 {len(tabs)} 个标签")
        for t in tabs:
            print(f"  标签: {t.text}")

        # 点击产品净值标签
        print("\n点击产品净值标签...")
        try:
            result = driver.execute_script("""
                var tabs = document.querySelectorAll('a');
                for (var i = 0; i < tabs.length; i++) {
                    if (tabs[i].innerText.indexOf('净值') >= 0 ||
                        tabs[i].className.indexOf('a2') >= 0) {
                        console.log('Found tab:', tabs[i].innerText);
                        tabs[i].click();
                        return tabs[i].innerText;
                    }
                }
                return 'not found';
            """)
            print(f"点击结果: {result}")
        except Exception as e:
            print(f"点击失败: {e}")

        time.sleep(3)

        # 4. 查找 iframe
        print("\n查找 iframe...")
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        print(f"找到 {len(iframes)} 个 iframe")
        for iframe in iframes:
            iframe_id = iframe.get_attribute('id')
            iframe_src = iframe.get_attribute('src')
            print(f"  ID: {iframe_id}, SRC: {iframe_src[:100] if iframe_src else 'None'}")

        # 5. 尝试切换到 iframe 并提取数据
        if iframes:
            print("\n切换到 fundValueframe...")
            # 找到正确的 iframe
            for iframe in iframes:
                if iframe.get_attribute('id') == 'fundValueframe':
                    driver.switch_to.frame(iframe)
                    break

            print("等待数据加载...")
            time.sleep(5)  # 增加等待时间

            # 等待表格中有数据
            for wait in range(10):
                row_count = driver.execute_script("""
                    var tables = document.querySelectorAll('table');
                    var total = 0;
                    for (var t = 0; t < tables.length; t++) {
                        var rows = tables[t].querySelectorAll('tr td');
                        total += rows.length;
                    }
                    return total;
                """)
                print(f"  等待 {wait+1}s, 单元格数: {row_count}")
                if row_count > 10:
                    break
                time.sleep(1)

            print(f"iframe 内容长度: {len(driver.page_source)}")

            # 查找表格
            tables = driver.find_elements(By.TAG_NAME, 'table')
            print(f"找到 {len(tables)} 个表格")

            # 先打印表格内容
            print("\n表格内容预览:")
            table_content = driver.execute_script("""
                var result = [];
                var tables = document.querySelectorAll('table');
                for (var t = 0; t < tables.length; t++) {
                    var rows = tables[t].querySelectorAll('tr');
                    result.push('=== Table ' + t + ' (' + rows.length + ' rows) ===');
                    for (var i = 0; i < Math.min(5, rows.length); i++) {
                        var cells = rows[i].querySelectorAll('td, th');
                        var rowText = [];
                        for (var j = 0; j < cells.length; j++) {
                            rowText.push(cells[j].innerText.trim().substring(0, 20));
                        }
                        result.push('Row ' + i + ': ' + rowText.join(' | '));
                    }
                }
                return result;
            """)
            for line in table_content:
                print(f"  {line}")

            # 提取数据
            result = driver.execute_script("""
                var result = [];
                var tables = document.querySelectorAll('table');
                for (var t = 0; t < tables.length; t++) {
                    var rows = tables[t].querySelectorAll('tr');
                    for (var i = 0; i < rows.length; i++) {
                        var cells = rows[i].querySelectorAll('td');
                        if (cells.length >= 2) {
                            var text0 = cells[0] ? cells[0].innerText.trim() : '';
                            var text1 = cells[1] ? cells[1].innerText.trim() : '';
                            if (/\\d{4}-\\d{2}-\\d{2}/.test(text0)) {
                                result.push({date: text0, nav: text1});
                            }
                        }
                    }
                }
                return result;
            """)

            print(f"\n提取到 {len(result)} 条净值数据:")
            for r in result[:5]:
                print(f"  {r}")

            driver.switch_to.default_content()

        print("\n测试完成！")

    except Exception as e:
        print(f"错误: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    test_nav_extract()
