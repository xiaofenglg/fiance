# -*- coding: utf-8 -*-
"""使用原版方法测试净值提取"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time

BASE_URL = "https://www.cebwm.com/wealth/grlc/index.html"
# 使用已成功爬取过的产品
TEST_URL = "https://www.cebwm.com/wealth/lcxx/lccp14/205687237/index.html"  # EB1008

def test_original_method():
    """完全模拟原版爬虫的方法"""
    print("启动新浏览器...")
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = uc.Chrome(options=options, use_subprocess=True)

    try:
        # 1. 先访问列表页（建立会话）
        print(f"\n1. 访问列表页: {BASE_URL}")
        driver.get(BASE_URL)

        # 等待页面加载
        for i in range(15):
            size = len(driver.page_source)
            if size > 10000:
                break
            time.sleep(1)
        print(f"   列表页大小: {len(driver.page_source)}")

        # 2. 使用JS导航到详情页
        print(f"\n2. JS导航到详情页: {TEST_URL}")
        driver.execute_script(f"window.location.href = '{TEST_URL}'")

        # 等待页面加载
        for i in range(15):
            size = len(driver.page_source)
            if size > 10000:
                break
            time.sleep(1)
        print(f"   详情页大小: {len(driver.page_source)}")
        print(f"   当前URL: {driver.current_url}")

        # 3. 点击"产品净值"标签
        print("\n3. 点击产品净值标签...")
        nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
        nav_tab.click()
        print("   等待 3 秒...")
        time.sleep(3)

        # 4. 切换到iframe
        print("\n4. 切换到 iframe...")
        iframe = driver.find_element(By.ID, "fundValueframe")
        print(f"   iframe src: {iframe.get_attribute('src')[:100]}")
        driver.switch_to.frame(iframe)
        print("   等待 3 秒...")
        time.sleep(3)

        # 5. 提取数据
        print("\n5. 提取净值数据...")
        print(f"   iframe 内容大小: {len(driver.page_source)}")

        # 打印所有表格内容
        tables = driver.find_elements(By.TAG_NAME, 'table')
        print(f"   找到 {len(tables)} 个表格")

        for t_idx, table in enumerate(tables):
            rows = table.find_elements(By.TAG_NAME, 'tr')
            print(f"\n   表格 {t_idx} ({len(rows)} 行):")
            for r_idx, row in enumerate(rows[:6]):
                cells = row.find_elements(By.CSS_SELECTOR, 'td, th')
                cell_texts = [c.text[:15] for c in cells]
                print(f"      行 {r_idx}: {cell_texts}")

        # 使用 JS 提取
        js_extract = """
        var result = [];
        var tables = document.querySelectorAll('table');
        for (var t = 0; t < tables.length; t++) {
            var rows = tables[t].querySelectorAll('tr');
            for (var i = 0; i < rows.length; i++) {
                var cells = rows[i].querySelectorAll('td');
                if (cells.length >= 2) {
                    var text0 = cells[0] ? cells[0].innerText.trim() : '';
                    var text1 = cells[1] ? cells[1].innerText.trim() : '';
                    var text2 = cells[2] ? cells[2].innerText.trim() : '';
                    if (/\\d{4}-\\d{2}-\\d{2}/.test(text0)) {
                        result.push({
                            date: text0,
                            unit_nav: parseFloat(text1) || null,
                            total_nav: parseFloat(text2) || null
                        });
                    }
                }
            }
        }
        return result;
        """
        nav_data = driver.execute_script(js_extract) or []

        print(f"\n提取到 {len(nav_data)} 条净值记录:")
        for nav in nav_data[:5]:
            print(f"   {nav}")

        driver.switch_to.default_content()

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
    test_original_method()
