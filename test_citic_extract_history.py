# -*- coding: utf-8 -*-
"""提取中信理财产品详情页的历史净值数据"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=options)

def extract_nav_history_from_detail(driver, wait_time=3):
    """从产品详情页提取历史净值"""
    time.sleep(wait_time)

    nav_history = {}

    # 方法1: 查找历史净值表格或列表
    # 查找包含日期和净值的元素
    try:
        # 查找图表区域或历史净值区域
        chart_area = driver.find_elements(By.CSS_SELECTOR, '[class*="chart"], [class*="history"]')
        for area in chart_area:
            text = area.text
            print(f"   图表区域内容: {text[:200]}...")

        # 查找表格
        tables = driver.find_elements(By.TAG_NAME, 'table')
        for table in tables:
            rows = table.find_elements(By.TAG_NAME, 'tr')
            print(f"   找到表格,共 {len(rows)} 行")
            for row in rows[:10]:
                cells = row.find_elements(By.TAG_NAME, 'td')
                cell_texts = [c.text.strip() for c in cells]
                print(f"      {cell_texts}")

        # 尝试点击"历史净值"tab
        history_tabs = driver.find_elements(By.XPATH, "//*[contains(text(), '历史净值')]")
        for tab in history_tabs:
            try:
                tab.click()
                time.sleep(2)
                print(f"   点击了历史净值tab")
            except:
                pass

        # 查找净值列表(可能是div列表)
        # 通常格式是: 日期 + 单位净值 + 累计净值
        nav_items = driver.find_elements(By.CSS_SELECTOR, '[class*="nav-item"], [class*="list-item"], tr')

        # 查找所有包含净值数据的元素
        body = driver.find_element(By.TAG_NAME, 'body')
        body_text = body.text

        # 尝试找日期+净值的模式
        # 日期格式: 2026-01-15 或 01-15
        date_pattern = r'(\d{4}-\d{2}-\d{2}|\d{2}-\d{2})'
        nav_pattern = r'(\d+\.\d{4})'

        lines = body_text.split('\n')
        current_date = None

        for i, line in enumerate(lines):
            line = line.strip()
            # 检查是否是日期行
            date_match = re.search(date_pattern, line)
            if date_match:
                current_date = date_match.group(1)
                # 如果同一行有净值
                nav_match = re.search(nav_pattern, line)
                if nav_match:
                    nav_history[current_date] = nav_match.group(1)
            elif current_date and re.match(nav_pattern, line):
                # 如果当前行只有净值
                nav_history[current_date] = line
                current_date = None

    except Exception as e:
        print(f"   提取历史净值出错: {e}")

    return nav_history

def main():
    print("=" * 60)
    print("提取中信理财产品历史净值")
    print("=" * 60)

    driver = None
    results = []

    try:
        print("\n1. 启动Chrome...")
        driver = setup_driver()

        print("\n2. 访问列表页...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(5)

        # 切换到iframe
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if iframes:
            driver.switch_to.frame(iframes[0])
            time.sleep(3)

        # 点击"每日购"
        print("\n3. 切换到每日购...")
        tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory, .typeCategory-select')
        for tab in tabs:
            if '每日购' in tab.text:
                tab.click()
                time.sleep(3)
                break

        # 获取前5个产品
        print("\n4. 获取产品列表...")
        product_items = driver.find_elements(By.CSS_SELECTOR, '.list-item')
        print(f"   找到 {len(product_items)} 个产品项")

        # 提取产品基本信息
        product_names = driver.find_elements(By.CSS_SELECTOR, '.item-productName')
        product_codes = driver.find_elements(By.CSS_SELECTOR, '.item-productCode div')

        print(f"   产品名称: {len(product_names)}, 产品代码: {len(product_codes)}")

        # 测试前3个产品
        for idx in range(min(3, len(product_names))):
            print(f"\n   === 产品 {idx+1} ===")
            name = product_names[idx].text.strip()
            code = product_codes[idx].text.strip() if idx < len(product_codes) else ''
            print(f"   名称: {name[:40]}...")
            print(f"   代码: {code}")

            # 点击产品查看详情
            print("   点击查看详情...")
            try:
                product_names[idx].click()
                time.sleep(3)

                # 提取详情页中的历史净值
                print("   提取历史净值...")

                # 查找历史净值区域
                # 首先找到详情弹窗
                detail_content = driver.find_element(By.TAG_NAME, 'body').text
                print(f"   页面文本长度: {len(detail_content)}")

                # 查找净值数据(通常在图表或表格中)
                # 尝试查找历史净值列表
                try:
                    # 查找历史净值区域的所有文本
                    history_sections = driver.find_elements(By.XPATH, "//*[contains(@class, 'history') or contains(@class, 'chart') or contains(@class, 'nav')]")
                    for section in history_sections:
                        section_text = section.text
                        if '1.' in section_text and len(section_text) > 50:
                            print(f"   历史区域: {section_text[:300]}")

                    # 提取日期和净值
                    # 在详情页中查找日期列表
                    date_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '2026-') or contains(text(), '2025-')]")
                    nav_elements = driver.find_elements(By.XPATH, "//*[text()[contains(., '1.0')]]")

                    print(f"   日期元素: {len(date_elements)}, 净值元素: {len(nav_elements)}")

                    # 提取所有可见的日期和净值
                    dates = []
                    navs = []

                    for el in date_elements[:20]:
                        date_text = el.text.strip()
                        if re.match(r'\d{4}-\d{2}-\d{2}', date_text):
                            dates.append(date_text)

                    # 找到表格数据
                    table_rows = driver.find_elements(By.CSS_SELECTOR, 'tr, [class*="row"], [class*="item"]')
                    for row in table_rows:
                        row_text = row.text
                        # 检查是否包含日期和净值
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', row_text)
                        nav_match = re.search(r'(1\.\d{4})', row_text)
                        if date_match and nav_match:
                            dates.append(date_match.group(1))
                            navs.append(nav_match.group(1))

                    print(f"   提取到 {len(dates)} 个日期, {len(navs)} 个净值")

                    # 保存结果
                    product_data = {
                        'code': code,
                        'name': name,
                        'nav_history': dict(zip(dates, navs)) if dates and navs else {}
                    }
                    results.append(product_data)

                except Exception as e:
                    print(f"   提取详情出错: {e}")

                # 关闭详情(点击空白处或按ESC)
                try:
                    # 点击关闭按钮
                    close_btn = driver.find_elements(By.CSS_SELECTOR, '.close, [class*="close"], .modal-close')
                    if close_btn:
                        close_btn[0].click()
                    else:
                        # 按ESC关闭
                        from selenium.webdriver.common.keys import Keys
                        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                    time.sleep(2)
                except:
                    pass

            except Exception as e:
                print(f"   处理产品出错: {e}")

        # 打印结果
        print("\n\n5. 结果汇总:")
        print(json.dumps(results, ensure_ascii=False, indent=2))

        # 保存截图用于分析
        driver.save_screenshot("D:\\AI-FINANCE\\citic_history_debug.png")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if driver:
            driver.quit()
            print("\n浏览器已关闭")

if __name__ == "__main__":
    main()
