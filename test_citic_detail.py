# -*- coding: utf-8 -*-
"""测试获取中信理财产品详情页的历史净值"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
import sys

# 设置控制台输出编码
sys.stdout.reconfigure(encoding='utf-8')

def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=options)

def main():
    print("=" * 60)
    print("测试获取中信理财产品详情的历史净值")
    print("=" * 60)

    driver = None

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

        # 找到第一个有净值的产品(不是募集期的)
        print("\n4. 查找有净值的产品...")
        product_items = driver.find_elements(By.CSS_SELECTOR, '.list-item')

        found_product = None
        for i, item in enumerate(product_items[:10]):
            try:
                name_el = item.find_element(By.CSS_SELECTOR, '.item-productName')
                name = name_el.text.strip()

                # 检查是否有净值数据
                nav_el = item.find_elements(By.CSS_SELECTOR, '.list-item-content-right')
                nav_text = nav_el[0].text if nav_el else ''

                print(f"   产品{i+1}: {name[:30]}... 净值: {nav_text[:20]}")

                # 如果净值是数字（不是--或空）,说明不是募集期
                if nav_text and nav_text[0].isdigit():
                    found_product = item
                    print(f"   >>> 找到有净值的产品!")
                    break
            except Exception as e:
                continue

        if not found_product:
            print("   未找到有净值的产品，使用第一个产品")
            found_product = product_items[0] if product_items else None

        if not found_product:
            print("   未找到任何产品!")
            return

        # 点击产品名称打开详情
        print("\n5. 点击产品查看详情...")
        name_el = found_product.find_element(By.CSS_SELECTOR, '.item-productName')
        product_name = name_el.text.strip()
        print(f"   产品名称: {product_name}")

        name_el.click()
        time.sleep(3)

        # 分析弹出的详情页
        print("\n6. 分析详情页内容...")

        # 查找详情弹窗
        detail_modal = driver.find_elements(By.CSS_SELECTOR, '.product-detail, .modal, .popup, .detail-container, [class*="detail"]')
        print(f"   找到 {len(detail_modal)} 个可能的详情容器")

        # 打印页面HTML结构(部分)
        page_html = driver.page_source

        # 查找包含"历史"或"净值"的元素
        print("\n7. 查找历史净值相关内容...")

        # 尝试查找各种可能的历史净值元素
        selectors_to_try = [
            "//div[contains(text(), '历史')]",
            "//div[contains(text(), '净值走势')]",
            "//div[contains(text(), '历史业绩')]",
            "//span[contains(text(), '历史')]",
            "//a[contains(text(), '历史')]",
            ".history-nav",
            ".nav-history",
            "[class*='history']",
            "[class*='chart']",
            "canvas",  # 可能有图表
            "table",
        ]

        for sel in selectors_to_try:
            try:
                if sel.startswith('//'):
                    elements = driver.find_elements(By.XPATH, sel)
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, sel)
                if elements:
                    print(f"   {sel}: 找到 {len(elements)} 个元素")
                    for el in elements[:3]:
                        text = el.text[:100] if el.text else ''
                        print(f"      - {text}")
            except:
                pass

        # 检查是否有tab切换
        print("\n8. 查找tab/按钮...")
        buttons = driver.find_elements(By.CSS_SELECTOR, 'button, .tab, [role="tab"], .btn')
        for btn in buttons[:20]:
            text = btn.text.strip()
            if text:
                print(f"   按钮: {text}")
                if '历史' in text or '净值' in text or '走势' in text:
                    print(f"   >>> 尝试点击: {text}")
                    try:
                        btn.click()
                        time.sleep(2)
                    except:
                        pass

        # 截图保存以便分析
        print("\n9. 截图保存...")
        driver.save_screenshot("D:\\AI-FINANCE\\citic_detail_screenshot.png")
        print("   已保存到 citic_detail_screenshot.png")

        # 检查Network请求中的API
        print("\n10. 检查页面中的数据...")

        # 在页面源码中查找JSON数据
        if 'navHistory' in page_html or 'historyNav' in page_html:
            print("   页面包含历史净值相关字段!")

        # 查找可能包含净值数据的script标签
        scripts = driver.find_elements(By.TAG_NAME, 'script')
        for script in scripts:
            src = script.get_attribute('src') or ''
            inner = script.get_attribute('innerHTML') or ''
            if 'nav' in inner.lower() or 'history' in inner.lower():
                print(f"   发现可能包含净值数据的script: {src[:50] if src else inner[:100]}")

        # 打印当前页面body文本(查找净值数据)
        print("\n11. 页面文本(查找净值相关)...")
        body = driver.find_element(By.TAG_NAME, 'body')
        body_text = body.text

        # 查找净值相关的行
        lines = body_text.split('\n')
        for line in lines:
            if '净值' in line or '历史' in line or any(c.isdigit() and '.' in line for c in line):
                if len(line) < 200:
                    print(f"   {line}")

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
