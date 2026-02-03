# -*- coding: utf-8 -*-
"""
光大理财净值获取测试 v2
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json


def test_nav_fetch():
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

    time.sleep(2)

    # 打印整个表格结构
    print("\n=== 分析表格结构 ===")

    tables = driver.find_elements(By.TAG_NAME, 'table')
    print(f"找到 {len(tables)} 个表格")

    for i, table in enumerate(tables):
        try:
            rows = table.find_elements(By.TAG_NAME, 'tr')
            print(f"\n表格 {i}: {len(rows)} 行")

            # 打印表头
            headers = table.find_elements(By.TAG_NAME, 'th')
            if headers:
                header_texts = [h.text.strip()[:15] for h in headers]
                print(f"  表头: {header_texts}")

            # 打印前3行
            for j, row in enumerate(rows[:3]):
                cells = row.find_elements(By.TAG_NAME, 'td')
                if cells:
                    cell_texts = [c.text.strip()[:20] for c in cells]
                    print(f"  行{j}: {cell_texts}")
        except Exception as e:
            print(f"  错误: {e}")

    # 查找包含"详"字的所有元素
    print("\n=== 查找详细按钮 ===")

    detail_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '详')]")
    print(f"找到 {len(detail_elements)} 个包含'详'的元素")

    for el in detail_elements[:20]:
        try:
            if el.is_displayed():
                tag = el.tag_name
                text = el.text[:30]
                onclick = el.get_attribute('onclick') or ''
                href = el.get_attribute('href') or ''
                cls = el.get_attribute('class') or ''
                print(f"  {tag}: '{text}' class={cls[:30]} onclick={onclick[:50]}")
        except:
            pass

    # 查找所有按钮
    print("\n=== 所有按钮 ===")
    buttons = driver.find_elements(By.TAG_NAME, 'button')
    for btn in buttons[:10]:
        try:
            if btn.is_displayed():
                print(f"  button: {btn.text[:30]} class={btn.get_attribute('class')[:30]}")
        except:
            pass

    # 查找所有带onclick的元素
    print("\n=== 带onclick的元素 ===")
    onclick_els = driver.find_elements(By.CSS_SELECTOR, '[onclick]')
    for el in onclick_els[:20]:
        try:
            if el.is_displayed():
                onclick = el.get_attribute('onclick')
                if 'detail' in onclick.lower() or 'xq' in onclick.lower() or 'view' in onclick.lower():
                    print(f"  {el.tag_name}: {el.text[:20]} - {onclick[:80]}")
        except:
            pass

    # 打印第一个产品行的完整HTML
    print("\n=== 第一个产品行完整HTML ===")
    try:
        first_row = driver.find_element(By.CSS_SELECTOR, 'table tbody tr')
        print(first_row.get_attribute('outerHTML')[:2000])
    except Exception as e:
        print(f"错误: {e}")

    # 尝试找到并点击第一个"详细"按钮
    print("\n=== 尝试点击详细 ===")

    # 多种选择器
    selectors = [
        "//a[contains(text(), '详细')]",
        "//span[contains(text(), '详细')]",
        "//button[contains(text(), '详细')]",
        "//*[contains(@onclick, 'detail')]",
        "//*[contains(@onclick, 'Detail')]",
        "//a[contains(text(), '查看')]",
        ".ui_rt a",
        "table tbody tr td:last-child a",
    ]

    for selector in selectors:
        try:
            if selector.startswith('//'):
                els = driver.find_elements(By.XPATH, selector)
            else:
                els = driver.find_elements(By.CSS_SELECTOR, selector)

            if els:
                print(f"\n选择器 '{selector}' 找到 {len(els)} 个元素")
                for el in els[:3]:
                    if el.is_displayed():
                        print(f"  文本: {el.text[:30]}, onclick: {el.get_attribute('onclick')}")
        except Exception as e:
            pass

    print("\n完成分析")


if __name__ == "__main__":
    test_nav_fetch()
