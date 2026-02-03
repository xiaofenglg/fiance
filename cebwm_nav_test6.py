# -*- coding: utf-8 -*-
"""
光大理财净值获取 - 直接点击左侧"产品净值"按钮
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time


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

    # 查找页面上所有包含"产品净值"的元素
    print("\n=== 查找产品净值按钮 ===")

    nav_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '产品净值')]")
    print(f"找到 {len(nav_elements)} 个包含'产品净值'的元素")

    for i, el in enumerate(nav_elements):
        try:
            tag = el.tag_name
            text = el.text.strip()
            cls = el.get_attribute('class') or ''
            displayed = el.is_displayed()
            print(f"  {i}: <{tag}> '{text[:30]}' class='{cls[:30]}' 可见={displayed}")
        except Exception as e:
            print(f"  {i}: 错误 {e}")

    # 查找左侧菜单
    print("\n=== 查找左侧菜单 ===")

    # 常见的左侧菜单选择器
    left_menu_selectors = [
        '.left-menu', '.sidebar', '.nav-left', '.menu-left',
        '[class*="left"]', '[class*="menu"]', '[class*="nav"]',
        '.lccp_l', '.lccplist_l'  # 光大理财特有的class
    ]

    for sel in left_menu_selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in els:
                if el.is_displayed():
                    text = el.text[:200]
                    if '净值' in text:
                        print(f"\n{sel}: 包含净值的菜单")
                        print(f"  内容: {text}")
        except:
            pass

    # 直接查找并点击产品净值
    print("\n=== 尝试点击产品净值 ===")

    # 尝试多种选择器
    selectors = [
        "//a[contains(text(), '产品净值')]",
        "//span[contains(text(), '产品净值')]",
        "//div[contains(text(), '产品净值')]",
        "//li[contains(text(), '产品净值')]",
        "//*[text()='产品净值']",
        "//a[text()='产品净值']",
    ]

    for sel in selectors:
        try:
            els = driver.find_elements(By.XPATH, sel)
            for el in els:
                if el.is_displayed():
                    print(f"\n找到可见的产品净值按钮: {el.tag_name}")
                    print(f"  文本: {el.text}")
                    print(f"  class: {el.get_attribute('class')}")

                    # 点击
                    print("  点击...")
                    actions = ActionChains(driver)
                    actions.move_to_element(el)
                    actions.pause(0.3)
                    actions.click()
                    actions.perform()

                    time.sleep(3)

                    # 检查结果
                    print("\n=== 点击后查找净值数据 ===")

                    # 查找表格
                    tables = driver.find_elements(By.TAG_NAME, 'table')
                    print(f"页面有 {len(tables)} 个表格")

                    for j, table in enumerate(tables):
                        rows = table.find_elements(By.TAG_NAME, 'tr')
                        if len(rows) >= 5:
                            print(f"\n表格 {j} ({len(rows)} 行):")
                            for row in rows[:25]:
                                cells = row.find_elements(By.TAG_NAME, 'td')
                                if not cells:
                                    cells = row.find_elements(By.TAG_NAME, 'th')
                                texts = [c.text.strip() for c in cells]
                                if any(texts):
                                    print(f"  {texts}")

                    # 查找包含日期的数据
                    print("\n查找日期数据...")
                    date_els = driver.find_elements(By.XPATH, "//*[contains(text(), '2026-') or contains(text(), '2025-')]")
                    print(f"找到 {len(date_els)} 个日期元素:")
                    for el in date_els[:30]:
                        try:
                            print(f"  {el.text[:80]}")
                        except:
                            pass

                    return

        except Exception as e:
            print(f"  选择器 {sel} 错误: {e}")

    print("\n未找到可点击的产品净值按钮")

    # 打印页面的左侧区域内容
    print("\n=== 页面左侧内容 ===")
    try:
        # 获取页面左侧1/3的元素
        left_divs = driver.find_elements(By.CSS_SELECTOR, 'div')
        for div in left_divs:
            try:
                location = div.location
                size = div.size
                if location['x'] < 300 and size['width'] > 50 and size['height'] > 20:
                    text = div.text.strip()
                    if text and len(text) < 200:
                        print(f"  x={location['x']}: {text[:100]}")
            except:
                pass
    except Exception as e:
        print(f"  错误: {e}")


if __name__ == "__main__":
    test()
