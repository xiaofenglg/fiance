# -*- coding: utf-8 -*-
"""
光大理财净值获取测试 v3 - 点击详细按钮获取净值
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


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

    # 找到产品表格（第3个表格，索引2）
    tables = driver.find_elements(By.TAG_NAME, 'table')
    print(f"找到 {len(tables)} 个表格")

    if len(tables) < 3:
        print("未找到产品表格")
        return

    product_table = tables[2]

    # 获取产品行
    rows = product_table.find_elements(By.TAG_NAME, 'tr')
    print(f"产品表格有 {len(rows)} 行")

    # 第一行是表头，从第二行开始是数据
    if len(rows) < 2:
        print("无产品数据")
        return

    data_row = rows[1]  # 第一个产品
    cells = data_row.find_elements(By.TAG_NAME, 'td')

    product_name = cells[0].text.strip()
    product_code = cells[1].text.strip()
    print(f"\n测试产品: {product_name} ({product_code})")

    # 最后一个单元格是操作列
    operation_cell = cells[-1]
    print(f"操作列内容: {operation_cell.text}")

    # 找到详细按钮
    detail_links = operation_cell.find_elements(By.TAG_NAME, 'a')
    print(f"找到 {len(detail_links)} 个链接")

    detail_btn = None
    for link in detail_links:
        if '详' in link.text:
            detail_btn = link
            print(f"找到详细按钮: {link.text}")
            break

    if not detail_btn:
        print("未找到详细按钮")
        return

    # 记录当前窗口
    main_window = driver.current_window_handle
    windows_before = len(driver.window_handles)

    # 点击详细
    print("\n点击详细按钮...")
    try:
        driver.execute_script("arguments[0].click();", detail_btn)
    except Exception as e:
        print(f"点击错误: {e}")
        return

    time.sleep(3)

    # 检查是否打开新窗口
    if len(driver.window_handles) > windows_before:
        print("打开了新窗口")
        new_window = [h for h in driver.window_handles if h != main_window][0]
        driver.switch_to.window(new_window)
        print(f"新窗口URL: {driver.current_url}")
        print(f"新窗口标题: {driver.title}")
    else:
        print("检查弹窗...")

    # 等待详情页加载
    time.sleep(2)

    # 查找"产品净值"按钮/标签
    print("\n查找产品净值...")

    nav_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '产品净值')]")
    print(f"找到 {len(nav_elements)} 个包含'产品净值'的元素")

    for el in nav_elements:
        try:
            if el.is_displayed():
                print(f"  {el.tag_name}: {el.text}, class={el.get_attribute('class')}")
        except:
            pass

    # 如果找到，点击
    nav_btn = None
    for el in nav_elements:
        try:
            if el.is_displayed() and el.text.strip() == '产品净值':
                nav_btn = el
                break
        except:
            pass

    if nav_btn:
        print(f"\n点击产品净值按钮...")
        driver.execute_script("arguments[0].click();", nav_btn)
        time.sleep(2)

        # 提取净值表格
        print("\n查找净值数据...")

        # 查找所有表格
        tables = driver.find_elements(By.TAG_NAME, 'table')
        print(f"页面有 {len(tables)} 个表格")

        for i, table in enumerate(tables):
            rows = table.find_elements(By.TAG_NAME, 'tr')
            if len(rows) > 5:  # 净值表应该有多行
                print(f"\n表格 {i} ({len(rows)} 行):")
                for row in rows[:25]:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if not cells:
                        cells = row.find_elements(By.TAG_NAME, 'th')
                    cell_texts = [c.text.strip() for c in cells]
                    if any(cell_texts):
                        print(f"  {cell_texts}")

        # 查找包含日期的元素
        print("\n查找日期数据...")
        date_els = driver.find_elements(By.XPATH, "//*[contains(text(), '2026-') or contains(text(), '2025-')]")
        for el in date_els[:30]:
            try:
                parent = el.find_element(By.XPATH, "./..")
                print(f"  {el.text} | 父元素: {parent.text[:50] if parent.text else ''}")
            except:
                print(f"  {el.text}")

    else:
        print("未找到产品净值按钮，打印页面内容...")

        # 打印页面所有可点击元素
        clickables = driver.find_elements(By.CSS_SELECTOR, 'a, button, [onclick], .tab, [class*="tab"]')
        print(f"\n可点击元素 ({len(clickables)}):")
        for c in clickables[:30]:
            try:
                if c.is_displayed() and c.text.strip():
                    print(f"  {c.tag_name}: {c.text[:40]}")
            except:
                pass

    print("\n测试完成")
    return driver


if __name__ == "__main__":
    test_nav_fetch()
