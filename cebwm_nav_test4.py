# -*- coding: utf-8 -*-
"""
光大理财净值获取测试 v4 - 详细分析点击后的DOM变化
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

    # 记录点击前的DOM
    print("\n=== 点击前状态 ===")
    body_before = len(driver.find_element(By.TAG_NAME, 'body').get_attribute('innerHTML'))
    divs_before = len(driver.find_elements(By.TAG_NAME, 'div'))
    print(f"Body长度: {body_before}, Div数量: {divs_before}")

    # 找到产品表格和详细按钮
    tables = driver.find_elements(By.TAG_NAME, 'table')
    product_table = tables[2]
    rows = product_table.find_elements(By.TAG_NAME, 'tr')
    data_row = rows[1]
    cells = data_row.find_elements(By.TAG_NAME, 'td')

    product_name = cells[0].text.strip()
    product_code = cells[1].text.strip()
    print(f"产品: {product_name} ({product_code})")

    operation_cell = cells[-1]
    detail_links = operation_cell.find_elements(By.TAG_NAME, 'a')
    detail_btn = None
    for link in detail_links:
        if '详' in link.text:
            detail_btn = link
            break

    if not detail_btn:
        print("未找到详细按钮")
        return

    # 获取详细按钮的href和onclick
    href = detail_btn.get_attribute('href')
    onclick = detail_btn.get_attribute('onclick')
    print(f"详细按钮 href: {href}")
    print(f"详细按钮 onclick: {onclick}")

    # 获取按钮的完整HTML
    print(f"按钮HTML: {detail_btn.get_attribute('outerHTML')}")

    # 点击
    print("\n点击详细...")
    main_window = driver.current_window_handle
    windows_before = driver.window_handles[:]

    driver.execute_script("arguments[0].click();", detail_btn)
    time.sleep(3)

    # 检查变化
    print("\n=== 点击后状态 ===")

    # 新窗口？
    windows_after = driver.window_handles
    if len(windows_after) > len(windows_before):
        new_windows = [w for w in windows_after if w not in windows_before]
        print(f"新窗口: {new_windows}")
        driver.switch_to.window(new_windows[0])
        print(f"新窗口URL: {driver.current_url}")
        print(f"新窗口标题: {driver.title}")
    else:
        print("没有新窗口")

    # DOM变化？
    body_after = len(driver.find_element(By.TAG_NAME, 'body').get_attribute('innerHTML'))
    divs_after = len(driver.find_elements(By.TAG_NAME, 'div'))
    print(f"Body长度: {body_after} (变化: {body_after - body_before})")
    print(f"Div数量: {divs_after} (变化: {divs_after - divs_before})")

    # 查找弹窗/模态框
    print("\n=== 查找弹窗 ===")

    modal_selectors = [
        '.modal', '.dialog', '.popup', '.layer', '.mask',
        '[class*="modal"]', '[class*="dialog"]', '[class*="popup"]',
        '[class*="layer"]', '[class*="mask"]', '[class*="overlay"]',
        '[style*="z-index"]', '[style*="position: fixed"]',
        '.el-dialog', '.ant-modal', '.ivu-modal',
        '#layui-layer1', '[id*="layer"]', '[id*="popup"]'
    ]

    for sel in modal_selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            visible = [e for e in els if e.is_displayed()]
            if visible:
                print(f"\n{sel}: 找到 {len(visible)} 个可见元素")
                for el in visible[:2]:
                    html = el.get_attribute('outerHTML')
                    print(f"  HTML片段: {html[:500]}")
        except:
            pass

    # 查找包含"净值"的新元素
    print("\n=== 查找净值相关 ===")
    nav_els = driver.find_elements(By.XPATH, "//*[contains(text(), '净值')]")
    for el in nav_els:
        try:
            if el.is_displayed():
                print(f"  {el.tag_name}: {el.text[:50]}")
        except:
            pass

    # 查找iframe
    print("\n=== 查找iframe ===")
    iframes = driver.find_elements(By.TAG_NAME, 'iframe')
    print(f"找到 {len(iframes)} 个iframe")
    for i, iframe in enumerate(iframes):
        src = iframe.get_attribute('src')
        print(f"  iframe {i}: {src}")
        if iframe.is_displayed():
            print("    (可见)")
            # 切换到iframe
            try:
                driver.switch_to.frame(iframe)
                iframe_body = driver.find_element(By.TAG_NAME, 'body').text[:500]
                print(f"    内容: {iframe_body}")
                driver.switch_to.default_content()
            except:
                pass

    print("\n完成")


if __name__ == "__main__":
    test()
