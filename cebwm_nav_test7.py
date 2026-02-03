# -*- coding: utf-8 -*-
"""
光大理财净值获取 v7 - 查找左侧净值相关按钮
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

    # 查找所有包含"净值"的元素
    print("\n=== 所有包含'净值'的元素 ===")

    nav_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '净值')]")
    print(f"找到 {len(nav_elements)} 个")

    for i, el in enumerate(nav_elements):
        try:
            if el.is_displayed():
                tag = el.tag_name
                text = el.text.strip()[:50]
                cls = el.get_attribute('class') or ''
                href = el.get_attribute('href') or ''
                onclick = el.get_attribute('onclick') or ''
                location = el.location

                print(f"\n{i}: <{tag}> 位置=({location['x']}, {location['y']})")
                print(f"   文本: '{text}'")
                print(f"   class: '{cls[:50]}'")
                if href:
                    print(f"   href: '{href[:80]}'")
                if onclick:
                    print(f"   onclick: '{onclick[:80]}'")
        except:
            pass

    # 查找产品名称列（可点击查看详情）
    print("\n\n=== 检查产品名称是否可点击 ===")

    tables = driver.find_elements(By.TAG_NAME, 'table')
    if len(tables) >= 3:
        product_table = tables[2]
        rows = product_table.find_elements(By.TAG_NAME, 'tr')

        if len(rows) > 1:
            data_row = rows[1]
            cells = data_row.find_elements(By.TAG_NAME, 'td')

            if cells:
                # 产品名称单元格
                name_cell = cells[0]
                print(f"产品名称单元格HTML:\n{name_cell.get_attribute('outerHTML')}")

                # 查找产品名称里的链接
                name_links = name_cell.find_elements(By.TAG_NAME, 'a')
                print(f"\n产品名称中的链接数: {len(name_links)}")

                for link in name_links:
                    href = link.get_attribute('href')
                    onclick = link.get_attribute('onclick')
                    print(f"  链接: {link.text}")
                    print(f"  href: {href}")
                    print(f"  onclick: {onclick}")

    # 查找表头中的"单位净值"列，看是否可点击查看历史
    print("\n\n=== 检查净值列是否可点击 ===")

    if len(tables) >= 3:
        product_table = tables[2]
        headers = product_table.find_elements(By.TAG_NAME, 'th')

        for th in headers:
            text = th.text.strip()
            if '净值' in text:
                print(f"表头: {text}")
                print(f"  HTML: {th.get_attribute('outerHTML')[:200]}")

    # 保存当前页面截图和HTML用于分析
    print("\n\n=== 保存页面用于分析 ===")

    # 保存HTML
    html = driver.page_source
    with open('D:/AI-FINANCE/cebwm_page_full.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("页面已保存到 cebwm_page_full.html")

    print("\n完成")


if __name__ == "__main__":
    test()
