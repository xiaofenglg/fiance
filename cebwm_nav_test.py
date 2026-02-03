# -*- coding: utf-8 -*-
"""
光大理财净值获取测试

测试点击详细->产品净值获取历史净值数据
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json


def test_nav_fetch():
    """测试获取净值数据"""
    print("连接Chrome浏览器...")

    options = Options()
    options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    driver = webdriver.Chrome(options=options)

    # 切换到光大理财页面
    for handle in driver.window_handles:
        driver.switch_to.window(handle)
        if 'cebwm' in driver.current_url:
            print(f"已连接: {driver.current_url}")
            break

    # 等待表格加载
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
    )

    # 获取第一个产品行
    rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')
    print(f"找到 {len(rows)} 行")

    if not rows:
        print("未找到产品行")
        return

    # 获取第一个产品的名称
    first_row = rows[0]
    cells = first_row.find_elements(By.TAG_NAME, 'td')
    product_name = cells[0].text.strip() if cells else "未知"
    product_code = cells[1].text.strip() if len(cells) > 1 else "未知"
    print(f"\n测试产品: {product_name} ({product_code})")

    # 查找"详细"按钮
    print("\n查找详细按钮...")

    # 打印最后一列（操作列）的内容
    if cells:
        last_cell = cells[-1]
        print(f"操作列HTML: {last_cell.get_attribute('outerHTML')[:500]}")

        # 查找详细链接/按钮
        detail_links = last_cell.find_elements(By.TAG_NAME, 'a')
        detail_buttons = last_cell.find_elements(By.TAG_NAME, 'button')
        detail_spans = last_cell.find_elements(By.TAG_NAME, 'span')

        print(f"找到链接: {len(detail_links)}, 按钮: {len(detail_buttons)}, span: {len(detail_spans)}")

        for link in detail_links:
            print(f"  链接: {link.text} - onclick: {link.get_attribute('onclick')}")

        # 尝试点击详细
        detail_btn = None
        for link in detail_links:
            if '详' in link.text or 'detail' in link.text.lower():
                detail_btn = link
                break

        if not detail_btn:
            # 尝试通过文本查找
            try:
                detail_btn = last_cell.find_element(By.XPATH, ".//*[contains(text(), '详')]")
            except:
                pass

        if detail_btn:
            print(f"\n找到详细按钮: {detail_btn.text}")
            print("点击详细...")

            # 记录当前窗口数
            windows_before = len(driver.window_handles)

            # 点击
            try:
                detail_btn.click()
            except:
                driver.execute_script("arguments[0].click();", detail_btn)

            time.sleep(2)

            # 检查是否打开新窗口
            if len(driver.window_handles) > windows_before:
                print("打开了新窗口")
                driver.switch_to.window(driver.window_handles[-1])
                print(f"新窗口URL: {driver.current_url}")
            else:
                print("未打开新窗口，可能是弹窗/模态框")

            # 查找弹窗/模态框
            print("\n查找弹窗内容...")

            # 等待弹窗出现
            time.sleep(1)

            # 查找可能的弹窗容器
            modal_selectors = [
                '.modal', '.dialog', '.popup', '.layer',
                '[class*="modal"]', '[class*="dialog"]', '[class*="popup"]',
                '.el-dialog', '.ant-modal', '.ivu-modal'
            ]

            modal = None
            for selector in modal_selectors:
                try:
                    modals = driver.find_elements(By.CSS_SELECTOR, selector)
                    for m in modals:
                        if m.is_displayed():
                            modal = m
                            print(f"找到弹窗: {selector}")
                            break
                except:
                    pass
                if modal:
                    break

            if modal:
                print(f"弹窗HTML: {modal.get_attribute('outerHTML')[:1000]}")

                # 查找"产品净值"按钮/标签
                print("\n查找产品净值按钮...")

                nav_btn = None
                try:
                    nav_btn = modal.find_element(By.XPATH, ".//*[contains(text(), '产品净值')]")
                except:
                    pass

                if not nav_btn:
                    try:
                        nav_btn = modal.find_element(By.XPATH, ".//*[contains(text(), '净值')]")
                    except:
                        pass

                if nav_btn:
                    print(f"找到净值按钮: {nav_btn.text}")
                    print("点击产品净值...")

                    try:
                        nav_btn.click()
                    except:
                        driver.execute_script("arguments[0].click();", nav_btn)

                    time.sleep(2)

                    # 提取净值数据
                    print("\n提取净值数据...")

                    # 查找净值表格
                    nav_tables = modal.find_elements(By.TAG_NAME, 'table')
                    print(f"找到 {len(nav_tables)} 个表格")

                    for table in nav_tables:
                        rows = table.find_elements(By.TAG_NAME, 'tr')
                        print(f"\n表格行数: {len(rows)}")
                        for row in rows[:25]:  # 显示前25行
                            cells = row.find_elements(By.TAG_NAME, 'td')
                            if not cells:
                                cells = row.find_elements(By.TAG_NAME, 'th')
                            cell_texts = [c.text.strip() for c in cells]
                            if any(cell_texts):
                                print(f"  {cell_texts}")

                    # 也查找列表形式的净值
                    nav_items = modal.find_elements(By.XPATH, ".//*[contains(text(), '2026') or contains(text(), '2025')]")
                    if nav_items:
                        print(f"\n找到日期相关元素: {len(nav_items)}")
                        for item in nav_items[:10]:
                            print(f"  {item.text}")

                else:
                    print("未找到产品净值按钮")
                    # 打印弹窗中所有可点击元素
                    clickables = modal.find_elements(By.CSS_SELECTOR, 'a, button, [onclick], .tab, [class*="tab"]')
                    print(f"\n弹窗中的可点击元素 ({len(clickables)}):")
                    for c in clickables:
                        print(f"  {c.tag_name}: {c.text[:50] if c.text else c.get_attribute('class')}")

            else:
                print("未找到弹窗，查找页面变化...")
                # 可能是在同一页面展开了详情
                body_html = driver.find_element(By.TAG_NAME, 'body').get_attribute('innerHTML')
                if '产品净值' in body_html:
                    print("页面中包含'产品净值'")

                # 查找所有包含"净值"的元素
                nav_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '净值')]")
                print(f"\n包含'净值'的元素: {len(nav_elements)}")
                for el in nav_elements[:10]:
                    if el.is_displayed():
                        print(f"  {el.tag_name}: {el.text[:50]}")

        else:
            print("未找到详细按钮")
            # 打印操作列所有内容
            print(f"操作列文本: {last_cell.text}")

    print("\n测试完成")
    return driver


if __name__ == "__main__":
    driver = test_nav_fetch()
    input("\n按Enter键退出...")
