# -*- coding: utf-8 -*-
"""
光大理财净值获取测试 v5 - 模拟真实点击
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
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

    # 找到详细按钮
    tables = driver.find_elements(By.TAG_NAME, 'table')
    product_table = tables[2]
    rows = product_table.find_elements(By.TAG_NAME, 'tr')
    data_row = rows[1]
    cells = data_row.find_elements(By.TAG_NAME, 'td')

    product_name = cells[0].text.strip()
    print(f"产品: {product_name}")

    # 打印操作列的所有链接
    operation_cell = cells[-1]
    print(f"\n操作列HTML:\n{operation_cell.get_attribute('innerHTML')}")

    all_links = operation_cell.find_elements(By.TAG_NAME, 'a')
    print(f"\n操作列有 {len(all_links)} 个链接:")
    for i, link in enumerate(all_links):
        print(f"  {i}: {link.text} - class={link.get_attribute('class')}")

    # 找详细按钮
    detail_btn = None
    for link in all_links:
        if '详' in link.text:
            detail_btn = link
            break

    if not detail_btn:
        print("\n未找到详细按钮，检查整行的所有链接...")
        all_row_links = data_row.find_elements(By.TAG_NAME, 'a')
        for link in all_row_links:
            print(f"  行内链接: {link.text}")
        return

    print(f"\n找到详细按钮")

    # 尝试多种点击方式
    print("\n=== 尝试不同点击方式 ===")

    main_window = driver.current_window_handle

    # 方式1: ActionChains模拟真实点击
    print("\n1. ActionChains点击...")
    try:
        actions = ActionChains(driver)
        actions.move_to_element(detail_btn)
        actions.pause(0.5)
        actions.click()
        actions.perform()
        time.sleep(2)
        if len(driver.window_handles) > 1:
            print("   成功！打开了新窗口")
            new_win = [w for w in driver.window_handles if w != main_window][0]
            driver.switch_to.window(new_win)
            print(f"   URL: {driver.current_url}")
            # 继续处理
            process_detail_page(driver)
            return
    except Exception as e:
        print(f"   失败: {e}")

    # 方式2: 触发mousedown/mouseup事件
    print("\n2. 触发鼠标事件...")
    try:
        driver.execute_script("""
            var element = arguments[0];
            var event = new MouseEvent('mousedown', {
                bubbles: true,
                cancelable: true,
                view: window
            });
            element.dispatchEvent(event);

            event = new MouseEvent('mouseup', {
                bubbles: true,
                cancelable: true,
                view: window
            });
            element.dispatchEvent(event);

            event = new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window
            });
            element.dispatchEvent(event);
        """, detail_btn)
        time.sleep(2)
        if len(driver.window_handles) > 1:
            print("   成功！")
    except Exception as e:
        print(f"   失败: {e}")

    # 方式3: 直接访问产品详情URL
    print("\n3. 尝试构造产品详情URL...")
    product_code = cells[1].text.strip()
    possible_urls = [
        f"https://www.cebwm.com/wealth/grlc/productDetail.html?productCode={product_code}",
        f"https://www.cebwm.com/wealth/grlc/product/{product_code}",
        f"https://www.cebwm.com/wealth/grlc/detail.html?code={product_code}",
        f"https://www.cebwm.com/wealth/grlc/lccp/{product_code}.html",
    ]

    for url in possible_urls:
        print(f"   尝试: {url}")
        try:
            driver.execute_script(f"window.open('{url}', '_blank');")
            time.sleep(2)
            if len(driver.window_handles) > 1:
                new_win = [w for w in driver.window_handles if w != main_window][-1]
                driver.switch_to.window(new_win)
                if 'cebwm' in driver.current_url and len(driver.page_source) > 1000:
                    print(f"   成功加载: {driver.current_url}")
                    print(f"   页面标题: {driver.title}")
                    print(f"   页面长度: {len(driver.page_source)}")

                    # 检查是否有净值内容
                    if '净值' in driver.page_source:
                        print("   包含净值数据！")
                        process_detail_page(driver)
                        return
                else:
                    print("   页面加载失败或为空")
                driver.close()
                driver.switch_to.window(main_window)
        except Exception as e:
            print(f"   错误: {e}")

    # 方式4: 查看页面的JS函数
    print("\n4. 查找页面JS函数...")
    js_funcs = driver.execute_script("""
        var funcs = [];
        for (var key in window) {
            if (typeof window[key] === 'function') {
                var name = key.toLowerCase();
                if (name.includes('detail') || name.includes('xq') ||
                    name.includes('product') || name.includes('view') ||
                    name.includes('show') || name.includes('open')) {
                    funcs.push(key);
                }
            }
        }
        return funcs;
    """)
    print(f"   相关函数: {js_funcs[:20]}")

    # 尝试调用这些函数
    product_code = cells[1].text.strip()
    for func in js_funcs[:5]:
        try:
            print(f"\n   尝试调用 {func}('{product_code}')...")
            driver.execute_script(f"if(typeof {func} === 'function') {func}('{product_code}');")
            time.sleep(1)
            if len(driver.window_handles) > 1:
                print("   触发了新窗口！")
                break
        except:
            pass

    print("\n所有尝试完成")


def process_detail_page(driver):
    """处理详情页，获取净值数据"""
    print("\n=== 处理详情页 ===")

    # 查找产品净值标签
    time.sleep(2)

    print("查找产品净值按钮/标签...")
    nav_tabs = driver.find_elements(By.XPATH, "//*[contains(text(), '产品净值')]")
    print(f"找到 {len(nav_tabs)} 个产品净值元素")

    for tab in nav_tabs:
        try:
            if tab.is_displayed():
                print(f"  点击: {tab.text}")
                driver.execute_script("arguments[0].click();", tab)
                time.sleep(2)
                break
        except:
            pass

    # 提取净值数据
    print("\n查找净值表格...")
    tables = driver.find_elements(By.TAG_NAME, 'table')
    print(f"页面有 {len(tables)} 个表格")

    for i, table in enumerate(tables):
        rows = table.find_elements(By.TAG_NAME, 'tr')
        if len(rows) >= 5:
            print(f"\n表格 {i} ({len(rows)} 行):")
            for j, row in enumerate(rows[:25]):
                cells = row.find_elements(By.TAG_NAME, 'td')
                if not cells:
                    cells = row.find_elements(By.TAG_NAME, 'th')
                texts = [c.text.strip() for c in cells]
                if any(texts):
                    print(f"  {texts}")


if __name__ == "__main__":
    test()
