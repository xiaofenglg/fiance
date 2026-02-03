# -*- coding: utf-8 -*-
"""
光大理财净值获取 v8 - 提取详情页URL并获取净值
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import re


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
    main_window = driver.current_window_handle

    # 找到产品行中的详情链接
    print("\n=== 查找详情链接 ===")

    # 查找所有包含"详情"的链接
    detail_links = driver.find_elements(By.XPATH, "//a[contains(text(), '详情')]")
    print(f"找到 {len(detail_links)} 个详情链接")

    if detail_links:
        first_link = detail_links[0]
        href = first_link.get_attribute('href')
        print(f"第一个详情链接 href: {href}")

        # 获取产品名称（同一行的第一个单元格）
        try:
            row = first_link.find_element(By.XPATH, "./ancestor::tr")
            cells = row.find_elements(By.TAG_NAME, 'td')
            if cells:
                product_name = cells[0].text.strip()
                product_code = cells[1].text.strip() if len(cells) > 1 else ""
                print(f"产品: {product_name} ({product_code})")
        except:
            pass

        # 直接访问详情页
        if href and href != 'javascript: void(0);':
            print(f"\n访问详情页: {href}")
            driver.execute_script(f"window.open('{href}', '_blank');")
            time.sleep(3)

            # 切换到新窗口
            if len(driver.window_handles) > 1:
                new_window = [h for h in driver.window_handles if h != main_window][0]
                driver.switch_to.window(new_window)
                print(f"新窗口URL: {driver.current_url}")
                print(f"页面标题: {driver.title}")

                # 等待页面加载
                time.sleep(2)

                # 查找产品净值标签/按钮
                print("\n=== 查找产品净值标签 ===")

                nav_tabs = driver.find_elements(By.XPATH, "//*[contains(text(), '产品净值')]")
                print(f"找到 {len(nav_tabs)} 个产品净值元素")

                for tab in nav_tabs:
                    try:
                        if tab.is_displayed():
                            print(f"  {tab.tag_name}: {tab.text}")
                            print(f"  class: {tab.get_attribute('class')}")

                            # 点击
                            print("  点击...")
                            driver.execute_script("arguments[0].click();", tab)
                            time.sleep(2)
                            break
                    except:
                        pass

                # 查找净值表格数据
                print("\n=== 查找净值数据 ===")

                # 查找表格
                tables = driver.find_elements(By.TAG_NAME, 'table')
                print(f"页面有 {len(tables)} 个表格")

                for i, table in enumerate(tables):
                    rows = table.find_elements(By.TAG_NAME, 'tr')
                    if len(rows) >= 3:
                        print(f"\n表格 {i} ({len(rows)} 行):")
                        for row in rows[:25]:
                            cells = row.find_elements(By.TAG_NAME, 'td')
                            if not cells:
                                cells = row.find_elements(By.TAG_NAME, 'th')
                            texts = [c.text.strip() for c in cells]
                            if any(texts):
                                print(f"  {texts}")

                # 查找包含日期的元素
                print("\n=== 查找日期数据 ===")
                date_els = driver.find_elements(By.XPATH, "//*[contains(text(), '2026-') or contains(text(), '2025-') or contains(text(), '2024-')]")
                print(f"找到 {len(date_els)} 个日期元素:")
                for el in date_els[:20]:
                    try:
                        print(f"  {el.text[:100]}")
                    except:
                        pass

                # 保存详情页HTML
                with open('D:/AI-FINANCE/cebwm_detail_page.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                print("\n详情页已保存到 cebwm_detail_page.html")

        else:
            print("链接无效，尝试从页面数据中提取URL...")

            # 从页面JavaScript中提取产品URL
            urls = driver.execute_script("""
                var urls = [];
                document.querySelectorAll('a').forEach(function(a) {
                    var href = a.getAttribute('href');
                    if (href && href.includes('.html') && !href.includes('void')) {
                        urls.push({text: a.innerText, href: href});
                    }
                });
                return urls;
            """)

            print(f"找到 {len(urls)} 个有效链接:")
            for url in urls[:20]:
                print(f"  {url['text'][:20]}: {url['href']}")

    print("\n完成")


if __name__ == "__main__":
    test()
