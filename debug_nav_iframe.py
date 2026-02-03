# -*- coding: utf-8 -*-
"""调试NAV iframe内容 - 使用Selenium原生点击"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(10):
    size = len(driver.page_source)
    print(f"{i+1}秒: {size} bytes")
    if size > 10000:
        break
    time.sleep(1)

main_window = driver.current_window_handle
windows_before = len(driver.window_handles)

# 方法1: 直接找链接元素并点击
print("\n查找产品链接...")
try:
    # 找所有 lb_title 类的链接
    links = driver.find_elements(By.CSS_SELECTOR, "a.lb_title")
    print(f"找到 {len(links)} 个产品链接")

    if links:
        first_link = links[0]
        product_name = first_link.text
        print(f"第一个产品: {product_name}")

        # 使用ActionChains模拟真实点击
        print("使用ActionChains点击...")
        actions = ActionChains(driver)
        actions.move_to_element(first_link)
        actions.click()
        actions.perform()

        time.sleep(5)

        print(f"当前窗口数: {len(driver.window_handles)}")
        print(f"当前URL: {driver.current_url}")

        if len(driver.window_handles) > windows_before:
            new_window = [h for h in driver.window_handles if h != main_window][-1]
            driver.switch_to.window(new_window)
            print(f"切换到新窗口")
            print(f"新窗口URL: {driver.current_url}")

            # 等待页面加载
            print("\n等待详情页加载...")
            for i in range(15):
                size = len(driver.page_source)
                print(f"{i+1}秒: {size} bytes")
                if size > 10000:
                    break
                time.sleep(1)

            # 检查是否有iframe
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            print(f"\n找到 {len(iframes)} 个iframe")

            # 保存页面源码
            with open("D:/AI-FINANCE/detail_page_source2.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("详情页源码已保存")

            # 找"产品净值"标签
            try:
                nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
                print(f"\n找到产品净值标签")
                nav_tab.click()
                time.sleep(3)

                # 找iframe
                try:
                    iframe = driver.find_element(By.ID, "fundValueframe")
                    print(f"找到iframe, src: {iframe.get_attribute('src')}")
                    driver.switch_to.frame(iframe)

                    time.sleep(3)

                    # 获取iframe内容
                    iframe_html = driver.page_source
                    print(f"iframe内容: {len(iframe_html)} bytes")

                    with open("D:/AI-FINANCE/nav_iframe_source.html", "w", encoding="utf-8") as f:
                        f.write(iframe_html)
                    print("iframe源码已保存")

                    # 分析内容
                    body_text = driver.execute_script("return document.body.innerText")

                    import re
                    dates = re.findall(r'\d{4}-\d{2}-\d{2}', body_text)
                    if dates:
                        print(f"\n找到日期: {dates[:10]}")

                    nav_pattern = re.findall(r'(\d{4}-\d{2}-\d{2})\s+([\d.]+)', body_text)
                    if nav_pattern:
                        print(f"找到净值数据: {nav_pattern[:5]}")

                except Exception as e:
                    print(f"iframe处理失败: {e}")
            except Exception as e:
                print(f"产品净值标签失败: {e}")

        else:
            print("没有打开新窗口")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("\n完成")
driver.quit()
