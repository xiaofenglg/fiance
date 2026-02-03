# -*- coding: utf-8 -*-
"""捕获弹出窗口URL并在主窗口导航"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import re

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(10):
    size = len(driver.page_source)
    if size > 10000:
        break
    time.sleep(1)

print(f"页面加载完成: {len(driver.page_source)} bytes")

main_window = driver.current_window_handle

# 点击第一个产品链接
print("\n查找并点击产品...")
links = driver.find_elements(By.CSS_SELECTOR, "a.lb_title")
if links:
    first_link = links[0]
    product_name = first_link.text
    print(f"产品: {product_name}")

    # 使用ActionChains点击（之前测试有效打开新窗口）
    actions = ActionChains(driver)
    actions.move_to_element(first_link)
    actions.click()
    actions.perform()

    # 等待新窗口
    time.sleep(3)

    if len(driver.window_handles) > 1:
        print(f"新窗口已打开 ({len(driver.window_handles)}个窗口)")

        # 切换到新窗口获取URL
        new_window = [h for h in driver.window_handles if h != main_window][0]
        driver.switch_to.window(new_window)
        detail_url = driver.current_url
        print(f"详情页URL: {detail_url}")

        # 关闭新窗口
        driver.close()
        driver.switch_to.window(main_window)

        # 在主窗口导航到详情URL
        if 'wealth/lcxx' in detail_url:
            print(f"\n在主窗口导航到详情页...")
            driver.get(detail_url)

            # 等待页面加载
            for i in range(15):
                size = len(driver.page_source)
                print(f"{i+1}秒: {size} bytes")
                if size > 10000:
                    break
                time.sleep(1)

            if len(driver.page_source) > 10000:
                print("\n详情页加载成功!")

                # 保存页面
                with open("D:/AI-FINANCE/detail_main_window.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)

                # 点击"产品净值"标签
                print("\n点击产品净值标签...")
                try:
                    nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
                    nav_tab.click()
                    time.sleep(3)

                    # 找到iframe
                    iframe = driver.find_element(By.ID, "fundValueframe")
                    iframe_src = iframe.get_attribute('src')
                    print(f"iframe src: {iframe_src}")

                    # 切换到iframe
                    driver.switch_to.frame(iframe)
                    time.sleep(3)

                    # 保存iframe内容
                    iframe_html = driver.page_source
                    with open("D:/AI-FINANCE/nav_iframe_main.html", "w", encoding="utf-8") as f:
                        f.write(iframe_html)
                    print(f"iframe内容: {len(iframe_html)} bytes")

                    # 分析净值数据
                    body_text = driver.execute_script("return document.body.innerText")
                    print(f"\nbody文本长度: {len(body_text)}")

                    # 查找日期
                    dates = re.findall(r'\d{4}-\d{2}-\d{2}', body_text)
                    if dates:
                        print(f"找到日期: {dates[:10]}")

                    # 查找净值数据模式
                    nav_pattern = re.findall(r'(\d{4}-\d{2}-\d{2})\s+([\d.]+)', body_text)
                    if nav_pattern:
                        print(f"\n净值数据: {nav_pattern[:5]}")
                    else:
                        # 打印部分文本
                        print("\n前1000字符:")
                        print(body_text[:1000] if body_text else "空内容")

                    driver.switch_to.default_content()

                except Exception as e:
                    print(f"净值提取失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("详情页加载被阻止")
        else:
            print(f"URL格式不正确: {detail_url}")
    else:
        print("未打开新窗口")
else:
    print("未找到产品链接")

print("\n完成")
driver.quit()
