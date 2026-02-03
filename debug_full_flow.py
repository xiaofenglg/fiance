# -*- coding: utf-8 -*-
"""完整流程测试 - 获取产品净值"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import re

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财列表页...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(10):
    size = len(driver.page_source)
    if size > 10000:
        break
    time.sleep(1)

print(f"列表页加载完成: {len(driver.page_source)} bytes")

main_window = driver.current_window_handle

# 点击产品获取详情页URL
print("\n点击产品获取详情URL...")
links = driver.find_elements(By.CSS_SELECTOR, "a.lb_title")
if links:
    first_link = links[0]
    product_name = first_link.text
    print(f"产品: {product_name}")

    # 使用ActionChains点击
    actions = ActionChains(driver)
    actions.move_to_element(first_link)
    actions.click()
    actions.perform()

    time.sleep(3)

    if len(driver.window_handles) > 1:
        # 切换到新窗口获取URL
        new_window = [h for h in driver.window_handles if h != main_window][0]
        driver.switch_to.window(new_window)
        detail_url = driver.current_url
        print(f"详情页URL: {detail_url}")

        # 关闭新窗口
        driver.close()
        driver.switch_to.window(main_window)

        # 使用JavaScript在主窗口导航
        print("\n使用JS导航到详情页...")
        driver.execute_script(f"window.location.href = '{detail_url}'")

        for i in range(15):
            size = len(driver.page_source)
            print(f"{i+1}秒: {size} bytes")
            if size > 10000:
                break
            time.sleep(1)

        if len(driver.page_source) > 10000:
            print("\n详情页加载成功!")

            # 保存页面
            with open("D:/AI-FINANCE/detail_js_nav.html", "w", encoding="utf-8") as f:
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

                # 等待iframe内容加载
                print("等待iframe加载...")
                for i in range(10):
                    size = len(driver.page_source)
                    print(f"iframe {i+1}秒: {size} bytes")
                    if size > 1000:
                        break
                    time.sleep(1)

                # 保存iframe内容
                iframe_html = driver.page_source
                with open("D:/AI-FINANCE/nav_iframe_js.html", "w", encoding="utf-8") as f:
                    f.write(iframe_html)
                print(f"\niframe内容: {len(iframe_html)} bytes")

                # 分析净值数据
                body_text = driver.execute_script("return document.body.innerText")
                print(f"body文本长度: {len(body_text)}")

                # 查找日期
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', body_text)
                if dates:
                    print(f"找到日期: {dates[:10]}")

                # 查找净值数据模式 - 日期后跟数字
                nav_pattern = re.findall(r'(\d{4}-\d{2}-\d{2})\s+([\d.]+)', body_text)
                if nav_pattern:
                    print(f"\n净值数据: {nav_pattern[:10]}")
                else:
                    # 尝试查找表格
                    print("\n尝试从表格提取...")
                    js_extract = """
                    var result = [];
                    var tables = document.querySelectorAll('table');
                    for (var t = 0; t < tables.length; t++) {
                        var rows = tables[t].querySelectorAll('tr');
                        for (var i = 0; i < rows.length; i++) {
                            var cells = rows[i].querySelectorAll('td');
                            if (cells.length >= 2) {
                                var text0 = cells[0] ? cells[0].innerText.trim() : '';
                                var text1 = cells[1] ? cells[1].innerText.trim() : '';
                                var text2 = cells[2] ? cells[2].innerText.trim() : '';
                                if (/\\d{4}-\\d{2}-\\d{2}/.test(text0)) {
                                    result.push({date: text0, unit: text1, total: text2});
                                }
                            }
                        }
                    }
                    return result;
                    """
                    table_data = driver.execute_script(js_extract)
                    if table_data:
                        print(f"表格净值数据: {table_data[:10]}")
                    else:
                        # 打印文本看看格式
                        print("\n前2000字符:")
                        print(body_text[:2000] if body_text else "空内容")

                driver.switch_to.default_content()

            except Exception as e:
                print(f"净值提取失败: {e}")
                import traceback
                traceback.print_exc()

            # 返回列表页
            print("\n返回列表页...")
            list_url = "https://www.cebwm.com/wealth/grlc/index.html"
            driver.execute_script(f"window.location.href = '{list_url}'")

            for i in range(10):
                size = len(driver.page_source)
                if size > 10000:
                    break
                time.sleep(1)

            print(f"返回列表页: {len(driver.page_source)} bytes")

        else:
            print("详情页加载失败")
    else:
        print("未打开新窗口")
else:
    print("未找到产品链接")

print("\n完成")
driver.quit()
