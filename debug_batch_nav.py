# -*- coding: utf-8 -*-
"""批量导航测试 - 获取所有URL后连续访问"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import re
import json

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

# 获取所有产品的详情URL
print("\n获取所有产品的详情URL...")
product_urls = []

links = driver.find_elements(By.CSS_SELECTOR, "a.lb_title")
print(f"找到 {len(links)} 个产品链接")

for i, link in enumerate(links[:5]):  # 测试前5个
    product_name = link.text
    print(f"\n[{i+1}] 获取: {product_name}")

    # 点击获取URL
    actions = ActionChains(driver)
    actions.move_to_element(link)
    actions.click()
    actions.perform()

    time.sleep(2)

    if len(driver.window_handles) > 1:
        new_window = [h for h in driver.window_handles if h != main_window][0]
        driver.switch_to.window(new_window)
        full_url = driver.current_url
        base_url = full_url.split('?')[0]
        print(f"   URL: {base_url}")
        product_urls.append({'name': product_name, 'url': base_url})

        # 关闭弹窗返回列表
        driver.close()
        driver.switch_to.window(main_window)
        time.sleep(1)

print(f"\n收集到 {len(product_urls)} 个URL")

# 现在依次访问这些URL
print("\n\n开始批量访问...")

# 先用about:blank重置
driver.get("about:blank")
time.sleep(1)

all_nav_data = {}

for i, item in enumerate(product_urls):
    name = item['name']
    url = item['url']

    print(f"\n[{i+1}/{len(product_urls)}] {name}")
    print(f"   导航到: {url}")

    # 直接导航
    driver.execute_script(f"window.location.href = '{url}'")

    # 等待加载
    success = False
    for j in range(15):
        size = len(driver.page_source)
        if size > 10000:
            success = True
            break
        time.sleep(1)

    if success:
        print(f"   页面加载成功: {size} bytes")

        # 提取产品代码
        try:
            code = driver.find_element(By.CSS_SELECTOR, "meta[name='keywords']").get_attribute('content')
            print(f"   产品代码: {code}")
        except:
            code = name

        # 点击净值标签
        try:
            nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
            nav_tab.click()
            time.sleep(2)

            # 切换到iframe
            iframe = driver.find_element(By.ID, "fundValueframe")
            driver.switch_to.frame(iframe)
            time.sleep(2)

            # 提取净值
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
            nav_data = driver.execute_script(js_extract)

            if nav_data:
                print(f"   净值数据: {len(nav_data)} 条")
                all_nav_data[code] = {
                    'name': name,
                    'nav_history': nav_data
                }
                for d in nav_data[:3]:
                    print(f"      {d['date']}: {d['unit']}")

            driver.switch_to.default_content()

        except Exception as e:
            print(f"   净值提取失败: {e}")

    else:
        print(f"   页面加载失败 ({size} bytes)")
        # 尝试重置
        print("   尝试重置...")
        driver.get("about:blank")
        time.sleep(2)

print("\n\n=== 结果汇总 ===")
print(f"成功获取 {len(all_nav_data)} 个产品的净值数据")

# 保存结果
with open("D:/AI-FINANCE/nav_batch_test.json", "w", encoding="utf-8") as f:
    json.dump(all_nav_data, f, ensure_ascii=False, indent=2)
print("结果已保存到 nav_batch_test.json")

print("\n完成")
driver.quit()
