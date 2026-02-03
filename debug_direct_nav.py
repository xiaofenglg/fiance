# -*- coding: utf-8 -*-
"""直接导航测试 - 不点击任何链接"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
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

# 不点击任何东西，直接导航到已知的产品详情页
detail_url = "https://www.cebwm.com/wealth/lcxx/lccp14/205687237/index.html"
print(f"\n直接导航到: {detail_url}")

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
    with open("D:/AI-FINANCE/detail_direct_nav.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)

    # 点击"产品净值"标签
    print("\n点击产品净值标签...")
    try:
        nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
        print(f"找到标签: {nav_tab.text}")
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
        with open("D:/AI-FINANCE/nav_iframe_direct.html", "w", encoding="utf-8") as f:
            f.write(iframe_html)
        print(f"\niframe内容: {len(iframe_html)} bytes")

        # 分析净值数据
        body_text = driver.execute_script("return document.body.innerText")
        print(f"body文本长度: {len(body_text)}")

        # 从表格提取
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
            print(f"\n净值数据 ({len(table_data)}条):")
            for d in table_data[:10]:
                print(f"  {d['date']}: 单位={d['unit']}, 累计={d['total']}")
        else:
            print("\n前1500字符:")
            safe_text = body_text.replace('\xa0', ' ') if body_text else "空内容"
            print(safe_text[:1500])

        driver.switch_to.default_content()

    except Exception as e:
        print(f"净值提取失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试能否返回列表页并再导航到另一个产品
    print("\n\n测试返回列表页...")
    list_url = "https://www.cebwm.com/wealth/grlc/index.html"
    driver.execute_script(f"window.location.href = '{list_url}'")

    for i in range(10):
        size = len(driver.page_source)
        if size > 10000:
            break
        time.sleep(1)

    print(f"返回列表页: {len(driver.page_source)} bytes")

    if len(driver.page_source) > 10000:
        # 导航到另一个产品 (假设ID存在)
        detail_url2 = "https://www.cebwm.com/wealth/lcxx/lccp14/205687238/index.html"
        print(f"\n导航到第二个产品: {detail_url2}")
        driver.execute_script(f"window.location.href = '{detail_url2}'")

        for i in range(10):
            size = len(driver.page_source)
            print(f"{i+1}秒: {size} bytes")
            if size > 1000:
                break
            time.sleep(1)

else:
    print("详情页加载失败")

print("\n完成")
driver.quit()
