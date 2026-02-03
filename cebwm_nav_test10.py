# -*- coding: utf-8 -*-
"""
光大理财净值获取 v10 - 尝试不同的详情页URL模式
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
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
    main_window = driver.current_window_handle

    product_code = "EB1008"

    # 尝试不同的URL模式
    url_patterns = [
        f"https://www.cebwm.com/wealth/grlc/productDetail.html?code={product_code}",
        f"https://www.cebwm.com/wealth/grlc/lccp/{product_code}.html",
        f"https://www.cebwm.com/wealth/grlc/detail/{product_code}.html",
        f"https://www.cebwm.com/wealth/grlc/product.html?cpcode={product_code}",
        f"https://www.cebwm.com/wealth/lccp/{product_code}.html",
        f"https://www.cebwm.com/wealth/product/{product_code}.html",
        f"https://www.cebwm.com/wealth/grlc/{product_code}.html",
        # 尝试搜索API
        f"https://www.cebwm.com/wealth/grlc/search.html?keyword={product_code}",
    ]

    for url in url_patterns:
        print(f"\n尝试: {url}")
        try:
            driver.get(url)
            time.sleep(3)

            # 检查页面是否有效
            page_len = len(driver.page_source)
            title = driver.title

            if page_len > 5000 and '404' not in title and '错误' not in title:
                print(f"  成功! 页面长度: {page_len}, 标题: {title}")

                # 查找产品净值
                nav_els = driver.find_elements(By.XPATH, "//*[contains(text(), '产品净值') or contains(text(), '净值')]")
                if nav_els:
                    print(f"  找到 {len(nav_els)} 个净值相关元素")
                    for el in nav_els[:5]:
                        try:
                            if el.is_displayed():
                                print(f"    {el.tag_name}: {el.text[:30]}")
                        except:
                            pass

                # 保存页面
                with open(f'D:/AI-FINANCE/cebwm_test_{product_code}.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"  页面已保存")

                break
            else:
                print(f"  页面无效 (长度: {page_len}, 标题: {title})")

        except Exception as e:
            print(f"  错误: {e}")

    # 回到主页面
    driver.get("https://www.cebwm.com/wealth/grlc/index.html")
    time.sleep(2)

    # 尝试另一种方法：查找data-*属性
    print("\n\n=== 分析产品行的data属性 ===")

    tables = driver.find_elements(By.TAG_NAME, 'table')
    if len(tables) >= 3:
        product_table = tables[2]
        rows = product_table.find_elements(By.TAG_NAME, 'tr')

        if len(rows) > 1:
            first_row = rows[1]
            print(f"行HTML: {first_row.get_attribute('outerHTML')[:500]}")

            # 查找所有data-*属性
            all_attrs = driver.execute_script("""
                var row = arguments[0];
                var attrs = {};
                for (var i = 0; i < row.attributes.length; i++) {
                    attrs[row.attributes[i].name] = row.attributes[i].value;
                }

                // 检查所有子元素
                row.querySelectorAll('*').forEach(function(el) {
                    for (var i = 0; i < el.attributes.length; i++) {
                        var attr = el.attributes[i];
                        if (attr.name.startsWith('data-') || attr.name.includes('url') || attr.name.includes('href')) {
                            attrs[el.tagName + '_' + attr.name] = attr.value;
                        }
                    }
                });

                return attrs;
            """, first_row)

            print(f"\n属性: {all_attrs}")

    print("\n完成")


if __name__ == "__main__":
    test()
