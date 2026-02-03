# -*- coding: utf-8 -*-
"""
光大理财净值获取 v9 - 拦截AJAX获取产品数据和URL
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json


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

    # 获取产品数据 - 通过JavaScript调用相同的AJAX
    print("\n=== 获取产品数据 ===")

    # 从页面获取pageId
    page_id = driver.execute_script("return document.getElementById('eprotalCurrentPageId').value;")
    print(f"pageId: {page_id}")

    # 调用产品查询API
    product_data = driver.execute_script("""
        var result = null;
        $.ajax({
            type: 'POST',
            url: '/eportal/ui?struts.portlet.action=/portlet/financeProductSearchList!searchFinanceList.action&pageId=' + arguments[0] + '&pageKey=financeProductSearchListAction',
            dataType: 'json',
            async: false,
            data: {
                inputQueryStr: '',
                mjbzArr: [],
                xsqdjeArr: [],
                cpfxdjArr: [],
                sfkfsArr: [],
                pageKeyStr: 'financeProductSearchListAction',
                currentPage: 1,
                isGrlcStr: '1',
                qclx: '',
                cpFlbs: '0'
            },
            success: function(data) {
                result = data;
            }
        });
        return result;
    """, page_id)

    if product_data:
        print(f"获取到产品数据")
        print(f"总数: {product_data.get('page', {}).get('totalCount', 0)}")

        result_list = product_data.get('resultList', [])
        print(f"当前页产品数: {len(result_list)}")

        if result_list:
            # 打印第一个产品的所有字段
            first = result_list[0]
            print(f"\n第一个产品的字段:")
            for key, value in first.items():
                print(f"  {key}: {value}")

            # 获取HTMLURL
            html_url = first.get('HTMLURL', '')
            print(f"\n详情页URL: {html_url}")

            if html_url:
                # 访问详情页
                full_url = f"https://www.cebwm.com{html_url}" if html_url.startswith('/') else html_url
                print(f"\n访问: {full_url}")

                driver.execute_script(f"window.open('{full_url}', '_blank');")
                time.sleep(3)

                if len(driver.window_handles) > 1:
                    new_window = [h for h in driver.window_handles if h != main_window][0]
                    driver.switch_to.window(new_window)
                    print(f"新窗口URL: {driver.current_url}")
                    print(f"页面标题: {driver.title}")

                    time.sleep(2)

                    # 查找产品净值标签
                    print("\n=== 查找产品净值 ===")

                    # 查找所有包含"净值"的可点击元素
                    nav_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '产品净值') or contains(text(), '净值走势') or contains(text(), '历史净值')]")
                    print(f"找到 {len(nav_elements)} 个净值相关元素")

                    for el in nav_elements:
                        try:
                            if el.is_displayed():
                                print(f"  {el.tag_name}: {el.text[:50]}")
                                # 尝试点击
                                driver.execute_script("arguments[0].click();", el)
                                time.sleep(2)
                                break
                        except:
                            pass

                    # 保存详情页HTML用于分析
                    with open('D:/AI-FINANCE/cebwm_detail_page.html', 'w', encoding='utf-8') as f:
                        f.write(driver.page_source)
                    print("\n详情页HTML已保存")

                    # 查找净值数据
                    print("\n=== 查找净值数据 ===")

                    tables = driver.find_elements(By.TAG_NAME, 'table')
                    print(f"找到 {len(tables)} 个表格")

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

    else:
        print("未能获取产品数据")

    print("\n完成")


if __name__ == "__main__":
    test()
