# -*- coding: utf-8 -*-
"""调试筛选和链接获取"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time

BASE_URL = "https://www.cebwm.com/wealth/grlc/index.html"

def debug():
    print("启动浏览器...")
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    driver = uc.Chrome(options=options, use_subprocess=True)

    try:
        print(f"访问: {BASE_URL}")
        driver.get(BASE_URL)
        time.sleep(5)

        # 点击开放式
        print("\n1. 点击开放式...")
        result = driver.execute_script("""
            var links = document.querySelectorAll('a');
            for (var i = 0; i < links.length; i++) {
                if (links[i].innerText.trim() === '开放式') {
                    links[i].click();
                    return '点击成功: ' + links[i].outerHTML.substring(0, 100);
                }
            }
            return '未找到';
        """)
        print(f"  结果: {result}")
        time.sleep(3)

        # 检查产品数量
        count = driver.execute_script("return document.querySelector('.total_count1')?.innerText || '未找到'")
        print(f"  当前产品数: {count}")

        # 点击5万-50万
        print("\n2. 点击5万-50万...")
        result = driver.execute_script("""
            var links = document.querySelectorAll('a');
            for (var i = 0; i < links.length; i++) {
                var text = links[i].innerText.trim();
                if (text.indexOf('5万-50万') >= 0) {
                    links[i].click();
                    return '点击成功: ' + text;
                }
            }
            return '未找到';
        """)
        print(f"  结果: {result}")
        time.sleep(3)

        # 再次检查产品数量
        count = driver.execute_script("return document.querySelector('.total_count1')?.innerText || '未找到'")
        pages = driver.execute_script("return document.querySelector('#totalpage1')?.innerText || '未找到'")
        print(f"  当前产品数: {count}, 页数: {pages}")

        # 查看第一个产品的链接结构
        print("\n3. 查看产品链接结构...")
        links_info = driver.execute_script("""
            var result = [];
            var table = document.querySelector('table');
            if (!table) return ['未找到表格'];

            var rows = table.querySelectorAll('tbody tr, tr');
            for (var i = 0; i < Math.min(3, rows.length); i++) {
                var row = rows[i];
                var cells = row.querySelectorAll('td');
                if (cells.length >= 2) {
                    var firstCell = cells[0];
                    var code = cells[1] ? cells[1].innerText.trim() : '';

                    // 查找所有链接
                    var links = firstCell.querySelectorAll('a');
                    var linkInfo = [];
                    for (var j = 0; j < links.length; j++) {
                        linkInfo.push({
                            text: links[j].innerText.trim().substring(0, 30),
                            class: links[j].className,
                            href: links[j].href ? links[j].href.substring(0, 50) : 'none'
                        });
                    }
                    result.push({
                        code: code,
                        links: linkInfo,
                        cellHTML: firstCell.innerHTML.substring(0, 200)
                    });
                }
            }
            return result;
        """)
        for info in links_info:
            print(f"  产品 {info.get('code', '?')}:")
            print(f"    链接: {info.get('links', [])}")
            print(f"    HTML: {info.get('cellHTML', '')[:100]}")

        # 尝试获取第一个产品的详情URL
        print("\n4. 尝试点击第一个产品...")
        result = driver.execute_script("""
            var table = document.querySelector('table');
            if (!table) return {error: '未找到表格'};

            var rows = table.querySelectorAll('tbody tr, tr');
            for (var i = 0; i < rows.length; i++) {
                var cells = rows[i].querySelectorAll('td');
                if (cells.length >= 2) {
                    var firstCell = cells[0];
                    // 尝试不同的链接选择器
                    var link = firstCell.querySelector('a.lb_title') ||
                               firstCell.querySelector('a[href*="lccp"]') ||
                               firstCell.querySelector('a');
                    if (link) {
                        return {
                            found: true,
                            text: link.innerText.trim(),
                            href: link.href,
                            class: link.className
                        };
                    }
                }
            }
            return {error: '未找到链接'};
        """)
        print(f"  结果: {result}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    debug()
