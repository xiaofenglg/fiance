# -*- coding: utf-8 -*-
"""调试 - 分析页面结构提取URL"""

import undetected_chromedriver as uc
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财列表页...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

for i in range(15):
    if len(driver.page_source) > 10000:
        break
    time.sleep(1)

print(f"页面加载完成")

# 分析产品链接结构
js_analyze = """
var result = [];
var tables = document.querySelectorAll('table');
for (var t = 0; t < tables.length; t++) {
    var rows = tables[t].querySelectorAll('tbody tr, tr');
    for (var i = 0; i < rows.length; i++) {
        var cells = rows[i].querySelectorAll('td');
        if (cells.length >= 10) {
            var code = cells[1] ? cells[1].innerText.trim() : '';
            if (code && (code.startsWith('EB') || code.startsWith('EW'))) {
                var link = cells[0].querySelector('a.lb_title');
                var onclick = link ? link.getAttribute('onclick') : null;
                var href = link ? link.getAttribute('href') : null;
                result.push({
                    code: code,
                    onclick: onclick,
                    href: href,
                    linkHTML: link ? link.outerHTML.substring(0, 200) : null
                });
            }
        }
    }
}
return result;
"""

links = driver.execute_script(js_analyze)
print(f"\n找到 {len(links)} 个产品链接:")
for link in links[:5]:
    print(f"\n  代码: {link['code']}")
    print(f"  onclick: {link['onclick']}")
    print(f"  href: {link['href']}")
    print(f"  HTML: {link['linkHTML']}")

# 尝试从onclick提取URL
print("\n\n分析onclick结构...")
if links and links[0]['onclick']:
    onclick = links[0]['onclick']
    print(f"第一个onclick: {onclick}")

    # 尝试提取参数
    import re
    match = re.search(r"lcxxUrl\s*\(\s*'([^']+)'", onclick)
    if match:
        print(f"提取到参数: {match.group(1)}")

print("\n完成")
driver.quit()
