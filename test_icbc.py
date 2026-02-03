# -*- coding: utf-8 -*-
"""测试工银理财网站访问"""

import undetected_chromedriver as uc
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开工银理财净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")

# 等待页面加载
for i in range(15):
    size = len(driver.page_source)
    print(f"  {i+1}秒: {size} bytes")
    if size > 10000:
        break
    time.sleep(1)

print(f"\n页面标题: {driver.title}")
print(f"页面大小: {len(driver.page_source)} bytes")

# 分析页面内容
if len(driver.page_source) > 1000:
    # 查找产品表格
    js_analyze = """
    var result = {
        tables: document.querySelectorAll('table').length,
        rows: document.querySelectorAll('tr').length,
        links: document.querySelectorAll('a').length,
        hasVue: typeof Vue !== 'undefined',
        hasReact: typeof React !== 'undefined'
    };
    // 查找包含"净值"的文本
    var body = document.body.innerText;
    result.hasNav = body.includes('净值');
    result.hasProduct = body.includes('产品');
    result.textLength = body.length;
    return result;
    """
    info = driver.execute_script(js_analyze)
    print(f"\n页面分析:")
    print(f"  表格数: {info['tables']}")
    print(f"  行数: {info['rows']}")
    print(f"  链接数: {info['links']}")
    print(f"  包含'净值': {info['hasNav']}")
    print(f"  包含'产品': {info['hasProduct']}")
    print(f"  文本长度: {info['textLength']}")

    # 打印页面部分文本
    body_text = driver.execute_script("return document.body.innerText.substring(0, 2000)")
    print(f"\n页面文本(前2000字符):\n{body_text}")
else:
    print("\n页面加载失败或被阻止")
    print(f"页面源码:\n{driver.page_source[:500]}")

print("\n完成")
driver.quit()
