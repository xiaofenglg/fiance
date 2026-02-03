# -*- coding: utf-8 -*-
"""测试工银理财网站 - 等待动态内容加载"""

import undetected_chromedriver as uc
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开工银理财净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")

print("\n等待页面和动态内容加载...")
for i in range(30):
    time.sleep(1)

    # 检查内容是否加载
    text_len = driver.execute_script("return document.body.innerText.length")
    tables = driver.execute_script("return document.querySelectorAll('table').length")
    divs = driver.execute_script("return document.querySelectorAll('div').length")

    print(f"  {i+1}秒: 文本={text_len}字, 表格={tables}, div={divs}")

    # 如果有足够的内容，停止等待
    if text_len > 500:
        print("  内容已加载!")
        break

print(f"\n页面标题: {driver.title}")
print(f"页面大小: {len(driver.page_source)} bytes")

# 打印页面文本
body_text = driver.execute_script("return document.body.innerText")
if body_text:
    print(f"\n页面文本({len(body_text)}字符):\n{body_text[:3000]}")
else:
    print("\n页面文本为空")

# 检查是否有iframe
iframes = driver.execute_script("return document.querySelectorAll('iframe').length")
print(f"\niframe数量: {iframes}")

# 检查Vue/React应用
js_check_app = """
return {
    hasVue: typeof Vue !== 'undefined' || !!document.querySelector('[data-v-]'),
    hasReact: typeof React !== 'undefined' || !!document.querySelector('[data-reactroot]'),
    hasAngular: typeof angular !== 'undefined' || !!document.querySelector('[ng-app]'),
    appDiv: document.querySelector('#app') ? 'found' : 'not found',
    rootDiv: document.querySelector('#root') ? 'found' : 'not found'
};
"""
app_info = driver.execute_script(js_check_app)
print(f"\n框架检测: {app_info}")

# 查看页面结构
html_structure = driver.execute_script("""
var html = document.documentElement.outerHTML;
return html.substring(0, 3000);
""")
print(f"\nHTML结构(前3000字符):\n{html_structure}")

print("\n完成")
driver.quit()
