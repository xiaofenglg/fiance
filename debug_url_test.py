# -*- coding: utf-8 -*-
"""测试不同URL格式"""

import undetected_chromedriver as uc
import time

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

# 先加载列表页建立会话
print("加载列表页建立会话...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")
for i in range(10):
    size = len(driver.page_source)
    if size > 10000:
        break
    time.sleep(1)
print(f"列表页: {len(driver.page_source)} bytes")

# 测试不同的URL格式
test_urls = [
    # 1. 不带参数的基础URL
    "https://www.cebwm.com/wealth/lcxx/lccp14/205687237/index.html",

    # 2. 带简单参数
    "https://www.cebwm.com/wealth/lcxx/lccp14/205687237/index.html?t=" + str(int(time.time())),

    # 3. 另一个产品页面
    "https://www.cebwm.com/wealth/lcxx/lccp14/index.html",
]

for url in test_urls:
    print(f"\n测试: {url[:80]}...")

    # 使用JS导航
    driver.execute_script(f"window.location.href = '{url}'")

    for i in range(10):
        size = len(driver.page_source)
        if i == 9 or size > 1000:
            print(f"结果: {size} bytes")
            break
        time.sleep(1)

    # 返回列表页
    driver.execute_script("window.location.href = 'https://www.cebwm.com/wealth/grlc/index.html'")
    time.sleep(3)

# 测试: 不返回列表页,连续导航
print("\n\n连续导航测试...")
driver.execute_script("window.location.href = 'https://www.cebwm.com/wealth/lcxx/lccp14/205687237/index.html'")
for i in range(15):
    size = len(driver.page_source)
    print(f"{i+1}秒: {size} bytes, URL: {driver.current_url[:60]}...")
    if size > 10000:
        break
    time.sleep(1)

if len(driver.page_source) > 10000:
    print("\n成功!")
    # 保存
    with open("D:/AI-FINANCE/detail_url_test.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)

print("\n完成")
driver.quit()
