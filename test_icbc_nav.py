# -*- coding: utf-8 -*-
"""测试工银理财净值详情API"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

# 注入XHR拦截器
intercept_script = """
(function() {
    window._xhrLog = [];
    var origOpen = XMLHttpRequest.prototype.open;
    var origSend = XMLHttpRequest.prototype.send;

    XMLHttpRequest.prototype.open = function(method, url) {
        this._method = method;
        this._url = url;
        return origOpen.apply(this, arguments);
    };

    XMLHttpRequest.prototype.send = function(body) {
        var xhr = this;
        var entry = {
            method: this._method,
            url: this._url,
            body: body,
            time: new Date().toISOString()
        };

        this.addEventListener('load', function() {
            entry.status = xhr.status;
            entry.response = xhr.responseText ? xhr.responseText.substring(0, 3000) : '';
            window._xhrLog.push(entry);
        });

        return origSend.apply(this, arguments);
    };
})();
"""

driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})

print("\n打开净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")
time.sleep(5)

# 清空之前的日志
driver.execute_script("window._xhrLog = []")

print("\n尝试点击第一个产品查看详情...")
try:
    # 查找产品链接/按钮
    # 从捕获的数据看，产品名称类似 "工银理财·鑫添益..."
    links = driver.find_elements(By.TAG_NAME, 'a')
    clicked = False

    for link in links:
        text = link.text.strip()
        if '工银理财' in text or '净值披露' in text[:20]:
            print(f"找到产品链接: {text[:50]}...")
            href = link.get_attribute('href')
            print(f"  href: {href}")

            # 点击
            link.click()
            clicked = True
            time.sleep(5)
            break

    if not clicked:
        # 尝试找div或其他可点击元素
        divs = driver.find_elements(By.CSS_SELECTOR, 'div[class*="item"], div[class*="product"], div[class*="row"]')
        for div in divs[:20]:
            text = div.text.strip()
            if '工银理财' in text or ('GS' in text and '净值' in text):
                print(f"找到产品div: {text[:50]}...")
                div.click()
                clicked = True
                time.sleep(5)
                break

except Exception as e:
    print(f"点击失败: {e}")

# 获取捕获的请求
print("\n=== 点击后捕获的请求 ===")
xhr_log = driver.execute_script("return window._xhrLog || []")
print(f"共 {len(xhr_log)} 个请求:")

for req in xhr_log:
    print(f"\n[{req.get('method')}] {req.get('url')}")
    if req.get('body'):
        try:
            body = json.loads(req.get('body'))
            print(f"  Body: {json.dumps(body, ensure_ascii=False)[:500]}")
        except:
            print(f"  Body: {req.get('body')[:300]}")

    if req.get('response'):
        try:
            resp = json.loads(req.get('response'))
            print(f"  Response keys: {list(resp.keys()) if isinstance(resp, dict) else 'list'}")
            # 查找净值数据
            if isinstance(resp, dict):
                for key in ['rows', 'data', 'list', 'content', 'navList', 'netWorthList']:
                    if key in resp:
                        items = resp[key]
                        if isinstance(items, list) and len(items) > 0:
                            print(f"  {key}: {len(items)} 项")
                            print(f"  第一项: {str(items[0])[:400]}")
        except:
            print(f"  Response: {req.get('response')[:300]}")

# 检查当前页面URL
print(f"\n当前URL: {driver.current_url}")
print(f"页面标题: {driver.title}")

# 如果打开了新页面，分析页面结构
if 'Detail' in driver.current_url or driver.current_url != "https://wm.icbc.com.cn/netWorthDisclosure":
    print("\n=== 详情页分析 ===")
    tables = driver.find_elements(By.TAG_NAME, 'table')
    print(f"表格数: {len(tables)}")

    for i, table in enumerate(tables[:3]):
        rows = table.find_elements(By.TAG_NAME, 'tr')
        print(f"\n表格{i+1}: {len(rows)} 行")
        for row in rows[:5]:
            print(f"  {row.text.strip()[:100]}")

print("\n完成")
driver.quit()
