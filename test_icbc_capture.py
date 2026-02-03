# -*- coding: utf-8 -*-
"""捕获工银理财的实际API调用"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
# 启用网络日志
options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
driver = uc.Chrome(options=options, use_subprocess=True)

# 注入XHR拦截器
print("注入请求拦截器...")
intercept_script = """
(function() {
    window._xhrLog = [];
    window._fetchLog = [];

    // 拦截XHR
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
            entry.response = xhr.responseText ? xhr.responseText.substring(0, 2000) : '';
            window._xhrLog.push(entry);
        });

        return origSend.apply(this, arguments);
    };

    // 拦截fetch
    var origFetch = window.fetch;
    window.fetch = function(url, options) {
        var entry = {
            url: url,
            method: options ? options.method : 'GET',
            body: options ? options.body : null,
            time: new Date().toISOString()
        };

        return origFetch.apply(this, arguments).then(function(response) {
            response.clone().text().then(function(text) {
                entry.status = response.status;
                entry.response = text.substring(0, 2000);
                window._fetchLog.push(entry);
            });
            return response;
        });
    };
})();
"""

driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})

print("\n打开净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")

print("等待页面加载和API调用...")
time.sleep(8)

# 获取捕获的请求
print("\n=== 捕获的XHR请求 ===")
xhr_log = driver.execute_script("return window._xhrLog || []")
print(f"共 {len(xhr_log)} 个XHR请求:")
for req in xhr_log:
    print(f"\n[{req.get('method')}] {req.get('url')}")
    if req.get('body'):
        print(f"  Body: {req.get('body')[:500]}")
    print(f"  Status: {req.get('status')}")
    if req.get('response'):
        resp = req.get('response', '')[:800]
        print(f"  Response: {resp}")

print("\n=== 捕获的Fetch请求 ===")
fetch_log = driver.execute_script("return window._fetchLog || []")
print(f"共 {len(fetch_log)} 个Fetch请求:")
for req in fetch_log:
    print(f"\n[{req.get('method')}] {req.get('url')}")
    if req.get('body'):
        print(f"  Body: {req.get('body')[:500]}")
    print(f"  Status: {req.get('status')}")
    if req.get('response'):
        resp = req.get('response', '')[:800]
        print(f"  Response: {resp}")

# 尝试点击分页按钮触发更多请求
print("\n\n=== 触发分页 ===")
try:
    # 查找分页按钮
    buttons = driver.find_elements(By.TAG_NAME, 'button')
    for btn in buttons:
        if '下一页' in btn.text or '2' == btn.text.strip():
            print(f"点击按钮: {btn.text}")
            btn.click()
            time.sleep(3)
            break

    # 再次获取请求日志
    xhr_log2 = driver.execute_script("return window._xhrLog || []")
    if len(xhr_log2) > len(xhr_log):
        print(f"\n分页后新增 {len(xhr_log2) - len(xhr_log)} 个请求:")
        for req in xhr_log2[len(xhr_log):]:
            print(f"\n[{req.get('method')}] {req.get('url')}")
            if req.get('body'):
                print(f"  Body: {req.get('body')[:500]}")
            if req.get('response'):
                print(f"  Response: {req.get('response')[:500]}")
except Exception as e:
    print(f"分页测试失败: {e}")

print("\n完成")
driver.quit()
