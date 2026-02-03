# -*- coding: utf-8 -*-
"""分析工银理财详情页获取净值数据"""

import undetected_chromedriver as uc
import time
import json
import re

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

# 注入XHR拦截器
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
            entry.response = xhr.responseText ? xhr.responseText.substring(0, 10000) : '';
            window._xhrLog.push(entry);
        });

        return origSend.apply(this, arguments);
    };

    // 拦截fetch
    var origFetch = window.fetch;
    window.fetch = function(url, options) {
        var entry = {url: url, options: options, time: new Date().toISOString()};
        return origFetch.apply(this, arguments).then(function(response) {
            return response.clone().text().then(function(text) {
                entry.response = text.substring(0, 10000);
                window._fetchLog.push(entry);
                return response;
            });
        });
    };
})();
"""

driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})

# 直接访问详情页
detail_url = "https://wm.icbc.com.cn/netWorthDisclosureDetails?type=03&id=2025101703300800008236&title=test"
print(f"\n直接打开详情页: {detail_url}")
driver.get(detail_url)

print("等待页面加载...")
time.sleep(8)

# 获取页面HTML
html = driver.page_source
print(f"\n页面HTML长度: {len(html)}")

# 查找净值数据格式
nav_pattern = r'\d\.\d{4}'
navs = re.findall(nav_pattern, html)
print(f"页面中的净值格式数据: {len(navs)} 个")
for nav in navs[:30]:
    print(f"  {nav}")

# 检查是否有表格
print("\n\n=== 检查表格 ===")
js_tables = """
var tables = document.querySelectorAll('table');
var result = [];
for (var i = 0; i < tables.length; i++) {
    var table = tables[i];
    var rows = table.querySelectorAll('tr');
    result.push({
        rows: rows.length,
        html: table.outerHTML.substring(0, 2000)
    });
}
return result;
"""
tables = driver.execute_script(js_tables)
print(f"找到 {len(tables)} 个表格:")
for i, t in enumerate(tables):
    print(f"  表格 {i+1}: {t['rows']} 行")
    print(f"  HTML: {t['html'][:500]}...")

# 检查页面主要内容
print("\n\n=== 页面主要内容 ===")
js_content = """
var content = document.querySelector('.content, .main, .detail, [class*="detail"], article, .wrapper');
if (content) {
    return {
        className: content.className,
        text: content.innerText.substring(0, 2000)
    };
}
// 如果找不到，返回body
return {
    className: 'body',
    text: document.body.innerText.substring(0, 2000)
};
"""
content = driver.execute_script(js_content)
print(f"内容类: {content.get('className')}")
print(f"文本:\n{content.get('text')}")

# 获取捕获的请求
print("\n\n=== 捕获的API请求 ===")
xhr_log = driver.execute_script("return window._xhrLog || []")
fetch_log = driver.execute_script("return window._fetchLog || []")

print(f"XHR请求: {len(xhr_log)}")
for req in xhr_log:
    url = req.get('url', '')
    if '/clt/' in url or 'api' in url.lower():
        print(f"\n[XHR] {req.get('method')} {url}")
        if req.get('body'):
            print(f"  Body: {req.get('body')[:500]}")
        if req.get('response'):
            try:
                resp = json.loads(req.get('response'))
                print(f"  Response keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
                if isinstance(resp, dict) and 'rows' in resp:
                    print(f"  rows: {len(resp['rows'])} 项")
                    if resp['rows']:
                        print(f"  第一项: {str(resp['rows'][0])[:500]}")
                else:
                    print(f"  Response: {str(resp)[:500]}")
            except:
                print(f"  Response: {req.get('response')[:500]}")

print(f"\nFetch请求: {len(fetch_log)}")
for req in fetch_log:
    url = req.get('url', '')
    if '/clt/' in str(url) or 'api' in str(url).lower():
        print(f"\n[Fetch] {url}")
        if req.get('response'):
            print(f"  Response: {req.get('response')[:500]}")

print("\n完成")
driver.quit()
