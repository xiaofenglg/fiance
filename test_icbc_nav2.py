# -*- coding: utf-8 -*-
"""测试工银理财净值详情 - 分析产品点击"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
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
            entry.response = xhr.responseText ? xhr.responseText.substring(0, 5000) : '';
            window._xhrLog.push(entry);
        });

        return origSend.apply(this, arguments);
    };
})();
"""

driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})

print("\n打开净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")
time.sleep(6)

# 分析页面结构找产品项
print("\n=== 分析页面结构 ===")

# 使用JavaScript查找包含产品代码的元素
js_find_products = """
var result = [];
var allElements = document.querySelectorAll('*');
for (var i = 0; i < allElements.length; i++) {
    var el = allElements[i];
    var text = el.innerText || '';
    // 查找包含产品代码格式的元素 (如 25GS0728, 24GS0006)
    if (/\\d{2}GS\\d{4}/.test(text) && text.length < 500) {
        var tagName = el.tagName;
        var classes = el.className;
        result.push({
            tag: tagName,
            classes: classes.substring(0, 100),
            text: text.substring(0, 200),
            clickable: el.onclick !== null || tagName === 'A' || tagName === 'BUTTON'
        });
    }
    if (result.length > 20) break;
}
return result;
"""

products = driver.execute_script(js_find_products)
print(f"找到 {len(products)} 个包含产品代码的元素:")
for p in products[:10]:
    print(f"  [{p['tag']}] class='{p['classes'][:40]}' clickable={p['clickable']}")
    print(f"    text: {p['text'][:100]}...")

# 查找可点击的产品项
print("\n\n=== 查找可点击元素 ===")
js_find_clickable = """
var result = [];
// 查找常见的列表项元素
var selectors = [
    '.el-table__row',
    '[class*="item"]',
    '[class*="list"]',
    '[class*="row"]',
    'li',
    'tr'
];

for (var s = 0; s < selectors.length; s++) {
    var els = document.querySelectorAll(selectors[s]);
    for (var i = 0; i < els.length; i++) {
        var el = els[i];
        var text = el.innerText || '';
        if (/GS\\d{4}/.test(text) || text.includes('工银理财')) {
            result.push({
                selector: selectors[s],
                index: i,
                tag: el.tagName,
                text: text.substring(0, 150),
                rect: el.getBoundingClientRect()
            });
        }
        if (result.length > 15) break;
    }
    if (result.length > 15) break;
}
return result;
"""

clickables = driver.execute_script(js_find_clickable)
print(f"找到 {len(clickables)} 个可能的产品项:")
for c in clickables[:8]:
    print(f"  [{c['selector']}:{c['index']}] {c['tag']}")
    print(f"    text: {c['text'][:80]}...")

# 清空请求日志
driver.execute_script("window._xhrLog = []")

# 尝试点击第一个产品项
if clickables:
    first = clickables[0]
    print(f"\n\n=== 点击第一个产品项 ===")
    print(f"selector: {first['selector']}:{first['index']}")

    js_click = f"""
    var els = document.querySelectorAll('{first['selector']}');
    if (els[{first['index']}]) {{
        els[{first['index']}].click();
        return 'clicked';
    }}
    return 'not found';
    """
    result = driver.execute_script(js_click)
    print(f"点击结果: {result}")

    time.sleep(5)

    # 检查是否有弹窗或新页面
    windows = driver.window_handles
    print(f"窗口数: {len(windows)}")

    if len(windows) > 1:
        print("切换到新窗口...")
        driver.switch_to.window(windows[-1])

    print(f"当前URL: {driver.current_url}")
    print(f"页面标题: {driver.title}")

# 获取捕获的请求
print("\n\n=== 点击后捕获的请求 ===")
xhr_log = driver.execute_script("return window._xhrLog || []")
print(f"共 {len(xhr_log)} 个请求:")

for req in xhr_log:
    url = req.get('url', '')
    if '/clt/' in url:  # 只显示API请求
        print(f"\n[{req.get('method')}] {url}")
        if req.get('body'):
            try:
                body = json.loads(req.get('body'))
                print(f"  Body: {json.dumps(body, ensure_ascii=False, indent=2)[:600]}")
            except:
                print(f"  Body: {req.get('body')[:300]}")

        if req.get('response'):
            try:
                resp = json.loads(req.get('response'))
                print(f"  Response keys: {list(resp.keys()) if isinstance(resp, dict) else 'list'}")
                if isinstance(resp, dict) and 'rows' in resp:
                    rows = resp['rows']
                    print(f"  rows: {len(rows)} 项")
                    if rows:
                        print(f"  第一项: {str(rows[0])[:500]}")
            except:
                print(f"  Response: {req.get('response')[:400]}")

print("\n完成")
driver.quit()
