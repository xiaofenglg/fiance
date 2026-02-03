# -*- coding: utf-8 -*-
"""检查工银理财产品点击后的弹窗/详情"""

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

# 清空请求日志
driver.execute_script("window._xhrLog = []")

# 获取点击前的DOM状态
before_html_len = len(driver.page_source)
before_modals = driver.execute_script("return document.querySelectorAll('.el-dialog, .modal, [class*=\"dialog\"], [class*=\"popup\"], [class*=\"overlay\"]').length")

print(f"点击前: HTML长度={before_html_len}, 弹窗数={before_modals}")

# 点击产品项
print("\n点击产品项...")
js_click = """
var items = document.querySelectorAll('.product-item');
if (items.length > 0) {
    items[0].click();
    return {success: true, count: items.length};
}
return {success: false};
"""
result = driver.execute_script(js_click)
print(f"点击结果: {result}")

time.sleep(3)

# 检查点击后的变化
after_html_len = len(driver.page_source)
after_modals = driver.execute_script("return document.querySelectorAll('.el-dialog, .modal, [class*=\"dialog\"], [class*=\"popup\"], [class*=\"overlay\"]').length")

print(f"点击后: HTML长度={after_html_len}, 弹窗数={after_modals}")
print(f"变化: HTML +{after_html_len - before_html_len} 字节, 弹窗 +{after_modals - before_modals}")

# 检查是否有visible的弹窗
js_check_visible = """
var result = [];
var modals = document.querySelectorAll('.el-dialog, .modal, [class*="dialog"], [class*="popup"], [class*="detail"]');
for (var i = 0; i < modals.length; i++) {
    var el = modals[i];
    var style = window.getComputedStyle(el);
    if (style.display !== 'none' && style.visibility !== 'hidden') {
        result.push({
            tag: el.tagName,
            classes: el.className.substring(0, 100),
            text: el.innerText.substring(0, 500),
            visible: true
        });
    }
}
return result;
"""
visible_modals = driver.execute_script(js_check_visible)
print(f"\n可见弹窗: {len(visible_modals)}")
for m in visible_modals[:3]:
    print(f"  [{m['tag']}] class='{m['classes']}'")
    print(f"    text: {m['text'][:200]}...")

# 检查新增的DOM元素
js_find_new = """
var result = [];
// 查找包含净值信息的元素
var els = document.querySelectorAll('*');
for (var i = 0; i < els.length; i++) {
    var el = els[i];
    var text = el.innerText || '';
    // 查找净值数据特征
    if ((text.includes('单位净值') || text.includes('累计净值') || /\\d\\.\\d{4}/.test(text)) && text.length < 1000) {
        var style = window.getComputedStyle(el);
        if (style.display !== 'none') {
            result.push({
                tag: el.tagName,
                classes: el.className.substring(0, 50),
                text: text.substring(0, 300)
            });
        }
    }
    if (result.length > 10) break;
}
return result;
"""
nav_elements = driver.execute_script(js_find_new)
print(f"\n包含净值信息的元素: {len(nav_elements)}")
for e in nav_elements[:5]:
    print(f"  [{e['tag']}] class='{e['classes']}'")
    print(f"    {e['text'][:150]}...")

# 获取捕获的请求
print("\n\n=== 捕获的API请求 ===")
xhr_log = driver.execute_script("return window._xhrLog || []")
print(f"共 {len(xhr_log)} 个请求")

for req in xhr_log:
    url = req.get('url', '')
    if '/clt/' in url:
        print(f"\n[POST] {url}")
        if req.get('body'):
            try:
                body = json.loads(req.get('body'))
                # 显示关键参数
                trans_code = body.get('head_trans_code', '')
                print(f"  trans_code: {trans_code}")
                for key in ['networth_id', 'product_code', 'page_num']:
                    if key in body:
                        print(f"  {key}: {body[key]}")
            except:
                pass

        if req.get('response'):
            try:
                resp = json.loads(req.get('response'))
                if 'rows' in resp:
                    print(f"  rows: {len(resp['rows'])} 项")
                    if resp['rows']:
                        first = resp['rows'][0]
                        print(f"  第一项keys: {list(first.keys()) if isinstance(first, dict) else first}")
            except:
                pass

# 检查产品详情页URL格式
print("\n\n=== 查找详情页链接 ===")
js_find_links = """
var links = [];
var els = document.querySelectorAll('a[href*="Detail"], a[href*="detail"], a[href*="netWorth"]');
for (var i = 0; i < els.length; i++) {
    links.push({
        href: els[i].href,
        text: els[i].innerText.substring(0, 50)
    });
}
return links;
"""
detail_links = driver.execute_script(js_find_links)
print(f"找到 {len(detail_links)} 个详情链接:")
for link in detail_links[:5]:
    print(f"  {link['text']}: {link['href']}")

print("\n完成")
driver.quit()
