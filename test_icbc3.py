# -*- coding: utf-8 -*-
"""测试工银理财网站 - 分析页面结构"""

import undetected_chromedriver as uc
import time
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开工银理财净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")

print("\n等待内容加载...")
time.sleep(5)

print(f"页面标题: {driver.title}")
print(f"页面大小: {len(driver.page_source)} bytes")

# 分析页面结构
js_analyze = """
var result = {
    tables: document.querySelectorAll('table').length,
    divs: document.querySelectorAll('div').length,
    links: [],
    buttons: [],
    inputs: [],
    hasNav: document.body.innerText.includes('净值'),
    hasProduct: document.body.innerText.includes('产品'),
    hasCode: document.body.innerText.includes('代码'),
    textPreview: document.body.innerText.substring(0, 1500).replace(/[\\r\\n]+/g, ' | ')
};

// 收集链接
var links = document.querySelectorAll('a');
for (var i = 0; i < Math.min(links.length, 20); i++) {
    result.links.push({
        text: links[i].innerText.substring(0, 50),
        href: links[i].href
    });
}

// 收集按钮
var buttons = document.querySelectorAll('button, .el-button, [class*="btn"]');
for (var i = 0; i < Math.min(buttons.length, 10); i++) {
    result.buttons.push(buttons[i].innerText.substring(0, 30));
}

// 收集输入框
var inputs = document.querySelectorAll('input, select');
for (var i = 0; i < Math.min(inputs.length, 10); i++) {
    result.inputs.push({
        type: inputs[i].type || inputs[i].tagName,
        placeholder: inputs[i].placeholder || '',
        name: inputs[i].name || ''
    });
}

return result;
"""

try:
    info = driver.execute_script(js_analyze)

    print(f"\n页面分析:")
    print(f"  表格: {info['tables']}")
    print(f"  div: {info['divs']}")
    print(f"  包含'净值': {info['hasNav']}")
    print(f"  包含'产品': {info['hasProduct']}")
    print(f"  包含'代码': {info['hasCode']}")

    print(f"\n链接({len(info['links'])}个):")
    for link in info['links'][:10]:
        text = link['text'].strip()[:40] if link['text'] else ''
        href = link['href'][:80] if link['href'] else ''
        if text or href:
            print(f"  {text} -> {href}")

    print(f"\n按钮: {info['buttons']}")
    print(f"\n输入框: {info['inputs']}")

    # 安全打印预览文本
    preview = info['textPreview']
    # 替换可能导致编码问题的字符
    preview = preview.encode('gbk', errors='replace').decode('gbk')
    print(f"\n文本预览:\n{preview}")

except Exception as e:
    print(f"分析出错: {e}")

# 检查网络请求（API调用）
print("\n\n=== 检查API请求 ===")
js_intercept = """
// 注入fetch拦截
if (!window._fetchLog) {
    window._fetchLog = [];
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        window._fetchLog.push({url: args[0], time: new Date().toISOString()});
        return originalFetch.apply(this, args);
    };
}
return window._fetchLog;
"""
driver.execute_script(js_intercept)

# 刷新页面让拦截生效
print("刷新页面以捕获API请求...")
driver.refresh()
time.sleep(5)

fetch_log = driver.execute_script("return window._fetchLog || []")
print(f"\n捕获到的fetch请求({len(fetch_log)}个):")
for req in fetch_log[:10]:
    print(f"  {req}")

# 检查XHR
print("\n检查performance entries...")
js_perf = """
var entries = performance.getEntriesByType('resource');
var apis = entries.filter(e => e.initiatorType === 'fetch' || e.initiatorType === 'xmlhttprequest');
return apis.map(e => ({name: e.name, type: e.initiatorType, duration: e.duration}));
"""
perf_entries = driver.execute_script(js_perf)
print(f"API请求({len(perf_entries)}个):")
for entry in perf_entries[:15]:
    print(f"  [{entry['type']}] {entry['name'][:100]}")

print("\n完成")
driver.quit()
