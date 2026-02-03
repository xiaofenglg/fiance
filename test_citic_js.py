"""分析中信理财app.js找API"""
import requests
import re
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
})
session.verify = False

base_url = "https://www.citic-wealth.com"

# 下载app.js
print("下载app.js...")
js_url = f"{base_url}/wechat/product/static/js/app.1165e0f4d708f887b333.js"
resp = session.get(js_url, timeout=30)
print(f"状态码: {resp.status_code}, 大小: {len(resp.text)} 字符")

js_content = resp.text

# 保存
with open("citic_app.js", "w", encoding="utf-8") as f:
    f.write(js_content)

# 查找API路径
print("\n" + "="*60)
print("查找API路径")
print("="*60)

# 查找包含api/http/url的字符串
api_patterns = [
    r'["\'](/[a-zA-Z0-9/_-]*(?:api|query|list|product|nav)[a-zA-Z0-9/_-]*)["\']',
    r'url:\s*["\']([^"\']+)["\']',
    r'baseURL:\s*["\']([^"\']+)["\']',
    r'axios\.[a-z]+\(["\']([^"\']+)["\']',
]

all_apis = set()
for pattern in api_patterns:
    matches = re.findall(pattern, js_content, re.I)
    for m in matches:
        if '/' in m and len(m) < 150 and not m.endswith('.js') and not m.endswith('.css'):
            all_apis.add(m)

print(f"找到 {len(all_apis)} 个可能的API路径:")
for api in sorted(all_apis):
    print(f"  {api}")

# 查找产品相关函数
print("\n" + "="*60)
print("查找产品相关关键词")
print("="*60)

keywords = ['productList', 'productDetail', 'navHistory', 'netValue', 'queryProduct', 'getProduct']
for kw in keywords:
    if kw.lower() in js_content.lower():
        # 找到附近的上下文
        idx = js_content.lower().find(kw.lower())
        context = js_content[max(0,idx-50):idx+100]
        print(f"\n{kw}:")
        print(f"  ...{context}...")

# 查找baseURL配置
print("\n" + "="*60)
print("查找baseURL配置")
print("="*60)
base_matches = re.findall(r'baseURL["\s:]+["\']([^"\']+)["\']', js_content)
for m in base_matches:
    print(f"  baseURL: {m}")

print("\n完成!")
