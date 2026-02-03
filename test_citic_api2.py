"""分析中信理财前端应用API"""
import requests
import re
import json
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Origin': 'https://www.citic-wealth.com',
    'Referer': 'https://www.citic-wealth.com/wechat/product/',
})
session.verify = False

base_url = "https://www.citic-wealth.com"

# 1. 获取前端应用页面
print("="*60)
print("1. 获取前端应用页面")
print("="*60)
resp = session.get(f"{base_url}/wechat/product/", timeout=30)
print(f"状态码: {resp.status_code}")
print(f"页面长度: {len(resp.text)} 字符")

# 保存用于分析
with open("citic_wechat.html", "w", encoding="utf-8") as f:
    f.write(resp.text)

# 查找JS文件
js_files = re.findall(r'src=["\']([^"\']*\.js[^"\']*)["\']', resp.text)
print(f"\nJS文件:")
for js in js_files[:10]:
    print(f"  {js}")

# 2. 尝试常见的API端点
print("\n" + "="*60)
print("2. 尝试wechat API端点")
print("="*60)

api_tests = [
    "/wechat/product/api/productList",
    "/wechat/product/api/list",
    "/wechat/api/product/list",
    "/wechat/api/productMarket",
    "/wechat/product/productList",
    "/wechat/productMarket/list",
    "/api/wechat/product/list",
    "/portal/product/list",
    "/portal/api/product/list",
    "/portal/api/queryProductList",
]

for api in api_tests:
    try:
        url = f"{base_url}{api}"
        resp = session.get(url, timeout=10)
        ct = resp.headers.get('Content-Type', '')[:40]
        print(f"GET {api}: {resp.status_code} ({ct})")
        if resp.status_code == 200 and 'json' in ct.lower():
            print(f"    数据: {resp.text[:300]}")
    except Exception as e:
        print(f"GET {api}: 错误")

# 3. 尝试POST请求
print("\n" + "="*60)
print("3. 尝试POST请求")
print("="*60)

post_tests = [
    ("/wechat/product/api/productList", {"type": "mrg", "pageNo": 1, "pageSize": 10}),
    ("/wechat/product/api/productList", {"productType": "1", "page": 1, "rows": 10}),
    ("/portal/api/product/queryList", {"pageNo": 1, "pageSize": 10, "type": "mrg"}),
    ("/portal/product/queryList", {"category": "每日购"}),
]

for api, data in post_tests:
    try:
        url = f"{base_url}{api}"
        resp = session.post(url, json=data, timeout=10)
        ct = resp.headers.get('Content-Type', '')[:40]
        print(f"POST {api}: {resp.status_code} ({ct})")
        if resp.status_code == 200 and ('json' in ct.lower() or resp.text.startswith('{')):
            print(f"    数据: {resp.text[:300]}")
    except Exception as e:
        print(f"POST {api}: 错误 - {str(e)[:50]}")

# 4. 分析JS文件找API
print("\n" + "="*60)
print("4. 分析主JS文件")
print("="*60)

# 找到main js文件
main_js = [js for js in js_files if 'main' in js.lower() or 'app' in js.lower() or 'chunk' in js.lower()]
if main_js:
    for js_url in main_js[:2]:
        if not js_url.startswith('http'):
            js_url = f"{base_url}/wechat/product/{js_url.lstrip('./')}"
        try:
            resp = session.get(js_url, timeout=20)
            if resp.status_code == 200:
                # 查找API路径
                apis = re.findall(r'["\']([^"\']*(?:api|query|list|product)[^"\']*)["\']', resp.text, re.I)
                print(f"\n{js_url} 中的API路径:")
                for api in list(set(apis))[:20]:
                    if '/' in api and len(api) < 100:
                        print(f"  {api}")
        except:
            pass

print("\n完成!")
