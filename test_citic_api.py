"""探测中信理财API结构"""
import requests
import re
import json
from bs4 import BeautifulSoup

# 禁用SSL警告
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
})
session.verify = False

base_url = "https://www.citic-wealth.com"

# 1. 访问净值查询主页
print("="*60)
print("1. 访问净值查询主页")
print("="*60)
try:
    resp = session.get(f"{base_url}/yymk/lccs/", timeout=30)
    print(f"状态码: {resp.status_code}")

    # 查找JS文件
    js_files = re.findall(r'src=["\']([^"\']*\.js[^"\']*)["\']', resp.text)
    print(f"找到 {len(js_files)} 个JS文件:")
    for js in js_files[:10]:
        print(f"  - {js}")

    # 查找可能的API调用
    api_patterns = re.findall(r'["\']([^"\']*(?:api|query|list|product)[^"\']*)["\']', resp.text, re.I)
    print(f"\n可能的API路径:")
    for api in set(api_patterns)[:20]:
        print(f"  - {api}")

    # 查找产品链接
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = soup.find_all('a', href=True)
    product_links = [a['href'] for a in links if 'product' in a['href'].lower() or 'detail' in a['href'].lower() or 'xq' in a['href'].lower()]
    print(f"\n可能的产品链接:")
    for link in set(product_links)[:10]:
        print(f"  - {link}")

except Exception as e:
    print(f"错误: {e}")

# 2. 尝试常见API路径
print("\n" + "="*60)
print("2. 尝试常见API路径")
print("="*60)

api_tests = [
    "/yymk/lccs/getProductList",
    "/yymk/lccs/productList",
    "/api/product/list",
    "/api/lccs/list",
    "/yymk/api/product",
    "/portal/api/product/list",
    "/wealth/product/list",
]

for api in api_tests:
    try:
        url = f"{base_url}{api}"
        resp = session.get(url, timeout=10)
        content_type = resp.headers.get('Content-Type', '')
        print(f"{api}: {resp.status_code} ({content_type[:30]})")
        if resp.status_code == 200 and 'json' in content_type:
            print(f"  数据: {resp.text[:200]}")
    except Exception as e:
        print(f"{api}: 错误 - {e}")

# 3. 尝试POST请求
print("\n" + "="*60)
print("3. 尝试POST请求")
print("="*60)

post_tests = [
    ("/yymk/lccs/getProductList", {"type": "1"}),
    ("/yymk/lccs/productList", {"category": "mrgxq"}),
    ("/api/product/query", {"pageNo": 1, "pageSize": 10}),
]

for api, data in post_tests:
    try:
        url = f"{base_url}{api}"
        resp = session.post(url, json=data, timeout=10)
        content_type = resp.headers.get('Content-Type', '')
        print(f"POST {api}: {resp.status_code} ({content_type[:30]})")
        if resp.status_code == 200:
            print(f"  响应: {resp.text[:300]}")
    except Exception as e:
        print(f"POST {api}: 错误 - {e}")

print("\n" + "="*60)
print("完成")
print("="*60)
