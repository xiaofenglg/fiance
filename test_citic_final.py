"""测试中信理财实际API"""
import requests
import json
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Content-Type': 'application/json',
    'Origin': 'https://www.citic-wealth.com',
    'Referer': 'https://www.citic-wealth.com/wechat/product/',
})
session.verify = False

base_url = "https://www.citic-wealth.com"

# 关键API列表
apis = [
    # 产品列表
    ("/cms.product/api/productInfo/search", "GET", {"pageNo": 1, "pageSize": 10}),
    ("/cms.product/api/productInfo/fundList", "GET", {"pageNo": 1, "pageSize": 10}),
    ("/cms.product/api/productInfo/fundTypeList", "GET", {}),
    ("/cms.product/api/custom/productInfo/search", "GET", {"pageNo": 1, "pageSize": 10}),
    ("/cms.product/api/productRecommended/list", "GET", {}),

    # 产品详情
    ("/cms.product/api/productInfo/fundDetail", "GET", {"productCode": "test"}),
    ("/cms.product/api/custom/productInfo/getTAProductDetail", "GET", {"productCode": "test"}),

    # 净值历史
    ("/cms.product/api/productInfo/fundNetValues", "GET", {"productCode": "test"}),
    ("/cms.product/api/custom/productInfo/getTAProductNav", "GET", {"productCode": "test"}),
    ("/cms.product/api/custom/productInfo/getHistoryProductNav", "GET", {"productCode": "test", "pageNo": 1, "pageSize": 30}),
]

print("="*70)
print("测试中信理财API")
print("="*70)

for api, method, params in apis:
    url = f"{base_url}{api}"
    try:
        if method == "GET":
            resp = session.get(url, params=params, timeout=15)
        else:
            resp = session.post(url, json=params, timeout=15)

        ct = resp.headers.get('Content-Type', '')[:40]
        print(f"\n{method} {api}")
        print(f"  状态: {resp.status_code}, 类型: {ct}")

        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"  响应: {json.dumps(data, ensure_ascii=False)[:500]}")
            except:
                print(f"  响应(文本): {resp.text[:200]}")
    except Exception as e:
        print(f"\n{method} {api}")
        print(f"  错误: {e}")

# 尝试不同的参数
print("\n" + "="*70)
print("尝试fundList的不同参数")
print("="*70)

fund_params = [
    {"pageNo": 1, "pageSize": 10},
    {"pageNo": 1, "pageSize": 10, "productType": "1"},
    {"pageNo": 1, "pageSize": 10, "productType": "mrg"},
    {"pageNo": 1, "pageSize": 10, "type": "每日购"},
    {"page": 1, "rows": 10},
]

for params in fund_params:
    try:
        resp = session.get(f"{base_url}/cms.product/api/productInfo/fundList", params=params, timeout=15)
        print(f"\n参数: {params}")
        print(f"  状态: {resp.status_code}")
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"  响应: {json.dumps(data, ensure_ascii=False)[:400]}")
            except:
                print(f"  响应: {resp.text[:200]}")
    except Exception as e:
        print(f"  错误: {e}")

print("\n完成!")
