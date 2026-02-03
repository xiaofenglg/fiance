"""搜索中信理财特定产品"""
import requests
from requests.adapters import HTTPAdapter
import ssl
import json
import urllib3
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings()

class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

def create_session():
    session = requests.Session()
    adapter = LegacySSLAdapter(pool_connections=10, pool_maxsize=10)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; MI 8) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://wechat.citic-wealth.com/wechat/product/',
    })
    session.verify = False
    return session

def main():
    session = create_session()
    session.get("https://wechat.citic-wealth.com/wechat/product/", timeout=15)

    print("=" * 60)
    print("Search for specific CITIC products")
    print("=" * 60)

    base_url = "https://wechat.citic-wealth.com"

    # Products seen in the iframe
    target_codes = ["AF251875D", "AF253315D", "AF252217B", "AF53042B"]

    # 1. Test search API with product code
    print("\n1. Test search API with product codes...")

    search_apis = [
        "/cms.product/api/productInfo/search",
        "/cms.product/api/productInfo/fundList",
        "/cms.product/api/custom/productInfo/search",
        "/cms.product/api/custom/productInfo/fundList",
    ]

    for code in target_codes[:2]:
        print(f"\n   Searching for: {code}")

        for api in search_apis:
            # Try different parameter names
            param_variants = [
                {"fundCode": code},
                {"productCode": code},
                {"code": code},
                {"keyword": code},
                {"searchKey": code},
                {"productName": code},
            ]

            for params in param_variants:
                full_params = {"pageNo": 1, "pageSize": 20, **params}
                try:
                    resp = session.get(f"{base_url}{api}", params=full_params, timeout=10)
                    if 'json' in resp.headers.get('Content-Type', '').lower():
                        data = resp.json()
                        if data.get('code') == '0000':
                            result = data.get('data', [])
                            if isinstance(result, list):
                                found = [p for p in result if code in p.get('fundCode', '')]
                                if found:
                                    print(f"      Found! API: {api}, param: {list(params.keys())[0]}")
                                    print(f"      Product: {json.dumps(found[0], ensure_ascii=False)[:200]}")
                except:
                    pass

    # 2. Get ALL products and search manually
    print("\n\n2. Get all products and search manually...")

    all_products = []
    page = 1
    while True:
        resp = session.get(
            f"{base_url}/cms.product/api/productInfo/fundList",
            params={"pageNo": page, "pageSize": 500},
            timeout=30
        )
        data = resp.json()
        products = data.get('data', [])
        if not products:
            break
        all_products.extend(products)
        page += 1
        if page > 10:  # Safety limit
            break

    print(f"   Total products retrieved: {len(all_products)}")

    # Search for target codes
    for code in target_codes:
        found = [p for p in all_products if code in p.get('fundCode', '')]
        if found:
            print(f"   Found {code}: Yes")
        else:
            print(f"   Found {code}: No")

    # 3. Try different API domains/paths
    print("\n\n3. Try alternative API paths...")

    alt_apis = [
        ("https://wechat.citic-wealth.com/cms.product/api/productMarket/list", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/cms.product/api/productMarket/fundList", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/cms.product/api/custom/productMarket/list", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/cms.product/api/custom/productMarket/fundList", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/cms.product/api/productList", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/cms.product/api/fundQuery/list", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/cms.product/api/productQuery/list", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/wechat/api/productList", {"pageNo": 1, "pageSize": 50}),
        ("https://wechat.citic-wealth.com/wechat/api/productMarket/list", {"pageNo": 1, "pageSize": 50}),
    ]

    for url, params in alt_apis:
        try:
            resp = session.get(url, params=params, timeout=10)
            ct = resp.headers.get('Content-Type', '')
            if 'json' in ct.lower():
                data = resp.json()
                code = data.get('code', '')
                if code == '0000' and data.get('data'):
                    result = data.get('data')
                    if isinstance(result, list) and len(result) > 0:
                        print(f"   {url.split('/')[-1]}: {len(result)} products")
                        # Check for target products
                        codes_found = [p.get('fundCode') for p in result if p.get('fundCode') in target_codes]
                        if codes_found:
                            print(f"      Found target codes: {codes_found}")
                    elif isinstance(result, dict):
                        records = result.get('records', []) or result.get('list', [])
                        if records:
                            print(f"   {url.split('/')[-1]}: {len(records)} products (in records)")
        except Exception as e:
            pass

    # 4. Check if products are retrieved with different category
    print("\n\n4. Test category parameters for productInfo/fundList...")

    category_tests = [
        {"mainCategory": "1"},  # 个人
        {"mainCategory": "2"},  # 机构
        {"typeCategory": "1"},  # 募集期
        {"typeCategory": "2"},  # 每日购
        {"typeCategory": "3"},  # 定开
        {"typeCategory": "4"},  # 货币
        {"typeCategory": "5"},  # 封闭
        {"customerType": "1"},
        {"customerType": "2"},
        {"channel": "web"},
        {"channel": "wechat"},
        {"source": "pc"},
        {"source": "h5"},
    ]

    for params in category_tests:
        full_params = {"pageNo": 1, "pageSize": 100, **params}
        try:
            resp = session.get(
                f"{base_url}/cms.product/api/productInfo/fundList",
                params=full_params,
                timeout=10
            )
            if 'json' in resp.headers.get('Content-Type', '').lower():
                data = resp.json()
                if data.get('code') == '0000':
                    products = data.get('data', [])
                    codes = [p.get('fundCode') for p in products[:5]]
                    # Check if any target code is found
                    has_target = any(c in target_codes for c in [p.get('fundCode') for p in products])
                    print(f"   {params}: {len(products)} products, has_target={has_target}")
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
