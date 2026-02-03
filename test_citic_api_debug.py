# -*- coding: utf-8 -*-
"""调试CITIC API"""
import requests
import ssl
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

class LegacySSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

def main():
    print("=" * 60)
    print("调试CITIC API")
    print("=" * 60)

    session = requests.Session()
    session.mount('https://', LegacySSLAdapter())

    # 测试fundList API
    api_base = "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo"

    # 尝试不同的productType值
    print("\n1. 测试不同的productType值...")
    for pt in [1, 2, 3, 4, 5]:
        url = f"{api_base}/fundList"
        params = {
            'pageNum': 1,
            'pageSize': 5,
            'productType': pt,
            'prodSaleCustom': 0,
        }

        try:
            resp = session.get(url, params=params, timeout=30)
            print(f"\n   productType={pt}:")
            print(f"   状态: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"   code: {data.get('code')}, msg: {data.get('msg')}")
                if 'data' in data and data['data']:
                    d = data['data']
                    if isinstance(d, dict):
                        print(f"   total: {d.get('total')}, list count: {len(d.get('list', []))}")
                        if d.get('list'):
                            p = d['list'][0]
                            print(f"   样例: {p.get('prodCode')} - {p.get('productName', '')[:30]}...")
                    else:
                        print(f"   data类型: {type(d)}")
        except Exception as e:
            print(f"   错误: {e}")

    # 测试直接URL(从网络捕获的)
    print("\n\n2. 测试网络捕获的完整URL...")
    full_urls = [
        "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList?pageNum=1&pageSize=20&productType=2&prodSaleCustom=0&investment=&respProductType=&riskLevel=",
        "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList?pageNum=1&pageSize=20&productType=3&prodSaleCustom=0&investment=&respProductType=&riskLevel=",
    ]

    for url in full_urls:
        print(f"\n   URL: {url[:80]}...")
        try:
            resp = session.get(url, timeout=30)
            print(f"   状态: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"   code: {data.get('code')}, msg: {data.get('msg')}")
                if 'data' in data and data['data']:
                    d = data['data']
                    print(f"   total: {d.get('total')}, list: {len(d.get('list', []))}")
        except Exception as e:
            print(f"   错误: {e}")

    # 测试添加headers
    print("\n\n3. 测试添加headers...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Origin': 'https://www.citic-wealth.com',
        'Referer': 'https://www.citic-wealth.com/',
    }

    url = f"{api_base}/fundList?pageNum=1&pageSize=20&productType=2&prodSaleCustom=0&investment=&respProductType=&riskLevel="
    try:
        resp = session.get(url, headers=headers, timeout=30)
        print(f"   状态: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   code: {data.get('code')}, msg: {data.get('msg')}")
            if 'data' in data and data['data']:
                d = data['data']
                print(f"   total: {d.get('total')}, list: {len(d.get('list', []))}")
                if d.get('list'):
                    print(f"\n   前3个产品:")
                    for p in d['list'][:3]:
                        print(f"      {p.get('prodCode')}: {p.get('productName', '')[:40]}")
    except Exception as e:
        print(f"   错误: {e}")

if __name__ == "__main__":
    main()
