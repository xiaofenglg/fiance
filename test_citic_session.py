# -*- coding: utf-8 -*-
"""测试CITIC API - 使用session获取cookies"""
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
    print("测试CITIC API - 使用session")
    print("=" * 60)

    session = requests.Session()
    session.mount('https://', LegacySSLAdapter())

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Origin': 'https://www.citic-wealth.com',
        'Referer': 'https://www.citic-wealth.com/',
    }
    session.headers.update(headers)

    # 1. 先访问主页获取session
    print("\n1. 访问主页建立session...")
    try:
        resp = session.get("https://www.citic-wealth.com/yymk/lccs/", timeout=30)
        print(f"   状态: {resp.status_code}")
        print(f"   Cookies: {dict(session.cookies)}")
    except Exception as e:
        print(f"   错误: {e}")

    # 2. 访问iframe页面
    print("\n2. 访问iframe页面...")
    try:
        resp = session.get("https://wechat.citic-wealth.com/app/lccp/", timeout=30)
        print(f"   状态: {resp.status_code}")
        print(f"   Cookies: {dict(session.cookies)}")
    except Exception as e:
        print(f"   错误: {e}")

    # 3. 尝试调用webid API
    print("\n3. 调用webid API...")
    try:
        resp = session.post("https://wechat.citic-wealth.com/dmas/webid", timeout=30)
        print(f"   状态: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   响应: {resp.text[:200]}")
        print(f"   Cookies: {dict(session.cookies)}")
    except Exception as e:
        print(f"   错误: {e}")

    # 4. 再次测试fundList API
    print("\n4. 测试fundList API...")
    api_url = "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList"
    params = {
        'pageNum': 1,
        'pageSize': 10,
        'productType': 2,
        'prodSaleCustom': 0,
        'investment': '',
        'respProductType': '',
        'riskLevel': '',
    }

    try:
        resp = session.get(api_url, params=params, timeout=30)
        print(f"   状态: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   code: {data.get('code')}, msg: {data.get('msg')}")
            if 'data' in data:
                d = data['data']
                print(f"   total: {d.get('total')}, list: {len(d.get('list', []))}")
                if d.get('list'):
                    print(f"\n   产品列表:")
                    for p in d['list'][:5]:
                        print(f"      {p.get('prodCode')}: {p.get('productName', '')[:40]}")
    except Exception as e:
        print(f"   错误: {e}")

    # 5. 测试旧API (不带custom路径)
    print("\n5. 测试旧API...")
    old_api_url = "https://wechat.citic-wealth.com/cms.product/api/productInfo/fundList"
    params = {
        'pageNum': 1,
        'pageSize': 10,
    }

    try:
        resp = session.get(old_api_url, params=params, timeout=30)
        print(f"   状态: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   code: {data.get('code')}, msg: {data.get('msg')}")
            if 'data' in data:
                d = data['data']
                if isinstance(d, list):
                    print(f"   返回 {len(d)} 个产品")
                    if d:
                        print(f"   样例: {d[0]}")
                elif isinstance(d, dict):
                    print(f"   total: {d.get('total')}, list: {len(d.get('list', []))}")
    except Exception as e:
        print(f"   错误: {e}")

if __name__ == "__main__":
    main()
