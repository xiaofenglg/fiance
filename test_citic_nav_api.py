# -*- coding: utf-8 -*-
"""测试中信理财历史净值API"""
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
    print("测试中信理财历史净值API")
    print("=" * 60)

    session = requests.Session()
    session.mount('https://', LegacySSLAdapter())

    # 测试产品代码
    product_code = "AF245689D"
    prod_type = 2  # 每日购

    # 测试NAV API
    print(f"\n1. 测试 getTAProductNav API...")
    nav_url = f"https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/getTAProductNav?prodCode={product_code}&queryUnit=1"
    print(f"   URL: {nav_url}")

    try:
        resp = session.get(nav_url, timeout=30)
        print(f"   状态: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   响应结构: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            print(f"   完整响应:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"   错误: {e}")

    # 测试Detail API
    print(f"\n2. 测试 getTAProductDetail API...")
    detail_url = f"https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/getTAProductDetail?prodCode={product_code}&prodType={prod_type}"
    print(f"   URL: {detail_url}")

    try:
        resp = session.get(detail_url, timeout=30)
        print(f"   状态: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   响应结构: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            # 只打印部分关键字段
            if 'data' in data:
                d = data['data']
                print(f"   产品名称: {d.get('productName', 'N/A')}")
                print(f"   产品代码: {d.get('prodCode', 'N/A')}")
                print(f"   风险等级: {d.get('riskLevel', 'N/A')}")
                print(f"   净值: {d.get('nav', 'N/A')}")
                print(f"   净值日期: {d.get('navDate', 'N/A')}")
                # 查找历史净值相关字段
                for key in d.keys():
                    if 'nav' in key.lower() or 'history' in key.lower():
                        print(f"   {key}: {d[key]}")
    except Exception as e:
        print(f"   错误: {e}")

    # 测试不同的queryUnit值
    print(f"\n3. 测试不同的queryUnit值...")
    for unit in [1, 2, 3, 4, 5]:
        nav_url = f"https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/getTAProductNav?prodCode={product_code}&queryUnit={unit}"
        try:
            resp = session.get(nav_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data and data['data']:
                    nav_list = data['data']
                    count = len(nav_list) if isinstance(nav_list, list) else 0
                    print(f"   queryUnit={unit}: {count} 条数据")
                    if count > 0 and isinstance(nav_list, list):
                        print(f"      第一条: {nav_list[0]}")
                        print(f"      最后一条: {nav_list[-1]}")
                else:
                    print(f"   queryUnit={unit}: 无数据")
        except Exception as e:
            print(f"   queryUnit={unit}: 错误 {e}")

    # 测试fundList API获取产品列表
    print(f"\n4. 测试 fundList API...")
    list_url = "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList?pageNum=1&pageSize=5&productType=2&prodSaleCustom=0"
    print(f"   URL: {list_url}")

    try:
        resp = session.get(list_url, timeout=30)
        print(f"   状态: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data and 'list' in data['data']:
                products = data['data']['list']
                print(f"   获取到 {len(products)} 个产品:")
                for p in products[:3]:
                    print(f"      - {p.get('prodCode')}: {p.get('productName', '')[:40]}...")
                    print(f"        净值: {p.get('nav')}, 日期: {p.get('navDate')}")
    except Exception as e:
        print(f"   错误: {e}")

if __name__ == "__main__":
    main()
