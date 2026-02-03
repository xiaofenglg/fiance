"""
宁波银行API深度测试
基于fund.js分析：url: '/ningbo-web' 是基础路径
"""

import requests
import ssl
import urllib3
import json
from requests.adapters import HTTPAdapter
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


def test():
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount('https://', adapter)
    session.verify = False

    base = "https://www.wmbnb.com"

    # 从fund.js分析：
    # 1. quertList() 函数调用 - 产品列表
    # 2. queryHotList() 函数调用 - 热门产品
    # 3. queryTap() 函数调用 - 产品分类
    # 4. querysearchList() 函数调用 - 搜索

    # 测试路径 - 基于Vue代码分析
    paths = [
        # 带不同参数组合的路径
        ("/ningbo-web/quertList", {"curPage": 1, "pageCount": 10}),
        ("/ningbo-web/queryHotList", {}),
        ("/ningbo-web/queryTap", {"tap": "productstate"}),
        ("/ningbo-web/queryTap", {"tap": "conduct"}),
        ("/ningbo-web/queryTap", {"tap": "profucat"}),

        # 可能的API命名
        ("/ningbo-web/api/product/list", {"pageNum": 1, "pageSize": 10}),
        ("/ningbo-web/api/fund/list", {"pageNum": 1, "pageSize": 10}),

        # 直接的路径
        ("/ningbo-web/productList", {"curPage": 1}),
        ("/ningbo-web/fundList", {"curPage": 1}),
    ]

    print("="*60)
    print("宁波银行 API 测试")
    print("="*60)

    for path, params in paths:
        url = base + path

        # 尝试GET
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://www.wmbnb.com/product/index.html',
            }

            resp = session.get(url, params=params, headers=headers, timeout=10)
            print(f"\n[GET] {path}")
            print(f"  Params: {params}")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                content = resp.text[:500]
                print(f"  Content: {content}")
                if resp.text.startswith('{') or resp.text.startswith('['):
                    try:
                        data = resp.json()
                        print(f"  JSON Keys: {list(data.keys()) if isinstance(data, dict) else 'array'}")
                    except:
                        pass
        except Exception as e:
            print(f"\n[GET] {path}: {e}")

        # 尝试POST (JSON)
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Content-Type': 'application/json',
                'Referer': 'https://www.wmbnb.com/product/index.html',
            }

            resp = session.post(url, json=params, headers=headers, timeout=10)
            if resp.status_code == 200 and (resp.text.startswith('{') or resp.text.startswith('[')):
                print(f"\n[POST JSON] {path}")
                print(f"  Status: {resp.status_code}")
                print(f"  Content: {resp.text[:500]}")
        except:
            pass

        # 尝试POST (Form)
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Referer': 'https://www.wmbnb.com/product/index.html',
            }

            resp = session.post(url, data=params, headers=headers, timeout=10)
            if resp.status_code == 200 and (resp.text.startswith('{') or resp.text.startswith('[')):
                print(f"\n[POST Form] {path}")
                print(f"  Status: {resp.status_code}")
                print(f"  Content: {resp.text[:500]}")
        except:
            pass

    # 尝试直接访问一些静态数据文件
    print("\n" + "="*60)
    print("测试静态数据文件")
    print("="*60)

    static_files = [
        "/data/products.json",
        "/data/fund.json",
        "/product/data.json",
        "/ningbo-web/data/products.json",
        "/api/products.json",
    ]

    for path in static_files:
        try:
            resp = session.get(base + path, timeout=10)
            print(f"\n{path}: {resp.status_code}")
            if resp.status_code == 200:
                print(f"  Content: {resp.text[:300]}")
        except Exception as e:
            print(f"\n{path}: {e}")


if __name__ == "__main__":
    test()
