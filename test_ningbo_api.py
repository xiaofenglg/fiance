"""
测试宁波银行理财API
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


def create_session():
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Content-Type': 'application/json',
        'Origin': 'https://www.wmbnb.com',
        'Referer': 'https://www.wmbnb.com/product/index.html',
    })
    session.verify = False
    return session


def test_ningbo():
    """测试宁波银行API"""
    print("="*60)
    print("宁波银行理财 API 测试")
    print("="*60)

    session = create_session()

    # 基于fund.js分析的API路径
    base_url = "https://www.wmbnb.com/ningbo-web"

    # 测试各种可能的端点
    endpoints = [
        # 产品列表
        ("/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("/product/queryList", "POST", {"pageNum": 1, "pageSize": 10, "curPage": 1}),
        ("/queryProductList", "POST", {"pageNum": 1, "pageSize": 10}),
        ("/fund/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("/fund/queryList", "POST", {"pageNum": 1, "pageSize": 10}),

        # 热门产品
        ("/product/hotList", "GET", {}),
        ("/queryHotList", "GET", {}),

        # 产品分类
        ("/product/tap", "GET", {}),
        ("/queryTap", "GET", {}),

        # 产品状态
        ("/product/state", "GET", {}),
    ]

    for path, method, data in endpoints:
        url = base_url + path
        try:
            if method == "POST":
                resp = session.post(url, json=data, timeout=15)
            else:
                resp = session.get(url, params=data, timeout=15)

            print(f"\n[{method}] {path}: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    result = resp.json()
                    print(f"  Response: {json.dumps(result, ensure_ascii=False)[:500]}")
                except:
                    print(f"  Content: {resp.text[:300]}")
        except Exception as e:
            print(f"\n[{method}] {path}: {e}")

    # 尝试form-data格式
    print("\n" + "-"*60)
    print("尝试 form-data 格式")
    print("-"*60)

    session.headers['Content-Type'] = 'application/x-www-form-urlencoded'

    form_endpoints = [
        ("/product/list", {"pageNum": 1, "pageSize": 10}),
        ("/queryProductList", {"pageNum": 1, "pageSize": 10}),
        ("/fund/list", {"curPage": 1, "pageCount": 10}),
    ]

    for path, data in form_endpoints:
        url = base_url + path
        try:
            resp = session.post(url, data=data, timeout=15)
            print(f"\n[POST form] {path}: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    result = resp.json()
                    print(f"  Response: {json.dumps(result, ensure_ascii=False)[:500]}")
                except:
                    print(f"  Content: {resp.text[:200]}")
        except Exception as e:
            print(f"\n[POST form] {path}: {e}")


if __name__ == "__main__":
    test_ningbo()
