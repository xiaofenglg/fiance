"""
探测各银行理财产品API接口
"""

import requests
import ssl
import urllib3
import json
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings()


class LegacySSLAdapter(HTTPAdapter):
    """支持旧版SSL的适配器"""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


def create_session():
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    })
    session.verify = False
    return session


def test_spdb():
    """浦发银行理财"""
    print("\n" + "="*60)
    print("浦发银行理财 (SPDB)")
    print("="*60)

    session = create_session()

    # 尝试常见的API端点
    endpoints = [
        ("https://www.spdb-wm.com/api/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.spdb-wm.com/api/products", "GET", {"page": 1, "size": 10}),
        ("https://www.spdb-wm.com/financialProducts/api/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.spdb-wm.com/spdbwm/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.spdb-wm.com/gw/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
    ]

    for url, method, params in endpoints:
        try:
            if method == "POST":
                resp = session.post(url, json=params, timeout=10)
            else:
                resp = session.get(url, params=params, timeout=10)

            print(f"\n[{method}] {url}")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  Response: {json.dumps(data, ensure_ascii=False)[:500]}")
                except:
                    print(f"  Content: {resp.text[:300]}")
        except Exception as e:
            print(f"\n[{method}] {url}")
            print(f"  Error: {e}")


def test_citic():
    """信银理财 (中信银行)"""
    print("\n" + "="*60)
    print("信银理财 (CITIC)")
    print("="*60)

    session = create_session()

    endpoints = [
        ("http://www.citic-wealth.com/api/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("http://www.citic-wealth.com/yymk/lccs/api/list", "GET", {}),
        ("http://www.citic-wealth.com/lccs/api/products", "POST", {"page": 1}),
        ("https://wealth.citicbank.com/wealth/api/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://wealth.citicbank.com/api/v1/products", "GET", {"page": 1}),
    ]

    for url, method, params in endpoints:
        try:
            if method == "POST":
                resp = session.post(url, json=params, timeout=10)
            else:
                resp = session.get(url, params=params, timeout=10)

            print(f"\n[{method}] {url}")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  Response: {json.dumps(data, ensure_ascii=False)[:500]}")
                except:
                    print(f"  Content: {resp.text[:300]}")
        except Exception as e:
            print(f"\n[{method}] {url}")
            print(f"  Error: {e}")


def test_ceb():
    """光大银行理财"""
    print("\n" + "="*60)
    print("光大银行理财 (CEB)")
    print("="*60)

    session = create_session()

    endpoints = [
        ("https://www.cebwm.com/api/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.cebwm.com/wealth/api/products", "GET", {"page": 1}),
        ("https://www.cebwm.com/grlc/api/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.cebwm.com/cebwm/product/queryList", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.cebwm.com/cebwm/v1/product/list", "POST", {"current": 1, "size": 10}),
    ]

    for url, method, params in endpoints:
        try:
            if method == "POST":
                resp = session.post(url, json=params, timeout=10)
            else:
                resp = session.get(url, params=params, timeout=10)

            print(f"\n[{method}] {url}")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  Response: {json.dumps(data, ensure_ascii=False)[:500]}")
                except:
                    print(f"  Content: {resp.text[:300]}")
        except Exception as e:
            print(f"\n[{method}] {url}")
            print(f"  Error: {e}")


def test_huaxia():
    """华夏银行理财"""
    print("\n" + "="*60)
    print("华夏银行理财 (Huaxia)")
    print("="*60)

    session = create_session()

    # 尝试JS文件和API
    endpoints = [
        ("https://www.hxwm.com.cn/common/js/gmjzbgData.js", "GET", {}),
        ("https://www.hxwm.com.cn/api/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.hxwm.com.cn/hxwm/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.hxwm.com.cn/xxpl/api/products", "GET", {"page": 1}),
    ]

    for url, method, params in endpoints:
        try:
            if method == "POST":
                resp = session.post(url, json=params, timeout=10)
            else:
                resp = session.get(url, params=params, timeout=10)

            print(f"\n[{method}] {url}")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  Response: {json.dumps(data, ensure_ascii=False)[:500]}")
                except:
                    print(f"  Content: {resp.text[:500]}")
        except Exception as e:
            print(f"\n[{method}] {url}")
            print(f"  Error: {e}")


def test_ningbo():
    """宁波银行理财"""
    print("\n" + "="*60)
    print("宁波银行理财 (Bank of Ningbo)")
    print("="*60)

    session = create_session()

    endpoints = [
        ("https://www.wmbnb.com/api/product/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.wmbnb.com/product/api/list", "GET", {"page": 1}),
        ("https://www.wmbnb.com/wmbnb/product/queryList", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://www.wmbnb.com/api/v1/products", "GET", {"page": 1, "size": 10}),
        ("https://www.wmbnb.com/nblc/api/product/list", "POST", {"current": 1, "size": 10}),
    ]

    for url, method, params in endpoints:
        try:
            if method == "POST":
                resp = session.post(url, json=params, timeout=10)
            else:
                resp = session.get(url, params=params, timeout=10)

            print(f"\n[{method}] {url}")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  Response: {json.dumps(data, ensure_ascii=False)[:500]}")
                except:
                    print(f"  Content: {resp.text[:300]}")
        except Exception as e:
            print(f"\n[{method}] {url}")
            print(f"  Error: {e}")


def test_all_home_pages():
    """测试所有首页是否可访问"""
    print("\n" + "="*60)
    print("测试各银行网站首页连通性")
    print("="*60)

    session = create_session()

    urls = [
        ("浦发银行", "https://www.spdb-wm.com/financialProducts/"),
        ("信银理财", "http://www.citic-wealth.com/yymk/lccs/"),
        ("光大银行", "https://www.cebwm.com/wealth/grlc/index.html"),
        ("华夏银行", "https://www.hxwm.com.cn/xxpl/gmcpgg/jzbg/index.shtml"),
        ("宁波银行", "https://www.wmbnb.com/product/index.html"),
    ]

    for name, url in urls:
        try:
            resp = session.get(url, timeout=15)
            print(f"\n{name}: {url}")
            print(f"  Status: {resp.status_code}")
            print(f"  Content-Type: {resp.headers.get('Content-Type', 'N/A')}")
            print(f"  Content Length: {len(resp.text)} chars")

            # 查找页面中可能的API线索
            text = resp.text.lower()
            if 'api' in text:
                # 尝试提取API URL
                import re
                api_matches = re.findall(r'["\']([^"\']*api[^"\']*)["\']', resp.text, re.IGNORECASE)
                if api_matches:
                    print(f"  Possible API endpoints found: {api_matches[:5]}")

        except Exception as e:
            print(f"\n{name}: {url}")
            print(f"  Error: {e}")


if __name__ == "__main__":
    # 先测试首页连通性
    test_all_home_pages()

    # 测试各银行API
    test_spdb()
    test_citic()
    test_ceb()
    test_huaxia()
    test_ningbo()

    print("\n" + "="*60)
    print("探测完成")
    print("="*60)
