"""
深入探测各银行理财产品API接口 v3
根据v2发现的线索进一步探测
"""

import requests
import ssl
import urllib3
import json
import re
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
    session.mount('http://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    })
    session.verify = False
    return session


def test_spdb_real():
    """浦发银行 - 真实理财产品页面"""
    print("\n" + "="*60)
    print("浦发银行理财 - 真实入口")
    print("="*60)

    session = create_session()

    # 访问真实的理财产品页面
    try:
        resp = session.get("https://per.spdb.com.cn/bank_financing/financial_product/", timeout=15)
        resp.encoding = 'utf-8'
        print(f"理财产品页面状态: {resp.status_code}")

        # 查找API
        if resp.status_code == 200:
            # 查找JS文件
            js_files = re.findall(r'src=["\']([^"\']*\.js[^"\']*)["\']', resp.text)
            print(f"JS文件: {js_files[:10]}")

            # 查找API路径
            api_patterns = re.findall(r'["\']([^"\']*(?:/api/|/product|/list|query)[^"\']*)["\']', resp.text, re.IGNORECASE)
            print(f"API路径: {list(set(api_patterns))[:15]}")

            # 查找数据URL
            data_urls = re.findall(r'(?:url|action|src)\s*[:=]\s*["\']([^"\']+)["\']', resp.text)
            print(f"数据URL: {list(set(data_urls))[:10]}")

    except Exception as e:
        print(f"Error: {e}")

    # 尝试常见API路径
    api_tests = [
        ("https://per.spdb.com.cn/bank_financing/financial_product/api/list", "POST", {"pageNum": 1, "pageSize": 10}),
        ("https://per.spdb.com.cn/was5/web/search", "POST", {"searchword": "理财", "page": 1}),
        ("https://per.spdb.com.cn/ares/search/list", "POST", {"keyword": "", "pageNum": 1, "pageSize": 10}),
    ]

    for url, method, data in api_tests:
        try:
            if method == "POST":
                resp = session.post(url, json=data, timeout=10)
            else:
                resp = session.get(url, timeout=10)
            print(f"\n[{method}] {url}: {resp.status_code}")
            if resp.status_code == 200:
                print(f"  Content: {resp.text[:300]}")
        except Exception as e:
            print(f"\n[{method}] {url}: {e}")


def test_ningbo_fund_js():
    """宁波银行 - 分析fund.js"""
    print("\n" + "="*60)
    print("宁波银行理财 - 分析fund.js")
    print("="*60)

    session = create_session()

    # 获取fund.js
    try:
        resp = session.get("https://www.wmbnb.com/js/fund.js", timeout=15)
        resp.encoding = 'utf-8'
        print(f"fund.js状态: {resp.status_code}")

        if resp.status_code == 200:
            content = resp.text
            # 查找API调用
            api_calls = re.findall(r'(?:get|post|ajax|fetch)\s*\(["\']([^"\']+)["\']', content, re.IGNORECASE)
            print(f"API调用: {api_calls}")

            # 查找URL模式
            urls = re.findall(r'["\']([^"\']*(?:api|query|list|product)[^"\']*)["\']', content, re.IGNORECASE)
            print(f"URL模式: {list(set(urls))[:10]}")

            # 打印部分内容看结构
            print(f"\n前1000字符:\n{content[:1000]}")

    except Exception as e:
        print(f"Error: {e}")

    # 获取index.html.js
    try:
        resp = session.get("https://www.wmbnb.com/common/index.html.js", timeout=15)
        resp.encoding = 'utf-8'
        print(f"\nindex.html.js状态: {resp.status_code}")

        if resp.status_code == 200:
            content = resp.text
            # 查找API调用
            api_calls = re.findall(r'(?:get|post|ajax|fetch)\s*\(["\']([^"\']+)["\']', content, re.IGNORECASE)
            print(f"API调用: {api_calls}")

            # 查找baseURL
            base_urls = re.findall(r'baseURL\s*[:=]\s*["\']([^"\']+)["\']', content)
            print(f"Base URLs: {base_urls}")

    except Exception as e:
        print(f"Error: {e}")


def test_citic_encoding_fix():
    """信银理财 - 修复编码问题"""
    print("\n" + "="*60)
    print("信银理财 - 修复编码并获取数据")
    print("="*60)

    session = create_session()

    try:
        # 尝试直接获取HTML并查看结构
        resp = session.get("http://www.citic-wealth.com/yymk/lccs/", timeout=15)
        # 不指定encoding，让它自动检测
        content = resp.content.decode('utf-8', errors='ignore')
        print(f"首页状态: {resp.status_code}")
        print(f"内容长度: {len(content)}")

        # 查找Vue/React数据
        data_match = re.search(r'(?:window\.__INITIAL_STATE__|__PRELOADED_STATE__|data\s*=)\s*({.*?})', content, re.DOTALL)
        if data_match:
            print(f"找到初始数据: {data_match.group(1)[:500]}")

        # 查找script标签中的数据
        script_data = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        for i, script in enumerate(script_data[:5]):
            if 'product' in script.lower() or 'list' in script.lower():
                print(f"\nScript {i}: {script[:500]}")

    except Exception as e:
        print(f"Error: {e}")


def test_ceb_with_cookies():
    """光大银行 - 尝试用完整浏览器头"""
    print("\n" + "="*60)
    print("光大银行理财 - 尝试完整请求")
    print("="*60)

    session = create_session()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }

    # 尝试直接访问API
    api_tests = [
        "https://www.cebwm.com/ebank/wealth/product/list",
        "https://www.cebwm.com/wealth/api/product/list",
        "https://www.cebwm.com/cebwm/api/product/queryList",
    ]

    for url in api_tests:
        try:
            resp = session.get(url, headers=headers, timeout=10)
            print(f"\n{url}: {resp.status_code}")
            print(f"Headers: {dict(resp.headers)}")
        except Exception as e:
            print(f"\n{url}: {e}")

    # 尝试找到替代入口
    alt_urls = [
        "https://www.cebbank.com/eportal/ui?pageId=477257&productType=LCCP",  # 光大银行网银
        "https://www.cebbank.com/site/gryw/yglc/lccp/index.html",
    ]

    for url in alt_urls:
        try:
            resp = session.get(url, headers=headers, timeout=10)
            print(f"\n{url}: {resp.status_code}")
            if resp.status_code == 200 and len(resp.text) > 100:
                # 查找API
                apis = re.findall(r'["\']([^"\']*(?:api|product|list)[^"\']*)["\']', resp.text, re.IGNORECASE)
                print(f"  API线索: {list(set(apis))[:10]}")
        except Exception as e:
            print(f"\n{url}: {e}")


def test_huaxia_product_list():
    """华夏银行 - 获取产品列表API"""
    print("\n" + "="*60)
    print("华夏银行理财 - 查找产品列表API")
    print("="*60)

    session = create_session()

    # 尝试不同的数据JS文件
    js_files = [
        "https://www.hxwm.com.cn/common/js/gmjzbgData.js",
        "https://www.hxwm.com.cn/common/js/productData.js",
        "https://www.hxwm.com.cn/common/js/cpggData.js",
        "https://www.hxwm.com.cn/js/productList.js",
    ]

    for url in js_files:
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                content = resp.content.decode('utf-8', errors='ignore')
                print(f"\n{url}: {resp.status_code}")
                print(f"  内容前500字符: {content[:500]}")
        except Exception as e:
            print(f"\n{url}: {e}")

    # 尝试API端点
    api_tests = [
        ("https://www.hxwm.com.cn/hxwm/api/product/list", "POST", {"pageNum": 1, "pageSize": 20}),
        ("https://www.hxwm.com.cn/api/wealth/products", "GET", {}),
    ]

    for url, method, data in api_tests:
        try:
            if method == "POST":
                resp = session.post(url, json=data, timeout=10)
            else:
                resp = session.get(url, timeout=10)
            print(f"\n[{method}] {url}: {resp.status_code}")
        except Exception as e:
            print(f"\n[{method}] {url}: {e}")


if __name__ == "__main__":
    test_spdb_real()
    test_ningbo_fund_js()
    test_citic_encoding_fix()
    test_huaxia_product_list()
    test_ceb_with_cookies()

    print("\n" + "="*60)
    print("探测完成")
    print("="*60)
