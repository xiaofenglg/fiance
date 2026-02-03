"""
深入探测各银行理财产品API接口 v2
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
        'Accept-Encoding': 'gzip, deflate, br',
    })
    session.verify = False
    return session


def test_citic_detailed():
    """信银理财 - 深入探测"""
    print("\n" + "="*60)
    print("信银理财 (CITIC) - 深入探测")
    print("="*60)

    session = create_session()

    # 先获取首页，看看有什么JS或API
    try:
        resp = session.get("http://www.citic-wealth.com/yymk/lccs/", timeout=15)
        resp.encoding = 'utf-8'
        print(f"首页状态: {resp.status_code}")

        # 查找JS文件和API
        js_files = re.findall(r'src=["\']([^"\']*\.js[^"\']*)["\']', resp.text)
        print(f"JS文件: {js_files[:10]}")

        api_hints = re.findall(r'["\']([^"\']*(?:api|query|list|product|data)[^"\']*)["\']', resp.text, re.IGNORECASE)
        print(f"API线索: {list(set(api_hints))[:10]}")

        # 查找data-url或ajax调用
        data_urls = re.findall(r'(?:url|href|src|data-url|action)\s*[:=]\s*["\']([^"\']+)["\']', resp.text)
        print(f"数据URL: {list(set(data_urls))[:15]}")

    except Exception as e:
        print(f"Error: {e}")

    # 测试可能的API
    api_tests = [
        ("http://www.citic-wealth.com/yymk/lccs/product/list", "POST", {}),
        ("http://www.citic-wealth.com/yymk/lccs/productList", "GET", {}),
        ("http://www.citic-wealth.com/yymk/lccs/data/products.json", "GET", {}),
        ("http://www.citic-wealth.com/lccs/product/queryList", "POST", {"pageNum": 1, "pageSize": 10}),
    ]

    for url, method, data in api_tests:
        try:
            if method == "POST":
                resp = session.post(url, json=data, timeout=10)
            else:
                resp = session.get(url, params=data, timeout=10)
            resp.encoding = 'utf-8'
            print(f"\n[{method}] {url}: {resp.status_code}")
            if resp.status_code == 200 and len(resp.text) < 5000:
                print(f"  Content: {resp.text[:500]}")
        except Exception as e:
            print(f"\n[{method}] {url}: {e}")


def test_ceb_detailed():
    """光大银行理财 - 深入探测"""
    print("\n" + "="*60)
    print("光大银行理财 (CEB) - 深入探测")
    print("="*60)

    session = create_session()

    # 添加更多headers尝试绕过412
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }

    try:
        resp = session.get("https://www.cebwm.com/wealth/grlc/index.html", headers=headers, timeout=15)
        print(f"首页状态: {resp.status_code}")
        if resp.status_code == 200:
            # 查找API
            api_hints = re.findall(r'["\']([^"\']*(?:api|query|list|product)[^"\']*)["\']', resp.text, re.IGNORECASE)
            print(f"API线索: {list(set(api_hints))[:10]}")
    except Exception as e:
        print(f"Error: {e}")

    # 尝试不同的域名
    alt_domains = [
        "https://www.cebwm.com",
        "https://cebwm.com",
        "https://www.ebanklc.com",  # 可能的备用域名
    ]

    for domain in alt_domains:
        try:
            resp = session.get(f"{domain}/", headers=headers, timeout=10)
            print(f"\n{domain}: {resp.status_code}")
        except Exception as e:
            print(f"\n{domain}: {e}")


def test_huaxia_detailed():
    """华夏银行理财 - 解析JS数据"""
    print("\n" + "="*60)
    print("华夏银行理财 (Huaxia) - 解析JS数据")
    print("="*60)

    session = create_session()

    # 获取JS数据文件
    try:
        resp = session.get("https://www.hxwm.com.cn/common/js/gmjzbgData.js", timeout=15)
        resp.encoding = 'utf-8'
        print(f"JS文件状态: {resp.status_code}")

        # 解析JS中的JSON数据
        js_content = resp.text
        # 提取var jzbgData = [...] 中的数组
        match = re.search(r'var\s+jzbgData\s*=\s*\[(.*?)\];', js_content, re.DOTALL)
        if match:
            json_str = '[' + match.group(1) + ']'
            # 清理JSON
            json_str = re.sub(r',\s*\]', ']', json_str)  # 移除末尾逗号
            json_str = re.sub(r',\s*}', '}', json_str)

            try:
                data = json.loads(json_str)
                print(f"解析到 {len(data)} 条产品数据")
                if data:
                    print(f"示例数据: {json.dumps(data[0], ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"前500字符: {json_str[:500]}")

    except Exception as e:
        print(f"Error: {e}")

    # 查找其他数据源
    try:
        resp = session.get("https://www.hxwm.com.cn/xxpl/gmcpgg/jzbg/index.shtml", timeout=15)
        resp.encoding = 'utf-8'

        # 查找其他JS文件
        js_files = re.findall(r'src=["\']([^"\']*\.js[^"\']*)["\']', resp.text)
        print(f"\nJS文件: {js_files}")

        # 查找产品列表API
        data_sources = re.findall(r'["\']([^"\']*(?:Data|List|product)[^"\']*\.js)["\']', resp.text, re.IGNORECASE)
        print(f"数据文件: {data_sources}")

    except Exception as e:
        print(f"Error: {e}")


def test_ningbo_detailed():
    """宁波银行理财 - 深入探测"""
    print("\n" + "="*60)
    print("宁波银行理财 (Ningbo) - 深入探测")
    print("="*60)

    session = create_session()

    try:
        resp = session.get("https://www.wmbnb.com/product/index.html", timeout=15)
        resp.encoding = 'utf-8'
        print(f"首页状态: {resp.status_code}")

        # 查找JS文件
        js_files = re.findall(r'src=["\']([^"\']*\.js[^"\']*)["\']', resp.text)
        print(f"JS文件: {js_files[:15]}")

        # 查找API线索
        api_hints = re.findall(r'["\']([^"\']*(?:api|query|list|product|data)[^"\']*)["\']', resp.text, re.IGNORECASE)
        print(f"API线索: {list(set(api_hints))[:15]}")

        # 查找接口调用
        ajax_calls = re.findall(r'(?:axios|fetch|ajax|post|get)\s*\(\s*["\']([^"\']+)["\']', resp.text, re.IGNORECASE)
        print(f"AJAX调用: {ajax_calls}")

    except Exception as e:
        print(f"Error: {e}")

    # 测试常见接口路径
    api_tests = [
        "https://www.wmbnb.com/product/queryProductList",
        "https://www.wmbnb.com/product/list",
        "https://www.wmbnb.com/api/product/queryList",
        "https://www.wmbnb.com/wmbnb/product/list",
        "https://www.wmbnb.com/nbwm/product/list",
    ]

    for url in api_tests:
        try:
            # 尝试POST
            resp = session.post(url, json={"pageNum": 1, "pageSize": 10}, timeout=10)
            print(f"\n[POST] {url}: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  Response: {json.dumps(data, ensure_ascii=False)[:300]}")
                except:
                    print(f"  Content: {resp.text[:200]}")
        except Exception as e:
            print(f"\n[POST] {url}: {e}")


def test_spdb_alternatives():
    """浦发银行 - 尝试其他域名"""
    print("\n" + "="*60)
    print("浦发银行理财 (SPDB) - 尝试其他入口")
    print("="*60)

    session = create_session()

    # 浦发可能的其他域名
    domains = [
        "https://ebank.spdb.com.cn",
        "https://per.spdb.com.cn",
        "https://www.spdb.com.cn",
        "https://wealth.spdb.com.cn",
    ]

    for domain in domains:
        try:
            resp = session.get(domain, timeout=10)
            print(f"\n{domain}: {resp.status_code}")
            if resp.status_code == 200:
                # 查找理财相关链接
                links = re.findall(r'href=["\']([^"\']*(?:wealth|product|licai|finance)[^"\']*)["\']', resp.text, re.IGNORECASE)
                if links:
                    print(f"  理财相关链接: {links[:5]}")
        except Exception as e:
            print(f"\n{domain}: {e}")


if __name__ == "__main__":
    test_citic_detailed()
    test_huaxia_detailed()
    test_ningbo_detailed()
    test_ceb_detailed()
    test_spdb_alternatives()

    print("\n" + "="*60)
    print("探测完成")
    print("="*60)
