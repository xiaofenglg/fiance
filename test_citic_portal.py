"""测试中信理财官网净值查询API"""
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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    })
    session.verify = False
    return session

def main():
    session = create_session()

    print("=" * 60)
    print("测试中信理财官网净值查询API")
    print("=" * 60)

    # 官网域名
    base_url = "https://www.citic-wealth.com"

    # 先访问净值查询页面
    print("\n1. 访问净值查询页面...")
    resp = session.get(f"{base_url}/yymk/lccs/", timeout=30)
    print(f"   状态: {resp.status_code}")
    print(f"   Cookies: {dict(session.cookies)}")

    # 查找页面中的API信息
    import re
    api_patterns = re.findall(r'["\']([^"\']*api[^"\']*)["\']', resp.text, re.I)
    print(f"\n   页面中的API引用: {len(api_patterns)}")
    for api in list(set(api_patterns))[:10]:
        if '/' in api and len(api) < 100:
            print(f"     {api}")

    # 2. 测试官网可能的API路径
    print("\n2. 测试官网API路径...")

    # 设置官网Referer
    session.headers.update({
        'Referer': f'{base_url}/yymk/lccs/',
        'Origin': base_url,
    })

    api_tests = [
        # 净值查询页面API
        "/yymk/lccs/api/productList",
        "/yymk/lccs/api/fundList",
        "/yymk/lccs/api/search",
        "/yymk/lccs/data/productList.json",
        "/yymk/api/productList",
        "/yymk/api/fundList",
        # CMS API (可能有不同的前缀)
        "/cms/api/productInfo/fundList",
        "/cms/api/product/list",
        "/cms.api/productInfo/fundList",
        # Portal API
        "/portal/api/productInfo/fundList",
        "/portal/api/product/list",
        # 其他可能
        "/api/lccs/productList",
        "/api/yymk/productList",
        "/data/product/list.json",
    ]

    for api in api_tests:
        try:
            url = f"{base_url}{api}"
            resp = session.get(url, params={"pageNo": 1, "pageSize": 20}, timeout=10)
            ct = resp.headers.get('Content-Type', '')

            if resp.status_code == 200:
                if 'json' in ct.lower():
                    try:
                        data = resp.json()
                        print(f"   [JSON] {api}: {json.dumps(data, ensure_ascii=False)[:200]}")
                    except:
                        print(f"   [?] {api}: 200 但非有效JSON")
                elif len(resp.text) < 500:
                    print(f"   [200] {api}: {resp.text[:100]}")
        except Exception as e:
            pass

    # 3. 尝试访问iframe内嵌页面的API
    print("\n3. 测试iframe相关API...")

    # iframe源是 /wechat/product/#/productMarket
    # 它可能加载数据从特定API

    iframe_apis = [
        "/wechat/product/api/productList",
        "/wechat/product/api/productMarket",
        "/wechat/product/api/fundList",
        "/wechat/api/productMarket/list",
        "/wechat/productMarket/api/list",
    ]

    for api in iframe_apis:
        try:
            url = f"{base_url}{api}"
            resp = session.get(url, params={"pageNo": 1, "pageSize": 20}, timeout=10)
            print(f"   {api}: {resp.status_code}")
            if resp.status_code == 200 and 'json' in resp.headers.get('Content-Type', '').lower():
                print(f"      {resp.text[:200]}")
        except Exception as e:
            pass

    # 4. 直接从iframe页面提取产品数据
    print("\n4. 从iframe页面提取产品数据...")

    # iframe URL
    iframe_url = f"{base_url}/wechat/product/"
    resp = session.get(iframe_url, timeout=30)
    print(f"   iframe页面状态: {resp.status_code}")

    # 查找JS文件
    js_files = re.findall(r'src=["\']([^"\']*static/js/[^"\']+\.js)["\']', resp.text)
    print(f"   找到 {len(js_files)} 个JS文件")

    # 下载并分析主JS文件
    if js_files:
        for js_file in js_files:
            if 'app' in js_file:
                js_url = f"{base_url}/wechat/product/{js_file.lstrip('./')}"
                print(f"\n   分析: {js_url}")
                try:
                    resp = session.get(js_url, timeout=30)
                    if resp.status_code == 200:
                        # 搜索API URL
                        api_urls = re.findall(r'["\']([^"\']*(?:api|list|query|fund|product)[^"\']{5,50})["\']', resp.text)
                        unique_apis = list(set([a for a in api_urls if '/' in a and not a.startswith('//')]))[:15]
                        print(f"   找到的API路径:")
                        for api in unique_apis:
                            print(f"     {api}")
                except Exception as e:
                    print(f"   获取JS失败: {e}")

if __name__ == "__main__":
    main()
