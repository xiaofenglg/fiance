"""
宁波银行 - 通过首页获取session后再测试API
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


def test():
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount('https://', adapter)
    session.verify = False

    base = "https://www.wmbnb.com"

    print("="*60)
    print("步骤1: 获取首页并建立session")
    print("="*60)

    # 先访问首页获取cookies
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }

    resp = session.get(f"{base}/product/index.html", headers=headers, timeout=30)
    print(f"首页状态: {resp.status_code}")
    print(f"Cookies: {[c.name for c in session.cookies]}")

    # 分析首页中的API调用
    content = resp.text

    # 查找baseUrl或apiUrl
    base_urls = re.findall(r'(?:baseUrl|apiUrl|baseURL|API_URL)\s*[:=]\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
    print(f"Base URLs found: {base_urls}")

    # 查找axios或ajax配置
    axios_config = re.findall(r'axios\.(?:get|post)\s*\(["\']([^"\']+)["\']', content)
    print(f"Axios calls: {axios_config}")

    # 查找所有可能的API路径
    api_paths = re.findall(r'["\']/(ningbo-web|api|product|fund)[^"\']*["\']', content)
    print(f"API paths in HTML: {list(set(api_paths))[:20]}")

    print("\n" + "="*60)
    print("步骤2: 获取fund.js并分析")
    print("="*60)

    resp = session.get(f"{base}/js/fund.js", timeout=30)
    fund_js = resp.text

    # 提取所有$.ajax调用的URL
    ajax_urls = re.findall(r'\$\.ajax\s*\(\s*\{[^}]*url\s*:\s*["\']?([^"\'}\s,]+)', fund_js, re.DOTALL)
    print(f"AJAX URLs: {ajax_urls}")

    # 查找_this.url + 后面的路径
    url_paths = re.findall(r'_this\.url\s*\+\s*["\']([^"\']+)["\']', fund_js)
    print(f"URL paths: {url_paths}")

    # 查找完整的API调用模式
    api_calls = re.findall(r'url\s*:\s*_this\.url\s*\+\s*["\']([^"\']+)["\']', fund_js)
    print(f"Full API calls: {api_calls}")

    print("\n" + "="*60)
    print("步骤3: 测试发现的API路径")
    print("="*60)

    # 测试发现的路径
    found_paths = list(set(url_paths + api_calls))

    for path in found_paths:
        url = f"{base}/ningbo-web{path}" if not path.startswith('/') else f"{base}/ningbo-web/{path}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json;charset=UTF-8',
            'Origin': base,
            'Referer': f'{base}/product/index.html',
            'X-Requested-With': 'XMLHttpRequest',
        }

        # GET请求
        try:
            resp = session.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                print(f"\n[GET] {path}: SUCCESS!")
                print(f"  Content: {resp.text[:300]}")
        except:
            pass

        # POST请求
        try:
            data = {"curPage": 1, "pageCount": 10, "pageNum": 1, "pageSize": 10}
            resp = session.post(url, json=data, headers=headers, timeout=10)
            if resp.status_code == 200:
                print(f"\n[POST] {path}: SUCCESS!")
                print(f"  Content: {resp.text[:300]}")
        except:
            pass


if __name__ == "__main__":
    test()
