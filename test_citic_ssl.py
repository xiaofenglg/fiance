"""使用LegacySSLAdapter测试中信理财API"""
import requests
from requests.adapters import HTTPAdapter
import ssl
import json
import urllib3
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings()

class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器，解决legacy renegotiation问题"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        # 启用legacy renegotiation
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

def create_session():
    """创建带SSL适配器的session"""
    session = requests.Session()
    adapter = LegacySSLAdapter(pool_connections=10, pool_maxsize=10)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    # 模拟微信内置浏览器
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; MI 8) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.62 XWEB/2691 MMWEBSDK/200901 Mobile Safari/537.36 MMWEBID/1234 MicroMessenger/7.0.20.1781(0x27001435) Process/tools WeChat/arm64 Weixin NetType/WIFI Language/zh_CN ABI/arm64',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'X-Requested-With': 'com.tencent.mm',
    })
    session.verify = False
    return session

def main():
    print("=" * 60)
    print("使用LegacySSLAdapter测试中信理财API")
    print("=" * 60)

    session = create_session()

    # 测试不同域名
    domains = [
        "https://wechat.citic-wealth.com",
        "https://www.citic-wealth.com",
    ]

    # 先访问主页获取cookie
    print("\n1. 访问页面获取cookie...")
    for domain in domains:
        try:
            urls_to_try = [
                f"{domain}/wechat/product/",
                f"{domain}/yymk/lccs/",
            ]
            for url in urls_to_try:
                try:
                    resp = session.get(url, timeout=15)
                    print(f"   {url}")
                    print(f"   状态: {resp.status_code}")
                    if resp.status_code == 200:
                        break
                except Exception as e:
                    print(f"   {url} - 错误: {e}")
        except Exception as e:
            print(f"   {domain} - 错误: {e}")

    print(f"\n   当前Cookies: {dict(session.cookies)}")

    # 测试API端点
    print("\n2. 测试产品列表API...")

    api_endpoints = [
        # wechat域名的API
        ("https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList", "POST"),
        ("https://wechat.citic-wealth.com/cms.product/api/productInfo/fundList", "POST"),
        ("https://wechat.citic-wealth.com/cms.product/api/productInfo/search", "POST"),
        # www域名的API
        ("https://www.citic-wealth.com/cms.product/api/custom/productInfo/fundList", "POST"),
        ("https://www.citic-wealth.com/cms.product/api/productInfo/fundList", "POST"),
    ]

    # 产品类型参数
    product_params = {
        "每日购": {"purchaseType": "每日购", "productSubType": ""},
        "定开": {"purchaseType": "定开", "productSubType": ""},
    }

    for url, method in api_endpoints:
        print(f"\n   测试: {url}")

        for param_name, extra_params in product_params.items():
            try:
                post_data = {
                    "pageNo": 1,
                    "pageSize": 10,
                    **extra_params
                }

                if method == "POST":
                    # JSON格式
                    headers = {'Content-Type': 'application/json'}
                    resp = session.post(url, json=post_data, headers=headers, timeout=15)
                else:
                    resp = session.get(url, params=post_data, timeout=15)

                ct = resp.headers.get('Content-Type', '')
                print(f"\n   [{param_name}] 状态: {resp.status_code}, 类型: {ct[:40]}")

                if 'json' in ct.lower() and resp.status_code == 200:
                    try:
                        data = resp.json()
                        print(f"   响应: {json.dumps(data, ensure_ascii=False)[:500]}")

                        # 检查是否有产品数据
                        if isinstance(data, dict):
                            if 'data' in data:
                                print(f"   data字段类型: {type(data['data'])}")
                                if isinstance(data['data'], dict) and 'records' in data['data']:
                                    records = data['data']['records']
                                    print(f"   找到 {len(records)} 条产品记录")
                                    if records:
                                        print(f"   第一条产品: {json.dumps(records[0], ensure_ascii=False)[:300]}")
                                elif isinstance(data['data'], list):
                                    print(f"   找到 {len(data['data'])} 条产品记录")
                            elif 'records' in data:
                                print(f"   找到 {len(data['records'])} 条产品记录")
                    except json.JSONDecodeError:
                        print(f"   响应内容: {resp.text[:300]}")
                elif resp.status_code == 200:
                    print(f"   响应内容: {resp.text[:300]}")

            except Exception as e:
                print(f"   [{param_name}] 错误: {e}")

    # 尝试其他可能的API格式
    print("\n3. 尝试其他API格式...")

    other_apis = [
        # 尝试不同的参数名
        {
            "url": "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList",
            "data": {"page": 1, "size": 10, "type": "每日购"}
        },
        {
            "url": "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList",
            "data": {"current": 1, "pageSize": 10, "category": "每日购"}
        },
    ]

    for api in other_apis:
        try:
            resp = session.post(api["url"], json=api["data"], timeout=15)
            ct = resp.headers.get('Content-Type', '')
            print(f"\n   POST {api['url']}")
            print(f"   参数: {api['data']}")
            print(f"   状态: {resp.status_code}, 类型: {ct[:40]}")
            if 'json' in ct.lower():
                print(f"   响应: {json.dumps(resp.json(), ensure_ascii=False)[:400]}")
        except Exception as e:
            print(f"   错误: {e}")

if __name__ == "__main__":
    main()
