"""使用GET请求测试中信理财API"""
import requests
from requests.adapters import HTTPAdapter
import ssl
import json
import urllib3
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings()

class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器"""

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
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; MI 8) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.62 XWEB/2691 MMWEBSDK/200901 Mobile Safari/537.36 MMWEBID/1234 MicroMessenger/7.0.20.1781(0x27001435) Process/tools WeChat/arm64 Weixin NetType/WIFI Language/zh_CN ABI/arm64',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://wechat.citic-wealth.com/wechat/product/',
    })
    session.verify = False
    return session

def main():
    print("=" * 60)
    print("使用GET请求测试中信理财API")
    print("=" * 60)

    session = create_session()

    # 先访问主页获取cookie
    print("\n1. 访问主页获取cookie...")
    resp = session.get("https://wechat.citic-wealth.com/wechat/product/", timeout=15)
    print(f"   状态: {resp.status_code}")

    # 测试GET API
    print("\n2. 测试产品列表API (GET)...")

    base_url = "https://wechat.citic-wealth.com/cms.product/api"

    # 尝试不同的API路径和参数
    api_tests = [
        # fundList API
        {
            "path": "/custom/productInfo/fundList",
            "params": {"pageNo": 1, "pageSize": 20}
        },
        {
            "path": "/custom/productInfo/fundList",
            "params": {"pageNo": 1, "pageSize": 20, "purchaseType": "每日购"}
        },
        {
            "path": "/custom/productInfo/fundList",
            "params": {"pageNo": 1, "pageSize": 20, "purchaseType": "定开"}
        },
        {
            "path": "/productInfo/fundList",
            "params": {"pageNo": 1, "pageSize": 20}
        },
        # search API
        {
            "path": "/productInfo/search",
            "params": {"pageNo": 1, "pageSize": 20}
        },
        {
            "path": "/custom/productInfo/search",
            "params": {"pageNo": 1, "pageSize": 20}
        },
        # 不同的参数名尝试
        {
            "path": "/custom/productInfo/fundList",
            "params": {"current": 1, "size": 20}
        },
        {
            "path": "/custom/productInfo/fundList",
            "params": {"page": 1, "size": 20}
        },
    ]

    for test in api_tests:
        url = f"{base_url}{test['path']}"
        try:
            resp = session.get(url, params=test['params'], timeout=15)
            ct = resp.headers.get('Content-Type', '')
            print(f"\n   GET {test['path']}")
            print(f"   参数: {test['params']}")
            print(f"   状态: {resp.status_code}, 类型: {ct[:40]}")

            if 'json' in ct.lower():
                try:
                    data = resp.json()
                    # 打印完整响应结构
                    if data.get('success'):
                        print(f"   成功! 响应: {json.dumps(data, ensure_ascii=False)[:600]}")

                        # 检查产品数量
                        result = data.get('result') or data.get('data')
                        if result:
                            if isinstance(result, dict):
                                records = result.get('records') or result.get('list') or []
                                total = result.get('total') or len(records)
                                print(f"   找到 {total} 条产品，当前页 {len(records)} 条")
                                if records:
                                    print(f"\n   第一条产品详情:")
                                    print(json.dumps(records[0], ensure_ascii=False, indent=2)[:800])
                            elif isinstance(result, list):
                                print(f"   找到 {len(result)} 条产品")
                                if result:
                                    print(f"\n   第一条产品详情:")
                                    print(json.dumps(result[0], ensure_ascii=False, indent=2)[:800])
                    else:
                        print(f"   失败: {data.get('message')}")
                except json.JSONDecodeError as e:
                    print(f"   JSON解析失败: {e}")
                    print(f"   响应内容: {resp.text[:200]}")
            else:
                print(f"   非JSON响应: {resp.text[:200]}")

        except Exception as e:
            print(f"   错误: {e}")

    # 尝试获取产品详情API
    print("\n3. 测试其他可能的API...")

    other_apis = [
        "/productInfo/list",
        "/productInfo/all",
        "/product/list",
        "/fund/list",
        "/productRecommended/list",
    ]

    for api_path in other_apis:
        url = f"{base_url}{api_path}"
        try:
            resp = session.get(url, params={"pageNo": 1, "pageSize": 10}, timeout=10)
            ct = resp.headers.get('Content-Type', '')
            print(f"\n   GET {api_path}: {resp.status_code}, {ct[:30]}")
            if 'json' in ct.lower():
                data = resp.json()
                if data.get('success'):
                    print(f"   成功! {json.dumps(data, ensure_ascii=False)[:300]}")
        except Exception as e:
            print(f"   错误: {e}")

if __name__ == "__main__":
    main()
