"""调试中信理财API响应"""
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
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; MI 8) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.62 XWEB/2691 MMWEBSDK/200901 Mobile Safari/537.36 MMWEBID/1234 MicroMessenger/7.0.20.1781(0x27001435) Process/tools WeChat/arm64 Weixin NetType/WIFI Language/zh_CN ABI/arm64',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://wechat.citic-wealth.com/wechat/product/',
    })
    session.verify = False
    return session

def main():
    session = create_session()

    # 先访问主页
    print("1. 访问主页...")
    resp = session.get("https://wechat.citic-wealth.com/wechat/product/", timeout=15)
    print(f"   Cookies: {dict(session.cookies)}")

    # 测试API并打印完整响应
    print("\n2. 测试API完整响应...")

    urls = [
        "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList?pageNo=1&pageSize=10",
        "https://wechat.citic-wealth.com/cms.product/api/productRecommended/list?pageNo=1&pageSize=10",
        "https://wechat.citic-wealth.com/cms.product/api/productInfo/fundList?pageNo=1&pageSize=10",
    ]

    for url in urls:
        print(f"\n   URL: {url}")
        try:
            resp = session.get(url, timeout=15)
            print(f"   状态: {resp.status_code}")
            print(f"   Headers: {dict(resp.headers)}")
            print(f"   完整响应:")
            try:
                data = resp.json()
                print(json.dumps(data, ensure_ascii=False, indent=2))
            except:
                print(resp.text[:1000])
        except Exception as e:
            print(f"   错误: {e}")

    # 检查是否需要特定的header
    print("\n3. 尝试添加更多headers...")

    extra_headers = {
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://wechat.citic-wealth.com',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
    }

    session.headers.update(extra_headers)

    resp = session.get(
        "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo/fundList",
        params={"pageNo": 1, "pageSize": 10},
        timeout=15
    )
    print(f"\n   带额外headers后的响应:")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
