"""尝试模拟微信环境访问中信理财API"""
import requests
import json
import urllib3
urllib3.disable_warnings()

session = requests.Session()

# 模拟微信内置浏览器
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Linux; Android 10; MI 8) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.62 XWEB/2691 MMWEBSDK/200901 Mobile Safari/537.36 MMWEBID/1234 MicroMessenger/7.0.20.1781(0x27001435) Process/tools WeChat/arm64 Weixin NetType/WIFI Language/zh_CN ABI/arm64',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'X-Requested-With': 'com.tencent.mm',
    'Origin': 'https://www.citic-wealth.com',
    'Referer': 'https://www.citic-wealth.com/wechat/product/',
})
session.verify = False

base_url = "https://www.citic-wealth.com"

# 先访问主页获取cookie
print("1. 访问主页获取cookie...")
resp = session.get(f"{base_url}/wechat/product/", timeout=30)
print(f"   状态: {resp.status_code}")
print(f"   Cookies: {dict(session.cookies)}")

# 测试API
print("\n2. 测试产品列表API...")
apis_to_test = [
    "/cms.product/api/productInfo/fundList",
    "/cms.product/api/productInfo/search",
    "/cms.product/api/custom/productInfo/search",
    "/cms.product/api/productRecommended/list",
]

for api in apis_to_test:
    try:
        resp = session.get(f"{base_url}{api}", params={"pageNo": 1, "pageSize": 10}, timeout=15)
        ct = resp.headers.get('Content-Type', '')
        print(f"\n   GET {api}")
        print(f"   状态: {resp.status_code}, 类型: {ct[:30]}")

        if 'json' in ct.lower():
            try:
                data = resp.json()
                print(f"   数据: {json.dumps(data, ensure_ascii=False)[:500]}")
            except:
                pass
        elif resp.status_code == 200 and len(resp.text) < 1000:
            print(f"   内容: {resp.text[:300]}")
    except Exception as e:
        print(f"   错误: {e}")

# 尝试POST请求
print("\n3. 测试POST请求...")
post_data = {
    "pageNo": 1,
    "pageSize": 10,
    "productType": "",
    "riskLevel": "",
    "orderBy": ""
}

for api in ["/cms.product/api/productInfo/fundList", "/cms.product/api/productInfo/search"]:
    try:
        # JSON格式
        resp = session.post(f"{base_url}{api}", json=post_data, timeout=15)
        ct = resp.headers.get('Content-Type', '')
        print(f"\n   POST {api} (JSON)")
        print(f"   状态: {resp.status_code}, 类型: {ct[:30]}")

        if 'json' in ct.lower():
            data = resp.json()
            print(f"   数据: {json.dumps(data, ensure_ascii=False)[:500]}")

        # 表单格式
        resp2 = session.post(f"{base_url}{api}", data=post_data, timeout=15)
        ct2 = resp2.headers.get('Content-Type', '')
        print(f"\n   POST {api} (FORM)")
        print(f"   状态: {resp2.status_code}, 类型: {ct2[:30]}")

        if 'json' in ct2.lower():
            data2 = resp2.json()
            print(f"   数据: {json.dumps(data2, ensure_ascii=False)[:500]}")

    except Exception as e:
        print(f"   错误: {e}")

print("\n完成!")
