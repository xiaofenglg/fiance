"""分析中信理财返回的HTML内容"""
import requests
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
})
session.verify = False

base_url = "https://www.citic-wealth.com"

# 测试API返回的内容
print("测试API返回内容:")
resp = session.get(f"{base_url}/cms.product/api/productInfo/fundList",
                   params={"pageNo": 1, "pageSize": 10}, timeout=30)
print(f"状态码: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")
print(f"内容长度: {len(resp.text)}")
print(f"\n内容预览:")
print(resp.text[:2000])

# 保存完整内容
with open("citic_api_response.html", "wb") as f:
    f.write(resp.content)
print("\n已保存到 citic_api_response.html")
