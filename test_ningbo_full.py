"""
宁波银行API完整测试
"""

import requests
import ssl
import urllib3
import json
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

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json;charset=UTF-8',
        'Origin': base,
        'Referer': f'{base}/product/index.html',
    }

    # 先访问首页建立session
    session.get(f"{base}/product/index.html", timeout=30)

    print("="*60)
    print("1. 获取产品列表 (分页)")
    print("="*60)

    # 获取第一页产品
    url = f"{base}/ningbo-web/product/list.json"
    params = {
        "curPage": 1,
        "pageCount": 20,  # 每页数量
        "productTapValue": "",
        "conductTapValue": "",
        "profucatTapValue": "",
    }

    resp = session.post(url, json=params, headers=headers, timeout=30)
    data = resp.json()

    print(f"状态: {data.get('status')}")
    print(f"总数: {data.get('total')}")

    if data.get('list'):
        print(f"本页产品数: {len(data['list'])}")
        print("\n第一个产品详情:")
        product = data['list'][0]
        for key, value in product.items():
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("2. 获取产品字典 (分类信息)")
    print("="*60)

    dict_names = ['productstate', 'conduct', 'profucat']
    for dict_name in dict_names:
        url = f"{base}/ningbo-web/product/dictionary.json"
        params = {"dictname": dict_name}
        resp = session.post(url, json=params, headers=headers, timeout=30)
        result = resp.json()
        print(f"\n{dict_name}:")
        if result.get('list'):
            for item in result['list'][:5]:
                print(f"  {item}")

    print("\n" + "="*60)
    print("3. 获取单个产品的净值报告")
    print("="*60)

    if data.get('list'):
        # 获取第一个产品的ID
        product_id = data['list'][0].get('id')
        project_code = data['list'][0].get('projectcode')
        print(f"产品ID: {product_id}")
        print(f"产品代码: {project_code}")

        # 获取该产品的附件（净值报告）
        url = f"{base}/ningbo-web/product/attachmentlist.json"
        params = {
            "project_id": product_id,
            "curPage": 1,
            "pageCount": 10,
        }

        resp = session.post(url, json=params, headers=headers, timeout=30)
        result = resp.json()
        print(f"\n该产品净值报告数: {result.get('total')}")
        if result.get('list'):
            print("最新报告:")
            for item in result['list'][:3]:
                print(f"  标题: {item.get('title')}")
                print(f"  URL: {item.get('url')}")
                print(f"  日期: {item.get('creation_date')}")
                print()

    print("\n" + "="*60)
    print("4. 搜索特定产品")
    print("="*60)

    url = f"{base}/ningbo-web/product/search.json"
    params = {
        "searchText": "持有期",
        "curPage": 1,
        "pageCount": 10,
    }

    resp = session.post(url, json=params, headers=headers, timeout=30)
    result = resp.json()
    print(f"搜索'持有期'结果数: {result.get('total')}")
    if result.get('list'):
        for item in result['list'][:3]:
            print(f"  - {item.get('projectname')}")

    print("\n" + "="*60)
    print("5. 获取全部产品数量统计")
    print("="*60)

    # 获取所有产品（分页获取）
    total_pages = (data.get('total', 0) + 99) // 100
    print(f"总产品数: {data.get('total')}")
    print(f"需要页数: {total_pages} (每页100个)")


if __name__ == "__main__":
    test()
