"""测试中信理财API过滤和分页"""
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
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; MI 8) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.62 XWEB/2691 MMWEBSDK/200901 Mobile Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://wechat.citic-wealth.com/wechat/product/',
    })
    session.verify = False
    return session

def main():
    session = create_session()

    # 访问主页
    session.get("https://wechat.citic-wealth.com/wechat/product/", timeout=15)

    base_url = "https://wechat.citic-wealth.com/cms.product/api/productInfo/fundList"

    print("=" * 60)
    print("测试中信理财API产品列表")
    print("=" * 60)

    # 1. 获取全部产品数量
    print("\n1. 获取全部产品...")
    resp = session.get(base_url, params={"pageNo": 1, "pageSize": 1000}, timeout=30)
    data = resp.json()
    all_products = data.get('data', [])
    print(f"   全部产品数量: {len(all_products)}")

    # 2. 统计各类别产品
    print("\n2. 按tags统计产品类别...")
    tags_count = {}
    status_count = {}

    for p in all_products:
        tag = p.get('tags', '未知')
        status = p.get('statusName', '未知')
        tags_count[tag] = tags_count.get(tag, 0) + 1
        status_count[status] = status_count.get(status, 0) + 1

    print("   产品类型分布:")
    for tag, count in sorted(tags_count.items(), key=lambda x: -x[1]):
        print(f"   - {tag}: {count}")

    print("\n   产品状态分布:")
    for status, count in sorted(status_count.items(), key=lambda x: -x[1]):
        print(f"   - {status}: {count}")

    # 3. 过滤每日购和定开产品
    print("\n3. 筛选目标产品(每日购+定开)...")

    target_products = []
    for p in all_products:
        tag = p.get('tags', '')
        status = p.get('statusName', '')
        # 筛选: 每日购或定开类型，且状态为"销售中"或"募集中"
        if tag in ['每日购', '每月定开', '双月定开', '季度定开', '半年定开', '一年定开', '两年定开', '三年定开']:
            target_products.append(p)

    print(f"   符合条件产品数量: {len(target_products)}")

    # 4. 展示部分产品数据
    print("\n4. 样例产品数据:")
    for i, p in enumerate(target_products[:5]):
        print(f"\n   产品{i+1}:")
        print(f"   - 代码: {p.get('fundCode')}")
        print(f"   - 名称: {p.get('fundName')}")
        print(f"   - 类型: {p.get('tags')}")
        print(f"   - 状态: {p.get('statusName')}")
        print(f"   - 单位净值: {p.get('perNetvalue')}")
        print(f"   - 累计净值: {p.get('totalNetvalue')}")
        print(f"   - 净值日期: {p.get('navDate')}")
        print(f"   - 风险等级: {p.get('riskLevelStr')}")
        print(f"   - 日涨幅: {p.get('dayInc')}")
        print(f"   - 月涨幅: {p.get('monthInc')}")
        print(f"   - 季涨幅: {p.get('seasonHinc')}")
        print(f"   - 半年涨幅: {p.get('halfYearHinc')}")
        print(f"   - 年涨幅: {p.get('yearInc')}")

    # 5. 检查是否有历史净值API
    print("\n5. 尝试获取历史净值API...")

    # 使用第一个产品测试
    if target_products:
        test_product = target_products[0]
        product_id = test_product.get('id')
        fund_code = test_product.get('fundCode')

        nav_apis = [
            f"/productInfo/nav?fundCode={fund_code}",
            f"/productInfo/navHistory?fundCode={fund_code}",
            f"/productInfo/navList?fundCode={fund_code}",
            f"/productInfo/detail?fundCode={fund_code}",
            f"/productInfo/detail?id={product_id}",
            f"/navHistory/list?fundCode={fund_code}",
        ]

        api_base = "https://wechat.citic-wealth.com/cms.product/api"
        for api in nav_apis:
            try:
                resp = session.get(f"{api_base}{api}", timeout=10)
                ct = resp.headers.get('Content-Type', '')
                if 'json' in ct.lower():
                    result = resp.json()
                    if result.get('code') == '0000':
                        print(f"\n   找到有效API: {api}")
                        print(f"   响应: {json.dumps(result, ensure_ascii=False)[:500]}")
                    else:
                        print(f"   {api}: {result.get('code')} - {result.get('msg')}")
            except Exception as e:
                print(f"   {api}: 错误 {e}")

if __name__ == "__main__":
    main()
