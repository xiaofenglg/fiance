"""完整探索中信理财API"""
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
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; MI 8) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://wechat.citic-wealth.com/wechat/product/',
    })
    session.verify = False
    return session

def main():
    session = create_session()

    # 访问主页获取cookie
    session.get("https://wechat.citic-wealth.com/wechat/product/", timeout=15)

    print("=" * 60)
    print("完整探索中信理财API")
    print("=" * 60)

    base_url = "https://wechat.citic-wealth.com"

    # 1. 获取所有产品并按状态分组
    print("\n1. 获取全部产品并分析...")
    resp = session.get(
        f"{base_url}/cms.product/api/productInfo/fundList",
        params={"pageNo": 1, "pageSize": 2000},
        timeout=30
    )
    data = resp.json()
    all_products = data.get('data', [])
    print(f"   总产品数: {len(all_products)}")

    # 统计状态
    by_status = {}
    by_tag = {}
    for p in all_products:
        status = p.get('statusName', '未知')
        tag = p.get('tags', '未分类') or '未分类'
        by_status[status] = by_status.get(status, 0) + 1
        by_tag[tag] = by_tag.get(tag, 0) + 1

    print("\n   按状态分布:")
    for s, c in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"     {s}: {c}")

    print("\n   按标签分布:")
    for t, c in sorted(by_tag.items(), key=lambda x: -x[1]):
        print(f"     {t}: {c}")

    # 2. 测试不同的状态过滤参数
    print("\n\n2. 测试状态过滤参数...")
    status_params = [
        {"status": "0"},  # 销售中
        {"status": "1"},  # 募集中
        {"fundState": "0"},
        {"fundState": "1"},
        {"statusName": "销售中"},
        {"productStatus": "销售中"},
    ]

    for params in status_params:
        full_params = {"pageNo": 1, "pageSize": 50, **params}
        try:
            resp = session.get(
                f"{base_url}/cms.product/api/productInfo/fundList",
                params=full_params,
                timeout=15
            )
            data = resp.json()
            products = data.get('data', [])
            print(f"   参数 {params}: 返回 {len(products)} 条")
            if products:
                # 显示第一条
                p = products[0]
                print(f"     示例: {p.get('fundCode')} - {p.get('statusName')} - {p.get('tags')}")
        except Exception as e:
            print(f"   参数 {params}: 错误 {e}")

    # 3. 测试产品类型/标签过滤参数
    print("\n\n3. 测试标签/类型过滤参数...")
    tag_params = [
        {"tags": "每日购"},
        {"tags": "定开"},
        {"purchaseType": "每日购"},
        {"purchaseType": "定开"},
        {"productType": "每日购"},
        {"fundType": "0"},  # 货币
        {"fundType": "1"},  # 固收
        {"category": "每日购"},
        {"mainCategory": "个人"},
        {"typeCategory": "每日购"},
    ]

    for params in tag_params:
        full_params = {"pageNo": 1, "pageSize": 50, **params}
        try:
            resp = session.get(
                f"{base_url}/cms.product/api/productInfo/fundList",
                params=full_params,
                timeout=15
            )
            data = resp.json()
            products = data.get('data', [])
            print(f"   参数 {params}: 返回 {len(products)} 条")
        except Exception as e:
            print(f"   参数 {params}: 错误 {e}")

    # 4. 尝试其他API端点
    print("\n\n4. 测试其他API端点...")
    other_apis = [
        "/cms.product/api/productInfo/search",
        "/cms.product/api/productMarket/list",
        "/cms.product/api/productMarket/search",
        "/cms.product/api/productMarket/queryByPage",
        "/cms.product/api/custom/productMarket/list",
        "/cms.product/api/custom/productMarket/fundList",
        "/cms.product/api/productQuery/list",
        "/cms.product/api/productQuery/search",
        "/cms.product/api/navQuery/list",
        "/cms.product/api/netValue/list",
    ]

    for api in other_apis:
        try:
            resp = session.get(
                f"{base_url}{api}",
                params={"pageNo": 1, "pageSize": 20},
                timeout=10
            )
            ct = resp.headers.get('Content-Type', '')
            if 'json' in ct.lower():
                data = resp.json()
                code = data.get('code', '')
                msg = data.get('msg', '')[:30]
                result = data.get('data')
                if code == '0000' and result:
                    if isinstance(result, list):
                        print(f"   ✓ {api}: {len(result)} 条")
                    elif isinstance(result, dict):
                        records = result.get('records') or result.get('list') or []
                        if records:
                            print(f"   ✓ {api}: {len(records)} 条 (total: {result.get('total', 'N/A')})")
                        else:
                            print(f"   ? {api}: 返回dict但无records")
                    else:
                        print(f"   ? {api}: code={code}, result类型={type(result)}")
                else:
                    print(f"   ✗ {api}: code={code}, msg={msg}")
            else:
                print(f"   ✗ {api}: 非JSON ({resp.status_code})")
        except Exception as e:
            print(f"   ✗ {api}: 错误 {str(e)[:30]}")

    # 5. 查看销售中产品的详细信息
    print("\n\n5. 销售中产品示例:")
    selling_products = [p for p in all_products if p.get('statusName') == '销售中']
    for i, p in enumerate(selling_products[:5]):
        print(f"\n   产品{i+1}:")
        print(f"   - 代码: {p.get('fundCode')}")
        print(f"   - 名称: {p.get('fundName')[:30]}")
        print(f"   - 标签: {p.get('tags')}")
        print(f"   - 类型: {p.get('fundTypeStr')}")
        print(f"   - 状态: {p.get('statusName')}")
        print(f"   - 净值: {p.get('perNetvalue')} (日期: {p.get('navDate')})")
        print(f"   - 风险: {p.get('riskLevelStr')}")

if __name__ == "__main__":
    main()
