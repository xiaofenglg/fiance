"""
诊断脚本：分析为什么只有200个产品被成功分析
"""

import requests
import ssl
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from collections import Counter
import time

class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


def main():
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
    })
    session.verify = False
    urllib3.disable_warnings()

    api_base = "https://www.cmbcwm.com.cn/gw/po_web"

    # 1. 获取全部产品列表
    print("=" * 60)
    print("步骤1: 获取产品列表")
    print("=" * 60)

    all_products = []
    page = 1
    while True:
        try:
            response = session.post(
                f"{api_base}/BTAProductQry",
                data={"pageNo": page, "pageSize": 100, "code_or_name": ""},
                timeout=30
            )
            data = response.json()
            products = data.get('list', [])
            if not products:
                break
            all_products.extend(products)
            print(f"  第{page}页: {len(products)}个产品")
            page += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"  第{page}页错误: {e}")
            break

    print(f"\n总产品数: {len(all_products)}")

    # 2. 分析产品代码情况
    print("\n" + "=" * 60)
    print("步骤2: 分析产品代码")
    print("=" * 60)

    no_code = 0
    with_code = 0
    for p in all_products:
        code = p.get('REAL_PRD_CODE') or p.get('PRD_CODE', '')
        if code:
            with_code += 1
        else:
            no_code += 1

    print(f"  有产品代码: {with_code}")
    print(f"  无产品代码: {no_code}")

    # 3. 抽样检查NAV数据
    print("\n" + "=" * 60)
    print("步骤3: 抽样检查NAV数据")
    print("=" * 60)

    sample_indices = [0, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3373]
    sample_indices = [i for i in sample_indices if i < len(all_products)]

    nav_results = []
    for idx in sample_indices:
        p = all_products[idx]
        code = p.get('REAL_PRD_CODE') or p.get('PRD_CODE', '')
        name = p.get('PRD_NAME', '')[:30]

        if not code:
            print(f"  [{idx}] {name} - 无产品代码")
            nav_results.append(('no_code', idx))
            continue

        try:
            response = session.post(
                f"{api_base}/BTADailyQry",
                data={
                    "chart_type": 1,
                    "real_prd_code": code,
                    "begin_date": "",
                    "end_date": "",
                    "pageNo": 1,
                    "pageSize": 30
                },
                timeout=15
            )
            data = response.json()
            nav_list = data.get('btaDailyAddFieldList') or data.get('list') or []

            if nav_list:
                print(f"  [{idx}] {name} - NAV数据: {len(nav_list)}条")
                nav_results.append(('has_nav', idx, len(nav_list)))
            else:
                print(f"  [{idx}] {name} - 无NAV数据")
                nav_results.append(('no_nav', idx))
        except Exception as e:
            print(f"  [{idx}] {name} - 请求失败: {e}")
            nav_results.append(('error', idx, str(e)))

        time.sleep(0.3)

    # 4. 批量检查前300个产品的NAV状态
    print("\n" + "=" * 60)
    print("步骤4: 检查前300个产品NAV状态")
    print("=" * 60)

    has_nav = 0
    no_nav = 0
    errors = 0

    for i, p in enumerate(all_products[:300]):
        code = p.get('REAL_PRD_CODE') or p.get('PRD_CODE', '')
        if not code:
            no_nav += 1
            continue

        try:
            response = session.post(
                f"{api_base}/BTADailyQry",
                data={
                    "chart_type": 1,
                    "real_prd_code": code,
                    "begin_date": "",
                    "end_date": "",
                    "pageNo": 1,
                    "pageSize": 10
                },
                timeout=10
            )
            data = response.json()
            nav_list = data.get('btaDailyAddFieldList') or data.get('list') or []

            if nav_list:
                has_nav += 1
            else:
                no_nav += 1

        except Exception as e:
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/300 - 有NAV: {has_nav}, 无NAV: {no_nav}, 错误: {errors}")

        time.sleep(0.1)

    print(f"\n前300个产品统计:")
    print(f"  有NAV数据: {has_nav}")
    print(f"  无NAV数据: {no_nav}")
    print(f"  请求错误: {errors}")

    # 5. 分析STATUS字段分布
    print("\n" + "=" * 60)
    print("步骤5: STATUS字段分布")
    print("=" * 60)

    status_counter = Counter()
    for p in all_products:
        status = p.get('STATUS')
        status_counter[status] += 1

    for status, count in status_counter.most_common():
        print(f"  STATUS={status}: {count}个")

    # 6. 分析产品名称中的关键词
    print("\n" + "=" * 60)
    print("步骤6: 产品名称关键词分析")
    print("=" * 60)

    keywords = ['持有', '定开', '封闭', '现金', '货币', '周周', '天天', '已到期', '募集中']
    keyword_counts = {kw: 0 for kw in keywords}

    for p in all_products:
        name = p.get('PRD_NAME', '')
        for kw in keywords:
            if kw in name:
                keyword_counts[kw] += 1

    for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
        print(f"  '{kw}': {count}个")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
