"""
测试银行API是否有查询次数限制
策略：
1. 从不同位置抽样查询产品NAV
2. 检查是否存在请求次数限制
3. 测试增加延迟是否能获取更多数据
"""

import requests
import ssl
import urllib3
import time
import json
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context


class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


def create_session():
    """创建新会话"""
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://www.cmbcwm.com.cn',
        'Referer': 'https://www.cmbcwm.com.cn/grlc/index.htm'
    })
    session.verify = False
    urllib3.disable_warnings()
    return session


def get_product_list(session, page=1, page_size=100):
    """获取产品列表"""
    url = "https://www.cmbcwm.com.cn/gw/po_web/BTAProductQry"
    response = session.post(url, data={
        "pageNo": page,
        "pageSize": page_size,
        "code_or_name": ""
    }, timeout=30)
    return response.json()


def get_nav_data(session, product_code):
    """获取NAV数据"""
    url = "https://www.cmbcwm.com.cn/gw/po_web/BTADailyQry"
    response = session.post(url, data={
        "chart_type": 1,
        "real_prd_code": product_code,
        "begin_date": "",
        "end_date": "",
        "pageNo": 1,
        "pageSize": 30
    }, timeout=15)
    data = response.json()
    return data.get('btaDailyAddFieldList') or data.get('list') or []


def test_1_continuous_requests():
    """测试1: 连续请求，看看从第几个开始失败"""
    print("=" * 60)
    print("测试1: 连续请求检测限制点")
    print("=" * 60)

    session = create_session()

    # 获取前500个产品
    products = []
    for page in range(1, 6):
        data = get_product_list(session, page, 100)
        products.extend(data.get('list', []))
        time.sleep(0.2)

    print(f"获取到 {len(products)} 个产品")

    # 连续请求NAV
    success_count = 0
    fail_start = None

    for i, p in enumerate(products):
        code = p.get('REAL_PRD_CODE', '')
        if not code:
            continue

        nav = get_nav_data(session, code)

        if nav:
            success_count += 1
        else:
            if fail_start is None:
                fail_start = i

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/500, 成功: {success_count}, 首次失败位置: {fail_start}")

        time.sleep(0.1)  # 100ms延迟

    print(f"\n结果: 成功 {success_count}/{len(products)}")
    if fail_start:
        print(f"首次失败位置: 第 {fail_start} 个产品")

    return success_count


def test_2_reverse_order():
    """测试2: 反向顺序请求，看是否是产品本身没数据"""
    print("\n" + "=" * 60)
    print("测试2: 反向顺序请求（从后往前）")
    print("=" * 60)

    session = create_session()

    # 获取后面的产品（第30-35页）
    products = []
    for page in range(30, 35):
        try:
            data = get_product_list(session, page, 100)
            products.extend(data.get('list', []))
            time.sleep(0.2)
        except:
            break

    print(f"获取到 {len(products)} 个产品（第30-35页）")

    success_count = 0
    for i, p in enumerate(products[:100]):
        code = p.get('REAL_PRD_CODE', '')
        if not code:
            continue

        nav = get_nav_data(session, code)
        if nav:
            success_count += 1

        time.sleep(0.1)

    print(f"结果: 后面的产品成功率 {success_count}/100")
    return success_count


def test_3_new_session_per_batch():
    """测试3: 每批次使用新会话"""
    print("\n" + "=" * 60)
    print("测试3: 每100个请求创建新会话")
    print("=" * 60)

    session = create_session()

    # 获取产品
    products = []
    for page in range(1, 4):
        data = get_product_list(session, page, 100)
        products.extend(data.get('list', []))
        time.sleep(0.2)

    print(f"获取到 {len(products)} 个产品")

    success_count = 0
    batch_size = 50

    for batch_start in range(0, min(300, len(products)), batch_size):
        # 每批次创建新会话
        session = create_session()
        batch_success = 0

        for i in range(batch_start, min(batch_start + batch_size, len(products))):
            p = products[i]
            code = p.get('REAL_PRD_CODE', '')
            if not code:
                continue

            nav = get_nav_data(session, code)
            if nav:
                batch_success += 1
                success_count += 1

            time.sleep(0.15)

        print(f"  批次 {batch_start//batch_size + 1}: 成功 {batch_success}/{batch_size}")

    print(f"\n结果: 总成功 {success_count}/300")
    return success_count


def test_4_slow_requests():
    """测试4: 慢速请求（500ms延迟）"""
    print("\n" + "=" * 60)
    print("测试4: 慢速请求（500ms延迟）")
    print("=" * 60)

    session = create_session()

    # 获取产品
    products = []
    for page in range(1, 4):
        data = get_product_list(session, page, 100)
        products.extend(data.get('list', []))
        time.sleep(0.3)

    print(f"获取到 {len(products)} 个产品")

    success_count = 0

    for i, p in enumerate(products[:150]):
        code = p.get('REAL_PRD_CODE', '')
        if not code:
            continue

        nav = get_nav_data(session, code)
        if nav:
            success_count += 1

        if (i + 1) % 30 == 0:
            print(f"  进度: {i+1}/150, 成功: {success_count}")

        time.sleep(0.5)  # 500ms延迟

    print(f"\n结果: 成功 {success_count}/150")
    return success_count


def test_5_check_specific_products():
    """测试5: 检查特定产品是否真的没有NAV数据"""
    print("\n" + "=" * 60)
    print("测试5: 检查失败产品是否真的无数据")
    print("=" * 60)

    session = create_session()

    # 获取产品
    data = get_product_list(session, 1, 100)
    products = data.get('list', [])

    # 先快速扫描找出"失败"的产品
    failed_products = []
    for p in products[:50]:
        code = p.get('REAL_PRD_CODE', '')
        nav = get_nav_data(session, code)
        if not nav:
            failed_products.append(p)
        time.sleep(0.05)

    print(f"快速扫描: 前50个产品中有 {len(failed_products)} 个无NAV数据")

    if failed_products:
        # 等待10秒后重新查询这些"失败"的产品
        print("等待10秒后重新查询...")
        time.sleep(10)

        session = create_session()  # 新会话
        retry_success = 0

        for p in failed_products[:10]:
            code = p.get('REAL_PRD_CODE', '')
            name = p.get('PRD_NAME', '')[:30]
            nav = get_nav_data(session, code)

            if nav:
                print(f"  重试成功: {name}")
                retry_success += 1
            else:
                print(f"  仍然失败: {name}")

            time.sleep(0.5)

        print(f"\n重试结果: {retry_success}/{len(failed_products[:10])} 成功")


def main():
    print("=" * 60)
    print("    民生银行API限制测试")
    print("=" * 60)
    print()

    try:
        # 测试1: 连续请求
        result1 = test_1_continuous_requests()

        # 测试2: 反向顺序
        result2 = test_2_reverse_order()

        # 测试3: 新会话
        result3 = test_3_new_session_per_batch()

        # 测试4: 慢速请求
        # result4 = test_4_slow_requests()  # 太慢，可选

        # 测试5: 重试失败产品
        test_5_check_specific_products()

        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"测试1 (连续请求): {result1}/500")
        print(f"测试2 (反向顺序): {result2}/100")
        print(f"测试3 (新会话): {result3}/300")

        if result2 < 50:
            print("\n结论: 后面的产品本身就没有NAV数据（已清盘/募集中）")
        elif result1 < 200 and result3 > result1:
            print("\n结论: 存在会话级别的请求限制，需要定期更换会话")
        elif result1 >= 400:
            print("\n结论: 大部分产品都有NAV数据，之前可能是网络问题")
        else:
            print("\n结论: 需要进一步分析")

    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
