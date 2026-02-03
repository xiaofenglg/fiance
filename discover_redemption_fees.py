# -*- coding: utf-8 -*-
"""
Phase 0: API探测 — 发现各银行API中的赎回费字段

对三家银行各选3-5个产品，调用详情API并dump完整JSON响应，
搜索费用相关字段。

运行方式:
    python discover_redemption_fees.py
"""

import os
import sys
import json
import re
import time
import requests
import ssl
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# 费用相关关键词
FEE_KEYWORDS = [
    'fee', 'Fee', 'FEE',
    '赎回', '费用', '费率', '手续费', '申赎',
    'redeem', 'redemption', 'charge',
    'SH_FEE', 'BUY_FEE', 'SELL_FEE',
    'penalty', 'cost',
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class LegacySSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


def search_fee_fields(data, prefix='', depth=0):
    """递归搜索JSON中的费用相关字段"""
    found = []
    if depth > 10:
        return found
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            # 检查字段名是否包含费用关键词
            for kw in FEE_KEYWORDS:
                if kw.lower() in key.lower():
                    found.append((full_key, value))
                    break
            # 递归搜索子对象
            if isinstance(value, (dict, list)):
                found.extend(search_fee_fields(value, full_key, depth + 1))
            # 检查字符串值中是否包含费用关键词
            elif isinstance(value, str) and len(value) < 500:
                for kw in ['赎回费', '费率', '手续费']:
                    if kw in value:
                        found.append((f"{full_key}(value)", value))
                        break
    elif isinstance(data, list):
        for i, item in enumerate(data[:5]):  # 只检查前5个元素
            found.extend(search_fee_fields(item, f"{prefix}[{i}]", depth + 1))
    return found


def discover_minsheng():
    """探测民生银行 BTAProductDetail API"""
    print("\n" + "=" * 70)
    print("  民生银行 — BTAProductDetail API 探测")
    print("=" * 70)

    try:
        from minsheng import CMBCWealthCrawler
        crawler = CMBCWealthCrawler()

        # 获取产品列表取前5个
        products = crawler.get_product_list()
        if not products:
            print("  无法获取产品列表")
            return

        test_products = products[:5]
        results = []

        for i, product in enumerate(test_products):
            code = product.get('REAL_PRD_CODE') or product.get('PRD_CODE', '')
            name = product.get('PRD_NAME', '')[:40]
            print(f"\n  [{i+1}] 产品: {name} ({code})")

            detail = crawler.get_product_detail(code)
            if not detail:
                print(f"      详情API返回空")
                continue

            # 保存完整响应
            results.append({'code': code, 'name': name, 'detail': detail})

            # 搜索费用字段
            fee_fields = search_fee_fields(detail)
            if fee_fields:
                print(f"      发现 {len(fee_fields)} 个费用相关字段:")
                for field_name, field_value in fee_fields:
                    val_str = str(field_value)[:100]
                    print(f"        {field_name} = {val_str}")
            else:
                print(f"      未发现费用相关字段")

            # 打印所有字段名（帮助发现未被关键词匹配的字段）
            if isinstance(detail, dict):
                all_keys = list(detail.keys())
                data_obj = detail.get('data', detail)
                if isinstance(data_obj, dict):
                    all_keys = list(data_obj.keys())
                print(f"      所有字段: {all_keys}")

            time.sleep(1)

        # 保存完整dump
        dump_path = os.path.join(OUTPUT_DIR, 'discover_minsheng_dump.json')
        with open(dump_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  完整响应已保存: {dump_path}")

    except Exception as e:
        print(f"  民生银行探测失败: {e}")
        import traceback
        traceback.print_exc()


def discover_spdb():
    """探测浦银理财 API"""
    print("\n" + "=" * 70)
    print("  浦银理财 — chlid=1002/1005 API 探测")
    print("=" * 70)

    try:
        from spdb_wm import SPDBWealthCrawler
        crawler = SPDBWealthCrawler()

        # 获取产品列表(chlid=1002)
        result_1002 = crawler._api_search(chlid=1002, size=5)
        results = []

        if result_1002 and result_1002.get('content'):
            items = result_1002['content'][:5]
            for i, item in enumerate(items):
                code = item.get('PRDC_CD', '')
                name = item.get('PRDC_NM', '')[:40]
                print(f"\n  [{i+1}] 产品: {name} ({code})")

                results.append({'code': code, 'name': name, 'chlid_1002': item})

                # 搜索费用字段
                fee_fields = search_fee_fields(item)
                if fee_fields:
                    print(f"      [chlid=1002] 发现 {len(fee_fields)} 个费用相关字段:")
                    for field_name, field_value in fee_fields:
                        val_str = str(field_value)[:100]
                        print(f"        {field_name} = {val_str}")
                else:
                    print(f"      [chlid=1002] 未发现费用相关字段")

                # 打印所有字段名
                print(f"      所有字段: {list(item.keys())}")

            # 尝试 chlid=1005 (可能是费用/费率频道)
            print(f"\n  尝试 chlid=1005...")
            result_1005 = crawler._api_search(chlid=1005, size=5)
            if result_1005:
                print(f"      chlid=1005 响应: {type(result_1005)}")
                if isinstance(result_1005, dict):
                    print(f"      keys: {list(result_1005.keys())}")
                    content = result_1005.get('content', [])
                    if content:
                        print(f"      第一条记录字段: {list(content[0].keys()) if content else '空'}")
                        fee_fields = search_fee_fields(content[0] if content else {})
                        if fee_fields:
                            print(f"      发现费用相关字段:")
                            for fn, fv in fee_fields:
                                print(f"        {fn} = {str(fv)[:100]}")
                        for r in results:
                            r['chlid_1005_sample'] = content[0] if content else None
            else:
                print(f"      chlid=1005 返回空")

        # 保存完整dump
        dump_path = os.path.join(OUTPUT_DIR, 'discover_spdb_dump.json')
        with open(dump_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  完整响应已保存: {dump_path}")

    except Exception as e:
        print(f"  浦银理财探测失败: {e}")
        import traceback
        traceback.print_exc()


def discover_citic():
    """探测中信银行 getTAProductDetail API"""
    print("\n" + "=" * 70)
    print("  中信银行 — getTAProductDetail API 探测")
    print("=" * 70)

    session = requests.Session()
    session.mount('https://', LegacySSLAdapter())

    API_BASE = "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo"

    # 已知测试产品
    test_products = [
        ("AF245689D", 2, "七彩象睿享同行1号"),
        ("AF245749D", 2, "七彩象安裕日开1号"),
        ("AF245750D", 2, "七彩象安裕日开2号"),
    ]

    results = []
    for code, prod_type, name in test_products:
        print(f"\n  产品: {name} ({code})")

        # getTAProductDetail
        detail_url = f"{API_BASE}/getTAProductDetail?prodCode={code}&prodType={prod_type}"
        try:
            resp = session.get(detail_url, timeout=30, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                results.append({'code': code, 'name': name, 'detail': data})

                # 搜索费用字段
                fee_fields = search_fee_fields(data)
                if fee_fields:
                    print(f"      发现 {len(fee_fields)} 个费用相关字段:")
                    for field_name, field_value in fee_fields:
                        val_str = str(field_value)[:200]
                        print(f"        {field_name} = {val_str}")
                else:
                    print(f"      未发现费用相关字段")

                # 打印data层所有字段
                data_obj = data.get('data', data)
                if isinstance(data_obj, dict):
                    print(f"      所有字段: {list(data_obj.keys())}")
                    # 特别检查产品描述字段
                    for desc_key in ['productDesc', 'prodDesc', 'remark', 'productName',
                                     'feeDesc', 'feeRate', 'redeemFee', 'redeemDesc']:
                        if desc_key in data_obj:
                            val = str(data_obj[desc_key])[:200]
                            print(f"      {desc_key}: {val}")
            else:
                print(f"      HTTP {resp.status_code}")
        except Exception as e:
            print(f"      请求失败: {e}")

        time.sleep(1)

    # 保存完整dump
    dump_path = os.path.join(OUTPUT_DIR, 'discover_citic_dump.json')
    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  完整响应已保存: {dump_path}")


def discover_name_patterns():
    """从现有产品数据中搜索产品名称里的赎回费线索"""
    print("\n" + "=" * 70)
    print("  Fallback: 产品名称中的赎回费线索")
    print("=" * 70)

    try:
        from nav_db_excel import NavDBReader
        db_reader = NavDBReader()

        fee_products = []
        total = 0

        for sheet in db_reader.sheet_names:
            df = db_reader.read_sheet(sheet)
            df.columns = [str(c).strip() for c in df.columns]
            if 'level_0' in df.columns:
                df = df.rename(columns={'level_0': '产品代码', 'level_1': '产品名称'})

            for _, row in df.iterrows():
                name = str(row.get('产品名称', ''))
                code = str(row.get('产品代码', ''))
                total += 1

                # 搜索赎回费相关关键词
                if any(kw in name for kw in ['赎回费', '手续费', '短期赎回']):
                    fee_products.append({
                        'bank': sheet,
                        'code': code,
                        'name': name,
                    })

        print(f"\n  扫描 {total} 个产品名称")
        print(f"  发现 {len(fee_products)} 个含赎回费关键词的产品:")

        from redemption_fee_db import parse_fee_from_name
        for p in fee_products:
            fee_info = parse_fee_from_name(p['name'])
            parsed = '是' if fee_info else '否'
            desc = fee_info.get('fee_description', '') if fee_info else ''
            print(f"    [{p['bank']}] {p['name'][:50]}")
            print(f"      代码: {p['code']}, 解析成功: {parsed}, 描述: {desc}")

    except Exception as e:
        print(f"  名称扫描失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 70)
    print("  赎回费API字段探测工具")
    print("=" * 70)

    # 可选参数: 只运行特定银行
    banks = sys.argv[1:] if len(sys.argv) > 1 else ['minsheng', 'spdb', 'citic', 'names']

    if 'minsheng' in banks:
        discover_minsheng()

    if 'spdb' in banks:
        discover_spdb()

    if 'citic' in banks:
        discover_citic()

    if 'names' in banks:
        discover_name_patterns()

    print("\n" + "=" * 70)
    print("  探测完成！请检查上方输出和生成的dump文件")
    print("=" * 70)
