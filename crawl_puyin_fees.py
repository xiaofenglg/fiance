# -*- coding: utf-8 -*-
"""
浦银理财费率采集器

API: https://www.spdb-wm.com/api/search
- chlid=1002: 产品列表
- chlid=1003: 净值历史
- chlid=1006: 最新净值

用法:
    python crawl_puyin_fees.py              # 采集全部
    python crawl_puyin_fees.py --max 100    # 限量测试
"""

import os
import sys
import ssl
import re
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_puyin_fees.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '浦银')

sys.path.insert(0, BASE_DIR)
from nav_db_excel import NAVDatabaseExcel
from redemption_fee_db import load_fee_db, save_fee_db, update_fee_info, get_fee_info

BASE_URL = "https://www.spdb-wm.com"
API_URL = f"{BASE_URL}/api/search"


# ============================================================
# HTTP客户端 (使用与spdb_wm.py相同的SSL处理)
# ============================================================

class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器，支持旧版SSL协商"""

    def __init__(self, *args, **kwargs):
        self.ssl_context = create_urllib3_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        self.ssl_context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs['ssl_context'] = self.ssl_context
        return super().init_poolmanager(connections, maxsize, block, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs['ssl_context'] = self.ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        return super().send(request, stream=stream, timeout=timeout,
                            verify=False, cert=cert, proxies=proxies)


def create_client():
    """创建requests客户端"""
    session = requests.Session()
    adapter = LegacySSLAdapter(pool_connections=30, pool_maxsize=30)
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
        'Origin': BASE_URL,
        'Referer': f'{BASE_URL}/financialProducts/',
    })
    return session


# ============================================================
# 浦银API
# ============================================================

def fetch_product_list(client: requests.Session, page: int = 1, size: int = 100) -> Dict:
    """获取产品列表

    chlid=1002: 产品列表
    """
    payload = {
        "chlid": 1002,
        "page": page,
        "size": size,
        "searchword": "",
        "cutsize": 150,
        "dynexpr": [],
        "dynidx": 1,
        "extopt": [],
        "orderby": ""
    }

    try:
        r = client.post(API_URL, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            logger.warning("触发限流，等待10秒...")
            time.sleep(10)
            return fetch_product_list(client, page, size)
        else:
            logger.error(f"获取产品列表失败: {r.status_code}")
            return {}
    except Exception as e:
        logger.error(f"获取产品列表异常: {e}")
        return {}


def fetch_product_detail(client: requests.Session, product_code: str) -> Dict:
    """获取产品详情

    通过searchword过滤特定产品
    """
    payload = {
        "chlid": 1002,
        "page": 1,
        "size": 10,
        "searchword": f"(REAL_PRD_CODE = '{product_code}')",
        "cutsize": 150,
        "dynexpr": [],
        "dynidx": 1,
        "extopt": [],
        "orderby": ""
    }

    try:
        r = client.post(API_URL, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            content = data.get('data', {}).get('content', [])
            return content[0] if content else {}
        return {}
    except Exception as e:
        logger.debug(f"获取产品详情失败: {e}")
        return {}


def get_all_products(client: requests.Session) -> List[Dict]:
    """获取全部产品列表"""
    logger.info("获取浦银理财产品列表...")

    all_products = []
    page = 1

    while True:
        data = fetch_product_list(client, page=page, size=100)

        content = data.get('data', {}).get('content', [])
        total = data.get('data', {}).get('totalElements', 0)

        if not content:
            break

        all_products.extend(content)

        logger.info(f"获取第{page}页, 本页{len(content)}个, 累计{len(all_products)}/{total}")

        if len(all_products) >= total:
            break

        page += 1
        time.sleep(0.3)

    logger.info(f"共获取 {len(all_products)} 个产品")
    return all_products


# ============================================================
# 净值数据库匹配
# ============================================================

def get_nav_products() -> Dict[str, str]:
    """从净值数据库获取浦银产品列表"""
    logger.info("读取净值数据库中的浦银产品...")

    try:
        db = NAVDatabaseExcel()
        puyin_df = db.data.get('浦银理财')

        if puyin_df is None or puyin_df.empty:
            logger.warning("净值数据库中无浦银数据")
            return {}

        products = {}
        for idx in puyin_df.index:
            code, name = idx
            products[code] = name

        logger.info(f"从净值数据库获取 {len(products)} 个浦银产品")
        return products

    except Exception as e:
        logger.error(f"读取净值数据库失败: {e}")
        return {}


def match_products(api_products: List[Dict], nav_products: Dict[str, str]) -> List[Dict]:
    """匹配API产品与NAV数据库"""
    logger.info(f"匹配 {len(api_products)} 个API产品与 {len(nav_products)} 个NAV产品...")

    matched = []
    nav_codes = set(nav_products.keys())

    for p in api_products:
        # 尝试多种代码字段
        code = p.get('PRDC_CD') or p.get('REAL_PRD_CODE') or p.get('PRD_CODE') or ''
        name = p.get('PRDC_NM') or p.get('PRD_NAME') or ''

        # 精确匹配
        if code in nav_codes:
            p['nav_code'] = code
            p['nav_name'] = nav_products.get(code, name)
            matched.append(p)
            continue

        # 名称匹配
        for nav_code, nav_name in nav_products.items():
            if name and nav_name and (name in nav_name or nav_name in name):
                p['nav_code'] = nav_code
                p['nav_name'] = nav_name
                matched.append(p)
                break

    logger.info(f"匹配成功 {len(matched)} 个产品 ({len(matched)*100//len(nav_products)}%)")
    return matched


# ============================================================
# 费率解析
# ============================================================

def extract_fees_from_product(product: Dict) -> Dict:
    """从产品信息中提取费率

    浦银API可能包含的费率字段:
    - MGR_RATE: 管理费率
    - TRUS_RATE: 托管费率
    - SLS_RATE: 销售费率
    - SUBS_RATE: 认购费率
    - PUR_RATE: 申购费率
    - RDM_RATE: 赎回费率
    """
    fees = {
        'management_fee': None,
        'custody_fee': None,
        'sales_service_fee': None,
        'subscription_fee': None,
        'purchase_fee': None,
        'redemption_fee': None,
        'has_redemption_fee': False,
        'fee_schedule': [],
    }

    # 尝试从API字段提取
    rate_fields = {
        'MGR_RATE': 'management_fee',
        'TRUS_RATE': 'custody_fee',
        'SLS_RATE': 'sales_service_fee',
        'SUBS_RATE': 'subscription_fee',
        'PUR_RATE': 'purchase_fee',
        'RDM_RATE': 'redemption_fee',
    }

    for api_field, fee_key in rate_fields.items():
        value = product.get(api_field)
        if value is not None:
            try:
                rate = float(value)
                # 判断是百分比还是小数
                if rate > 1:
                    rate = rate / 100
                fees[fee_key] = rate
            except (ValueError, TypeError):
                pass

    # 从产品名称解析赎回费
    name = product.get('PRDC_NM') or product.get('PRD_NAME') or ''
    desc = product.get('PRDC_DESC') or product.get('PRD_DESC') or ''
    text = name + ' ' + desc

    # 检查是否有赎回费
    if fees.get('redemption_fee') and fees['redemption_fee'] > 0:
        fees['has_redemption_fee'] = True
    elif '赎回费' in text:
        if '不收取赎回费' in text or '赎回费0' in text or '赎回费：0' in text:
            fees['has_redemption_fee'] = False
            fees['redemption_fee'] = 0
        else:
            fees['has_redemption_fee'] = True
            # 尝试提取赎回费率
            match = re.search(r'赎回费[率]?[：:为\s]*([\d.]+)\s*%', text)
            if match:
                fees['redemption_fee'] = float(match.group(1)) / 100

    return fees


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='浦银理财费率采集器')
    parser.add_argument('--max', type=int, default=0, help='限制处理数量')
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)

    print("=" * 60)
    print("  浦银理财费率采集器")
    print("=" * 60)

    # 1. 获取NAV产品
    nav_products = get_nav_products()
    if not nav_products:
        print("净值数据库中无浦银产品")
        return 1

    # 2. 获取API产品列表
    client = create_client()
    api_products = get_all_products(client)

    if not api_products:
        print("未获取到API产品")
        client.close()
        return 1

    # 3. 匹配产品
    matched = match_products(api_products, nav_products)

    if not matched:
        print("无匹配产品")
        client.close()
        return 1

    # 4. 提取费率
    total = len(matched) if args.max == 0 else min(len(matched), args.max)
    stats = {'success': 0, 'fail': 0, 'skip': 0}

    logger.info(f"开始提取 {total} 个产品的费率...")

    for i, product in enumerate(matched[:total]):
        nav_code = product.get('nav_code', '')

        # 检查是否已有数据
        existing = get_fee_info('浦银理财', nav_code)
        if existing and existing.get('source') == 'prospectus':
            stats['skip'] += 1
            continue

        # 提取费率
        fees = extract_fees_from_product(product)

        # 检查是否有有效数据
        has_data = any(v is not None for k, v in fees.items()
                      if k not in ('has_redemption_fee', 'fee_schedule'))

        if has_data:
            fee_info = {
                'has_redemption_fee': fees.get('has_redemption_fee', False),
                'fee_schedule': fees.get('fee_schedule', []),
                'fee_description': '',
                'source': 'api',  # 来源是API而非PDF
                'subscription_fee': fees.get('subscription_fee'),
                'purchase_fee': fees.get('purchase_fee'),
                'sales_service_fee': fees.get('sales_service_fee'),
                'custody_fee': fees.get('custody_fee'),
                'management_fee': fees.get('management_fee'),
            }
            update_fee_info('浦银理财', nav_code, fee_info)
            stats['success'] += 1
        else:
            stats['fail'] += 1

        if (i + 1) % 100 == 0:
            logger.info(f"进度: {i+1}/{total}")
            save_fee_db()

    client.close()
    save_fee_db()

    print("\n" + "=" * 60)
    print(f"  费率采集完成")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['fail']}")
    print(f"  跳过: {stats['skip']}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
