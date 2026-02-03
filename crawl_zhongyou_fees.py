# -*- coding: utf-8 -*-
"""
中邮理财费率采集器

API: https://www.psbc-wm.com/pswm-api
- /product/search: 产品列表
- /product/nvlist: 净值历史

注意: 需要绕过WAF (状态码412)

用法:
    python crawl_zhongyou_fees.py              # 采集全部
    python crawl_zhongyou_fees.py --max 100    # 限量测试
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

import httpx

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_zhongyou_fees.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '中邮')

sys.path.insert(0, BASE_DIR)
from nav_db_excel import NAVDatabaseExcel
from redemption_fee_db import load_fee_db, save_fee_db, update_fee_info, get_fee_info

BASE_URL = "https://www.psbc-wm.com"
API_BASE = f"{BASE_URL}/pswm-api"


# ============================================================
# HTTP客户端
# ============================================================

def create_client():
    """创建httpx客户端"""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
    except AttributeError:
        pass

    return httpx.Client(
        verify=ctx,
        timeout=60,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Origin': BASE_URL,
            'Referer': f'{BASE_URL}/products/index.html',
        },
        follow_redirects=True,
    )


def get_cookies_via_selenium() -> Dict[str, str]:
    """使用Selenium获取cookies绕过WAF"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        logger.error("需要安装selenium: pip install selenium")
        return {}

    logger.info("启动浏览器获取cookies (绕过WAF)...")

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)

        # 访问产品页面
        driver.get(f"{BASE_URL}/products/index.html")
        time.sleep(3)

        # 获取cookies
        cookies = {c['name']: c['value'] for c in driver.get_cookies()}
        logger.info(f"获取到 {len(cookies)} 个cookies")
        return cookies

    except Exception as e:
        logger.error(f"Selenium获取cookies失败: {e}")
        return {}
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


# ============================================================
# 中邮API
# ============================================================

def fetch_product_list(client: httpx.Client, page: int = 1, size: int = 10) -> Dict:
    """获取产品列表"""
    params = {
        'sale_object_indi': '1',  # 个人理财
        'pageNum': page,
        'pageSize': size,
        'sort': '',
        'order': '',
    }

    try:
        r = client.get(f"{API_BASE}/product/search", params=params)

        if r.status_code == 412:
            logger.warning("触发WAF (412)，需要重新获取cookies")
            return {'waf_blocked': True}

        if r.status_code == 200:
            return r.json()
        else:
            logger.error(f"获取产品列表失败: {r.status_code}")
            return {}

    except Exception as e:
        logger.error(f"获取产品列表异常: {e}")
        return {}


def get_all_products(client: httpx.Client) -> List[Dict]:
    """获取全部产品列表"""
    logger.info("获取中邮理财产品列表...")

    all_products = []
    page = 1
    retry_count = 0

    while True:
        data = fetch_product_list(client, page=page, size=100)

        # 检查WAF
        if data.get('waf_blocked'):
            if retry_count < 3:
                logger.warning("尝试重新获取cookies...")
                cookies = get_cookies_via_selenium()
                if cookies:
                    for name, value in cookies.items():
                        client.cookies.set(name, value, domain='www.psbc-wm.com')
                    retry_count += 1
                    continue
            logger.error("WAF绕过失败")
            break

        if data.get('state') != 'ok':
            break

        products = data.get('data', {}).get('list', [])
        total = data.get('data', {}).get('totalRow', 0)
        total_page = data.get('data', {}).get('totalPage', 0)

        if not products:
            break

        all_products.extend(products)

        logger.info(f"获取第{page}/{total_page}页, 本页{len(products)}个, 累计{len(all_products)}/{total}")

        if page >= total_page:
            break

        page += 1
        time.sleep(0.5)
        retry_count = 0

    logger.info(f"共获取 {len(all_products)} 个产品")
    return all_products


# ============================================================
# 净值数据库匹配
# ============================================================

def get_nav_products() -> Dict[str, str]:
    """从净值数据库获取中邮产品列表"""
    logger.info("读取净值数据库中的中邮产品...")

    try:
        db = NAVDatabaseExcel()
        df = db.data.get('中邮理财')

        if df is None or df.empty:
            logger.warning("净值数据库中无中邮数据")
            return {}

        products = {}
        for idx in df.index:
            code, name = idx
            products[code] = name

        logger.info(f"从净值数据库获取 {len(products)} 个中邮产品")
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
        code = p.get('wp_code') or p.get('prd_code') or ''
        name = p.get('wp_name') or p.get('prd_name') or ''

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

    logger.info(f"匹配成功 {len(matched)} 个产品 ({len(matched)*100//max(len(nav_products),1)}%)")
    return matched


# ============================================================
# 费率解析
# ============================================================

def extract_fees_from_product(product: Dict) -> Dict:
    """从产品信息中提取费率"""
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

    # 中邮API可能的费率字段
    rate_fields = {
        'mng_rate': 'management_fee',
        'trust_rate': 'custody_fee',
        'sale_rate': 'sales_service_fee',
        'subs_rate': 'subscription_fee',
        'pur_rate': 'purchase_fee',
        'rdm_rate': 'redemption_fee',
    }

    for api_field, fee_key in rate_fields.items():
        value = product.get(api_field)
        if value is not None:
            try:
                rate = float(value)
                if rate > 1:
                    rate = rate / 100
                fees[fee_key] = rate
            except (ValueError, TypeError):
                pass

    # 从名称解析
    name = product.get('wp_name') or product.get('prd_name') or ''

    if fees.get('redemption_fee') and fees['redemption_fee'] > 0:
        fees['has_redemption_fee'] = True

    return fees


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='中邮理财费率采集器')
    parser.add_argument('--max', type=int, default=0, help='限制处理数量')
    parser.add_argument('--use-selenium', action='store_true', help='使用Selenium获取cookies')
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)

    print("=" * 60)
    print("  中邮理财费率采集器")
    print("=" * 60)

    # 1. 获取NAV产品
    nav_products = get_nav_products()
    if not nav_products:
        print("净值数据库中无中邮产品")
        return 1

    # 2. 创建客户端
    client = create_client()

    # 如果需要，用Selenium获取cookies
    if args.use_selenium:
        cookies = get_cookies_via_selenium()
        if cookies:
            for name, value in cookies.items():
                client.cookies.set(name, value, domain='www.psbc-wm.com')

    # 3. 获取API产品列表
    api_products = get_all_products(client)

    if not api_products:
        print("未获取到API产品，可能需要 --use-selenium 参数绕过WAF")
        client.close()
        return 1

    # 4. 匹配产品
    matched = match_products(api_products, nav_products)

    if not matched:
        print("无匹配产品")
        client.close()
        return 1

    # 5. 提取费率
    total = len(matched) if args.max == 0 else min(len(matched), args.max)
    stats = {'success': 0, 'fail': 0, 'skip': 0}

    logger.info(f"开始提取 {total} 个产品的费率...")

    for i, product in enumerate(matched[:total]):
        nav_code = product.get('nav_code', '')

        existing = get_fee_info('中邮理财', nav_code)
        if existing and existing.get('source') == 'prospectus':
            stats['skip'] += 1
            continue

        fees = extract_fees_from_product(product)

        has_data = any(v is not None for k, v in fees.items()
                      if k not in ('has_redemption_fee', 'fee_schedule'))

        if has_data:
            fee_info = {
                'has_redemption_fee': fees.get('has_redemption_fee', False),
                'fee_schedule': fees.get('fee_schedule', []),
                'fee_description': '',
                'source': 'api',
                'subscription_fee': fees.get('subscription_fee'),
                'purchase_fee': fees.get('purchase_fee'),
                'sales_service_fee': fees.get('sales_service_fee'),
                'custody_fee': fees.get('custody_fee'),
                'management_fee': fees.get('management_fee'),
            }
            update_fee_info('中邮理财', nav_code, fee_info)
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
