# -*- coding: utf-8 -*-
"""
宁银理财费率采集器

策略:
1. 扫描附件列表，筛选销售文件 (_07_ 格式)
2. 从标题提取产品名称，与NAV库匹配
3. 下载匹配的PDF并解析费率

用法:
    python crawl_ningyin_fees.py              # 采集全部
    python crawl_ningyin_fees.py --max 50     # 限量测试
    python crawl_ningyin_fees.py --parse-only # 仅解析已下载的PDF
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
from typing import Dict, List, Optional, Set, Tuple

import httpx

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_ningyin_fees.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '宁银')
INDEX_FILE = os.path.join(PDF_DIR, 'index.json')

sys.path.insert(0, BASE_DIR)
from nav_db_excel import NAVDatabaseExcel
from redemption_fee_db import load_fee_db, save_fee_db, update_fee_info, get_fee_info

BASE_URL = "https://www.wmbnb.com"
API_PREFIX = "/ningbo-web/;a"


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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': f'{BASE_URL}/info/funddetails/index.html',
        },
        follow_redirects=True,
    )


# ============================================================
# 净值数据库产品列表
# ============================================================

def get_nav_products() -> Dict[str, str]:
    """从净值数据库获取宁银产品列表

    Returns:
        {product_code: product_name}
    """
    logger.info("读取净值数据库中的宁银产品...")

    try:
        db = NAVDatabaseExcel()
        ningyin_df = db.data.get('宁银理财')

        if ningyin_df is None or ningyin_df.empty:
            logger.warning("净值数据库中无宁银数据")
            return {}

        # DataFrame的index是MultiIndex(产品代码, 产品名称)
        products = {}
        for idx in ningyin_df.index:
            code, name = idx
            products[code] = name

        logger.info(f"从净值数据库获取 {len(products)} 个宁银产品")
        return products

    except Exception as e:
        logger.error(f"读取净值数据库失败: {e}")
        return {}


def normalize_for_match(name: str) -> str:
    """规范化产品名称用于模糊匹配

    提取: 系列 + 产品号 + 份额类型
    """
    # 移除"宁银理财"前缀
    name = name.replace('宁银理财', '')

    parts = []

    # 系列关键词
    series = ['宁欣', '晶耀', '皎月', '甄享', '沁宁', '宁享', '天天鎏金', '日日薪']
    for s in series:
        if s in name:
            parts.append(s)
            break

    # 产品类型
    types = ['封闭式', '日开', '定开', '定期开放', '日申周赎', '日申半年赎', '现金管理']
    for t in types:
        if t in name:
            parts.append(t)
            break

    # 产品号
    num_match = re.search(r'(\d+)号', name)
    if num_match:
        parts.append(num_match.group(1) + '号')

    # 份额类型
    share_match = re.search(r'-([A-Z])(?:份额)?', name)
    if share_match:
        parts.append('-' + share_match.group(1))

    return '|'.join(parts)


def build_nav_index(nav_products: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """构建NAV产品索引用于匹配

    Returns:
        {normalized_key: (nav_code, nav_name)}
    """
    index = {}
    for code, name in nav_products.items():
        key = normalize_for_match(name)
        if key and key not in index:
            index[key] = (code, name)
    return index


# ============================================================
# 扫描销售文件
# ============================================================

def scan_sales_documents(client: httpx.Client, nav_products: Dict[str, str],
                         max_pages: int = 5000) -> List[Dict]:
    """扫描附件列表，查找匹配NAV库的销售文件

    销售文件URL格式: /report/{产品代码}_07_{描述}_{时间戳}.pdf

    Returns:
        匹配的文档列表 [{title, url, nav_code, nav_name}, ...]
    """
    logger.info(f"扫描销售文件 (_07_ 格式)，匹配 {len(nav_products)} 个NAV产品...")

    # 构建匹配索引
    nav_index = build_nav_index(nav_products)
    logger.info(f"构建NAV索引: {len(nav_index)} 个唯一标识")

    matched_docs = []
    seen_nav_codes = set()  # 每个NAV产品只需一份说明书
    page = 1
    total_scanned = 0

    while page <= max_pages:
        try:
            r = client.get(
                f"{BASE_URL}{API_PREFIX}/product/attachmentlist.json",
                params={'request_num': 100, 'request_pageno': page}
            )

            if r.status_code != 200:
                break

            data = r.json()
            if data.get('status') != 'success':
                break

            docs = data.get('list', [])
            if not docs:
                break

            for doc in docs:
                url = doc.get('url', '')
                title = doc.get('title', '')

                # 只处理销售文件 (_07_ 格式)
                if '_07_' not in url:
                    continue

                # 从标题规范化
                title_key = normalize_for_match(title)

                # 尝试匹配NAV产品
                if title_key in nav_index:
                    nav_code, nav_name = nav_index[title_key]
                    if nav_code not in seen_nav_codes:
                        seen_nav_codes.add(nav_code)
                        matched_docs.append({
                            'title': title,
                            'url': url,
                            'nav_code': nav_code,
                            'nav_name': nav_name,
                        })

            total = data.get('total', 0)
            total_scanned += len(docs)

            if page % 100 == 0:
                logger.info(f"扫描第{page}页, 已匹配{len(matched_docs)}个, "
                           f"已扫描{total_scanned}/{total}")

            # 如果已找到大部分产品，提前退出
            if len(seen_nav_codes) >= len(nav_products) * 0.8:
                logger.info(f"已匹配80%以上产品 ({len(seen_nav_codes)}/{len(nav_products)})")
                break

            page += 1
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"获取第{page}页失败: {e}")
            break

    logger.info(f"扫描完成: 共{len(matched_docs)}个匹配 "
               f"(覆盖率: {len(matched_docs)*100//len(nav_products)}%)")
    return matched_docs


def download_pdf(client: httpx.Client, doc: Dict, pdf_dir: str) -> str:
    """下载PDF文件

    Returns:
        'ok' | 'skip' | 'fail'
    """
    url = doc.get('url', '')
    if not url:
        return 'fail'

    filename = url.split('/')[-1] if '/' in url else f"{hash(url)}.pdf"
    filepath = os.path.join(pdf_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return 'skip'

    full_url = f"{BASE_URL}{url}" if url.startswith('/') else url

    try:
        r = client.get(full_url, timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return 'ok'
        return 'fail'
    except Exception as e:
        logger.debug(f"下载失败: {e}")
        return 'fail'


# ============================================================
# PDF解析
# ============================================================

def parse_pdf_fees(pdf_path: str) -> Dict:
    """解析PDF提取费率信息"""
    result = {
        'management_fee': None,
        'custody_fee': None,
        'sales_service_fee': None,
        'subscription_fee': None,
        'purchase_fee': None,
        'redemption_fee': None,
        'has_redemption_fee': False,
        'fee_schedule': [],
    }

    if pdfplumber is None:
        return result

    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ''
            for page in pdf.pages[:15]:  # 读前15页
                text = page.extract_text()
                if text:
                    full_text += text + '\n'

            if not full_text:
                return result

            result.update(parse_fees_from_text(full_text))

    except Exception as e:
        logger.debug(f"PDF解析失败 {pdf_path}: {e}")

    return result


def parse_fees_from_text(text: str) -> Dict:
    """从文本中解析费率"""
    result = {}

    # 费率正则模式 - 支持【】包裹格式
    patterns = [
        (r'(?:固定)?管理费[率]?[：:为\s]*【?([\d.]+)\s*%】?\s*/?\s*年?', 'management_fee'),
        (r'托管费[率]?[：:为\s]*【?([\d.]+)\s*%】?\s*/?\s*年?', 'custody_fee'),
        (r'销售(?:服务)?费[率]?[：:为\s]*【?([\d.]+)\s*%】?\s*/?\s*年?', 'sales_service_fee'),
        (r'认购费[率]?[：:为\s]*【?([\d.]+)\s*%】?', 'subscription_fee'),
        (r'申购费[率]?[：:为\s]*【?([\d.]+)\s*%】?', 'purchase_fee'),
        (r'赎回费[率]?[：:为\s]*【?([\d.]+)\s*%】?', 'redemption_fee'),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                rate_pct = float(match.group(1))
                result[key] = rate_pct / 100
            except ValueError:
                pass

    # 检查是否有赎回费
    if '不收取赎回费' in text or '赎回费：0' in text or '赎回费率：0' in text:
        result['has_redemption_fee'] = False
        result['redemption_fee'] = 0.0
    elif '赎回费' in text and result.get('redemption_fee'):
        result['has_redemption_fee'] = True

    # 解析赎回费阶梯
    schedule_pattern = r'持有[少不满]?[于]?\s*(\d+)\s*(?:天|日)[^，,;；]*(?:赎回费[率]?)\s*([\d.]+)\s*%'
    schedule_matches = re.findall(schedule_pattern, text)
    if schedule_matches:
        schedule = []
        prev_days = 0
        for days_str, rate_str in schedule_matches:
            days = int(days_str)
            rate = float(rate_str) / 100
            schedule.append({
                'min_days': prev_days,
                'max_days': days,
                'fee_rate': rate
            })
            prev_days = days
        if schedule:
            schedule.append({
                'min_days': prev_days,
                'max_days': 999999,
                'fee_rate': 0.0
            })
            result['fee_schedule'] = schedule
            result['has_redemption_fee'] = any(s['fee_rate'] > 0 for s in schedule)

    return result


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='宁银理财费率采集器')
    parser.add_argument('--max', type=int, default=0, help='限制处理数量')
    parser.add_argument('--parse-only', action='store_true', help='仅解析已下载的PDF')
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)

    print("=" * 60)
    print("  宁银理财费率采集器")
    print("=" * 60)

    # 1. 获取净值库中的产品
    nav_products = get_nav_products()
    if not nav_products:
        print("净值数据库中无宁银产品")
        return 1

    logger.info(f"净值库中有 {len(nav_products)} 个宁银产品")

    if not args.parse_only:
        # 2. 扫描并下载销售文件
        client = create_client()

        docs = scan_sales_documents(client, nav_products)

        if not docs:
            print("未找到匹配的销售文件")
            client.close()
            return 1

        # 保存索引
        index = {}
        for doc in docs:
            url = doc.get('url', '')
            filename = url.split('/')[-1] if '/' in url else ''
            if filename:
                index[filename] = {
                    'title': doc.get('title', ''),
                    'url': url,
                    'nav_code': doc.get('nav_code', ''),
                    'nav_name': doc.get('nav_name', ''),
                }

        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        # 下载PDF
        total = len(docs) if args.max == 0 else min(len(docs), args.max)
        stats = {'ok': 0, 'skip': 0, 'fail': 0}

        logger.info(f"开始下载 {total} 个文件...")

        for i, doc in enumerate(docs[:total]):
            result = download_pdf(client, doc, PDF_DIR)
            stats[result] += 1

            if (i + 1) % 20 == 0:
                logger.info(f"下载进度: {i+1}/{total} - 新下载:{stats['ok']} 跳过:{stats['skip']} 失败:{stats['fail']}")

            if result == 'ok':
                time.sleep(0.3)

        client.close()
        logger.info(f"下载完成: 新下载{stats['ok']} 跳过{stats['skip']} 失败{stats['fail']}")

    # 3. 解析PDF提取费率
    if pdfplumber is None:
        logger.error("pdfplumber未安装，无法解析PDF")
        return 1

    # 加载索引
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            index = json.load(f)
    else:
        index = {}

    logger.info(f"开始解析PDF提取费率 ({len(index)} 个文件)...")

    parse_stats = {'success': 0, 'fail': 0, 'skip': 0}
    total_parse = len(index) if args.max == 0 else min(len(index), args.max)

    for i, (filename, info) in enumerate(list(index.items())[:total_parse]):
        nav_code = info.get('nav_code', '')
        if not nav_code:
            parse_stats['skip'] += 1
            continue

        # 检查是否已有prospectus数据
        existing = get_fee_info('宁银理财', nav_code)
        if existing and existing.get('source') == 'prospectus':
            parse_stats['skip'] += 1
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        if not os.path.exists(pdf_path):
            parse_stats['fail'] += 1
            continue

        # 解析PDF
        fees = parse_pdf_fees(pdf_path)

        # 检查是否有有效数据
        has_data = any(v is not None for k, v in fees.items()
                      if k not in ('has_redemption_fee', 'fee_schedule'))

        if has_data:
            fee_info = {
                'has_redemption_fee': fees.get('has_redemption_fee', False),
                'fee_schedule': fees.get('fee_schedule', []),
                'fee_description': '',
                'source': 'prospectus',
                'subscription_fee': fees.get('subscription_fee'),
                'purchase_fee': fees.get('purchase_fee'),
                'sales_service_fee': fees.get('sales_service_fee'),
                'custody_fee': fees.get('custody_fee'),
                'management_fee': fees.get('management_fee'),
            }
            update_fee_info('宁银理财', nav_code, fee_info)
            parse_stats['success'] += 1
        else:
            parse_stats['fail'] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"解析进度: {i+1}/{total_parse}")
            save_fee_db()  # 定期保存

    # 保存
    save_fee_db()

    print("\n" + "=" * 60)
    print(f"  费率采集完成")
    print(f"  解析成功: {parse_stats['success']}")
    print(f"  解析失败: {parse_stats['fail']}")
    print(f"  跳过: {parse_stats['skip']}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
