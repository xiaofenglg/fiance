# -*- coding: utf-8 -*-
"""
华夏理财费率采集器

数据来源: https://www.hxwm.com.cn
- 销售文件JS: /common/js/gmcpxswjData.js
- prdType="1": 理财产品说明书

用法:
    python crawl_huaxia_fees.py              # 采集全部
    python crawl_huaxia_fees.py --max 100    # 限量测试
    python crawl_huaxia_fees.py --download   # 仅下载PDF
"""

import os
import sys
import re
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import requests

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_huaxia_fees.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '华夏')

sys.path.insert(0, BASE_DIR)
from nav_db_excel import NAVDatabaseExcel
from redemption_fee_db import load_fee_db, save_fee_db, update_fee_info, get_fee_info

BASE_URL = "https://www.hxwm.com.cn"
DATA_JS_URL = f"{BASE_URL}/common/js/gmcpxswjData.js"


# ============================================================
# HTTP客户端
# ============================================================

def create_session():
    """创建requests会话"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    })
    return session


# ============================================================
# 获取产品说明书列表
# ============================================================

def fetch_prospectus_list(session: requests.Session) -> List[Dict]:
    """获取产品说明书列表

    从gmcpxswjData.js解析数据，筛选prdType="1"的说明书
    """
    logger.info("获取华夏理财产品说明书列表...")

    try:
        r = session.get(DATA_JS_URL, timeout=60)
        if r.status_code != 200:
            logger.error(f"获取JS文件失败: {r.status_code}")
            return []

        # 解析JS数组
        # 格式: var cpxswjData = [{...}, {...}, ...]
        text = r.text
        match = re.search(r'var\s+cpxswjData\s*=\s*\[(.*)\]', text, re.DOTALL)
        if not match:
            logger.error("无法解析JS数据")
            return []

        # 提取每个对象
        items = []
        content = match.group(1)

        # 使用正则提取每个对象
        pattern = r'\{[^{}]*"address"\s*:\s*"([^"]+)"[^{}]*"title"\s*:\s*"([^"]+)"[^{}]*"prdType"\s*:\s*"([^"]*)"[^{}]*\}'
        for m in re.finditer(pattern, content, re.DOTALL):
            address, title, prd_type = m.groups()
            if prd_type == "1":  # 只要产品说明书
                items.append({
                    'address': address,
                    'title': title,
                    'prd_type': prd_type,
                })

        logger.info(f"共获取 {len(items)} 个产品说明书")
        return items

    except Exception as e:
        logger.error(f"获取说明书列表失败: {e}")
        return []


# ============================================================
# 净值数据库匹配
# ============================================================

def get_nav_products() -> Dict[str, str]:
    """从净值数据库获取华夏产品列表"""
    logger.info("读取净值数据库中的华夏产品...")

    try:
        db = NAVDatabaseExcel()
        df = db.data.get('华夏银行')

        if df is None or df.empty:
            logger.warning("净值数据库中无华夏数据")
            return {}

        products = {}
        for idx in df.index:
            code, name = idx
            products[code] = name

        logger.info(f"从净值数据库获取 {len(products)} 个华夏产品")
        return products

    except Exception as e:
        logger.error(f"读取净值数据库失败: {e}")
        return {}


def extract_product_key(name: str) -> List[str]:
    """提取产品关键标识（可能返回多个候选）"""
    keys = []

    # 清理前缀后缀
    name = re.sub(r'^华夏理财', '', name)
    name = re.sub(r'有限责任公司', '', name)
    name = re.sub(r'说明书$', '', name)
    name = re.sub(r'[（(][^)）]*[)）]', '', name)

    # 提取产品编号
    number_match = re.search(r'(\d+)号', name)
    number = number_match.group(1) if number_match else None

    if number:
        # 尝试提取系列名（通常是开头2-3个字）
        # 合盈、悦安、龙盈、鑫安、稳盈、增盈、天天盈等
        series_match = re.match(r'^([\u4e00-\u9fa5]{2,3})', name)
        if series_match:
            series = series_match.group(1)
            keys.append(f"{series}{number}号")

        # 也尝试完整系列名 (如"固定收益债权型封闭式")
        full_match = re.search(r'([\u4e00-\u9fa5]+?)(?:理财产品)?(\d+)号', name)
        if full_match:
            full_series = full_match.group(1)
            keys.append(f"{full_series}{number}号")

    # 处理字母款式 (A款, B款等)
    letter_match = re.search(r'([A-Z])款', name)
    if letter_match:
        letter = letter_match.group(1)
        series_match = re.match(r'^([\u4e00-\u9fa5]+)', name)
        if series_match:
            series = series_match.group(1)
            keys.append(f"{series}{letter}款")

    return keys


def match_prospectus_to_nav(items: List[Dict], nav_products: Dict[str, str]) -> List[Dict]:
    """匹配说明书与NAV产品"""
    logger.info(f"匹配 {len(items)} 个说明书与 {len(nav_products)} 个NAV产品...")

    matched = []
    matched_codes = set()  # 避免重复匹配

    # 预处理NAV产品，提取关键标识
    nav_keys = {}  # key -> (code, name)
    for code, name in nav_products.items():
        keys = extract_product_key(name)
        for key in keys:
            if key not in nav_keys:
                nav_keys[key] = (code, name)

    # 按系列名+编号匹配
    for item in items:
        title = item['title']
        title_keys = extract_product_key(title)

        found = False
        for title_key in title_keys:
            if title_key in nav_keys:
                nav_code, nav_name = nav_keys[title_key]
                if nav_code not in matched_codes:
                    item['nav_code'] = nav_code
                    item['nav_name'] = nav_name
                    matched.append(item)
                    matched_codes.add(nav_code)
                    found = True
                    break

        if found:
            continue

        # 备用：直接名称包含匹配
        for nav_code, nav_name in nav_products.items():
            if nav_code in matched_codes:
                continue
            if nav_name and title:
                # 去掉"华夏理财"前缀后比较
                nav_clean = nav_name.replace('华夏理财', '')
                title_clean = title.replace('华夏理财', '').replace('说明书', '')
                if nav_clean in title_clean or title_clean in nav_clean:
                    item['nav_code'] = nav_code
                    item['nav_name'] = nav_name
                    matched.append(item)
                    matched_codes.add(nav_code)
                    break

    logger.info(f"匹配成功 {len(matched)} 个 ({len(matched)*100//max(len(nav_products),1)}%)")
    return matched


# ============================================================
# PDF下载与解析
# ============================================================

def download_pdf(session: requests.Session, item: Dict) -> Optional[str]:
    """下载PDF文件"""
    address = item['address']
    url = f"{BASE_URL}{address}"

    # 生成文件名
    filename = os.path.basename(address)
    filepath = os.path.join(PDF_DIR, filename)

    # 检查是否已下载
    if os.path.exists(filepath):
        return filepath

    try:
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return filepath
        else:
            logger.debug(f"下载失败 {r.status_code}: {filename}")
            return None

    except Exception as e:
        logger.debug(f"下载异常: {e}")
        return None


def extract_fees_from_pdf(pdf_path: str) -> Dict:
    """从PDF中提取费率信息"""
    if not pdfplumber:
        logger.warning("pdfplumber未安装，无法解析PDF")
        return {}

    fees = {
        'management_fee': None,
        'custody_fee': None,
        'sales_service_fee': None,
        'subscription_fee': None,
        'purchase_fee': None,
        'redemption_fee': None,
        'has_redemption_fee': False,
        'fee_schedule': [],
        'fee_description': '',
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages[:10]:  # 只读前10页
                text += page.extract_text() or ''

            if not text:
                return fees

            # 费率正则模式
            patterns = {
                'management_fee': [
                    r'(?:管理费|固定管理费)[率]?[：:为\s【]*([0-9.]+)\s*%',
                    r'管理费率[为：:\s【]*([0-9.]+)\s*%',
                ],
                'custody_fee': [
                    r'(?:托管费|保管费)[率]?[：:为\s【]*([0-9.]+)\s*%',
                    r'托管费率[为：:\s【]*([0-9.]+)\s*%',
                ],
                'sales_service_fee': [
                    r'(?:销售服务费|销售费)[率]?[：:为\s【]*([0-9.]+)\s*%',
                    r'销售服务费率[为：:\s【]*([0-9.]+)\s*%',
                ],
                'subscription_fee': [
                    r'(?:认购费|申购费)[率]?[：:为\s【]*([0-9.]+)\s*%',
                    r'认购费率[为：:\s【]*([0-9.]+)\s*%',
                ],
                'purchase_fee': [
                    r'申购费[率]?[：:为\s【]*([0-9.]+)\s*%',
                ],
            }

            for fee_key, regexes in patterns.items():
                for pattern in regexes:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            fees[fee_key] = float(match.group(1)) / 100
                        except ValueError:
                            pass
                        break

            # 赎回费解析 - 可能有阶梯
            redemption_patterns = [
                r'赎回费[率]?[：:为\s【]*([0-9.]+)\s*%',
                r'([0-9.]+)\s*%.*赎回费',
            ]

            # 检查是否有赎回费
            if '不收取赎回费' in text or '赎回费为0' in text or '赎回费率0' in text:
                fees['has_redemption_fee'] = False
                fees['redemption_fee'] = 0
            else:
                for pattern in redemption_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            rate = float(match.group(1)) / 100
                            fees['redemption_fee'] = rate
                            fees['has_redemption_fee'] = rate > 0
                        except ValueError:
                            pass
                        break

                # 解析阶梯费率
                schedule_pattern = r'([0-9]+)(?:天|日|个月)[以内\-至到~]*([0-9]+)?(?:天|日|个月)?[：:,，\s]*([0-9.]+)\s*%'
                schedules = []
                for match in re.finditer(schedule_pattern, text):
                    try:
                        start_days = int(match.group(1))
                        end_days = int(match.group(2)) if match.group(2) else None
                        rate = float(match.group(3)) / 100
                        schedules.append({
                            'min_days': start_days,
                            'max_days': end_days,
                            'rate': rate
                        })
                    except (ValueError, TypeError):
                        pass

                if schedules:
                    fees['fee_schedule'] = schedules
                    fees['has_redemption_fee'] = any(s.get('rate', 0) > 0 for s in schedules)

    except Exception as e:
        logger.debug(f"解析PDF失败: {e}")

    return fees


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='华夏理财费率采集器')
    parser.add_argument('--max', type=int, default=0, help='限制处理数量')
    parser.add_argument('--download', action='store_true', help='仅下载PDF')
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)

    print("=" * 60)
    print("  华夏理财费率采集器")
    print("=" * 60)

    # 1. 获取NAV产品
    nav_products = get_nav_products()
    if not nav_products:
        print("净值数据库中无华夏产品")
        return 1

    # 2. 获取说明书列表
    session = create_session()
    items = fetch_prospectus_list(session)

    if not items:
        print("未获取到说明书")
        return 1

    # 3. 匹配产品
    matched = match_prospectus_to_nav(items, nav_products)

    if not matched:
        print("无匹配产品")
        return 1

    # 4. 下载并解析
    total = len(matched) if args.max == 0 else min(len(matched), args.max)
    stats = {'success': 0, 'fail': 0, 'skip': 0, 'download': 0}

    logger.info(f"开始处理 {total} 个产品...")

    for i, item in enumerate(matched[:total]):
        nav_code = item.get('nav_code', '')

        # 检查是否已有数据
        existing = get_fee_info('华夏银行', nav_code)
        if existing and existing.get('source') == 'prospectus':
            stats['skip'] += 1
            continue

        # 下载PDF
        pdf_path = download_pdf(session, item)
        if not pdf_path:
            stats['fail'] += 1
            continue

        stats['download'] += 1

        if args.download:
            continue

        # 解析费率
        fees = extract_fees_from_pdf(pdf_path)

        has_data = any(v is not None for k, v in fees.items()
                      if k not in ('has_redemption_fee', 'fee_schedule', 'fee_description'))

        if has_data:
            fee_info = {
                'has_redemption_fee': fees.get('has_redemption_fee', False),
                'fee_schedule': fees.get('fee_schedule', []),
                'fee_description': fees.get('fee_description', ''),
                'source': 'prospectus',
                'subscription_fee': fees.get('subscription_fee'),
                'purchase_fee': fees.get('purchase_fee'),
                'sales_service_fee': fees.get('sales_service_fee'),
                'custody_fee': fees.get('custody_fee'),
                'management_fee': fees.get('management_fee'),
            }
            update_fee_info('华夏银行', nav_code, fee_info)
            stats['success'] += 1
        else:
            stats['fail'] += 1

        if (i + 1) % 100 == 0:
            logger.info(f"进度: {i+1}/{total}")
            save_fee_db()

        time.sleep(0.2)  # 避免请求过快

    save_fee_db()

    print("\n" + "=" * 60)
    print(f"  费率采集完成")
    print(f"  下载: {stats['download']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['fail']}")
    print(f"  跳过: {stats['skip']}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
