# -*- coding: utf-8 -*-
"""
中信理财PDF费率解析器

解析已下载的PDF文件，提取费率信息并写入数据库
"""

import os
import sys
import re
import json
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    print("pdfplumber未安装，请运行: pip install pdfplumber")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from redemption_fee_db import load_fee_db, save_fee_db, update_fee_info, get_fee_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parse_citic_fees.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '中信')
PROGRESS_FILE = os.path.join(PDF_DIR, 'parse_progress.json')

# 产品代码正则 - 从PDF内容提取
PRODUCT_CODE_IN_PDF_RE = re.compile(r'产品代码[：:]\s*【?([A-Z]{2}\d{5,8}[A-Z]?)】?')
# 备用: 从文件名提取
PRODUCT_CODE_RE = re.compile(r'\b([A-Z]{2}\d{5,8}[A-Z]?)\b')

# 线程安全
stats_lock = Lock()
stats = {'success': 0, 'failed': 0, 'skipped': 0}


def _extract_fee_section(text):
    """提取费用相关段落"""
    patterns = [
        r'(?:产品费用|费用说明|收费标准|费率说明)[\s\S]{0,2000}?(?=\n\n|\Z)',
        r'(?:认购费|申购费|赎回费|管理费|托管费|销售服务费)[\s\S]{0,1500}',
    ]

    sections = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sections.extend(matches)

    return '\n'.join(sections) if sections else text[:5000]


def _parse_rate_fee(text, fee_name):
    """解析费率类型费用 - 支持【】包裹格式"""
    patterns = [
        # 格式: 销售服务费：费率【0.40%】/年
        rf'{fee_name}[率]?[：:]\s*费率\s*【([\d.]+)\s*%】',
        # 格式: 销售服务费：【0.40%】/年 或 销售服务费：0.40%/年
        rf'{fee_name}[率]?[：:为\s]*【?([\d.]+)\s*%】?\s*/?\s*年?',
        rf'{fee_name}[：:]\s*【?([\d.]+)\s*%】?',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1)) / 100
            except ValueError:
                pass
    return None


def parse_redemption_fee(text):
    """解析赎回费阶梯结构"""
    result = {
        'has_redemption_fee': False,
        'fee_schedule': [],
        'fee_description': '',
    }

    # 检查是否不收取赎回费
    no_fee_patterns = [
        r'不收取赎回费',
        r'赎回费[率]?[：:为\s]*[0零无]',
        r'无赎回费',
        r'免收赎回费',
    ]

    for pattern in no_fee_patterns:
        if re.search(pattern, text):
            result['fee_description'] = '无赎回费'
            return result

    # 解析阶梯费率
    schedule_patterns = [
        r'持有[期]?[少不满]?[于]?\s*(\d+)\s*(?:天|日|个月)[^，,;；\n]{0,50}?(?:赎回费[率]?)[为：:\s]*【?([\d.]+)\s*%】?',
        r'(\d+)\s*(?:天|日)[以内及][^，,;；\n]{0,30}?【?([\d.]+)\s*%】?',
    ]

    schedule = []
    for pattern in schedule_patterns:
        matches = re.findall(pattern, text)
        if matches:
            prev_days = 0
            for days_str, rate_str in matches:
                try:
                    days = int(days_str)
                    rate = float(rate_str) / 100
                    schedule.append({
                        'min_days': prev_days,
                        'max_days': days,
                        'fee_rate': rate
                    })
                    prev_days = days
                except ValueError:
                    continue

            if schedule:
                schedule.append({
                    'min_days': prev_days,
                    'max_days': 999999,
                    'fee_rate': 0.0
                })
                break

    if schedule:
        result['has_redemption_fee'] = any(s['fee_rate'] > 0 for s in schedule)
        result['fee_schedule'] = schedule
        result['fee_description'] = '; '.join(
            f"{s['min_days']}-{s['max_days']}天:{s['fee_rate']*100:.2f}%"
            for s in schedule if s['max_days'] < 999999
        )

    return result


def parse_pdf_fees(pdf_path):
    """解析单个PDF提取费率和产品代码"""
    result = {
        'product_code': None,
        'management_fee': None,
        'custody_fee': None,
        'sales_service_fee': None,
        'subscription_fee': None,
        'purchase_fee': None,
        'has_redemption_fee': False,
        'fee_schedule': [],
        'fee_description': '',
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ''
            for page in pdf.pages[:20]:  # 读前20页
                text = page.extract_text()
                if text:
                    full_text += text + '\n'

            if not full_text:
                return None

            # 提取产品代码
            code_match = PRODUCT_CODE_IN_PDF_RE.search(full_text)
            if code_match:
                result['product_code'] = code_match.group(1)

            # 提取费用相关段落
            fee_section = _extract_fee_section(full_text)

            # 解析各项费率
            result['management_fee'] = _parse_rate_fee(fee_section, '(?:固定)?管理费')
            result['custody_fee'] = _parse_rate_fee(fee_section, '托管费')
            result['sales_service_fee'] = _parse_rate_fee(fee_section, '销售(?:服务)?费')
            result['subscription_fee'] = _parse_rate_fee(fee_section, '认购费')
            result['purchase_fee'] = _parse_rate_fee(fee_section, '申购费')

            # 解析赎回费
            redemption_info = parse_redemption_fee(fee_section)
            result.update(redemption_info)

            return result

    except Exception as e:
        logger.debug(f"PDF解析失败 {pdf_path}: {e}")
        return None


def process_one_pdf(pdf_file, parsed_codes):
    """处理单个PDF文件"""
    pdf_path = os.path.join(PDF_DIR, pdf_file)

    # 解析PDF (包含提取产品代码)
    fees = parse_pdf_fees(pdf_path)

    if not fees:
        with stats_lock:
            stats['failed'] += 1
        return None

    # 从PDF内容获取产品代码
    product_code = fees.get('product_code')
    if not product_code:
        with stats_lock:
            stats['skipped'] += 1
        return None

    # 检查是否已有prospectus数据
    if product_code in parsed_codes:
        with stats_lock:
            stats['skipped'] += 1
        return None

    existing = get_fee_info('中信银行', product_code)
    if existing and existing.get('source') == 'prospectus':
        with stats_lock:
            stats['skipped'] += 1
        return None

    # 检查是否有有效数据
    has_data = any(v is not None for k, v in fees.items()
                  if k not in ('product_code', 'has_redemption_fee', 'fee_schedule', 'fee_description'))

    if has_data or fees.get('fee_schedule'):
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
        update_fee_info('中信银行', product_code, fee_info)

        with stats_lock:
            stats['success'] += 1
        return product_code

    with stats_lock:
        stats['failed'] += 1
    return None


def save_progress(parsed_codes, new_parsed):
    """保存解析进度"""
    all_parsed = list(parsed_codes) + new_parsed
    progress = {'parsed_codes': all_parsed}
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def migrate_to_sqlite():
    """将费率数据同步到SQLite"""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'migrate_fees_to_sqlite.py'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.info("SQLite同步完成")
        else:
            logger.warning(f"SQLite同步警告: {result.stderr}")
    except Exception as e:
        logger.warning(f"SQLite同步失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='中信理财PDF费率解析器')
    parser.add_argument('--workers', type=int, default=4, help='并行线程数 (默认4)')
    parser.add_argument('--max', type=int, default=None, help='限制处理数量')
    parser.add_argument('--batch', type=int, default=50, help='批次保存间隔 (默认50)')
    parser.add_argument('--sync-sqlite', action='store_true', help='每批次同步到SQLite')
    args = parser.parse_args()

    if not os.path.exists(PDF_DIR):
        print(f"PDF目录不存在: {PDF_DIR}")
        return 1

    print("=" * 60)
    print(f"  中信理财PDF费率解析器")
    print(f"  线程数: {args.workers}, 批次保存: 每{args.batch}个")
    print("=" * 60)

    # 获取PDF文件列表
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    logger.info(f"共有 {len(pdf_files)} 个PDF文件")

    if args.max:
        pdf_files = pdf_files[:args.max]
        logger.info(f"限制处理: {args.max} 个")

    # 加载已解析的代码
    parsed_codes = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            parsed_codes = set(progress.get('parsed_codes', []))
    logger.info(f"已解析产品代码: {len(parsed_codes)}")

    # 并行解析
    new_parsed = []
    batch_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_pdf, f, parsed_codes): f for f in pdf_files}

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result:
                    new_parsed.append(result)
                    batch_count += 1
            except Exception as e:
                logger.error(f"处理异常: {e}")

            # 批次保存
            if batch_count >= args.batch:
                logger.info(f"进度: {i+1}/{len(pdf_files)} - "
                           f"成功:{stats['success']} 失败:{stats['failed']} 跳过:{stats['skipped']}")
                save_fee_db()
                save_progress(parsed_codes, new_parsed)
                if args.sync_sqlite:
                    migrate_to_sqlite()
                batch_count = 0

            # 每500个显示进度
            if (i + 1) % 500 == 0:
                logger.info(f"进度: {i+1}/{len(pdf_files)} - "
                           f"成功:{stats['success']} 失败:{stats['failed']} 跳过:{stats['skipped']}")

    # 最终保存
    save_fee_db()
    save_progress(parsed_codes, new_parsed)

    # 最终同步到SQLite
    logger.info("执行最终SQLite同步...")
    migrate_to_sqlite()

    print("\n" + "=" * 60)
    print(f"  解析完成!")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"  总产品: {len(parsed_codes) + len(new_parsed)}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
