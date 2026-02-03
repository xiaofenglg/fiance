# -*- coding: utf-8 -*-
"""
批量解析PDF提取赎回费 — 多进程并行版本
支持民生银行产品说明书PDF解析

使用方法:
    python batch_parse_pdf_fees.py --bank 民生 --workers 8
"""

import os
import sys
import json
import re
import time
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz  # PyMuPDF

# 从现有模块导入解析函数
from crawl_citic_prospectus import parse_redemption_fee

# 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_BASE = os.path.join(BASE_DIR, 'pdfs')
PDF_ARCHIVE = os.path.join(BASE_DIR, 'pdf_archive')
FEE_DB_PATH = os.path.join(BASE_DIR, '赎回费数据库.json')

# 银行目录映射
BANK_PATHS = {
    '中信': os.path.join(PDF_BASE, '中信'),
    '民生': os.path.join(PDF_ARCHIVE, 'minsheng'),
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF提取文本（单个文件）"""
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return None


def extract_product_code_from_text(text: str, bank: str) -> str:
    """从PDF文本中提取产品代码"""
    if bank == '民生':
        # 民生产品代码格式: FBAEXXXXX, FBAGXXXXX
        patterns = [
            r'产品代码[：:]\s*(FBA[EG]\d+[A-Z]?)',
            r'理财登记编码[：:]\s*(FBA[EG]\d+[A-Z]?)',
            r'(FBA[EG]\d{5,}[A-Z]?)',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
    elif bank == '中信':
        # 中信产品代码格式: AFXXXXXX, ABXXXXXX
        patterns = [
            r'产品代码[：:]\s*(A[BF]\d+[A-Z]?)',
            r'(A[BF]\d{6,}[A-Z]?)',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
    return None


def process_single_pdf(args):
    """处理单个PDF文件（用于多进程）"""
    pdf_path, doc_id, bank_name = args

    try:
        # 提取文本
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 100:
            return None, 'empty_text'

        # 提取产品代码
        product_code = extract_product_code_from_text(text, bank_name)
        if not product_code:
            return None, 'no_code'

        # 解析赎回费
        redemption_fees = parse_redemption_fee(text)

        result = {
            'product_code': product_code,
            'bank': bank_name,
            'has_redemption_fee': len(redemption_fees) > 0,
            'fee_schedule': redemption_fees,
            'source': 'pdf_parse',
            'doc_id': doc_id,
            'parse_time': datetime.now().isoformat(),
        }

        return result, 'success'

    except Exception as e:
        return None, f'error: {str(e)[:50]}'


def collect_pdf_files(bank_name: str) -> list:
    """收集银行的所有PDF文件

    Returns:
        list of (pdf_path, doc_id, bank_name) tuples
    """
    bank_folder = BANK_PATHS.get(bank_name)
    if not bank_folder or not os.path.exists(bank_folder):
        return []

    tasks = []

    if bank_name == '中信':
        # 中信: 平面目录结构 pdfs/中信/*.pdf
        for filename in os.listdir(bank_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(bank_folder, filename)
                doc_id = filename[:-4]  # 去掉.pdf后缀
                tasks.append((pdf_path, doc_id, bank_name))

    elif bank_name == '民生':
        # 民生: 嵌套目录结构 pdf_archive/minsheng/{product_code}/*.pdf
        for product_dir in os.listdir(bank_folder):
            product_path = os.path.join(bank_folder, product_dir)
            if os.path.isdir(product_path):
                for filename in os.listdir(product_path):
                    if filename.endswith('.pdf'):
                        pdf_path = os.path.join(product_path, filename)
                        doc_id = f"{product_dir}/{filename[:-4]}"
                        tasks.append((pdf_path, doc_id, bank_name))

    return tasks


def batch_parse_pdfs(bank_name: str, max_workers: int = None, limit: int = None):
    """批量解析PDF文件"""

    if max_workers is None:
        max_workers = min(cpu_count(), 16)  # 最多16个进程

    print(f"=" * 60)
    print(f"  批量解析PDF赎回费 — {bank_name}银行")
    print(f"  并行进程数: {max_workers}")
    print(f"=" * 60)

    # 收集PDF文件
    tasks = collect_pdf_files(bank_name)
    if not tasks:
        print(f"错误: 未找到{bank_name}的PDF文件")
        return

    print(f"发现PDF文件数: {len(tasks)}")

    if limit:
        tasks = tasks[:limit]

    print(f"待处理PDF数: {len(tasks)}")

    # 加载现有费率数据库
    with open(FEE_DB_PATH, 'r', encoding='utf-8') as f:
        fee_db = json.load(f)
    products = fee_db.get('products', {})

    # 统计
    stats = {
        'total': len(tasks),
        'success': 0,
        'has_fee': 0,
        'no_fee': 0,
        'no_code': 0,
        'empty_text': 0,
        'error': 0,
    }

    start_time = time.time()

    # 并行处理
    print(f"\n开始解析...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdf, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            result, status = future.result()

            if status == 'success' and result:
                product_key = f"{result['bank']}银行|{result['product_code']}"

                # 只有当：1)产品不存在 或 2)新结果有费率 时才更新
                # 避免无费率的PDF覆盖有费率的结果
                existing = products.get(product_key)
                should_update = (
                    existing is None or
                    result['has_redemption_fee'] or
                    not existing.get('has_redemption_fee')
                )

                if should_update:
                    products[product_key] = {
                        'has_redemption_fee': result['has_redemption_fee'],
                        'fee_schedule': result['fee_schedule'],
                        'source': result['source'],
                        'doc_id': result['doc_id'],
                    }

                stats['success'] += 1
                if result['has_redemption_fee']:
                    stats['has_fee'] += 1
                else:
                    stats['no_fee'] += 1
            elif status == 'no_code':
                stats['no_code'] += 1
            elif status == 'empty_text':
                stats['empty_text'] += 1
            else:
                stats['error'] += 1

            # 进度显示
            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / speed if speed > 0 else 0
                print(f"  进度: {i+1}/{len(tasks)} ({100*(i+1)/len(tasks):.1f}%) "
                      f"| 速度: {speed:.1f}/s | ETA: {eta:.0f}s "
                      f"| 有费:{stats['has_fee']} 无费:{stats['no_fee']}")

            # 每500个保存一次
            if (i + 1) % 500 == 0:
                fee_db['products'] = products
                fee_db['last_update'] = datetime.now().isoformat()
                with open(FEE_DB_PATH, 'w', encoding='utf-8') as f:
                    json.dump(fee_db, f, ensure_ascii=False, indent=2)
                print(f"  [保存] 已写入{stats['success']}条记录")

    # 最终保存
    fee_db['products'] = products
    fee_db['last_update'] = datetime.now().isoformat()
    with open(FEE_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(fee_db, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  解析完成!")
    print(f"{'='*60}")
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"    - 有赎回费: {stats['has_fee']}")
    print(f"    - 无赎回费: {stats['no_fee']}")
    print(f"  无产品代码: {stats['no_code']}")
    print(f"  文本为空: {stats['empty_text']}")
    print(f"  错误: {stats['error']}")
    print(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    print(f"  速度: {stats['total']/elapsed:.1f} PDF/秒")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='批量解析PDF提取赎回费')
    parser.add_argument('--bank', type=str, default='民生', help='银行名称 (民生/中信)')
    parser.add_argument('--workers', type=int, default=None, help='并行进程数')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量(测试用)')

    args = parser.parse_args()

    batch_parse_pdfs(
        bank_name=args.bank,
        max_workers=args.workers,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
