# -*- coding: utf-8 -*-
"""
简化版PDF费率解析器 - 单线程顺序处理，更可靠
"""

import os
import sys
import json
import time

# 强制刷新输出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from parse_citic_fees import parse_pdf_fees
from redemption_fee_db import load_fee_db, save_fee_db, update_fee_info, get_fee_info

PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '中信')
PROGRESS_FILE = os.path.join(PDF_DIR, 'parse_progress.json')

def main():
    # 加载已解析的代码
    parsed_codes = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            parsed_codes = set(progress.get('parsed_codes', []))

    print(f"已解析产品代码: {len(parsed_codes)}")

    # 获取PDF列表
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    print(f"总PDF文件: {len(pdf_files)}")

    # 统计
    success = 0
    failed = 0
    skipped = 0
    new_codes = []

    start_time = time.time()

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_DIR, pdf_file)

        # 解析PDF
        result = parse_pdf_fees(pdf_path)

        if not result:
            failed += 1
            continue

        product_code = result.get('product_code')
        if not product_code:
            skipped += 1
            continue

        # 检查是否已解析
        if product_code in parsed_codes:
            skipped += 1
            continue

        # 检查是否已有prospectus数据
        existing = get_fee_info('中信银行', product_code)
        if existing and existing.get('source') == 'prospectus':
            skipped += 1
            continue

        # 保存费率信息
        has_data = any(v is not None for k, v in result.items()
                      if k not in ('product_code', 'has_redemption_fee', 'fee_schedule', 'fee_description'))

        if has_data or result.get('fee_schedule'):
            fee_info = {
                'has_redemption_fee': result.get('has_redemption_fee', False),
                'fee_schedule': result.get('fee_schedule', []),
                'fee_description': result.get('fee_description', ''),
                'source': 'prospectus',
                'subscription_fee': result.get('subscription_fee'),
                'purchase_fee': result.get('purchase_fee'),
                'sales_service_fee': result.get('sales_service_fee'),
                'custody_fee': result.get('custody_fee'),
                'management_fee': result.get('management_fee'),
            }
            update_fee_info('中信银行', product_code, fee_info)
            success += 1
            new_codes.append(product_code)
            parsed_codes.add(product_code)
        else:
            failed += 1

        # 每50个新产品保存一次
        if success > 0 and success % 50 == 0:
            save_fee_db()
            # 保存进度
            progress = {'parsed_codes': list(parsed_codes)}
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"[{i+1}/{len(pdf_files)}] 成功:{success} 失败:{failed} 跳过:{skipped} ({rate:.1f} PDF/s)")

        # 每500个文件显示进度
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"[{i+1}/{len(pdf_files)}] 成功:{success} 失败:{failed} 跳过:{skipped} ({rate:.1f} PDF/s)")

    # 最终保存
    save_fee_db()
    progress = {'parsed_codes': list(parsed_codes)}
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    # 同步到SQLite
    print("\n同步到SQLite...")
    import subprocess
    subprocess.run([sys.executable, 'migrate_fees_to_sqlite.py'])

    elapsed = time.time() - start_time
    print(f"\n完成! 耗时: {elapsed:.1f}秒")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print(f"总产品: {len(parsed_codes)}")

if __name__ == '__main__':
    main()
