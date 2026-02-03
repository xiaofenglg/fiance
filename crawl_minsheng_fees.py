# -*- coding: utf-8 -*-
"""
民生银行费率独立采集程序

功能：
1. 获取产品列表
2. 下载发行公告 + 费率优惠公告 PDF（保存到本地）
3. 解析费率数据并存入费率数据库

用法：
    python crawl_minsheng_fees.py              # 增量采集（跳过已有数据）
    python crawl_minsheng_fees.py --full       # 全量采集（重新下载所有PDF）
    python crawl_minsheng_fees.py --limit 50   # 只处理前50个产品（测试用）
"""

import sys
import os
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_minsheng_fees.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from minsheng import CMBCWealthCrawler
from fee_rate_db import (load_fee_rate_db, save_fee_rate_db,
                         update_product_fees, needs_update, get_fee_rate_summary)


def crawl_fees(full_mode: bool = False, limit: int = 0):
    """采集民生银行费率数据

    Args:
        full_mode: True=全量采集, False=增量（跳过已有）
        limit: >0 时只处理前 N 个产品
    """
    mode_str = "全量" if full_mode else "增量"
    print("=" * 70)
    print(f"        民生银行费率采集程序 [{mode_str}模式]")
    print("=" * 70)

    crawler = CMBCWealthCrawler()
    crawler._warmup_session()

    # 1. 获取产品列表
    logger.info("获取产品列表...")
    products = crawler.get_product_list()
    if not products:
        logger.error("无法获取产品列表")
        return

    if limit > 0:
        products = products[:limit]
        logger.info(f"测试模式: 只处理前 {limit} 个产品")

    logger.info(f"共 {len(products)} 个产品待处理")

    # 统计
    stats = {
        'total': len(products),
        'new_base': 0,
        'new_discount': 0,
        'skipped': 0,
        'failed': 0,
        'pdf_downloaded': 0,
        'pdf_cached': 0,
    }

    # 2. 分批处理（每批使用独立会话）
    batch_size = 100
    batches = [products[i:i+batch_size] for i in range(0, len(products), batch_size)]

    logger.info(f"分 {len(batches)} 批处理，每批 {batch_size} 个")

    start_time = time.time()

    for bi, batch in enumerate(batches):
        logger.info(f"处理批次 {bi+1}/{len(batches)}...")
        session = crawler._create_session()

        for pi, product in enumerate(batch):
            code = product.get('REAL_PRD_CODE') or product.get('PRD_CODE', '')
            name = product.get('PRD_NAME', '')

            if not code:
                continue

            # 获取公告列表
            try:
                announcements = crawler.get_fee_announcements(code, session)
            except Exception as e:
                logger.debug(f"获取公告失败 {code}: {e}")
                stats['failed'] += 1
                continue

            issuance_date = None
            discount_date = None
            if announcements.get('issuance'):
                issuance_date = announcements['issuance'].get('date')
            if announcements.get('discount'):
                discount_date = announcements['discount'].get('date')

            # 增量模式: 检查是否需要更新
            if not full_mode and not needs_update('民生', code, issuance_date, discount_date):
                stats['skipped'] += 1
                continue

            # 下载并解析
            try:
                # 检查PDF是否已缓存
                iss_url = announcements.get('issuance', {}).get('url', '')
                disc_url = announcements.get('discount', {}).get('url', '')

                iss_cached = False
                disc_cached = False
                if iss_url:
                    local_path = crawler._get_pdf_local_path(iss_url)
                    if local_path and os.path.exists(local_path):
                        iss_cached = True
                if disc_url:
                    local_path = crawler._get_pdf_local_path(disc_url)
                    if local_path and os.path.exists(local_path):
                        disc_cached = True

                fee_data = crawler.scrape_product_fees(code, session)

                if fee_data:
                    fee_data['product_name'] = name
                    update_product_fees('民生', code, fee_data)

                    if fee_data.get('base_fees'):
                        stats['new_base'] += 1
                        if not iss_cached:
                            stats['pdf_downloaded'] += 1
                        else:
                            stats['pdf_cached'] += 1

                    if fee_data.get('discounts'):
                        stats['new_discount'] += 1
                        if not disc_cached:
                            stats['pdf_downloaded'] += 1
                        else:
                            stats['pdf_cached'] += 1
                else:
                    stats['failed'] += 1

            except Exception as e:
                logger.debug(f"费率采集失败 {code}: {e}")
                stats['failed'] += 1

            # 进度显示
            total_done = bi * batch_size + pi + 1
            if total_done % 20 == 0:
                logger.info(f"进度: {total_done}/{stats['total']}, "
                           f"基础费率: {stats['new_base']}, "
                           f"优惠: {stats['new_discount']}, "
                           f"跳过: {stats['skipped']}")

            time.sleep(0.1)  # 避免请求过快

        session.close()

        # 每批保存一次
        save_fee_rate_db()

    elapsed = time.time() - start_time

    # 3. 最终保存
    save_fee_rate_db()

    # 4. 打印报告
    print("\n" + "=" * 70)
    print("                    费率采集完成")
    print("=" * 70)
    print(f"耗时: {elapsed:.0f} 秒")
    print(f"总产品数: {stats['total']}")
    print(f"新增基础费率: {stats['new_base']}")
    print(f"新增费率优惠: {stats['new_discount']}")
    print(f"跳过(已最新): {stats['skipped']}")
    print(f"失败: {stats['failed']}")
    print(f"PDF下载: {stats['pdf_downloaded']}, 缓存命中: {stats['pdf_cached']}")

    # 数据库统计
    summary = get_fee_rate_summary()
    print(f"\n费率数据库统计:")
    print(f"  总产品: {summary['total']}")
    print(f"  有基础费率: {summary['has_base']}")
    print(f"  有费率优惠: {summary['has_discount']}")

    # PDF存储统计
    pdf_dir = crawler.PDF_STORAGE_DIR
    if os.path.exists(pdf_dir):
        pdf_count = sum(len(files) for _, _, files in os.walk(pdf_dir))
        pdf_size = sum(os.path.getsize(os.path.join(root, f))
                      for root, _, files in os.walk(pdf_dir) for f in files)
        print(f"\nPDF本地存储:")
        print(f"  目录: {pdf_dir}")
        print(f"  文件数: {pdf_count}")
        print(f"  总大小: {pdf_size / 1024 / 1024:.1f} MB")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='民生银行费率独立采集程序')
    parser.add_argument('--full', '-f', action='store_true',
                        help='全量采集（忽略已有数据，重新下载）')
    parser.add_argument('--limit', '-l', type=int, default=0,
                        help='只处理前N个产品（测试用）')

    args = parser.parse_args()

    crawl_fees(full_mode=args.full, limit=args.limit)


if __name__ == "__main__":
    main()
