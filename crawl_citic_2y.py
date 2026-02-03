# -*- coding: utf-8 -*-
"""
中信银行2年净值数据抓取 — 使用 startDate/endDate 方式

用法: python crawl_citic_2y.py
"""

import sys
import os
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s — %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('crawl_citic_2y')

from bank_crawler import CITICCrawler
from nav_db_excel import NAVDatabaseExcel

def main():
    max_months = 24
    logger.info(f"=== 中信银行 {max_months} 个月净值抓取 (startDate/endDate 模式) ===")

    crawler = CITICCrawler()

    # 1. 获取产品列表
    logger.info("获取产品列表（Selenium）...")
    products = crawler.get_product_list()
    if not products:
        logger.error("未获取到产品列表")
        return
    logger.info(f"共 {len(products)} 个产品")

    # 2. 获取每个产品的2年净值
    session = crawler._get_session()
    all_products_for_db = []
    total = len(products)
    t0 = time.time()

    for i, product in enumerate(products):
        product_code = product.get('product_code', '')
        product_name = product.get('product_name', '')

        nav_history = crawler.get_nav_history(
            product_code, session,
            full_history=False,
            max_months=max_months,
        )

        if nav_history:
            all_products_for_db.append({
                'product_code': product_code,
                'product_name': product_name,
                'nav_history': nav_history,
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%), "
                f"有效: {len(all_products_for_db)}, "
                f"平均: {len(nav_history) if nav_history else 0} 条/产品, "
                f"ETA: {eta/60:.1f}分钟"
            )

    logger.info(f"抓取完成: {len(all_products_for_db)}/{total} 个产品有净值数据")

    # 3. 导入净值数据库
    if all_products_for_db:
        logger.info("导入净值数据库...")
        db = NAVDatabaseExcel()

        # 显示导入前状态
        stats_before = db.get_stats().get('中信银行', {})
        logger.info(f"导入前: {stats_before.get('products', 0)} 产品, "
                    f"{stats_before.get('dates', 0)} 日期, "
                    f"范围 {stats_before.get('earliest_date')} ~ {stats_before.get('latest_date')}")

        stats = db.update_nav('中信', all_products_for_db)
        db.save()

        # 显示导入后状态
        stats_after = db.get_stats().get('中信银行', {})
        logger.info(f"导入后: {stats_after.get('products', 0)} 产品, "
                    f"{stats_after.get('dates', 0)} 日期, "
                    f"范围 {stats_after.get('earliest_date')} ~ {stats_after.get('latest_date')}")
        logger.info(f"新增 {stats.get('new_products', 0)} 产品, "
                    f"新增 {len(stats.get('new_dates', []))} 日期, "
                    f"更新 {stats.get('updated_cells', 0)} 单元格")

    logger.info("=== 完成 ===")


if __name__ == '__main__':
    main()
