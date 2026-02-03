# -*- coding: utf-8 -*-
import logging
import pandas as pd
from typing import Dict, List
import sys
import os

# 确保项目根目录在 Python 路径中，以便能找到其他模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 在导入我们自己的模块之前配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from .database import Database, initialize_database
    from nav_db_excel import NavDBReader, FEE_COLUMNS
    from redemption_fee_db import load_fee_db, get_fee_info, format_redemption_fee, format_rate
except (ImportError, ModuleNotFoundError):
    # Fallback for running script directly
    from database import Database, initialize_database
    from nav_db_excel import NavDBReader, FEE_COLUMNS
    from redemption_fee_db import load_fee_db, get_fee_info, format_redemption_fee, format_rate



def migrate_bank_data(db: Database, reader: NavDBReader, bank_name: str):
    """迁移单个银行的数据。"""
    logger.info(f"====== 开始迁移银行: {bank_name} ======")
    
    df_wide = reader.read_sheet(bank_name)
    if df_wide.empty:
        logger.warning(f"银行 '{bank_name}' 的数据为空，跳过。")
        return

    products_data: List[Dict] = []
    
    meta_cols = {'产品代码', '产品名称'} | set(FEE_COLUMNS)
    date_cols = [col for col in df_wide.columns if col not in meta_cols]

    df_wide.columns = [c.replace('.', '-') for c in df_wide.columns]
    date_cols = [c.replace('.', '-') for c in date_cols]

    for record in df_wide.to_dict('records'):
        product_code = record.get('产品代码')
        if not product_code or pd.isna(product_code):
            continue

        nav_history = []
        for date_col in date_cols:
            nav_value = record.get(date_col)
            if nav_value and pd.notna(nav_value) and str(nav_value).strip():
                nav_history.append({'date': date_col, 'unit_nav': nav_value})
        
        fee_info = get_fee_info(bank_name, product_code) or {}

        product_dict = {
            'product_code': product_code,
            'product_name': record.get('产品名称', ''),
            'nav_history': nav_history,
            'subscription_fee': format_rate(fee_info.get('subscription_fee')),
            'purchase_fee': format_rate(fee_info.get('purchase_fee')),
            'redemption_fee': format_redemption_fee(fee_info),
            'sales_service_fee': format_rate(fee_info.get('sales_service_fee')),
            'custody_fee': format_rate(fee_info.get('custody_fee')),
            'management_fee': format_rate(fee_info.get('management_fee')),
        }
        products_data.append(product_dict)
    
    logger.info(f"转换了 {len(products_data)} 个产品的数据结构。")

    if products_data:
        db.update_products_and_nav(bank_name, products_data)
        logger.info(f"成功将 {len(products_data)} 个产品的数据写入新数据库。")

    logger.info(f"====== 完成迁移银行: {bank_name} ======\n")


def main():
    """执行完整的迁移流程。"""
    logger.info("***** 开始数据库迁移 *****")

    logger.info("步骤 1/4: 初始化目标数据库 (SQLite)...")
    initialize_database()
    
    logger.info("步骤 2/4: 加载源数据 (Parquet/Excel)...")
    reader = NavDBReader()
    if not reader.sheet_names:
        logger.error("未能从旧数据库加载任何数据。请检查 '净值数据库.parquet' 或 '.xlsx' 文件是否存在且包含数据。")
        return

    logger.info("步骤 3/4: 加载费率数据库...")
    load_fee_db()

    logger.info("步骤 4/4: 开始逐个银行迁移数据...")
    with Database() as db:
        # 删除旧的数据库文件，以确保从头开始迁移
        if os.path.exists(db.db_file):
            logger.warning(f"发现旧的数据库文件 {db.db_file}，将删除后重新生成。")
            os.remove(db.db_file)
            # 重新创建表
            db._create_tables()

        for bank_name in reader.sheet_names:
            try:
                migrate_bank_data(db, reader, bank_name)
            except Exception as e:
                logger.error(f"迁移银行 '{bank_name}' 时发生严重错误: {e}", exc_info=True)

    logger.info("***** 数据库迁移完成 *****")


if __name__ == "__main__":
    main()
