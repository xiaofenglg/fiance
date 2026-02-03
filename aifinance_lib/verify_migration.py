import logging
import sys
import os
import pandas as pd
import numpy as np

# 确保能找到根目录和 aifinance_lib 包中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from .database import Database
    from nav_db_excel import NavDBReader
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"无法导入必要的模块: {e}")
    # Fallback for running script directly for testing
    from database import Database
    from nav_db_excel import NavDBReader

def verify_migration(bank_name: str):
    """对指定银行的数据进行验证。"""
    logger.info(f"========== 开始验证银行: {bank_name} ==========")
    
    # 1. 从旧数据源加载数据
    logger.info("--- (1/3) 从旧数据源 (Parquet) 加载...")
    old_reader = NavDBReader()
    if bank_name not in old_reader.sheet_names:
        logger.error(f"在旧数据源中找不到银行 '{bank_name}'。")
        return
    df_old_wide = old_reader.read_sheet(bank_name)
    logger.info("旧数据源加载完毕。")

    # 2. 从新数据源加载数据
    logger.info("--- (2/3) 从新数据源 (SQLite) 加载...")
    with Database() as db:
        df_new_wide = db.get_nav_wide_format(bank_name)
    logger.info("新数据源加载完毕。")

    # 3. 开始比较
    logger.info("--- (3/3) 开始比较关键指标...")
    errors = []

    # 比较产品数量 (行数)
    old_products = len(df_old_wide)
    new_products = len(df_new_wide)
    if old_products == new_products:
        logger.info(f"[成功] 产品数量一致: {old_products}")
    else:
        msg = f"[失败] 产品数量不一致: 旧={old_products}, 新={new_products}"
        logger.error(msg)
        errors.append(msg)

    # 比较日期数量 (列数)
    old_date_cols = [c for c in df_old_wide.columns if str(c).startswith('20')]
    new_date_cols = [c for c in df_new_wide.columns if str(c).startswith('20')]
    if len(old_date_cols) == len(new_date_cols):
        logger.info(f"[成功] 日期列数量一致: {len(old_date_cols)}")
    else:
        msg = f"[失败] 日期列数量不一致: 旧={len(old_date_cols)}, 新={len(new_date_cols)}"
        logger.error(msg)
        errors.append(msg)

    # 比较总数据点数量 (非空净值单元格) - 修正验证逻辑
    old_nav_points = (df_old_wide[old_date_cols].apply(
        lambda s: pd.to_numeric(s, errors='coerce').notna()
    )).sum().sum()
    new_nav_points = df_new_wide[new_date_cols].apply(
        lambda s: pd.to_numeric(s, errors='coerce').notna()
    ).sum().sum()
    
    if old_nav_points == new_nav_points:
        logger.info(f"[成功] 总净值数据点数量一致: {old_nav_points}")
    else:
        msg = f"[失败] 总净值数据点数量不一致: 旧={old_nav_points}, 新={new_nav_points}"
        logger.error(msg)
        errors.append(msg)

    # 抽样比较某一天的数据总和 - 修正兼容性问题
    if old_date_cols and new_date_cols:
        sample_date = old_date_cols[-1] # 选择最近一天
        old_sum = pd.to_numeric(df_old_wide[sample_date], errors='coerce').sum()
        new_sum = pd.to_numeric(df_new_wide[sample_date], errors='coerce').sum()
        
        if np.isclose(old_sum, new_sum):
            logger.info(f"[成功] 抽样日期 '{sample_date}' 的净值总和一致: {old_sum:.4f}")
        else:
            msg = f"[失败] 抽样日期 '{sample_date}' 的净值总和不一致: 旧={old_sum:.4f}, 新={new_sum:.4f}"
            logger.error(msg)
            errors.append(msg)

    logger.info("========== 验证结束 ==========")
    
    return errors

if __name__ == "__main__":
    # 我们选择一个数据量较大的银行进行验证
    target_bank = "中信银行"
    
    all_errors = verify_migration(target_bank)
    
    if all_errors:
        print("\n*** 验证失败，发现以下问题: ***")
        for err in all_errors:
            print(f"- {err}")
    else:
        print("\n*** 所有验证项均成功通过！数据迁移准确无误。 ***")
