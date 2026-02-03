# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
"""
import logging
import pandas as pd
import sys
import os

from ..database import Database

logger = logging.getLogger(__name__)

def is_date_col(col_name):
    if not isinstance(col_name, str):
        return False
    return len(col_name) == 10 and col_name[4] == '-' and col_name[7] == '-'

def _merge_unidentified_rows(df: pd.DataFrame, date_cols: List[str], bank_name: str) -> pd.DataFrame:
    """
    合并有代码行和无代码行的数据。
    (从 bank_product_strategy_v6.py 迁移而来)
    """
    df_with_code = df[df['产品代码'].notna()].copy()
    df_no_code = df[df['产品代码'].isna()].copy()

    if df_no_code.empty:
        return df_with_code

    nav_with_code = df_with_code[date_cols].apply(pd.to_numeric, errors='coerce')
    nav_no_code = df_no_code[date_cols].apply(pd.to_numeric, errors='coerce')

    # 寻找有重叠数据的日期
    overlapping_dates = [
        d for d in date_cols
        if nav_with_code[d].notna().sum() > 0 and nav_no_code[d].notna().sum() > 0
    ]

    if len(overlapping_dates) < 3:
        return df_with_code

    # 使用最近的20个重叠日期作为匹配基准
    match_dates = overlapping_dates[-20:]
    
    # 为有代码的行创建基于净值序列的哈希索引
    match_index = {}
    for idx, row in df_with_code.iterrows():
        key_vals = []
        for d in match_dates:
            val = row.get(d)
            if pd.notna(val):
                try:
                    key_vals.append(str(round(float(val), 4)))
                except (ValueError, TypeError):
                    pass
        if len(key_vals) >= 3:
            match_key = '|'.join(key_vals)
            match_index[match_key] = idx

    result_dict = {idx: row.to_dict() for idx, row in df_with_code.iterrows()}
    matched_count = 0

    # 遍历无代码行，尝试匹配
    for idx, row in df_no_code.iterrows():
        key_vals = []
        for d in match_dates:
            val = row.get(d)
            if pd.notna(val):
                try:
                    key_vals.append(str(round(float(val), 4)))
                except (ValueError, TypeError):
                    pass
        
        if len(key_vals) >= 3:
            key = '|'.join(key_vals)
            if key in match_index:
                original_idx = match_index[key]
                # 合并数据：用无代码行的数据填充有代码行的空值
                for d in date_cols:
                    if pd.notna(row.get(d)) and pd.isna(result_dict[original_idx].get(d)):
                        result_dict[original_idx][d] = row[d]
                matched_count += 1
    
    if matched_count > 0:
        logger.info(f"[{bank_name}] 智能合并: 成功匹配并合并了 {matched_count} 行无代码数据。")
        return pd.DataFrame.from_dict(result_dict, orient='index')

    return df_with_code


def load_prepared_data(bank_name: str) -> pd.DataFrame:
    """
    从数据库加载指定银行的数据，并进行预处理。
    这是替代旧 NavDBReader 的新入口。
    """
    logger.info(f"开始为银行 '{bank_name}' 加载和准备数据...")
    
    with Database() as db:
        df = db.get_nav_wide_format(bank_name)

    if df.empty:
        logger.warning(f"银行 '{bank_name}' 没有从数据库加载到任何数据。")
        return pd.DataFrame()

    logger.info(f"从数据库成功加载 {len(df)} 行 '{bank_name}' 的原始数据。")

    # 执行数据合并逻辑
    date_cols = sorted([c for c in df.columns if is_date_col(c)])
    if not date_cols:
        logger.warning(f"银行 '{bank_name}' 的数据中没有找到日期列。")
        return df

    df_prepared = _merge_unidentified_rows(df, date_cols, bank_name)
    
    logger.info(f"数据准备完毕，返回 {len(df_prepared)} 行处理后的数据。")
    
    return df_prepared

if __name__ == '__main__':
    # 测试
    logging.basicConfig(level=logging.INFO)
    test_bank = "中信银行"
    prepared_df = load_prepared_data(test_bank)
    
    print(f"\n为银行 '{test_bank}' 加载并处理完成的数据:")
    print(prepared_df.head())
    print(f"\n数据维度: {prepared_df.shape}")
