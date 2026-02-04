# -*- coding: utf-8 -*-
"""
NAV 数据加载模块

支持:
- 从 SQLite 数据库加载
- 从 Parquet 文件加载
- 数据预处理和合并
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def is_date_col(col_name: str) -> bool:
    """判断列名是否为日期格式 (YYYY-MM-DD)"""
    if not isinstance(col_name, str):
        return False
    return len(col_name) == 10 and col_name[4] == "-" and col_name[7] == "-"


class NavLoader:
    """NAV 数据加载器"""

    def __init__(self, db_path: Optional[str] = None, parquet_dir: Optional[str] = None):
        """
        Args:
            db_path: SQLite 数据库路径
            parquet_dir: Parquet 文件目录
        """
        self.db_path = db_path
        self.parquet_dir = parquet_dir

    def load_from_parquet(self, bank_name: str) -> pd.DataFrame:
        """从 Parquet 文件加载数据

        Args:
            bank_name: 银行名称

        Returns:
            NAV 数据 DataFrame
        """
        if not self.parquet_dir:
            raise ValueError("未设置 parquet_dir")

        parquet_path = Path(self.parquet_dir) / f"{bank_name}.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet 文件不存在: {parquet_path}")
            return pd.DataFrame()

        df = pd.read_parquet(parquet_path)
        logger.info(f"从 {parquet_path} 加载 {len(df)} 行数据")
        return df

    def load_from_db(self, bank_name: str) -> pd.DataFrame:
        """从 SQLite 数据库加载数据

        Args:
            bank_name: 银行名称

        Returns:
            NAV 数据 DataFrame (宽格式)
        """
        if not self.db_path:
            raise ValueError("未设置 db_path")

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        try:
            # 先获取所有日期
            dates_df = pd.read_sql_query(
                "SELECT DISTINCT nav_date FROM nav_data WHERE bank_name = ? ORDER BY nav_date",
                conn,
                params=(bank_name,),
            )

            if dates_df.empty:
                logger.warning(f"数据库中没有 {bank_name} 的数据")
                return pd.DataFrame()

            dates = dates_df["nav_date"].tolist()

            # 获取产品信息和 NAV 数据
            df = pd.read_sql_query(
                """
                SELECT product_code, product_name, nav_date, nav_value
                FROM nav_data
                WHERE bank_name = ?
                ORDER BY product_code, nav_date
                """,
                conn,
                params=(bank_name,),
            )

            if df.empty:
                return pd.DataFrame()

            # 转换为宽格式
            pivot_df = df.pivot_table(
                index=["product_code", "product_name"],
                columns="nav_date",
                values="nav_value",
                aggfunc="first",
            )
            pivot_df = pivot_df.reset_index()

            logger.info(f"从数据库加载 {len(pivot_df)} 个产品的 NAV 数据")
            return pivot_df

        finally:
            conn.close()

    def load(self, bank_name: str, source: str = "auto") -> pd.DataFrame:
        """加载 NAV 数据

        Args:
            bank_name: 银行名称
            source: 数据源 ("parquet", "db", "auto")

        Returns:
            NAV 数据 DataFrame
        """
        if source == "parquet" or (source == "auto" and self.parquet_dir):
            return self.load_from_parquet(bank_name)
        elif source == "db" or (source == "auto" and self.db_path):
            return self.load_from_db(bank_name)
        else:
            raise ValueError("未指定有效的数据源")

    def prepare_matrices(
        self, df: pd.DataFrame, min_valid_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """准备收益率矩阵和掩码矩阵

        Args:
            df: NAV 数据 DataFrame
            min_valid_ratio: 最小有效数据比例阈值

        Returns:
            (nav_matrix, returns, masks, dates, product_codes)
            - nav_matrix: [n_products, n_dates] NAV 矩阵
            - returns: [n_products, n_dates] 年化收益率矩阵 (%)
            - masks: [n_products, n_dates] 有效数据掩码
            - dates: 日期列表
            - product_codes: 产品代码列表
        """
        # 识别日期列
        date_cols = sorted([c for c in df.columns if is_date_col(c)])
        if not date_cols:
            raise ValueError("数据中没有日期列")

        # 提取产品代码
        if "产品代码" in df.columns:
            product_codes = df["产品代码"].fillna("Unknown").astype(str).tolist()
        elif "product_code" in df.columns:
            product_codes = df["product_code"].fillna("Unknown").astype(str).tolist()
        else:
            product_codes = [f"P{i}" for i in range(len(df))]

        # 提取 NAV 矩阵
        nav_matrix = df[date_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        n_products, n_dates = nav_matrix.shape

        # 创建掩码
        masks = (~np.isnan(nav_matrix)).astype(np.float32)

        # 计算年化收益率
        returns = np.zeros_like(nav_matrix, dtype=np.float32)

        for p in range(n_products):
            for d in range(1, n_dates):
                if masks[p, d] > 0 and masks[p, d - 1] > 0:
                    nav_today = nav_matrix[p, d]
                    nav_yesterday = nav_matrix[p, d - 1]
                    if nav_yesterday > 0:
                        daily_return = (nav_today - nav_yesterday) / nav_yesterday
                        returns[p, d] = daily_return * 365 * 100  # 年化百分比

        # 过滤低质量产品
        valid_ratios = masks.sum(axis=1) / n_dates
        valid_products = valid_ratios >= min_valid_ratio

        if valid_products.sum() < n_products:
            logger.info(
                f"过滤低质量产品: {n_products} -> {valid_products.sum()} "
                f"(有效率阈值: {min_valid_ratio:.0%})"
            )

        nav_matrix = nav_matrix[valid_products]
        returns = returns[valid_products]
        masks = masks[valid_products]
        product_codes = [c for c, v in zip(product_codes, valid_products) if v]

        # 清理 NaN
        nav_matrix = np.nan_to_num(nav_matrix, nan=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=100.0, neginf=-100.0)

        logger.info(
            f"数据矩阵准备完成: {nav_matrix.shape[0]} 产品 x {nav_matrix.shape[1]} 天"
        )

        return nav_matrix, returns, masks, date_cols, product_codes

    def get_first_valid_idx(self, masks: np.ndarray) -> np.ndarray:
        """获取每个产品的首个有效数据索引

        Args:
            masks: [n_products, n_dates] 掩码矩阵

        Returns:
            [n_products] 首个有效索引
        """
        n_products, n_dates = masks.shape
        first_valid = np.full(n_products, n_dates, dtype=np.int32)

        for p in range(n_products):
            for d in range(n_dates):
                if masks[p, d] > 0:
                    first_valid[p] = d
                    break

        return first_valid


def load_nav_data(
    bank_name: str,
    db_path: Optional[str] = None,
    parquet_dir: Optional[str] = None,
    min_valid_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """便捷函数: 加载并准备 NAV 数据

    Args:
        bank_name: 银行名称
        db_path: SQLite 数据库路径
        parquet_dir: Parquet 文件目录
        min_valid_ratio: 最小有效数据比例

    Returns:
        (nav_matrix, returns, masks, dates, product_codes)
    """
    loader = NavLoader(db_path=db_path, parquet_dir=parquet_dir)
    df = loader.load(bank_name)

    if df.empty:
        raise ValueError(f"无法加载 {bank_name} 的数据")

    return loader.prepare_matrices(df, min_valid_ratio=min_valid_ratio)
