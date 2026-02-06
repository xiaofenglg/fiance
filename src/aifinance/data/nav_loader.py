# -*- coding: utf-8 -*-
"""
NAV 数据加载模块 — V2 周频版本

支持:
- 从 SQLite 数据库加载
- 从 Parquet 文件加载
- 数据预处理和合并
- 周频重采样 (W-WED) — Phase 1 改进

变更日志:
- V2: 添加 resample_to_weekly() 强制周频对齐
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 周频重采样函数 — Phase 1 关键组件
# ═════════════════════════════════════════════════════════════════════════════

def resample_to_weekly(
    nav_matrix: np.ndarray,
    dates: List[str],
    anchor_day: str = "WED",
    method: str = "last",
) -> Tuple[np.ndarray, List[str]]:
    """将日频 NAV 数据重采样为周频 (W-WED)

    强制将所有输入数据重采样为周频，使用 .last() 获取收盘价。
    这是 Phase 1 架构的核心组件，确保所有下游计算基于统一频率。

    Args:
        nav_matrix: [n_products, n_dates] 日频 NAV 矩阵
        dates: 日期列表 (YYYY-MM-DD 格式)
        anchor_day: 锚定日 ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
        method: 重采样方法 ("last", "mean", "first")

    Returns:
        (weekly_nav, weekly_dates)
        - weekly_nav: [n_products, n_weeks] 周频 NAV 矩阵
        - weekly_dates: 周日期列表 (每周锚定日的日期)

    Example:
        >>> nav, dates = resample_to_weekly(daily_nav, daily_dates, anchor_day="WED")
        >>> print(f"日频 {len(daily_dates)} 天 -> 周频 {len(dates)} 周")
    """
    if nav_matrix.ndim != 2:
        raise ValueError(f"nav_matrix 必须是 2D 数组, 当前: {nav_matrix.ndim}D")

    n_products, n_dates = nav_matrix.shape

    if len(dates) != n_dates:
        raise ValueError(f"dates 长度 ({len(dates)}) 与 nav_matrix 列数 ({n_dates}) 不匹配")

    # 转换日期为 DatetimeIndex
    try:
        date_index = pd.DatetimeIndex(dates)
    except Exception as e:
        logger.error(f"日期转换失败: {e}")
        raise ValueError(f"无法解析日期格式: {dates[:3]}...")

    # 创建 DataFrame 用于重采样
    # 转置以便按日期重采样
    df = pd.DataFrame(nav_matrix.T, index=date_index)

    # Pandas 周频代码映射
    day_map = {
        "MON": "W-MON",
        "TUE": "W-TUE",
        "WED": "W-WED",
        "THU": "W-THU",
        "FRI": "W-FRI",
        "SAT": "W-SAT",
        "SUN": "W-SUN",
    }
    freq = day_map.get(anchor_day.upper(), "W-WED")

    # 重采样
    if method == "last":
        df_weekly = df.resample(freq).last()
    elif method == "mean":
        df_weekly = df.resample(freq).mean()
    elif method == "first":
        df_weekly = df.resample(freq).first()
    else:
        raise ValueError(f"不支持的重采样方法: {method}")

    # 移除全 NaN 行
    df_weekly = df_weekly.dropna(how="all")

    # 转换回 numpy 格式
    weekly_nav = df_weekly.values.T  # 转回 [n_products, n_weeks]
    weekly_dates = [d.strftime("%Y-%m-%d") for d in df_weekly.index]

    logger.info(
        f"周频重采样完成: {n_dates} 天 -> {len(weekly_dates)} 周 "
        f"(锚定日: {anchor_day}, 方法: {method})"
    )

    return weekly_nav, weekly_dates


def compute_weekly_returns(
    nav_matrix: np.ndarray,
    masks: np.ndarray,
    dates: List[str],
    anchor_day: str = "WED",
    annualize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """计算周频收益率矩阵

    先重采样到周频，再计算收益率。这是 Phase 1 的标准数据处理流程。

    Args:
        nav_matrix: [n_products, n_dates] 日频 NAV 矩阵
        masks: [n_products, n_dates] 有效数据掩码
        dates: 日期列表
        anchor_day: 锚定日 (默认 "WED")
        annualize: 是否年化 (默认 True)

    Returns:
        (weekly_returns, weekly_masks, weekly_dates)
        - weekly_returns: [n_products, n_weeks] 周频收益率 (年化%)
        - weekly_masks: [n_products, n_weeks] 有效数据掩码
        - weekly_dates: 周日期列表
    """
    # 先重采样 NAV 到周频
    weekly_nav, weekly_dates = resample_to_weekly(
        nav_matrix, dates, anchor_day=anchor_day, method="last"
    )

    # 同样重采样 masks
    weekly_masks, _ = resample_to_weekly(
        masks, dates, anchor_day=anchor_day, method="last"
    )
    # 将 mask 二值化
    weekly_masks = (weekly_masks > 0.5).astype(np.float32)

    n_products, n_weeks = weekly_nav.shape

    # 计算周收益率
    weekly_returns = np.zeros_like(weekly_nav, dtype=np.float32)

    # 有效数据对
    valid_pairs = (weekly_masks[:, 1:] > 0) & (weekly_masks[:, :-1] > 0)
    nav_prev = weekly_nav[:, :-1]

    # 避免除以零
    safe_nav_prev = np.where(nav_prev > 0, nav_prev, np.nan)
    ret = (weekly_nav[:, 1:] / safe_nav_prev) - 1

    # 年化: 周收益率 * 52
    if annualize:
        ret = ret * 52 * 100  # 转为年化百分比

    weekly_returns[:, 1:] = np.where(valid_pairs, ret, 0)

    # 清理 NaN
    weekly_returns = np.nan_to_num(weekly_returns, nan=0.0, posinf=100.0, neginf=-100.0)

    logger.info(
        f"周频收益率计算完成: {n_products} 产品 x {n_weeks} 周"
    )

    return weekly_returns, weekly_masks, weekly_dates


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

        支持两种格式:
        1. 单银行文件: {bank_name}.parquet
        2. 统一数据库: 净值数据库.parquet (包含所有银行)

        Args:
            bank_name: 银行名称

        Returns:
            NAV 数据 DataFrame (宽格式)
        """
        if not self.parquet_dir:
            raise ValueError("未设置 parquet_dir")

        parquet_dir = Path(self.parquet_dir)

        # 尝试方式1: 单银行文件
        parquet_path = parquet_dir / f"{bank_name}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            logger.info(f"从 {parquet_path} 加载 {len(df)} 行数据")
            return df

        # 尝试方式2: 统一数据库文件
        unified_path = parquet_dir / "净值数据库.parquet"
        if not unified_path.exists():
            logger.warning(f"Parquet 文件不存在: {parquet_path} 或 {unified_path}")
            return pd.DataFrame()

        # 加载统一数据库并筛选
        df = pd.read_parquet(unified_path)
        logger.info(f"从 {unified_path} 加载统一数据库")

        # 筛选指定银行
        bank_col = "银行" if "银行" in df.columns else "bank_name"
        if bank_col not in df.columns:
            logger.error(f"数据中没有银行列: {df.columns.tolist()}")
            return pd.DataFrame()

        df = df[df[bank_col] == bank_name].copy()
        if df.empty:
            logger.warning(f"没有找到 {bank_name} 的数据")
            return pd.DataFrame()

        logger.info(f"筛选 {bank_name} 数据: {len(df)} 行")

        # 转换为宽格式 (pivot)
        # 列名映射
        code_col = "产品代码" if "产品代码" in df.columns else "product_code"
        name_col = "产品名称" if "产品名称" in df.columns else "product_name"
        date_col = "日期" if "日期" in df.columns else "date"
        nav_col = "净值" if "净值" in df.columns else "nav"

        # Pivot 转换
        pivot_df = df.pivot_table(
            index=[code_col, name_col] if name_col in df.columns else [code_col],
            columns=date_col,
            values=nav_col,
            aggfunc="first",
        )
        pivot_df = pivot_df.reset_index()

        # 将 Timestamp 列名转换为字符串格式 "YYYY-MM-DD"
        new_columns = []
        for col in pivot_df.columns:
            if isinstance(col, pd.Timestamp):
                new_columns.append(col.strftime("%Y-%m-%d"))
            else:
                new_columns.append(col)
        pivot_df.columns = new_columns

        # 重命名列以统一
        if code_col != "product_code" and code_col in pivot_df.columns:
            pivot_df = pivot_df.rename(columns={code_col: "product_code"})
        if name_col != "product_name" and name_col in pivot_df.columns:
            pivot_df = pivot_df.rename(columns={name_col: "product_name"})

        logger.info(f"转换为宽格式: {len(pivot_df)} 产品 x {len(pivot_df.columns)-2} 天")
        return pivot_df

    def load_from_db(
        self,
        bank_name: Union[str, List[str]],  # 支持单个或多个银行
        lookback_days: Optional[int] = None,
        max_products: Optional[int] = None,
        min_records: int = 60,
    ) -> pd.DataFrame:
        """从 SQLite 数据库加载数据

        Args:
            bank_name: 银行名称 (单个银行或银行列表，如 ["华夏银行", "民生银行"])
            lookback_days: 限制加载最近N天的数据 (None = 全部)
            max_products: 限制最多加载N个产品 (None = 全部)
            min_records: 最少NAV记录数 (默认60, 宁银等低频数据可降低)

        Returns:
            NAV 数据 DataFrame (宽格式)
        """
        if not self.db_path:
            raise ValueError("未设置 db_path")

        import sqlite3
        from datetime import datetime, timedelta

        # 支持单个或多个银行
        if isinstance(bank_name, str):
            bank_names = [bank_name]
        else:
            bank_names = bank_name

        conn = sqlite3.connect(self.db_path)
        try:
            # 检查表结构，支持两种 schema
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]

            # 计算日期过滤条件
            date_filter = ""
            if lookback_days:
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                date_filter = f" AND h.date >= '{start_date}'" if "nav_history" in tables else f" AND nav_date >= '{start_date}'"
                logger.info(f"日期过滤: >= {start_date} (最近 {lookback_days} 天)")

            # 构建 IN 子句
            placeholders = ','.join('?' * len(bank_names))

            if "nav_data" in tables:
                # 旧 schema: nav_data 表
                query = f"""
                    SELECT product_code, product_name, nav_date, nav_value
                    FROM nav_data
                    WHERE bank_name IN ({placeholders}){date_filter.replace('h.date', 'nav_date')}
                    ORDER BY product_code, nav_date
                """
                df = pd.read_sql_query(query, conn, params=bank_names)
            elif "products" in tables and "nav_history" in tables:
                # 新 schema: products + nav_history 表
                # 优化: 使用 CTE (Common Table Expression) 高效查询

                # 计算日期过滤
                start_date_clause = ""
                if lookback_days:
                    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                    start_date_clause = f"AND h.date >= '{start_date}'"

                # 限制产品数量 (默认 200)
                product_limit = max_products or 200

                # 使用 CTE 一次性高效查询
                # 改进: 选择有NAV波动的产品 (排除货币基金等NAV=1.0的产品)
                query = f"""
                    WITH BankProducts AS (
                        SELECT product_code, product_name
                        FROM products
                        WHERE bank_name IN ({placeholders})
                    ),
                    ProductStats AS (
                        SELECT h.product_code,
                               COUNT(*) as cnt,
                               MIN(h.nav) as min_nav,
                               MAX(h.nav) as max_nav
                        FROM nav_history h
                        WHERE h.product_code IN (SELECT product_code FROM BankProducts)
                        {start_date_clause}
                        GROUP BY h.product_code
                    ),
                    TopProducts AS (
                        SELECT product_code, cnt
                        FROM ProductStats
                        WHERE ((max_nav - min_nav) > 0.001  -- 只选有NAV波动的产品
                           OR max_nav > 1.01 OR min_nav < 0.99)
                           AND cnt >= {min_records}  -- 最少记录数(可配置, 低频银行可降低)
                        ORDER BY cnt DESC  -- 按数据量排序，避免前瞻偏差
                        LIMIT {product_limit}
                    )
                    SELECT bp.product_code, bp.product_name, h.date as nav_date, h.nav as nav_value
                    FROM nav_history h
                    JOIN BankProducts bp ON h.product_code = bp.product_code
                    WHERE h.product_code IN (SELECT product_code FROM TopProducts)
                    {start_date_clause}
                    ORDER BY h.product_code, h.date
                """
                bank_list_str = ', '.join(bank_names)
                logger.info(f"加载 {bank_list_str} 数据 (最多 {product_limit} 产品, 最近 {lookback_days or '全部'} 天)")
                df = pd.read_sql_query(query, conn, params=bank_names)
            else:
                logger.error(f"数据库表结构不支持: {tables}")
                return pd.DataFrame()

            if df.empty:
                logger.warning(f"数据库中没有 {bank_list_str} 的数据")
                return pd.DataFrame()

            # 使用 float32 减少内存
            df['nav_value'] = df['nav_value'].astype(np.float32)

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

    def load(
        self,
        bank_name: str,
        source: str = "auto",
        lookback_days: Optional[int] = None,
        max_products: Optional[int] = None,
        min_records: int = 60,
    ) -> pd.DataFrame:
        """加载 NAV 数据

        Args:
            bank_name: 银行名称
            source: 数据源 ("parquet", "db", "auto")
            lookback_days: 限制加载最近N天的数据 (None = 全部)
            max_products: 限制最多加载N个产品 (None = 全部)
            min_records: 最少NAV记录数 (默认60)

        Returns:
            NAV 数据 DataFrame
        """
        if source == "parquet" or (source == "auto" and self.parquet_dir):
            return self.load_from_parquet(bank_name)
        elif source == "db" or (source == "auto" and self.db_path):
            return self.load_from_db(bank_name, lookback_days=lookback_days, max_products=max_products, min_records=min_records)
        else:
            raise ValueError("未指定有效的数据源")

    def prepare_matrices(
        self,
        df: pd.DataFrame,
        min_valid_ratio: float = 0.3,
        frequency: str = "daily",
        anchor_day: str = "WED",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """准备收益率矩阵和掩码矩阵

        Args:
            df: NAV 数据 DataFrame
            min_valid_ratio: 最小有效数据比例阈值
            frequency: 输出频率 ("daily" 或 "weekly")
            anchor_day: 周频锚定日 (默认 "WED")

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

        # 计算年化收益率 (向量化)
        returns = np.zeros_like(nav_matrix, dtype=np.float32)
        
        # 使用切片计算日收益率: (NAV[t] / NAV[t-1]) - 1
        # 只有当今天和昨天都有有效数据时才计算
        valid_pairs = (masks[:, 1:] > 0) & (masks[:, :-1] > 0)
        nav_prev = nav_matrix[:, :-1]
        
        # 避免除以零
        safe_nav_prev = np.where(nav_prev > 0, nav_prev, np.nan)
        daily_ret = (nav_matrix[:, 1:] / safe_nav_prev) - 1
        
        # 填充到 returns 矩阵 (从索引1开始)
        returns[:, 1:] = np.where(valid_pairs, daily_ret * 365 * 100, 0)

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

        # 清理 NaN: 保留 NaN 而不是填充为0，避免产生虚假的0→1.0收益率
        # 下游代码 (run_pipeline.py) 会正确处理 NaN
        # 仅清理 returns 中的 inf 值
        returns = np.nan_to_num(returns, nan=0.0, posinf=100.0, neginf=-100.0)

        # ══════════════════════════════════════════════════════════════════════
        # Phase 1: 可选周频重采样
        # ══════════════════════════════════════════════════════════════════════
        if frequency.lower() == "weekly":
            logger.info(f"启用周频模式 (锚定日: {anchor_day})")
            returns, masks, date_cols = compute_weekly_returns(
                nav_matrix, masks, date_cols, anchor_day=anchor_day, annualize=True
            )
            nav_matrix, date_cols = resample_to_weekly(
                nav_matrix, date_cols, anchor_day=anchor_day, method="last"
            )
            logger.info(
                f"周频数据准备完成: {nav_matrix.shape[0]} 产品 x {nav_matrix.shape[1]} 周"
            )
        else:
            logger.info(
                f"日频数据准备完成: {nav_matrix.shape[0]} 产品 x {nav_matrix.shape[1]} 天"
            )

        return nav_matrix, returns, masks, date_cols, product_codes

    def get_first_valid_idx(self, masks: np.ndarray) -> np.ndarray:
        """获取每个产品的首个有效数据索引 (向量化)

        Args:
            masks: [n_products, n_dates] 掩码矩阵

        Returns:
            [n_products] 首个有效索引
        """
        n_products, n_dates = masks.shape
        # argmax 返回第一个 True 的位置；对于全 False 行返回 0
        has_valid = masks.any(axis=1)
        first_valid = np.where(has_valid, masks.argmax(axis=1), n_dates)
        return first_valid.astype(np.int32)


def load_nav_data(
    bank_name: Union[str, List[str]],  # 支持单个或多个银行
    db_path: Optional[str] = None,
    parquet_dir: Optional[str] = None,
    min_valid_ratio: float = 0.3,
    frequency: str = "daily",
    anchor_day: str = "WED",
    lookback_days: Optional[int] = None,
    max_products: Optional[int] = None,
    min_records: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """便捷函数: 加载并准备 NAV 数据

    Args:
        bank_name: 银行名称 (单个银行或银行列表，如 ["华夏银行", "民生银行"])
        db_path: SQLite 数据库路径
        parquet_dir: Parquet 文件目录
        min_valid_ratio: 最小有效数据比例
        frequency: 输出频率 ("daily" 或 "weekly")
        anchor_day: 周频锚定日 (默认 "WED")
        lookback_days: 限制加载最近N天的数据 (None = 全部, 推荐 365-730)
        max_products: 限制最多加载N个产品 (None = 全部)
        min_records: 最少NAV记录数 (默认60, 宁银等低频银行可降低)

    Returns:
        (nav_matrix, returns, masks, dates, product_codes)
    """
    loader = NavLoader(db_path=db_path, parquet_dir=parquet_dir)
    df = loader.load(bank_name, lookback_days=lookback_days, max_products=max_products, min_records=min_records)

    if df.empty:
        raise ValueError(f"无法加载 {bank_name} 的数据")

    return loader.prepare_matrices(
        df,
        min_valid_ratio=min_valid_ratio,
        frequency=frequency,
        anchor_day=anchor_day,
    )


def load_nav_data_weekly(
    bank_name: str,
    db_path: Optional[str] = None,
    parquet_dir: Optional[str] = None,
    min_valid_ratio: float = 0.3,
    anchor_day: str = "WED",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """便捷函数: 加载周频 NAV 数据 — Phase 1 推荐接口

    这是 V11 Phase 1 架构的推荐数据加载接口。
    强制使用周频 (W-WED) 重采样，确保与 GLM 去平滑和 HRP 优化的兼容性。

    Args:
        bank_name: 银行名称
        db_path: SQLite 数据库路径
        parquet_dir: Parquet 文件目录
        min_valid_ratio: 最小有效数据比例
        anchor_day: 周频锚定日 (默认 "WED")

    Returns:
        (nav_matrix, returns, masks, dates, product_codes)
        - 所有数据均为周频
        - returns 为年化百分比
    """
    return load_nav_data(
        bank_name=bank_name,
        db_path=db_path,
        parquet_dir=parquet_dir,
        min_valid_ratio=min_valid_ratio,
        frequency="weekly",
        anchor_day=anchor_day,
    )
