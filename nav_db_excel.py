# -*- coding: utf-8 -*-
"""
NAV 数据库访问层 — 从 Parquet 文件读取

提供与旧 Excel 接口相同的 API。
"""

import os
import pandas as pd
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 数据库文件路径
_PARQUET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '净值数据库.parquet')


class NAVDatabaseExcel:
    """
    NAV 数据库访问层 (从 Parquet 读取)

    提供与旧接口相同的 API:
    - self.data: Dict[sheet_name, DataFrame]
    - _get_sheet_name(bank_name): 获取 sheet 名
    - _is_date_column(col): 判断是否日期列
    - get_stats(): 获取统计信息
    """

    def __init__(self, parquet_file: str = None):
        self.parquet_file = parquet_file or _PARQUET_FILE
        self._data = None
        self._bank_names = None
        self._stats = None
        self._load_data()

    def _load_data(self):
        """从 Parquet 加载数据为宽表格式"""
        if not os.path.exists(self.parquet_file):
            logger.warning(f"Parquet 文件不存在: {self.parquet_file}")
            self._data = {}
            self._bank_names = []
            self._stats = {}
            return

        try:
            # 读取 Parquet (长表格式: 银行, 产品代码, 产品名称, 日期, 净值)
            df_long = pd.read_parquet(self.parquet_file)

            # 获取所有银行名称
            self._bank_names = sorted(df_long['银行'].unique().tolist())

            self._data = {}
            self._stats = {}

            for bank_name in self._bank_names:
                bank_df = df_long[df_long['银行'] == bank_name]

                # 转换为宽表: 行是(产品代码, 产品名称), 列是日期
                df_wide = bank_df.pivot_table(
                    index=['产品代码', '产品名称'],
                    columns='日期',
                    values='净值',
                    aggfunc='first'
                )

                # 将日期列转换为字符串格式 YYYY-MM-DD
                df_wide.columns = [c.strftime('%Y-%m-%d') if hasattr(c, 'strftime') else str(c)
                                   for c in df_wide.columns]

                # 排序日期列
                date_cols = sorted([c for c in df_wide.columns if self._is_date_column(str(c))])
                if date_cols:
                    df_wide = df_wide[date_cols]

                self._data[bank_name] = df_wide

                # 计算统计
                self._stats[bank_name] = {
                    'products': len(df_wide),
                    'dates': len(date_cols),
                    'earliest_date': date_cols[0] if date_cols else '',
                    'latest_date': date_cols[-1] if date_cols else '',
                }

                logger.debug(f"加载 {bank_name}: {len(df_wide)} 产品, {len(date_cols)} 日期")

            logger.info(f"NAV 数据库加载完成: {len(self._data)} 银行, "
                        f"{sum(len(df) for df in self._data.values())} 产品")

        except Exception as e:
            logger.error(f"加载 NAV Parquet 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._data = {}
            self._bank_names = []
            self._stats = {}

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """兼容旧接口: 返回 {sheet_name: DataFrame} 字典"""
        return self._data

    def _get_sheet_name(self, bank_name: str) -> Optional[str]:
        """根据银行名获取 sheet 名 (兼容旧接口)"""
        if bank_name in self._data:
            return bank_name

        # 尝试模糊匹配
        for name in self._data.keys():
            if bank_name in name or name in bank_name:
                return name

        return None

    def _is_date_column(self, col) -> bool:
        """判断列名是否为日期格式 (兼容旧接口)"""
        if not isinstance(col, str):
            col = str(col)

        # 匹配 YYYY-MM-DD 格式
        return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', col))

    def get_bank_names(self) -> List[str]:
        """获取所有银行名称"""
        return list(self._data.keys())

    def get_stats(self) -> Dict[str, Dict]:
        """
        获取各银行的统计信息

        Returns:
            {
                'bank_name': {
                    'products': int,
                    'dates': int,
                    'earliest_date': str,
                    'latest_date': str,
                },
                ...
            }
        """
        return self._stats

    def get_product_nav(self, bank_name: str, product_code: str) -> pd.Series:
        """获取单个产品的净值序列"""
        sheet = self._get_sheet_name(bank_name)
        if not sheet or sheet not in self._data:
            return pd.Series()

        df = self._data[sheet]

        # 在 MultiIndex 中查找
        for idx in df.index:
            if idx[0] == product_code:
                return df.loc[idx]

        return pd.Series()

    def get_all_products(self) -> List[Dict]:
        """获取所有产品列表"""
        products = []
        for bank_name, df in self._data.items():
            for idx in df.index:
                products.append({
                    'bank': bank_name,
                    'code': idx[0],
                    'name': idx[1] if len(idx) > 1 else '',
                })
        return products


# 单例模式
_instance = None


def get_nav_database() -> NAVDatabaseExcel:
    """获取 NAV 数据库单例"""
    global _instance
    if _instance is None:
        _instance = NAVDatabaseExcel()
    return _instance


# ── SQLite 写入功能 ──

_SQLITE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aifinance.sqlite3')

# 银行简称到全称映射
_BANK_NAME_MAP = {
    '民生': '民生银行',
    '华夏': '华夏银行',
    '中信': '中信银行',
    '浦银': '浦银理财',
    '宁银': '宁银理财',
    '中邮': '中邮理财',
}


def update_nav_database(bank_short: str, products: List[Dict], db_path: str = None) -> Dict:
    """
    将抓取的净值数据保存到 SQLite 数据库

    Args:
        bank_short: 银行简称 (民生/华夏/中信/浦银/宁银/中邮)
        products: 产品列表，每个产品包含:
            - product_code: 产品代码
            - product_name: 产品名称
            - nav_history: 净值历史列表 [{date: str, nav: float}, ...]
        db_path: SQLite 数据库路径，默认使用 aifinance.sqlite3

    Returns:
        {
            'products_added': int,  # 新增产品数
            'nav_rows_added': int,  # 新增净值记录数
            'nav_rows_updated': int,  # 更新净值记录数
        }
    """
    import sqlite3
    from datetime import datetime

    db = db_path or _SQLITE_DB
    bank_name = _BANK_NAME_MAP.get(bank_short, bank_short)

    stats = {
        'products_added': 0,
        'nav_rows_added': 0,
        'nav_rows_updated': 0,
    }

    if not os.path.exists(db):
        logger.warning(f"SQLite 数据库不存在: {db}")
        return stats

    try:
        conn = sqlite3.connect(db, timeout=30)
        cursor = conn.cursor()

        for product in products:
            code = product.get('product_code', '')
            name = product.get('product_name', '')
            nav_history = product.get('nav_history', [])

            if not code:
                continue

            # 插入或更新产品信息
            cursor.execute("""
                INSERT INTO products (product_code, product_name, bank_name)
                VALUES (?, ?, ?)
                ON CONFLICT(product_code) DO UPDATE SET
                    product_name = COALESCE(NULLIF(excluded.product_name, ''), products.product_name)
            """, (code, name, bank_name))

            if cursor.rowcount > 0:
                stats['products_added'] += 1

            # 插入净值历史
            for nav_entry in nav_history:
                nav_date = nav_entry.get('date', '')
                nav_value = nav_entry.get('nav') or nav_entry.get('net_value')

                if not nav_date or nav_value is None:
                    continue

                # 标准化日期格式
                if isinstance(nav_date, str):
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d']:
                        try:
                            nav_date = datetime.strptime(nav_date, fmt).strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue

                try:
                    nav_value = float(nav_value)
                except (ValueError, TypeError):
                    continue

                # 插入或更新净值
                cursor.execute("""
                    INSERT INTO nav_history (product_code, date, nav)
                    VALUES (?, ?, ?)
                    ON CONFLICT(product_code, date) DO UPDATE SET nav = excluded.nav
                """, (code, nav_date, nav_value))

                if cursor.rowcount > 0:
                    stats['nav_rows_added'] += 1

        conn.commit()
        conn.close()

        logger.info(f"[{bank_name}] 数据库更新完成: "
                   f"{stats['products_added']} 新产品, "
                   f"{stats['nav_rows_added']} 净值记录")

    except Exception as e:
        logger.error(f"[{bank_name}] 数据库更新失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    db = NAVDatabaseExcel()
    print(f"银行数量: {len(db.data)}")
    print(f"银行列表: {list(db.data.keys())}")

    stats = db.get_stats()
    for bank, info in list(stats.items())[:3]:
        print(f"\n{bank}:")
        print(f"  产品数: {info['products']}")
        print(f"  日期数: {info['dates']}")
        print(f"  日期范围: {info['earliest_date']} ~ {info['latest_date']}")
