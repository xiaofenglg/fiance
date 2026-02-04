# -*- coding: utf-8 -*-
"""
宏观数据加载模块

加载宏观经济指标:
- SHIBOR (上海银行间同业拆放利率)
- 国债收益率曲线
- 银行间回购利率
- PMI 等经济指标
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroDataLoader:
    """宏观数据加载器"""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: 数据文件目录
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_shibor(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tenors: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """加载 SHIBOR 数据

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期
            tenors: 期限列表 (如 ["O/N", "1W", "1M", "3M"])

        Returns:
            SHIBOR 数据 DataFrame, 索引为日期
        """
        cache_key = "shibor"
        if cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            df = self._load_shibor_from_file()
            if df is not None:
                self._cache[cache_key] = df

        if df is None or df.empty:
            logger.warning("SHIBOR 数据不可用,生成模拟数据")
            df = self._generate_mock_shibor(start_date, end_date)

        # 过滤日期范围
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # 过滤期限
        if tenors:
            available_tenors = [t for t in tenors if t in df.columns]
            if available_tenors:
                df = df[available_tenors]

        return df

    def _load_shibor_from_file(self) -> Optional[pd.DataFrame]:
        """从文件加载 SHIBOR"""
        if not self.data_dir:
            return None

        shibor_path = self.data_dir / "shibor.csv"
        if not shibor_path.exists():
            return None

        try:
            df = pd.read_csv(shibor_path, index_col=0, parse_dates=True)
            df.index = df.index.strftime("%Y-%m-%d")
            return df
        except Exception as e:
            logger.error(f"加载 SHIBOR 文件失败: {e}")
            return None

    def _generate_mock_shibor(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """生成模拟 SHIBOR 数据"""
        if not start_date:
            start_date = "2020-01-01"
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        dates = pd.date_range(start=start_date, end=end_date, freq="B")

        # 基准利率 + 随机波动
        np.random.seed(42)
        n = len(dates)

        # 各期限基准值 (%)
        tenors = ["O/N", "1W", "2W", "1M", "3M", "6M", "9M", "1Y"]
        base_rates = [1.5, 2.0, 2.2, 2.5, 2.7, 2.8, 2.9, 3.0]

        data = {}
        for tenor, base in zip(tenors, base_rates):
            # AR(1) 过程模拟
            rates = [base]
            for _ in range(1, n):
                shock = np.random.randn() * 0.05
                new_rate = 0.98 * rates[-1] + 0.02 * base + shock
                rates.append(max(0.5, new_rate))
            data[tenor] = rates

        df = pd.DataFrame(data, index=dates.strftime("%Y-%m-%d"))
        return df

    def load_bond_yields(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        maturities: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """加载国债收益率曲线

        Args:
            start_date: 开始日期
            end_date: 结束日期
            maturities: 期限列表 (如 ["1Y", "3Y", "5Y", "10Y"])

        Returns:
            国债收益率 DataFrame
        """
        cache_key = "bond_yields"
        if cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            df = self._load_bond_yields_from_file()
            if df is not None:
                self._cache[cache_key] = df

        if df is None or df.empty:
            logger.warning("国债收益率数据不可用,生成模拟数据")
            df = self._generate_mock_bond_yields(start_date, end_date)

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        if maturities:
            available = [m for m in maturities if m in df.columns]
            if available:
                df = df[available]

        return df

    def _load_bond_yields_from_file(self) -> Optional[pd.DataFrame]:
        """从文件加载国债收益率"""
        if not self.data_dir:
            return None

        path = self.data_dir / "bond_yields.csv"
        if not path.exists():
            return None

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = df.index.strftime("%Y-%m-%d")
            return df
        except Exception as e:
            logger.error(f"加载国债收益率文件失败: {e}")
            return None

    def _generate_mock_bond_yields(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """生成模拟国债收益率"""
        if not start_date:
            start_date = "2020-01-01"
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        np.random.seed(43)
        n = len(dates)

        maturities = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "30Y"]
        base_yields = [2.0, 2.3, 2.5, 2.7, 2.9, 3.0, 3.5]

        data = {}
        for mat, base in zip(maturities, base_yields):
            yields = [base]
            for _ in range(1, n):
                shock = np.random.randn() * 0.03
                new_yield = 0.99 * yields[-1] + 0.01 * base + shock
                yields.append(max(0.5, new_yield))
            data[mat] = yields

        df = pd.DataFrame(data, index=dates.strftime("%Y-%m-%d"))
        return df

    def get_term_spread(
        self, bond_yields: pd.DataFrame, short_tenor: str = "1Y", long_tenor: str = "10Y"
    ) -> pd.Series:
        """计算期限利差

        Args:
            bond_yields: 国债收益率 DataFrame
            short_tenor: 短期限
            long_tenor: 长期限

        Returns:
            期限利差序列
        """
        if short_tenor not in bond_yields.columns or long_tenor not in bond_yields.columns:
            logger.warning(f"期限 {short_tenor} 或 {long_tenor} 不在数据中")
            return pd.Series(index=bond_yields.index, dtype=float)

        return bond_yields[long_tenor] - bond_yields[short_tenor]

    def get_rate_regime(
        self, shibor: pd.DataFrame, tenor: str = "3M", window: int = 20
    ) -> pd.Series:
        """识别利率环境

        Args:
            shibor: SHIBOR 数据
            tenor: 参考期限
            window: 移动窗口

        Returns:
            利率环境指标 (-1: 下行, 0: 平稳, 1: 上行)
        """
        if tenor not in shibor.columns:
            return pd.Series(index=shibor.index, data=0, dtype=int)

        rates = shibor[tenor]
        ma = rates.rolling(window=window).mean()
        diff = rates - ma

        regime = pd.Series(index=shibor.index, dtype=int)
        regime[diff > 0.1] = 1  # 上行
        regime[diff < -0.1] = -1  # 下行
        regime[(diff >= -0.1) & (diff <= 0.1)] = 0  # 平稳

        return regime.fillna(0).astype(int)

    def align_to_dates(
        self, macro_df: pd.DataFrame, target_dates: List[str]
    ) -> pd.DataFrame:
        """对齐宏观数据到目标日期

        Args:
            macro_df: 宏观数据 DataFrame
            target_dates: 目标日期列表

        Returns:
            对齐后的 DataFrame
        """
        # 创建目标日期索引
        target_index = pd.Index(target_dates)

        # 前向填充对齐
        aligned = macro_df.reindex(target_index, method="ffill")

        return aligned


def load_macro_features(
    dates: List[str], data_dir: Optional[str] = None
) -> pd.DataFrame:
    """便捷函数: 加载宏观特征

    Args:
        dates: 日期列表
        data_dir: 数据目录

    Returns:
        宏观特征 DataFrame
    """
    loader = MacroDataLoader(data_dir=data_dir)

    start_date = dates[0]
    end_date = dates[-1]

    # 加载数据
    shibor = loader.load_shibor(start_date, end_date)
    bond_yields = loader.load_bond_yields(start_date, end_date)

    # 构建特征
    features = pd.DataFrame(index=dates)

    # SHIBOR 特征
    shibor_aligned = loader.align_to_dates(shibor, dates)
    for tenor in ["O/N", "1M", "3M"]:
        if tenor in shibor_aligned.columns:
            features[f"shibor_{tenor}"] = shibor_aligned[tenor]

    # 期限利差
    bond_aligned = loader.align_to_dates(bond_yields, dates)
    term_spread = loader.get_term_spread(bond_aligned)
    features["term_spread"] = term_spread

    # 利率环境
    regime = loader.get_rate_regime(shibor_aligned)
    features["rate_regime"] = regime

    return features.fillna(method="ffill").fillna(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    loader = MacroDataLoader()

    # 加载 SHIBOR
    shibor = loader.load_shibor("2024-01-01", "2024-12-31")
    print("SHIBOR 数据:")
    print(shibor.head())
    print(f"Shape: {shibor.shape}")

    # 加载国债收益率
    bonds = loader.load_bond_yields("2024-01-01", "2024-12-31")
    print("\n国债收益率:")
    print(bonds.head())

    # 计算期限利差
    spread = loader.get_term_spread(bonds)
    print(f"\n期限利差 (10Y-1Y) 均值: {spread.mean():.2f}")

    # 加载宏观特征
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B").strftime("%Y-%m-%d").tolist()
    features = load_macro_features(dates)
    print(f"\n宏观特征 Shape: {features.shape}")
    print(features.head())
