# -*- coding: utf-8 -*-
"""
Qlib 数据集成模块

将 NAV 数据转换为 Qlib 格式,支持:
- 自定义数据 Provider
- 与 Qlib 模型框架集成
- Alpha 因子表达式

注意: 此模块为可选依赖,需要安装 qlib
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Qlib 是可选依赖
_qlib_available = False
try:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    _qlib_available = True
except ImportError:
    logger.debug("qlib 未安装,Qlib 集成功能不可用")


class NavQlibProvider:
    """NAV 数据 Qlib Provider"""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Qlib 格式数据存储路径
        """
        self.data_path = Path(data_path)
        self._initialized = False

    def init_qlib(self, region: str = "cn") -> bool:
        """初始化 Qlib

        Args:
            region: 市场区域

        Returns:
            是否成功初始化
        """
        if not _qlib_available:
            logger.error("qlib 未安装,无法初始化")
            return False

        try:
            qlib.init(provider_uri=str(self.data_path), region=region)
            self._initialized = True
            logger.info(f"Qlib 初始化成功: {self.data_path}")
            return True
        except Exception as e:
            logger.error(f"Qlib 初始化失败: {e}")
            return False

    def convert_nav_to_qlib(
        self,
        nav_df: pd.DataFrame,
        date_cols: List[str],
        product_col: str = "产品代码",
        output_path: Optional[str] = None,
    ) -> bool:
        """将 NAV DataFrame 转换为 Qlib 格式

        Qlib 格式要求:
        - 每个 instrument 一个目录
        - 每个字段一个 .bin 文件
        - calendar.txt, instruments 等元数据

        Args:
            nav_df: NAV 数据 (宽格式)
            date_cols: 日期列列表
            product_col: 产品代码列名
            output_path: 输出路径

        Returns:
            是否成功转换
        """
        if output_path is None:
            output_path = self.data_path

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # 转换为长格式
            products = nav_df[product_col].tolist() if product_col in nav_df.columns else None
            if products is None:
                products = [f"PROD_{i}" for i in range(len(nav_df))]

            # 创建长格式数据
            records = []
            for idx, row in nav_df.iterrows():
                product = products[idx] if idx < len(products) else f"PROD_{idx}"
                for date in date_cols:
                    nav = row.get(date)
                    if pd.notna(nav):
                        records.append(
                            {
                                "instrument": str(product).replace("/", "_"),
                                "datetime": date,
                                "nav": float(nav),
                            }
                        )

            if not records:
                logger.warning("没有有效的 NAV 数据")
                return False

            long_df = pd.DataFrame(records)
            long_df["datetime"] = pd.to_datetime(long_df["datetime"])

            # 计算收益率
            long_df = long_df.sort_values(["instrument", "datetime"])
            long_df["return"] = long_df.groupby("instrument")["nav"].pct_change()

            # 保存 calendar
            calendar = sorted(long_df["datetime"].unique())
            calendar_path = output_path / "calendars"
            calendar_path.mkdir(exist_ok=True)
            with open(calendar_path / "day.txt", "w") as f:
                for dt in calendar:
                    f.write(dt.strftime("%Y-%m-%d") + "\n")

            # 保存 instruments
            instruments = sorted(long_df["instrument"].unique())
            inst_path = output_path / "instruments"
            inst_path.mkdir(exist_ok=True)
            with open(inst_path / "all.txt", "w") as f:
                for inst in instruments:
                    start = long_df[long_df["instrument"] == inst]["datetime"].min()
                    end = long_df[long_df["instrument"] == inst]["datetime"].max()
                    f.write(f"{inst}\t{start.strftime('%Y-%m-%d')}\t{end.strftime('%Y-%m-%d')}\n")

            # 保存特征数据
            features_path = output_path / "features"
            features_path.mkdir(exist_ok=True)

            for inst in instruments:
                inst_data = long_df[long_df["instrument"] == inst].copy()
                inst_data = inst_data.set_index("datetime").sort_index()

                inst_dir = features_path / inst
                inst_dir.mkdir(exist_ok=True)

                # 保存 nav 和 return
                for field in ["nav", "return"]:
                    if field in inst_data.columns:
                        values = inst_data[field].values.astype(np.float32)
                        values.tofile(inst_dir / f"{field}.bin")

            logger.info(
                f"Qlib 数据转换完成: {len(instruments)} 个产品, "
                f"{len(calendar)} 天, 路径: {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Qlib 数据转换失败: {e}")
            return False

    def create_dataset(
        self,
        instruments: Union[str, List[str]] = "all",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        fit_start_time: Optional[str] = None,
        fit_end_time: Optional[str] = None,
    ):
        """创建 Qlib DatasetH

        Args:
            instruments: 产品列表或 "all"
            start_time: 数据开始时间
            end_time: 数据结束时间
            fit_start_time: 训练开始时间
            fit_end_time: 训练结束时间

        Returns:
            DatasetH 对象
        """
        if not _qlib_available:
            raise ImportError("qlib 未安装")

        if not self._initialized:
            self.init_qlib()

        # 定义特征
        fields = [
            "$nav",
            "$return",
            "Ref($nav, 1)",
            "Ref($nav, 5)",
            "Mean($return, 5)",
            "Mean($return, 20)",
            "Std($return, 5)",
            "Std($return, 20)",
        ]
        names = [
            "nav",
            "return",
            "nav_lag1",
            "nav_lag5",
            "return_ma5",
            "return_ma20",
            "return_std5",
            "return_std20",
        ]

        # 创建 Handler
        handler_config = {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": fit_start_time,
            "fit_end_time": fit_end_time,
            "instruments": instruments,
            "data_loader": {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (fields, names),
                        "label": (["Ref($return, -1)"], ["label"]),
                    }
                },
            },
        }

        dataset = DatasetH(handler=handler_config)
        return dataset

    def get_features(
        self,
        instruments: Union[str, List[str]] = "all",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取特征数据

        Args:
            instruments: 产品列表
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            特征 DataFrame
        """
        if not _qlib_available:
            raise ImportError("qlib 未安装")

        if not self._initialized:
            self.init_qlib()

        fields = ["$nav", "$return"]
        df = D.features(instruments, fields, start_time, end_time)

        return df


def check_qlib_available() -> bool:
    """检查 Qlib 是否可用"""
    return _qlib_available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Qlib 可用: {check_qlib_available()}")

    if check_qlib_available():
        # 测试
        provider = NavQlibProvider("./qlib_data")

        # 模拟数据
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
        date_cols = dates.strftime("%Y-%m-%d").tolist()

        nav_df = pd.DataFrame(
            {
                "产品代码": ["PROD_A", "PROD_B", "PROD_C"],
                **{d: np.random.rand(3) + 1 for d in date_cols},
            }
        )

        # 转换
        success = provider.convert_nav_to_qlib(nav_df, date_cols)
        print(f"转换成功: {success}")
