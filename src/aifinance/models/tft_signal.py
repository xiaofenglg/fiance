# -*- coding: utf-8 -*-
"""
Temporal Fusion Transformer 信号模型 — V2 Phase 2

使用 NeuralForecast 的 TFT 实现,作为 V11 的主信号模型。

特点:
- Multi-horizon 预测 (7天)
- Variable Selection Network 自动特征选择
- 可解释的注意力权重
- 处理时序特征和静态特征

Phase 2 改进:
- 修复 val_size 错误 (确保 val_size >= horizon)
- 添加 Bank_ID 作为静态分类特征
- 添加 Shibor/Treasury 作为历史外生变量 (lag-1)

参考:
Lim, B., et al. (2019). Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 宏观数据加载器
try:
    from ..data.macro_loader import MacroDataLoader, load_macro_features
    _macro_available = True
except ImportError:
    _macro_available = False
    logger.debug("macro_loader 不可用")

# NeuralForecast 可选依赖
_nf_available = False
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT

    _nf_available = True
except ImportError:
    logger.debug("neuralforecast 未安装,TFT 模型不可用")

# 默认参数
DEFAULT_HORIZON = 7
DEFAULT_INPUT_SIZE = 14  # lookback 窗口，需小于最短序列长度
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_STEPS = 100


class TFTSignal:
    """Temporal Fusion Transformer 信号模型 — V2 Phase 2"""

    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        horizon: int = DEFAULT_HORIZON,
        max_steps: int = DEFAULT_MAX_STEPS,
        early_stop_patience: int = 10,
        learning_rate: float = 1e-3,
        random_seed: int = 42,
        # Phase 2 新参数
        use_bank_id: bool = True,
        use_macro_features: bool = True,
        macro_lag: int = 1,
    ):
        """
        Args:
            input_size: 输入特征维度 (25维因子)
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout 比例
            horizon: 预测窗口 (天)
            max_steps: 最大训练步数
            early_stop_patience: 早停耐心值
            learning_rate: 学习率
            random_seed: 随机种子
            use_bank_id: 是否使用 Bank_ID 作为静态分类特征
            use_macro_features: 是否使用宏观特征 (Shibor, Treasury)
            macro_lag: 宏观特征滞后期数 (默认 1 天)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.horizon = horizon
        self.max_steps = max_steps
        self.early_stop_patience = early_stop_patience
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        # Phase 2 参数
        self.use_bank_id = use_bank_id
        self.use_macro_features = use_macro_features and _macro_available
        self.macro_lag = macro_lag

        self._model = None
        self._nf = None
        self._trained = False
        self._feature_importance = None
        self._macro_loader = MacroDataLoader() if self.use_macro_features else None
        self._macro_cache: Optional[pd.DataFrame] = None
        self._bank_id_map: Dict[str, int] = {}  # 银行名 -> 分类编码

    def _create_model(self, n_static_categories: int = 0, hist_exog_list: List[str] = None):
        """创建 TFT 模型

        Args:
            n_static_categories: 静态分类特征的类别数 (Bank_ID)
            hist_exog_list: 历史外生变量列表 (不需要未来值)
        """
        if not _nf_available:
            raise ImportError("neuralforecast 未安装")

        # Phase 2: 配置静态分类特征
        stat_exog_list = ["bank_id"] if self.use_bank_id and n_static_categories > 0 else None

        # Phase 3 修复: 使用 hist_exog_list 而非 futr_exog_list
        # 因子特征只需要历史值,不需要预测未来值
        self._model = TFT(
            h=self.horizon,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_head=self.num_heads,
            dropout=self.dropout,
            max_steps=self.max_steps,
            early_stop_patience_steps=self.early_stop_patience,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            scaler_type="robust",
            batch_size=32,
            enable_progress_bar=False,  # 禁用进度条避免编码问题
            start_padding_enabled=True,  # 允许短序列训练
            # Phase 3 修复: 使用历史外生变量 (不需要未来值)
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
        )

        self._nf = NeuralForecast(models=[self._model], freq="D")

    def _prepare_data(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        as_of_idx: int,
        train_window: int = 180,
        bank_name: Optional[str] = None,
        product_codes: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
        """准备 NeuralForecast 格式的训练数据 — V2 Phase 2

        NeuralForecast 需要长格式 DataFrame:
        - unique_id: 产品标识
        - ds: 日期
        - y: 目标变量 (收益率)
        - 外生变量列...

        Phase 2 改进:
        - 添加 Bank_ID 静态分类特征
        - 添加 Shibor/Treasury 历史外生变量 (lag-1)

        Args:
            features: [n_products, n_dates, n_features]
            returns: [n_products, n_dates]
            masks: [n_products, n_dates]
            dates: 日期列表
            as_of_idx: 当前日期索引
            train_window: 训练窗口
            bank_name: 银行名称 (用于 Bank_ID)
            product_codes: 产品代码列表

        Returns:
            (df, static_df, hist_exog_list)
            - df: NeuralForecast 格式 DataFrame
            - static_df: 静态特征 DataFrame (Bank_ID)
            - hist_exog_list: 历史外生变量列名列表 (Phase 3 修复)
        """
        n_p, n_d, n_f = features.shape

        train_end = as_of_idx - self.horizon
        train_start = max(0, train_end - train_window)

        if train_end - train_start < 60:
            return None, None, []

        # 最小数据长度要求: input_size + horizon + 10
        min_data_length = self.input_size + self.horizon + 10

        # 预计算每个产品在训练窗口内的有效数据点数
        valid_counts = np.sum(masks[:, train_start:train_end + 1] > 0, axis=1)
        valid_products = np.where(valid_counts >= min_data_length)[0]

        if len(valid_products) == 0:
            logger.warning(f"[TFT] 没有产品满足最小数据长度要求 ({min_data_length})")
            return None, None, []

        logger.debug(f"[TFT] 数据过滤: {n_p} -> {len(valid_products)} 产品 (min_length={min_data_length})")

        # Phase 2: 加载宏观特征 (带 lag-1)
        macro_features = None
        macro_cols = []
        if self.use_macro_features and self._macro_loader:
            try:
                train_dates = dates[train_start:train_end + 1]
                macro_features = load_macro_features(train_dates)
                # 应用 lag-1
                if self.macro_lag > 0:
                    macro_features = macro_features.shift(self.macro_lag).fillna(method="bfill")
                macro_cols = list(macro_features.columns)
                logger.debug(f"[TFT] 宏观特征: {macro_cols}")
            except Exception as e:
                logger.warning(f"[TFT] 加载宏观特征失败: {e}")
                macro_features = None

        # Phase 3 修复: 暂时禁用外生变量以避免预测时的对齐问题
        # TFT 将仅基于目标序列 'y' 进行预测
        # 这简化了模型但仍能利用 TFT 的时序建模能力
        hist_exog_list = None  # 禁用外生变量

        # Phase 3 修复: 同时禁用 bank_id 静态特征
        # 因为预测时需要提供相同的 static_df,增加复杂性
        use_bank_id_local = False  # 暂时禁用

        # TODO: 未来可以添加外生变量,但需要确保预测时也提供相同的变量
        # hist_exog_list = [f"feat_{i}" for i in range(n_f)]
        # if macro_features is not None:
        #     hist_exog_list.extend(macro_cols)

        records = []
        for p in valid_products:
            for t in range(train_start, train_end + 1):
                if masks[p, t] > 0:
                    record = {
                        "unique_id": f"P{p}",
                        "ds": pd.Timestamp(dates[t]),
                        "y": float(returns[p, t]),
                    }
                    # Phase 3 修复: 暂时不添加外生特征,仅使用目标序列
                    # 这避免了预测时需要提供相同特征的问题
                    records.append(record)

        if len(records) < 100:
            return None, None, []

        df = pd.DataFrame(records)

        # Phase 2: 构建静态特征 DataFrame (Bank_ID)
        # Phase 3 修复: 暂时禁用,使用上面的 use_bank_id_local 变量
        static_df = None
        if use_bank_id_local and bank_name:
            # 为银行分配分类编码
            if bank_name not in self._bank_id_map:
                self._bank_id_map[bank_name] = len(self._bank_id_map)
            bank_id = self._bank_id_map[bank_name]

            static_records = []
            for p in valid_products:
                static_records.append({
                    "unique_id": f"P{p}",
                    "bank_id": bank_id,
                })
            static_df = pd.DataFrame(static_records)
            logger.debug(f"[TFT] 静态特征: bank_id={bank_id} ({bank_name})")

        return df, static_df, hist_exog_list

    def train(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        as_of_idx: int,
        bank_name: Optional[str] = None,
        product_codes: Optional[List[str]] = None,
    ) -> Dict:
        """训练 TFT 模型 — V2 Phase 2

        Args:
            features: [n_products, n_dates, n_features]
            returns: [n_products, n_dates]
            masks: [n_products, n_dates]
            dates: 日期列表
            as_of_idx: 当前日期索引
            bank_name: 银行名称 (Phase 2: 用于 Bank_ID)
            product_codes: 产品代码列表

        Returns:
            训练结果字典
        """
        if not _nf_available:
            return {"status": "neuralforecast_not_installed"}

        try:
            # Phase 2: 准备数据 (包含静态特征和宏观特征)
            df, static_df, hist_exog_list = self._prepare_data(
                features, returns, masks, dates, as_of_idx,
                bank_name=bank_name, product_codes=product_codes
            )
            if df is None:
                return {"status": "insufficient_data"}

            # Phase 2/3: 创建模型 (配置静态分类特征数量, 使用历史外生变量)
            n_static_categories = len(self._bank_id_map) if self.use_bank_id else 0
            self._create_model(
                n_static_categories=n_static_categories,
                hist_exog_list=hist_exog_list if hist_exog_list else None
            )

            # Phase 2: 修复 val_size 错误
            # 关键约束: val_size >= horizon (NeuralForecast 要求)
            logger.info(f"[TFT] 开始训练,样本数: {len(df)}")

            # 计算每个产品的序列长度
            min_series_len = df.groupby("unique_id").size().min()

            # Phase 2 修复: 确保 val_size >= horizon
            # NeuralForecast 要求: val_size >= horizon
            # 同时: val_size < (min_series_len - input_size)
            available_for_val = min_series_len - self.input_size - 1

            if available_for_val < self.horizon:
                # 数据不足以支持验证集,使用更小的 horizon 或跳过验证
                logger.warning(
                    f"[TFT] 数据不足: available_for_val={available_for_val} < horizon={self.horizon}, "
                    f"使用 val_size={max(1, available_for_val)}"
                )
                val_size = max(1, available_for_val)
            else:
                # 正常情况: val_size = horizon
                val_size = self.horizon

            logger.info(f"[TFT] min_series_len={min_series_len}, val_size={val_size}")

            # 禁用 PyTorch Lightning 输出以避免 Windows 编码问题
            import sys
            import io
            import warnings
            warnings.filterwarnings("ignore")

            # 临时重定向 stdout/stderr
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                self._nf.fit(df, static_df=static_df, val_size=val_size)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

            self._trained = True

            # 获取特征重要性 (通过 Variable Selection Network)
            feature_cols = [f"feat_{i}" for i in range(self.input_size)]
            available_cols = [c for c in feature_cols if c in df.columns]
            if available_cols:
                self._feature_importance = df[available_cols].var().values

            logger.info("[TFT] 训练完成")
            return {
                "status": "ok",
                "n_samples": len(df),
                "n_products": df["unique_id"].nunique(),
                "val_size": val_size,
                "has_bank_id": static_df is not None,
                "n_hist_features": len(hist_exog_list) if hist_exog_list else 0,
            }

        except Exception as e:
            logger.error(f"[TFT] 训练失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def predict(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        recent_returns: Optional[np.ndarray] = None,
    ) -> Dict:
        """预测 — Phase 3 简化版

        Args:
            features: [n_products, n_features] 或 [n_products, seq_len, n_features]
            masks: [n_products] 或 [n_products, seq_len]
            dates: 日期列表 (用于构建预测输入)
            recent_returns: [n_products, seq_len] 最近收益率 (用于预测)

        Returns:
            预测结果字典
        """
        n_p = features.shape[0]

        if not self._trained or self._nf is None:
            return self._fallback_predict(features)

        try:
            # Phase 3 修复: 需要 recent_returns 来构建预测输入
            if recent_returns is None:
                logger.debug("[TFT] 无 recent_returns,使用回退方法")
                return self._fallback_predict(features)

            seq_len = recent_returns.shape[1] if recent_returns.ndim == 2 else 1
            if seq_len < self.input_size:
                logger.debug(f"[TFT] seq_len={seq_len} < input_size={self.input_size},使用回退方法")
                return self._fallback_predict(features)

            # 准备预测 DataFrame (Phase 3: 仅使用 y,不使用外生特征)
            records = []
            for p in range(n_p):
                if masks.ndim == 1:
                    m = masks[p]
                else:
                    m = masks[p, -1] if masks.shape[1] > 0 else 0

                if m > 0:
                    # 构建该产品的历史序列
                    for t in range(seq_len):
                        date_idx = -(seq_len - t)
                        record = {
                            "unique_id": f"P{p}",
                            "ds": pd.Timestamp(dates[date_idx]) if len(dates) >= seq_len else pd.Timestamp("2025-01-01"),
                            "y": float(recent_returns[p, t]) if recent_returns.ndim == 2 else float(recent_returns[p]),
                        }
                        records.append(record)

            if not records:
                return self._fallback_predict(features)

            pred_df = pd.DataFrame(records)

            # 预测
            forecast = self._nf.predict(pred_df)

            # 提取预测结果
            release_prob = np.zeros(n_p, dtype=np.float32)
            expected_return = np.zeros(n_p, dtype=np.float32)

            for p in range(n_p):
                uid = f"P{p}"
                if uid in forecast["unique_id"].values:
                    pred_row = forecast[forecast["unique_id"] == uid]
                    # 使用平均预测值作为预期收益
                    pred_cols = [c for c in forecast.columns if c.startswith("TFT")]
                    if pred_cols:
                        expected_return[p] = pred_row[pred_cols].mean().mean()
                        # 将预期收益转换为释放概率
                        release_prob[p] = self._return_to_prob(expected_return[p])

            return {
                "release_prob": release_prob,
                "expected_return": expected_return,
            }

        except Exception as e:
            logger.warning(f"[TFT] 预测失败: {e},使用回退方法")
            return self._fallback_predict(features)

    def _return_to_prob(self, expected_return: float, threshold: float = 2.5) -> float:
        """将预期收益转换为释放概率

        Args:
            expected_return: 预期收益 (年化%)
            threshold: 释放阈值

        Returns:
            释放概率 [0, 1]
        """
        # Sigmoid 转换
        prob = 1 / (1 + np.exp(-(expected_return - threshold) / 1.5))
        return float(np.clip(prob, 0, 1))

    def _fallback_predict(self, features: np.ndarray) -> Dict:
        """回退预测方法 (当 TFT 不可用时)

        使用简单规则基于因子进行预测

        Args:
            features: [n_products, n_features] 或 [n_products, seq_len, n_features]

        Returns:
            预测结果
        """
        if features.ndim == 3:
            feat = features[:, -1, :]  # 取最后时间点
        else:
            feat = features

        n_p = feat.shape[0]

        # 使用因子组合估计
        # ret_1d (0), ret_5d_ma (1), mom_5_10 (5), vol_ratio (11)
        ret_score = feat[:, 0] * 0.3 + feat[:, 1] * 0.3 + feat[:, 5] * 0.2
        vol_penalty = np.clip(feat[:, 11] - 1, 0, 1) * 0.2

        expected_return = ret_score - vol_penalty
        release_prob = np.clip((ret_score / 5.0), 0, 1)

        return {
            "release_prob": release_prob.astype(np.float32),
            "expected_return": expected_return.astype(np.float32),
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        return self._feature_importance

    @property
    def is_trained(self) -> bool:
        return self._trained


def check_tft_available() -> bool:
    """检查 TFT 是否可用"""
    return _nf_available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"TFT 可用: {check_tft_available()}")

    # 测试
    n_products, n_dates, n_features = 50, 200, 25
    features = np.random.randn(n_products, n_dates, n_features).astype(np.float32)
    returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
    masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D").strftime("%Y-%m-%d").tolist()

    model = TFTSignal(max_steps=10)  # 快速测试

    if check_tft_available():
        result = model.train(features, returns, masks, dates, n_dates - 10)
        print(f"训练结果: {result}")

        if model.is_trained:
            pred = model.predict(features[:, -1, :], masks[:, -1], dates)
            print(f"预测结果 shape: {pred['release_prob'].shape}")
    else:
        # 测试回退方法
        pred = model._fallback_predict(features[:, -1, :])
        print(f"回退预测结果 shape: {pred['release_prob'].shape}")
