# -*- coding: utf-8 -*-
"""
Temporal Fusion Transformer 信号模型

使用 NeuralForecast 的 TFT 实现,作为 V11 的主信号模型。

特点:
- Multi-horizon 预测 (7天)
- Variable Selection Network 自动特征选择
- 可解释的注意力权重
- 处理时序特征和静态特征

参考:
Lim, B., et al. (2019). Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
DEFAULT_INPUT_SIZE = 25
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_STEPS = 100


class TFTSignal:
    """Temporal Fusion Transformer 信号模型"""

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

        self._model = None
        self._nf = None
        self._trained = False
        self._feature_importance = None

    def _create_model(self):
        """创建 TFT 模型"""
        if not _nf_available:
            raise ImportError("neuralforecast 未安装")

        self._model = TFT(
            h=self.horizon,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            attention_head_size=self.num_heads,
            dropout_rate=self.dropout,
            max_steps=self.max_steps,
            early_stop_patience_steps=self.early_stop_patience,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            scaler_type="robust",
            batch_size=32,
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
    ) -> Optional[pd.DataFrame]:
        """准备 NeuralForecast 格式的训练数据

        NeuralForecast 需要长格式 DataFrame:
        - unique_id: 产品标识
        - ds: 日期
        - y: 目标变量 (收益率)
        - 外生变量列...

        Args:
            features: [n_products, n_dates, n_features]
            returns: [n_products, n_dates]
            masks: [n_products, n_dates]
            dates: 日期列表
            as_of_idx: 当前日期索引
            train_window: 训练窗口

        Returns:
            NeuralForecast 格式 DataFrame
        """
        n_p, n_d, n_f = features.shape

        train_end = as_of_idx - self.horizon
        train_start = max(0, train_end - train_window)

        if train_end - train_start < 60:
            return None

        records = []
        for p in range(n_p):
            for t in range(train_start, train_end + 1):
                if masks[p, t] > 0:
                    record = {
                        "unique_id": f"P{p}",
                        "ds": pd.Timestamp(dates[t]),
                        "y": float(returns[p, t]),
                    }
                    # 添加特征
                    for f_idx in range(n_f):
                        record[f"feat_{f_idx}"] = float(features[p, t, f_idx])
                    records.append(record)

        if len(records) < 100:
            return None

        df = pd.DataFrame(records)
        return df

    def train(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        as_of_idx: int,
    ) -> Dict:
        """训练 TFT 模型

        Args:
            features: [n_products, n_dates, n_features]
            returns: [n_products, n_dates]
            masks: [n_products, n_dates]
            dates: 日期列表
            as_of_idx: 当前日期索引

        Returns:
            训练结果字典
        """
        if not _nf_available:
            return {"status": "neuralforecast_not_installed"}

        try:
            # 准备数据
            df = self._prepare_data(features, returns, masks, dates, as_of_idx)
            if df is None:
                return {"status": "insufficient_data"}

            # 创建模型
            self._create_model()

            # 外生变量列表
            futr_exog_list = [f"feat_{i}" for i in range(self.input_size)]

            # 训练
            logger.info(f"[TFT] 开始训练,样本数: {len(df)}")
            self._nf.fit(df, static_df=None)

            self._trained = True

            # 获取特征重要性 (通过 Variable Selection Network)
            # TFT 的 VSN 可以提供特征重要性,但需要额外处理
            # 这里简化为使用特征方差作为重要性代理
            feature_cols = [f"feat_{i}" for i in range(self.input_size)]
            self._feature_importance = df[feature_cols].var().values

            logger.info("[TFT] 训练完成")
            return {
                "status": "ok",
                "n_samples": len(df),
                "n_products": df["unique_id"].nunique(),
            }

        except Exception as e:
            logger.error(f"[TFT] 训练失败: {e}")
            return {"status": "error", "message": str(e)}

    def predict(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        recent_returns: Optional[np.ndarray] = None,
    ) -> Dict:
        """预测

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
            # 构建预测输入
            if features.ndim == 2:
                # 单时间点,需要历史数据
                if recent_returns is None:
                    return self._fallback_predict(features)
                seq_features = features[:, np.newaxis, :]
            else:
                seq_features = features

            # 准备预测 DataFrame
            records = []
            seq_len = seq_features.shape[1] if seq_features.ndim == 3 else 1

            for p in range(n_p):
                for t in range(seq_len):
                    if masks.ndim == 1:
                        m = masks[p]
                    else:
                        m = masks[p, t]

                    if m > 0:
                        record = {
                            "unique_id": f"P{p}",
                            "ds": pd.Timestamp(dates[-(seq_len - t)]) if len(dates) >= seq_len else pd.Timestamp("2025-01-01"),
                            "y": float(recent_returns[p, t]) if recent_returns is not None else 0.0,
                        }
                        for f_idx in range(self.input_size):
                            if seq_features.ndim == 3:
                                record[f"feat_{f_idx}"] = float(seq_features[p, t, f_idx])
                            else:
                                record[f"feat_{f_idx}"] = float(seq_features[p, f_idx])
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
