# -*- coding: utf-8 -*-
"""
V11 模型集成框架

集成策略:
- TFT (Temporal Fusion Transformer): 70% 权重 (主信号模型)
- LightGBM: 30% 权重 (辅助模型)

特点:
- 动态权重调整 (基于验证集表现)
- 信号一致性检查
- 回退机制保证稳定性
"""

import logging
from typing import Dict, List, Optional


import numpy as np

from .lgbm_signal import LightGBMSignal, check_lgbm_available
from .tft_signal import TFTSignal, check_tft_available

logger = logging.getLogger(__name__)

# 默认权重
DEFAULT_WEIGHTS = {
    "tft": 0.70,  # TFT 作为主模型
    "lgbm": 0.30,  # LightGBM 辅助
}


class EnsembleModel:
    """V11 模型集成框架"""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        tft_params: Optional[Dict] = None,
        lgbm_params: Optional[Dict] = None,
        device=None,
    ):
        """
        Args:
            weights: 模型权重 {"tft": 0.7, "lgbm": 0.3}
            tft_params: TFT 模型参数
            lgbm_params: LightGBM 模型参数
            device: 计算设备
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # 初始化模型
        tft_params = tft_params or {}
        lgbm_params = lgbm_params or {}

        self.tft = TFTSignal(**tft_params)
        self.lgbm = LightGBMSignal(**lgbm_params)

        self._trained = False
        self._train_results = {}
        self._model_status = {"tft": False, "lgbm": False}

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
        """训练集成模型 — V2 Phase 2

        Args:
            features: [n_products, n_dates, n_features]
            returns: [n_products, n_dates]
            masks: [n_products, n_dates]
            dates: 日期列表
            as_of_idx: 当前日期索引
            bank_name: 银行名称 (Phase 2: 用于 TFT Bank_ID)
            product_codes: 产品代码列表

        Returns:
            训练结果汇总
        """
        results = {}

        # 训练 TFT (Phase 2: 传递 bank_name)
        if check_tft_available():
            logger.info("[Ensemble] 开始训练 TFT...")
            tft_result = self.tft.train(
                features, returns, masks, dates, as_of_idx,
                bank_name=bank_name, product_codes=product_codes
            )
            results["tft"] = tft_result
            self._model_status["tft"] = tft_result.get("status") == "ok"
        else:
            results["tft"] = {"status": "not_available"}
            self._model_status["tft"] = False

        # 训练 LightGBM
        if check_lgbm_available():
            logger.info("[Ensemble] 开始训练 LightGBM...")
            lgbm_result = self.lgbm.train(features, returns, masks, as_of_idx)
            results["lgbm"] = lgbm_result
            self._model_status["lgbm"] = lgbm_result.get("status") == "ok"
        else:
            results["lgbm"] = {"status": "not_available"}
            self._model_status["lgbm"] = False

        # 动态调整权重
        self._adjust_weights(results)

        self._trained = any(self._model_status.values())
        results["weights"] = self.weights.copy()
        results["trained"] = self._trained
        self._train_results = results

        logger.info(
            f"[Ensemble] 训练完成, 权重: TFT={self.weights['tft']:.2f}, "
            f"LGBM={self.weights['lgbm']:.2f}"
        )

        return results

    def _adjust_weights(self, train_results: Dict) -> None:
        """动态调整模型权重

        基于训练结果调整权重:
        - 如果某模型训练失败,将其权重分配给其他模型
        - 可以基于验证集 AUC 进行更精细的调整

        Args:
            train_results: 训练结果字典
        """
        tft_ok = self._model_status["tft"]
        lgbm_ok = self._model_status["lgbm"]

        if tft_ok and lgbm_ok:
            # 两个模型都可用,使用默认权重
            self.weights = DEFAULT_WEIGHTS.copy()

            # 可选: 基于验证指标微调
            lgbm_auc = train_results.get("lgbm", {}).get("val_auc", 0.5)
            if lgbm_auc > 0.65:
                # LightGBM 表现很好,稍微提高权重
                self.weights["lgbm"] = min(0.4, self.weights["lgbm"] + 0.05)
                self.weights["tft"] = 1 - self.weights["lgbm"]

        elif tft_ok:
            self.weights = {"tft": 1.0, "lgbm": 0.0}
        elif lgbm_ok:
            self.weights = {"tft": 0.0, "lgbm": 1.0}
        else:
            # 两个模型都不可用,保持原权重 (使用回退方法)
            self.weights = {"tft": 0.5, "lgbm": 0.5}

    def predict(
        self,
        features: np.ndarray,
        features_seq: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        masks_seq: Optional[np.ndarray] = None,
        dates: Optional[List[str]] = None,
        recent_returns: Optional[np.ndarray] = None,
    ) -> Dict:
        """集成预测

        Args:
            features: [n_products, n_features] 当前时刻因子 (LightGBM用)
            features_seq: [n_products, seq_len, n_features] 时序因子 (TFT用)
            masks: [n_products] 当前时刻掩码
            masks_seq: [n_products, seq_len] 序列掩码
            dates: 日期列表
            recent_returns: [n_products, seq_len] 最近收益率

        Returns:
            集成预测结果
        """
        n_p = features.shape[0]

        # LightGBM 预测
        lgbm_pred = self.lgbm.predict(features)
        lgbm_prob = lgbm_pred["release_prob"]
        lgbm_ret = lgbm_pred["expected_return"]

        # TFT 预测
        if features_seq is not None and masks_seq is not None and dates is not None:
            tft_pred = self.tft.predict(features_seq, masks_seq, dates, recent_returns)
            tft_prob = tft_pred["release_prob"]
            tft_ret = tft_pred["expected_return"]
        else:
            # TFT 使用回退
            tft_pred = self.tft._fallback_predict(features)
            tft_prob = tft_pred["release_prob"]
            tft_ret = tft_pred["expected_return"]

        # 加权融合
        w_tft = self.weights.get("tft", 0.7)
        w_lgbm = self.weights.get("lgbm", 0.3)

        ensemble_prob = tft_prob * w_tft + lgbm_prob * w_lgbm
        ensemble_ret = tft_ret * w_tft + lgbm_ret * w_lgbm

        # 计算综合信号强度 (传入收益用于方向一致性检查)
        signal_strength = self._compute_signal_strength(
            ensemble_prob, ensemble_ret, tft_prob, lgbm_prob, tft_ret, lgbm_ret
        )

        return {
            "release_prob": ensemble_prob,
            "expected_return": ensemble_ret,
            "signal_strength": signal_strength,
            "tft_prob": tft_prob,
            "lgbm_prob": lgbm_prob,
            "tft_ret": tft_ret,
            "lgbm_ret": lgbm_ret,
        }

    def _compute_signal_strength(
        self,
        prob: np.ndarray,
        ret: np.ndarray,
        tft_prob: np.ndarray,
        lgbm_prob: np.ndarray,
        tft_ret: Optional[np.ndarray] = None,
        lgbm_ret: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算综合信号强度 - V2 优化版

        针对银行理财产品优化:
        - 释放概率 (35%)
        - 预期收益 (35%) - 提高收益权重
        - 模型一致性 (15%) - 降低一致性要求
        - 基础信号加成 (15%) - 确保有足够的信号输出

        Args:
            prob: 集成释放概率
            ret: 集成预期收益
            tft_prob: TFT 释放概率
            lgbm_prob: LightGBM 释放概率
            tft_ret: TFT 预期收益
            lgbm_ret: LightGBM 预期收益

        Returns:
            signal_strength: [0, 1]
        """
        # 概率贡献 - 使用更宽松的范围
        prob_score = np.clip(prob, 0, 1)

        # 收益贡献 - 针对银行理财产品使用更低的归一化标准
        # 银行理财年化收益通常在 1-4%，所以用 3.0 而非 5.0
        ret_score = np.clip(ret / 3.0, 0, 1)

        # 一致性贡献 (两个模型越一致越好)
        prob_diff = np.abs(tft_prob - lgbm_prob)
        consistency = 1 - prob_diff

        # 基础信号加成 - 确保即使模型不确定，也有基础信号
        base_signal = 0.15

        signal_strength = (
            prob_score * 0.35 +
            ret_score * 0.35 +
            consistency * 0.15 +
            base_signal
        )

        # === 软性一致性过滤 ===
        # 当两个模型概率差距 > 0.60 时，信号衰减 15% (更宽松)
        inconsistent_mask = prob_diff > 0.60
        signal_strength = np.where(inconsistent_mask, signal_strength * 0.85, signal_strength)

        # 确保输出在合理范围
        return np.clip(signal_strength, 0.05, 1.0).astype(np.float32)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性 (优先使用 LightGBM)"""
        lgbm_imp = self.lgbm.get_feature_importance()
        if lgbm_imp is not None:
            return lgbm_imp

        tft_imp = self.tft.get_feature_importance()
        return tft_imp

    @property
    def is_trained(self) -> bool:
        return self._trained


def get_ensemble_signals(
    features: np.ndarray,
    returns: np.ndarray,
    masks: np.ndarray,
    dates: List[str],
    as_of_date: Optional[str] = None,
) -> Dict:
    """便捷函数: 获取集成信号

    Args:
        features: [n_products, n_dates, n_features]
        returns: [n_products, n_dates]
        masks: [n_products, n_dates]
        dates: 日期列表
        as_of_date: 预测日期

    Returns:
        {product_idx: {signal_strength, release_prob, expected_return, ...}}
    """
    if as_of_date is None:
        as_of_idx = len(dates) - 1
    else:
        try:
            as_of_idx = dates.index(as_of_date)
        except ValueError:
            as_of_idx = len(dates) - 1

    model = EnsembleModel()
    train_result = model.train(features, returns, masks, dates, as_of_idx)

    if not model.is_trained:
        logger.warning("[Ensemble] 模型训练失败,返回空信号")
        return {}

    # 准备预测输入
    current_features = features[:, as_of_idx, :]
    seq_start = max(0, as_of_idx - 59)
    features_seq = features[:, seq_start : as_of_idx + 1, :]
    masks_seq = masks[:, seq_start : as_of_idx + 1]
    recent_returns = returns[:, seq_start : as_of_idx + 1]

    # 预测
    pred = model.predict(
        current_features,
        features_seq,
        masks[:, as_of_idx],
        masks_seq,
        dates[seq_start : as_of_idx + 1],
        recent_returns,
    )

    # 构建结果
    result = {}
    n_p = features.shape[0]

    for i in range(n_p):
        if masks[i, as_of_idx] > 0:
            result[i] = {
                "signal_strength": float(pred["signal_strength"][i]),
                "release_prob": float(pred["release_prob"][i]),
                "expected_return": float(pred["expected_return"][i]),
                "tft_prob": float(pred["tft_prob"][i]),
                "lgbm_prob": float(pred["lgbm_prob"][i]),
            }

    return result


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    n_products, n_dates, n_features = 50, 200, 25
    features = np.random.randn(n_products, n_dates, n_features).astype(np.float32)
    returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
    masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
    dates = (
        pd.date_range("2024-01-01", periods=n_dates, freq="D")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    model = EnsembleModel()
    train_result = model.train(features, returns, masks, dates, n_dates - 10)
    print(f"训练结果: {train_result}")

    if model.is_trained:
        current_feat = features[:, -1, :]
        seq_feat = features[:, -60:, :]
        seq_mask = masks[:, -60:]
        recent_ret = returns[:, -60:]

        pred = model.predict(
            current_feat,
            seq_feat,
            masks[:, -1],
            seq_mask,
            dates[-60:],
            recent_ret,
        )
        print(f"Signal strengths: {pred['signal_strength'][:5]}")
