# -*- coding: utf-8 -*-
"""
LightGBM 信号模型

作为 V11 的辅助模型 (权重 30%),与 TFT 模型集成。

特点:
- 使用25维因子作为输入
- 改进的标签定义 (相对强度 + 分位数截断)
- Walk-forward 训练
- 支持分类和回归双任务
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# LightGBM (可选)
_lgbm_available = False
lgb = None

try:
    import lightgbm as _lgb

    lgb = _lgb
    _lgbm_available = True
except ImportError:
    logger.debug("lightgbm 未安装,将使用统计回退")

# 超参数
TRAIN_WINDOW = 180  # 训练窗口 (天)
PRED_HORIZON = 20  # 预测窗口 (天)
MIN_SAMPLES = 500  # 最少训练样本


class LightGBMSignal:
    """LightGBM 信号生成模型"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.clf_model = None  # 分类模型 (是否释放)
        self.reg_model = None  # 回归模型 (预期收益)
        self._feature_importance = None
        self._trained = False

    def _create_labels(
        self,
        returns: np.ndarray,
        masks: np.ndarray,
        start_idx: int,
        horizon: int = PRED_HORIZON,
        target_pos_rate: float = 0.30,  # 增加正样本比例
    ) -> Tuple[np.ndarray, np.ndarray]:
        """创建相对强度标签 - V2 优化版

        标签逻辑 (针对银行理财产品优化):
        1. 绝对底线过滤: 年化收益 > 0.5% (降低阈值)
        2. 综合评分: 50% 收益排名 + 30% 稳定性排名 + 20% 动量
        3. 分位数截断: 取评分 Top 30% 作为正样本 (扩大正样本池)

        Args:
            returns: [n_products, n_dates] 年化收益率矩阵 (%)
            masks: [n_products, n_dates] 有效掩码
            start_idx: 当前时间索引
            horizon: 预测窗口
            target_pos_rate: 目标正样本率

        Returns:
            y_cls: 分类标签 (0/1)
            y_reg: 回归标签 (超额收益)
        """
        n_p, n_d = returns.shape
        end_idx = min(start_idx + horizon, n_d)

        if end_idx <= start_idx + 5:
            return np.zeros(n_p), np.zeros(n_p)

        fut_ret = returns[:, start_idx:end_idx]
        fut_mask = masks[:, start_idx:end_idx]

        # 计算区间平均年化收益
        valid_cnt = fut_mask.sum(axis=1)
        avg_yield = np.where(
            valid_cnt > 0,
            (fut_ret * fut_mask).sum(axis=1) / np.maximum(valid_cnt, 1),
            0,
        )

        # 计算波动率
        std_yield = np.nanstd(np.where(fut_mask > 0, fut_ret, np.nan), axis=1)
        std_yield = np.nan_to_num(std_yield, nan=10.0)

        # 计算动量 (过去表现的延续性)
        if start_idx >= 5:
            past_ret = returns[:, start_idx - 5:start_idx]
            past_mask = masks[:, start_idx - 5:start_idx]
            past_cnt = past_mask.sum(axis=1)
            momentum = np.where(
                past_cnt > 0,
                (past_ret * past_mask).sum(axis=1) / np.maximum(past_cnt, 1),
                0,
            )
        else:
            momentum = np.zeros(n_p)

        # 绝对底线过滤 - 针对银行理财产品降低阈值
        min_valid_ratio = 0.3  # 降低数据完整度要求
        abs_yield_floor = 0.5  # 降低收益门槛到0.5%
        valid_mask = (valid_cnt >= horizon * min_valid_ratio) & (
            avg_yield > abs_yield_floor
        )

        if valid_mask.sum() < 10:
            valid_mask = valid_cnt >= 2

        # 计算排名得分 - 收益排名
        yield_rank = np.zeros(n_p)
        valid_yields = avg_yield[valid_mask]
        if len(valid_yields) > 0:
            sorted_idx = np.argsort(valid_yields)
            ranks = np.zeros(len(valid_yields))
            ranks[sorted_idx] = np.arange(len(valid_yields)) / max(
                len(valid_yields) - 1, 1
            )
            yield_rank[valid_mask] = ranks

        # 稳定性排名 (低波动更好)
        risk_rank = np.zeros(n_p)
        valid_stds = std_yield[valid_mask]
        if len(valid_stds) > 0:
            sorted_idx = np.argsort(valid_stds)[::-1]  # 波动率低的排名高
            ranks = np.zeros(len(valid_stds))
            ranks[sorted_idx] = np.arange(len(valid_stds)) / max(
                len(valid_stds) - 1, 1
            )
            risk_rank[valid_mask] = ranks

        # 动量排名
        momentum_rank = np.zeros(n_p)
        valid_momentum = momentum[valid_mask]
        if len(valid_momentum) > 0:
            sorted_idx = np.argsort(valid_momentum)
            ranks = np.zeros(len(valid_momentum))
            ranks[sorted_idx] = np.arange(len(valid_momentum)) / max(
                len(valid_momentum) - 1, 1
            )
            momentum_rank[valid_mask] = ranks

        # 综合评分: 50% 收益 + 30% 稳定性 + 20% 动量
        composite_score = 0.50 * yield_rank + 0.30 * risk_rank + 0.20 * momentum_rank

        # 分位数截断 - 扩大正样本池
        valid_scores = composite_score[valid_mask]
        if len(valid_scores) > 0:
            cutoff_quantile = 1.0 - target_pos_rate
            score_threshold = np.percentile(valid_scores, cutoff_quantile * 100)
        else:
            score_threshold = 1.0

        # 生成标签
        y_cls = (valid_mask & (composite_score >= score_threshold)).astype(np.float32)

        # 回归标签: 超额收益 (相对市场中位数)
        market_median = np.median(avg_yield[valid_mask]) if valid_mask.sum() > 0 else 0
        y_reg = (avg_yield - market_median).astype(np.float32)

        return y_cls, y_reg

    def _prepare_train_data(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
        as_of_idx: int,
    ) -> Optional[Dict]:
        """准备训练数据

        Args:
            features: [n_products, n_dates, n_features]
            returns: [n_products, n_dates]
            masks: [n_products, n_dates]
            as_of_idx: 当前日期索引

        Returns:
            训练数据字典
        """
        n_p, n_d, n_f = features.shape
        train_end = as_of_idx - PRED_HORIZON
        train_start = max(0, train_end - TRAIN_WINDOW)

        if train_end - train_start < 60:
            return None

        X_list, y_cls_list, y_reg_list = [], [], []

        # 每3天采样一次
        for t in range(train_start, train_end - PRED_HORIZON, 3):
            x_t = features[:, t, :]
            m_t = masks[:, t]

            y_cls, y_reg = self._create_labels(returns, masks, t + 1)

            valid = m_t > 0
            if valid.sum() < 10:
                continue

            X_list.append(x_t[valid])
            y_cls_list.append(y_cls[valid])
            y_reg_list.append(y_reg[valid])

        if not X_list:
            return None

        X_all = np.vstack(X_list)
        y_cls_all = np.concatenate(y_cls_list)
        y_reg_all = np.concatenate(y_reg_list)

        if len(X_all) < MIN_SAMPLES:
            return None

        return {
            "X": X_all,
            "y_cls": y_cls_all,
            "y_reg": y_reg_all,
        }

    def train(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
        as_of_idx: int,
    ) -> Dict:
        """训练模型

        Args:
            features: [n_products, n_dates, n_features]
            returns: [n_products, n_dates]
            masks: [n_products, n_dates]
            as_of_idx: 当前日期索引

        Returns:
            训练结果字典
        """
        if not _lgbm_available:
            return {"status": "lgbm_not_installed"}

        data = self._prepare_train_data(features, returns, masks, as_of_idx)
        if data is None:
            return {"status": "insufficient_data"}

        X = data["X"]
        y_cls = data["y_cls"]
        y_reg = data["y_reg"]

        n_samples = len(X)
        split = int(n_samples * 0.85)

        # 时间序列切分
        indices = np.arange(n_samples)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_cls_train, y_cls_val = y_cls[train_idx], y_cls[val_idx]
        y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]

        # 处理 NaN 值 - 移除包含 NaN 的样本
        train_valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_cls_train) | np.isnan(y_reg_train))
        val_valid = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_cls_val) | np.isnan(y_reg_val))

        if train_valid.sum() < 100 or val_valid.sum() < 10:
            logger.warning(f"[LightGBM] 过滤 NaN 后数据不足: train={train_valid.sum()}, val={val_valid.sum()}")
            return {"status": "insufficient_data_after_nan_filter"}

        X_train, y_cls_train, y_reg_train = X_train[train_valid], y_cls_train[train_valid], y_reg_train[train_valid]
        X_val, y_cls_val, y_reg_val = X_val[val_valid], y_cls_val[val_valid], y_reg_val[val_valid]

        pos_rate = y_cls_train.mean()
        scale_pos_weight = min((1 - pos_rate) / max(pos_rate, 0.05), 10.0)

        # 分类模型
        clf_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": scale_pos_weight,
            "min_child_samples": 20,
            "verbosity": -1,
            "seed": self.random_state,
        }

        train_data_cls = lgb.Dataset(X_train, label=y_cls_train)
        val_data_cls = lgb.Dataset(X_val, label=y_cls_val, reference=train_data_cls)

        self.clf_model = lgb.train(
            clf_params,
            train_data_cls,
            num_boost_round=500,
            valid_sets=[val_data_cls],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # 回归模型
        reg_params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "verbosity": -1,
            "seed": self.random_state,
        }

        train_data_reg = lgb.Dataset(X_train, label=y_reg_train)
        val_data_reg = lgb.Dataset(X_val, label=y_reg_val, reference=train_data_reg)

        self.reg_model = lgb.train(
            reg_params,
            train_data_reg,
            num_boost_round=300,
            valid_sets=[val_data_reg],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        self._feature_importance = self.clf_model.feature_importance()
        self._trained = True

        # 验证集指标
        val_pred_cls = self.clf_model.predict(X_val)
        val_pred_reg = self.reg_model.predict(X_val)

        try:
            from sklearn.metrics import mean_absolute_error, roc_auc_score

            # 过滤可能的 NaN 值
            valid_mask = ~(np.isnan(y_reg_val) | np.isnan(val_pred_reg))
            if valid_mask.sum() > 0:
                val_mae = mean_absolute_error(y_reg_val[valid_mask], val_pred_reg[valid_mask])
            else:
                val_mae = 1.0

            cls_valid = ~(np.isnan(y_cls_val) | np.isnan(val_pred_cls))
            if cls_valid.sum() > 0 and len(np.unique(y_cls_val[cls_valid])) > 1:
                val_auc = roc_auc_score(y_cls_val[cls_valid], val_pred_cls[cls_valid])
            else:
                val_auc = 0.5
        except (ImportError, ValueError) as e:
            logger.warning(f"[LightGBM] 验证指标计算失败: {e}")
            val_auc = 0.5
            val_mae = 1.0

        logger.info(
            f"[LightGBM] 训练完成: {n_samples} 样本, "
            f"pos_rate={pos_rate:.3f}, val_auc={val_auc:.4f}, val_mae={val_mae:.4f}"
        )

        return {
            "status": "ok",
            "n_samples": n_samples,
            "pos_rate": round(pos_rate, 4),
            "val_auc": round(val_auc, 4),
            "val_mae": round(val_mae, 4),
        }

    def predict(self, features: np.ndarray) -> Dict:
        """预测

        Args:
            features: [n_products, n_features] 当前因子

        Returns:
            预测结果字典
        """
        if self.clf_model is None or self.reg_model is None:
            return self._statistical_fallback(features)

        release_prob = self.clf_model.predict(features)
        expected_ret = self.reg_model.predict(features)

        return {
            "release_prob": release_prob.astype(np.float32),
            "expected_return": expected_ret.astype(np.float32),
        }

    def _statistical_fallback(self, features: np.ndarray) -> Dict:
        """统计回退"""
        n_p = features.shape[0]

        ret_1d = features[:, 0] if features.shape[1] > 0 else np.zeros(n_p)
        ret_5d = features[:, 1] if features.shape[1] > 1 else np.zeros(n_p)

        release_prob = np.clip((ret_5d / 5.0), 0, 1)
        expected_ret = ret_1d

        return {
            "release_prob": release_prob.astype(np.float32),
            "expected_return": expected_ret.astype(np.float32),
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        return self._feature_importance

    @property
    def is_trained(self) -> bool:
        return self._trained


def check_lgbm_available() -> bool:
    """检查 LightGBM 是否可用"""
    return _lgbm_available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"LightGBM 可用: {check_lgbm_available()}")

    # 测试
    n_products, n_dates, n_features = 100, 200, 25
    features = np.random.randn(n_products, n_dates, n_features).astype(np.float32)
    returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
    masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)

    model = LightGBMSignal()
    result = model.train(features, returns, masks, n_dates - 10)
    print(f"训练结果: {result}")

    if result["status"] == "ok":
        pred = model.predict(features[:, -1, :])
        print(
            f"预测 shape: release_prob={pred['release_prob'].shape}, "
            f"expected_return={pred['expected_return'].shape}"
        )
