# -*- coding: utf-8 -*-
"""
Hierarchical Risk Parity (HRP) 组合优化器 — V2 Phase 3

替代 Kelly Criterion,实现更稳健的组合优化。

HRP 优点:
- 不需要协方差矩阵求逆 (避免 MVO 的不稳定性)
- 层次聚类捕捉资产相关性结构
- 风险分散更均衡

Phase 3 改进:
- Ledoit-Wolf 协方差收缩 (更鲁棒的协方差估计)
- IVP (Inverse Variance Portfolio) 回退机制
- 增强的后处理约束

参考:
López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
Journal of Portfolio Management, 42(4), 59-69.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional
covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411.

需要安装: riskfolio-lib
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Ledoit-Wolf 协方差收缩
# ═════════════════════════════════════════════════════════════════════════════

def ledoit_wolf_shrinkage(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf 协方差收缩估计

    实现 Ledoit & Wolf (2004) 的收缩估计器。
    目标: 在样本协方差矩阵和结构化目标(对角矩阵)之间寻找最优权衡。

    公式: Σ_shrunk = δ * F + (1 - δ) * S
    其中:
    - S: 样本协方差矩阵
    - F: 收缩目标 (缩放的单位矩阵)
    - δ: 最优收缩强度

    Args:
        returns: [n_samples, n_assets] 收益率矩阵

    Returns:
        (shrunk_cov, shrinkage_intensity)
        - shrunk_cov: 收缩后的协方差矩阵
        - shrinkage_intensity: 收缩强度 δ ∈ [0, 1]
    """
    n_samples, n_assets = returns.shape

    if n_samples < 2 or n_assets < 1:
        return np.eye(max(n_assets, 1)), 1.0

    # 1. 计算样本协方差矩阵
    X = returns - returns.mean(axis=0)
    sample_cov = np.dot(X.T, X) / n_samples

    # 2. 计算收缩目标 F (缩放的单位矩阵)
    # 使用样本协方差的平均方差作为目标
    trace = np.trace(sample_cov)
    mu = trace / n_assets
    F = mu * np.eye(n_assets)

    # 3. 计算最优收缩强度 δ
    # Ledoit-Wolf 公式
    delta = sample_cov - F

    # 估计 π (方差项的预期平方误差)
    X2 = X ** 2
    sample_cov2 = np.dot(X2.T, X2) / n_samples
    pi_mat = sample_cov2 - sample_cov ** 2
    pi = np.sum(pi_mat)

    # 估计 rho (收缩目标的贡献)
    rho_diag = np.sum(np.diag(pi_mat))
    rho_off = pi - rho_diag

    # 估计 gamma (delta 的 Frobenius 范数)
    gamma = np.sum(delta ** 2)

    # 计算最优 δ (收缩强度)
    kappa = (pi - rho_off) / gamma if gamma > 1e-10 else 1.0
    shrinkage = max(0, min(1, kappa / n_samples))

    # 4. 应用收缩
    shrunk_cov = shrinkage * F + (1 - shrinkage) * sample_cov

    # 确保正定性
    min_eigenvalue = np.min(np.linalg.eigvalsh(shrunk_cov))
    if min_eigenvalue < 1e-8:
        shrunk_cov += (1e-8 - min_eigenvalue) * np.eye(n_assets)

    logger.debug(f"[Ledoit-Wolf] shrinkage intensity: {shrinkage:.4f}")

    return shrunk_cov, shrinkage


def inverse_variance_portfolio(cov: np.ndarray) -> np.ndarray:
    """逆方差组合 (IVP) — Phase 3 回退机制

    当 HRP 失败时使用的简单但稳健的组合方法。
    权重与方差成反比: w_i = (1/σ²_i) / Σ(1/σ²_j)

    Args:
        cov: [n_assets, n_assets] 协方差矩阵

    Returns:
        [n_assets] 权重向量
    """
    n_assets = cov.shape[0]

    # 提取方差 (对角线元素)
    variances = np.diag(cov)

    # 避免除零
    variances = np.maximum(variances, 1e-10)

    # 逆方差权重
    inv_var = 1.0 / variances
    weights = inv_var / np.sum(inv_var)

    return weights

# riskfolio-lib 可选依赖
_rp_available = False
_HCPortfolio = None
try:
    import riskfolio as rp
    from riskfolio import HCPortfolio as _HCPortfolio

    _rp_available = True
except ImportError:
    logger.debug("riskfolio-lib 未安装,将使用内置 HRP 实现")


@dataclass
class HRPResult:
    """HRP 优化结果"""

    weights: Dict[int, float]  # {product_idx: weight}
    raw_weights: np.ndarray  # 原始权重数组
    risk_contribution: Dict[int, float]  # 风险贡献
    cluster_labels: Optional[np.ndarray]  # 聚类标签


class HRPOptimizer:
    """Hierarchical Risk Parity 组合优化器"""

    def __init__(
        self,
        risk_measure: str = "MV",
        linkage_method: str = "ward",
        max_weight: float = 0.15,  # 降低最大权重以提高分散度
        min_weight: float = 0.02,  # 降低最小权重以允许更多产品
        rf_rate: float = 0.02,
    ):
        """
        Args:
            risk_measure: 风险度量 ("MV", "MAD", "CVaR", "CDaR")
            linkage_method: 聚类方法 ("single", "complete", "average", "ward")
            max_weight: 单个资产最大权重
            min_weight: 单个资产最小权重
            rf_rate: 年化无风险利率
        """
        self.risk_measure = risk_measure
        self.linkage_method = linkage_method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.rf_rate = rf_rate

    def optimize(
        self,
        returns: pd.DataFrame,
        signals: Optional[Dict[int, float]] = None,
        use_riskfolio: bool = True,
    ) -> HRPResult:
        """基于 HRP 算法优化权重

        Args:
            returns: 候选产品的历史收益率 [n_dates, n_products]
                    列名应为产品索引
            signals: 模型信号 {product_idx: signal_strength}

        Returns:
            HRPResult 优化结果
        """
        if returns.empty:
            return HRPResult(
                weights={}, raw_weights=np.array([]), risk_contribution={}, cluster_labels=None
            )

        # 数据清洗: 移除包含 NaN/Inf 的列和行
        returns = self._clean_returns(returns)

        if returns.empty or returns.shape[1] < 2:
            logger.warning("[HRP] 清洗后数据不足,使用等权")
            return self._equal_weight_result(returns, signals)

        n_assets = returns.shape[1]

        if _rp_available and use_riskfolio:
            result = self._optimize_riskfolio(returns, signals)
        else:
            result = self._optimize_native(returns, signals)

        return result

    def _clean_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """清洗收益率数据,移除 NaN 和 Inf (不捏造数据)"""
        original_cols = len(returns.columns)
        original_rows = len(returns)

        # 1. 替换 Inf 为 NaN
        returns = returns.replace([np.inf, -np.inf], np.nan)

        # 2. 删除全空列
        returns = returns.dropna(axis=1, how="all")

        # 3. 删除含有过多 NaN 的列 (超过 70%)
        nan_ratio = returns.isna().sum() / len(returns)
        valid_cols = nan_ratio[nan_ratio < 0.7].index
        returns = returns[valid_cols]

        # 4. 删除方差为0的列 (常数列会导致相关矩阵问题)
        # pd.std() 默认忽略 NaN
        valid_cols = []
        for col in returns.columns:
            if returns[col].std() > 1e-8:
                valid_cols.append(col)
        returns = returns[valid_cols]

        logger.debug(
            f"[HRP] 数据清洗: {original_cols}列x{original_rows}行 -> {len(returns.columns)}列 (无数据填充)"
        )

        return returns

    def _optimize_riskfolio(
        self, returns: pd.DataFrame, signals: Optional[Dict[int, float]] = None
    ) -> HRPResult:
        """使用 riskfolio-lib 进行 HRP 优化

        Args:
            returns: 收益率 DataFrame
            signals: 信号强度

        Returns:
            HRPResult
        """
        n_assets = returns.shape[1]
        product_indices = [int(c) for c in returns.columns]

        try:
            # 对于大量资产，直接使用 IVP (riskfolio HRP 可能很慢)
            if n_assets > 200:
                logger.info(f"[HRP] 资产数 {n_assets} > 200, 直接使用 IVP")
                return self._ivp_fallback(returns, product_indices, signals)

            # 创建 HCPortfolio 对象 (用于层次聚类优化)
            port = _HCPortfolio(returns=returns)

            # HRP 优化
            w = port.optimization(
                model="HRP",
                codependence="pearson",
                rm=self.risk_measure,
                rf=self.rf_rate / 252,  # 日无风险利率
                linkage=self.linkage_method,
                leaf_order=True,
            )

            if w is None or w.empty:
                logger.warning("[HRP] riskfolio 优化失败,使用 IVP 回退")
                return self._ivp_fallback(returns, product_indices, signals)

            # 转换为字典
            raw_weights = w.values.flatten()

            # 检查权重有效性
            if np.any(np.isnan(raw_weights)) or np.sum(raw_weights) < 1e-10:
                logger.warning("[HRP] riskfolio 产生无效权重,使用 IVP 回退")
                return self._ivp_fallback(returns, product_indices, signals)

            logger.info(f"[HRP] riskfolio 优化成功, 非零权重: {np.sum(raw_weights > 1e-6)}/{n_assets}")

            # 应用信号调整和权重约束
            adjusted_weights = self._apply_constraints(
                raw_weights, product_indices, signals
            )

            # 计算风险贡献
            risk_contrib = self._compute_risk_contribution(returns, adjusted_weights)

            return HRPResult(
                weights=adjusted_weights,
                raw_weights=raw_weights,
                risk_contribution=risk_contrib,
                cluster_labels=None,
            )

        except Exception as e:
            logger.warning(f"[HRP] riskfolio 优化异常: {e}, 使用 IVP 回退")
            return self._ivp_fallback(returns, product_indices, signals)

    def _optimize_native(
        self, returns: pd.DataFrame, signals: Optional[Dict[int, float]] = None
    ) -> HRPResult:
        """原生 HRP 实现 (不依赖 riskfolio) — Phase 3 改进版

        改进点:
        1. 使用 Ledoit-Wolf 协方差收缩 (更鲁棒)
        2. IVP 回退机制 (当 HRP 失败时)
        """
        n_assets = returns.shape[1]
        product_indices = [int(c) for c in returns.columns]

        try:
            # ═══════════════════════════════════════════════════════════
            # Phase 3: 使用 Ledoit-Wolf 协方差收缩
            # ═══════════════════════════════════════════════════════════

            # 准备收益率矩阵 (处理 NaN)
            returns_clean = returns.fillna(0).values
            n_samples = returns_clean.shape[0]

            # 使用 Ledoit-Wolf 收缩估计协方差矩阵
            cov, shrinkage_intensity = ledoit_wolf_shrinkage(returns_clean)
            logger.info(f"[HRP] Ledoit-Wolf 收缩强度: {shrinkage_intensity:.4f}")

            # 从协方差矩阵计算相关矩阵
            std_dev = np.sqrt(np.diag(cov))
            std_dev = np.maximum(std_dev, 1e-8)
            corr = cov / np.outer(std_dev, std_dev)
            np.fill_diagonal(corr, 1.0)

            # 确保相关系数在 [-1, 1] 范围内
            corr = np.clip(corr, -1.0, 1.0)

            # 距离矩阵: d = sqrt(0.5 * (1 - corr))
            dist = np.sqrt(0.5 * np.clip(1 - corr, 0, 2))
            np.fill_diagonal(dist, 0)

            # 2. 层次聚类
            condensed_dist = squareform(dist, checks=False)
            link = linkage(condensed_dist, method=self.linkage_method)

            # 3. 准序列化 (Quasi-Diagonalization)
            sort_ix = self._get_quasi_diag(link)

            # 4. 递归二分 (Recursive Bisection)
            # 使用 Ledoit-Wolf 收缩后的协方差矩阵
            raw_weights = self._recursive_bisection(cov, sort_ix)

            # 检查权重有效性
            if np.any(np.isnan(raw_weights)) or np.sum(raw_weights) < 1e-10:
                raise ValueError("HRP 产生无效权重")

            # 应用约束
            adjusted_weights = self._apply_constraints(
                raw_weights, product_indices, signals
            )

            # 风险贡献
            risk_contrib = self._compute_risk_contribution(returns, adjusted_weights)

            return HRPResult(
                weights=adjusted_weights,
                raw_weights=raw_weights,
                risk_contribution=risk_contrib,
                cluster_labels=None,
            )

        except Exception as e:
            logger.warning(f"[HRP] 原生优化异常: {e}, 使用 IVP 回退")

            # ═══════════════════════════════════════════════════════════
            # Phase 3: IVP 回退机制
            # ═══════════════════════════════════════════════════════════
            return self._ivp_fallback(returns, product_indices, signals)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """获取准对角化排序

        Args:
            link: 聚类链接矩阵

        Returns:
            排序后的索引
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def _recursive_bisection(
        self, cov: np.ndarray, sort_ix: List[int]
    ) -> np.ndarray:
        """递归二分算法

        Args:
            cov: 协方差矩阵
            sort_ix: 排序索引

        Returns:
            权重数组
        """
        n = len(sort_ix)
        w = np.ones(n)

        # 递归分配
        clusters = [sort_ix]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # 分成两半
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # 计算每个子簇的方差
                var_left = self._cluster_variance(cov, left)
                var_right = self._cluster_variance(cov, right)

                # 分配权重 (反比于方差)
                alpha = 1 - var_left / (var_left + var_right + 1e-10)

                for i in left:
                    w[i] *= alpha
                for i in right:
                    w[i] *= 1 - alpha

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        # 归一化
        w = w / w.sum()
        return w

    def _cluster_variance(self, cov: np.ndarray, indices: List[int]) -> float:
        """计算子簇方差

        Args:
            cov: 协方差矩阵
            indices: 资产索引

        Returns:
            子簇方差
        """
        sub_cov = cov[np.ix_(indices, indices)]
        # 逆方差权重
        inv_diag = 1 / (np.diag(sub_cov) + 1e-10)
        inv_diag = inv_diag / inv_diag.sum()
        var = np.dot(inv_diag, np.dot(sub_cov, inv_diag))
        return var

    def _ivp_fallback(
        self,
        returns: pd.DataFrame,
        product_indices: List[int],
        signals: Optional[Dict[int, float]] = None,
    ) -> HRPResult:
        """IVP (Inverse Variance Portfolio) 回退机制 — Phase 3

        当 HRP 优化失败时使用的稳健回退方法。
        权重与方差成反比,对协方差矩阵奇异性不敏感。

        Args:
            returns: 收益率 DataFrame
            product_indices: 产品索引列表
            signals: 信号强度

        Returns:
            HRPResult 使用 IVP 权重
        """
        try:
            n_assets = len(product_indices)

            # 使用 Ledoit-Wolf 收缩估计协方差
            returns_clean = returns.fillna(0).values
            cov, shrinkage = ledoit_wolf_shrinkage(returns_clean)

            # 计算 IVP 权重
            raw_weights = inverse_variance_portfolio(cov)

            logger.info(f"[HRP] IVP 回退成功, 收缩强度: {shrinkage:.4f}")

            # 应用约束和信号调整
            adjusted_weights = self._apply_constraints(
                raw_weights, product_indices, signals
            )

            # 风险贡献 (IVP 的风险贡献近似相等)
            risk_contrib = {idx: 1.0 / len(adjusted_weights) for idx in adjusted_weights}

            return HRPResult(
                weights=adjusted_weights,
                raw_weights=raw_weights,
                risk_contribution=risk_contrib,
                cluster_labels=None,
            )

        except Exception as e:
            logger.error(f"[HRP] IVP 回退也失败: {e}, 使用等权")
            return self._equal_weight_result(returns, signals)

    def _apply_constraints(
        self,
        raw_weights: np.ndarray,
        product_indices: List[int],
        signals: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """应用权重约束和信号调整

        Args:
            raw_weights: 原始权重
            product_indices: 产品索引
            signals: 信号强度

        Returns:
            调整后的权重字典
        """
        weights = raw_weights.copy()
        n_assets = len(weights)

        if n_assets == 0:
            return {}

        # 处理 NaN 和无效值
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        # 如果所有权重都是0，使用等权
        if np.sum(weights) < 1e-10:
            weights = np.ones(n_assets) / n_assets

        # 1. 信号调整 (可选)
        if signals:
            for i, idx in enumerate(product_indices):
                signal = signals.get(idx, 0.5)
                # 信号强度作为乘数 (0.5-1.5 范围)
                multiplier = 0.5 + signal
                weights[i] *= multiplier

        # 2. 归一化
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            weights = np.ones(n_assets) / n_assets

        # 3. 应用权重约束
        # 当产品数很多时，动态调整约束以确保合理的权重分布
        if n_assets > 100:
            # 对于大量资产，放宽约束
            effective_min = max(1e-6, 0.3 / n_assets)
            effective_max = min(self.max_weight, 3.0 / n_assets)
        else:
            effective_min = min(self.min_weight, 0.5 / n_assets)
            effective_max = self.max_weight

        weights = np.clip(weights, effective_min, effective_max)

        # 4. 再次归一化
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            weights = np.ones(n_assets) / n_assets

        # 5. 转换为字典 - 保留所有非零权重
        result = {}
        for i, idx in enumerate(product_indices):
            if weights[i] > 1e-8:
                result[idx] = float(weights[i])

        # 6. 如果结果为空，返回等权的前 N 个产品
        if not result:
            logger.warning(f"[HRP] _apply_constraints 产生空权重，使用等权回退")
            n_keep = min(n_assets, 50)  # 最多保留50个
            equal_weight = 1.0 / n_keep
            for i in range(n_keep):
                result[product_indices[i]] = equal_weight

        return result

    def _compute_risk_contribution(
        self, returns: pd.DataFrame, weights: Dict[int, float]
    ) -> Dict[int, float]:
        """计算风险贡献"""
        if not weights:
            return {}

        product_indices = list(weights.keys())
        w = np.array([weights.get(idx, 0) for idx in product_indices])

        # 提取相关列
        cols = [str(idx) for idx in product_indices if str(idx) in returns.columns]
        if not cols:
            return {}

        # 使用 pairwise covariance 处理缺失值
        cov_df = returns[cols].cov(min_periods=30)
        cov = cov_df.fillna(0).values
        np.fill_diagonal(cov, np.maximum(np.diag(cov), 1e-8))

        if cov.ndim == 0 or len(cols) == 1:
            return {product_indices[0] if product_indices else 0: 1.0}

        # 边际风险贡献
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        if port_vol < 1e-10:
            return {idx: 1.0 / len(weights) for idx in product_indices}

        marginal_contrib = np.dot(cov, w) / port_vol
        risk_contrib = w * marginal_contrib
        risk_contrib = risk_contrib / risk_contrib.sum()

        return {idx: float(risk_contrib[i]) for i, idx in enumerate(product_indices)}

    def _equal_weight_result(
        self, returns: pd.DataFrame, signals: Optional[Dict[int, float]] = None
    ) -> HRPResult:
        """等权回退

        Args:
            returns: 收益率
            signals: 信号

        Returns:
            等权 HRPResult
        """
        n = returns.shape[1]
        if n == 0:
            return HRPResult(weights={}, raw_weights=np.array([]), risk_contribution={}, cluster_labels=None)

        product_indices = [int(c) for c in returns.columns]
        raw_weights = np.ones(n) / n

        weights = self._apply_constraints(raw_weights, product_indices, signals)
        risk_contrib = {idx: 1.0 / len(weights) for idx in weights}

        return HRPResult(
            weights=weights,
            raw_weights=raw_weights,
            risk_contribution=risk_contrib,
            cluster_labels=None,
        )


def optimize_portfolio(
    returns: np.ndarray,
    masks: np.ndarray,
    product_indices: List[int],
    signals: Optional[Dict[int, float]] = None,
    lookback: int = 60,
    current_idx: int = -1,
    **kwargs,
) -> Dict[int, float]:
    """便捷函数: 优化组合权重

    Args:
        returns: [n_products, n_dates] 收益率矩阵
        masks: [n_products, n_dates] 掩码
        product_indices: 候选产品索引列表
        signals: 信号强度
        lookback: 回望窗口
        current_idx: 当前日期索引
        **kwargs: HRPOptimizer 参数

    Returns:
        权重字典
    """
    n_p, n_d = returns.shape
    if current_idx < 0:
        current_idx = n_d + current_idx

    start_idx = max(0, current_idx - lookback + 1)

    # 构建收益率 DataFrame - 只选择有足够有效数据的产品
    records = {}
    valid_product_count = 0
    for idx in product_indices:
        if idx < n_p:
            ret = returns[idx, start_idx : current_idx + 1]
            mask = masks[idx, start_idx : current_idx + 1]

            # 检查该产品在窗口内有多少有效数据
            valid_days = np.sum(mask > 0)
            if valid_days < lookback * 0.3:  # 至少需要30%的有效数据
                continue

            valid_ret = np.where(mask > 0, ret, np.nan)
            records[str(idx)] = valid_ret
            valid_product_count += 1

    logger.debug(f"[HRP] 构建DataFrame: {len(product_indices)} 候选 -> {valid_product_count} 有效产品")

    df = pd.DataFrame(records)
    df = df.dropna(how="all", axis=1)  # 删除全空列

    if df.empty or len(df.columns) < 2:
        logger.warning(f"[HRP] DataFrame 为空或产品不足: {len(df.columns)} 列, 使用信号排序等权")
        # 使用信号强度排序的等权作为回退
        if signals and product_indices:
            sorted_indices = sorted(product_indices, key=lambda x: signals.get(x, 0), reverse=True)
            n_top = min(20, len(sorted_indices))  # 最多20个
            equal_weight = 1.0 / n_top
            return {idx: equal_weight for idx in sorted_indices[:n_top]}
        return {}

    optimizer = HRPOptimizer(**kwargs)
    result = optimizer.optimize(df, signals)

    return result.weights


def check_riskfolio_available() -> bool:
    """检查 riskfolio-lib 是否可用"""
    return _rp_available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"riskfolio-lib 可用: {check_riskfolio_available()}")

    # 测试
    np.random.seed(42)
    n_assets, n_days = 10, 100

    # 生成模拟收益率
    returns = np.random.randn(n_days, n_assets) * 2 + 3
    returns_df = pd.DataFrame(returns, columns=[str(i) for i in range(n_assets)])

    # 模拟信号
    signals = {i: 0.5 + 0.5 * np.random.rand() for i in range(n_assets)}

    optimizer = HRPOptimizer()
    result = optimizer.optimize(returns_df, signals)

    print("\nHRP 优化结果:")
    print(f"权重: {result.weights}")
    print(f"风险贡献: {result.risk_contribution}")
    print(f"权重和: {sum(result.weights.values()):.4f}")
