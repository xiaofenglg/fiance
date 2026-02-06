# -*- coding: utf-8 -*-
"""
GLM NAV 去平滑算法 — V2 资产过滤器版本

实现 Getmansky-Lo-Makarov (2004) 算法，将其从"预处理步骤"转变为"资产过滤器"。

核心改进:
1. 周频强制重采样 (W-WED)
2. theta_0 < 0.10 的资产被标记为"僵尸资产"并剔除
3. 完整的验证指标 (ACF, Variance, Adjusted Sharpe)

参考文献:
Getmansky, M., Lo, A. W., & Makarov, I. (2004).
An econometric model of serial correlation and illiquidity in hedge fund returns.
Journal of Financial Economics, 74(3), 529-609.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 自定义异常
# ═════════════════════════════════════════════════════════════════════════════

class IlliquidityException(Exception):
    """非流动性异常 — 当资产的 theta_0 < 阈值时抛出"""

    def __init__(self, asset_id: str, theta_0: float, threshold: float = 0.10):
        self.asset_id = asset_id
        self.theta_0 = theta_0
        self.threshold = threshold
        super().__init__(
            f"Asset Rejected: High Latency — {asset_id} has theta_0={theta_0:.4f} < {threshold}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 验证指标数据类
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class UnsmoothingMetrics:
    """去平滑前后的验证指标"""
    asset_id: str
    theta: np.ndarray
    theta_0: float
    smoothing_index: float

    # 去平滑前
    acf_lag1_before: float
    variance_before: float
    sharpe_before: float

    # 去平滑后
    acf_lag1_after: float
    variance_after: float
    sharpe_after: float
    adjusted_sharpe: float

    # 状态
    is_valid: bool
    rejection_reason: Optional[str] = None

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else f"INVALID ({self.rejection_reason})"
        return (
            f"[{self.asset_id}] {status}\n"
            f"  θ = [{', '.join(f'{t:.3f}' for t in self.theta)}], θ_0 = {self.theta_0:.3f}\n"
            f"  Before: ACF(1)={self.acf_lag1_before:.4f}, Var={self.variance_before:.6f}, SR={self.sharpe_before:.2f}\n"
            f"  After:  ACF(1)={self.acf_lag1_after:.4f}, Var={self.variance_after:.6f}, SR={self.sharpe_after:.2f}\n"
            f"  Adjusted Sharpe Ratio = {self.adjusted_sharpe:.3f}"
        )


class GLMUnsmoothing:
    """Getmansky-Lo-Makarov 去平滑处理器 — V2 资产过滤器版本

    核心改进:
    1. 强制 theta 约束: Sum(theta) = 1, 0 <= theta <= 1
    2. theta_0 阈值检查: 低于阈值抛出 IlliquidityException
    3. 完整的验证指标输出
    """

    # 默认阈值常量
    THETA_0_MIN_THRESHOLD = 0.10  # theta_0 最小阈值
    THETA_0_WARN_THRESHOLD = 0.30  # theta_0 警告阈值

    def __init__(
        self,
        max_lag: int = 2,
        method: str = "mle",
        theta_0_threshold: float = 0.10,
        strict_filter: bool = True,
    ):
        """
        Args:
            max_lag: 最大滞后阶数 (周频建议 k=2)
            method: 估计方法 ("mle", "acf")
            theta_0_threshold: theta_0 最小阈值，低于此值视为僵尸资产
            strict_filter: True=抛出异常, False=仅警告
        """
        self.max_lag = max_lag
        self.method = method
        self.theta_0_threshold = theta_0_threshold
        self.strict_filter = strict_filter
        self._theta = None
        self._metrics: Optional[UnsmoothingMetrics] = None

    def estimate_theta(self, observed_returns: np.ndarray) -> np.ndarray:
        """估计平滑系数 θ — 带严格约束的优化

        模型: R_t^o = θ_0 * R_t + θ_1 * R_{t-1} + ... + θ_k * R_{t-k}
        约束: Σθ_i = 1 且 0 <= θ_i <= 1

        Args:
            observed_returns: 观测收益率序列 [n_samples]

        Returns:
            theta: 平滑系数 [max_lag + 1]
        """
        n = len(observed_returns)

        # 周频数据至少需要 12 周 (一个季度)
        min_samples = 12
        if n < min_samples:
            logger.warning(f"样本量过小 ({n} < {min_samples}),使用默认 θ=[1,0,0]")
            return np.array([1.0] + [0.0] * self.max_lag)

        # 计算自相关函数
        acf = self._compute_acf(observed_returns, self.max_lag)

        if self.method == "acf":
            theta = self._theta_from_acf(acf)
        else:
            # MLE 估计 — 使用约束优化
            theta = self._constrained_mle_estimate(observed_returns, acf)

        self._theta = theta
        return theta

    def _compute_acf(self, returns: np.ndarray, max_lag: int) -> np.ndarray:
        """计算自相关函数

        Args:
            returns: 收益率序列
            max_lag: 最大滞后阶数

        Returns:
            acf: 自相关系数 [max_lag + 1]
        """
        n = len(returns)
        mean = np.mean(returns)
        var = np.var(returns)

        if var < 1e-10:
            return np.zeros(max_lag + 1)

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, max_lag + 1):
            cov = np.mean((returns[lag:] - mean) * (returns[:-lag] - mean))
            acf[lag] = cov / var

        return acf

    def _theta_from_acf(self, acf: np.ndarray) -> np.ndarray:
        """从 ACF 估计 θ (简化方法)

        利用关系: ρ_j = Σ_{i=0}^{k-j} θ_i * θ_{i+j}

        Args:
            acf: 自相关系数

        Returns:
            theta: 平滑系数
        """
        k = self.max_lag

        # 初始化: θ_0 = sqrt(1 - 2*Σρ_j)
        rho_sum = sum(acf[1 : k + 1])

        # 避免负数
        theta_0_sq = max(1 - 2 * rho_sum, 0.1)
        theta_0 = np.sqrt(theta_0_sq)

        # 估计其他 θ
        theta = np.zeros(k + 1)
        theta[0] = theta_0

        for j in range(1, k + 1):
            if theta_0 > 1e-6:
                # 近似: θ_j ≈ ρ_j / θ_0
                theta[j] = max(acf[j] / theta_0, 0)

        # 归一化确保 Σθ = 1
        theta_sum = theta.sum()
        if theta_sum > 0:
            theta = theta / theta_sum

        return theta

    def _constrained_mle_estimate(self, returns: np.ndarray, acf: np.ndarray) -> np.ndarray:
        """约束 MLE 估计 θ — Getmansky-Lo-Makarov 算法

        使用 scipy.optimize.minimize 进行约束优化:
        - 目标: 最小化观测 ACF 与理论 ACF 的差异
        - 约束: Sum(θ) = 1, 0 <= θ_i <= 1

        Args:
            returns: 观测收益率
            acf: 自相关函数

        Returns:
            theta: MLE 估计的 θ
        """
        k = self.max_lag
        n_params = k + 1

        # 初始值: 均匀分布
        theta_init = np.ones(n_params) / n_params

        def objective(theta):
            """ACF 匹配损失函数"""
            # 计算理论 ACF: ρ_j = Σ_{i=0}^{k-j} θ_i * θ_{i+j}
            theoretical_acf = np.zeros(n_params)
            for j in range(n_params):
                for i in range(n_params - j):
                    theoretical_acf[j] += theta[i] * theta[i + j]

            # 加权 ACF 匹配损失 (给 lag-1 更高权重)
            weights = np.array([1.0] + [2.0] * k)
            loss = np.sum(weights * (theoretical_acf - acf) ** 2)
            return loss

        # 约束: Sum(θ) = 1
        constraints = {"type": "eq", "fun": lambda theta: np.sum(theta) - 1.0}

        # 边界: 0 <= θ_i <= 1
        bounds = [(0.0, 1.0) for _ in range(n_params)]

        # 执行优化
        result = optimize.minimize(
            objective,
            theta_init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning(f"GLM 优化未收敛: {result.message}, 使用 ACF 方法回退")
            return self._theta_from_acf(acf)

        theta = result.x

        # 确保非负并归一化 (数值安全)
        theta = np.maximum(theta, 0)
        theta_sum = theta.sum()
        if theta_sum > 1e-10:
            theta = theta / theta_sum
        else:
            # 极端情况: 回退到默认
            theta = np.array([1.0] + [0.0] * k)

        return theta

    def _mle_estimate(self, returns: np.ndarray, acf: np.ndarray) -> np.ndarray:
        """旧版 MLE 估计 — 保留兼容性"""
        return self._constrained_mle_estimate(returns, acf)

    def _gmm_estimate(self, returns: np.ndarray, acf: np.ndarray) -> np.ndarray:
        """GMM 估计 θ

        Args:
            returns: 观测收益率
            acf: 自相关函数

        Returns:
            theta: GMM 估计的 θ
        """
        # 简化实现: 使用 ACF 方法
        return self._theta_from_acf(acf)

    def unsmooth(
        self,
        observed_returns: np.ndarray,
        theta: Optional[np.ndarray] = None,
        asset_id: str = "unknown",
    ) -> np.ndarray:
        """去平滑: 恢复真实收益率 — 带资产过滤逻辑

        使用反卷积恢复真实收益率序列。
        如果 theta_0 < 阈值，根据 strict_filter 设置决定是否抛出异常。

        Args:
            observed_returns: 观测收益率 [n_samples] 或 [n_products, n_dates]
            theta: 平滑系数 (如果为 None,则先估计)
            asset_id: 资产标识符 (用于日志和异常)

        Returns:
            true_returns: 去平滑后的真实收益率

        Raises:
            IlliquidityException: 当 theta_0 < 阈值且 strict_filter=True
        """
        # 处理 2D 输入 — 逐资产处理
        if observed_returns.ndim == 2:
            n_products, n_dates = observed_returns.shape
            true_returns = np.zeros_like(observed_returns)

            for p in range(n_products):
                try:
                    true_returns[p] = self.unsmooth(
                        observed_returns[p], theta, asset_id=f"product_{p}"
                    )
                except IlliquidityException:
                    # 2D 模式下，标记无效但不中断
                    true_returns[p] = np.nan
                    raise

            return true_returns

        # ══════════════════════════════════════════════════════════════════════
        # 1D 处理
        # ══════════════════════════════════════════════════════════════════════
        n = len(observed_returns)
        if n < 10:
            return observed_returns.copy()

        # 估计或使用给定的 θ
        if theta is None:
            theta = self.estimate_theta(observed_returns)

        theta_0 = theta[0]
        self._theta = theta

        # ══════════════════════════════════════════════════════════════════════
        # 核心检查: theta_0 阈值
        # ══════════════════════════════════════════════════════════════════════
        if theta_0 < self.theta_0_threshold:
            msg = f"Asset Rejected: High Latency — {asset_id} theta_0={theta_0:.4f} < {self.theta_0_threshold}"
            logger.warning(msg)

            if self.strict_filter:
                raise IlliquidityException(asset_id, theta_0, self.theta_0_threshold)
            else:
                # 非严格模式: 返回原始数据
                return observed_returns.copy()

        # 警告级别 (不阻止)
        if theta_0 < self.THETA_0_WARN_THRESHOLD:
            logger.info(
                f"Asset Warning: Moderate Latency — {asset_id} theta_0={theta_0:.4f}"
            )

        # 反卷积恢复真实收益率
        try:
            true_returns = self._inverse_filter(observed_returns, theta)
        except Exception as e:
            logger.warning(f"反卷积失败 ({asset_id}): {e}, 返回原始收益率")
            return observed_returns.copy()

        return true_returns

    def _inverse_filter(self, observed: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """逆滤波器

        R_t = (R_t^o - θ_1 * R_{t-1} - ... - θ_k * R_{t-k}) / θ_0

        Args:
            observed: 观测收益率
            theta: 平滑系数

        Returns:
            true: 真实收益率
        """
        n = len(observed)
        k = len(theta) - 1
        theta_0 = theta[0]

        if theta_0 < 0.1:
            # θ_0 过小,不稳定
            return observed.copy()

        true = np.zeros(n)

        for t in range(n):
            smoothed_part = 0
            for j in range(1, min(k + 1, t + 1)):
                smoothed_part += theta[j] * true[t - j]

            true[t] = (observed[t] - smoothed_part) / theta_0

        # 限制极端值
        std_obs = np.std(observed)
        true = np.clip(true, -10 * std_obs, 10 * std_obs)

        return true

    def compute_verification_metrics(
        self,
        observed_returns: np.ndarray,
        unsmoothed_returns: np.ndarray,
        asset_id: str = "unknown",
        rf_rate: float = 0.02,
    ) -> UnsmoothingMetrics:
        """计算去平滑前后的验证指标

        包括:
        - Autocorrelation (Lag-1)
        - Variance
        - Sharpe Ratio
        - Adjusted Sharpe Ratio (根据 GLM 论文公式)

        Args:
            observed_returns: 观测收益率
            unsmoothed_returns: 去平滑收益率
            asset_id: 资产标识符
            rf_rate: 年化无风险利率

        Returns:
            UnsmoothingMetrics 数据对象
        """
        theta = self._theta if self._theta is not None else np.array([1.0, 0.0, 0.0])
        theta_0 = float(theta[0])

        # 计算去平滑前指标
        acf_before = self._compute_acf(observed_returns, 1)
        var_before = float(np.var(observed_returns))
        sharpe_before = self._compute_sharpe(observed_returns, rf_rate)

        # 计算去平滑后指标
        acf_after = self._compute_acf(unsmoothed_returns, 1)
        var_after = float(np.var(unsmoothed_returns))
        sharpe_after = self._compute_sharpe(unsmoothed_returns, rf_rate)

        # 计算调整夏普比率 (GLM 公式)
        # Adjusted SR = SR_observed * sqrt(1 + 2*Σρ_j)
        # 其中 ρ_j 是观测收益率的自相关
        acf_full = self._compute_acf(observed_returns, self.max_lag)
        rho_sum = np.sum(acf_full[1:])
        adjustment_factor = np.sqrt(max(1 + 2 * rho_sum, 0.1))
        adjusted_sharpe = sharpe_before * adjustment_factor

        # 判断有效性
        is_valid = theta_0 >= self.theta_0_threshold
        rejection_reason = None if is_valid else f"theta_0={theta_0:.4f} < {self.theta_0_threshold}"

        # Smoothing Index
        smoothing_index = 1 - theta_0 ** 2

        metrics = UnsmoothingMetrics(
            asset_id=asset_id,
            theta=theta,
            theta_0=theta_0,
            smoothing_index=smoothing_index,
            acf_lag1_before=float(acf_before[1]) if len(acf_before) > 1 else 0.0,
            variance_before=var_before,
            sharpe_before=sharpe_before,
            acf_lag1_after=float(acf_after[1]) if len(acf_after) > 1 else 0.0,
            variance_after=var_after,
            sharpe_after=sharpe_after,
            adjusted_sharpe=adjusted_sharpe,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )

        self._metrics = metrics
        return metrics

    def _compute_sharpe(self, returns: np.ndarray, rf_rate: float = 0.02) -> float:
        """计算夏普比率

        Args:
            returns: 收益率序列 (年化 %)
            rf_rate: 年化无风险利率

        Returns:
            Sharpe Ratio
        """
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret < 1e-8:
            return 0.0

        # 假设输入是周频年化收益率
        # 转换为年化夏普 (周频有 52 周)
        excess_return = mean_ret - rf_rate * 100  # rf_rate 是小数，returns 是百分比
        sharpe = excess_return / std_ret

        return float(sharpe)

    def get_smoothing_info(self) -> dict:
        """获取平滑信息

        Returns:
            包含 θ 估计和平滑程度的字典
        """
        if self._theta is None:
            return {"theta": None, "smoothing_index": None, "is_valid": True}

        theta_0 = float(self._theta[0])
        smoothing_index = 1 - theta_0 ** 2

        return {
            "theta": self._theta.tolist(),
            "theta_0": theta_0,
            "smoothing_index": float(smoothing_index),
            "max_lag": self.max_lag,
            "is_valid": theta_0 >= self.theta_0_threshold,
            "threshold": self.theta_0_threshold,
        }

    def get_metrics(self) -> Optional[UnsmoothingMetrics]:
        """获取最近一次计算的验证指标"""
        return self._metrics


def glm_unsmooth(
    observed_returns: np.ndarray, max_lag: int = 2, method: str = "mle"
) -> Tuple[np.ndarray, dict]:
    """便捷函数: GLM 去平滑

    Args:
        observed_returns: 观测收益率 [n_products, n_dates] 或 [n_dates]
        max_lag: 最大滞后阶数
        method: 估计方法

    Returns:
        (true_returns, info)
        - true_returns: 去平滑后的收益率
        - info: 平滑信息
    """
    glm = GLMUnsmoothing(max_lag=max_lag, method=method)
    true_returns = glm.unsmooth(observed_returns)
    info = glm.get_smoothing_info()

    return true_returns, info


# ═════════════════════════════════════════════════════════════════════════════
# Rolling GLM — 解决前视偏差 (Look-ahead Bias)
# ═════════════════════════════════════════════════════════════════════════════

class RollingGLMUnsmoothing(GLMUnsmoothing):
    """滚动窗口 GLM 去平滑处理器 — 避免前视偏差

    关键改进 (hisensho 审计建议):
    - 使用滚动窗口估计 theta，而非全局历史数据
    - 在回测的每个时间点，仅使用该时间点之前的数据
    - 默认窗口: 180 天 (约 26 周)
    """

    def __init__(
        self,
        max_lag: int = 2,
        method: str = "mle",
        theta_0_threshold: float = 0.10,
        strict_filter: bool = True,
        window_size: int = 180,  # 滚动窗口大小 (天)
        min_window: int = 60,    # 最小窗口大小
    ):
        """
        Args:
            max_lag: 最大滞后阶数
            method: 估计方法 ("mle", "acf")
            theta_0_threshold: theta_0 最小阈值
            strict_filter: True=抛出异常, False=仅警告
            window_size: 滚动窗口大小 (天数)
            min_window: 最小窗口大小 (少于此值使用默认 theta)
        """
        super().__init__(max_lag, method, theta_0_threshold, strict_filter)
        self.window_size = window_size
        self.min_window = min_window
        self._theta_history: List[np.ndarray] = []

    def estimate_theta_rolling(
        self,
        observed_returns: np.ndarray,
        current_idx: int,
    ) -> np.ndarray:
        """滚动估计 theta — 仅使用 current_idx 之前的数据

        Args:
            observed_returns: 完整的观测收益率序列
            current_idx: 当前时间点索引 (0-based)

        Returns:
            theta: 基于历史数据估计的平滑系数
        """
        # 确定窗口范围 [start, current_idx)
        start_idx = max(0, current_idx - self.window_size)
        window_data = observed_returns[start_idx:current_idx]

        n = len(window_data)

        # 窗口过小，使用默认值
        if n < self.min_window:
            default_theta = np.array([1.0] + [0.0] * self.max_lag)
            return default_theta

        # 调用父类方法估计 theta
        theta = self.estimate_theta(window_data)
        return theta

    def unsmooth_rolling(
        self,
        observed_returns: np.ndarray,
        asset_id: str = "unknown",
        reestimate_freq: int = 26,  # 每 26 周 (半年) 重新估计
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """滚动去平滑 — 避免前视偏差

        在回测场景中，每隔一段时间重新估计 theta。

        Args:
            observed_returns: 观测收益率序列
            asset_id: 资产标识符
            reestimate_freq: 重新估计频率 (周数)

        Returns:
            (unsmoothed_returns, theta_history)
            - unsmoothed_returns: 去平滑后的收益率
            - theta_history: theta 估计历史
        """
        n = len(observed_returns)

        if n < self.min_window:
            logger.warning(f"数据量过小 ({n} < {self.min_window}), 返回原始数据")
            return observed_returns.copy(), []

        unsmoothed = np.zeros(n)
        theta_history = []
        current_theta = None

        for t in range(n):
            # 决定是否重新估计 theta
            should_reestimate = (
                current_theta is None or  # 首次估计
                (t >= self.min_window and t % reestimate_freq == 0)  # 定期重估
            )

            if should_reestimate and t >= self.min_window:
                try:
                    current_theta = self.estimate_theta_rolling(observed_returns, t)
                    theta_history.append((t, current_theta.copy()))

                    # 检查 theta_0 阈值
                    if current_theta[0] < self.theta_0_threshold:
                        if self.strict_filter:
                            raise IlliquidityException(
                                asset_id, current_theta[0], self.theta_0_threshold
                            )
                        else:
                            logger.warning(
                                f"Rolling GLM: {asset_id} theta_0={current_theta[0]:.4f} "
                                f"< {self.theta_0_threshold} at t={t}"
                            )
                            current_theta = np.array([1.0] + [0.0] * self.max_lag)
                except Exception as e:
                    if isinstance(e, IlliquidityException):
                        raise
                    logger.warning(f"Rolling theta estimation failed at t={t}: {e}")
                    current_theta = np.array([1.0] + [0.0] * self.max_lag)

            # 使用当前 theta 进行去平滑
            if current_theta is None:
                # 还没有足够数据估计，使用原始值
                unsmoothed[t] = observed_returns[t]
            else:
                # 逆滤波: R_t = (R_t^o - Σ θ_j * R_{t-j}) / θ_0
                theta_0 = current_theta[0]
                if theta_0 < 0.1:
                    unsmoothed[t] = observed_returns[t]
                else:
                    smoothed_part = 0.0
                    for j in range(1, min(len(current_theta), t + 1)):
                        smoothed_part += current_theta[j] * unsmoothed[t - j]
                    unsmoothed[t] = (observed_returns[t] - smoothed_part) / theta_0

        # 限制极端值
        std_obs = np.std(observed_returns)
        if std_obs > 1e-8:
            unsmoothed = np.clip(unsmoothed, -10 * std_obs, 10 * std_obs)

        self._theta_history = theta_history
        return unsmoothed, theta_history

    def get_theta_history(self) -> List[Tuple[int, np.ndarray]]:
        """获取 theta 估计历史"""
        return self._theta_history


def rolling_glm_unsmooth(
    observed_returns: np.ndarray,
    max_lag: int = 2,
    window_size: int = 180,
    reestimate_freq: int = 26,
) -> Tuple[np.ndarray, dict]:
    """便捷函数: 滚动 GLM 去平滑 (无前视偏差)

    Args:
        observed_returns: 观测收益率
        max_lag: 最大滞后阶数
        window_size: 滚动窗口大小
        reestimate_freq: 重新估计频率 (周)

    Returns:
        (true_returns, info)
    """
    glm = RollingGLMUnsmoothing(
        max_lag=max_lag,
        window_size=window_size,
        strict_filter=False,  # 滚动模式不抛异常
    )
    true_returns, theta_history = glm.unsmooth_rolling(
        observed_returns,
        reestimate_freq=reestimate_freq,
    )

    info = {
        "window_size": window_size,
        "reestimate_freq": reestimate_freq,
        "n_reestimates": len(theta_history),
        "theta_history": theta_history,
    }

    return true_returns, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    np.random.seed(42)
    n = 200

    # 生成真实收益率
    true_ret = np.random.randn(n) * 2

    # 模拟平滑: R^o = 0.6*R_t + 0.3*R_{t-1} + 0.1*R_{t-2}
    theta_true = np.array([0.6, 0.3, 0.1])
    observed_ret = np.convolve(true_ret, theta_true, mode="same")

    print("真实 θ:", theta_true)
    print("真实收益率 std:", true_ret.std())
    print("观测收益率 std:", observed_ret.std())

    # 去平滑
    glm = GLMUnsmoothing(max_lag=2, method="mle")
    estimated_theta = glm.estimate_theta(observed_ret)
    print("估计 θ:", estimated_theta)

    unsmoothed_ret = glm.unsmooth(observed_ret, estimated_theta)
    print("去平滑收益率 std:", unsmoothed_ret.std())

    # 计算相关性
    corr = np.corrcoef(true_ret, unsmoothed_ret)[0, 1]
    print(f"真实 vs 去平滑 相关性: {corr:.4f}")
