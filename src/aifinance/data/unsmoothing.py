# -*- coding: utf-8 -*-
"""
GLM NAV 去平滑算法

实现 Getmansky-Lo-Makarov (2004) 算法处理银行理财产品净值的平滑问题。

背景:
银行理财产品净值往往被平滑处理,导致:
1. 波动率被低估
2. 真实风险被掩盖
3. 相关性失真

GLM 算法通过分析收益率的自相关结构,还原真实收益率序列。

参考文献:
Getmansky, M., Lo, A. W., & Makarov, I. (2004).
An econometric model of serial correlation and illiquidity in hedge fund returns.
Journal of Financial Economics, 74(3), 529-609.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.signal import deconvolve

logger = logging.getLogger(__name__)


class GLMUnsmoothing:
    """Getmansky-Lo-Makarov 去平滑处理器"""

    def __init__(self, max_lag: int = 2, method: str = "mle"):
        """
        Args:
            max_lag: 最大滞后阶数 (论文建议 k=2)
            method: 估计方法 ("mle", "gmm", "acf")
        """
        self.max_lag = max_lag
        self.method = method
        self._theta = None  # 估计的平滑系数

    def estimate_theta(self, observed_returns: np.ndarray) -> np.ndarray:
        """估计平滑系数 θ

        模型: R_t^o = θ_0 * R_t + θ_1 * R_{t-1} + ... + θ_k * R_{t-k}
        约束: Σθ_i = 1, θ_i ≥ 0

        Args:
            observed_returns: 观测收益率序列 [n_samples]

        Returns:
            theta: 平滑系数 [max_lag + 1]
        """
        n = len(observed_returns)
        if n < 30:
            logger.warning("样本量过小,使用默认 θ")
            return np.array([1.0] + [0.0] * self.max_lag)

        # 计算自相关函数
        acf = self._compute_acf(observed_returns, self.max_lag)

        if self.method == "acf":
            # 简单方法: 通过 ACF 反推 θ
            theta = self._theta_from_acf(acf)
        elif self.method == "mle":
            # MLE 估计
            theta = self._mle_estimate(observed_returns, acf)
        else:
            # GMM 估计
            theta = self._gmm_estimate(observed_returns, acf)

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

    def _mle_estimate(self, returns: np.ndarray, acf: np.ndarray) -> np.ndarray:
        """MLE 估计 θ

        最大化对数似然函数 (假设正态分布)

        Args:
            returns: 观测收益率
            acf: 自相关函数

        Returns:
            theta: MLE 估计的 θ
        """
        k = self.max_lag

        # 初始值
        theta_init = np.ones(k + 1) / (k + 1)

        def neg_log_likelihood(theta_raw):
            # 转换为满足约束的 θ
            theta = np.abs(theta_raw)
            theta = theta / theta.sum()

            # 计算理论 ACF
            theoretical_acf = np.zeros(k + 1)
            for j in range(k + 1):
                for i in range(k + 1 - j):
                    theoretical_acf[j] += theta[i] * theta[i + j]

            # 负对数似然 (简化为 ACF 匹配)
            loss = np.sum((theoretical_acf - acf) ** 2)
            return loss

        result = optimize.minimize(
            neg_log_likelihood,
            theta_init,
            method="L-BFGS-B",
            bounds=[(0, 1) for _ in range(k + 1)],
        )

        theta = np.abs(result.x)
        theta = theta / theta.sum()

        return theta

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
        self, observed_returns: np.ndarray, theta: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """去平滑: 恢复真实收益率

        使用反卷积恢复真实收益率序列

        Args:
            observed_returns: 观测收益率 [n_samples] 或 [n_products, n_dates]
            theta: 平滑系数 (如果为 None,则先估计)

        Returns:
            true_returns: 去平滑后的真实收益率
        """
        # 处理 2D 输入
        if observed_returns.ndim == 2:
            n_products, n_dates = observed_returns.shape
            true_returns = np.zeros_like(observed_returns)

            for p in range(n_products):
                true_returns[p] = self.unsmooth(observed_returns[p], theta)

            return true_returns

        # 1D 处理
        n = len(observed_returns)
        if n < 10:
            return observed_returns.copy()

        # 估计或使用给定的 θ
        if theta is None:
            theta = self.estimate_theta(observed_returns)

        # 检查 θ 的有效性
        if theta[0] < 0.3:
            logger.warning(f"θ_0 = {theta[0]:.3f} 过小,可能导致不稳定,使用原始收益率")
            return observed_returns.copy()

        # 反卷积恢复真实收益率
        try:
            # 使用 Wiener 反卷积的简化版本
            true_returns = self._inverse_filter(observed_returns, theta)
        except Exception as e:
            logger.warning(f"反卷积失败: {e},返回原始收益率")
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

    def get_smoothing_info(self) -> dict:
        """获取平滑信息

        Returns:
            包含 θ 估计和平滑程度的字典
        """
        if self._theta is None:
            return {"theta": None, "smoothing_index": None}

        # Smoothing Index: 1 - θ_0^2
        # 完全平滑: SI = 0, 无平滑: SI = 1
        smoothing_index = 1 - self._theta[0] ** 2

        return {
            "theta": self._theta.tolist(),
            "theta_0": float(self._theta[0]),
            "smoothing_index": float(smoothing_index),
            "max_lag": self.max_lag,
        }


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
