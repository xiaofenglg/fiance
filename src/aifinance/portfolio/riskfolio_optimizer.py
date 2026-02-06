# -*- coding: utf-8 -*-
"""
V12 Phase 4: Riskfolio-Lib Portfolio Optimizer

Advanced portfolio optimization with:
- k=6 cardinality constraint (max 6 assets)
- Ledoit-Wolf covariance shrinkage
- HRP (Hierarchical Risk Parity)
- IVP (Inverse Variance Portfolio) fallback
- Alpha signal integration

hisensho quant audit compliance:
- Ledoit-Wolf shrinkage for robust covariance estimation
- Hard cardinality constraint with proper weight renormalization
- No look-ahead bias in return/covariance estimation

Usage:
    optimizer = RiskfolioOptimizer(max_assets=6, method="HRP")
    weights = optimizer.optimize(returns, expected_returns=alpha_signal)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Optional riskfolio-lib import
try:
    import riskfolio as rp
    _rp_available = True
except ImportError:
    _rp_available = False
    logger.debug("riskfolio-lib not available, using native implementation")


@dataclass
class OptimizationResult:
    """Optimization result"""
    weights: pd.Series  # asset -> weight
    method: str
    cov_method: str
    shrinkage_intensity: float
    risk_contribution: Dict[str, float]
    expected_return: float
    expected_risk: float


def ledoit_wolf_shrinkage(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf covariance shrinkage estimator.

    Implements Ledoit & Wolf (2004) shrinkage to improve covariance estimation
    for small samples or high-dimensional data.

    Formula: Σ_shrunk = δ * F + (1 - δ) * S
    Where:
    - S: sample covariance matrix
    - F: shrinkage target (scaled identity matrix)
    - δ: optimal shrinkage intensity

    hisensho quant audit: This is critical for numerical stability when
    n_samples < n_assets or when assets are highly correlated.

    Args:
        returns: [n_samples, n_assets] return matrix

    Returns:
        (shrunk_cov, shrinkage_intensity)
    """
    n_samples, n_assets = returns.shape

    if n_samples < 2 or n_assets < 1:
        return np.eye(max(n_assets, 1)), 1.0

    # Handle NaN/Inf values - replace with 0 (no return)
    returns_clean = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Check for all-zero columns (assets with no data)
    valid_cols = np.any(returns_clean != 0, axis=0)
    if not np.any(valid_cols):
        logger.warning("[Ledoit-Wolf] All returns are zero/NaN, returning identity")
        return np.eye(n_assets), 1.0

    # Center returns
    X = returns_clean - returns_clean.mean(axis=0)

    # Sample covariance
    sample_cov = np.dot(X.T, X) / n_samples

    # Shrinkage target F (scaled identity)
    trace = np.trace(sample_cov)
    mu = trace / n_assets
    F = mu * np.eye(n_assets)

    # Compute optimal shrinkage intensity using Ledoit-Wolf formula
    delta = sample_cov - F

    # Estimate asymptotic variance of off-diagonal elements
    X2 = X ** 2
    sample_cov2 = np.dot(X2.T, X2) / n_samples
    pi_mat = sample_cov2 - sample_cov ** 2
    pi = np.sum(pi_mat)

    # Diagonal contribution
    rho_diag = np.sum(np.diag(pi_mat))
    rho_off = pi - rho_diag

    # Frobenius norm of delta
    gamma = np.sum(delta ** 2)

    # Optimal shrinkage intensity
    kappa = (pi - rho_off) / gamma if gamma > 1e-10 else 1.0
    shrinkage = max(0, min(1, kappa / n_samples))

    # Apply shrinkage
    shrunk_cov = shrinkage * F + (1 - shrinkage) * sample_cov

    # Ensure positive definite
    try:
        min_eigenvalue = np.min(np.linalg.eigvalsh(shrunk_cov))
        if min_eigenvalue < 1e-8:
            shrunk_cov += (1e-8 - min_eigenvalue) * np.eye(n_assets)
    except np.linalg.LinAlgError as e:
        logger.warning(f"[Ledoit-Wolf] Eigenvalue computation failed: {e}, adding regularization")
        # Fallback: add strong regularization
        shrunk_cov = 0.5 * shrunk_cov + 0.5 * np.diag(np.diag(shrunk_cov))
        shrunk_cov += 1e-6 * np.eye(n_assets)
        shrinkage = 0.5

    return shrunk_cov, shrinkage


def inverse_variance_portfolio(cov: np.ndarray) -> np.ndarray:
    """
    Inverse Variance Portfolio (IVP).

    Simple but robust portfolio where weights are inversely proportional to variance.
    Used as fallback when HRP or MVO fails.

    w_i = (1/σ²_i) / Σ(1/σ²_j)

    Args:
        cov: [n_assets, n_assets] covariance matrix

    Returns:
        [n_assets] weight vector
    """
    variances = np.diag(cov)
    variances = np.maximum(variances, 1e-10)  # Avoid division by zero

    inv_var = 1.0 / variances
    weights = inv_var / np.sum(inv_var)

    return weights


class RiskfolioOptimizer:
    """Riskfolio-Lib Portfolio Optimizer with Cardinality Constraint"""

    def __init__(
        self,
        max_assets: int = 6,
        method: str = "HRP",
        risk_measure: str = "MV",
        cov_method: str = "ledoit_wolf",
        rf_rate: float = 0.02,
        signal_weight: float = 0.3,
    ):
        """
        Args:
            max_assets: Maximum number of assets (k=6 cardinality constraint)
            method: Optimization method ("HRP", "MVO", "CVaR", "IVP")
            risk_measure: Risk measure ("MV", "CVaR", "CDaR")
            cov_method: Covariance estimation method ("ledoit_wolf", "sample")
            rf_rate: Annual risk-free rate
            signal_weight: Weight for alpha signal in expected return estimation
        """
        self.max_assets = max_assets
        self.method = method
        self.risk_measure = risk_measure
        self.cov_method = cov_method
        self.rf_rate = rf_rate
        self.signal_weight = signal_weight

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
    ) -> OptimizationResult:
        """
        Execute portfolio optimization.

        hisensho quant audit:
        - Uses Ledoit-Wolf shrinkage for robust covariance
        - Applies k=6 cardinality constraint post-optimization
        - Properly renormalizes weights after constraint application

        Args:
            returns: [n_samples, n_assets] historical returns DataFrame
            expected_returns: Optional expected returns from alpha model

        Returns:
            OptimizationResult with weights and diagnostics
        """
        if returns.empty or returns.shape[1] < 2:
            logger.warning("[Riskfolio] Insufficient data for optimization")
            return self._equal_weight_result(returns)

        # Clean data
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

        if returns_clean.shape[1] < 2:
            return self._equal_weight_result(returns)

        # Estimate covariance
        cov, shrinkage = self._estimate_covariance(returns_clean.values)

        # Compute expected returns
        mu = self._estimate_expected_returns(returns_clean, expected_returns)

        # Select optimization method
        if _rp_available and self.method in ["HRP", "MVO"]:
            weights = self._optimize_riskfolio(returns_clean, mu, cov)
        elif self.method == "HRP":
            weights = self._optimize_hrp_native(returns_clean, cov)
        elif self.method == "IVP":
            weights = inverse_variance_portfolio(cov)
            weights = pd.Series(weights, index=returns_clean.columns)
        else:
            weights = self._optimize_mvo_native(returns_clean, mu, cov)

        # Apply cardinality constraint (k=6)
        weights = self._apply_cardinality_constraint(weights, self.max_assets)

        # Compute risk contribution
        risk_contrib = self._compute_risk_contribution(weights.values, cov)

        # Compute expected portfolio metrics
        w = weights.values
        expected_return = np.dot(w, mu) if mu is not None else 0.0
        expected_risk = np.sqrt(np.dot(w, np.dot(cov, w)))

        return OptimizationResult(
            weights=weights,
            method=self.method,
            cov_method=self.cov_method,
            shrinkage_intensity=shrinkage,
            risk_contribution={k: v for k, v in zip(weights.index, risk_contrib)},
            expected_return=expected_return,
            expected_risk=expected_risk,
        )

    def optimize_with_signal(
        self,
        returns: pd.DataFrame,
        alpha_signal: pd.Series,
    ) -> OptimizationResult:
        """
        Optimize with alpha signal integration.

        Combines historical expected returns with alpha model predictions using
        Bayesian shrinkage.

        μ_adjusted = (1 - λ) * μ_hist + λ * α_signal

        Args:
            returns: Historical returns
            alpha_signal: Alpha signal (expected return proxy)

        Returns:
            OptimizationResult
        """
        # Align indices
        common_assets = returns.columns.intersection(alpha_signal.index)
        returns_aligned = returns[common_assets]
        signal_aligned = alpha_signal[common_assets]

        # Compute adjusted expected returns
        mu_hist = returns_aligned.mean()

        # Normalize signal to same scale as historical returns
        signal_scaled = (signal_aligned - signal_aligned.mean()) / signal_aligned.std()
        signal_scaled = signal_scaled * mu_hist.std() + mu_hist.mean()

        # Bayesian shrinkage
        mu_adjusted = (1 - self.signal_weight) * mu_hist + self.signal_weight * signal_scaled

        return self.optimize(returns_aligned, expected_returns=mu_adjusted)

    def _estimate_covariance(
        self,
        returns: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Estimate covariance matrix"""
        if self.cov_method == "ledoit_wolf":
            return ledoit_wolf_shrinkage(returns)
        else:
            # Sample covariance
            cov = np.cov(returns, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            return cov, 0.0

    def _estimate_expected_returns(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series],
    ) -> np.ndarray:
        """Estimate expected returns"""
        if expected_returns is not None:
            # Align and fill missing
            mu = expected_returns.reindex(returns.columns).fillna(returns.mean())
            return mu.values

        # Historical mean
        return returns.mean().values

    def _optimize_riskfolio(
        self,
        returns: pd.DataFrame,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> pd.Series:
        """Optimize using riskfolio-lib"""
        try:
            if self.method == "HRP":
                # HRP requires HCPortfolio, not Portfolio
                port = rp.HCPortfolio(returns=returns)
                w = port.optimization(
                    model="HRP",
                    codependence="pearson",
                    method_cov="ledoit" if self.cov_method == "ledoit_wolf" else "hist",
                    rm=self.risk_measure,
                    rf=self.rf_rate / 252,
                    linkage="ward",
                    leaf_order=True,
                )
            else:
                # MVO uses Portfolio
                port = rp.Portfolio(returns=returns)
                port.mu = pd.Series(mu, index=returns.columns)
                port.cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
                w = port.optimization(
                    model="Classic",
                    rm=self.risk_measure,
                    obj="Sharpe",
                    rf=self.rf_rate / 252,
                    l=0,
                )

            if w is None or w.empty:
                raise ValueError("Optimization returned empty weights")

            return w.squeeze()

        except Exception as e:
            logger.warning(f"[Riskfolio] riskfolio optimization failed: {e}")
            # Fallback to IVP
            weights = inverse_variance_portfolio(cov)
            return pd.Series(weights, index=returns.columns)

    def _optimize_hrp_native(
        self,
        returns: pd.DataFrame,
        cov: np.ndarray,
    ) -> pd.Series:
        """Native HRP implementation"""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        n_assets = returns.shape[1]
        assets = returns.columns.tolist()

        # Correlation from covariance
        std = np.sqrt(np.diag(cov))
        std = np.maximum(std, 1e-8)
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        # Distance matrix
        dist = np.sqrt(0.5 * np.clip(1 - corr, 0, 2))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method="ward")

        # Quasi-diagonalization
        sort_ix = self._get_quasi_diag(link)

        # Recursive bisection
        weights = self._recursive_bisection(cov, sort_ix)

        return pd.Series(weights, index=assets)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal order from linkage matrix"""
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
        self,
        cov: np.ndarray,
        sort_ix: List[int],
    ) -> np.ndarray:
        """Recursive bisection for HRP weights"""
        n = len(sort_ix)
        w = np.ones(n)

        clusters = [sort_ix]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = self._cluster_variance(cov, left)
                var_right = self._cluster_variance(cov, right)

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

        return w / w.sum()

    def _cluster_variance(self, cov: np.ndarray, indices: List[int]) -> float:
        """Compute cluster variance using IVP weights"""
        sub_cov = cov[np.ix_(indices, indices)]
        inv_diag = 1 / (np.diag(sub_cov) + 1e-10)
        inv_diag = inv_diag / inv_diag.sum()
        return np.dot(inv_diag, np.dot(sub_cov, inv_diag))

    def _optimize_mvo_native(
        self,
        returns: pd.DataFrame,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> pd.Series:
        """Native MVO (Max Sharpe) implementation"""
        from scipy.optimize import minimize

        n_assets = len(mu)
        rf = self.rf_rate / 252

        def neg_sharpe(w):
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
            return -(port_ret - rf) / (port_vol + 1e-10)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        init_weights = np.ones(n_assets) / n_assets

        result = minimize(
            neg_sharpe,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=returns.columns)
        else:
            # Fallback to IVP
            weights = inverse_variance_portfolio(cov)
            return pd.Series(weights, index=returns.columns)

    def _apply_cardinality_constraint(
        self,
        weights: pd.Series,
        k: int,
    ) -> pd.Series:
        """
        Apply k-cardinality constraint: only keep top-k assets.

        hisensho quant audit: Hard constraint implementation.
        After selecting top-k, properly renormalize to sum to 1.

        Args:
            weights: Original weights
            k: Maximum number of assets

        Returns:
            Constrained weights
        """
        if len(weights) <= k:
            # No constraint needed
            return weights / weights.sum()

        # Get top-k by weight
        top_k = weights.nlargest(k)

        # Zero out others
        result = pd.Series(0.0, index=weights.index)
        result.loc[top_k.index] = top_k.values

        # Renormalize
        total = result.sum()
        if total > 1e-10:
            result = result / total
        else:
            # Fallback to equal weight
            result.loc[top_k.index] = 1.0 / k

        logger.debug(
            f"[Riskfolio] Applied k={k} constraint: "
            f"{len(weights)} -> {(result > 0).sum()} assets"
        )

        return result

    def _compute_risk_contribution(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Compute marginal risk contribution"""
        port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))

        if port_vol < 1e-10:
            return np.ones(len(weights)) / len(weights)

        marginal_risk = np.dot(cov, weights) / port_vol
        risk_contrib = weights * marginal_risk
        risk_contrib = risk_contrib / risk_contrib.sum()

        return risk_contrib

    def _equal_weight_result(
        self,
        returns: pd.DataFrame,
    ) -> OptimizationResult:
        """Equal weight fallback"""
        n_assets = returns.shape[1] if not returns.empty else 1
        n_assets = min(n_assets, self.max_assets)

        if returns.empty:
            weights = pd.Series([1.0])
        else:
            weights = pd.Series(1.0 / n_assets, index=returns.columns[:n_assets])

        return OptimizationResult(
            weights=weights,
            method="equal_weight",
            cov_method=self.cov_method,
            shrinkage_intensity=0.0,
            risk_contribution={k: 1.0 / n_assets for k in weights.index},
            expected_return=0.0,
            expected_risk=0.0,
        )


def create_riskfolio_optimizer(
    max_assets: int = 6,
    method: str = "HRP",
    **kwargs,
) -> RiskfolioOptimizer:
    """
    Convenience function to create RiskfolioOptimizer.

    Args:
        max_assets: Maximum assets (k=6)
        method: Optimization method
        **kwargs: Additional arguments

    Returns:
        RiskfolioOptimizer instance
    """
    return RiskfolioOptimizer(max_assets=max_assets, method=method, **kwargs)


def check_riskfolio_available() -> bool:
    """Check if riskfolio-lib is available"""
    return _rp_available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic data
    np.random.seed(42)
    n_assets, n_samples = 20, 200

    # Generate synthetic returns
    returns_data = np.random.randn(n_samples, n_assets) * 0.02 + 0.0003
    returns = pd.DataFrame(
        returns_data,
        columns=[f"A{i:02d}" for i in range(n_assets)]
    )

    # Create optimizer
    optimizer = RiskfolioOptimizer(max_assets=6, method="HRP")

    # Run optimization
    result = optimizer.optimize(returns)

    print("\n=== Optimization Results ===")
    print(f"Method: {result.method}")
    print(f"Covariance method: {result.cov_method}")
    print(f"Shrinkage intensity: {result.shrinkage_intensity:.4f}")
    print(f"Expected return: {result.expected_return:.4%}")
    print(f"Expected risk: {result.expected_risk:.4%}")
    print(f"\nWeights (top 6):")
    for asset, weight in result.weights.nlargest(10).items():
        if weight > 0:
            print(f"  {asset}: {weight:.4f}")
    print(f"\nWeight sum: {result.weights.sum():.6f}")
    print(f"Non-zero weights: {(result.weights > 0).sum()}")
