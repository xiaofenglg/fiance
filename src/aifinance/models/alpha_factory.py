# -*- coding: utf-8 -*-
"""
V12 Phase 2: Alpha Factory

Qlib-style factor generation and alpha signal computation.

Features:
- Qlib expression-based factor computation
- IC (Information Coefficient) analysis
- Factor orthogonalization
- LightGBM/TFT model training for alpha prediction

Factors:
- Momentum: 5/21/63 day returns
- Volatility: 20-day rolling std
- Sharpe: 60-day rolling Sharpe ratio
- Skew/Kurt: 20-day higher moments
- MaxDD: 60-day max drawdown distance
- RSI/MACD: Technical indicators

hisensho quant audit compliance:
- Walk-forward training to avoid look-ahead bias
- Proper label shifting (forward_returns.shift(-N))
- Rank normalization for cross-sectional signals
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Optional LightGBM import
try:
    import lightgbm as lgb
    _lgb_available = True
except ImportError:
    _lgb_available = False
    logger.debug("LightGBM not available")


@dataclass
class FactorConfig:
    """Factor computation configuration"""
    name: str
    window: int
    expression: str  # Qlib-style expression
    category: str  # momentum, volatility, quality, etc.


@dataclass
class ICResult:
    """Information Coefficient analysis result"""
    factor_name: str
    ic_mean: float
    ic_std: float
    ir: float  # IC / IC_std
    ic_series: pd.Series
    t_stat: float
    p_value: float


@dataclass
class AlphaSignal:
    """Alpha signal output"""
    date: str
    signals: Dict[str, float]  # product_code -> signal strength
    raw_scores: np.ndarray
    rank_scores: np.ndarray  # Rank normalized [0, 1]


class AlphaFactory:
    """Qlib-style Alpha Factor Factory"""

    # Default factor definitions (Qlib expression style)
    # hisensho approved: Extended from 10 to 20 factors for improved IC
    # hisensho revision (2026-02-06): Disabled window > 40 factors due to insufficient data (171 days)
    DEFAULT_FACTORS = [
        # === Momentum factors (multi-timeframe) ===
        FactorConfig("Momentum_5", 5, "Ref($close, 5) / $close - 1", "momentum"),
        FactorConfig("Momentum_10", 10, "Ref($close, 10) / $close - 1", "momentum"),
        FactorConfig("Momentum_21", 21, "Ref($close, 21) / $close - 1", "momentum"),
        # FactorConfig("Momentum_63", 63, "Ref($close, 63) / $close - 1", "momentum"),  # Disabled: window > 40
        # FactorConfig("Momentum_126", 126, "Ref($close, 126) / $close - 1", "momentum"),  # Disabled: window > 40, only 35 samples

        # === Reversal factor (short-term) ===
        FactorConfig("Reversal_5", 5, "-(Ref($close, 5) / $close - 1)", "reversal"),

        # === Volatility factors (multi-timeframe) ===
        FactorConfig("Vol_10", 10, "Std($return, 10)", "volatility"),
        FactorConfig("Vol_20", 20, "Std($return, 20)", "volatility"),
        # FactorConfig("Vol_60", 60, "Std($return, 60)", "volatility"),  # Disabled: window > 40

        # === Quality factors ===
        FactorConfig("Sharpe_30", 30, "Mean($return, 30) / Std($return, 30)", "quality"),
        # FactorConfig("Sharpe_60", 60, "Mean($return, 60) / Std($return, 60)", "quality"),  # Disabled: window > 40
        FactorConfig("Skew_20", 20, "Skew($return, 20)", "quality"),
        FactorConfig("Kurt_20", 20, "Kurt($return, 20)", "quality"),
        # FactorConfig("Sortino_60", 60, "Mean($return, 60) / DownsideStd($return, 60)", "quality"),  # Disabled: window > 40

        # === Risk factors ===
        FactorConfig("MaxDD_30", 30, "Max($close, 30) - $close", "risk"),
        # FactorConfig("MaxDD_60", 60, "Max($close, 60) - $close", "risk"),  # Disabled: window > 40

        # === Technical indicators ===
        FactorConfig("RSI_14", 14, "SMA(Max($return,0),14)/SMA(Abs($return),14)*100", "technical"),
        FactorConfig("MACD", 26, "EMA($close,12) - EMA($close,26)", "technical"),
        FactorConfig("BB_Width", 20, "2*Std($close,20) / Mean($close,20)", "technical"),
        FactorConfig("ATR_14", 14, "Mean(TrueRange, 14)", "technical"),
    ]

    def __init__(
        self,
        qlib_data_dir: Optional[str] = None,
        factors: Optional[List[FactorConfig]] = None,
        ic_threshold: float = 0.05,  # hisensho approved: 0.03 → 0.05 (stricter)
        train_window: int = 252,
        predict_horizon: int = 10,  # hisensho approved: 5 → 10 (lower turnover)
    ):
        """
        Args:
            qlib_data_dir: Qlib data directory (optional, for future integration)
            factors: List of factor configs (default: DEFAULT_FACTORS)
            ic_threshold: Minimum IC threshold for factor selection (raised to 0.05)
            train_window: Training window (days)
            predict_horizon: Prediction horizon (days, extended to 10 for stability)
        """
        self.qlib_data_dir = Path(qlib_data_dir).expanduser() if qlib_data_dir else None
        self.factors = factors or self.DEFAULT_FACTORS
        self.ic_threshold = ic_threshold
        self.train_window = train_window
        self.predict_horizon = predict_horizon

        # Model storage
        self._model = None
        self._selected_factors = []
        self._factor_weights = {}

    def compute_factors(
        self,
        nav_matrix: np.ndarray,
        dates: List[str],
        product_codes: List[str],
    ) -> pd.DataFrame:
        """
        Compute all factors from NAV matrix.

        Args:
            nav_matrix: [n_products, n_dates] NAV matrix
            dates: List of dates (YYYY-MM-DD)
            product_codes: List of product codes

        Returns:
            DataFrame with MultiIndex (date, product) and factor columns
        """
        n_products, n_dates = nav_matrix.shape
        logger.info(f"[AlphaFactory] Computing factors: {n_products} products, {n_dates} dates")

        # Calculate returns with safeguards against zero NAV
        returns = np.zeros_like(nav_matrix, dtype=np.float32)
        prev_nav = nav_matrix[:, :-1]
        # Replace zeros and very small values to avoid div by zero
        safe_prev = np.where(prev_nav > 1e-6, prev_nav, np.nan)
        returns[:, 1:] = nav_matrix[:, 1:] / safe_prev - 1
        # Clip extreme returns (>100% daily change is likely data error)
        returns = np.clip(returns, -0.5, 0.5)

        # Compute each factor
        factor_data = {}

        for factor_cfg in self.factors:
            logger.debug(f"[AlphaFactory] Computing {factor_cfg.name}...")
            factor_values = self._compute_single_factor(
                nav_matrix, returns, factor_cfg
            )
            factor_data[factor_cfg.name] = factor_values

        # Build DataFrame with MultiIndex
        records = []
        for t in range(n_dates):
            date = dates[t]
            for p in range(n_products):
                row = {
                    "date": date,
                    "product_code": product_codes[p],
                }
                for factor_name, values in factor_data.items():
                    row[factor_name] = values[p, t]
                records.append(row)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "product_code"])

        logger.info(f"[AlphaFactory] Computed {len(self.factors)} factors")
        return df

    def _compute_single_factor(
        self,
        nav: np.ndarray,
        returns: np.ndarray,
        config: FactorConfig,
    ) -> np.ndarray:
        """Compute a single factor based on config"""
        n_products, n_dates = nav.shape
        window = config.window
        result = np.full((n_products, n_dates), np.nan, dtype=np.float32)

        if config.name.startswith("Momentum_"):
            # Momentum: (NAV[t] / NAV[t-window]) - 1
            if n_dates > window:
                base_nav = nav[:, :-window]
                # Safeguard against zero/small NAV values
                safe_base = np.where(base_nav > 1e-6, base_nav, np.nan)
                result[:, window:] = nav[:, window:] / safe_base - 1
                # Clip extreme momentum values
                result = np.clip(result, -5, 5)

        elif config.name.startswith("Vol_"):
            # Volatility: rolling std
            result = self._rolling_std(returns, window)

        elif config.name.startswith("Sharpe_"):
            # Sharpe: mean / std
            mean = self._rolling_mean(returns, window)
            std = self._rolling_std(returns, window)
            safe_std = np.maximum(std, 1e-8)
            result = mean / safe_std

        elif config.name.startswith("Skew_"):
            # Skewness
            result = self._rolling_skew(returns, window)

        elif config.name.startswith("Kurt_"):
            # Kurtosis
            result = self._rolling_kurt(returns, window)

        elif config.name.startswith("MaxDD_"):
            # Max drawdown distance
            result = self._rolling_max_drawdown(nav, window)

        elif config.name.startswith("RSI_"):
            # RSI
            result = self._compute_rsi(returns, window)

        elif config.name == "MACD":
            # MACD
            result = self._compute_macd(nav)

        elif config.name.startswith("Reversal_"):
            # Reversal: negative momentum (mean reversion)
            if n_dates > window:
                base_nav = nav[:, :-window]
                safe_base = np.where(base_nav > 1e-6, base_nav, np.nan)
                result[:, window:] = -(nav[:, window:] / safe_base - 1)
                result = np.clip(result, -5, 5)

        elif config.name.startswith("Sortino_"):
            # Sortino ratio (downside risk adjusted)
            result = self._compute_sortino(returns, window)

        elif config.name == "BB_Width":
            # Bollinger Bands Width
            result = self._compute_bb_width(nav, window)

        elif config.name.startswith("ATR_"):
            # Average True Range
            result = self._compute_atr(nav, window)

        # hisensho suggestion: Clip extreme values (especially for Sortino)
        result = np.clip(result, -10, 10)
        return result

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean using cumsum trick"""
        n_p, n_d = data.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        cs = np.cumsum(np.nan_to_num(data, 0), axis=1)
        result[:, window - 1:] = (cs[:, window - 1:] - np.concatenate([
            np.zeros((n_p, 1)), cs[:, :-window]
        ], axis=1)) / window

        return result

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation"""
        n_p, n_d = data.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        # Use variance = E[X^2] - (E[X])^2
        data_clean = np.nan_to_num(data, 0)
        mean = self._rolling_mean(data_clean, window)
        mean_sq = self._rolling_mean(data_clean ** 2, window)
        var = np.maximum(mean_sq - mean ** 2, 0)
        result = np.sqrt(var)

        return result

    def _rolling_skew(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling skewness (Vectorized using pandas for performance)

        hisensho fix: Replaced Python double loop with pandas vectorized rolling.
        Performance: O(n) instead of O(n*p*window) scipy calls.
        """
        n_p, n_d = data.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        # Use pandas rolling - much faster than Python loops
        df = pd.DataFrame(data.T)  # transpose to (n_dates, n_products)
        result = df.rolling(window=window, min_periods=window // 2).skew().T.values.astype(np.float32)

        return result

    def _rolling_kurt(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling kurtosis (Vectorized using pandas for performance)

        hisensho fix: Replaced Python double loop with pandas vectorized rolling.
        Performance: O(n) instead of O(n*p*window) scipy calls.
        """
        n_p, n_d = data.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        # Use pandas rolling - much faster than Python loops
        df = pd.DataFrame(data.T)  # transpose to (n_dates, n_products)
        result = df.rolling(window=window, min_periods=window // 2).kurt().T.values.astype(np.float32)

        return result

    def _rolling_max_drawdown(self, nav: np.ndarray, window: int) -> np.ndarray:
        """Rolling max drawdown distance (from rolling max)"""
        n_p, n_d = nav.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        for t in range(window - 1, n_d):
            window_nav = nav[:, t - window + 1:t + 1]
            rolling_max = np.nanmax(window_nav, axis=1)
            current = nav[:, t]
            result[:, t] = (rolling_max - current) / rolling_max

        return result

    def _compute_rsi(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Relative Strength Index"""
        n_p, n_d = returns.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        gains = np.maximum(returns, 0)
        losses = np.maximum(-returns, 0)

        avg_gain = self._rolling_mean(gains, window)
        avg_loss = self._rolling_mean(losses, window)

        safe_loss = np.maximum(avg_loss, 1e-10)
        rs = avg_gain / safe_loss
        result = 100 - 100 / (1 + rs)

        return result

    def _compute_macd(self, nav: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """MACD (EMA12 - EMA26)"""
        n_p, n_d = nav.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < slow:
            return result

        ema_fast = self._ema(nav, fast)
        ema_slow = self._ema(nav, slow)

        result = ema_fast - ema_slow
        return result

    def _ema(self, data: np.ndarray, span: int) -> np.ndarray:
        """Exponential Moving Average (Vectorized using pandas)

        hisensho suggestion: Use pandas ewm for better performance with large datasets.
        Pandas uses C implementation under the hood.
        """
        df = pd.DataFrame(data.T)  # transpose to (n_dates, n_products)
        result = df.ewm(span=span, adjust=False).mean().T.values.astype(np.float32)
        return result

    def _compute_sortino(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Sortino Ratio: Mean / Downside_Std

        hisensho approved: Only considers negative returns for risk calculation.
        More suitable for non-normal distributions common in financial data.

        Args:
            returns: [n_products, n_dates] return matrix
            window: Rolling window size

        Returns:
            Sortino ratio array
        """
        n_p, n_d = returns.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        mean = self._rolling_mean(returns, window)

        # Downside deviation: only calculate std of negative returns
        downside_returns = np.where(returns < 0, returns, 0)
        downside_var = self._rolling_mean(downside_returns ** 2, window)
        downside_std = np.sqrt(np.maximum(downside_var, 0))

        # Prevent division by zero
        safe_std = np.maximum(downside_std, 1e-8)
        result = mean / safe_std

        return result

    def _compute_bb_width(self, nav: np.ndarray, window: int) -> np.ndarray:
        """Bollinger Bands Width: (Upper - Lower) / Middle = (4 * Std) / Mean

        hisensho corrected: Standard formula is (Mean + 2Std) - (Mean - 2Std) / Mean = 4Std / Mean
        Captures volatility squeeze and expansion patterns.

        Args:
            nav: [n_products, n_dates] NAV matrix
            window: Rolling window size

        Returns:
            BB width array
        """
        n_p, n_d = nav.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window:
            return result

        mean = self._rolling_mean(nav, window)
        std = self._rolling_std(nav, window)

        # Prevent division by zero
        safe_mean = np.maximum(mean, 1e-8)
        # hisensho fix: Standard BB Width = (Upper - Lower) / Middle = 4 * Std / Mean
        result = 4 * std / safe_mean

        return result

    def _compute_atr(self, nav: np.ndarray, window: int) -> np.ndarray:
        """Average True Range (simplified for NAV data)

        hisensho approved: For NAV data (only close price), use abs(close[t] - close[t-1])
        as proxy for True Range.

        Args:
            nav: [n_products, n_dates] NAV matrix
            window: Rolling window size

        Returns:
            ATR array
        """
        n_p, n_d = nav.shape
        result = np.full((n_p, n_d), np.nan, dtype=np.float32)

        if n_d < window + 1:
            return result

        # True range proxy: abs(close[t] - close[t-1])
        true_range = np.abs(nav[:, 1:] - nav[:, :-1])

        # Pad with NaN for first element to maintain shape
        true_range = np.concatenate([
            np.full((n_p, 1), np.nan),
            true_range
        ], axis=1)

        # Rolling average of true range
        result = self._rolling_mean(true_range, window)

        return result

    def rank_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional rank normalization.

        For each date, rank factors across products and normalize to [0, 1].
        This removes outliers and makes factors comparable.

        Args:
            df: Factor DataFrame with MultiIndex (date, product)

        Returns:
            Rank-normalized DataFrame
        """
        factor_cols = [col for col in df.columns if col not in ["date", "product_code"]]

        # Group by date and rank
        result = df.copy()
        for col in factor_cols:
            result[col] = df.groupby(level="date")[col].rank(pct=True, method="average")

        logger.info(f"[AlphaFactory] Rank normalized {len(factor_cols)} factors")
        return result

    def compute_ic(
        self,
        factors_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        horizon: int = 5,
    ) -> Dict[str, ICResult]:
        """
        Compute Information Coefficient for each factor.

        IC = Correlation(factor[t], forward_return[t+horizon])

        hisensho audit: Uses proper forward return shifting to avoid look-ahead bias.

        Args:
            factors_df: Factor DataFrame with MultiIndex (date, product)
            returns_df: Forward returns DataFrame
            horizon: Prediction horizon

        Returns:
            Dict of ICResult per factor
        """
        factor_cols = [col for col in factors_df.columns if col not in ["date", "product_code"]]
        results = {}

        for factor_name in factor_cols:
            ic_series = []

            # Group by date
            dates = factors_df.index.get_level_values("date").unique()

            for i, date in enumerate(dates[:-horizon]):
                # Get factor values at t
                if date not in factors_df.index.get_level_values("date"):
                    continue

                factor_at_t = factors_df.loc[date, factor_name]

                # hisensho CRITICAL FIX: returns_df already contains forward returns
                # aligned at each date. The forward return at 'date' IS the return
                # from date to date+horizon. Do NOT shift by horizon again!
                #
                # Before (WRONG): used dates[i + horizon] -> double shift
                # After (CORRECT): use current date directly
                if date not in returns_df.index.get_level_values("date"):
                    continue

                returns_at_date = returns_df.loc[date, "return"] if "return" in returns_df.columns else returns_df.loc[date].values

                # Align products
                if isinstance(factor_at_t, pd.Series) and isinstance(returns_at_date, pd.Series):
                    common_products = factor_at_t.index.intersection(returns_at_date.index)
                    if len(common_products) < 10:
                        continue

                    f = factor_at_t.loc[common_products].values
                    r = returns_at_date.loc[common_products].values

                    # Compute rank IC (Spearman)
                    valid = ~(np.isnan(f) | np.isnan(r))
                    if valid.sum() < 10:
                        continue

                    ic, _ = stats.spearmanr(f[valid], r[valid])
                    if not np.isnan(ic):
                        ic_series.append(ic)

            if not ic_series:
                continue

            ic_arr = np.array(ic_series)
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr)
            ir = ic_mean / ic_std if ic_std > 0 else 0

            # T-test
            t_stat, p_value = stats.ttest_1samp(ic_arr, 0)

            results[factor_name] = ICResult(
                factor_name=factor_name,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ir=ir,
                ic_series=pd.Series(ic_arr),
                t_stat=t_stat,
                p_value=p_value,
            )

            logger.info(
                f"[AlphaFactory] {factor_name}: IC={ic_mean:.4f}, IR={ir:.4f}, "
                f"t={t_stat:.2f}, p={p_value:.4f}"
            )

        return results

    def select_factors(
        self,
        ic_results: Dict[str, ICResult],
        max_corr: float = 0.7,
    ) -> List[str]:
        """
        Select effective factors based on IC and correlation.

        Args:
            ic_results: IC analysis results
            max_corr: Maximum allowed correlation between selected factors

        Returns:
            List of selected factor names
        """
        # Filter by IC threshold (hisensho fix: restore abs() - negative IC factors are valuable too)
        candidates = [
            name for name, result in ic_results.items()
            if abs(result.ic_mean) >= self.ic_threshold and result.p_value < 0.1
        ]

        if not candidates:
            logger.warning("[AlphaFactory] No factors passed IC threshold")
            return list(ic_results.keys())[:5]  # Fallback to top 5

        # Sort by |IR| (absolute value since negative IC factors are valid with sign flip)
        candidates.sort(key=lambda x: abs(ic_results[x].ir), reverse=True)

        logger.info(f"[AlphaFactory] Selected {len(candidates)} factors: {candidates}")
        self._selected_factors = candidates
        return candidates

    def train_model(
        self,
        factors_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        model_type: str = "lightgbm",
        selected_factors: Optional[List[str]] = None,
    ) -> Any:
        """
        Train alpha prediction model.

        hisensho audit: Uses walk-forward training.
        - Training data: [0, train_end]
        - Labels: Forward returns at [train_end + horizon]

        Args:
            factors_df: Factor DataFrame
            returns_df: Forward returns DataFrame
            model_type: Model type ("lightgbm", "linear")
            selected_factors: Factors to use (default: all)

        Returns:
            Trained model
        """
        if model_type == "lightgbm" and not _lgb_available:
            logger.warning("[AlphaFactory] LightGBM not available, using linear")
            model_type = "linear"

        selected_factors = selected_factors or self._selected_factors or list(factors_df.columns)

        # Prepare training data
        X = factors_df[selected_factors].values
        y = returns_df["return"].values if "return" in returns_df.columns else returns_df.values.flatten()

        # Remove NaN rows
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid]
        y_clean = y[valid]

        if len(X_clean) < 100:
            logger.warning("[AlphaFactory] Insufficient training data")
            return None

        if model_type == "lightgbm":
            # hisensho approved: Optimized params for financial time series
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 15,  # Reduced 31 → 15 to prevent overfitting
                "learning_rate": 0.02,  # Reduced 0.05 → 0.02 for stability
                "feature_fraction": 0.7,  # Reduced 0.8 → 0.7 for generalization
                "bagging_fraction": 0.7,  # Reduced 0.8 → 0.7
                "bagging_freq": 5,
                "min_child_samples": 20,  # New: minimum samples per leaf
                "reg_alpha": 0.1,  # New: L1 regularization
                "reg_lambda": 0.1,  # New: L2 regularization
                "seed": 42,  # New: reproducibility
                "verbose": -1,
            }

            train_data = lgb.Dataset(X_clean, label=y_clean)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,  # Increased 100 → 200
                valid_sets=[train_data],
            )

            # Store feature importance
            self._factor_weights = dict(zip(
                selected_factors,
                model.feature_importance("gain") / model.feature_importance("gain").sum()
            ))

        else:
            # Simple linear model
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_clean, y_clean)

            self._factor_weights = dict(zip(
                selected_factors,
                np.abs(model.coef_) / np.abs(model.coef_).sum()
            ))

        self._model = model
        logger.info(f"[AlphaFactory] Trained {model_type} model on {len(X_clean)} samples")
        return model

    def predict_signal(
        self,
        factors_df: pd.DataFrame,
        date: Optional[str] = None,
        selected_factors: Optional[List[str]] = None,
    ) -> AlphaSignal:
        """
        Generate alpha signals for prediction.

        Args:
            factors_df: Factor DataFrame
            date: Prediction date (default: latest)
            selected_factors: Factors to use

        Returns:
            AlphaSignal with signals per product
        """
        if self._model is None:
            logger.warning("[AlphaFactory] Model not trained, using factor average")
            return self._fallback_signal(factors_df, date)

        selected_factors = selected_factors or self._selected_factors

        # Get factor values at date
        if date:
            date_dt = pd.to_datetime(date)
            if date_dt in factors_df.index.get_level_values("date"):
                X = factors_df.loc[date_dt, selected_factors].values
                product_codes = factors_df.loc[date_dt].index.tolist()
            else:
                return self._fallback_signal(factors_df, date)
        else:
            # Latest date
            latest_date = factors_df.index.get_level_values("date").max()
            X = factors_df.loc[latest_date, selected_factors].values
            product_codes = factors_df.loc[latest_date].index.tolist()
            date = latest_date.strftime("%Y-%m-%d")

        # Handle NaN
        X_clean = np.nan_to_num(X, 0)

        # Predict
        if hasattr(self._model, "predict"):
            raw_scores = self._model.predict(X_clean)
        else:
            raw_scores = X_clean.mean(axis=1)

        # Rank normalize to [0, 1]
        rank_scores = stats.rankdata(raw_scores) / len(raw_scores)

        # Build signal dict
        signals = dict(zip(product_codes, rank_scores))

        return AlphaSignal(
            date=date,
            signals=signals,
            raw_scores=raw_scores,
            rank_scores=rank_scores,
        )

    def _fallback_signal(
        self,
        factors_df: pd.DataFrame,
        date: Optional[str],
    ) -> AlphaSignal:
        """Fallback signal using simple factor average"""
        if date:
            date_dt = pd.to_datetime(date)
            if date_dt in factors_df.index.get_level_values("date"):
                X = factors_df.loc[date_dt].values
                product_codes = factors_df.loc[date_dt].index.tolist()
            else:
                return AlphaSignal(date=date, signals={}, raw_scores=np.array([]), rank_scores=np.array([]))
        else:
            latest_date = factors_df.index.get_level_values("date").max()
            X = factors_df.loc[latest_date].values
            product_codes = factors_df.loc[latest_date].index.tolist()
            date = latest_date.strftime("%Y-%m-%d")

        # Simple average
        raw_scores = np.nanmean(X, axis=1)
        rank_scores = stats.rankdata(np.nan_to_num(raw_scores, 0)) / len(raw_scores)

        return AlphaSignal(
            date=date,
            signals=dict(zip(product_codes, rank_scores)),
            raw_scores=raw_scores,
            rank_scores=rank_scores,
        )

    def generate_signals(
        self,
        nav_matrix: np.ndarray,
        dates: List[str],
        product_codes: List[str],
        train_end_idx: Optional[int] = None,
        walk_forward: bool = True,
        retrain_freq: int = 21,  # Retrain every 21 trading days (~monthly)
    ) -> Dict[str, AlphaSignal]:
        """
        Full pipeline: compute factors, train model, generate signals.

        hisensho CRITICAL FIX: Implements walk-forward training to avoid look-ahead bias.

        Walk-forward process:
        1. At time T, train model using data from [0, T-horizon]
        2. Predict signal for time T using factor values at T
        3. Move to T+1 and repeat (retrain every retrain_freq days)

        This ensures NO future data is used for training at any prediction point.

        Args:
            nav_matrix: [n_products, n_dates] NAV matrix
            dates: Date list
            product_codes: Product codes
            train_end_idx: Training end index (deprecated, use walk_forward)
            walk_forward: Enable walk-forward training (default: True)
            retrain_freq: Days between model retraining (default: 21)

        Returns:
            Dict of date -> AlphaSignal
        """
        # Compute factors
        factors_df = self.compute_factors(nav_matrix, dates, product_codes)
        factors_df = self.rank_normalize(factors_df)

        # Compute forward returns for IC analysis
        # NOTE: forward_returns[t] = nav[t+horizon] / nav[t] - 1
        # This is the LABEL for predicting at time t
        n_products, n_dates = nav_matrix.shape
        forward_returns = np.zeros((n_products, n_dates), dtype=np.float32)
        forward_returns[:, :-self.predict_horizon] = (
            nav_matrix[:, self.predict_horizon:] / nav_matrix[:, :-self.predict_horizon] - 1
        )

        # Build returns DataFrame
        returns_records = []
        for t in range(n_dates):
            for p in range(n_products):
                returns_records.append({
                    "date": dates[t],
                    "product_code": product_codes[p],
                    "return": forward_returns[p, t],
                })

        returns_df = pd.DataFrame(returns_records)
        returns_df["date"] = pd.to_datetime(returns_df["date"])
        returns_df = returns_df.set_index(["date", "product_code"])

        # IC analysis on training portion only (avoid lookahead)
        train_cutoff = self.train_window
        train_dates = dates[:train_cutoff]

        train_factors = factors_df[
            factors_df.index.get_level_values("date") <= pd.to_datetime(train_dates[-1])
        ]
        train_returns = returns_df[
            returns_df.index.get_level_values("date") <= pd.to_datetime(train_dates[-1])
        ]

        ic_results = self.compute_ic(train_factors, train_returns, self.predict_horizon)

        # Select factors based on IC (from training period only)
        selected = self.select_factors(ic_results)

        # Generate signals with walk-forward training
        signals = {}
        last_train_idx = 0

        for i, date in enumerate(dates[self.train_window:], start=self.train_window):
            # Check if we need to retrain
            need_retrain = (
                self._model is None or
                (walk_forward and (i - last_train_idx) >= retrain_freq)
            )

            if need_retrain:
                # hisensho: Walk-forward - train ONLY on data up to (i - horizon)
                # to avoid using future label information
                train_end = max(0, i - self.predict_horizon)

                if train_end > self.train_window // 2:
                    # Get training data up to train_end
                    train_dates_wf = dates[:train_end]
                    train_factors_wf = factors_df[
                        factors_df.index.get_level_values("date") <= pd.to_datetime(train_dates_wf[-1])
                    ]
                    train_returns_wf = returns_df[
                        returns_df.index.get_level_values("date") <= pd.to_datetime(train_dates_wf[-1])
                    ]

                    # Retrain model
                    self.train_model(train_factors_wf, train_returns_wf, selected_factors=selected)
                    last_train_idx = i
                    logger.debug(f"[AlphaFactory] Walk-forward retrain at {date}, using data up to {train_dates_wf[-1]}")

            # Generate signal for current date
            signal = self.predict_signal(factors_df, date, selected)
            signals[date] = signal

        logger.info(f"[AlphaFactory] Generated {len(signals)} signals with walk-forward={walk_forward}")
        return signals


def create_alpha_factory(
    qlib_data_dir: Optional[str] = None,
    factors: Optional[List[str]] = None,
    **kwargs,
) -> AlphaFactory:
    """
    Convenience function to create AlphaFactory.

    Args:
        qlib_data_dir: Qlib data directory
        factors: Factor names to use (default: all)
        **kwargs: Additional arguments for AlphaFactory

    Returns:
        AlphaFactory instance
    """
    factory = AlphaFactory(qlib_data_dir=qlib_data_dir, **kwargs)
    return factory


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic data
    np.random.seed(42)
    n_products, n_dates = 50, 300

    # Generate synthetic NAV (random walk with drift)
    nav_matrix = np.ones((n_products, n_dates), dtype=np.float32)
    for t in range(1, n_dates):
        drift = 0.0002 + np.random.randn(n_products) * 0.005
        nav_matrix[:, t] = nav_matrix[:, t - 1] * (1 + drift)

    dates = [(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(n_dates)]
    product_codes = [f"P{i:03d}" for i in range(n_products)]

    # Create factory and generate signals
    factory = AlphaFactory(train_window=100, predict_horizon=5)
    signals = factory.generate_signals(nav_matrix, dates, product_codes)

    print(f"\nGenerated signals for {len(signals)} dates")
    if signals:
        last_date = list(signals.keys())[-1]
        last_signal = signals[last_date]
        print(f"Last signal ({last_date}):")
        print(f"  Top 5 products: {sorted(last_signal.signals.items(), key=lambda x: -x[1])[:5]}")
