# -*- coding: utf-8 -*-
"""
V11 25维因子计算引擎 - GPU/CPU双模式

因子体系:
    收益因子(5): ret_1d, ret_5d_ma, ret_10d_ma, ret_zscore, ret_rank
    动量因子(4): mom_5_10, mom_5_20, ret_acceleration, trend_strength
    波动因子(3): vol_5d, vol_20d, vol_ratio
    周期因子(4): days_since_release, phase_in_cycle, bday_of_month, weekday
    质量因子(4): pulse_width, pulse_density, hype_ratio, maturity_days
    宏观/期限因子(4): rate_regime, yield_spread, quarter_effect, month_end_effect
    新鲜度因子(1): nav_freshness
"""

import calendar
import logging
import math
from datetime import datetime
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch (可选)
_torch_available = False
torch = None
F = None

try:
    import torch as _torch
    import torch.nn.functional as _F

    torch = _torch
    F = _F
    _torch_available = True
except ImportError:
    pass

# 常量
FEATURE_DIM = 25
RELEASE_THRESHOLD = 2.5  # 释放事件阈值 (年化%)


class FactorEngine:
    """25维因子计算引擎"""

    FACTOR_NAMES = [
        # 收益因子 (0-4)
        "ret_1d",  # 当日年化收益率
        "ret_5d_ma",  # 5日均收益
        "ret_10d_ma",  # 10日均收益
        "ret_zscore",  # 收益Z-score (标准化)
        "ret_rank",  # 跨产品收益排名百分位
        # 动量因子 (5-8)
        "mom_5_10",  # 短期动量 MA5/MA10
        "mom_5_20",  # 中期动量 MA5/MA20
        "ret_acceleration",  # 收益加速度
        "trend_strength",  # 趋势强度
        # 波动因子 (9-11)
        "vol_5d",  # 5日波动率
        "vol_20d",  # 20日波动率
        "vol_ratio",  # 波动率比 vol_5d/vol_20d
        # 周期因子 (12-15)
        "days_since_release",  # 距上次释放天数
        "phase_in_cycle",  # 周期内相位
        "bday_of_month",  # 月内工作日序号
        "weekday",  # 星期编码
        # 质量因子 (16-19)
        "pulse_width",  # 脉冲宽度 (释放窗口天数)
        "pulse_density",  # 脉冲密度 (近30日释放比例)
        "hype_ratio",  # 炒作比 (峰值/均值)
        "maturity_days",  # 产品存续天数
        # 宏观/期限因子 (20-23)
        "rate_regime",  # 利率环境 (20日均收益变化方向)
        "yield_spread",  # 收益利差 (产品收益 - 市场均值)
        "quarter_effect",  # 季末效应 (距季末天数编码)
        "month_end_effect",  # 月末效应 (距月末天数编码)
        # 新鲜度因子 (24)
        "nav_freshness",  # 净值新鲜度 (距上次更新天数)
    ]

    def __init__(self, device=None):
        if _torch_available:
            self.device = device or (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = "cpu"
        self._bday_cache = {}

    def compute(
        self,
        returns: np.ndarray,
        masks: np.ndarray,
        dates: Optional[List[str]] = None,
        first_valid_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算25维因子

        Args:
            returns: 年化收益率矩阵 [n_products, n_dates]
            masks: 有效数据掩码 [n_products, n_dates]
            dates: 日期列表 (YYYY-MM-DD)
            first_valid_idx: 每个产品首个有效数据的日期索引 [n_products]

        Returns:
            features: [n_products, n_dates, 25]
        """
        use_torch = _torch_available and torch.is_tensor(returns)
        if use_torch:
            return self._compute_torch(returns, masks, dates, first_valid_idx)
        else:
            return self._compute_numpy(
                np.asarray(returns, dtype=np.float32),
                np.asarray(masks, dtype=np.float32),
                dates,
                first_valid_idx,
            )

    @torch.no_grad()
    def _compute_torch(self, returns, masks, dates, first_valid_idx):
        """PyTorch GPU版本"""
        n_p, n_d = returns.shape
        device = returns.device
        features = torch.zeros(n_p, n_d, FEATURE_DIM, device=device)
        ret_masked = returns * masks

        # === 收益因子 (0-4) ===
        features[:, :, 0] = ret_masked

        if n_d >= 5:
            features[:, :, 1] = self._rolling_mean_torch(ret_masked, masks, 5)

        if n_d >= 10:
            features[:, :, 2] = self._rolling_mean_torch(ret_masked, masks, 10)

        if n_d >= 20:
            ma20 = self._rolling_mean_torch(ret_masked, masks, 20)
            std20 = self._rolling_std_torch(ret_masked, masks, 20)
            safe_std = std20.clamp(min=0.1)
            features[:, :, 3] = ((ret_masked - ma20) / safe_std * masks).clamp(-3, 3)

        features[:, :, 4] = self._cross_rank_torch(ret_masked, masks)

        # === 动量因子 (5-8) ===
        if n_d >= 10:
            ma5 = self._rolling_mean_torch(ret_masked, masks, 5)
            ma10 = self._rolling_mean_torch(ret_masked, masks, 10)
            safe_ma10 = ma10.clamp(min=0.01)
            features[:, :, 5] = ((ma5 / safe_ma10 - 1) * masks).clamp(-1, 1)

        if n_d >= 20:
            if n_d < 10:
                ma5 = self._rolling_mean_torch(ret_masked, masks, 5)
            ma20_mom = self._rolling_mean_torch(ret_masked, masks, 20)
            safe_ma20 = ma20_mom.clamp(min=0.01)
            features[:, :, 6] = ((ma5 / safe_ma20 - 1) * masks).clamp(-1, 1)

        if n_d >= 3:
            diff1 = torch.zeros_like(returns)
            diff1[:, 1:] = (
                (returns[:, 1:] - returns[:, :-1]) * masks[:, 1:] * masks[:, :-1]
            )
            diff2 = torch.zeros_like(returns)
            diff2[:, 2:] = (diff1[:, 2:] - diff1[:, 1:-1]) * masks[:, 2:]
            features[:, :, 7] = diff2.clamp(-5, 5)

        if n_d >= 10:
            features[:, :, 8] = self._trend_strength_torch(ret_masked, masks, 10)

        # === 波动因子 (9-11) ===
        if n_d >= 5:
            features[:, :, 9] = self._rolling_std_torch(ret_masked, masks, 5)

        if n_d >= 20:
            features[:, :, 10] = self._rolling_std_torch(ret_masked, masks, 20)
            vol5 = features[:, :, 9]
            vol20 = features[:, :, 10].clamp(min=0.1)
            features[:, :, 11] = ((vol5 / vol20) * masks).clamp(0, 3)

        # === 周期因子 (12-15) ===
        features[:, :, 12] = self._days_since_release_torch(
            returns, masks, RELEASE_THRESHOLD
        )
        dsr = features[:, :, 12] * 30
        features[:, :, 13] = torch.sin(2 * math.pi * dsr / 21)

        if dates:
            time_feats = self._encode_time_torch(dates, n_p, n_d, device)
            features[:, :, 14] = time_feats[:, :, 0]
            features[:, :, 15] = time_feats[:, :, 1]

        # === 质量因子 (16-19) ===
        features[:, :, 16] = self._pulse_width_torch(returns, masks, RELEASE_THRESHOLD)

        if n_d >= 30:
            is_release = ((returns > RELEASE_THRESHOLD) & (masks > 0)).float()
            features[:, :, 17] = self._rolling_mean_torch(is_release, masks, 30)

        if n_d >= 20:
            ma20_hype = self._rolling_mean_torch(ret_masked, masks, 20).clamp(min=0.1)
            features[:, :, 18] = ((ret_masked / ma20_hype) * masks).clamp(0, 5)

        if first_valid_idx is not None:
            first_idx = (
                torch.from_numpy(first_valid_idx).to(device).float().unsqueeze(1)
            )
            day_idx = torch.arange(n_d, device=device).float().unsqueeze(0)
            maturity = (day_idx - first_idx).clamp(min=0) / 365.0
            features[:, :, 19] = (maturity * masks).clamp(0, 3)

        # === 宏观/期限因子 (20-23) ===
        if n_d >= 20:
            valid_counts = masks.sum(dim=0).clamp(min=1)
            market_avg = ret_masked.sum(dim=0) / valid_counts
            min_valid = 10
            for d in range(1, n_d):
                if valid_counts[d] < min_valid:
                    market_avg[d] = market_avg[d - 1]
            market_ma20 = torch.zeros(n_d, device=device)
            for d in range(19, n_d):
                market_ma20[d] = market_avg[d - 19 : d + 1].mean()
            market_trend = torch.zeros(n_d, device=device)
            market_trend[20:] = (market_ma20[20:] - market_ma20[:-20]) / 20
            features[:, :, 20] = (
                market_trend.unsqueeze(0).expand(n_p, -1) * masks
            ).clamp(-1, 1)

        market_daily_avg = ret_masked.sum(dim=0) / masks.sum(dim=0).clamp(min=1)
        spread = ret_masked - market_daily_avg.unsqueeze(0)
        features[:, :, 21] = (spread * masks).clamp(-5, 5)

        if dates:
            quarter_enc = torch.zeros(n_d, device=device)
            for i, d in enumerate(dates):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    month = dt.month
                    quarter_end_month = ((month - 1) // 3 + 1) * 3
                    if quarter_end_month in [3, 12]:
                        quarter_end_day = 31
                    else:
                        quarter_end_day = 30
                    days_to_qend = (quarter_end_month - month) * 30 + (
                        quarter_end_day - dt.day
                    )
                    quarter_enc[i] = max(0, 1 - days_to_qend / 30)
                except ValueError:
                    pass
            features[:, :, 22] = quarter_enc.unsqueeze(0).expand(n_p, -1) * masks

            month_enc = torch.zeros(n_d, device=device)
            for i, d in enumerate(dates):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    _, last_day = calendar.monthrange(dt.year, dt.month)
                    days_to_mend = last_day - dt.day
                    month_enc[i] = max(0, 1 - days_to_mend / 10)
                except ValueError:
                    pass
            features[:, :, 23] = month_enc.unsqueeze(0).expand(n_p, -1) * masks

        # === 净值新鲜度因子 (24) ===
        nav_freshness = torch.zeros(n_p, n_d, device=device)
        for p in range(n_p):
            last_valid = -1
            for d in range(n_d):
                if masks[p, d] > 0:
                    last_valid = d
                    nav_freshness[p, d] = 0
                elif last_valid >= 0:
                    days_stale = d - last_valid
                    nav_freshness[p, d] = min(days_stale / 7.0, 1.0)
                else:
                    nav_freshness[p, d] = 1.0
        features[:, :, 24] = nav_freshness

        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        return features

    def _rolling_mean_torch(self, values, masks, window):
        """GPU滚动均值"""
        n_p, n_d = values.shape
        device = values.device

        val_masked = values * masks
        cs_val = torch.cumsum(F.pad(val_masked, (1, 0), value=0), dim=1)
        cs_mask = torch.cumsum(F.pad(masks, (1, 0), value=0), dim=1)

        result = torch.zeros(n_p, n_d, device=device)

        if n_d >= window:
            roll_sum = cs_val[:, window:] - cs_val[:, :-window]
            roll_cnt = (cs_mask[:, window:] - cs_mask[:, :-window]).clamp(min=1)
            result[:, window - 1 :] = roll_sum / roll_cnt

        for i in range(1, min(window, n_d)):
            c = cs_mask[:, i + 1].clamp(min=1)
            result[:, i] = cs_val[:, i + 1] / c

        return result * masks

    def _rolling_std_torch(self, values, masks, window):
        """GPU滚动标准差"""
        n_p, n_d = values.shape
        device = values.device

        if n_d < window:
            return torch.zeros(n_p, n_d, device=device)

        padded_v = F.pad(values * masks, (window - 1, 0), value=0)
        padded_m = F.pad(masks, (window - 1, 0), value=0)

        win_v = padded_v.unfold(1, window, 1)
        win_m = padded_m.unfold(1, window, 1)

        cnt = win_m.sum(dim=2).clamp(min=1)
        mean = (win_v * win_m).sum(dim=2) / cnt
        var = (((win_v - mean.unsqueeze(2)) ** 2) * win_m).sum(dim=2) / cnt

        return torch.sqrt(var) * masks

    def _cross_rank_torch(self, values, masks):
        """GPU跨产品排名"""
        n_p, n_d = values.shape
        device = values.device

        big_neg = torch.full_like(values, -1e9)
        active_vals = torch.where(masks > 0, values, big_neg)

        order = active_vals.argsort(dim=0)
        ranks = torch.zeros_like(values)
        row_indices = torch.arange(n_p, device=device).unsqueeze(1).expand(-1, n_d)
        ranks.scatter_(0, order, row_indices.float())

        n_valid = (masks > 0).sum(dim=0).clamp(min=1).float()
        result = (ranks / (n_valid.unsqueeze(0) - 1).clamp(min=1)) * masks
        return result

    def _days_since_release_torch(self, returns, masks, threshold):
        """GPU计算距上次释放天数"""
        n_p, n_d = returns.shape
        device = returns.device

        is_release = ((returns > threshold) & (masks > 0)).float()
        indices = (
            torch.arange(n_d, device=device).float().unsqueeze(0).expand(n_p, -1)
        )

        release_idx_masked = torch.where(
            is_release > 0, indices, torch.tensor(-1.0, device=device)
        )
        last_release_idx = torch.cummax(release_idx_masked, dim=1).values

        gap = indices - last_release_idx
        gap = torch.where(
            last_release_idx >= 0, gap, torch.tensor(30.0, device=device)
        )

        return (gap / 30.0).clamp(0, 1) * masks

    def _trend_strength_torch(self, values, masks, window):
        """GPU计算趋势强度"""
        n_p, n_d = values.shape
        device = values.device

        if n_d < window:
            return torch.zeros(n_p, n_d, device=device)

        diff = torch.zeros_like(values)
        if n_d > window:
            diff[:, window:] = (values[:, window:] - values[:, :-window]) / window
        return (diff * masks).clamp(-1, 1)

    def _pulse_width_torch(self, returns, masks, threshold):
        """GPU计算当前脉冲宽度"""
        n_p, n_d = returns.shape
        device = returns.device

        is_high = ((returns > threshold * 0.5) & (masks > 0)).float()

        result = torch.zeros(n_p, n_d, device=device)
        for d in range(n_d):
            if d == 0:
                result[:, d] = is_high[:, d]
            else:
                result[:, d] = torch.where(
                    is_high[:, d] > 0,
                    result[:, d - 1] + 1,
                    torch.zeros(n_p, device=device),
                )

        return (result / 10.0).clamp(0, 1) * masks

    def _encode_time_torch(self, dates, n_p, n_d, device):
        """GPU时间编码"""
        bday = torch.zeros(n_d, device=device)
        wday = torch.zeros(n_d, device=device)

        for i, d in enumerate(dates):
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
                bday[i] = min(dt.day / 23.0, 1.0)
                wday[i] = dt.weekday() / 4.0 if dt.weekday() < 5 else 1.0
            except ValueError:
                pass

        time_feats = torch.stack([bday, wday], dim=1)
        return time_feats.unsqueeze(0).expand(n_p, -1, -1)

    def _compute_numpy(self, returns, masks, dates, first_valid_idx):
        """NumPy CPU版本"""
        n_p, n_d = returns.shape
        features = np.zeros((n_p, n_d, FEATURE_DIM), dtype=np.float32)
        ret_masked = returns * masks

        # === 收益因子 (0-4) ===
        features[:, :, 0] = ret_masked

        if n_d >= 5:
            features[:, :, 1] = self._rolling_mean_np(ret_masked, masks, 5)
        if n_d >= 10:
            features[:, :, 2] = self._rolling_mean_np(ret_masked, masks, 10)
        if n_d >= 20:
            ma20 = self._rolling_mean_np(ret_masked, masks, 20)
            std20 = self._rolling_std_np(ret_masked, masks, 20)
            safe_std = np.maximum(std20, 0.1)
            features[:, :, 3] = np.clip((ret_masked - ma20) / safe_std * masks, -3, 3)

        features[:, :, 4] = self._cross_rank_np(ret_masked, masks)

        # === 动量因子 (5-8) ===
        if n_d >= 10:
            ma5 = self._rolling_mean_np(ret_masked, masks, 5)
            ma10 = self._rolling_mean_np(ret_masked, masks, 10)
            safe_ma10 = np.maximum(ma10, 0.01)
            features[:, :, 5] = np.clip((ma5 / safe_ma10 - 1) * masks, -1, 1)

        if n_d >= 20:
            if n_d < 10:
                ma5 = self._rolling_mean_np(ret_masked, masks, 5)
            ma20_mom = self._rolling_mean_np(ret_masked, masks, 20)
            safe_ma20 = np.maximum(ma20_mom, 0.01)
            features[:, :, 6] = np.clip((ma5 / safe_ma20 - 1) * masks, -1, 1)

        if n_d >= 3:
            diff1 = np.zeros_like(returns)
            diff1[:, 1:] = (
                (returns[:, 1:] - returns[:, :-1]) * masks[:, 1:] * masks[:, :-1]
            )
            diff2 = np.zeros_like(returns)
            diff2[:, 2:] = (diff1[:, 2:] - diff1[:, 1:-1]) * masks[:, 2:]
            features[:, :, 7] = np.clip(diff2, -5, 5)

        if n_d >= 10:
            features[:, :, 8] = self._trend_strength_np(ret_masked, masks, 10)

        # === 波动因子 (9-11) ===
        if n_d >= 5:
            features[:, :, 9] = self._rolling_std_np(ret_masked, masks, 5)
        if n_d >= 20:
            features[:, :, 10] = self._rolling_std_np(ret_masked, masks, 20)
            vol5 = features[:, :, 9]
            vol20 = np.maximum(features[:, :, 10], 0.1)
            features[:, :, 11] = np.clip((vol5 / vol20) * masks, 0, 3)

        # === 周期因子 (12-15) ===
        features[:, :, 12] = self._days_since_release_np(
            returns, masks, RELEASE_THRESHOLD
        )
        dsr = features[:, :, 12] * 30
        features[:, :, 13] = np.sin(2 * np.pi * dsr / 21) * masks

        if dates:
            for i, d in enumerate(dates):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    features[:, i, 14] = min(dt.day / 23.0, 1.0) * masks[:, i]
                    features[:, i, 15] = (dt.weekday() / 4.0 if dt.weekday() < 5 else 1.0) * masks[:, i]
                except ValueError:
                    pass

        # === 质量因子 (16-19) ===
        features[:, :, 16] = self._pulse_width_np(returns, masks, RELEASE_THRESHOLD)

        if n_d >= 30:
            is_release = ((returns > RELEASE_THRESHOLD) & (masks > 0)).astype(
                np.float32
            )
            features[:, :, 17] = self._rolling_mean_np(is_release, masks, 30)

        if n_d >= 20:
            ma20_hype = np.maximum(
                self._rolling_mean_np(ret_masked, masks, 20), 0.1
            )
            features[:, :, 18] = np.clip((ret_masked / ma20_hype) * masks, 0, 5)

        if first_valid_idx is not None:
            first_idx = first_valid_idx.reshape(-1, 1)
            day_idx = np.arange(n_d).reshape(1, -1)
            maturity = np.maximum(day_idx - first_idx, 0) / 365.0
            features[:, :, 19] = np.clip(maturity * masks, 0, 3)

        # === 宏观/期限因子 (20-23) ===
        if n_d >= 20:
            valid_counts = np.maximum(masks.sum(axis=0), 1)
            market_avg = ret_masked.sum(axis=0) / valid_counts
            min_valid = 10
            for d in range(1, n_d):
                if valid_counts[d] < min_valid:
                    market_avg[d] = market_avg[d - 1]
            market_ma20 = np.zeros(n_d, dtype=np.float32)
            for d in range(19, n_d):
                market_ma20[d] = market_avg[d - 19 : d + 1].mean()
            market_trend = np.zeros(n_d, dtype=np.float32)
            market_trend[20:] = (market_ma20[20:] - market_ma20[:-20]) / 20
            features[:, :, 20] = np.clip(
                np.broadcast_to(market_trend, (n_p, n_d)) * masks, -1, 1
            )

        mask_sum_daily = np.maximum(masks.sum(axis=0), 1)
        market_daily_avg = ret_masked.sum(axis=0) / mask_sum_daily
        spread = ret_masked - market_daily_avg
        features[:, :, 21] = np.clip(spread * masks, -5, 5)

        if dates:
            quarter_enc = np.zeros(n_d, dtype=np.float32)
            for i, d in enumerate(dates):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    month = dt.month
                    quarter_end_month = ((month - 1) // 3 + 1) * 3
                    if quarter_end_month in [3, 12]:
                        quarter_end_day = 31
                    else:
                        quarter_end_day = 30
                    days_to_qend = (quarter_end_month - month) * 30 + (
                        quarter_end_day - dt.day
                    )
                    quarter_enc[i] = max(0, 1 - days_to_qend / 30)
                except ValueError:
                    pass
            features[:, :, 22] = np.broadcast_to(quarter_enc, (n_p, n_d)) * masks

            month_enc = np.zeros(n_d, dtype=np.float32)
            for i, d in enumerate(dates):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    _, last_day = calendar.monthrange(dt.year, dt.month)
                    days_to_mend = last_day - dt.day
                    month_enc[i] = max(0, 1 - days_to_mend / 10)
                except ValueError:
                    pass
            features[:, :, 23] = np.broadcast_to(month_enc, (n_p, n_d)) * masks

        # === 净值新鲜度因子 (24) ===
        nav_freshness = np.zeros((n_p, n_d), dtype=np.float32)
        for p in range(n_p):
            last_valid = -1
            for d in range(n_d):
                if masks[p, d] > 0:
                    last_valid = d
                    nav_freshness[p, d] = 0
                elif last_valid >= 0:
                    days_stale = d - last_valid
                    nav_freshness[p, d] = min(days_stale / 7.0, 1.0)
                else:
                    nav_freshness[p, d] = 1.0
        features[:, :, 24] = nav_freshness

        return np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

    def _rolling_mean_np(self, values, masks, window):
        """NumPy滚动均值"""
        n_p, n_d = values.shape
        result = np.zeros((n_p, n_d), dtype=np.float32)

        val_masked = values * masks
        cs_val = np.cumsum(np.pad(val_masked, ((0, 0), (1, 0))), axis=1)
        cs_mask = np.cumsum(np.pad(masks, ((0, 0), (1, 0))), axis=1)

        if n_d >= window:
            roll_sum = cs_val[:, window:] - cs_val[:, :-window]
            roll_cnt = np.maximum(cs_mask[:, window:] - cs_mask[:, :-window], 1)
            result[:, window - 1 :] = roll_sum / roll_cnt

        for i in range(1, min(window, n_d)):
            c = np.maximum(cs_mask[:, i + 1], 1)
            result[:, i] = cs_val[:, i + 1] / c

        return result * masks

    def _rolling_std_np(self, values, masks, window):
        """NumPy滚动标准差"""
        n_p, n_d = values.shape
        result = np.zeros((n_p, n_d), dtype=np.float32)

        for d in range(window - 1, n_d):
            w = values[:, d - window + 1 : d + 1] * masks[:, d - window + 1 : d + 1]
            m = masks[:, d - window + 1 : d + 1]
            cnt = np.maximum(m.sum(axis=1), 1)
            mean = (w * m).sum(axis=1) / cnt
            var = (((w - mean[:, None]) ** 2) * m).sum(axis=1) / cnt
            result[:, d] = np.sqrt(var)

        return result * masks

    def _cross_rank_np(self, values, masks):
        """NumPy跨产品排名"""
        n_p, n_d = values.shape
        result = np.zeros((n_p, n_d), dtype=np.float32)

        for d in range(n_d):
            col = values[:, d]
            valid = masks[:, d] > 0
            nv = valid.sum()
            if nv > 1:
                order = col.argsort()
                ranks = np.zeros(n_p, dtype=np.float32)
                ranks[order] = np.arange(n_p, dtype=np.float32)
                result[:, d] = (ranks / max(nv - 1, 1)) * valid.astype(np.float32)

        return result

    def _days_since_release_np(self, returns, masks, threshold):
        """NumPy计算距上次释放天数"""
        n_p, n_d = returns.shape
        release = (returns > threshold) & (masks > 0)
        result = np.zeros((n_p, n_d), dtype=np.float32)

        for p in range(n_p):
            last = -1
            for d in range(n_d):
                if release[p, d]:
                    last = d
                if last >= 0 and masks[p, d] > 0:
                    result[p, d] = min((d - last) / 30.0, 1.0)
                else:
                    result[p, d] = 0.0 if masks[p, d] <= 0 else 1.0

        return result * masks

    def _trend_strength_np(self, values, masks, window):
        """NumPy计算趋势强度"""
        n_p, n_d = values.shape
        result = np.zeros((n_p, n_d), dtype=np.float32)

        if n_d > window:
            result[:, window:] = (values[:, window:] - values[:, :-window]) / window

        return np.clip(result * masks, -1, 1)

    def _pulse_width_np(self, returns, masks, threshold):
        """NumPy计算脉冲宽度"""
        n_p, n_d = returns.shape
        is_high = ((returns > threshold * 0.5) & (masks > 0)).astype(np.float32)
        result = np.zeros((n_p, n_d), dtype=np.float32)

        for p in range(n_p):
            count = 0
            for d in range(n_d):
                if is_high[p, d] > 0:
                    count += 1
                else:
                    count = 0
                result[p, d] = count

        return np.clip(result / 10.0, 0, 1) * masks


def compute_all_factors(
    returns: np.ndarray,
    masks: np.ndarray,
    dates: Optional[List[str]] = None,
    first_valid_idx: Optional[np.ndarray] = None,
    device=None,
) -> np.ndarray:
    """便捷函数: 计算25维因子"""
    engine = FactorEngine(device=device)
    return engine.compute(returns, masks, dates, first_valid_idx)


if __name__ == "__main__":
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)

    n_products, n_dates = 100, 365
    returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
    masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
    dates = [
        (datetime(2025, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(n_dates)
    ]

    engine = FactorEngine()
    features = engine.compute(returns, masks, dates)

    print(f"Features shape: {features.shape}")
    print(f"Factor names: {FactorEngine.FACTOR_NAMES}")
    print(f"Sample feature (product 0, date -1): {features[0, -1, :]}")
