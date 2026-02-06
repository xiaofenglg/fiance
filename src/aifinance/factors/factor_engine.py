# -*- coding: utf-8 -*-
"""
V11 26维因子计算引擎 - GPU/CPU双模式

因子体系:
    收益因子(4): ret_1d, ret_5d_ma, ret_zscore, ret_rank
    动量因子(3): mom_5_10, ret_acceleration, trend_strength
    波动因子(3): vol_5d, vol_20d, vol_ratio
    周期因子(4): days_since_release, phase_in_cycle, bday_of_month, weekday
    质量因子(4): pulse_width, pulse_density, hype_ratio, maturity_days
    宏观/期限因子(4): rate_regime, yield_spread, quarter_effect, month_end_effect
    新鲜度因子(1): nav_freshness
    稳定性因子(1): return_stability
    市场脉冲因子(2): market_pulse, leader_signal (领先3-5天的预警信号)
"""

import calendar
import logging
import math
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

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
FEATURE_DIM = 26  # 新增 market_pulse (领先信号) + leader_signal (滞后脉冲)
RELEASE_THRESHOLD = 1.0  # 释放事件阈值 (年化%) - 针对银行理财产品降低
LEADER_LAG_DAYS = 4  # 领先信号滞后天数 (3-5天的中点)


class FactorEngine:
    """26维因子计算引擎 (含市场脉冲和领先信号)"""

    FACTOR_NAMES = [
        # 收益因子 (0-3) - 删除了 ret_10d_ma (与 ret_5d_ma 共线性高)
        "ret_1d",  # 当日年化收益率
        "ret_5d_ma",  # 5日均收益
        "ret_zscore",  # 收益Z-score (标准化)
        "ret_rank",  # 跨产品收益排名百分位
        # 动量因子 (4-6) - 删除了 mom_5_20 (与 mom_5_10 共线性高)
        "mom_5_10",  # 短期动量 MA5/MA10
        "ret_acceleration",  # 收益加速度
        "trend_strength",  # 趋势强度
        # 波动因子 (7-9)
        "vol_5d",  # 5日波动率
        "vol_20d",  # 20日波动率
        "vol_ratio",  # 波动率比 vol_5d/vol_20d
        # 周期因子 (10-13)
        "days_since_release",  # 距上次释放天数
        "phase_in_cycle",  # 周期内相位
        "bday_of_month",  # 月内工作日序号
        "weekday",  # 星期编码
        # 质量因子 (14-17)
        "pulse_width",  # 脉冲宽度 (释放窗口天数)
        "pulse_density",  # 脉冲密度 (近30日释放比例)
        "hype_ratio",  # 炒作比 (峰值/均值)
        "maturity_days",  # 产品存续天数
        # 宏观/期限因子 (18-21)
        "rate_regime",  # 利率环境 (20日均收益变化方向)
        "yield_spread",  # 收益利差 (产品收益 - 市场均值)
        "quarter_effect",  # 季末效应 (距季末天数编码)
        "month_end_effect",  # 月末效应 (距月末天数编码)
        # 新鲜度因子 (22)
        "nav_freshness",  # 净值新鲜度 (距上次更新天数)
        # 质量因子 (23) - 新增
        "return_stability",  # 收益稳定性 (信息比率: mean/std)
        # 市场脉冲因子 (24-25) - 领先信号
        "market_pulse",  # 全市场释放率 (当日释放产品占比)
        "leader_signal",  # 滞后脉冲 (3-5天前的市场释放率) - 预测当前释放的领先指标
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
        """计算26维因子

        Args:
            returns: 年化收益率矩阵 [n_products, n_dates]
            masks: 有效数据掩码 [n_products, n_dates]
            dates: 日期列表 (YYYY-MM-DD)
            first_valid_idx: 每个产品首个有效数据的日期索引 [n_products]

        Returns:
            features: [n_products, n_dates, 26]
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
        """PyTorch GPU版本 - 26维因子 (与NumPy版本索引对齐)

        因子索引 (26维):
        0-3: 收益因子 (ret_1d, ret_5d_ma, ret_zscore, ret_rank)
        4-6: 动量因子 (mom_5_10, ret_acceleration, trend_strength)
        7-9: 波动因子 (vol_5d, vol_20d, vol_ratio)
        10-13: 周期因子 (days_since_release, phase_in_cycle, bday_of_month, weekday)
        14-17: 质量因子 (pulse_width, pulse_density, hype_ratio, maturity_days)
        18-21: 宏观因子 (rate_regime, yield_spread, quarter_effect, month_end_effect)
        22: nav_freshness
        23: return_stability
        24: market_pulse
        25: leader_signal
        """
        n_p, n_d = returns.shape
        device = returns.device
        features = torch.zeros(n_p, n_d, FEATURE_DIM, device=device)
        ret_masked = returns * masks

        # === 收益因子 (0-3) ===
        features[:, :, 0] = ret_masked  # ret_1d

        if n_d >= 5:
            features[:, :, 1] = self._rolling_mean_torch(ret_masked, masks, 5)  # ret_5d_ma

        if n_d >= 20:
            ma20 = self._rolling_mean_torch(ret_masked, masks, 20)
            std20 = self._rolling_std_torch(ret_masked, masks, 20)
            safe_std = std20.clamp(min=0.1)
            features[:, :, 2] = ((ret_masked - ma20) / safe_std * masks).clamp(-3, 3)  # ret_zscore

        features[:, :, 3] = self._cross_rank_torch(ret_masked, masks)  # ret_rank

        # === 动量因子 (4-6) ===
        if n_d >= 10:
            ma5 = self._rolling_mean_torch(ret_masked, masks, 5)
            ma10 = self._rolling_mean_torch(ret_masked, masks, 10)
            safe_ma10 = ma10.clamp(min=0.01)
            features[:, :, 4] = ((ma5 / safe_ma10 - 1) * masks).clamp(-1, 1)  # mom_5_10

        if n_d >= 3:
            diff1 = torch.zeros_like(returns)
            diff1[:, 1:] = (
                (returns[:, 1:] - returns[:, :-1]) * masks[:, 1:] * masks[:, :-1]
            )
            diff2 = torch.zeros_like(returns)
            diff2[:, 2:] = (diff1[:, 2:] - diff1[:, 1:-1]) * masks[:, 2:]
            features[:, :, 5] = diff2.clamp(-5, 5)  # ret_acceleration

        if n_d >= 10:
            features[:, :, 6] = self._trend_strength_torch(ret_masked, masks, 10)  # trend_strength

        # === 波动因子 (7-9) ===
        if n_d >= 5:
            features[:, :, 7] = self._rolling_std_torch(ret_masked, masks, 5)  # vol_5d

        if n_d >= 20:
            features[:, :, 8] = self._rolling_std_torch(ret_masked, masks, 20)  # vol_20d
            vol5 = features[:, :, 7]
            vol20 = features[:, :, 8].clamp(min=0.1)
            features[:, :, 9] = ((vol5 / vol20) * masks).clamp(0, 3)  # vol_ratio

        # === 周期因子 (10-13) ===
        features[:, :, 10] = self._days_since_release_torch(
            returns, masks, RELEASE_THRESHOLD
        )  # days_since_release
        dsr = features[:, :, 10] * 30
        features[:, :, 11] = torch.sin(2 * math.pi * dsr / 21) * masks  # phase_in_cycle

        if dates:
            time_feats = self._encode_time_torch(dates, n_p, n_d, device)
            features[:, :, 12] = time_feats[:, :, 0]  # bday_of_month
            features[:, :, 13] = time_feats[:, :, 1]  # weekday

        # === 质量因子 (14-17) ===
        features[:, :, 14] = self._pulse_width_torch(returns, masks, RELEASE_THRESHOLD)  # pulse_width

        if n_d >= 30:
            is_release_qual = ((returns > RELEASE_THRESHOLD) & (masks > 0)).float()
            features[:, :, 15] = self._rolling_mean_torch(is_release_qual, masks, 30)  # pulse_density

        if n_d >= 20:
            ma20_hype = self._rolling_mean_torch(ret_masked, masks, 20).clamp(min=0.1)
            features[:, :, 16] = ((ret_masked / ma20_hype) * masks).clamp(0, 5)  # hype_ratio

        if first_valid_idx is not None:
            first_idx = (
                torch.from_numpy(first_valid_idx).to(device).float().unsqueeze(1)
            )
            day_idx = torch.arange(n_d, device=device).float().unsqueeze(0)
            maturity = (day_idx - first_idx).clamp(min=0) / 365.0
            features[:, :, 17] = (maturity * masks).clamp(0, 3)  # maturity_days

        # === 宏观/期限因子 (18-21) ===
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
            features[:, :, 18] = (
                market_trend.unsqueeze(0).expand(n_p, -1) * masks
            ).clamp(-1, 1)  # rate_regime

        market_daily_avg = ret_masked.sum(dim=0) / masks.sum(dim=0).clamp(min=1)
        spread = ret_masked - market_daily_avg.unsqueeze(0)
        features[:, :, 19] = (spread * masks).clamp(-5, 5)  # yield_spread

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
            features[:, :, 20] = quarter_enc.unsqueeze(0).expand(n_p, -1) * masks  # quarter_effect

            month_enc = torch.zeros(n_d, device=device)
            for i, d in enumerate(dates):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    _, last_day = calendar.monthrange(dt.year, dt.month)
                    days_to_mend = last_day - dt.day
                    month_enc[i] = max(0, 1 - days_to_mend / 10)
                except ValueError:
                    pass
            features[:, :, 21] = month_enc.unsqueeze(0).expand(n_p, -1) * masks  # month_end_effect

        # === 净值新鲜度因子 (22) ===
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
        features[:, :, 22] = nav_freshness  # nav_freshness

        # === 收益稳定性因子 (23) ===
        if n_d >= 20:
            rolling_mean = self._rolling_mean_torch(ret_masked, masks, 20)
            rolling_std = self._rolling_std_torch(ret_masked, masks, 20)
            safe_std = rolling_std.clamp(min=0.1)
            return_stability = ((rolling_mean / safe_std) * masks).clamp(-3, 3)
            features[:, :, 23] = return_stability  # return_stability

        # === 市场脉冲因子 (24) ===
        is_release = ((returns > RELEASE_THRESHOLD) & (masks > 0)).float()
        valid_count = masks.sum(dim=0).clamp(min=1)
        market_release_rate = is_release.sum(dim=0) / valid_count
        features[:, :, 24] = market_release_rate.unsqueeze(0).expand(n_p, -1) * masks  # market_pulse

        # === 领先信号因子 (25) ===
        leader_signal = torch.zeros(n_d, device=device)
        if n_d > LEADER_LAG_DAYS:
            leader_signal[LEADER_LAG_DAYS:] = market_release_rate[:-LEADER_LAG_DAYS]
        features[:, :, 25] = leader_signal.unsqueeze(0).expand(n_p, -1) * masks  # leader_signal

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
        """NumPy CPU版本

        因子索引 (26维):
        0-3: 收益因子 (ret_1d, ret_5d_ma, ret_zscore, ret_rank)
        4-6: 动量因子 (mom_5_10, ret_acceleration, trend_strength)
        7-9: 波动因子 (vol_5d, vol_20d, vol_ratio)
        10-13: 周期因子 (days_since_release, phase_in_cycle, bday_of_month, weekday)
        14-17: 质量因子 (pulse_width, pulse_density, hype_ratio, maturity_days)
        18-21: 宏观因子 (rate_regime, yield_spread, quarter_effect, month_end_effect)
        22: nav_freshness
        23: return_stability
        24: market_pulse (全市场释放率)
        25: leader_signal (滞后脉冲 - 领先指标)
        """
        n_p, n_d = returns.shape
        features = np.zeros((n_p, n_d, FEATURE_DIM), dtype=np.float32)
        ret_masked = returns * masks

        # === 收益因子 (0-3) ===
        features[:, :, 0] = ret_masked  # ret_1d

        if n_d >= 5:
            features[:, :, 1] = self._rolling_mean_np(ret_masked, masks, 5)  # ret_5d_ma

        if n_d >= 20:
            ma20 = self._rolling_mean_np(ret_masked, masks, 20)
            std20 = self._rolling_std_np(ret_masked, masks, 20)
            safe_std = np.maximum(std20, 0.1)
            features[:, :, 2] = np.clip((ret_masked - ma20) / safe_std * masks, -3, 3)  # ret_zscore

        features[:, :, 3] = self._cross_rank_np(ret_masked, masks)  # ret_rank

        # === 动量因子 (4-6) ===
        if n_d >= 10:
            ma5 = self._rolling_mean_np(ret_masked, masks, 5)
            ma10 = self._rolling_mean_np(ret_masked, masks, 10)
            safe_ma10 = np.maximum(ma10, 0.01)
            features[:, :, 4] = np.clip((ma5 / safe_ma10 - 1) * masks, -1, 1)  # mom_5_10

        # 删除了 mom_5_20 (与 mom_5_10 共线性高)

        if n_d >= 3:
            diff1 = np.zeros_like(returns)
            diff1[:, 1:] = (
                (returns[:, 1:] - returns[:, :-1]) * masks[:, 1:] * masks[:, :-1]
            )
            diff2 = np.zeros_like(returns)
            diff2[:, 2:] = (diff1[:, 2:] - diff1[:, 1:-1]) * masks[:, 2:]
            features[:, :, 5] = np.clip(diff2, -5, 5)  # ret_acceleration

        if n_d >= 10:
            features[:, :, 6] = self._trend_strength_np(ret_masked, masks, 10)  # trend_strength

        # === 波动因子 (7-9) ===
        if n_d >= 5:
            features[:, :, 7] = self._rolling_std_np(ret_masked, masks, 5)  # vol_5d
        if n_d >= 20:
            features[:, :, 8] = self._rolling_std_np(ret_masked, masks, 20)  # vol_20d
            vol5 = features[:, :, 7]
            vol20 = np.maximum(features[:, :, 8], 0.1)
            features[:, :, 9] = np.clip((vol5 / vol20) * masks, 0, 3)  # vol_ratio

        # === 周期因子 (10-13) ===
        features[:, :, 10] = self._days_since_release_np(
            returns, masks, RELEASE_THRESHOLD
        )  # days_since_release
        dsr = features[:, :, 10] * 30
        features[:, :, 11] = np.sin(2 * np.pi * dsr / 21) * masks  # phase_in_cycle

        if dates:
            # 向量化时间因子计算
            date_arr = pd.to_datetime(dates, errors='coerce')
            valid_mask = ~pd.isna(date_arr)

            # bday_of_month: day / 23, capped at 1
            days = np.where(valid_mask, date_arr.day, 0)
            bday_factor = np.minimum(days / 23.0, 1.0)
            features[:, :, 12] = bday_factor[np.newaxis, :] * masks  # bday_of_month

            # weekday: weekday / 4 for Mon-Fri, 1.0 for Sat-Sun
            weekdays = np.where(valid_mask, date_arr.weekday, 0)
            wday_factor = np.where(weekdays < 5, weekdays / 4.0, 1.0)
            features[:, :, 13] = wday_factor[np.newaxis, :] * masks  # weekday

        # === 质量因子 (14-17) ===
        features[:, :, 14] = self._pulse_width_np(returns, masks, RELEASE_THRESHOLD)  # pulse_width

        if n_d >= 30:
            is_release = ((returns > RELEASE_THRESHOLD) & (masks > 0)).astype(
                np.float32
            )
            features[:, :, 15] = self._rolling_mean_np(is_release, masks, 30)  # pulse_density

        if n_d >= 20:
            ma20_hype = np.maximum(
                self._rolling_mean_np(ret_masked, masks, 20), 0.1
            )
            features[:, :, 16] = np.clip((ret_masked / ma20_hype) * masks, 0, 5)  # hype_ratio

        if first_valid_idx is not None:
            first_idx = first_valid_idx.reshape(-1, 1)
            day_idx = np.arange(n_d).reshape(1, -1)
            maturity = np.maximum(day_idx - first_idx, 0) / 365.0
            features[:, :, 17] = np.clip(maturity * masks, 0, 3)  # maturity_days

        # === 宏观/期限因子 (18-21) ===
        if n_d >= 20:
            valid_counts = np.maximum(masks.sum(axis=0), 1)
            market_avg = ret_masked.sum(axis=0) / valid_counts

            # 向量化填充低有效数据的日期 (前向填充)
            min_valid = 10
            low_valid_mask = valid_counts < min_valid
            if low_valid_mask.any():
                # 使用 pandas ffill 进行向量化前向填充
                market_avg_series = pd.Series(market_avg)
                market_avg_series[low_valid_mask] = np.nan
                market_avg = market_avg_series.ffill().bfill().values.astype(np.float32)

            # 向量化计算 MA20: 使用 cumsum 技巧
            cs = np.cumsum(np.concatenate([[0], market_avg]))
            market_ma20 = np.zeros(n_d, dtype=np.float32)
            market_ma20[19:] = (cs[20:] - cs[:-20]) / 20

            # 计算趋势
            market_trend = np.zeros(n_d, dtype=np.float32)
            market_trend[20:] = (market_ma20[20:] - market_ma20[:-20]) / 20
            features[:, :, 18] = np.clip(
                market_trend[np.newaxis, :] * masks, -1, 1
            )  # rate_regime

        mask_sum_daily = np.maximum(masks.sum(axis=0), 1)
        market_daily_avg = ret_masked.sum(axis=0) / mask_sum_daily
        spread = ret_masked - market_daily_avg
        features[:, :, 19] = np.clip(spread * masks, -5, 5)  # yield_spread

        if dates:
            # 向量化季末和月末效应计算
            date_arr = pd.to_datetime(dates, errors='coerce')
            valid_mask = ~pd.isna(date_arr)

            months = np.where(valid_mask, date_arr.month, 1)
            days = np.where(valid_mask, date_arr.day, 1)

            # Quarter effect: 距离季末天数
            quarter_end_month = ((months - 1) // 3 + 1) * 3
            quarter_end_day = np.where(np.isin(quarter_end_month, [3, 12]), 31, 30)
            days_to_qend = (quarter_end_month - months) * 30 + (quarter_end_day - days)
            quarter_enc = np.maximum(0, 1 - days_to_qend / 30).astype(np.float32)
            features[:, :, 20] = quarter_enc[np.newaxis, :] * masks  # quarter_effect

            # Month end effect: 距离月末天数
            days_to_mend = np.maximum(30 - days, 0)
            month_enc = np.maximum(0, 1 - days_to_mend / 10).astype(np.float32)
            features[:, :, 21] = month_enc[np.newaxis, :] * masks  # month_end_effect

        # === 净值新鲜度因子 (22) - 向量化 ===
        day_indices = np.arange(n_d)[np.newaxis, :]
        valid_indices = np.where(masks > 0, day_indices, -np.inf)
        last_valid = np.maximum.accumulate(valid_indices, axis=1)
        days_stale = day_indices - last_valid
        never_valid = last_valid < 0
        days_stale = np.where(never_valid, 7.0, days_stale)
        nav_freshness = np.minimum(days_stale / 7.0, 1.0).astype(np.float32)
        nav_freshness = np.where(masks > 0, 0, nav_freshness)
        features[:, :, 22] = nav_freshness  # nav_freshness

        # === 收益稳定性因子 (23) - 新增: 信息比率 mean/std ===
        if n_d >= 20:
            rolling_mean = self._rolling_mean_np(ret_masked, masks, 20)
            rolling_std = self._rolling_std_np(ret_masked, masks, 20)
            safe_std = np.maximum(rolling_std, 0.1)
            # 信息比率: mean / std，值越高表示收益越稳定
            return_stability = (rolling_mean / safe_std) * masks
            features[:, :, 23] = np.clip(return_stability, -3, 3)  # return_stability

        # === 市场脉冲因子 (24) - 全市场释放率 ===
        # 当多个产品同时释放时,往往预示着更多产品即将释放
        is_release = ((returns > RELEASE_THRESHOLD) & (masks > 0)).astype(np.float32)
        valid_count = np.maximum(masks.sum(axis=0), 1)  # 每日有效产品数
        market_release_rate = is_release.sum(axis=0) / valid_count  # 当日释放占比 [0, 1]
        features[:, :, 24] = market_release_rate[np.newaxis, :] * masks  # 广播到所有产品

        # === 领先信号因子 (25) - 滞后脉冲 (3-5天前的市场释放率) ===
        # 当3-5天前市场大量释放时,当前/未来可能也会释放 (周期性模式)
        leader_signal = np.zeros(n_d, dtype=np.float32)
        if n_d > LEADER_LAG_DAYS:
            # 使用滞后LEADER_LAG_DAYS天的market_release_rate
            leader_signal[LEADER_LAG_DAYS:] = market_release_rate[:-LEADER_LAG_DAYS]
        features[:, :, 25] = leader_signal[np.newaxis, :] * masks

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
        """NumPy滚动标准差 (向量化实现)"""
        n_p, n_d = values.shape
        if n_d < window:
            return np.zeros((n_p, n_d), dtype=np.float32)

        # 使用 var = E[X^2] - (E[X])^2 公式
        val_masked = values * masks
        
        # 计算滚动均值 E[X]
        mean = self._rolling_mean_np(values, masks, window)
        
        # 计算滚动平方均值 E[X^2]
        val_sq = (values ** 2) * masks
        mean_sq = self._rolling_mean_np(val_sq, masks, window)
        
        # 计算方差
        var = np.maximum(mean_sq - (mean ** 2), 0)
        return np.sqrt(var) * masks

    def _cross_rank_np(self, values, masks):
        """NumPy跨产品排名 (向量化)"""
        n_p, n_d = values.shape

        # 将无效值设为极小值以排在最后
        masked_values = np.where(masks > 0, values, -np.inf)

        # 使用 argsort 两次得到排名
        order = masked_values.argsort(axis=0)
        ranks = np.zeros_like(values, dtype=np.float32)
        row_idx = np.arange(n_p)[:, np.newaxis]
        ranks[order, np.arange(n_d)] = row_idx

        # 归一化: rank / (n_valid - 1)
        n_valid = np.maximum(masks.sum(axis=0) - 1, 1)
        result = (ranks / n_valid[np.newaxis, :]) * masks

        return result

    def _days_since_release_np(self, returns, masks, threshold):
        """NumPy计算距上次释放天数 (向量化)"""
        n_p, n_d = returns.shape
        release = (returns > threshold) & (masks > 0)

        # 创建日期索引矩阵
        day_indices = np.arange(n_d)[np.newaxis, :]  # [1, n_d]

        # 将 release 事件的日期索引提取出来，非 release 设为 -inf
        release_indices = np.where(release, day_indices, -np.inf)

        # 使用 cummax 找到每个位置之前最近的 release 日期
        last_release = np.maximum.accumulate(release_indices, axis=1)

        # 计算距离上次 release 的天数
        days_gap = day_indices - last_release

        # 处理从未 release 过的情况 (last_release == -inf)
        never_released = last_release < 0
        days_gap = np.where(never_released, 30.0, days_gap)  # 默认30天

        # 归一化到 [0, 1]
        result = np.minimum(days_gap / 30.0, 1.0).astype(np.float32)

        return result * masks

    def _trend_strength_np(self, values, masks, window):
        """NumPy计算趋势强度"""
        n_p, n_d = values.shape
        result = np.zeros((n_p, n_d), dtype=np.float32)

        if n_d > window:
            result[:, window:] = (values[:, window:] - values[:, :-window]) / window

        return np.clip(result * masks, -1, 1)

    def _pulse_width_np(self, returns, masks, threshold):
        """NumPy计算脉冲宽度 (向量化)

        计算连续高值的长度
        """
        n_p, n_d = returns.shape
        is_high = ((returns > threshold * 0.5) & (masks > 0)).astype(np.float32)

        # 方法: 使用 cumsum 和重置点技巧
        # 1. 计算累加和
        cumsum = np.cumsum(is_high, axis=1)

        # 2. 找到重置点 (从高变低的位置)
        # 在每个 0 的位置记录当前 cumsum 值
        reset_values = np.where(is_high == 0, cumsum, 0)

        # 3. 使用 cummax 传播重置值
        last_reset = np.maximum.accumulate(reset_values, axis=1)

        # 4. 当前长度 = cumsum - last_reset
        result = cumsum - last_reset

        # 只在高值位置保留计数
        result = result * is_high

        return np.clip(result / 10.0, 0, 1).astype(np.float32) * masks


def compute_all_factors(
    returns: np.ndarray,
    masks: np.ndarray,
    dates: Optional[List[str]] = None,
    first_valid_idx: Optional[np.ndarray] = None,
    device=None,
) -> np.ndarray:
    """便捷函数: 计算26维因子"""
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
