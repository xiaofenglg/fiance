# -*- coding: utf-8 -*-
"""
因子计算模块

25维因子体系:
- 收益因子 (5): ret_1d, ret_5d_ma, ret_10d_ma, ret_zscore, ret_rank
- 动量因子 (4): mom_5_10, mom_5_20, ret_acceleration, trend_strength
- 波动因子 (3): vol_5d, vol_20d, vol_ratio
- 周期因子 (4): days_since_release, phase_in_cycle, bday_of_month, weekday
- 质量因子 (4): pulse_width, pulse_density, hype_ratio, maturity_days
- 宏观因子 (4): rate_regime, yield_spread, quarter_effect, month_end_effect
- 新鲜度因子 (1): nav_freshness
"""

from .factor_engine import FactorEngine, compute_all_factors, FEATURE_DIM

__all__ = [
    "FactorEngine",
    "compute_all_factors",
    "FEATURE_DIM",
]
