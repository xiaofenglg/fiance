# -*- coding: utf-8 -*-
"""
组合优化模块

包含:
- HRP (Hierarchical Risk Parity) 优化器
- CVaR 风险控制
- ATR 动态止损
- 仓位管理
"""

from .hrp_optimizer import HRPOptimizer
from .risk_control import CVaRController, check_portfolio_risk
from .position_sizer import PositionSizer, PositionAllocation

__all__ = [
    "HRPOptimizer",
    "CVaRController",
    "check_portfolio_risk",
    "PositionSizer",
    "PositionAllocation",
]
