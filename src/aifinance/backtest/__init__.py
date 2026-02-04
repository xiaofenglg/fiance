# -*- coding: utf-8 -*-
"""
回测引擎模块

功能:
- Walk-forward 回测
- 性能指标计算
- 交易成本模拟
- 结果可视化
"""

from .engine import BacktestEngine, BacktestResult

__all__ = [
    "BacktestEngine",
    "BacktestResult",
]
