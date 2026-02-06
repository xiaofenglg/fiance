# -*- coding: utf-8 -*-
"""
AIFinance V11 - Bank Wealth Management Product Selection & Release Strategy

核心升级:
- GLM NAV 去平滑处理
- Temporal Fusion Transformer 信号模型
- Hierarchical Risk Parity 组合优化
- 工程规范化 (pyproject.toml, 标准包结构)

目标指标:
- 年化收益率 ≥ 6%
- 夏普比率 ≥ 1.9
- 最大回撤 < 5%
"""

__version__ = "11.0.0"
__author__ = "AI Finance Team"

# Lazy imports to avoid loading heavy dependencies (PyTorch/Lightning) at startup
# Only import lightweight modules by default
from . import data
from . import factors
from . import portfolio
from . import utils

# Heavy modules (models, backtest) are imported lazily when needed
# to avoid slow startup times in web apps
def __getattr__(name):
    """Lazy import for heavy modules"""
    if name == "models":
        from . import models
        return models
    elif name == "backtest":
        from . import backtest
        return backtest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
