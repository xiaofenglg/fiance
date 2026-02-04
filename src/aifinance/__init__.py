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

from . import data
from . import factors
from . import models
from . import portfolio
from . import backtest
from . import utils
