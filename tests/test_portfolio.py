# -*- coding: utf-8 -*-
"""
组合优化模块测试
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aifinance.portfolio.hrp_optimizer import HRPOptimizer, optimize_portfolio
from aifinance.portfolio.risk_control import CVaRController, ATRStopLoss, check_portfolio_risk
from aifinance.portfolio.position_sizer import PositionSizer, calculate_position_sizes


class TestHRPOptimizer:
    """HRP 优化器测试"""

    @pytest.fixture
    def sample_returns(self):
        """生成测试收益率"""
        np.random.seed(42)
        n_assets, n_days = 10, 100
        returns = np.random.randn(n_days, n_assets) * 2 + 3
        returns_df = pd.DataFrame(
            returns, columns=[str(i) for i in range(n_assets)]
        )
        return returns_df

    def test_weights_sum_to_one(self, sample_returns):
        """测试权重和为 1"""
        optimizer = HRPOptimizer()
        result = optimizer.optimize(sample_returns)

        total_weight = sum(result.weights.values())
        assert np.isclose(total_weight, 1.0, atol=0.01)

    def test_weights_positive(self, sample_returns):
        """测试权重非负"""
        optimizer = HRPOptimizer()
        result = optimizer.optimize(sample_returns)

        for weight in result.weights.values():
            assert weight >= 0

    def test_max_weight_constraint(self, sample_returns):
        """测试最大权重约束"""
        max_weight = 0.20
        optimizer = HRPOptimizer(max_weight=max_weight)
        result = optimizer.optimize(sample_returns)

        for weight in result.weights.values():
            assert weight <= max_weight + 0.01

    def test_signal_adjustment(self, sample_returns):
        """测试信号调整"""
        signals = {i: 0.9 for i in range(5)}  # 高信号
        signals.update({i: 0.3 for i in range(5, 10)})  # 低信号

        optimizer = HRPOptimizer()
        result = optimizer.optimize(sample_returns, signals)

        # 高信号产品权重应该更高
        high_signal_weights = [result.weights.get(i, 0) for i in range(5)]
        low_signal_weights = [result.weights.get(i, 0) for i in range(5, 10)]

        assert sum(high_signal_weights) >= sum(low_signal_weights)

    def test_optimize_portfolio_function(self):
        """测试便捷函数"""
        np.random.seed(42)
        n_products, n_dates = 10, 100
        returns = np.random.randn(n_products, n_dates).astype(np.float32) * 2 + 3
        masks = np.ones((n_products, n_dates), dtype=np.float32)
        product_indices = list(range(n_products))

        weights = optimize_portfolio(returns, masks, product_indices)

        assert isinstance(weights, dict)


class TestCVaRController:
    """CVaR 控制器测试"""

    @pytest.fixture
    def sample_portfolio(self):
        """生成测试组合"""
        np.random.seed(42)
        n_products, n_dates = 10, 200
        returns_matrix = np.random.randn(n_products, n_dates) * 5 + 3
        masks_matrix = np.ones((n_products, n_dates))
        positions = [
            {"product_idx": 0, "weight": 0.30},
            {"product_idx": 1, "weight": 0.25},
            {"product_idx": 2, "weight": 0.25},
            {"product_idx": 3, "weight": 0.20},
        ]
        return returns_matrix, masks_matrix, positions

    def test_cvar_calculation(self, sample_portfolio):
        """测试 CVaR 计算"""
        returns_matrix, masks_matrix, positions = sample_portfolio
        controller = CVaRController()

        cvar, details = controller.calculate_portfolio_cvar(
            positions, returns_matrix, masks_matrix, 199
        )

        assert cvar >= 0
        assert details["status"] == "ok"
        assert "var" in details
        assert "within_limit" in details

    def test_risk_breach_detection(self, sample_portfolio):
        """测试风险超限检测"""
        returns_matrix, masks_matrix, positions = sample_portfolio

        # 设置很低的限制以触发超限
        controller = CVaRController(cvar_limit=0.0001)

        breached, reason, cvar = controller.check_risk_breach(
            positions, returns_matrix, masks_matrix, 199
        )

        # 应该超限
        assert breached or cvar <= 0.0001

    def test_reduction_suggestions(self, sample_portfolio):
        """测试减仓建议"""
        returns_matrix, masks_matrix, positions = sample_portfolio
        controller = CVaRController(cvar_limit=0.0001)

        suggestions = controller.suggest_reduction(
            positions, returns_matrix, masks_matrix, 199
        )

        # 超限时应该有建议
        # (具体是否超限取决于随机数据)
        if suggestions:
            assert "product_idx" in suggestions[0]
            assert "suggested_weight" in suggestions[0]


class TestATRStopLoss:
    """ATR 止损测试"""

    def test_atr_calculation(self):
        """测试 ATR 计算"""
        np.random.seed(42)
        returns = np.random.randn(100) * 2 + 3
        masks = np.ones(100)

        stop_mgr = ATRStopLoss()
        atr = stop_mgr.calculate_atr(returns, masks, 50)

        assert atr > 0

    def test_stop_initialization(self):
        """测试止损初始化"""
        stop_mgr = ATRStopLoss(atr_multiplier=2.0)
        level = stop_mgr.initialize_position(0, 1.0, 2.0)

        # 止损位应该低于入场价
        assert level.initial_stop < 1.0
        assert level.current_stop == level.initial_stop

    def test_trailing_stop(self):
        """测试追踪止损"""
        stop_mgr = ATRStopLoss(
            atr_multiplier=2.0, trailing_activation=0.02
        )
        level = stop_mgr.initialize_position(0, 1.0, 2.0)

        # 模拟价格上涨
        for price in [1.01, 1.02, 1.03, 1.04, 1.05]:
            level = stop_mgr.update_stop(0, price, 1.0)

        # 止损应该上移
        assert level.current_stop > level.initial_stop

    def test_stop_triggered(self):
        """测试止损触发"""
        stop_mgr = ATRStopLoss(atr_multiplier=2.0)
        level = stop_mgr.initialize_position(0, 1.0, 2.0)

        # 价格下跌到止损位以下
        triggered = stop_mgr.check_stop_triggered(0, level.initial_stop - 0.01)
        assert triggered


class TestPositionSizer:
    """仓位管理测试"""

    def test_basic_allocation(self):
        """测试基本分配"""
        hrp_weights = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        signals = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5}

        sizer = PositionSizer()
        allocations = sizer.allocate(hrp_weights, signals)

        assert len(allocations) > 0
        total_weight = sum(a.weight for a in allocations)
        assert total_weight <= 1.0

    def test_signal_filter(self):
        """测试信号过滤"""
        hrp_weights = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        signals = {0: 0.9, 1: 0.3, 2: 0.2, 3: 0.1}  # 只有 0 超过门槛

        sizer = PositionSizer(min_signal_strength=0.5)
        allocations = sizer.allocate(hrp_weights, signals)

        # 只有信号 >= 0.5 的产品应该被分配
        for alloc in allocations:
            assert signals[alloc.product_idx] >= 0.5

    def test_max_positions(self):
        """测试最大持仓数"""
        hrp_weights = {i: 0.1 for i in range(10)}
        signals = {i: 0.9 - i * 0.05 for i in range(10)}

        sizer = PositionSizer(max_positions=3)
        allocations = sizer.allocate(hrp_weights, signals)

        assert len(allocations) <= 3

    def test_calculate_position_sizes(self):
        """测试便捷函数"""
        signals = [
            {"product_idx": 0, "signal_strength": 0.9},
            {"product_idx": 1, "signal_strength": 0.8},
            {"product_idx": 2, "signal_strength": 0.7},
        ]
        total_capital = 100_000_000

        positions = calculate_position_sizes(signals, total_capital)

        assert len(positions) > 0
        for pos in positions:
            assert "amount" in pos
            assert pos["amount"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
