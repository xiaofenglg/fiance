# -*- coding: utf-8 -*-
"""
因子引擎测试
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aifinance.factors.factor_engine import FactorEngine, compute_all_factors, FEATURE_DIM


class TestFactorEngine:
    """因子引擎测试"""

    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n_products, n_dates = 50, 100
        returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
        masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
        dates = [
            (datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_dates)
        ]
        return returns, masks, dates

    def test_feature_dimension(self, sample_data):
        """测试特征维度"""
        returns, masks, dates = sample_data
        engine = FactorEngine()
        features = engine.compute(returns, masks, dates)

        assert features.shape == (
            returns.shape[0],
            returns.shape[1],
            FEATURE_DIM,
        ), f"Expected shape {(returns.shape[0], returns.shape[1], FEATURE_DIM)}, got {features.shape}"

    def test_factor_names_count(self):
        """测试因子名称数量"""
        assert len(FactorEngine.FACTOR_NAMES) == FEATURE_DIM

    def test_no_nan_in_output(self, sample_data):
        """测试输出无 NaN"""
        returns, masks, dates = sample_data
        engine = FactorEngine()
        features = engine.compute(returns, masks, dates)

        assert not np.isnan(features).any(), "Features contain NaN values"

    def test_masked_values_zero(self, sample_data):
        """测试掩码位置的因子值应为 0 (除了 nav_freshness)"""
        returns, masks, dates = sample_data
        engine = FactorEngine()

        # 对于完全无效的产品,因子应为 0 (除了 nav_freshness=1.0)
        zero_mask_product = 0
        masks[zero_mask_product, :] = 0

        features2 = engine.compute(returns, masks, dates)

        # nav_freshness (index 24) should be 1.0 for fully masked products
        # All other factors should be 0 (use atol to handle -0.0 floating point)
        factors_except_freshness = features2[zero_mask_product, :, :24]
        assert np.allclose(
            factors_except_freshness, 0, atol=1e-7
        ), f"Fully masked product should have zero features (except nav_freshness), got max abs value: {np.abs(factors_except_freshness).max()}"

        # nav_freshness should be 1.0 (max staleness)
        assert np.allclose(
            features2[zero_mask_product, :, 24], 1.0
        ), "nav_freshness should be 1.0 for fully masked product"

    def test_compute_all_factors_function(self, sample_data):
        """测试便捷函数"""
        returns, masks, dates = sample_data
        features = compute_all_factors(returns, masks, dates)

        assert features.shape[2] == FEATURE_DIM

    def test_first_valid_idx(self, sample_data):
        """测试 first_valid_idx 参数"""
        returns, masks, dates = sample_data
        n_products = returns.shape[0]

        first_valid = np.random.randint(0, 10, size=n_products).astype(np.int32)

        engine = FactorEngine()
        features = engine.compute(returns, masks, dates, first_valid_idx=first_valid)

        # maturity_days 因子应该使用 first_valid_idx
        assert features.shape[2] == FEATURE_DIM


class TestFactorValues:
    """因子值测试"""

    def test_ret_1d_equals_input(self):
        """测试 ret_1d 因子等于输入"""
        np.random.seed(42)
        returns = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        masks = np.ones_like(returns)
        dates = [f"2024-01-0{i+1}" for i in range(5)]

        engine = FactorEngine()
        features = engine.compute(returns, masks, dates)

        np.testing.assert_array_almost_equal(
            features[0, :, 0], returns[0], decimal=5
        )

    def test_volatility_factor_positive(self):
        """测试波动率因子非负"""
        np.random.seed(42)
        n_products, n_dates = 10, 50
        returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
        masks = np.ones((n_products, n_dates), dtype=np.float32)
        dates = [
            (datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_dates)
        ]

        engine = FactorEngine()
        features = engine.compute(returns, masks, dates)

        # vol_5d (index 9) 和 vol_20d (index 10) 应该非负
        assert (features[:, :, 9] >= 0).all(), "vol_5d should be non-negative"
        assert (features[:, :, 10] >= 0).all(), "vol_20d should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
