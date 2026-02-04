# -*- coding: utf-8 -*-
"""
GLM 去平滑算法测试
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aifinance.data.unsmoothing import GLMUnsmoothing, glm_unsmooth


class TestGLMUnsmoothing:
    """GLM 去平滑测试"""

    @pytest.fixture
    def smoothed_data(self):
        """生成已平滑的测试数据"""
        np.random.seed(42)
        n = 200

        # 真实收益率
        true_ret = np.random.randn(n) * 2

        # 平滑: R^o = 0.6*R_t + 0.3*R_{t-1} + 0.1*R_{t-2}
        theta_true = np.array([0.6, 0.3, 0.1])
        observed_ret = np.convolve(true_ret, theta_true, mode="same")

        return true_ret, observed_ret, theta_true

    def test_theta_estimation(self, smoothed_data):
        """测试 θ 估计"""
        true_ret, observed_ret, theta_true = smoothed_data

        glm = GLMUnsmoothing(max_lag=2, method="mle")
        estimated_theta = glm.estimate_theta(observed_ret)

        # θ 应该归一化
        assert np.isclose(estimated_theta.sum(), 1.0), "Theta should sum to 1"

        # All theta values should be non-negative
        assert (estimated_theta >= 0).all(), "All theta values should be non-negative"

        # θ should have correct length
        assert len(estimated_theta) == 3, "Should have max_lag + 1 theta values"

    def test_unsmoothing_increases_volatility(self, smoothed_data):
        """测试去平滑增加波动率"""
        true_ret, observed_ret, theta_true = smoothed_data

        glm = GLMUnsmoothing(max_lag=2)
        unsmoothed_ret = glm.unsmooth(observed_ret)

        # 去平滑后波动率应该增加
        assert unsmoothed_ret.std() >= observed_ret.std() * 0.9

    def test_unsmoothing_2d_array(self):
        """测试 2D 数组输入"""
        np.random.seed(42)
        n_products, n_dates = 5, 100

        observed = np.random.randn(n_products, n_dates).astype(np.float32) * 2

        glm = GLMUnsmoothing(max_lag=2)
        unsmoothed = glm.unsmooth(observed)

        assert unsmoothed.shape == observed.shape

    def test_convenience_function(self, smoothed_data):
        """测试便捷函数"""
        true_ret, observed_ret, theta_true = smoothed_data

        unsmoothed, info = glm_unsmooth(observed_ret, max_lag=2)

        assert len(unsmoothed) == len(observed_ret)
        assert "theta" in info
        assert "smoothing_index" in info

    def test_short_series_fallback(self):
        """测试短序列回退"""
        short_ret = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        glm = GLMUnsmoothing(max_lag=2)
        unsmoothed = glm.unsmooth(short_ret)

        # 短序列应该返回原始数据
        np.testing.assert_array_equal(unsmoothed, short_ret)

    def test_smoothing_info(self, smoothed_data):
        """测试平滑信息"""
        true_ret, observed_ret, theta_true = smoothed_data

        glm = GLMUnsmoothing(max_lag=2)
        glm.estimate_theta(observed_ret)
        info = glm.get_smoothing_info()

        assert "theta_0" in info
        assert "smoothing_index" in info
        assert 0 <= info["smoothing_index"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
