# -*- coding: utf-8 -*-
"""
模型模块测试
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aifinance.models.lgbm_signal import LightGBMSignal, check_lgbm_available
from aifinance.models.tft_signal import TFTSignal, check_tft_available
from aifinance.models.ensemble import EnsembleModel, get_ensemble_signals


class TestLightGBMSignal:
    """LightGBM 模型测试"""

    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n_products, n_dates, n_features = 50, 200, 25
        features = np.random.randn(n_products, n_dates, n_features).astype(
            np.float32
        )
        returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
        masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
        return features, returns, masks

    @pytest.mark.skipif(not check_lgbm_available(), reason="LightGBM not installed")
    def test_train(self, sample_data):
        """测试训练"""
        features, returns, masks = sample_data
        model = LightGBMSignal()

        result = model.train(features, returns, masks, 190)

        assert result["status"] == "ok"
        assert "val_auc" in result
        assert model.is_trained

    @pytest.mark.skipif(not check_lgbm_available(), reason="LightGBM not installed")
    def test_predict(self, sample_data):
        """测试预测"""
        features, returns, masks = sample_data
        model = LightGBMSignal()

        model.train(features, returns, masks, 190)
        pred = model.predict(features[:, -1, :])

        assert "release_prob" in pred
        assert "expected_return" in pred
        assert len(pred["release_prob"]) == features.shape[0]

    def test_fallback_predict(self, sample_data):
        """测试回退预测"""
        features, returns, masks = sample_data
        model = LightGBMSignal()

        # 不训练直接预测
        pred = model.predict(features[:, -1, :])

        assert "release_prob" in pred
        assert len(pred["release_prob"]) == features.shape[0]

    @pytest.mark.skipif(not check_lgbm_available(), reason="LightGBM not installed")
    def test_feature_importance(self, sample_data):
        """测试特征重要性"""
        features, returns, masks = sample_data
        model = LightGBMSignal()

        model.train(features, returns, masks, 190)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == features.shape[2]


class TestTFTSignal:
    """TFT 模型测试"""

    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n_products, n_dates, n_features = 30, 150, 25
        features = np.random.randn(n_products, n_dates, n_features).astype(
            np.float32
        )
        returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
        masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
        dates = (
            pd.date_range("2024-01-01", periods=n_dates, freq="D")
            .strftime("%Y-%m-%d")
            .tolist()
        )
        return features, returns, masks, dates

    def test_fallback_predict(self, sample_data):
        """测试回退预测"""
        features, returns, masks, dates = sample_data
        model = TFTSignal()

        # 不训练直接使用回退
        pred = model._fallback_predict(features[:, -1, :])

        assert "release_prob" in pred
        assert len(pred["release_prob"]) == features.shape[0]

    @pytest.mark.skipif(not check_tft_available(), reason="NeuralForecast not installed")
    def test_train(self, sample_data):
        """测试训练"""
        features, returns, masks, dates = sample_data
        model = TFTSignal(max_steps=5)  # 快速测试

        result = model.train(features, returns, masks, dates, 140)

        # 可能因为数据不足而失败,但不应该报错
        assert "status" in result


class TestEnsembleModel:
    """集成模型测试"""

    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n_products, n_dates, n_features = 30, 150, 25
        features = np.random.randn(n_products, n_dates, n_features).astype(
            np.float32
        )
        returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2
        masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
        dates = (
            pd.date_range("2024-01-01", periods=n_dates, freq="D")
            .strftime("%Y-%m-%d")
            .tolist()
        )
        return features, returns, masks, dates

    def test_default_weights(self):
        """测试默认权重"""
        model = EnsembleModel()

        assert "tft" in model.weights
        assert "lgbm" in model.weights
        assert np.isclose(sum(model.weights.values()), 1.0)

    @pytest.mark.skipif(
        not check_lgbm_available(),
        reason="LightGBM not installed",
    )
    def test_train(self, sample_data):
        """测试训练"""
        features, returns, masks, dates = sample_data
        model = EnsembleModel()

        result = model.train(features, returns, masks, dates, 140)

        assert "weights" in result
        assert "trained" in result

    def test_predict_without_training(self, sample_data):
        """测试未训练时的预测"""
        features, returns, masks, dates = sample_data
        model = EnsembleModel()

        pred = model.predict(features[:, -1, :])

        assert "release_prob" in pred
        assert "signal_strength" in pred

    def test_signal_strength_range(self, sample_data):
        """测试信号强度范围"""
        features, returns, masks, dates = sample_data
        model = EnsembleModel()

        pred = model.predict(features[:, -1, :])

        assert (pred["signal_strength"] >= 0).all()
        assert (pred["signal_strength"] <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
