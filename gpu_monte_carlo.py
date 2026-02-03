# -*- coding: utf-8 -*-
"""
蒙特卡洛组合优化

核心目标：替代固定 Kelly 仓位，用概率模拟找最优配置
在 GPU 上并行模拟 100,000 条投资组合路径

方法：
- 从每个产品的历史收益分布中随机采样未来30天路径
- 计算组合级别的 VaR, CVaR, 最优仓位权重
- 使用 mean-CVaR 优化替代等权/Kelly
"""

import os
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from gpu_engine import (
    get_device, is_gpu_available, is_torch_available,
    load_nav_tensors, to_numpy, to_tensor
)

_torch_ok = False
torch = None

try:
    import torch as _torch
    torch = _torch
    _torch_ok = True
except ImportError:
    pass

# ── 蒙特卡洛参数 ──
DEFAULT_N_SIM = 100_000       # GPU 模拟次数
FALLBACK_N_SIM = 10_000       # CPU 回退模拟次数
DEFAULT_HORIZON = 30          # 前瞻天数
CVAR_LAMBDA = 2.0             # CVaR 惩罚系数
MAX_SINGLE_WEIGHT = 0.20      # 单仓上限 20%
HIST_WINDOW = 90              # 历史窗口（天）


class MonteCarloOptimizer:
    """蒙特卡洛组合优化器"""

    def __init__(self, device=None, n_simulations=None, horizon=DEFAULT_HORIZON):
        if device is None:
            device = get_device()
        self.device = device

        if n_simulations is None:
            self.n_sim = DEFAULT_N_SIM if is_gpu_available() else FALLBACK_N_SIM
        else:
            self.n_sim = n_simulations

        self.horizon = horizon

    def simulate(self, product_keys: List[Tuple],
                 current_positions: List[Dict] = None,
                 total_capital: float = 100_000_000) -> dict:
        """运行蒙特卡洛模拟

        Args:
            product_keys: 候选产品列表 [(bank, code), ...]
            current_positions: 当前持仓 [{'key': (b,c), 'amount': float}, ...]
            total_capital: 总资本

        Returns: 模拟结果字典
        """
        try:
            # 加载数据
            data = load_nav_tensors()
            if len(data['product_keys']) == 0:
                return self._empty_result()

            returns_all = to_numpy(data['returns'])
            masks_all = to_numpy(data['masks'])
            all_keys = data['product_keys']

            # 筛选目标产品
            key_indices = []
            valid_keys = []
            for key in product_keys:
                if key in all_keys:
                    idx = all_keys.index(key)
                    key_indices.append(idx)
                    valid_keys.append(key)

            if not key_indices:
                return self._empty_result()

            n_products = len(key_indices)

            # 提取每个产品最近 HIST_WINDOW 天的有效收益
            product_returns = []
            for idx in key_indices:
                ret = returns_all[idx]
                mask = masks_all[idx]
                valid = ret[mask > 0]
                # 取最后 HIST_WINDOW 天
                valid = valid[-HIST_WINDOW:] if len(valid) > HIST_WINDOW else valid
                product_returns.append(valid)

            # 拟合收益分布参数 (均值, 标准差)
            means = np.array([r.mean() if len(r) > 0 else 0 for r in product_returns])
            stds = np.array([r.std() if len(r) > 1 else 1.0 for r in product_returns])

            # GPU 或 CPU 路径模拟
            if _torch_ok and is_gpu_available():
                sim_returns = self._simulate_torch(means, stds, n_products)
            else:
                sim_returns = self._simulate_numpy(means, stds, n_products)

            # 优化权重
            optimal_weights = self.optimize_weights(sim_returns, n_products)

            # 计算组合统计
            weighted_returns = sim_returns  # 已在 optimize 时用过
            # 用最优权重计算组合收益
            if _torch_ok and torch.is_tensor(sim_returns):
                w = torch.tensor(optimal_weights, device=sim_returns.device, dtype=sim_returns.dtype)
                portfolio_returns = (sim_returns * w.unsqueeze(0).unsqueeze(2)).sum(dim=1)
                # 累积收益 (horizon天)
                cum_returns = portfolio_returns.sum(dim=1)  # (n_sim,)
                cum_np = to_numpy(cum_returns)
            else:
                w = np.array(optimal_weights)
                portfolio_returns = (sim_returns * w[np.newaxis, :, np.newaxis]).sum(axis=1)
                cum_returns = portfolio_returns.sum(axis=1)
                cum_np = cum_returns

            # VaR / CVaR (95%)
            sorted_returns = np.sort(cum_np)
            var_idx = int(len(sorted_returns) * 0.05)
            var_95 = float(-sorted_returns[var_idx])  # 损失为正
            cvar_95 = float(-sorted_returns[:var_idx + 1].mean()) if var_idx > 0 else var_95

            # 期望收益
            expected_return = float(cum_np.mean())

            # 最大回撤分布 (95%分位)
            if _torch_ok and torch.is_tensor(portfolio_returns):
                dd_np = to_numpy(portfolio_returns)
            else:
                dd_np = portfolio_returns

            max_drawdowns = []
            for i in range(min(1000, len(dd_np))):
                cum = np.cumsum(dd_np[i])
                peak = np.maximum.accumulate(cum)
                dd = (peak - cum)
                max_dd = dd.max() if len(dd) > 0 else 0
                max_drawdowns.append(max_dd)
            max_drawdown_95 = float(np.percentile(max_drawdowns, 95)) if max_drawdowns else 0

            # 收益分布直方图
            hist, bin_edges = np.histogram(cum_np, bins=50)
            return_dist = {
                'counts': hist.tolist(),
                'bin_edges': [round(float(b), 4) for b in bin_edges.tolist()],
            }

            # 仓位建议
            weight_dict = {}
            position_sizing = {}
            for i, key in enumerate(valid_keys):
                w_val = float(optimal_weights[i])
                weight_dict[f"{key[0]}|{key[1]}"] = round(w_val, 4)
                position_sizing[f"{key[0]}|{key[1]}"] = round(w_val * total_capital, 0)

            return {
                'var_95': round(var_95, 4),
                'cvar_95': round(cvar_95, 4),
                'expected_return': round(expected_return, 4),
                'max_drawdown_95': round(max_drawdown_95, 4),
                'optimal_weights': weight_dict,
                'position_sizing': position_sizing,
                'return_distribution': return_dist,
                'n_simulations': self.n_sim,
                'horizon': self.horizon,
                'n_products': n_products,
                'product_keys': [f"{k[0]}|{k[1]}" for k in valid_keys],
                'path_summary': {
                    'mean': round(float(cum_np.mean()), 4),
                    'std': round(float(cum_np.std()), 4),
                    'median': round(float(np.median(cum_np)), 4),
                    'p5': round(float(np.percentile(cum_np, 5)), 4),
                    'p95': round(float(np.percentile(cum_np, 95)), 4),
                },
            }

        except Exception as e:
            logger.error(f"[Monte Carlo] 模拟失败: {e}\n{traceback.format_exc()}")
            return self._empty_result()

    def _simulate_torch(self, means, stds, n_products):
        """GPU 路径模拟"""
        device = self.device if isinstance(self.device, str) is False else torch.device(self.device)
        means_t = torch.tensor(means, device=device, dtype=torch.float32)
        stds_t = torch.tensor(stds, device=device, dtype=torch.float32)

        # (n_sim, n_products, horizon) 随机路径
        noise = torch.randn(self.n_sim, n_products, self.horizon, device=device)
        sim_returns = means_t.unsqueeze(0).unsqueeze(2) + stds_t.unsqueeze(0).unsqueeze(2) * noise
        return sim_returns

    def _simulate_numpy(self, means, stds, n_products):
        """CPU 路径模拟"""
        noise = np.random.randn(self.n_sim, n_products, self.horizon).astype(np.float32)
        sim_returns = means[np.newaxis, :, np.newaxis] + stds[np.newaxis, :, np.newaxis] * noise
        return sim_returns

    def optimize_weights(self, sim_returns, n_products) -> np.ndarray:
        """均值-CVaR 优化

        目标: max E[R] - lambda * CVaR(R), lambda=2.0
        约束: 权重和=1, 0 <= w <= 0.2 (单仓不超20%)

        使用网格搜索 + 随机优化的简化方法
        """
        if n_products <= 1:
            return np.array([1.0])

        best_weights = np.ones(n_products) / n_products  # 等权初始化
        best_obj = float('-inf')

        # 随机搜索 1000 组权重
        n_search = 1000
        for _ in range(n_search):
            # 生成随机权重 (Dirichlet分布)
            w = np.random.dirichlet(np.ones(n_products))
            # 限制单仓上限
            w = np.clip(w, 0, MAX_SINGLE_WEIGHT)
            w /= w.sum()

            # 计算组合收益
            if _torch_ok and torch.is_tensor(sim_returns):
                w_t = torch.tensor(w, device=sim_returns.device, dtype=sim_returns.dtype)
                port_ret = (sim_returns * w_t.unsqueeze(0).unsqueeze(2)).sum(dim=1).sum(dim=1)
                port_np = to_numpy(port_ret)
            else:
                port_ret = (sim_returns * w[np.newaxis, :, np.newaxis]).sum(axis=1).sum(axis=1)
                port_np = port_ret

            # 目标: E[R] - lambda * CVaR
            mean_ret = port_np.mean()
            sorted_ret = np.sort(port_np)
            var_idx = int(len(sorted_ret) * 0.05)
            cvar = -sorted_ret[:max(var_idx, 1)].mean()

            obj = mean_ret - CVAR_LAMBDA * cvar

            if obj > best_obj:
                best_obj = obj
                best_weights = w.copy()

        return best_weights

    def _empty_result(self) -> dict:
        return {
            'var_95': 0,
            'cvar_95': 0,
            'expected_return': 0,
            'max_drawdown_95': 0,
            'optimal_weights': {},
            'position_sizing': {},
            'return_distribution': {'counts': [], 'bin_edges': []},
            'n_simulations': 0,
            'horizon': self.horizon,
            'n_products': 0,
            'product_keys': [],
            'path_summary': {},
        }
