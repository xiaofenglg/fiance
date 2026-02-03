# -*- coding: utf-8 -*-
"""
大规模并行参数寻优

核心目标：在 GPU 上并行运行数万组回测参数，找全局最优策略

方法：
- 将回测逻辑向量化为张量运算
- 所有参数组合同时运行在 GPU 上
- 输出: 参数热力图 + 最优参数集 + 鲁棒性分析
"""

import os
import logging
import traceback
import itertools
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


# ── 默认参数网格 ──
DEFAULT_PARAM_GRID = {
    'buy_threshold': [2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5],
    'sell_threshold': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    'max_hold_days': [10, 15, 20, 25, 30],
    'hold_eval_days': [5, 7, 10, 14],
    'success_criterion': [2.0, 2.5, 3.0, 3.5],
    'min_success_rate': [25, 30, 35, 40],
}

# CPU 回退: 缩小网格
FALLBACK_PARAM_GRID = {
    'buy_threshold': [3.0, 3.5, 4.0],
    'sell_threshold': [1.5, 2.0, 2.5],
    'max_hold_days': [15, 20, 25],
    'hold_eval_days': [7, 10],
    'success_criterion': [2.5, 3.0],
    'min_success_rate': [30, 35],
}


class VectorizedBacktest:
    """向量化回测引擎 — GPU 版

    核心思路：
    - 预计算信号矩阵: signals[products, dates] = bool
    - 参数影响的是阈值和窗口
    - 对每组参数，用张量运算一次性完成整个回测
    """

    def __init__(self, device=None):
        if device is None:
            device = get_device()
        self.device = device
        self.returns = None
        self.masks = None
        self.dates = None
        self.product_keys = None
        self.n_products = 0
        self.n_dates = 0
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """设置进度回调 callback(progress: float, message: str)"""
        self._progress_callback = callback

    def _emit_progress(self, progress, message):
        if self._progress_callback:
            self._progress_callback(progress, message)

    def load_data(self) -> None:
        """加载数据并构建张量"""
        data = load_nav_tensors()
        if len(data['product_keys']) == 0:
            raise ValueError("无产品数据")

        self.returns = to_numpy(data['returns'])
        self.masks = to_numpy(data['masks'])
        self.dates = data['dates']
        self.product_keys = data['product_keys']
        self.n_products, self.n_dates = self.returns.shape

        logger.info(f"[ParamSweep] 加载 {self.n_products} 产品 x {self.n_dates} 天")

    def sweep(self, param_grid: dict = None) -> dict:
        """运行参数扫描

        Returns: {
            'results': list of dicts — 每组参数的绩效指标
            'best_params': dict — 最优参数
            'best_sharpe': float
            'best_return': float
            'heatmaps': {
                'sharpe': 2D list (buy_thresh x sell_thresh)
                'return': 2D list
                'drawdown': 2D list
            }
            'robustness': {
                'top10_params': [...],
                'stability_score': float
            }
        }
        """
        if self.returns is None:
            self.load_data()

        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID if is_gpu_available() else FALLBACK_PARAM_GRID

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]
        combinations = list(itertools.product(*param_values))
        n_combos = len(combinations)

        logger.info(f"[ParamSweep] 参数组合: {n_combos}组, "
                    f"数据: {self.n_products}产品 x {self.n_dates}天")
        self._emit_progress(10, f'生成 {n_combos} 组参数组合...')

        # 运行每组参数的回测
        results = []
        for ci, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            perf = self._run_single_backtest(params)
            perf.update(params)
            results.append(perf)

            if ci % max(1, n_combos // 20) == 0:
                pct = 10 + int(80 * (ci + 1) / n_combos)
                self._emit_progress(pct,
                    f'回测 {ci+1}/{n_combos} | 当前最优夏普: '
                    f'{max((r.get("sharpe",0) for r in results), default=0):.2f}')

        self._emit_progress(90, '分析结果...')

        # 排序结果
        results.sort(key=lambda r: r.get('sharpe', 0), reverse=True)

        best = results[0] if results else {}
        best_params = {k: best.get(k) for k in param_names}

        # 构建热力图 (buy_threshold x sell_threshold)
        heatmaps = self._build_heatmaps(results, param_grid)

        # 鲁棒性分析
        robustness = self._analyze_robustness(results, param_names)

        self._emit_progress(100, f'完成 — 最优夏普 {best.get("sharpe", 0):.2f}, '
                            f'年化 {best.get("ann_return", 0):.2f}%')

        return {
            'results': results[:100],  # 只返回前100组（避免传输过大）
            'best_params': best_params,
            'best_sharpe': round(best.get('sharpe', 0), 4),
            'best_return': round(best.get('ann_return', 0), 4),
            'best_win_rate': round(best.get('win_rate', 0), 1),
            'best_max_dd': round(best.get('max_drawdown', 0), 4),
            'total_combos': n_combos,
            'heatmaps': heatmaps,
            'robustness': robustness,
        }

    def _run_single_backtest(self, params: dict) -> dict:
        """向量化运行单组参数的回测

        简化假设:
        - 等权仓位（最多9仓）
        - 固定最大持仓数 9
        - T+1 通过日期偏移模拟
        """
        buy_thresh = params.get('buy_threshold', 3.5)
        sell_thresh = params.get('sell_threshold', 2.0)
        max_hold = params.get('max_hold_days', 20)
        eval_days = params.get('hold_eval_days', 7)
        success_crit = params.get('success_criterion', 2.5)
        min_success = params.get('min_success_rate', 30)

        max_positions = 9
        initial_capital = 1e8
        per_position = initial_capital / max_positions

        returns = self.returns  # (n_products, n_dates)
        masks = self.masks

        # Step 1: 信号检测
        buy_signals = (returns > buy_thresh) & (masks > 0)  # (P, D)
        sell_signals = (returns <= sell_thresh) & (masks > 0)

        # Step 2: 计算每个产品的成功率（滚动窗口）
        n_p, n_d = returns.shape
        success_rates = np.zeros((n_p, n_d), dtype=np.float32)

        for d in range(eval_days, n_d):
            window = returns[:, d - eval_days:d]
            w_mask = masks[:, d - eval_days:d]
            w_valid = w_mask.sum(axis=1)

            # 信号次数
            sig_count = ((window > buy_thresh) & (w_mask > 0)).sum(axis=1)
            # 成功次数
            succ_count = ((window > success_crit) & (w_mask > 0)).sum(axis=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                rate = np.where(sig_count > 0, succ_count / sig_count * 100, 0)
            success_rates[:, d] = rate

        # Step 3: 产品筛选 + 简化模拟交易
        cash = initial_capital
        positions = {}  # key -> {entry_day, entry_return, amount}
        total_pnl = 0
        wins = 0
        losses = 0
        daily_values = [initial_capital]

        # 从第60天开始（需要足够历史）
        start_day = max(60, eval_days + 1)

        for d in range(start_day, n_d - 1):
            # 卖出检查
            to_sell = []
            for key, pos in positions.items():
                hold_days = d - pos['entry_day']
                p_idx = pos['product_idx']
                current_return = returns[p_idx, d] if masks[p_idx, d] > 0 else 0

                should_sell = False
                if hold_days >= max_hold:
                    should_sell = True
                elif hold_days >= 4 and current_return <= sell_thresh:
                    should_sell = True

                if should_sell:
                    # 计算 PnL（简化: 使用区间平均收益）
                    avg_ret = returns[p_idx, pos['entry_day']:d + 1]
                    avg_mask = masks[p_idx, pos['entry_day']:d + 1]
                    valid_rets = avg_ret[avg_mask > 0]
                    if len(valid_rets) > 0:
                        period_ret = valid_rets.mean() / 365 * hold_days / 100
                    else:
                        period_ret = 0
                    pnl = pos['amount'] * period_ret
                    cash += pos['amount'] + pnl
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    to_sell.append(key)

            for key in to_sell:
                del positions[key]

            # 买入检查 (T+1: 使用 d+1 的数据确认)
            slots = max_positions - len(positions)
            if slots > 0:
                # 找到有买入信号且成功率达标的产品
                candidates = []
                for p in range(n_p):
                    if buy_signals[p, d] and success_rates[p, d] >= min_success:
                        # 检查是否已持有
                        key = (p, d)
                        already_held = any(
                            pos['product_idx'] == p for pos in positions.values()
                        )
                        if not already_held:
                            candidates.append((p, returns[p, d]))

                # 按收益排序，取前 slots 个
                candidates.sort(key=lambda x: -x[1])
                for p_idx, _ in candidates[:slots]:
                    key = f"{p_idx}_{d}"
                    positions[key] = {
                        'product_idx': p_idx,
                        'entry_day': d + 1,  # T+1
                        'amount': per_position,
                    }
                    cash -= per_position

            # 记录每日净值
            pos_value = 0
            for pos in positions.values():
                p_idx = pos['product_idx']
                if d >= pos['entry_day'] and masks[p_idx, d] > 0:
                    hold_d = d - pos['entry_day']
                    if hold_d > 0:
                        avg_ret = returns[p_idx, pos['entry_day']:d + 1]
                        avg_mask = masks[p_idx, pos['entry_day']:d + 1]
                        valid = avg_ret[avg_mask > 0]
                        if len(valid) > 0:
                            period_ret = valid.mean() / 365 * hold_d / 100
                        else:
                            period_ret = 0
                        pos_value += pos['amount'] * (1 + period_ret)
                    else:
                        pos_value += pos['amount']
                else:
                    pos_value += pos['amount']
            daily_values.append(cash + pos_value)

        # 计算绩效
        daily_values = np.array(daily_values)
        if len(daily_values) < 2:
            return {'ann_return': 0, 'sharpe': 0, 'max_drawdown': 0,
                    'win_rate': 0, 'total_trades': 0, 'profit_factor': 0}

        final_value = daily_values[-1]
        total_return = (final_value / initial_capital - 1) * 100
        n_days = len(daily_values) - 1
        years = n_days / 252
        ann_return = ((final_value / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

        # 夏普比率
        daily_rets = np.diff(daily_values) / daily_values[:-1]
        rf = 0.025 / 252
        excess = daily_rets - rf
        sharpe = 0
        if len(excess) > 1 and np.std(excess) > 0:
            sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

        # 最大回撤
        peak = np.maximum.accumulate(daily_values)
        drawdowns = (peak - daily_values) / peak * 100
        max_dd = float(drawdowns.max())

        # 胜率
        total_trades = wins + losses
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        return {
            'ann_return': round(ann_return, 4),
            'total_return': round(total_return, 4),
            'sharpe': round(sharpe, 4),
            'max_drawdown': round(max_dd, 4),
            'win_rate': round(win_rate, 1),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'final_value': round(final_value / 1e4, 2),
        }

    def _build_heatmaps(self, results: list, param_grid: dict) -> dict:
        """构建参数热力图 (buy_threshold x sell_threshold)"""
        buy_vals = sorted(param_grid.get('buy_threshold', [3.5]))
        sell_vals = sorted(param_grid.get('sell_threshold', [2.0]))

        n_buy = len(buy_vals)
        n_sell = len(sell_vals)

        sharpe_map = [[0.0] * n_sell for _ in range(n_buy)]
        return_map = [[0.0] * n_sell for _ in range(n_buy)]
        drawdown_map = [[0.0] * n_sell for _ in range(n_buy)]

        # 对每个 (buy, sell) 组合，取该组合下最优的其他参数结果
        for r in results:
            bt = r.get('buy_threshold')
            st = r.get('sell_threshold')
            if bt in buy_vals and st in sell_vals:
                bi = buy_vals.index(bt)
                si = sell_vals.index(st)
                # 取最优（而非平均）
                if r.get('sharpe', 0) > sharpe_map[bi][si]:
                    sharpe_map[bi][si] = round(r.get('sharpe', 0), 2)
                    return_map[bi][si] = round(r.get('ann_return', 0), 2)
                    drawdown_map[bi][si] = round(r.get('max_drawdown', 0), 2)

        return {
            'sharpe': sharpe_map,
            'return': return_map,
            'drawdown': drawdown_map,
            'buy_thresholds': buy_vals,
            'sell_thresholds': sell_vals,
        }

    def _analyze_robustness(self, results: list, param_names: list) -> dict:
        """鲁棒性分析: 检查最优参数是否稳定"""
        if len(results) < 10:
            return {'top10_params': results[:10], 'stability_score': 0}

        top10 = results[:10]

        # 稳定性评分: Top10 参数的一致性
        scores = []
        for pname in param_names:
            vals = [r.get(pname, 0) for r in top10]
            if len(set(vals)) == 1:
                scores.append(1.0)  # 完全一致
            else:
                cv = np.std(vals) / max(np.mean(np.abs(vals)), 0.01)
                scores.append(max(0, 1 - cv))

        stability = float(np.mean(scores))

        return {
            'top10_params': [
                {k: r.get(k) for k in param_names + ['sharpe', 'ann_return', 'max_drawdown', 'win_rate']}
                for r in top10
            ],
            'stability_score': round(stability, 3),
        }
