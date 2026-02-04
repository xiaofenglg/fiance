# -*- coding: utf-8 -*-
"""
V11 回测引擎

功能:
- Walk-forward 回测
- 性能指标计算 (年化收益率, 夏普比率, 最大回撤)
- 交易成本模拟
- 结果可视化
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.nav_loader import NavLoader
from ..data.unsmoothing import glm_unsmooth
from ..factors.factor_engine import FactorEngine
from ..models.ensemble import EnsembleModel
from ..portfolio.hrp_optimizer import HRPOptimizer, optimize_portfolio
from ..portfolio.position_sizer import PositionSizer
from ..portfolio.risk_control import CVaRController, ATRStopLoss

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果"""

    # 性能指标
    total_return: float  # 总收益率
    annual_return: float  # 年化收益率
    sharpe_ratio: float  # 夏普比率
    max_drawdown: float  # 最大回撤
    win_rate: float  # 胜率
    profit_factor: float  # 盈亏比

    # 交易统计
    n_trades: int  # 交易次数
    avg_holding_days: float  # 平均持仓天数
    turnover: float  # 换手率

    # 时间序列
    equity_curve: pd.Series  # 净值曲线
    daily_returns: pd.Series  # 日收益率
    positions_history: List[Dict]  # 持仓历史

    # 元数据
    start_date: str
    end_date: str
    n_days: int
    config: Dict = field(default_factory=dict)


class BacktestEngine:
    """V11 回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        train_window: int = 180,
        rebalance_freq: int = 5,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        use_unsmoothing: bool = True,
        risk_free_rate: float = 0.02,
    ):
        """
        Args:
            initial_capital: 初始资金
            train_window: 模型训练窗口 (天)
            rebalance_freq: 再平衡频率 (天)
            transaction_cost: 交易成本 (比例)
            slippage: 滑点 (比例)
            use_unsmoothing: 是否使用 GLM 去平滑
            risk_free_rate: 年化无风险利率
        """
        self.initial_capital = initial_capital
        self.train_window = train_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.use_unsmoothing = use_unsmoothing
        self.risk_free_rate = risk_free_rate

        # 组件
        self.factor_engine = FactorEngine()
        self.model = None
        self.hrp_optimizer = HRPOptimizer()
        self.position_sizer = PositionSizer()
        self.cvar_controller = CVaRController()
        self.atr_stop_loss = ATRStopLoss()

    def run(
        self,
        returns: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        product_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """运行回测

        Args:
            returns: [n_products, n_dates] 年化收益率矩阵 (%)
            masks: [n_products, n_dates] 有效掩码
            dates: 日期列表
            product_codes: 产品代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            BacktestResult
        """
        n_products, n_dates = returns.shape

        # 确定回测区间
        if start_date:
            try:
                start_idx = dates.index(start_date)
            except ValueError:
                start_idx = self.train_window + 30
        else:
            start_idx = self.train_window + 30

        if end_date:
            try:
                end_idx = dates.index(end_date)
            except ValueError:
                end_idx = n_dates - 1
        else:
            end_idx = n_dates - 1

        # GLM 去平滑
        if self.use_unsmoothing:
            logger.info("[Backtest] 应用 GLM 去平滑...")
            returns_unsmoothed, smooth_info = glm_unsmooth(returns)
            logger.info(f"[Backtest] 平滑信息: {smooth_info}")
        else:
            returns_unsmoothed = returns

        # 计算因子
        logger.info("[Backtest] 计算因子...")
        first_valid_idx = self._get_first_valid_idx(masks)
        features = self.factor_engine.compute(
            returns_unsmoothed, masks, dates, first_valid_idx
        )

        # 初始化模型
        self.model = EnsembleModel()

        # 回测状态
        capital = self.initial_capital
        positions: Dict[int, Dict] = {}  # {product_idx: {weight, entry_price, entry_idx}}
        equity_curve = []
        daily_returns_list = []
        positions_history = []
        trades = []

        logger.info(
            f"[Backtest] 开始回测: {dates[start_idx]} -> {dates[end_idx]}"
        )

        for t in range(start_idx, end_idx + 1):
            current_date = dates[t]

            # 计算当日持仓收益
            daily_pnl = 0.0
            for idx, pos in list(positions.items()):
                if masks[idx, t] > 0:
                    daily_return = returns[idx, t] / 365 / 100  # 日收益率
                    position_pnl = capital * pos["weight"] * daily_return
                    daily_pnl += position_pnl

                    # 检查止损
                    nav_today = 1 + returns[idx, t] / 365 / 100
                    if self.atr_stop_loss.check_stop_triggered(idx, nav_today):
                        # 触发止损,平仓
                        cost = capital * pos["weight"] * self.transaction_cost
                        capital -= cost
                        trades.append(
                            {
                                "date": current_date,
                                "product_idx": idx,
                                "action": "stop_loss",
                                "weight": pos["weight"],
                            }
                        )
                        del positions[idx]
                        self.atr_stop_loss.remove_position(idx)

            capital += daily_pnl

            # 计算日收益率
            if equity_curve:
                daily_ret = (capital - equity_curve[-1]) / equity_curve[-1]
            else:
                daily_ret = 0.0

            equity_curve.append(capital)
            daily_returns_list.append(daily_ret)

            # 记录持仓
            positions_history.append(
                {
                    "date": current_date,
                    "capital": capital,
                    "positions": {k: v["weight"] for k, v in positions.items()},
                }
            )

            # 再平衡
            if (t - start_idx) % self.rebalance_freq == 0 and t >= start_idx + 30:
                new_positions, new_trades = self._rebalance(
                    features,
                    returns_unsmoothed,
                    masks,
                    dates,
                    t,
                    positions,
                    capital,
                )

                # 更新持仓
                positions = new_positions
                trades.extend(new_trades)

        # 计算性能指标
        equity_series = pd.Series(equity_curve, index=dates[start_idx : end_idx + 1])
        returns_series = pd.Series(
            daily_returns_list, index=dates[start_idx : end_idx + 1]
        )

        result = self._compute_metrics(
            equity_series,
            returns_series,
            trades,
            positions_history,
            dates[start_idx],
            dates[end_idx],
        )

        return result

    def _rebalance(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        current_idx: int,
        current_positions: Dict[int, Dict],
        capital: float,
    ) -> Tuple[Dict[int, Dict], List[Dict]]:
        """再平衡

        Args:
            features: 因子矩阵
            returns: 收益率矩阵
            masks: 掩码
            dates: 日期列表
            current_idx: 当前索引
            current_positions: 当前持仓
            capital: 当前资金

        Returns:
            (new_positions, trades)
        """
        n_products = features.shape[0]
        current_date = dates[current_idx]
        trades = []

        # 训练模型
        train_result = self.model.train(
            features, returns, masks, dates, current_idx
        )

        if not self.model.is_trained:
            return current_positions, trades

        # 获取信号
        current_features = features[:, current_idx, :]
        seq_start = max(0, current_idx - 59)
        features_seq = features[:, seq_start : current_idx + 1, :]
        masks_seq = masks[:, seq_start : current_idx + 1]
        recent_returns = returns[:, seq_start : current_idx + 1]

        pred = self.model.predict(
            current_features,
            features_seq,
            masks[:, current_idx],
            masks_seq,
            dates[seq_start : current_idx + 1],
            recent_returns,
        )

        # 筛选候选产品
        signal_threshold = 0.5
        candidates = []
        for i in range(n_products):
            if masks[i, current_idx] > 0 and pred["signal_strength"][i] >= signal_threshold:
                candidates.append(i)

        if not candidates:
            return current_positions, trades

        # HRP 优化
        hrp_weights = optimize_portfolio(
            returns,
            masks,
            candidates,
            signals={i: float(pred["signal_strength"][i]) for i in candidates},
            lookback=60,
            current_idx=current_idx,
        )

        # 仓位分配
        signals_dict = {i: float(pred["signal_strength"][i]) for i in candidates}
        allocations = self.position_sizer.allocate(hrp_weights, signals_dict)

        # 风险检查
        new_positions_list = [
            {"product_idx": a.product_idx, "weight": a.weight} for a in allocations
        ]
        breached, reason, cvar = self.cvar_controller.check_risk_breach(
            new_positions_list, returns, masks, current_idx
        )

        if breached:
            allocations = self.position_sizer.adjust_for_risk(
                allocations, cvar, 0.03
            )

        # 构建新持仓
        new_positions = {}
        for alloc in allocations:
            idx = alloc.product_idx

            # 计算交易成本
            old_weight = current_positions.get(idx, {}).get("weight", 0)
            weight_change = abs(alloc.weight - old_weight)
            cost = capital * weight_change * (self.transaction_cost + self.slippage)

            new_positions[idx] = {
                "weight": alloc.weight,
                "entry_price": 1.0,  # NAV 基准
                "entry_idx": current_idx,
            }

            # 初始化止损
            atr = self.atr_stop_loss.calculate_atr(
                returns[idx], masks[idx], current_idx
            )
            self.atr_stop_loss.initialize_position(idx, 1.0, atr)

            # 记录交易
            if weight_change > 0.01:
                action = "buy" if alloc.weight > old_weight else "reduce"
                trades.append(
                    {
                        "date": current_date,
                        "product_idx": idx,
                        "action": action,
                        "weight": alloc.weight,
                        "cost": cost,
                    }
                )

        # 清理不再持有的仓位
        for idx in current_positions:
            if idx not in new_positions:
                trades.append(
                    {
                        "date": current_date,
                        "product_idx": idx,
                        "action": "sell",
                        "weight": 0,
                    }
                )
                self.atr_stop_loss.remove_position(idx)

        return new_positions, trades

    def _get_first_valid_idx(self, masks: np.ndarray) -> np.ndarray:
        """获取每个产品的首个有效索引"""
        n_products, n_dates = masks.shape
        first_valid = np.full(n_products, n_dates, dtype=np.int32)

        for p in range(n_products):
            for d in range(n_dates):
                if masks[p, d] > 0:
                    first_valid[p] = d
                    break

        return first_valid

    def _compute_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: List[Dict],
        positions_history: List[Dict],
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """计算性能指标

        Args:
            equity: 净值曲线
            returns: 日收益率
            trades: 交易记录
            positions_history: 持仓历史
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            BacktestResult
        """
        n_days = len(equity)

        # 总收益率
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

        # 年化收益率
        years = n_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 夏普比率
        excess_returns = returns - self.risk_free_rate / 252
        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        else:
            sharpe_ratio = 0

        # 最大回撤
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # 胜率
        positive_days = (returns > 0).sum()
        win_rate = positive_days / len(returns) if len(returns) > 0 else 0

        # 盈亏比
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float("inf")

        # 交易统计
        n_trades = len([t for t in trades if t["action"] in ["buy", "sell"]])

        # 平均持仓天数
        holding_periods = []
        position_starts = {}
        for ph in positions_history:
            for idx in ph["positions"]:
                if idx not in position_starts:
                    position_starts[idx] = ph["date"]
            for idx in list(position_starts.keys()):
                if idx not in ph["positions"]:
                    try:
                        start = datetime.strptime(position_starts[idx], "%Y-%m-%d")
                        end = datetime.strptime(ph["date"], "%Y-%m-%d")
                        holding_periods.append((end - start).days)
                    except ValueError:
                        pass
                    del position_starts[idx]

        avg_holding_days = np.mean(holding_periods) if holding_periods else 0

        # 换手率
        total_turnover = sum(
            t.get("weight", 0)
            for t in trades
            if t["action"] in ["buy", "sell", "reduce"]
        )
        turnover = total_turnover / years if years > 0 else 0

        result = BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            n_trades=n_trades,
            avg_holding_days=avg_holding_days,
            turnover=turnover,
            equity_curve=equity,
            daily_returns=returns,
            positions_history=positions_history,
            start_date=start_date,
            end_date=end_date,
            n_days=n_days,
            config={
                "initial_capital": self.initial_capital,
                "train_window": self.train_window,
                "rebalance_freq": self.rebalance_freq,
                "transaction_cost": self.transaction_cost,
                "use_unsmoothing": self.use_unsmoothing,
            },
        )

        logger.info(
            f"[Backtest] 结果: 年化收益={annual_return:.2%}, "
            f"夏普={sharpe_ratio:.2f}, 最大回撤={max_drawdown:.2%}"
        )

        return result


def run_backtest(
    returns: np.ndarray,
    masks: np.ndarray,
    dates: List[str],
    **kwargs,
) -> BacktestResult:
    """便捷函数: 运行回测

    Args:
        returns: 收益率矩阵
        masks: 掩码
        dates: 日期列表
        **kwargs: BacktestEngine 参数

    Returns:
        BacktestResult
    """
    engine = BacktestEngine(**kwargs)
    return engine.run(returns, masks, dates)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    np.random.seed(42)
    n_products, n_dates = 50, 400

    # 生成模拟数据
    returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2.5
    masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
    dates = (
        pd.date_range("2023-01-01", periods=n_dates, freq="D")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    # 运行回测
    engine = BacktestEngine(
        initial_capital=100_000_000,
        train_window=180,
        rebalance_freq=5,
    )

    result = engine.run(returns, masks, dates)

    print("\n=== 回测结果 ===")
    print(f"回测区间: {result.start_date} -> {result.end_date} ({result.n_days} 天)")
    print(f"总收益率: {result.total_return:.2%}")
    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2%}")
    print(f"胜率: {result.win_rate:.2%}")
    print(f"交易次数: {result.n_trades}")
