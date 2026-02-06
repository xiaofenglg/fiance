# -*- coding: utf-8 -*-
"""
风险控制模块

包含:
- CVaR (Conditional Value at Risk) 风险控制器
- ATR 动态止损
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CVaRController:
    """CVaR 风险控制器

    CVaR (Conditional Value at Risk) = 条件风险价值
    也叫 Expected Shortfall (ES)

    约束:
    - 组合 CVaR(95%) ≤ 3%
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        cvar_limit: float = 0.03,
        lookback_window: int = 60,
    ):
        """
        Args:
            confidence_level: 置信水平 (如 0.95 表示 95%)
            cvar_limit: CVaR 上限 (如 0.03 表示 3%)
            lookback_window: 计算 CVaR 的历史窗口
        """
        self.confidence_level = confidence_level
        self.cvar_limit = cvar_limit
        self.lookback_window = lookback_window

    def calculate_var(self, returns: np.ndarray) -> float:
        """计算 VaR (Value at Risk)

        Args:
            returns: 收益率序列

        Returns:
            VaR 值 (正数表示亏损)
        """
        if len(returns) < 5:
            return 0.0

        alpha = 1 - self.confidence_level
        var = -np.percentile(returns, alpha * 100)

        return max(var, 0)

    def calculate_cvar(self, returns: np.ndarray) -> float:
        """计算 CVaR (Conditional Value at Risk)

        CVaR = 超过 VaR 的平均损失

        Args:
            returns: 收益率序列

        Returns:
            CVaR 值 (正数表示亏损)
        """
        if len(returns) < 5:
            return 0.0

        var = self.calculate_var(returns)
        tail_returns = returns[returns < -var]

        if len(tail_returns) == 0:
            return var

        cvar = -np.mean(tail_returns)
        return max(cvar, var)

    def _batch_calculate_cvar(
        self, returns_matrix: np.ndarray, product_indices: List[int]
    ) -> Dict[int, float]:
        """批量计算多个产品的 CVaR (向量化)

        Args:
            returns_matrix: [n_products, n_dates] 收益率矩阵
            product_indices: 产品索引列表

        Returns:
            {product_idx: cvar}
        """
        n_products, n_dates = returns_matrix.shape

        if n_dates < 5:
            return {idx: 0.0 for idx in product_indices}

        alpha = 1 - self.confidence_level

        # 向量化计算 VaR: 对每行取分位数
        # np.percentile 支持 axis 参数
        var_values = -np.percentile(returns_matrix, alpha * 100, axis=1)
        var_values = np.maximum(var_values, 0)

        # 计算 CVaR: 尾部均值
        # 创建掩码: returns < -var (每行独立比较)
        var_expanded = var_values[:, np.newaxis]  # [n_products, 1]
        tail_mask = returns_matrix < -var_expanded  # [n_products, n_dates]

        # 计算每行尾部均值
        tail_sum = np.where(tail_mask, returns_matrix, 0).sum(axis=1)
        tail_count = tail_mask.sum(axis=1)

        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            cvar_values = np.where(
                tail_count > 0,
                -tail_sum / tail_count,
                var_values
            )
        cvar_values = np.maximum(cvar_values, var_values)

        return {idx: float(cvar_values[i]) for i, idx in enumerate(product_indices)}

    def calculate_portfolio_cvar(
        self,
        positions: List[Dict],
        returns_matrix: np.ndarray,
        masks_matrix: np.ndarray,
        current_idx: int,
    ) -> Tuple[float, Dict]:
        """计算组合 CVaR

        Args:
            positions: 持仓列表, 每个包含:
                - product_idx: 产品索引
                - weight: 权重
            returns_matrix: [n_products, n_dates] 收益率矩阵
            masks_matrix: [n_products, n_dates] 有效掩码
            current_idx: 当前日期索引

        Returns:
            (portfolio_cvar, details)
        """
        if not positions or current_idx < self.lookback_window:
            return 0.0, {"status": "insufficient_data"}

        start_idx = max(0, current_idx - self.lookback_window + 1)
        
        # 提取相关产品的收益率子矩阵
        product_indices = [pos["product_idx"] for pos in positions if pos["product_idx"] < returns_matrix.shape[0]]
        weights = np.array([pos["weight"] for pos in positions if pos["product_idx"] < returns_matrix.shape[0]])
        
        if not product_indices:
            return 0.0, {"status": "no_valid_products"}

        # [n_selected_products, window_len]
        sub_returns = returns_matrix[product_indices, start_idx : current_idx + 1]
        sub_masks = masks_matrix[product_indices, start_idx : current_idx + 1]
        
        # 应用掩码
        masked_returns = np.where(sub_masks > 0, sub_returns, 0)
        
        # 计算组合日收益率序列: weights @ masked_returns / 365 / 100
        # [window_len]
        portfolio_returns = (weights @ masked_returns) / 365 / 100

        portfolio_cvar = self.calculate_cvar(portfolio_returns)

        # 单产品 CVaR (批量计算)
        daily_sub_returns = masked_returns / 365 / 100
        individual_cvars = self._batch_calculate_cvar(daily_sub_returns, product_indices)

        details = {
            "status": "ok",
            "portfolio_cvar": portfolio_cvar,
            "var": self.calculate_var(portfolio_returns),
            "individual_cvars": individual_cvars,
            "window_len": sub_returns.shape[1],
            "within_limit": portfolio_cvar <= self.cvar_limit,
        }

        return portfolio_cvar, details

    def check_risk_breach(
        self,
        positions: List[Dict],
        returns_matrix: np.ndarray,
        masks_matrix: np.ndarray,
        current_idx: int,
    ) -> Tuple[bool, str, float]:
        """检查风险是否超限

        Args:
            positions: 持仓列表
            returns_matrix: 收益率矩阵
            masks_matrix: 有效掩码
            current_idx: 当前日期索引

        Returns:
            (breached, reason, cvar_value)
        """
        cvar, details = self.calculate_portfolio_cvar(
            positions, returns_matrix, masks_matrix, current_idx
        )

        if details.get("status") != "ok":
            return False, "", 0.0

        breached = cvar > self.cvar_limit

        if breached:
            reason = (
                f"CVaR({self.confidence_level:.0%})={cvar:.2%} > "
                f"限制{self.cvar_limit:.2%}"
            )
            logger.warning(f"[CVaR] 风险超限! {reason}")
        else:
            reason = ""

        return breached, reason, cvar

    def suggest_reduction(
        self,
        positions: List[Dict],
        returns_matrix: np.ndarray,
        masks_matrix: np.ndarray,
        current_idx: int,
    ) -> List[Dict]:
        """建议减仓方案

        Args:
            positions: 持仓列表
            returns_matrix: 收益率矩阵
            masks_matrix: 有效掩码
            current_idx: 当前日期索引

        Returns:
            减仓建议列表
        """
        cvar, details = self.calculate_portfolio_cvar(
            positions, returns_matrix, masks_matrix, current_idx
        )

        if cvar <= self.cvar_limit:
            return []

        individual_cvars = details.get("individual_cvars", {})
        if not individual_cvars:
            return []

        # 按 CVaR 风险贡献排序
        risk_contributions = []
        for pos in positions:
            idx = pos["product_idx"]
            ind_cvar = individual_cvars.get(idx, 0)
            risk_contribution = ind_cvar * pos["weight"]
            risk_contributions.append(
                {
                    "product_idx": idx,
                    "weight": pos["weight"],
                    "individual_cvar": ind_cvar,
                    "risk_contribution": risk_contribution,
                }
            )

        risk_contributions.sort(key=lambda x: x["risk_contribution"], reverse=True)

        # 建议减仓
        suggestions = []
        excess_risk = cvar - self.cvar_limit

        for rc in risk_contributions:
            if excess_risk <= 0:
                break

            reduction_ratio = min(
                excess_risk / max(rc["risk_contribution"], 1e-6), 0.5
            )
            new_weight = rc["weight"] * (1 - reduction_ratio)

            suggestions.append(
                {
                    "product_idx": rc["product_idx"],
                    "current_weight": rc["weight"],
                    "suggested_weight": new_weight,
                    "reduction_pct": reduction_ratio * 100,
                    "reason": f"CVaR贡献={rc['risk_contribution']:.2%}",
                }
            )

            excess_risk -= rc["risk_contribution"] * reduction_ratio

        return suggestions


@dataclass
class StopLossLevel:
    """止损水平"""

    initial_stop: float  # 初始止损价
    current_stop: float  # 当前止损价 (追踪后)
    atr: float  # ATR 值
    highest_since_entry: float  # 入场后最高价


@dataclass
class ProfitProtectionLevel:
    """盈利保护水平"""

    entry_price: float  # 入场价
    highest_profit_pct: float  # 历史最高盈利百分比
    protection_line: float  # 保护线 (当前锁定的盈利比例)
    triggered: bool  # 是否触发保护


class ProfitProtection:
    """盈利回吐保护器

    当仓位盈利超过阈值时，设置盈利保护线。
    若后续回吐超过保护线，则触发平仓。

    特点:
    - 盈利超过 5% 时激活保护 (针对银行理财产品提高阈值)
    - 保护线 = 最高盈利 * 40% (保护更少，追求更高收益)
    - 只升不降
    """

    def __init__(
        self,
        activation_threshold: float = 0.05,  # 激活阈值 (5%)
        protection_ratio: float = 0.4,  # 保护比例 (保护最高盈利的40%)
    ):
        """
        Args:
            activation_threshold: 盈利超过此阈值时激活保护
            protection_ratio: 保护线为最高盈利的多少比例
        """
        self.activation_threshold = activation_threshold
        self.protection_ratio = protection_ratio
        self._positions: Dict[int, ProfitProtectionLevel] = {}

    def initialize_position(self, product_idx: int, entry_price: float) -> ProfitProtectionLevel:
        """初始化持仓保护

        Args:
            product_idx: 产品索引
            entry_price: 入场价

        Returns:
            保护水平
        """
        level = ProfitProtectionLevel(
            entry_price=entry_price,
            highest_profit_pct=0.0,
            protection_line=0.0,
            triggered=False,
        )
        self._positions[product_idx] = level
        return level

    def update(self, product_idx: int, current_price: float) -> Tuple[bool, float]:
        """更新盈利保护状态

        Args:
            product_idx: 产品索引
            current_price: 当前价格

        Returns:
            (是否触发保护平仓, 当前盈利比例)
        """
        if product_idx not in self._positions:
            return False, 0.0

        level = self._positions[product_idx]
        current_profit_pct = (current_price - level.entry_price) / level.entry_price

        # 更新最高盈利
        if current_profit_pct > level.highest_profit_pct:
            level.highest_profit_pct = current_profit_pct

            # 如果超过激活阈值，更新保护线
            if current_profit_pct >= self.activation_threshold:
                new_protection = current_profit_pct * self.protection_ratio
                level.protection_line = max(level.protection_line, new_protection)
                logger.debug(
                    f"[ProfitProtection] 产品{product_idx}: "
                    f"最高盈利={current_profit_pct:.2%}, 保护线={level.protection_line:.2%}"
                )

        # 检查是否触发保护
        if level.protection_line > 0 and current_profit_pct < level.protection_line:
            level.triggered = True
            logger.info(
                f"[ProfitProtection] 产品{product_idx}: 触发盈利保护! "
                f"当前盈利={current_profit_pct:.2%} < 保护线={level.protection_line:.2%}"
            )
            return True, current_profit_pct

        return False, current_profit_pct

    def remove_position(self, product_idx: int) -> None:
        """移除持仓追踪"""
        if product_idx in self._positions:
            del self._positions[product_idx]

    def get_all_protections(self) -> Dict[int, ProfitProtectionLevel]:
        """获取所有保护水平"""
        return self._positions.copy()


class ATRStopLoss:
    """ATR 动态止损管理器

    特点:
    - 止损位 = 入场价 - 5×ATR (针对银行理财产品大幅放宽)
    - 追踪止损: 只上移不下移
    - 支持日内更新
    """

    def __init__(
        self,
        atr_multiplier: float = 5.0,  # 大幅放宽止损位，银行理财波动小
        atr_window: int = 20,  # 增加ATR窗口
        use_trailing: bool = True,
        trailing_activation: float = 0.03,  # 追踪激活门槛3%
    ):
        """
        Args:
            atr_multiplier: ATR 倍数
            atr_window: ATR 计算窗口
            use_trailing: 是否使用追踪止损
            trailing_activation: 追踪止损激活门槛
        """
        self.atr_multiplier = atr_multiplier
        self.atr_window = atr_window
        self.use_trailing = use_trailing
        self.trailing_activation = trailing_activation

        self._positions: Dict[int, StopLossLevel] = {}

    def calculate_atr(
        self, returns: np.ndarray, masks: np.ndarray, current_idx: int
    ) -> float:
        """计算 ATR

        Args:
            returns: [n_dates] 收益率序列
            masks: [n_dates] 有效掩码
            current_idx: 当前索引

        Returns:
            ATR 值 (收益率%)
        """
        start_idx = max(0, current_idx - self.atr_window + 1)
        window_ret = returns[start_idx : current_idx + 1]
        window_mask = masks[start_idx : current_idx + 1]

        valid_ret = window_ret[window_mask > 0]

        if len(valid_ret) < 3:
            return 1.0

        true_range = np.abs(np.diff(valid_ret))
        if len(true_range) == 0:
            return 1.0

        atr = np.mean(true_range)
        return max(atr, 0.1)

    def initialize_position(
        self, product_idx: int, entry_price: float, atr: float
    ) -> StopLossLevel:
        """初始化持仓止损

        Args:
            product_idx: 产品索引
            entry_price: 入场价 (NAV)
            atr: ATR 值

        Returns:
            止损水平
        """
        stop_distance = self.atr_multiplier * atr / 100
        initial_stop = entry_price * (1 - stop_distance)

        level = StopLossLevel(
            initial_stop=initial_stop,
            current_stop=initial_stop,
            atr=atr,
            highest_since_entry=entry_price,
        )

        self._positions[product_idx] = level

        logger.debug(
            f"[ATR Stop] 产品{product_idx}: 入场价={entry_price:.4f}, "
            f"ATR={atr:.2f}%, 止损={initial_stop:.4f}"
        )

        return level

    def update_stop(
        self, product_idx: int, current_price: float, entry_price: float
    ) -> Optional[StopLossLevel]:
        """更新追踪止损

        Args:
            product_idx: 产品索引
            current_price: 当前价格 (NAV)
            entry_price: 入场价

        Returns:
            更新后的止损水平
        """
        if product_idx not in self._positions:
            return None

        level = self._positions[product_idx]

        if current_price > level.highest_since_entry:
            level.highest_since_entry = current_price

            if self.use_trailing:
                profit_ratio = current_price / entry_price - 1
                if profit_ratio >= self.trailing_activation:
                    stop_distance = self.atr_multiplier * level.atr / 100
                    new_stop = current_price * (1 - stop_distance)

                    if new_stop > level.current_stop:
                        level.current_stop = new_stop
                        logger.debug(
                            f"[ATR Stop] 产品{product_idx}: "
                            f"追踪止损上移至 {new_stop:.4f}"
                        )

        return level

    def check_stop_triggered(
        self, product_idx: int, current_price: float
    ) -> bool:
        """检查是否触发止损

        Args:
            product_idx: 产品索引
            current_price: 当前价格

        Returns:
            是否触发
        """
        if product_idx not in self._positions:
            return False

        level = self._positions[product_idx]
        triggered = current_price <= level.current_stop

        if triggered:
            logger.info(
                f"[ATR Stop] 产品{product_idx}: 触发止损! "
                f"当前价={current_price:.4f} <= 止损位={level.current_stop:.4f}"
            )

        return triggered

    def remove_position(self, product_idx: int) -> None:
        """移除持仓追踪"""
        if product_idx in self._positions:
            del self._positions[product_idx]

    def get_all_stops(self) -> Dict[int, StopLossLevel]:
        """获取所有止损水平"""
        return self._positions.copy()


def check_portfolio_risk(
    positions: List[Dict],
    returns_matrix: np.ndarray,
    masks_matrix: np.ndarray,
    current_idx: int,
    cvar_limit: float = 0.03,
) -> Dict:
    """便捷函数: 检查组合风险

    Returns:
        风险检查结果字典
    """
    controller = CVaRController(cvar_limit=cvar_limit)

    breached, reason, cvar = controller.check_risk_breach(
        positions, returns_matrix, masks_matrix, current_idx
    )

    result = {
        "breached": breached,
        "reason": reason,
        "portfolio_cvar": cvar,
        "cvar_limit": cvar_limit,
    }

    if breached:
        result["reduction_suggestions"] = controller.suggest_reduction(
            positions, returns_matrix, masks_matrix, current_idx
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试 CVaR
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

    controller = CVaRController(cvar_limit=0.03)
    cvar, details = controller.calculate_portfolio_cvar(
        positions, returns_matrix, masks_matrix, n_dates - 1
    )

    print(f"组合 CVaR(95%): {cvar:.4%}")
    print(f"组合 VaR: {details.get('var', 0):.4%}")
    print(f"是否超限: {not details.get('within_limit', True)}")

    # 测试 ATR 止损
    print("\n--- ATR 止损测试 ---")
    returns = np.random.randn(100) * 2 + 3
    masks = np.ones(100)
    nav = [1.0]
    for i in range(1, 100):
        nav.append(nav[-1] * (1 + returns[i] / 100 / 365))
    nav = np.array(nav)

    stop_mgr = ATRStopLoss()
    atr = stop_mgr.calculate_atr(returns, masks, 50)
    level = stop_mgr.initialize_position(0, nav[50], atr)

    print(f"ATR: {atr:.2f}%")
    print(f"入场价: {nav[50]:.4f}")
    print(f"初始止损: {level.initial_stop:.4f}")
