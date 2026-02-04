# -*- coding: utf-8 -*-
"""
仓位管理模块

结合 HRP 优化结果和信号强度进行仓位分配。

特点:
- 基于 HRP 的风险预算分配
- 信号强度调节
- 单仓位上限约束
- 波动率约束
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionAllocation:
    """仓位分配结果"""

    product_idx: int
    weight: float  # 仓位权重 (0-1)
    signal_strength: float
    tier: str  # 仓位等级
    risk_contribution: float  # 风险贡献


class PositionSizer:
    """仓位管理器

    结合 HRP 优化和信号强度进行仓位分配
    """

    def __init__(
        self,
        max_single_weight: float = 0.25,
        max_positions: int = 6,
        min_signal_strength: float = 0.5,
        target_vol_contribution: float = 0.01,
    ):
        """
        Args:
            max_single_weight: 单仓最大权重
            max_positions: 最大持仓数
            min_signal_strength: 最低信号强度门槛
            target_vol_contribution: 单仓目标波动率贡献
        """
        self.max_single_weight = max_single_weight
        self.max_positions = max_positions
        self.min_signal_strength = min_signal_strength
        self.target_vol_contribution = target_vol_contribution

        # 信号强度到仓位等级的映射
        self.tier_config = [
            (0.8, "TOP"),
            (0.6, "HIGH"),
            (0.5, "MEDIUM"),
        ]

    def _signal_to_tier(self, signal_strength: float) -> str:
        """信号强度转换为仓位等级"""
        for threshold, tier in self.tier_config:
            if signal_strength >= threshold:
                return tier
        return "BELOW"

    def allocate(
        self,
        hrp_weights: Dict[int, float],
        signals: Dict[int, float],
        volatilities: Optional[Dict[int, float]] = None,
    ) -> List[PositionAllocation]:
        """分配仓位

        结合 HRP 优化权重和信号强度

        Args:
            hrp_weights: HRP 优化的权重 {product_idx: weight}
            signals: 信号强度 {product_idx: strength}
            volatilities: 波动率 {product_idx: vol}

        Returns:
            仓位分配列表
        """
        if not hrp_weights:
            return []

        # 过滤低于门槛的信号
        valid_products = [
            idx
            for idx in hrp_weights.keys()
            if signals.get(idx, 0) >= self.min_signal_strength
        ]

        if not valid_products:
            logger.info("[PositionSizer] 无有效信号")
            return []

        # 按信号强度排序
        valid_products.sort(key=lambda x: signals.get(x, 0), reverse=True)

        # 限制持仓数量
        valid_products = valid_products[: self.max_positions]

        allocations = []
        total_weight = 0.0

        for idx in valid_products:
            signal_strength = signals.get(idx, 0)
            hrp_weight = hrp_weights.get(idx, 0)
            volatility = (
                volatilities.get(idx, 0.02) if volatilities else 0.02
            )

            # 基础权重: HRP 权重
            base_weight = hrp_weight

            # 信号调整: 信号强度作为乘数
            signal_multiplier = 0.5 + signal_strength * 0.5  # 0.5-1.0
            adjusted_weight = base_weight * signal_multiplier

            # 波动率约束
            if volatility > 0:
                vol_limit_weight = self.target_vol_contribution / volatility
                adjusted_weight = min(adjusted_weight, vol_limit_weight)

            # 单仓上限
            adjusted_weight = min(adjusted_weight, self.max_single_weight)
            adjusted_weight = max(adjusted_weight, 0)

            # 检查总权重
            if total_weight + adjusted_weight > 1.0:
                adjusted_weight = max(1.0 - total_weight, 0)

            if adjusted_weight > 0:
                tier = self._signal_to_tier(signal_strength)
                allocations.append(
                    PositionAllocation(
                        product_idx=idx,
                        weight=adjusted_weight,
                        signal_strength=signal_strength,
                        tier=tier,
                        risk_contribution=hrp_weight,  # 使用 HRP 权重作为风险贡献代理
                    )
                )
                total_weight += adjusted_weight

        # 归一化 (如果总权重超过 100%)
        if total_weight > 1.0:
            scale = 1.0 / total_weight
            for alloc in allocations:
                alloc.weight *= scale

        logger.info(
            f"[PositionSizer] 分配完成: {len(allocations)} 个持仓, "
            f"总权重={sum(a.weight for a in allocations):.2%}"
        )

        return allocations

    def allocate_from_signals(
        self,
        signals: List[Dict],
        volatilities: Optional[Dict[int, float]] = None,
    ) -> List[PositionAllocation]:
        """从信号列表直接分配 (不使用 HRP)

        Args:
            signals: 信号列表, 每个包含:
                - product_idx: 产品索引
                - signal_strength: 信号强度
            volatilities: 波动率

        Returns:
            仓位分配列表
        """
        # 过滤低于门槛的信号
        valid_signals = [
            s
            for s in signals
            if s.get("signal_strength", 0) >= self.min_signal_strength
        ]

        if not valid_signals:
            return []

        # 按信号强度排序
        valid_signals.sort(
            key=lambda x: x.get("signal_strength", 0), reverse=True
        )
        valid_signals = valid_signals[: self.max_positions]

        # 计算权重 (基于信号强度的比例)
        total_signal = sum(s.get("signal_strength", 0) for s in valid_signals)

        allocations = []
        total_weight = 0.0

        for s in valid_signals:
            product_idx = s.get("product_idx", -1)
            signal_strength = s.get("signal_strength", 0)
            volatility = (
                volatilities.get(product_idx, 0.02) if volatilities else 0.02
            )

            # 信号比例权重
            if total_signal > 0:
                base_weight = signal_strength / total_signal
            else:
                base_weight = 1.0 / len(valid_signals)

            # 波动率约束
            if volatility > 0:
                vol_limit = self.target_vol_contribution / volatility
                adjusted_weight = min(base_weight, vol_limit)
            else:
                adjusted_weight = base_weight

            # 单仓上限
            adjusted_weight = min(adjusted_weight, self.max_single_weight)

            # 总权重检查
            if total_weight + adjusted_weight > 1.0:
                adjusted_weight = max(1.0 - total_weight, 0)

            if adjusted_weight > 0:
                tier = self._signal_to_tier(signal_strength)
                allocations.append(
                    PositionAllocation(
                        product_idx=product_idx,
                        weight=adjusted_weight,
                        signal_strength=signal_strength,
                        tier=tier,
                        risk_contribution=adjusted_weight,
                    )
                )
                total_weight += adjusted_weight

        # 归一化
        if total_weight > 1.0:
            scale = 1.0 / total_weight
            for alloc in allocations:
                alloc.weight *= scale

        return allocations

    def adjust_for_risk(
        self,
        allocations: List[PositionAllocation],
        portfolio_risk: float,
        risk_limit: float = 0.03,
    ) -> List[PositionAllocation]:
        """根据组合风险调整仓位

        Args:
            allocations: 原始仓位分配
            portfolio_risk: 当前组合风险 (如 CVaR)
            risk_limit: 风险上限

        Returns:
            调整后的仓位分配
        """
        if portfolio_risk <= risk_limit:
            return allocations

        # 按比例缩减
        scale = risk_limit / portfolio_risk
        scale = max(scale, 0.5)  # 最少保留 50%

        for alloc in allocations:
            alloc.weight *= scale

        logger.info(
            f"[PositionSizer] 风险调整: 组合风险={portfolio_risk:.2%} > "
            f"限制={risk_limit:.2%}, 仓位缩减 {scale:.1%}"
        )

        return allocations


def calculate_position_sizes(
    signals: List[Dict],
    total_capital: float,
    hrp_weights: Optional[Dict[int, float]] = None,
    **kwargs,
) -> List[Dict]:
    """便捷函数: 计算仓位金额

    Args:
        signals: 信号列表
        total_capital: 总资金
        hrp_weights: HRP 权重 (可选)
        **kwargs: PositionSizer 参数

    Returns:
        仓位分配列表
    """
    sizer = PositionSizer(**kwargs)

    if hrp_weights:
        signals_dict = {
            s.get("product_idx"): s.get("signal_strength", 0) for s in signals
        }
        allocations = sizer.allocate(hrp_weights, signals_dict)
    else:
        allocations = sizer.allocate_from_signals(signals)

    result = []
    for alloc in allocations:
        result.append(
            {
                "product_idx": alloc.product_idx,
                "weight": alloc.weight,
                "amount": alloc.weight * total_capital,
                "tier": alloc.tier,
                "signal_strength": alloc.signal_strength,
                "risk_contribution": alloc.risk_contribution,
            }
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    signals = [
        {"product_idx": 0, "signal_strength": 0.85},
        {"product_idx": 1, "signal_strength": 0.72},
        {"product_idx": 2, "signal_strength": 0.55},
        {"product_idx": 3, "signal_strength": 0.45},  # 低于门槛
        {"product_idx": 4, "signal_strength": 0.35},  # 低于门槛
        {"product_idx": 5, "signal_strength": 0.90},  # 最强
    ]

    hrp_weights = {
        0: 0.18,
        1: 0.16,
        2: 0.15,
        3: 0.14,
        4: 0.12,
        5: 0.25,
    }

    signals_dict = {s["product_idx"]: s["signal_strength"] for s in signals}

    sizer = PositionSizer()
    allocations = sizer.allocate(hrp_weights, signals_dict)

    print("\n仓位分配:")
    for alloc in allocations:
        print(
            f"  产品 {alloc.product_idx}: 权重={alloc.weight:.2%}, "
            f"信号={alloc.signal_strength:.2f}, 等级={alloc.tier}"
        )

    # 金额分配
    total_capital = 100_000_000
    positions = calculate_position_sizes(
        signals, total_capital, hrp_weights=hrp_weights
    )

    print(f"\n资金分配 (总资金: {total_capital / 1e6:.0f}M):")
    for pos in positions:
        print(
            f"  产品 {pos['product_idx']}: "
            f"{pos['amount'] / 1e6:.1f}M ({pos['weight']:.1%}) [{pos['tier']}]"
        )
