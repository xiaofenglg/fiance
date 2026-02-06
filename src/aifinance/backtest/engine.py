# -*- coding: utf-8 -*-
"""
V11 回测引擎

功能:
- Walk-forward 回测
- 性能指标计算 (年化收益率, 夏普比率, 最大回撤)
- 交易成本模拟
- 结果可视化
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.nav_loader import NavLoader, resample_to_weekly, compute_weekly_returns
from ..data.unsmoothing import (
    glm_unsmooth,
    GLMUnsmoothing,
    RollingGLMUnsmoothing,
    IlliquidityException,
)
from ..factors.factor_engine import FactorEngine
from ..models.ensemble import EnsembleModel
from ..models.vol_scaled_momentum import VolScaledMomentum, VolScaledMomentumConfig
from ..models.sharpe_optimizer import SharpeOptimizer, SharpeOptimizerConfig, create_sharpe_optimizer
from ..portfolio.hrp_optimizer import HRPOptimizer, optimize_portfolio
from ..portfolio.position_sizer import PositionSizer
from ..portfolio.risk_control import CVaRController, ATRStopLoss, ProfitProtection


class FeeDatabase:
    """赎回费数据库 - 支持 SQLite 和 JSON"""

    def __init__(self, db_path: Optional[str] = None, use_sqlite: bool = True):
        project_root = Path(__file__).parent.parent.parent.parent

        # 优先使用 SQLite
        self.sqlite_path = project_root / "aifinance.sqlite3"
        self.json_path = project_root / "赎回费数据库.json"

        self.use_sqlite = use_sqlite and self.sqlite_path.exists()
        self.products = {}
        self._conn = None
        self._load()

    def _load(self):
        """加载费率数据"""
        if self.use_sqlite:
            self._load_from_sqlite()
        else:
            self._load_from_json()

    def _load_from_sqlite(self):
        """从 SQLite 加载"""
        import sqlite3
        try:
            self._conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products WHERE redemption_fee IS NOT NULL")
            count = cursor.fetchone()[0]
            logger.debug(f"[FeeDB] SQLite 加载 {count} 个产品费率")
        except Exception as e:
            logger.warning(f"[FeeDB] SQLite 加载失败: {e}, 回退到 JSON")
            self.use_sqlite = False
            self._load_from_json()

    def _load_from_json(self):
        """从 JSON 加载"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.products = data.get('products', {})
            logger.debug(f"[FeeDB] JSON 加载 {len(self.products)} 个产品费率")
        except Exception as e:
            logger.debug(f"[FeeDB] 加载失败: {e}")

    def get_fee(self, bank: str, product_code: str, holding_days: int = 0) -> float:
        """获取赎回费率"""
        if self.use_sqlite and self._conn:
            return self._get_fee_sqlite(bank, product_code, holding_days)
        return self._get_fee_json(bank, product_code, holding_days)

    def _get_fee_sqlite(self, bank: str, product_code: str, holding_days: int) -> float:
        """从 SQLite 获取费率 (V3: 优先使用预解析表)"""
        try:
            cursor = self._conn.cursor()

            # V3 hisensho: 优先查询预解析费率表
            cursor.execute('''
                SELECT days_threshold, fee_rate, comparison_type
                FROM parsed_fees
                WHERE product_code = ? AND bank_name = ?
                ORDER BY days_threshold ASC
            ''', (product_code, bank))
            rows = cursor.fetchall()

            if rows:
                # 使用预解析的费率规则
                for days_threshold, rate, cmp_type in rows:
                    if cmp_type == 'less' and holding_days < days_threshold:
                        return rate
                    elif cmp_type == 'greater_eq' and holding_days >= days_threshold:
                        return rate
                # 如果没有匹配，返回最后一条规则的费率
                return rows[-1][1]

            # 回退: 查询原始费率字符串并实时解析
            cursor.execute(
                "SELECT redemption_fee FROM products WHERE product_code = ? AND bank_name = ?",
                (product_code, bank)
            )
            row = cursor.fetchone()
            if row and row[0]:
                fee_str = row[0]
                if '无' in fee_str or not fee_str.strip():
                    return 0.0
                return self._parse_fee_string(fee_str, holding_days)
        except Exception as e:
            logger.debug(f"[FeeDB] SQLite 查询失败: {e}")
        return 0.001  # 默认

    def _parse_fee_string(self, fee_str: str, holding_days: int) -> float:
        """解析费率字符串，支持多种格式

        支持的格式:
        - "7天内:0.50%", "7日内:0.50%"
        - "小于7天 0.50%", "少于7天 0.50%"
        - "7天(含)以内 0.50%"
        - "7天以上:0.00%"
        - "持有期限<7天，赎回费率1.50%"
        """
        import re
        fee_tiers = []

        # Pattern 1: "N天/日(内/以内/以下)" + rate
        pattern1 = re.findall(r'(\d+)\s*[天日]\s*(?:\(含\))?\s*(?:内|以内|以下)[：:\s]*(\d+\.?\d*)\s*%', fee_str)
        for days_str, rate_str in pattern1:
            fee_tiers.append((int(days_str), float(rate_str) / 100, 'less'))

        # Pattern 2: "小于/少于N天/日" + rate
        pattern2 = re.findall(r'[小少]于\s*(\d+)\s*[天日][：:\s,，]*(\d+\.?\d*)\s*%', fee_str)
        for days_str, rate_str in pattern2:
            fee_tiers.append((int(days_str), float(rate_str) / 100, 'less'))

        # Pattern 3: "<N天" format
        pattern3 = re.findall(r'<\s*(\d+)\s*[天日].*?(\d+\.?\d*)\s*%', fee_str)
        for days_str, rate_str in pattern3:
            fee_tiers.append((int(days_str), float(rate_str) / 100, 'less'))

        # Pattern 4: "N天/日(以上/及以上)" + rate
        pattern4 = re.findall(r'(\d+)\s*[天日]\s*(?:\(含\))?\s*(?:以上|及以上)[：:\s]*(\d+\.?\d*)\s*%', fee_str)
        for days_str, rate_str in pattern4:
            fee_tiers.append((int(days_str), float(rate_str) / 100, 'greater_eq'))

        # Fallback: simple "N天.*rate%" pattern
        if not fee_tiers:
            pattern_simple = re.findall(r'(\d+)\s*[天日].*?(\d+\.?\d*)\s*%', fee_str)
            for days_str, rate_str in pattern_simple:
                fee_tiers.append((int(days_str), float(rate_str) / 100, 'less'))

        # Sort by days threshold
        fee_tiers.sort(key=lambda x: x[0])

        # Find applicable fee
        for days_threshold, rate, cmp_type in fee_tiers:
            if cmp_type == 'less' and holding_days < days_threshold:
                return rate
            elif cmp_type == 'greater_eq' and holding_days >= days_threshold:
                return rate

        return 0.0

    def _get_fee_json(self, bank: str, product_code: str, holding_days: int) -> float:
        """从 JSON 获取费率"""
        key = f"{bank}|{product_code}"
        if key not in self.products:
            return 0.001  # 默认0.1%

        info = self.products[key]
        if not info.get('has_redemption_fee', False):
            return 0.0

        fee_schedule = info.get('fee_schedule', [])
        for tier in fee_schedule:
            if holding_days < tier.get('days', 0):
                return tier.get('rate', 0.0)

        return 0.0

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
        rebalance_freq: int = 7,  # 周度再平衡
        train_freq: int = 30,      # 默认每30天重新训练模型
        transaction_cost: float = 0.0001,  # 银行理财低交易成本
        slippage: float = 0.0001,  # 低滑点
        use_unsmoothing: bool = True,
        risk_free_rate: float = 0.02,
        use_fee_db: bool = True,  # 使用真实费率
        bank_name: str = "中信银行",  # 银行名称
        # Phase 1 新参数
        glm_theta0_threshold: float = 0.10,  # GLM theta_0 阈值
        glm_strict_filter: bool = True,  # 严格过滤模式
        frequency: str = "daily",  # 数据频率 ("daily" 或 "weekly")
        # V3 hisensho 审计: Rolling GLM 避免前视偏差
        use_rolling_glm: bool = True,  # 使用滚动窗口 GLM (避免 look-ahead bias)
        glm_window_size: int = 180,  # Rolling GLM 窗口大小 (天)
        # 可配置参数
        momentum_target_return: float = 5.0,  # 动量归一化目标收益率 (%)
        signal_threshold: float = 0.05,  # 信号强度阈值
        momentum_window: int = 20,  # 动量计算窗口
        momentum_weight: float = 0.5,  # 动量信号权重 (1-权重=模型信号权重)
        # V2 hisensho 审计建议: 提取硬编码参数到配置
        signal_weights: Optional[Dict[str, float]] = None,  # 信号融合权重
        drawdown_thresholds: Optional[Dict[str, float]] = None,  # 回撤控制阈值
        target_volatility: float = 0.10,  # 目标年化波动率 (10%)
    ):
        """
        Args:
            initial_capital: 初始资金
            train_window: 模型训练窗口 (天/周)
            rebalance_freq: 再平衡频率 (天/周)
            train_freq: 模型训练频率 (天/周)
            transaction_cost: 交易成本 (比例)
            slippage: 滑点 (比例)
            use_unsmoothing: 是否使用 GLM 去平滑
            risk_free_rate: 年化无风险利率
            glm_theta0_threshold: GLM theta_0 最小阈值 (低于此值的资产将被剔除)
            glm_strict_filter: True=抛出异常剔除资产, False=仅警告
            frequency: 数据频率 ("daily" 或 "weekly")
            momentum_target_return: 动量归一化目标收益率 (%), 用于将动量归一化到 [0, 1]
            signal_threshold: 信号强度阈值，低于此值的产品将被排除
            momentum_window: 动量计算窗口天数
            momentum_weight: 动量信号在最终信号中的权重 (0.0-1.0), 模型信号权重为 1-momentum_weight
        """
        self.initial_capital = initial_capital
        self.train_window = train_window
        self.rebalance_freq = rebalance_freq
        self.train_freq = train_freq
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.use_unsmoothing = use_unsmoothing
        self.risk_free_rate = risk_free_rate
        self.use_fee_db = use_fee_db
        self.bank_name = bank_name
        # Phase 1 参数
        self.glm_theta0_threshold = glm_theta0_threshold
        self.glm_strict_filter = glm_strict_filter
        self.frequency = frequency
        # V3 hisensho 审计: Rolling GLM
        self.use_rolling_glm = use_rolling_glm
        self.glm_window_size = glm_window_size
        # 可配置参数
        self.momentum_target_return = momentum_target_return
        self.signal_threshold = signal_threshold
        self.momentum_window = momentum_window
        self.target_volatility = target_volatility

        # V2 hisensho 审计建议: 可配置的信号权重和回撤阈值
        self.base_signal_weights = signal_weights or {
            "model": 0.25,
            "sharpe": 0.40,
            "momentum": 0.25,
            "regime": 0.10,
        }
        self.signal_weights = self.base_signal_weights.copy()  # 动态调整的权重
        self.drawdown_thresholds = drawdown_thresholds or {
            "severe": 0.04,    # > 4% 紧急减仓 50%
            "moderate": 0.03,  # > 3% 适度减仓 30%
            "light": 0.02,     # > 2% 轻微减仓 15%
        }
        self.momentum_weight = momentum_weight
        # V3 hisensho: 自适应权重开关
        self.use_adaptive_weights = True

        # 组件
        self.factor_engine = FactorEngine()
        self.model = None
        self.hrp_optimizer = HRPOptimizer(max_weight=0.20, min_weight=0.001)
        self.position_sizer = PositionSizer(
            min_signal_strength=0.05,  # 极低门槛
            max_positions=100,  # 高分散
            max_single_weight=0.05,  # 单仓上限5%
        )
        self.cvar_controller = CVaRController(cvar_limit=0.10)  # CVaR限制10% - 非常宽松
        self.atr_stop_loss = ATRStopLoss(atr_multiplier=10.0)  # 极宽止损 - 基本不触发
        self.profit_protection = ProfitProtection(
            activation_threshold=0.10,  # 10%盈利才激活 - 基本不触发
            protection_ratio=0.3,  # 保护30%利润
        )

        # 波动率缩放动量模块 (GitHub 最佳实践)
        self.vol_scaled_momentum = VolScaledMomentum(
            VolScaledMomentumConfig(
                vol_short_window=5,
                vol_long_window=20,
                vol_floor_ratio=0.5,  # 底板保护
                momentum_windows=[5, 10, 20],  # 多窗口动量
                signal_cap=2.0,  # 信号封顶
                target_annual_vol=0.03,  # 银行理财3%目标波动率
            )
        )

        # Sharpe 优化器 (基于 Momentum Transformer 思想)
        # 核心改进: 直接优化夏普比率 + Changepoint Detection
        self.sharpe_optimizer = create_sharpe_optimizer(
            momentum_windows=(5, 21, 63),  # 短/中/长期动量
            target_vol=0.03,  # 银行理财3%目标波动率
        )

        # 费率数据库
        self.fee_db = FeeDatabase() if use_fee_db else None

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

        # ══════════════════════════════════════════════════════════════════════
        # Phase 1: 可选周频转换
        # ══════════════════════════════════════════════════════════════════════
        if self.frequency.lower() == "weekly":
            logger.info("[Backtest] 启用周频模式 (W-WED)")
            returns, masks, dates = compute_weekly_returns(
                returns, masks, dates, anchor_day="WED", annualize=True
            )
            n_products, n_dates = returns.shape
            logger.info(f"[Backtest] 周频数据: {n_products} 产品 x {n_dates} 周")

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

        # ══════════════════════════════════════════════════════════════════════
        # Phase 1: GLM 去平滑 + 资产过滤
        # V3 hisensho 审计修复: 使用 Rolling GLM 避免前视偏差 (look-ahead bias)
        # ══════════════════════════════════════════════════════════════════════
        if self.use_unsmoothing:
            glm_mode = "Rolling" if self.use_rolling_glm else "Global"
            logger.info(
                f"[Backtest] 应用 {glm_mode} GLM 去平滑 "
                f"(θ_0 阈值: {self.glm_theta0_threshold}, 窗口: {self.glm_window_size}天)"
            )

            # 根据配置选择 GLM 实现
            if self.use_rolling_glm:
                glm = RollingGLMUnsmoothing(
                    max_lag=2,
                    method="mle",
                    theta_0_threshold=self.glm_theta0_threshold,
                    strict_filter=False,
                    window_size=self.glm_window_size,
                    min_window=60,
                )
            else:
                glm = GLMUnsmoothing(
                    max_lag=2,
                    method="mle",
                    theta_0_threshold=self.glm_theta0_threshold,
                    strict_filter=False,  # 先设为 False 收集信息
                )

            returns_unsmoothed = np.zeros_like(returns)
            valid_assets = []  # 通过 GLM 过滤的资产索引
            filtered_assets = []  # 被过滤的资产 (theta_0 < 阈值)
            glm_metrics = []

            for p in range(n_products):
                product_returns = returns[p, :]
                product_code = product_codes[p] if product_codes is not None else f"P{p}"

                # 检查是否有足够数据
                valid_count = (masks[p, :] > 0).sum()
                if valid_count < 20:  # 需要至少 20 个有效数据点
                    filtered_assets.append((p, "insufficient_data", 0.0))
                    returns_unsmoothed[p, :] = product_returns  # 保留原数据
                    continue

                try:
                    # V3: 根据配置选择 GLM 方法
                    if self.use_rolling_glm:
                        # Rolling GLM: 避免前视偏差
                        unsmoothed, theta_history = glm.unsmooth_rolling(
                            product_returns,
                            asset_id=product_code,
                            reestimate_freq=26,  # 每半年重估
                        )
                        returns_unsmoothed[p, :] = unsmoothed

                        # 从 theta 历史获取最新 theta_0 (如果有)
                        if theta_history:
                            _, last_theta = theta_history[-1]
                            theta_0 = last_theta[0]
                        else:
                            theta_0 = 1.0  # 默认无平滑

                        if theta_0 >= self.glm_theta0_threshold:
                            valid_assets.append(p)
                        else:
                            filtered_assets.append((p, "low_theta0", theta_0))
                    else:
                        # Global GLM: 使用全量历史数据 (学术回测标准)
                        theta = glm.estimate_theta(product_returns)
                        theta_0 = theta[0]

                        if theta_0 < self.glm_theta0_threshold:
                            # 资产被过滤
                            filtered_assets.append((p, "low_theta0", theta_0))
                            returns_unsmoothed[p, :] = product_returns  # 保留原数据

                            if self.glm_strict_filter:
                                logger.debug(
                                    f"[GLM] 过滤资产 {product_code}: θ_0={theta_0:.4f} < {self.glm_theta0_threshold}"
                                )
                        else:
                            # 资产通过过滤
                            valid_assets.append(p)
                            unsmoothed = glm.unsmooth(product_returns, theta, asset_id=product_code)
                            returns_unsmoothed[p, :] = unsmoothed

                            # 计算验证指标
                            metrics = glm.compute_verification_metrics(
                                product_returns, unsmoothed, asset_id=product_code
                            )
                            glm_metrics.append(metrics)

                except IlliquidityException:
                    filtered_assets.append((p, "illiquid", 0.0))
                    returns_unsmoothed[p, :] = product_returns
                except Exception as e:
                    logger.debug(f"[GLM] 处理 {product_code} 异常: {e}")
                    filtered_assets.append((p, "error", 0.0))
                    returns_unsmoothed[p, :] = product_returns

            logger.info(
                f"[Backtest] GLM 过滤结果: {len(valid_assets)} 有效资产, "
                f"{len(filtered_assets)} 被过滤 (阈值 θ_0 < {self.glm_theta0_threshold})"
            )

            # 如果启用严格过滤，创建过滤后的数据集
            if self.glm_strict_filter and filtered_assets:
                filtered_indices = set(f[0] for f in filtered_assets)
                # 将被过滤资产的 mask 设为 0，阻止其参与交易
                for idx in filtered_indices:
                    masks[idx, :] = 0.0
                logger.info(f"[Backtest] 严格模式: 已禁用 {len(filtered_indices)} 个低流动性资产")
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

        # 止损追踪: 记录最近被止损的产品及其止损日期索引
        # 在止损后的冷却期内(如7天)不允许重新买入
        stopped_positions: Dict[int, int] = {}  # {product_idx: stop_date_idx}
        stop_cooldown_days = 7

        logger.info(
            f"[Backtest] 开始回测: {dates[start_idx]} -> {dates[end_idx]}"
        )

        for t in range(start_idx, end_idx + 1):
            current_date = dates[t]

            # 1. 模型训练 (根据训练频率)
            days_passed = t - start_idx
            if days_passed % self.train_freq == 0:
                logger.info(f"[Backtest] 触发模型训练 @ {current_date}")
                # Phase 2: 传递 bank_name 用于 TFT Bank_ID 特征
                self.model.train(
                    features, returns_unsmoothed, masks, dates, t,
                    bank_name=self.bank_name, product_codes=product_codes
                )

            # 2. 计算当日持仓收益
            # hisensho 审核修复: 根据频率调整年化收益率去年化除数
            # 日频: /365, 周频: /52
            period_divisor = 52 if self.frequency.lower() == "weekly" else 365

            daily_pnl = 0.0
            for idx, pos in list(positions.items()):
                if masks[idx, t] > 0:
                    period_return = returns[idx, t] / period_divisor / 100  # 期间收益率
                    position_pnl = capital * pos["weight"] * period_return
                    daily_pnl += position_pnl

                    # 检查止损
                    nav_today = 1 + returns[idx, t] / period_divisor / 100
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
                        self.profit_protection.remove_position(idx)
                        # 记录止损，进入冷却期
                        stopped_positions[idx] = t
                        continue  # 跳过盈利保护检查

                    # 检查盈利保护
                    profit_triggered, current_profit = self.profit_protection.update(idx, nav_today)
                    if profit_triggered:
                        # 触发盈利保护,平仓
                        cost = capital * pos["weight"] * self.transaction_cost
                        capital -= cost
                        trades.append(
                            {
                                "date": current_date,
                                "product_idx": idx,
                                "action": "profit_protect",
                                "weight": pos["weight"],
                                "profit_pct": current_profit,
                            }
                        )
                        del positions[idx]
                        self.atr_stop_loss.remove_position(idx)
                        self.profit_protection.remove_position(idx)
                        # 盈利保护不进入冷却期，可以重新买入

            capital += daily_pnl

            # 计算日收益率
            if equity_curve:
                daily_ret = (capital - equity_curve[-1]) / equity_curve[-1]
            else:
                daily_ret = 0.0

            # 动态回撤控制已禁用 (测试表明导致性能下降)
            # 保留代码供未来优化使用
            # if len(equity_curve) > 10:
            #     peak_capital = max(equity_curve)
            #     current_drawdown = (peak_capital - capital) / peak_capital
            #     # 仅在极端情况下触发 (>15%)
            #     if current_drawdown > 0.15:
            #         for idx in list(positions.keys()):
            #             positions[idx]["weight"] *= 0.5

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

            # 清理过期的止损冷却记录
            expired_stops = [idx for idx, stop_t in stopped_positions.items()
                            if t - stop_t > stop_cooldown_days]
            for idx in expired_stops:
                del stopped_positions[idx]

            # 3. 再平衡
            if days_passed % self.rebalance_freq == 0 and t >= start_idx + 30:
                new_positions, new_trades = self._rebalance(
                    features,
                    returns_unsmoothed,
                    masks,
                    dates,
                    t,
                    positions,
                    capital,
                    stopped_positions,  # 传递止损记录
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

    def _compute_adaptive_weights(
        self,
        returns: np.ndarray,
        current_idx: int,
        lookback: int = 60,
    ) -> Dict[str, float]:
        """V3 hisensho: 根据市场状态自适应调整信号权重

        市场状态判断:
        - 高波动期: 增加 regime 权重，降低 momentum 权重 (趋势不稳定)
        - 强趋势期: 增加 momentum 权重，降低 model 权重 (趋势延续)
        - 低波动期: 标准权重 (正常状态)

        Args:
            returns: [n_products, n_dates] 收益率矩阵
            current_idx: 当前时间索引
            lookback: 计算窗口

        Returns:
            调整后的信号权重字典
        """
        if not self.use_adaptive_weights:
            return self.base_signal_weights.copy()

        # 获取最近的市场数据
        start_idx = max(0, current_idx - lookback)
        recent_returns = returns[:, start_idx:current_idx]

        if recent_returns.shape[1] < 20:
            return self.base_signal_weights.copy()

        # 计算市场整体指标 (使用所有产品的平均)
        mean_returns = np.nanmean(recent_returns, axis=0)

        # 1. 市场波动率 (20日滚动标准差的年化值)
        if len(mean_returns) >= 20:
            recent_vol = np.nanstd(mean_returns[-20:]) * np.sqrt(252)
        else:
            recent_vol = np.nanstd(mean_returns) * np.sqrt(252)

        # 2. 市场趋势强度 (20日动量的符号一致性)
        if len(mean_returns) >= 20:
            momentum_20d = np.nansum(mean_returns[-20:])
            momentum_5d = np.nansum(mean_returns[-5:]) if len(mean_returns) >= 5 else 0
            # 趋势一致性: 短期和长期动量同向
            trend_strength = abs(momentum_20d) / (np.nanstd(mean_returns[-20:]) * 20 + 1e-8)
            trend_consistent = np.sign(momentum_20d) == np.sign(momentum_5d)
        else:
            trend_strength = 0
            trend_consistent = False

        # 3. 根据市场状态调整权重
        weights = self.base_signal_weights.copy()

        # 波动率阈值 (年化)
        vol_high_threshold = 0.15  # 15% 以上视为高波动
        vol_low_threshold = 0.05   # 5% 以下视为低波动

        # 趋势阈值
        trend_strong_threshold = 1.5  # 标准差的1.5倍视为强趋势

        if recent_vol > vol_high_threshold:
            # 高波动期: regime 权重提升，momentum 权重降低
            weights["regime"] = min(0.25, weights["regime"] * 1.5)
            weights["momentum"] = max(0.10, weights["momentum"] * 0.6)
            weights["sharpe"] = max(0.25, weights["sharpe"] * 0.9)
            logger.debug(f"[Adaptive] 高波动期 (vol={recent_vol:.2%}), 提升 regime 权重")

        elif trend_strength > trend_strong_threshold and trend_consistent:
            # 强趋势期: momentum 权重提升，model 权重降低
            weights["momentum"] = min(0.40, weights["momentum"] * 1.4)
            weights["model"] = max(0.15, weights["model"] * 0.7)
            logger.debug(f"[Adaptive] 强趋势期 (trend={trend_strength:.2f}), 提升 momentum 权重")

        elif recent_vol < vol_low_threshold:
            # 低波动期: sharpe 权重提升 (稳定收益更重要)
            weights["sharpe"] = min(0.50, weights["sharpe"] * 1.2)
            weights["regime"] = max(0.05, weights["regime"] * 0.7)
            logger.debug(f"[Adaptive] 低波动期 (vol={recent_vol:.2%}), 提升 sharpe 权重")

        # 归一化确保权重和为 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _rebalance(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
        dates: List[str],
        current_idx: int,
        current_positions: Dict[int, Dict],
        capital: float,
        stopped_positions: Optional[Dict[int, int]] = None,
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
            stopped_positions: 最近止损的产品 {product_idx: stop_date_idx}

        Returns:
            (new_positions, trades)
        """
        n_products = features.shape[0]
        current_date = dates[current_idx]
        trades = []
        stopped_positions = stopped_positions or {}

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

        # === Sharpe Optimizer 策略 (基于 Momentum Transformer) ===
        # 核心改进: 直接优化夏普比率 + Changepoint Detection + 多时间尺度动量

        # 1. 使用 SharpeOptimizer 计算增强信号
        sharpe_signals = self.sharpe_optimizer.compute_signals(
            returns, masks, current_idx,
            base_signals=pred["signal_strength"]  # 传入模型基础信号
        )

        # 2. 使用 VolScaledMomentum 计算风险权重
        vol_signals = self.vol_scaled_momentum.compute_signals(returns, masks, current_idx)
        risk_weights = vol_signals["risk_weights"]

        # 3. 四因子融合 (V2 优化):
        # - 模型信号 (30%)
        # - Sharpe优化信号 (35%) - 核心改进
        # - 波动率缩放动量 (20%)
        # - Regime信号 (15%) - Changepoint Detection
        model_signal = pred["signal_strength"]
        sharpe_signal = sharpe_signals["signal_strength"]
        vol_momentum_signal = vol_signals["signal_strength"]
        regime_signal_raw = sharpe_signals["regime_signals"]

        # hisensho 审计修复: 信号标准化
        # 保留原始 min-max 归一化 (rank 归一化导致性能下降)
        def normalize_signal(sig: np.ndarray) -> np.ndarray:
            """将信号归一化到 [0, 1]"""
            sig_min, sig_max = sig.min(), sig.max()
            if sig_max - sig_min > 1e-8:
                return (sig - sig_min) / (sig_max - sig_min)
            return np.clip(sig, 0, 1)

        # 标准化各信号 (保持原始实现)
        model_signal_norm = np.clip(model_signal, 0, 1)
        sharpe_signal_norm = np.clip(sharpe_signal, 0, 1)
        vol_momentum_norm = np.clip(vol_momentum_signal, 0, 1)
        regime_signal_norm = (regime_signal_raw + 2) / 4  # [-2, 2] -> [0, 1]
        regime_signal_norm = np.clip(regime_signal_norm, 0, 1)

        # V3 hisensho: 自适应权重调整
        sw = self._compute_adaptive_weights(returns, current_idx)
        self.signal_weights = sw  # 保存当前权重用于日志

        # 信号融合 (使用自适应权重)
        enhanced_signal = (
            model_signal_norm * sw["model"] +
            sharpe_signal_norm * sw["sharpe"] +
            vol_momentum_norm * sw["momentum"] +
            regime_signal_norm * sw["regime"]
        )

        # 4. CPD 调整: 在 changepoint 时降低信号强度
        cpd_adjustment = sharpe_signals["cpd_adjustment"]
        enhanced_signal = enhanced_signal * cpd_adjustment

        # 5. 应用等风险贡献作为后处理权重调整因子
        # 高波动产品信号衰减，低波动产品信号增强 (恢复原始实现)
        vol_adjustment = np.clip(risk_weights * n_products, 0.5, 2.0)
        enhanced_signal = enhanced_signal * vol_adjustment

        # 筛选候选产品 (使用可配置阈值)
        # 排除仍在止损冷却期的产品
        candidates = []
        for i in range(n_products):
            if masks[i, current_idx] > 0 and enhanced_signal[i] >= self.signal_threshold:
                # 检查是否在止损冷却期
                if i not in stopped_positions:
                    candidates.append(i)
                else:
                    logger.debug(f"[Backtest] 产品 {i} 在止损冷却期，跳过")

        logger.debug(f"[Backtest] 候选产品: {len(candidates)}, 最强增强信号: {enhanced_signal.max():.3f}")

        if not candidates:
            return current_positions, trades

        # HRP 优化 - 使用增强信号
        hrp_weights = optimize_portfolio(
            returns,
            masks,
            candidates,
            signals={i: float(enhanced_signal[i]) for i in candidates},
            lookback=60,
            current_idx=current_idx,
        )

        logger.debug(f"[Backtest] HRP 权重数量: {len(hrp_weights)}, 总权重: {sum(hrp_weights.values()) if hrp_weights else 0:.3f}")

        if not hrp_weights:
            logger.warning(f"[Backtest] HRP 返回空权重, 候选数: {len(candidates)}")
            return current_positions, trades

        # 仓位分配 - 使用增强信号
        signals_dict = {i: float(enhanced_signal[i]) for i in candidates}
        allocations = self.position_sizer.allocate(hrp_weights, signals_dict)

        logger.debug(f"[Backtest] 仓位分配: {len(allocations)} 个, 信号强度范围: {min(signals_dict.values()):.3f}-{max(signals_dict.values()):.3f}")

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

            # 初始化盈利保护
            self.profit_protection.initialize_position(idx, 1.0)

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
                self.profit_protection.remove_position(idx)

        return new_positions, trades

    def _get_first_valid_idx(self, masks: np.ndarray) -> np.ndarray:
        """获取每个产品的首个有效索引 (向量化)"""
        n_products, n_dates = masks.shape
        # 使用 argmax 找到每行第一个 True 的位置
        # 对于全0行，argmax 返回0，需要特殊处理
        has_valid = masks.any(axis=1)
        first_valid = np.where(has_valid, masks.argmax(axis=1), n_dates)
        return first_valid.astype(np.int32)

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

        # V3 hisensho: 夏普比率计算优化 (兼容短期回测)
        # 根据数据长度选择最佳计算方式
        sharpe_ratio = self._compute_sharpe_ratio(equity, returns, n_days)

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

    def _compute_sharpe_ratio(
        self,
        equity: pd.Series,
        returns: pd.Series,
        n_days: int,
    ) -> float:
        """V3 hisensho: 自适应夏普比率计算

        根据数据长度选择最佳计算频率:
        - >= 12个月: 使用月度收益率 (最准确)
        - >= 3个月: 使用周度收益率 (折中方案)
        - < 3个月: 使用日度收益率 (有统计噪音警告)

        Args:
            equity: 净值序列
            returns: 日收益率序列
            n_days: 交易天数

        Returns:
            年化夏普比率
        """
        # 最低波动率下限 (年化 1%)
        MIN_ANNUAL_VOL = 0.01

        try:
            equity_dt = equity.copy()
            equity_dt.index = pd.to_datetime(equity_dt.index)

            # 根据数据长度选择计算频率
            n_months = n_days / 21  # 约21个交易日/月
            n_weeks = n_days / 5    # 约5个交易日/周

            if n_months >= 12:
                # 足够长: 使用月度收益率
                return self._sharpe_monthly(equity_dt, MIN_ANNUAL_VOL)

            elif n_months >= 3:
                # 中等长度: 使用周度收益率
                return self._sharpe_weekly(equity_dt, MIN_ANNUAL_VOL)

            elif n_days >= 20:
                # 短期: 使用日度收益率 (带警告)
                logger.debug(f"[Sharpe] 数据较短 ({n_days}天), 使用日度计算, 结果可能有统计噪音")
                return self._sharpe_daily(returns, MIN_ANNUAL_VOL)

            else:
                # 极短期: 返回保守估计
                logger.warning(f"[Sharpe] 数据过短 ({n_days}天), 夏普比率可能不可靠")
                return self._sharpe_daily(returns, MIN_ANNUAL_VOL)

        except Exception as e:
            logger.debug(f"[Sharpe] 计算异常: {e}, 使用日度回退")
            return self._sharpe_daily(returns, MIN_ANNUAL_VOL)

    def _sharpe_monthly(self, equity_dt: pd.Series, min_vol: float) -> float:
        """月度夏普比率计算"""
        try:
            monthly_equity = equity_dt.resample('ME').last()
        except ValueError:
            monthly_equity = equity_dt.resample('M').last()  # pandas < 2.0

        monthly_returns = monthly_equity.pct_change().dropna()

        if len(monthly_returns) < 2:
            return 0.0

        monthly_mean = monthly_returns.mean()
        monthly_std = max(monthly_returns.std(), min_vol / np.sqrt(12))

        rf_monthly = self.risk_free_rate / 12
        excess_monthly = monthly_mean - rf_monthly

        return float(excess_monthly / monthly_std * np.sqrt(12))

    def _sharpe_weekly(self, equity_dt: pd.Series, min_vol: float) -> float:
        """周度夏普比率计算 (中等数据量的折中方案)"""
        try:
            weekly_equity = equity_dt.resample('W').last()
        except ValueError:
            weekly_equity = equity_dt.resample('W-FRI').last()

        weekly_returns = weekly_equity.pct_change().dropna()

        if len(weekly_returns) < 4:
            return 0.0

        weekly_mean = weekly_returns.mean()
        weekly_std = max(weekly_returns.std(), min_vol / np.sqrt(52))

        rf_weekly = self.risk_free_rate / 52
        excess_weekly = weekly_mean - rf_weekly

        return float(excess_weekly / weekly_std * np.sqrt(52))

    def _sharpe_daily(self, returns: pd.Series, min_vol: float) -> float:
        """日度夏普比率计算 (短期数据)"""
        if len(returns) < 5:
            return 0.0

        daily_mean = returns.mean()
        daily_std = max(returns.std(), min_vol / np.sqrt(252))

        rf_daily = self.risk_free_rate / 252
        excess_daily = daily_mean - rf_daily

        return float(excess_daily / daily_std * np.sqrt(252))

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
