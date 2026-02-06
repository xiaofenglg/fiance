# -*- coding: utf-8 -*-
"""
V12 Phase 3: VectorBT PRO Portfolio Simulator

High-fidelity backtesting with:
- T+N redemption constraint modeling
- Real fee schedule lookup
- Multi-asset portfolio simulation
- VectorBT PRO integration (optional)

hisensho quant audit compliance:
- frozen_pool mechanism for T+N redemption delay
- Fee lookup from parsed_fees table
- No future data leakage in order execution

Usage:
    simulator = VBTPortfolioSimulator(prices, signals, t_plus_n=1)
    portfolio = simulator.run_backtest(rebalance_freq="W-WED", max_positions=6)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional VectorBT PRO import
try:
    import vectorbtpro as vbt
    _vbt_available = True
except ImportError:
    try:
        import vectorbt as vbt
        _vbt_available = True
    except ImportError:
        _vbt_available = False
        logger.debug("VectorBT not available, using native simulator")


@dataclass
class Order:
    """Order representation for T+N tracking"""
    product_code: str
    order_date: datetime
    settle_date: datetime
    quantity: float
    nav_at_order: float
    fee_rate: float
    is_buy: bool
    status: str = "pending"  # pending, settled, cancelled
    nav_at_settle: Optional[float] = None
    proceeds: Optional[float] = None


@dataclass
class Position:
    """Position tracking with proper units (not weights)

    hisensho audit fix: Track actual units instead of weights.
    Units = Cash invested / NAV at purchase
    Market value = Units * Current NAV
    """
    product_code: str
    units: float  # Number of units held (not weight!)
    avg_cost_per_unit: float  # Average cost per unit (NAV at purchase)
    entry_date: datetime
    holding_days: int = 0

    def market_value(self, current_nav: float) -> float:
        """Calculate current market value via Mark-to-Market"""
        return self.units * current_nav

    def cost_basis(self) -> float:
        """Calculate total cost basis"""
        return self.units * self.avg_cost_per_unit

    def unrealized_pnl(self, current_nav: float) -> float:
        """Calculate unrealized P&L"""
        return self.market_value(current_nav) - self.cost_basis()


@dataclass
class Receivable:
    """T+N pending settlement receivable

    hisensho audit: Track receivables separately from cash.
    Receivables represent money that will be received after T+N days.

    hisensho audit fix (iteration 2): Added fee_rate for proper settlement calculation.
    """
    product_code: str
    settle_date: datetime
    expected_amount: float  # Estimated amount (calculated at order time)
    order_nav: float  # NAV at order time
    units_sold: float  # Units being redeemed
    fee_rate: float = 0.0  # Fee rate for settlement calculation


@dataclass
class SimulationResult:
    """Simulation result"""
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions_history: List[Dict]
    trades: List[Dict]
    metrics: Dict[str, float]
    cash_history: Optional[pd.Series] = None  # Track cash over time


class FeeScheduleLookup:
    """Fee schedule lookup from database"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to SQLite database with parsed_fees table
        """
        self.db_path = db_path
        self._fee_cache: Dict[Tuple[str, str], List[Tuple[int, float, str]]] = {}
        self._load_fees()

    def _load_fees(self):
        """Load fee schedule from database"""
        if not self.db_path or not Path(self.db_path).exists():
            logger.debug("[FeeSchedule] No database, using default fees")
            return

        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if parsed_fees table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='parsed_fees'"
            )
            if not cursor.fetchone():
                logger.debug("[FeeSchedule] parsed_fees table not found")
                conn.close()
                return

            # Load fee rules
            cursor.execute("""
                SELECT bank_name, product_code, days_threshold, fee_rate, comparison_type
                FROM parsed_fees
                ORDER BY bank_name, product_code, days_threshold
            """)

            for row in cursor.fetchall():
                bank, product, days_thresh, rate, cmp_type = row
                key = (bank, product)
                if key not in self._fee_cache:
                    self._fee_cache[key] = []
                self._fee_cache[key].append((days_thresh, rate, cmp_type))

            conn.close()
            logger.info(f"[FeeSchedule] Loaded fees for {len(self._fee_cache)} products")

        except Exception as e:
            logger.warning(f"[FeeSchedule] Failed to load fees: {e}")

    def get_fee(
        self,
        bank_name: str,
        product_code: str,
        holding_days: int,
    ) -> float:
        """
        Get redemption fee rate.

        Args:
            bank_name: Bank name
            product_code: Product code
            holding_days: Days held

        Returns:
            Fee rate (0.0 - 1.0)
        """
        key = (bank_name, product_code)

        if key in self._fee_cache:
            for days_thresh, rate, cmp_type in self._fee_cache[key]:
                if cmp_type == "less" and holding_days < days_thresh:
                    return rate
                elif cmp_type == "greater_eq" and holding_days >= days_thresh:
                    return rate
            # Default to last rule
            if self._fee_cache[key]:
                return self._fee_cache[key][-1][1]

        # Default fee schedule
        if holding_days < 7:
            return 0.015  # 1.5% for < 7 days
        elif holding_days < 30:
            return 0.005  # 0.5% for 7-30 days
        elif holding_days < 180:
            return 0.002  # 0.2% for 30-180 days
        else:
            return 0.0  # Free for >= 180 days


class TNRedemptionSimulator:
    """T+N Redemption Constraint Simulator

    hisensho audit compliance:
    - Proper cash ledger (self.cash)
    - Positions tracked in units, not weights
    - Receivables tracked separately
    - Mark-to-Market for equity calculation

    hisensho audit fix (iteration 3):
    - Added settlement_pricing parameter for WMP business logic
    """

    def __init__(
        self,
        fee_lookup: FeeScheduleLookup,
        default_t_plus_n: int = 1,
        product_t_plus_n: Optional[Dict[str, int]] = None,
        bank_name: str = "中信银行",
        init_cash: float = 0.0,
        settlement_pricing: str = "order_date",
    ):
        """
        Args:
            fee_lookup: Fee schedule lookup instance
            default_t_plus_n: Default T+N days
            product_t_plus_n: Product-specific T+N configuration
            bank_name: Bank name for fee lookup
            init_cash: Initial cash balance
            settlement_pricing: "order_date" (use NAV at T, typical for WMP) or
                              "settle_date" (use NAV at T+N, for some funds)
        """
        self.fee_lookup = fee_lookup
        self.default_n = default_t_plus_n
        self.product_n = product_t_plus_n or {}
        self.bank_name = bank_name

        # hisensho audit fix: Settlement pricing mode
        # "order_date": Standard WMP - redemption at order date NAV
        # "settle_date": Some funds settle at T+N NAV
        self.settlement_pricing = settlement_pricing

        # === hisensho audit fix: Proper state variables ===
        # 1. Cash ledger - dynamic cash balance
        self.cash: float = init_cash

        # 2. Positions - tracked in UNITS, not weights
        self.positions: Dict[str, Position] = {}

        # 3. Receivables - T+N pending settlements
        self.receivables: List[Receivable] = []

        # Order tracking (for audit trail)
        self.pending_orders: List[Order] = []
        self.settled_orders: List[Order] = []

        # Frozen capital (T+N pending) - for backward compatibility
        self.frozen_pool: float = 0.0

    def execute_buy(
        self,
        product_code: str,
        amount: float,
        current_date: datetime,
        nav: float,
    ) -> Order:
        """
        Execute buy order (T+0 settlement for buy).

        hisensho audit fix:
        - Parameter is now 'amount' (cash to invest), not 'quantity'
        - Units = amount / NAV
        - Deduct cash immediately

        Args:
            product_code: Product code
            amount: Cash amount to invest (NOT units!)
            current_date: Current date
            nav: Current NAV

        Returns:
            Order object
        """
        # Check sufficient cash
        if amount > self.cash:
            logger.warning(
                f"[T+N] Insufficient cash for buy: need {amount:.2f}, have {self.cash:.2f}"
            )
            amount = self.cash  # Buy what we can afford

        if amount <= 0:
            return Order(
                product_code=product_code,
                order_date=current_date,
                settle_date=current_date,
                quantity=0,
                nav_at_order=nav,
                fee_rate=0.0,
                is_buy=True,
                status="cancelled",
            )

        # Calculate units purchased
        # Guard against invalid NAV (0, NaN, inf)
        if nav <= 0 or np.isnan(nav) or np.isinf(nav):
            logger.warning(f"[T+N] Invalid NAV {nav} for {product_code}, skipping buy")
            return Order(
                product_code=product_code,
                order_date=current_date,
                settle_date=current_date,
                quantity=0,
                nav_at_order=nav,
                fee_rate=0.0,
                is_buy=True,
                status="cancelled",
            )
        units = amount / nav

        order = Order(
            product_code=product_code,
            order_date=current_date,
            settle_date=current_date,  # T+0 for buy
            quantity=units,  # Store actual units
            nav_at_order=nav,
            fee_rate=0.0,  # No fee for buying
            is_buy=True,
            status="settled",
            nav_at_settle=nav,
        )

        # Deduct cash
        self.cash -= amount

        # Update position with proper unit tracking
        if product_code in self.positions:
            pos = self.positions[product_code]
            total_units = pos.units + units
            # Weighted average cost
            pos.avg_cost_per_unit = (
                pos.avg_cost_per_unit * pos.units + nav * units
            ) / total_units
            pos.units = total_units
        else:
            self.positions[product_code] = Position(
                product_code=product_code,
                units=units,
                avg_cost_per_unit=nav,
                entry_date=current_date,
            )

        self.settled_orders.append(order)

        logger.debug(
            f"[T+N] Buy {product_code}: amount={amount:.2f}, units={units:.4f}, "
            f"nav={nav:.4f}, cash_remain={self.cash:.2f}"
        )

        return order

    def execute_sell(
        self,
        product_code: str,
        units_to_sell: float,
        current_date: datetime,
        nav: float,
    ) -> Order:
        """
        Execute sell order (T+N settlement).

        hisensho audit fix:
        - Parameter is 'units_to_sell', not weight
        - Create Receivable for T+N settlement
        - Don't add to cash until settlement

        Args:
            product_code: Product code
            units_to_sell: Number of units to sell
            current_date: Current date
            nav: Current NAV

        Returns:
            Order object
        """
        # Check sufficient units
        if product_code not in self.positions:
            logger.warning(f"[T+N] No position to sell: {product_code}")
            return Order(
                product_code=product_code,
                order_date=current_date,
                settle_date=current_date,
                quantity=0,
                nav_at_order=nav,
                fee_rate=0.0,
                is_buy=False,
                status="cancelled",
            )

        pos = self.positions[product_code]
        if units_to_sell > pos.units:
            logger.warning(
                f"[T+N] Selling more than held: {units_to_sell:.4f} > {pos.units:.4f}"
            )
            units_to_sell = pos.units

        if units_to_sell <= 0:
            return Order(
                product_code=product_code,
                order_date=current_date,
                settle_date=current_date,
                quantity=0,
                nav_at_order=nav,
                fee_rate=0.0,
                is_buy=False,
                status="cancelled",
            )

        # Guard against invalid NAV
        if np.isnan(nav) or nav <= 0:
            logger.warning(f"[T+N] Invalid NAV {nav} for sell {product_code}, skipping")
            return Order(
                product_code=product_code,
                order_date=current_date,
                settle_date=current_date,
                quantity=0,
                nav_at_order=0,
                fee_rate=0.0,
                is_buy=False,
                status="cancelled",
            )

        n_days = self.product_n.get(product_code, self.default_n)
        settle_date = self._add_trading_days(current_date, n_days)

        # Calculate holding days
        delta = current_date - pos.entry_date
        holding_days = delta.days

        # Get fee rate
        fee_rate = self.fee_lookup.get_fee(
            self.bank_name, product_code, holding_days
        )

        # Calculate expected proceeds (based on current NAV)
        gross_proceeds = units_to_sell * nav
        expected_proceeds = gross_proceeds * (1 - fee_rate)

        order = Order(
            product_code=product_code,
            order_date=current_date,
            settle_date=settle_date,
            quantity=units_to_sell,
            nav_at_order=nav,
            fee_rate=fee_rate,
            is_buy=False,
            status="pending",
        )

        # Create receivable for T+N tracking
        receivable = Receivable(
            product_code=product_code,
            settle_date=settle_date,
            expected_amount=expected_proceeds,
            order_nav=nav,
            units_sold=units_to_sell,
            fee_rate=fee_rate,  # hisensho fix: Store fee_rate for settlement
        )
        self.receivables.append(receivable)

        # Frozen pool for backward compatibility
        self.frozen_pool += expected_proceeds
        self.pending_orders.append(order)

        # Update position (remove units immediately - they're locked for settlement)
        pos.units -= units_to_sell
        if pos.units <= 1e-9:  # Clean up tiny residuals
            del self.positions[product_code]

        logger.debug(
            f"[T+N] Sell {product_code}: units={units_to_sell:.4f}, "
            f"nav={nav:.4f}, fee={fee_rate:.4f}, "
            f"settle={settle_date.strftime('%Y-%m-%d')}, "
            f"expected_proceeds={expected_proceeds:.2f}"
        )

        return order

    def settle_orders(
        self,
        current_date: datetime,
        nav_lookup: Dict[str, float],
    ) -> Tuple[float, List[Order]]:
        """
        Settle due orders.

        hisensho audit fix:
        - Settle receivables and add to cash
        - Use NAV at settlement time (may differ from order NAV)
        - Properly update frozen pool

        Args:
            current_date: Current date
            nav_lookup: Dict of product_code -> current NAV

        Returns:
            (released_capital, settled_orders)
        """
        released = 0.0
        settled = []

        # Settle receivables
        for receivable in list(self.receivables):
            if receivable.settle_date <= current_date:
                # hisensho audit fix (iteration 3): Settlement pricing mode
                # For WMP products: typically use order date NAV
                # For some funds: may use settlement date NAV
                if self.settlement_pricing == "order_date":
                    # Standard WMP: Redemption price = NAV at order date (T)
                    # This is the expected_amount already calculated at order time
                    actual_proceeds = receivable.expected_amount
                    nav_used = receivable.order_nav
                else:
                    # Alternative: Redemption price = NAV at settlement date (T+N)
                    nav_settle = nav_lookup.get(
                        receivable.product_code, receivable.order_nav
                    )
                    actual_proceeds = (
                        receivable.units_sold * nav_settle * (1 - receivable.fee_rate)
                    )
                    nav_used = nav_settle

                # Add to cash
                self.cash += actual_proceeds
                released += actual_proceeds

                # Update frozen pool (remove the estimated amount)
                self.frozen_pool = max(0, self.frozen_pool - receivable.expected_amount)

                # Remove from receivables
                self.receivables.remove(receivable)

                logger.debug(
                    f"[T+N] Settled {receivable.product_code}: "
                    f"units={receivable.units_sold:.4f}, nav_used={nav_used:.4f}, "
                    f"pricing={self.settlement_pricing}, "
                    f"actual_proceeds={actual_proceeds:.2f}, cash_now={self.cash:.2f}"
                )

        # Update order status
        for order in list(self.pending_orders):
            if order.settle_date <= current_date and order.status == "pending":
                nav_settle = nav_lookup.get(order.product_code, order.nav_at_order)
                order.nav_at_settle = nav_settle

                if order.is_buy:
                    order.proceeds = 0.0
                else:
                    order.proceeds = order.quantity * nav_settle * (1 - order.fee_rate)

                order.status = "settled"
                settled.append(order)
                self.pending_orders.remove(order)
                self.settled_orders.append(order)

        return released, settled

    def get_available_capital(self, total_capital: float = None) -> float:
        """Get capital available for new orders.

        hisensho audit fix: Use self.cash instead of external total_capital.
        """
        # New behavior: return actual cash balance
        return self.cash

    def get_total_equity(self, nav_lookup: Dict[str, float]) -> float:
        """
        Calculate total portfolio equity via Mark-to-Market.

        hisensho audit fix:
        Equity = Cash + Sum(Units * Current_NAV) + Receivables

        Args:
            nav_lookup: Dict of product_code -> current NAV

        Returns:
            Total portfolio equity
        """
        equity = self.cash

        # Mark-to-Market: Value positions at current NAV
        for product_code, pos in self.positions.items():
            current_nav = nav_lookup.get(product_code, pos.avg_cost_per_unit)
            # Guard against NaN/invalid NAV
            if pd.isna(current_nav) or current_nav <= 0:
                current_nav = pos.avg_cost_per_unit
            pos_value = pos.market_value(current_nav)
            if not np.isnan(pos_value):
                equity += pos_value

        # Add receivables (money in transit)
        for receivable in self.receivables:
            if not np.isnan(receivable.expected_amount):
                equity += receivable.expected_amount

        return equity

    def get_positions_value(self, nav_lookup: Dict[str, float]) -> float:
        """Get total value of all positions"""
        total = 0.0
        for product_code, pos in self.positions.items():
            current_nav = nav_lookup.get(product_code, pos.avg_cost_per_unit)
            total += pos.market_value(current_nav)
        return total

    def get_receivables_value(self) -> float:
        """Get total value of pending receivables"""
        return sum(r.expected_amount for r in self.receivables)

    def update_holding_days(self):
        """Update holding days for all positions"""
        for pos in self.positions.values():
            pos.holding_days += 1

    def _add_trading_days(self, date: datetime, n_days: int) -> datetime:
        """Add N trading days (simplified: skip weekends)"""
        result = date
        added = 0
        while added < n_days:
            result += timedelta(days=1)
            if result.weekday() < 5:  # Mon-Fri
                added += 1
        return result


class VBTPortfolioSimulator:
    """VectorBT PRO Portfolio Simulator"""

    def __init__(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        t_plus_n: int = 1,
        db_path: Optional[str] = None,
        bank_name: str = "中信银行",
        settlement_pricing: str = "order_date",
    ):
        """
        Args:
            prices: DataFrame [dates, products] with NAV prices
            signals: DataFrame [dates, products] with alpha signals
            t_plus_n: T+N redemption days
            db_path: Path to SQLite for fee lookup
            bank_name: Bank name
            settlement_pricing: "order_date" (use NAV at T, typical for WMP) or
                              "settle_date" (use NAV at T+N, for some funds)
        """
        self.prices = prices
        self.signals = signals
        self.t_plus_n = t_plus_n
        self.bank_name = bank_name
        self.settlement_pricing = settlement_pricing

        # Fee lookup
        self.fee_lookup = FeeScheduleLookup(db_path)

        # T+N simulator
        self.tn_simulator = TNRedemptionSimulator(
            fee_lookup=self.fee_lookup,
            default_t_plus_n=t_plus_n,
            bank_name=bank_name,
            settlement_pricing=settlement_pricing,
        )

        # hisensho audit fix: Deferred buy queue for T+N cash flow management
        # When we can't buy due to insufficient cash (pending T+N settlement),
        # we defer the buy order to execute when cash becomes available
        self._deferred_buys: List[Dict] = []

    def run_backtest(
        self,
        rebalance_freq: str = "W-WED",
        max_positions: int = 6,
        init_cash: float = 1e8,
        use_vbt: bool = True,
    ) -> SimulationResult:
        """
        Run portfolio backtest.

        Args:
            rebalance_freq: Rebalance frequency (pandas offset string)
            max_positions: Maximum number of positions (k=6 cardinality constraint)
            init_cash: Initial capital
            use_vbt: Whether to use VectorBT PRO (if available)

        Returns:
            SimulationResult
        """
        if use_vbt and _vbt_available:
            return self._run_vbt_backtest(rebalance_freq, max_positions, init_cash)
        else:
            return self._run_native_backtest(rebalance_freq, max_positions, init_cash)

    def _run_vbt_backtest(
        self,
        rebalance_freq: str,
        max_positions: int,
        init_cash: float,
    ) -> SimulationResult:
        """Run backtest using VectorBT PRO"""
        logger.info("[VBT] Running VectorBT backtest")

        # Generate rebalance dates
        rebalance_dates = self.prices.resample(rebalance_freq).last().index

        # Compute target weights
        target_weights = self._compute_target_weights(rebalance_dates, max_positions)

        # Apply T+N delay
        delayed_weights = target_weights.shift(self.t_plus_n).fillna(0)

        # Get fee array
        fee_array = self._get_fee_array()

        try:
            # VectorBT portfolio simulation
            portfolio = vbt.Portfolio.from_orders(
                close=self.prices,
                size=delayed_weights,
                size_type="targetpercent",
                init_cash=init_cash,
                fees=fee_array,
                freq="D",
            )

            # Extract results
            equity_curve = portfolio.value()
            daily_returns = portfolio.returns()

            metrics = {
                "total_return": portfolio.total_return(),
                "annual_return": portfolio.annualized_return(),
                "sharpe_ratio": portfolio.sharpe_ratio(),
                "max_drawdown": portfolio.max_drawdown(),
            }

            return SimulationResult(
                equity_curve=equity_curve,
                daily_returns=daily_returns,
                positions_history=[],
                trades=[],
                metrics=metrics,
            )

        except Exception as e:
            logger.warning(f"[VBT] VectorBT failed: {e}, falling back to native")
            return self._run_native_backtest(rebalance_freq, max_positions, init_cash)

    def _run_native_backtest(
        self,
        rebalance_freq: str,
        max_positions: int,
        init_cash: float,
    ) -> SimulationResult:
        """
        Run backtest using native simulator.

        hisensho audit fix: Complete rewrite with proper accounting
        - Cash ledger (self.tn_simulator.cash)
        - Positions in units (not weights)
        - Mark-to-Market equity calculation
        - Receivables tracking
        """
        logger.info("[VBT] Running native backtest (hisensho compliant)")

        # Reset T+N simulator with initial cash
        self.tn_simulator = TNRedemptionSimulator(
            fee_lookup=self.fee_lookup,
            default_t_plus_n=self.t_plus_n,
            bank_name=self.bank_name,
            init_cash=init_cash,  # Set initial cash balance
            settlement_pricing=self.settlement_pricing,
        )

        # Reset deferred buy queue
        self._deferred_buys = []

        dates = self.prices.index.tolist()
        products = self.prices.columns.tolist()
        n_dates = len(dates)

        # Generate rebalance dates (使用实际交易日历,避免假日跳过)
        if rebalance_freq in ("BMS", "MS"):
            # 每月第一个实际交易日
            rebalance_dates = set(
                self.prices.groupby(self.prices.index.to_period('M')).apply(
                    lambda x: x.index[0]
                ).tolist()
            )
        else:
            rebalance_dates = set(self.prices.resample(rebalance_freq).last().index.tolist())

        # State tracking
        equity_curve = []
        cash_curve = []
        daily_returns_list = []
        positions_history = []
        trades = []

        for t, date in enumerate(dates):
            current_date = (
                pd.Timestamp(date).to_pydatetime()
                if not isinstance(date, datetime)
                else date
            )

            # Build NAV lookup for this date (exclude NaN/invalid values)
            nav_lookup = {}
            for p in products:
                try:
                    nav = self.prices.loc[date, p]
                    # Only include valid NAV values
                    if pd.notna(nav) and nav > 0.1:  # NAV should be > 0.1 for valid products
                        nav_lookup[p] = float(nav)
                except KeyError:
                    pass

            # === Step 1: Settle receivables (T+N settlements) ===
            released, settled = self.tn_simulator.settle_orders(current_date, nav_lookup)

            # === Step 1.5: Process deferred buys (hisensho audit fix for cash drag) ===
            if released > 0 and self._deferred_buys:
                deferred_trades = self._process_deferred_buys(current_date, nav_lookup)
                trades.extend(deferred_trades)

            # === Step 2: Mark-to-Market equity calculation ===
            # Equity = Cash + Positions_Value + Receivables_Value
            equity = self.tn_simulator.get_total_equity(nav_lookup)
            equity_curve.append(equity)
            cash_curve.append(self.tn_simulator.cash)

            # === Step 3: Calculate daily return ===
            if t > 0 and equity_curve[-2] > 0:
                daily_ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
            else:
                daily_ret = 0.0
            daily_returns_list.append(daily_ret)

            # === Step 4: Record positions snapshot ===
            position_snapshot = {}
            for prod, pos in self.tn_simulator.positions.items():
                position_snapshot[prod] = {
                    "units": pos.units,
                    "avg_cost": pos.avg_cost_per_unit,
                    "market_value": pos.market_value(nav_lookup.get(prod, pos.avg_cost_per_unit)),
                }

            positions_history.append({
                "date": date,
                "equity": equity,
                "cash": self.tn_simulator.cash,
                "positions_value": self.tn_simulator.get_positions_value(nav_lookup),
                "receivables_value": self.tn_simulator.get_receivables_value(),
                "positions": position_snapshot,
                "n_positions": len(self.tn_simulator.positions),
            })

            # === Step 5: Rebalance on scheduled dates ===
            if date in rebalance_dates and t >= 5:
                rebal_trades = self._rebalance_with_units(
                    date, nav_lookup, max_positions
                )
                trades.extend(rebal_trades)

            # === Step 6: Update holding days ===
            self.tn_simulator.update_holding_days()

        # Convert to Series
        equity_series = pd.Series(equity_curve, index=dates)
        cash_series = pd.Series(cash_curve, index=dates)
        returns_series = pd.Series(daily_returns_list, index=dates)

        # Compute metrics
        metrics = self._compute_metrics(equity_series, returns_series)

        return SimulationResult(
            equity_curve=equity_series,
            daily_returns=returns_series,
            positions_history=positions_history,
            trades=trades,
            metrics=metrics,
            cash_history=cash_series,
        )

    def _compute_target_weights(
        self,
        rebalance_dates: pd.DatetimeIndex,
        max_positions: int,
    ) -> pd.DataFrame:
        """
        Compute target weights based on alpha signals.

        hisensho audit: Only uses signals available at rebalance date (no future data).

        Args:
            rebalance_dates: Dates to rebalance
            max_positions: Maximum positions (k=6)

        Returns:
            DataFrame with target weights
        """
        weights = pd.DataFrame(
            0.0, index=self.prices.index, columns=self.prices.columns
        )

        for date in rebalance_dates:
            if date not in self.signals.index:
                continue

            scores = self.signals.loc[date]

            # Get top-k products with equal weights
            valid_scores = scores.dropna()
            if len(valid_scores) == 0:
                continue
            if len(valid_scores) <= max_positions:
                top_k = valid_scores.index
            else:
                top_k = valid_scores.nlargest(max_positions).index

            weight = 1.0 / len(top_k) if len(top_k) > 0 else 0.0
            weights.loc[date, top_k] = weight

        # Forward fill weights
        weights = weights.ffill()

        return weights

    def _rebalance_with_units(
        self,
        date,
        nav_lookup: Dict[str, float],
        max_positions: int,
    ) -> List[Dict]:
        """
        Execute rebalance with proper unit-based accounting.

        hisensho audit fix:
        - Calculate target allocation in cash amounts
        - Convert to units using current NAV
        - Sell first to free up cash, then buy
        """
        trades = []
        current_date = (
            pd.Timestamp(date).to_pydatetime()
            if not isinstance(date, datetime)
            else date
        )

        if date not in self.signals.index:
            return trades

        scores = self.signals.loc[date]
        valid_scores = scores.dropna()

        if len(valid_scores) == 0:
            return trades

        # Get top-k products
        if len(valid_scores) <= max_positions:
            top_k_scores = valid_scores
        else:
            top_k_scores = valid_scores.nlargest(max_positions)

        # Filter to products we have prices for
        top_k_scores = top_k_scores[[p for p in top_k_scores.index if p in nav_lookup]]

        # hisensho fix: 排除当前NAV过低的问题产品(使用当前时点数据,无前瞻偏差)
        # 过滤NAV < 0.9的产品,避免买入已大幅亏损的distressed产品
        top_k_scores = top_k_scores[
            [p for p in top_k_scores.index if nav_lookup.get(p, 0) >= 0.9]
        ]
        top_k = set(top_k_scores.index)

        if len(top_k) == 0:
            return trades

        # Current holdings
        current_holdings = set(self.tn_simulator.positions.keys())

        # === Step 1: Sell positions not in top-k ===
        positions_to_sell = current_holdings - top_k
        for product in positions_to_sell:
            if product in self.tn_simulator.positions:
                pos = self.tn_simulator.positions[product]
                nav = nav_lookup.get(product, pos.avg_cost_per_unit)

                order = self.tn_simulator.execute_sell(
                    product_code=product,
                    units_to_sell=pos.units,  # Sell all units
                    current_date=current_date,
                    nav=nav,
                )

                trades.append({
                    "date": date,
                    "product": product,
                    "action": "sell",
                    "units": order.quantity,
                    "nav": nav,
                    "fee_rate": order.fee_rate,
                    "status": order.status,
                })

        # === Step 2: Calculate target allocation (equal weight) ===
        total_equity = self.tn_simulator.get_total_equity(nav_lookup)
        target_per_position = total_equity / len(top_k) if len(top_k) > 0 else 0

        # === Step 3: Buy new positions and rebalance existing ===
        for product in top_k:
            nav = nav_lookup.get(product, 1.0)

            if product in self.tn_simulator.positions:
                # Existing position - check if rebalance needed
                pos = self.tn_simulator.positions[product]
                current_value = pos.market_value(nav)
                diff = target_per_position - current_value

                if abs(diff) > target_per_position * 0.05:  # 5% threshold
                    if diff > 0:
                        # Need to buy more
                        buy_amount = min(diff, self.tn_simulator.cash)
                        if buy_amount > 100:  # Min buy amount
                            order = self.tn_simulator.execute_buy(
                                product_code=product,
                                amount=buy_amount,
                                current_date=current_date,
                                nav=nav,
                            )
                            trades.append({
                                "date": date,
                                "product": product,
                                "action": "buy",
                                "amount": buy_amount,
                                "units": order.quantity,
                                "nav": nav,
                            })
                    else:
                        # Need to sell some
                        sell_value = abs(diff)
                        units_to_sell = sell_value / nav
                        if units_to_sell > 0.01:
                            order = self.tn_simulator.execute_sell(
                                product_code=product,
                                units_to_sell=units_to_sell,
                                current_date=current_date,
                                nav=nav,
                            )
                            trades.append({
                                "date": date,
                                "product": product,
                                "action": "sell",
                                "units": order.quantity,
                                "nav": nav,
                                "fee_rate": order.fee_rate,
                            })
            else:
                # New position - buy
                available_cash = self.tn_simulator.cash
                buy_amount = min(target_per_position, available_cash)

                if buy_amount > 100:  # Min buy amount
                    order = self.tn_simulator.execute_buy(
                        product_code=product,
                        amount=buy_amount,
                        current_date=current_date,
                        nav=nav,
                    )
                    trades.append({
                        "date": date,
                        "product": product,
                        "action": "buy",
                        "amount": buy_amount,
                        "units": order.quantity,
                        "nav": nav,
                    })

                    # hisensho audit fix: If partial buy, defer the rest
                    if buy_amount < target_per_position * 0.95:
                        deferred_amount = target_per_position - buy_amount
                        self._deferred_buys.append({
                            "product": product,
                            "amount": deferred_amount,
                            "original_date": date,
                        })
                        logger.debug(
                            f"[T+N] Deferred buy queued: {product}, "
                            f"amount={deferred_amount:.2f}"
                        )

                elif target_per_position > 100:
                    # hisensho audit fix: No cash available, defer entire buy
                    self._deferred_buys.append({
                        "product": product,
                        "amount": target_per_position,
                        "original_date": date,
                    })
                    logger.debug(
                        f"[T+N] Buy deferred due to T+N: {product}, "
                        f"amount={target_per_position:.2f}"
                    )

        return trades

    def _process_deferred_buys(
        self,
        current_date: datetime,
        nav_lookup: Dict[str, float],
    ) -> List[Dict]:
        """
        Process deferred buy orders after T+N settlement releases cash.

        hisensho audit fix: Lazy buying to handle T+N cash flow constraint.
        """
        trades = []
        remaining_deferred = []

        for deferred in self._deferred_buys:
            product = deferred["product"]
            target_amount = deferred["amount"]

            if product not in nav_lookup:
                remaining_deferred.append(deferred)
                continue

            nav = nav_lookup[product]
            available_cash = self.tn_simulator.cash

            if available_cash >= target_amount * 0.95:  # Allow 5% tolerance
                buy_amount = min(target_amount, available_cash)
                if buy_amount > 100:
                    order = self.tn_simulator.execute_buy(
                        product_code=product,
                        amount=buy_amount,
                        current_date=current_date,
                        nav=nav,
                    )
                    trades.append({
                        "date": current_date,
                        "product": product,
                        "action": "deferred_buy",
                        "amount": buy_amount,
                        "units": order.quantity,
                        "nav": nav,
                    })
                    logger.debug(
                        f"[T+N] Deferred buy executed: {product}, "
                        f"amount={buy_amount:.2f}"
                    )
            elif available_cash > 100:
                # Partial buy with available cash
                order = self.tn_simulator.execute_buy(
                    product_code=product,
                    amount=available_cash * 0.95,  # Keep small reserve
                    current_date=current_date,
                    nav=nav,
                )
                trades.append({
                    "date": current_date,
                    "product": product,
                    "action": "partial_deferred_buy",
                    "amount": available_cash * 0.95,
                    "units": order.quantity,
                    "nav": nav,
                })
                # Reduce deferred amount
                deferred["amount"] = target_amount - available_cash * 0.95
                remaining_deferred.append(deferred)
            else:
                # Not enough cash, keep deferred
                remaining_deferred.append(deferred)

        self._deferred_buys = remaining_deferred
        return trades

    def _rebalance(
        self,
        date,
        current_positions: Dict[str, float],
        capital: float,
        max_positions: int,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """Legacy rebalance method - deprecated, use _rebalance_with_units"""
        logger.warning("_rebalance is deprecated, use _rebalance_with_units")
        nav_lookup = {
            p: self.prices.loc[date, p]
            for p in self.prices.columns
            if date in self.prices.index
        }
        trades = self._rebalance_with_units(date, nav_lookup, max_positions)
        return current_positions, trades

    def _get_fee_array(self) -> Union[float, np.ndarray]:
        """Get fee array for VectorBT"""
        # For simplicity, return average fee
        # In production, this would be a time-varying array
        return 0.005  # 0.5% average

    def _compute_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """Compute performance metrics"""
        n_days = len(equity)
        years = n_days / 252

        # Total return
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

        # Annual return
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio (using 0.5% rf_rate for WMP products)
        rf_daily = 0.005 / 252  # 0.5% annual = 0.002% daily
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                  if excess_returns.std() > 0 else 0)

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "n_days": n_days,
        }


def create_vbt_simulator(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    t_plus_n: int = 1,
    **kwargs,
) -> VBTPortfolioSimulator:
    """
    Convenience function to create VBTPortfolioSimulator.

    Args:
        prices: Price DataFrame
        signals: Signal DataFrame
        t_plus_n: T+N days
        **kwargs: Additional arguments

    Returns:
        VBTPortfolioSimulator instance
    """
    return VBTPortfolioSimulator(
        prices=prices,
        signals=signals,
        t_plus_n=t_plus_n,
        **kwargs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic data
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    products = [f"P{i:03d}" for i in range(20)]

    # Generate synthetic prices (random walk)
    prices_data = np.ones((len(dates), len(products)))
    for t in range(1, len(dates)):
        drift = 0.0002 + np.random.randn(len(products)) * 0.01
        prices_data[t] = prices_data[t - 1] * (1 + drift)

    prices = pd.DataFrame(prices_data, index=dates, columns=products)

    # Generate synthetic signals
    signals = pd.DataFrame(
        np.random.rand(len(dates), len(products)),
        index=dates,
        columns=products,
    )

    # Run simulation
    simulator = VBTPortfolioSimulator(prices, signals, t_plus_n=1)
    result = simulator.run_backtest(
        rebalance_freq="W-WED",
        max_positions=6,
        init_cash=1e8,
    )

    print("\n=== Backtest Results ===")
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Annual Return: {result.metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
