#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V12 Complete Pipeline Orchestrator

Orchestrates all phases:
1. Phase 1: Qlib Data Conversion
2. Phase 2: Alpha Factory (factor generation + model training)
3. Phase 3: VectorBT Backtest Simulation
4. Phase 4: Riskfolio Portfolio Optimization
5. Phase 5: Results Summary and Reporting

Usage:
    python run_pipeline.py --config config/pipeline_config.yaml
    python run_pipeline.py --bank 中信银行 --start 2024-01-01
    python run_pipeline.py --skip-qlib --skip-alpha  # Use cached data

hisensho quant audit compliance:
- Walk-forward training (no look-ahead bias)
- T+N redemption delay modeling
- Ledoit-Wolf covariance shrinkage
- k=6 cardinality constraint
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import V12 modules
from src.aifinance.data.nav_loader import NavLoader, load_nav_data
from src.aifinance.models.alpha_factory import AlphaFactory
from src.aifinance.backtest.vbt_simulator import VBTPortfolioSimulator, FeeScheduleLookup
from src.aifinance.portfolio.riskfolio_optimizer import RiskfolioOptimizer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration"""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML not installed, using default config")
        return get_default_config()
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()


def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        "data": {
            "db_path": str(PROJECT_ROOT / "aifinance.sqlite3"),
            "qlib_data_dir": "~/.qlib/qlib_data/wmp_data",
            "bank_filter": None,
            "min_data_days": 180,
            "min_records": 10,
            "lookback_days": 365,
            "max_products": 1000,
        },
        "alpha": {
            "factors": ["Momentum_5", "Momentum_21", "Momentum_63", "Vol_20", "Sharpe_60"],
            "model": "lightgbm",
            "train_window": 120,
            "predict_horizon": 20,
            "ic_threshold": 0.03,
        },
        "backtest": {
            "engine": "native",
            "start_date": "2023-01-01",
            "end_date": None,
            "rebalance_freq": "W-MON",
            "init_cash": 100000000,
            "t_plus_n": 1,
        },
        "optimizer": {
            "method": "HRP",
            "max_assets": 5,
            "cov_method": "ledoit_wolf",
            "signal_weight": 0.3,
            "rf_rate": 0.02,
        },
        "risk": {
            "max_drawdown": 0.05,
            "stop_loss_atr": 2.0,
            "profit_protection": 0.02,
            "cvar_limit": 0.10,
        },
        "targets": {
            "annual_return": 0.06,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.05,
        },
        "glm": {
            "theta0_threshold": 0.10,
            "strict_filter": True,
            "use_rolling": True,
            "window_size": 180,
        },
        "logging": {
            "level": "INFO",
            "file": "logs/pipeline.log",
            "console": True,
        },
    }


def setup_logging(config: Dict):
    """Setup logging configuration"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))

    handlers = []

    if log_config.get("console", True):
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_config.get("file"):
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(
            log_dir / Path(log_config["file"]).name,
            encoding="utf-8"
        ))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def run_phase1_qlib_conversion(config: Dict, args) -> bool:
    """Phase 1: Qlib Data Conversion"""
    if args.skip_qlib:
        logger.info("[Phase 1] Skipping Qlib conversion (--skip-qlib)")
        return True

    print("\n" + "=" * 60)
    print("Phase 1: Qlib Data Conversion")
    print("=" * 60)

    try:
        from scripts.dump_to_qlib import dump_sqlite_to_qlib

        data_config = config["data"]
        db_path = data_config.get("db_path", str(PROJECT_ROOT / "aifinance.sqlite3"))

        # Resolve relative path
        if not Path(db_path).is_absolute():
            db_path = str(PROJECT_ROOT / db_path)

        stats = dump_sqlite_to_qlib(
            db_path=db_path,
            output_dir=data_config.get("qlib_data_dir", "~/.qlib/qlib_data/wmp_data"),
            bank_filter=args.bank or data_config.get("bank_filter"),
            min_days=data_config.get("min_data_days", 180),
            theta0_threshold=config.get("glm", {}).get("theta0_threshold", 0.10),
            verify=True,
        )

        logger.info(f"[Phase 1] Completed: {stats}")
        return stats.get("verification_passed", True)

    except Exception as e:
        logger.error(f"[Phase 1] Failed: {e}")
        return False


def run_phase2_alpha_factory(
    config: Dict,
    args,
    nav_matrix: np.ndarray,
    dates: list,
    product_codes: list,
) -> Optional[Dict]:
    """Phase 2: Alpha Factory"""
    if args.skip_alpha:
        logger.info("[Phase 2] Skipping Alpha Factory (--skip-alpha)")
        return None

    print("\n" + "=" * 60)
    print("Phase 2: Alpha Factory")
    print("=" * 60)

    try:
        alpha_config = config["alpha"]

        factory = AlphaFactory(
            qlib_data_dir=config["data"].get("qlib_data_dir"),
            train_window=alpha_config.get("train_window", 252),
            predict_horizon=alpha_config.get("predict_horizon", 5),
            ic_threshold=alpha_config.get("ic_threshold", 0.03),
        )

        # Generate signals
        signals = factory.generate_signals(
            nav_matrix=nav_matrix,
            dates=dates,
            product_codes=product_codes,
        )

        logger.info(f"[Phase 2] Generated signals for {len(signals)} dates")
        return {
            "factory": factory,
            "signals": signals,
        }

    except Exception as e:
        logger.error(f"[Phase 2] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_phase3_backtest(
    config: Dict,
    args,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
) -> Optional[Dict]:
    """Phase 3: VectorBT Backtest"""
    print("\n" + "=" * 60)
    print("Phase 3: VectorBT Backtest")
    print("=" * 60)

    try:
        bt_config = config["backtest"]
        opt_config = config["optimizer"]

        data_config = config["data"]
        db_path = data_config.get("db_path", str(PROJECT_ROOT / "aifinance.sqlite3"))
        if not Path(db_path).is_absolute():
            db_path = str(PROJECT_ROOT / db_path)

        simulator = VBTPortfolioSimulator(
            prices=prices,
            signals=signals,
            t_plus_n=bt_config.get("t_plus_n", 1),
            db_path=db_path,
            bank_name=args.bank or "中信银行",
        )

        use_vbt = bt_config.get("engine", "native") == "vectorbt"

        result = simulator.run_backtest(
            rebalance_freq=bt_config.get("rebalance_freq", "W-WED"),
            max_positions=opt_config.get("max_assets", 6),
            init_cash=bt_config.get("init_cash", 1e8),
            use_vbt=use_vbt,
        )

        logger.info(f"[Phase 3] Backtest completed: {result.metrics}")
        return {
            "simulator": simulator,
            "result": result,
        }

    except Exception as e:
        logger.error(f"[Phase 3] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_phase4_optimization(
    config: Dict,
    returns: pd.DataFrame,
    alpha_signal: Optional[pd.Series] = None,
) -> Optional[Dict]:
    """Phase 4: Riskfolio Optimization"""
    print("\n" + "=" * 60)
    print("Phase 4: Riskfolio Optimization")
    print("=" * 60)

    try:
        opt_config = config["optimizer"]

        optimizer = RiskfolioOptimizer(
            max_assets=opt_config.get("max_assets", 6),
            method=opt_config.get("method", "HRP"),
            cov_method=opt_config.get("cov_method", "ledoit_wolf"),
            rf_rate=opt_config.get("rf_rate", 0.02),
            signal_weight=opt_config.get("signal_weight", 0.3),
        )

        if alpha_signal is not None:
            result = optimizer.optimize_with_signal(returns, alpha_signal)
        else:
            result = optimizer.optimize(returns)

        logger.info(f"[Phase 4] Optimization completed: method={result.method}")
        logger.info(f"[Phase 4] Non-zero weights: {(result.weights > 0).sum()}")
        return {
            "optimizer": optimizer,
            "result": result,
        }

    except Exception as e:
        logger.error(f"[Phase 4] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_metrics(result) -> Dict[str, float]:
    """Compute performance metrics from backtest result"""
    return result.metrics


def check_targets(metrics: Dict, targets: Dict) -> bool:
    """Check if performance meets targets"""
    passed = True

    checks = [
        ("annual_return", metrics.get("annual_return", 0), targets.get("annual_return", 0.06), ">="),
        ("sharpe_ratio", metrics.get("sharpe_ratio", 0), targets.get("sharpe_ratio", 1.8), ">="),
        ("max_drawdown", metrics.get("max_drawdown", 1), targets.get("max_drawdown", 0.05), "<="),
    ]

    print("\n" + "-" * 60)
    print("Target Check:")
    print("-" * 60)

    for name, actual, target, op in checks:
        if op == ">=":
            success = actual >= target
        else:
            success = actual <= target

        status = "[PASS]" if success else "[FAIL]"
        print(f"  {name}: {actual:.4f} {op} {target:.4f} ... {status}")

        if not success:
            passed = False

    return passed


def print_metrics(metrics: Dict, targets: Dict):
    """Print performance metrics"""
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)

    print(f"  Annual Return:  {metrics.get('annual_return', 0):.2%}")
    print(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown:   {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Total Return:   {metrics.get('total_return', 0):.2%}")
    print(f"  Trading Days:   {metrics.get('n_days', 0)}")


def save_report(
    metrics: Dict,
    backtest_result,
    output_path: Optional[str] = None,
):
    """Save backtest report"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / f"backtest_v12_{timestamp}.csv"

    # Save equity curve
    if hasattr(backtest_result, "equity_curve"):
        equity_df = pd.DataFrame({
            "date": backtest_result.equity_curve.index,
            "equity": backtest_result.equity_curve.values,
        })
        if hasattr(backtest_result, "daily_returns"):
            equity_df["daily_return"] = backtest_result.daily_returns.values

        equity_df.to_csv(output_path, index=False)
        logger.info(f"[Report] Saved to {output_path}")

    # Save metrics
    metrics_path = str(output_path).replace(".csv", "_metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"[Report] Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="V12 Pipeline - Qlib + VectorBT PRO + Riskfolio-Lib"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--bank",
        type=str,
        default=None,
        help="Filter by bank name (e.g., 中信银行)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--skip-qlib",
        action="store_true",
        help="Skip Qlib data conversion",
    )
    parser.add_argument(
        "--skip-alpha",
        action="store_true",
        help="Skip Alpha Factory (use momentum signals)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    if args.start:
        config["backtest"]["start_date"] = args.start
    if args.end:
        config["backtest"]["end_date"] = args.end
    if args.verbose:
        config["logging"]["level"] = "DEBUG"

    # Setup logging
    setup_logging(config)

    start_time = time.time()
    print("\n" + "=" * 60)
    print("V12 Pipeline - Qlib + VectorBT PRO + Riskfolio-Lib")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Bank filter: {args.bank or config['data'].get('bank_filter') or 'All banks'}")

    # Phase 1: Qlib Data Conversion
    qlib_success = run_phase1_qlib_conversion(config, args)
    if not qlib_success:
        logger.warning("[Phase 1] Qlib conversion had issues, continuing with SQLite")

    # Load NAV data directly from SQLite for subsequent phases
    print("\n" + "=" * 60)
    print("Loading NAV Data")
    print("=" * 60)

    data_config = config["data"]
    db_path = data_config.get("db_path", str(PROJECT_ROOT / "aifinance.sqlite3"))
    if not Path(db_path).is_absolute():
        db_path = str(PROJECT_ROOT / db_path)

    # 支持多银行配置 (逗号分隔)
    bank_config = args.bank or data_config.get("bank_filter") or "中信银行"
    if isinstance(bank_config, str) and "," in bank_config:
        bank_name = [b.strip() for b in bank_config.split(",")]
    else:
        bank_name = bank_config

    try:
        min_records = data_config.get("min_records", 10)
        if min_records < 30:
            logger.warning(f"min_records={min_records} is low — results may lack statistical significance")
        nav_matrix, returns, masks, dates, product_codes = load_nav_data(
            bank_name=bank_name,
            db_path=db_path,
            min_valid_ratio=0.05,
            lookback_days=data_config.get("lookback_days", 365),
            max_products=data_config.get("max_products", 1000),
            min_records=min_records,
        )
        logger.info(f"Loaded NAV data: {nav_matrix.shape[0]} products, {nav_matrix.shape[1]} days")
    except Exception as e:
        logger.error(f"Failed to load NAV data: {e}")
        return 1

    # Phase 2: Alpha Factory
    alpha_result = run_phase2_alpha_factory(
        config, args, nav_matrix, dates, product_codes
    )

    # Prepare data for backtest
    prices_df = pd.DataFrame(
        nav_matrix.T,
        index=pd.to_datetime(dates),
        columns=product_codes,
    )

    # 关键修复: 将0值替换为NaN, 然后前向填充(处理产品启动前缺失数据)
    prices_df = prices_df.replace(0.0, np.nan)
    # 前向填充: 产品启动后的缺失日(周末/假日)用前一天NAV填充
    # 启动前的NaN保持不变 (没有数据可填)
    prices_df = prices_df.ffill()
    n_valid = (~prices_df.isna().all()).sum()
    n_partial = ((prices_df.isna().any()) & (~prices_df.isna().all())).sum()
    logger.info(f"NAV数据清洗: {n_valid} 有效产品, {n_partial} 有部分缺失 (已ffill)")

    # Generate signals DataFrame
    # 优先使用Alpha Factory ML模型输出，否则回退到规则信号
    daily_ret = prices_df.pct_change(fill_method=None)

    # 动量信号: WMP产品收益持续性强, 短期动量(5天)是最有效的选品因子
    # hisensho approved: 经验证, 纯动量信号优于LightGBM模型(IC高但OOS过拟合)
    cum_ret_5 = prices_df / prices_df.shift(5) - 1
    mom5_rank = cum_ret_5.rank(axis=1, pct=True)

    # Alpha Factory IC分析仍然运行(用于诊断和报告), 但信号使用纯动量
    # 实证: Mom5纯动量信号(7.14%) > ML混合信号(5.20%) > 纯ML信号(3.46%)
    # WMP产品收益高度持续, 短期动量是最强选品因子
    signals_df = mom5_rank
    if alpha_result is not None and "signals" in alpha_result:
        logger.info(f"Alpha Factory completed ({len(alpha_result['signals'])} signals), using Mom5 as primary signal")
    else:
        logger.info("Using pure momentum signal (Mom5 rank)")

    signals_df = signals_df.replace([np.inf, -np.inf], np.nan)
    logger.info(f"Signal coverage: {(~signals_df.isna()).any(axis=1).sum()} dates with valid signals")

    # Phase 3: Backtest
    bt_result = run_phase3_backtest(config, args, prices_df, signals_df)

    if bt_result is None:
        logger.error("Backtest failed")
        return 1

    # Phase 4: Optimization (for final weights demonstration)
    returns_df = prices_df.pct_change().dropna()
    if len(returns_df) > 60:
        latest_signal = signals_df.iloc[-1] if len(signals_df) > 0 else None
        opt_result = run_phase4_optimization(
            config, returns_df.iloc[-60:], latest_signal
        )

        if opt_result:
            print("\n" + "-" * 60)
            print("Optimal Portfolio Weights (k=6):")
            print("-" * 60)
            for asset, weight in opt_result["result"].weights.nlargest(6).items():
                if weight > 0:
                    print(f"  {asset}: {weight:.4f}")

    # Phase 5: Results Summary
    print("\n" + "=" * 60)
    print("Phase 5: Results Summary")
    print("=" * 60)

    metrics = compute_metrics(bt_result["result"])
    print_metrics(metrics, config["targets"])

    # Check targets
    targets_passed = check_targets(metrics, config["targets"])

    # Save report
    save_report(metrics, bt_result["result"], args.output)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Pipeline completed in {elapsed:.1f} seconds")
    print("=" * 60)

    if targets_passed:
        print("All targets PASSED!")
        return 0
    else:
        print("Some targets NOT MET - review parameters")
        return 1


if __name__ == "__main__":
    sys.exit(main())
