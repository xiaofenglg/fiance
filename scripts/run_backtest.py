#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V11 回测运行脚本

使用方法:
    python scripts/run_backtest.py --bank "中信银行" --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aifinance.backtest.engine import BacktestEngine, run_backtest
from aifinance.data.nav_loader import NavLoader, load_nav_data
from aifinance.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="V11 回测运行脚本")
    parser.add_argument("--bank", type=str, default="中信银行", help="银行名称")
    parser.add_argument("--db-path", type=str, default=None, help="数据库路径")
    parser.add_argument("--parquet-dir", type=str, default=None, help="Parquet 目录")
    parser.add_argument("--start", type=str, default=None, help="开始日期")
    parser.add_argument("--end", type=str, default=None, help="结束日期")
    parser.add_argument("--capital", type=float, default=1e8, help="初始资金")
    parser.add_argument("--train-window", type=int, default=180, help="训练窗口")
    parser.add_argument("--rebalance-freq", type=int, default=14, help="再平衡频率")
    parser.add_argument("--no-unsmooth", action="store_true", help="禁用 GLM 去平滑")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"=== V11 回测: {args.bank} ===")

    # 加载数据
    try:
        if args.db_path or args.parquet_dir:
            nav_matrix, returns, masks, dates, product_codes = load_nav_data(
                args.bank,
                db_path=args.db_path,
                parquet_dir=args.parquet_dir,
            )
        else:
            # 使用模拟数据进行演示
            logger.warning("未指定数据源,使用模拟数据")
            np.random.seed(42)
            n_products, n_dates = 100, 500
            returns = np.random.randn(n_products, n_dates).astype(np.float32) * 3 + 2.5
            masks = (np.random.rand(n_products, n_dates) > 0.1).astype(np.float32)
            dates = (
                pd.date_range("2023-01-01", periods=n_dates, freq="D")
                .strftime("%Y-%m-%d")
                .tolist()
            )
            product_codes = [f"PROD_{i}" for i in range(n_products)]

        logger.info(f"数据加载完成: {returns.shape[0]} 产品, {returns.shape[1]} 天")

    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return 1

    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=args.capital,
        train_window=args.train_window,
        rebalance_freq=args.rebalance_freq,
        use_unsmoothing=not args.no_unsmooth,
        use_fee_db=True,  # 使用真实费率
        bank_name=args.bank,
    )

    # 运行回测
    try:
        result = engine.run(
            returns,
            masks,
            dates,
            product_codes=product_codes,
            start_date=args.start,
            end_date=args.end,
        )
    except Exception as e:
        logger.error(f"回测运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 打印结果
    print("\n" + "=" * 60)
    print("                    V11 回测结果")
    print("=" * 60)
    print(f"回测区间: {result.start_date} -> {result.end_date}")
    print(f"交易天数: {result.n_days}")
    print("-" * 60)
    print(f"总收益率:     {result.total_return:>10.2%}")
    print(f"年化收益率:   {result.annual_return:>10.2%}")
    print(f"夏普比率:     {result.sharpe_ratio:>10.2f}")
    print(f"最大回撤:     {result.max_drawdown:>10.2%}")
    print(f"胜率:         {result.win_rate:>10.2%}")
    print(f"盈亏比:       {result.profit_factor:>10.2f}")
    print("-" * 60)
    print(f"交易次数:     {result.n_trades:>10}")
    print(f"平均持仓天数: {result.avg_holding_days:>10.1f}")
    print(f"年化换手率:   {result.turnover:>10.2f}")
    print("=" * 60)

    # 目标检查
    print("\n目标达成检查:")
    target_annual = 0.06
    target_sharpe = 1.85  # 月度计算夏普
    target_mdd = 0.05

    annual_ok = result.annual_return >= target_annual
    sharpe_ok = result.sharpe_ratio >= target_sharpe
    mdd_ok = result.max_drawdown <= target_mdd

    print(f"  年化收益 >= {target_annual:.0%}: {'[OK]' if annual_ok else '[FAIL]'} ({result.annual_return:.2%})")
    print(f"  夏普比率 >= {target_sharpe}: {'[OK]' if sharpe_ok else '[FAIL]'} ({result.sharpe_ratio:.2f})")
    print(f"  最大回撤 <= {target_mdd:.0%}: {'[OK]' if mdd_ok else '[FAIL]'} ({result.max_drawdown:.2%})")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存净值曲线
        result.equity_curve.to_csv(output_path.with_suffix(".equity.csv"))

        # 保存日收益率
        result.daily_returns.to_csv(output_path.with_suffix(".returns.csv"))

        # 保存汇总
        summary = {
            "bank": args.bank,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "n_days": result.n_days,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "n_trades": result.n_trades,
        }
        pd.Series(summary).to_csv(output_path.with_suffix(".summary.csv"))

        logger.info(f"结果已保存至: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
