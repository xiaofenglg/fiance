# -*- coding: utf-8 -*-
"""
回测系统适配层 V12 — 连接 Flask API 与 V12 VBTPortfolioSimulator 引擎

数据流: load_nav_data → prices_df → Mom5信号 → VBTPortfolioSimulator → SimulationResult → 前端JSON

线程安全地执行回测、缓存结果供 API 返回。
"""

import os
import threading
import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── V12 模块导入 ──
from src.aifinance.data.nav_loader import load_nav_data
from src.aifinance.backtest.vbt_simulator import VBTPortfolioSimulator

# ── 银行名称映射 (crawler key → DB bank_name) ──
BANK_KEY_MAP = {
    'minsheng': '民生银行',
    'huaxia': '华夏银行',
    'citic': '中信银行',
    'spdb': '浦银理财',
    'ningyin': '宁银理财',
    'psbc': '中邮理财',
}
BANK_KEY_MAP_REVERSE = {v: k for k, v in BANK_KEY_MAP.items()}


def _load_config():
    """加载 pipeline_config.yaml 配置"""
    config_path = os.path.join(BASE_DIR, 'config', 'pipeline_config.yaml')
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")
        return {}
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return {}


# ── 全局状态 ──
_lock = threading.Lock()
_state = {
    'status': 'idle',        # idle / running / done / error
    'progress': 0,
    'message': '',
    'error': '',
    'started_at': None,
    'finished_at': None,
    # 缓存的结果
    'summary': {},
    'nav_series': [],
    'trades': [],
    'monthly_returns': [],
}


def _set(key, value):
    with _lock:
        _state[key] = value


def _get(key, default=None):
    with _lock:
        v = _state.get(key, default)
        return v


def get_status():
    with _lock:
        return {
            'status': _state['status'],
            'progress': _state['progress'],
            'message': _state['message'],
            'error': _state['error'],
            'started_at': _state['started_at'],
            'finished_at': _state['finished_at'],
        }


def _emit(progress, message):
    with _lock:
        _state['progress'] = progress
        _state['message'] = message
    logger.info(f"[Backtest] {progress}% — {message}")


# ── 回测执行（后台线程） ──

def run_backtest(params=None):
    """在后台线程中执行回测"""
    if _get('status') == 'running':
        return False, '回测正在运行中'

    _set('status', 'running')
    _set('progress', 0)
    _set('message', '启动中...')
    _set('error', '')
    _set('started_at', datetime.now().isoformat())
    _set('finished_at', None)

    t = threading.Thread(target=_run_backtest, args=(params,), daemon=True)
    t.start()
    return True, '已启动'


def _run_backtest(params):
    """V12 回测核心: load_nav_data → Mom5 → VBTPortfolioSimulator"""
    try:
        _emit(5, '初始化 V12 Quantitative Pipeline...')

        # ── Step 1: 加载配置 ──
        config = _load_config()
        data_config = config.get('data', {})
        bt_config = config.get('backtest', {})
        opt_config = config.get('optimizer', {})

        db_path = os.path.join(BASE_DIR, data_config.get('db_path', 'aifinance.sqlite3'))
        rebalance_freq = bt_config.get('rebalance_freq', 'W-MON')
        init_cash = bt_config.get('init_cash', 100_000_000)
        t_plus_n = bt_config.get('t_plus_n', 1)
        max_assets = opt_config.get('max_assets', 5)
        lookback_days = data_config.get('lookback_days', 365)
        max_products = data_config.get('max_products', 1000)
        min_records = data_config.get('min_records', 10)

        # ── Step 2: 确定银行 ──
        bank_name = '华夏银行'  # 默认
        if params:
            if 'bank' in params:
                raw = params['bank']
                bank_name = BANK_KEY_MAP.get(raw, raw)
            if 'max_positions' in params:
                max_assets = int(params['max_positions'])
            if 'rebalance_freq' in params:
                rebalance_freq = params['rebalance_freq']

        _emit(10, f'加载 {bank_name} 净值数据...')

        # ── Step 3: 加载 NAV 数据 ──
        nav_matrix, returns, masks, dates, product_codes = load_nav_data(
            bank_name=bank_name,
            db_path=db_path,
            min_valid_ratio=0.05,
            lookback_days=lookback_days,
            max_products=max_products,
            min_records=min_records,
        )

        n_products = nav_matrix.shape[0]
        n_dates = nav_matrix.shape[1]
        _emit(30, f'数据加载完成: {n_products} 产品, {n_dates} 天')

        # ── Step 4: 构建 prices_df ──
        prices_df = pd.DataFrame(
            nav_matrix.T,
            index=pd.to_datetime(dates),
            columns=product_codes,
        )
        # 清洗: 0 → NaN, 前向填充
        prices_df = prices_df.replace(0.0, np.nan)
        prices_df = prices_df.ffill()

        _emit(40, '生成 Mom5 动量信号...')

        # ── Step 5: Mom5 信号 (复用 run_pipeline.py 逻辑) ──
        cum_ret_5 = prices_df / prices_df.shift(5) - 1
        mom5_rank = cum_ret_5.rank(axis=1, pct=True)
        signals_df = mom5_rank.replace([np.inf, -np.inf], np.nan)

        _emit(50, f'V12 回测运行中 ({rebalance_freq}, Top-{max_assets})...')

        # ── Step 6: VBTPortfolioSimulator 回测 ──
        simulator = VBTPortfolioSimulator(
            prices=prices_df,
            signals=signals_df,
            t_plus_n=t_plus_n,
            db_path=db_path,
            bank_name=bank_name,
        )
        sim_result = simulator.run_backtest(
            rebalance_freq=rebalance_freq,
            max_positions=max_assets,
            init_cash=init_cash,
            use_vbt=False,  # 使用 native simulator (hisensho 审计合规)
        )

        _emit(85, '计算绩效指标...')

        # ── Step 7: 转换结果为前端格式 ──
        summary = _build_summary_v12(sim_result, init_cash, bank_name)
        _set('summary', summary)

        nav_series = _build_nav_series(sim_result)
        _set('nav_series', nav_series)

        trades = _build_trades_v12(sim_result, bank_name)
        _set('trades', trades)

        monthly = _calc_monthly_returns_v12(sim_result.equity_curve)
        _set('monthly_returns', monthly)

        _emit(100, f'V12 回测完成 — 年化{summary.get("ann_return", 0):.2f}%  夏普{summary.get("sharpe", 0):.2f}')
        _set('status', 'done')
        _set('finished_at', datetime.now().isoformat())

    except Exception as e:
        logger.error(f"回测执行失败: {e}\n{traceback.format_exc()}")
        _set('status', 'error')
        _set('error', str(e))
        _set('finished_at', datetime.now().isoformat())


# ── V12 结果转换函数 ──

def _build_summary_v12(sim_result, init_cash, bank_name):
    """SimulationResult.metrics → 前端 summary dict"""
    m = sim_result.metrics
    equity = sim_result.equity_curve

    fv = float(equity.iloc[-1])
    total_ret = (fv / init_cash - 1) * 100
    ann_ret = m.get('annual_return', 0) * 100
    sharpe = m.get('sharpe_ratio', 0)
    max_dd = m.get('max_drawdown', 0) * 100

    # 回测期间
    dates = equity.index
    period = f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}"
    n_days = len(dates)
    yrs = n_days / 252

    # 基准 (2.5% 年化存款利率)
    bm_ret = 2.5 * yrs
    excess = total_ret - bm_ret

    # 最大回撤日期
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    dd_date = ''
    if len(drawdown) > 0:
        dd_idx = drawdown.idxmin()
        dd_date = dd_idx.strftime('%Y-%m-%d') if hasattr(dd_idx, 'strftime') else str(dd_idx)

    # 交易统计
    trades = sim_result.trades
    buy_trades = [t for t in trades if t.get('action') in ('buy', 'deferred_buy', 'partial_deferred_buy')]
    sell_trades = [t for t in trades if t.get('action') == 'sell']

    return {
        'period': period,
        'trading_days': n_days,
        'initial_capital': round(init_cash / 1e4, 0),
        'final_value': round(fv / 1e4, 0),
        'total_return': round(total_ret, 4),
        'ann_return': round(ann_ret, 4),
        'total_pnl': round((fv - init_cash) / 1e4, 2),
        'benchmark_return': round(bm_ret, 4),
        'excess_return': round(excess, 4),
        'max_drawdown': round(max_dd, 4),
        'max_dd_date': dd_date,
        'sharpe': round(float(sharpe), 4),
        'profit_factor': 0,  # V12 不计算 profit_factor
        'total_trades': len(buy_trades) + len(sell_trades),
        'wins': 0,
        'losses': 0,
        'win_rate': 0,
        'avg_hold_days': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'bank_stats': {bank_name: {'trades': len(trades), 'pnl': round((fv - init_cash) / 1e4, 2)}},
    }


def _build_nav_series(sim_result):
    """equity_curve + cash_history → [{date, value(万), cash(万), position(万)}]"""
    equity = sim_result.equity_curve
    cash = sim_result.cash_history

    result = []
    for i, date in enumerate(equity.index):
        eq_val = float(equity.iloc[i])
        cash_val = float(cash.iloc[i]) if cash is not None and i < len(cash) else 0
        pos_val = eq_val - cash_val
        result.append({
            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            'value': round(eq_val / 1e4, 2),
            'cash': round(cash_val / 1e4, 2),
            'position': round(pos_val / 1e4, 2),
        })
    return result


def _build_trades_v12(sim_result, bank_name):
    """SimulationResult.trades → 前端交易明细列表"""
    result = []
    for t in sim_result.trades:
        date = t.get('date', '')
        if hasattr(date, 'strftime'):
            date = date.strftime('%Y-%m-%d')
        elif hasattr(date, 'isoformat'):
            date = date.isoformat()[:10]
        else:
            date = str(date)[:10]

        action = t.get('action', '')
        product = t.get('product', '')
        nav = t.get('nav', 0)
        amount = t.get('amount', 0)
        units = t.get('units', 0)

        # 用 amount 或 units*nav 计算金额
        if amount:
            trade_amount = float(amount)
        elif units and nav:
            trade_amount = float(units) * float(nav)
        else:
            trade_amount = 0

        result.append({
            'date': date,
            'action': '买入' if 'buy' in action else '卖出',
            'bank': bank_name,
            'code': product,
            'name': '',
            'nav': round(float(nav), 6) if nav else 0,
            'amount': round(trade_amount / 1e4, 2),
            'pnl': 0,
            'hold_days': 0,
            'reason': f'V12 Mom5 {action}',
        })
    return result


def _calc_monthly_returns_v12(equity_curve):
    """从 equity_curve Series 计算月度收益"""
    if len(equity_curve) < 2:
        return []
    monthly_equity = equity_curve.resample('ME').last().dropna()
    if len(monthly_equity) < 2:
        return []

    result = []
    for i in range(1, len(monthly_equity)):
        prev = monthly_equity.iloc[i - 1]
        curr = monthly_equity.iloc[i]
        if prev > 0:
            ret = (curr / prev - 1) * 100
            ym = monthly_equity.index[i].strftime('%Y-%m')
            result.append({'month': ym, 'return_pct': round(float(ret), 4)})
    return result


# ── 数据访问 ──

def get_results():
    return _get('summary', {})


def get_nav():
    return _get('nav_series', [])


def get_trades():
    return _get('trades', [])


def get_monthly():
    return _get('monthly_returns', [])


def get_sweep():
    """敏感性分析数据（当前参数的摘要，实际sweep需多次运行）"""
    summary = _get('summary', {})
    if not summary:
        return []
    return [{
        'rebalance_freq': 'W-MON',
        'max_assets': 5,
        'ann_return': summary.get('ann_return', 0),
        'sharpe': summary.get('sharpe', 0),
        'max_drawdown': summary.get('max_drawdown', 0),
    }]


def get_patterns_snapshot():
    """V12 不使用 pattern 机制，返回空列表"""
    return []


# ══════════════════════════════════════════
# GPU 参数寻优 (保留原有功能)
# ══════════════════════════════════════════

_sweep_lock = threading.Lock()
_sweep_state = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'error': '',
    'started_at': None,
    'finished_at': None,
    'results': {},
}


def _sweep_set(key, value):
    with _sweep_lock:
        _sweep_state[key] = value


def _sweep_get(key, default=None):
    with _sweep_lock:
        return _sweep_state.get(key, default)


def get_sweep_status():
    with _sweep_lock:
        return {
            'status': _sweep_state['status'],
            'progress': _sweep_state['progress'],
            'message': _sweep_state['message'],
            'error': _sweep_state['error'],
            'started_at': _sweep_state['started_at'],
            'finished_at': _sweep_state['finished_at'],
        }


def run_param_sweep(param_grid=None):
    """启动 GPU 参数寻优（后台线程）"""
    if _sweep_get('status') == 'running':
        return False, '参数寻优正在运行中'

    _sweep_set('status', 'running')
    _sweep_set('progress', 0)
    _sweep_set('message', '启动参数寻优...')
    _sweep_set('error', '')
    _sweep_set('started_at', datetime.now().isoformat())
    _sweep_set('finished_at', None)

    t = threading.Thread(target=_run_param_sweep, args=(param_grid,), daemon=True)
    t.start()
    return True, '已启动'


def _run_param_sweep(param_grid):
    try:
        from gpu_param_sweep import VectorizedBacktest

        def progress_cb(pct, msg):
            _sweep_set('progress', pct)
            _sweep_set('message', msg)

        engine = VectorizedBacktest()
        engine.set_progress_callback(progress_cb)

        _sweep_set('message', '加载数据...')
        engine.load_data()

        results = engine.sweep(param_grid)
        _sweep_set('results', results)
        _sweep_set('progress', 100)
        _sweep_set('message', f'完成 — 最优夏普 {results.get("best_sharpe", 0):.2f}')
        _sweep_set('status', 'done')
        _sweep_set('finished_at', datetime.now().isoformat())

    except Exception as e:
        logger.error(f"参数寻优失败: {e}\n{traceback.format_exc()}")
        _sweep_set('status', 'error')
        _sweep_set('error', str(e))
        _sweep_set('finished_at', datetime.now().isoformat())


def get_sweep_results():
    """获取寻优结果"""
    return _sweep_get('results', {})
