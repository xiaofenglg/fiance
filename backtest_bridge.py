# -*- coding: utf-8 -*-
"""
回测系统适配层 — 连接 Flask API 与 backtest_pattern_v4.py 引擎

线程安全地执行回测、缓存结果供 API 返回。
"""

import threading
import logging
import traceback
import copy
from datetime import datetime

logger = logging.getLogger(__name__)

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
    'engine': None,          # BacktestEngine 实例
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
    try:
        import backtest_pattern_v4 as bt
        import numpy as np
        import pandas as pd
        from collections import defaultdict

        _emit(5, '初始化回测引擎 (Pattern V4)...')

        # 覆盖参数（如果有）
        if params:
            if 'buy_threshold' in params:
                bt.信号阈值_买入 = float(params['buy_threshold'])
            if 'sell_threshold' in params:
                bt.信号阈值_卖出 = float(params['sell_threshold'])
            if 'max_positions' in params:
                bt.最大持仓数 = int(params['max_positions'])
            if 'max_hold_days' in params:
                bt.最大持有交易日 = int(params['max_hold_days'])

        engine = bt.PatternBacktestEngine()

        _emit(10, '加载净值数据库...')
        engine.load_data()

        _emit(30, f'数据加载完成，{len(engine.products)}个产品，运行 Mode B 回测...')

        # 运行 Mode B (优选模式)
        result = engine.run_single_mode('B')

        _emit(92, '计算绩效指标...')

        # 存储引擎引用（供 get_patterns_snapshot 使用）
        _set('engine', engine)

        # 存储释放规律快照
        _set('patterns_snapshot', result.get('patterns_snapshot', {}))

        # 构建摘要 — 直接从 result dict
        summary = _build_summary(result, bt, engine)
        _set('summary', summary)

        # 构建 NAV 序列 — 从 result['daily_values']
        nav_series = [{'date': d, 'value': round(v / 1e4, 2), 'cash': round(c / 1e4, 2),
                        'position': round(pv / 1e4, 2)}
                       for d, v, c, pv in result['daily_values']]
        _set('nav_series', nav_series)

        # 构建交易明细 — 从 result['trade_log']
        trades = []
        for t in result['trade_log']:
            trades.append({
                'date': t['date'],
                'action': t['action'],
                'bank': t['bank'],
                'code': t['code'],
                'name': t['name'][:25] if t['name'] else '',
                'nav': round(t['nav'], 6) if t['nav'] else 0,
                'amount': round(t['amount'] / 1e4, 2) if t['amount'] else 0,
                'pnl': round(t['pnl'] / 1e4, 2),
                'hold_days': t.get('hold_days', 0),
                'reason': t['reason'],
            })
        _set('trades', trades)

        # 月度收益 — 从 daily_values 手动计算
        monthly = _calc_monthly_returns(result['daily_values'])
        _set('monthly_returns', monthly)

        _emit(100, f'回测完成 — 年化{summary.get("ann_return", 0):.2f}%  夏普{summary.get("sharpe", 0):.2f}')
        _set('status', 'done')
        _set('finished_at', datetime.now().isoformat())

    except Exception as e:
        logger.error(f"回测执行失败: {e}\n{traceback.format_exc()}")
        _set('status', 'error')
        _set('error', str(e))
        _set('finished_at', datetime.now().isoformat())


def _calc_monthly_returns(daily_values):
    """从 daily_values [(date, total, cash, pos_val), ...] 计算月度收益"""
    if len(daily_values) < 2:
        return []
    monthly = {}
    for d, v, _, _ in daily_values:
        ym = d[:7]  # 'YYYY-MM'
        if ym not in monthly:
            monthly[ym] = {'first': v, 'last': v}
        monthly[ym]['last'] = v
    result = []
    prev_last = None
    for ym in sorted(monthly.keys()):
        base = prev_last if prev_last is not None else monthly[ym]['first']
        if base > 0:
            ret = (monthly[ym]['last'] / base - 1) * 100
            result.append({'month': ym, 'return_pct': round(ret, 4)})
        prev_last = monthly[ym]['last']
    return result


def _build_summary(result, bt, engine):
    import numpy as np
    import pandas as pd

    daily_values = result.get('daily_values', [])
    closed_trades = result.get('closed_trades', [])

    if not daily_values:
        return {}

    initial = bt.初始资金
    fv = daily_values[-1][1]
    total_ret = (fv / initial - 1) * 100
    t0 = pd.Timestamp(engine.all_sim_dates[0])
    t1 = pd.Timestamp(engine.all_sim_dates[-1])
    yrs = (t1 - t0).days / 365
    ann_ret = ((fv / initial) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0

    peak = initial
    max_dd = 0
    dd_date = ''
    for d, v, _, _ in daily_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
            dd_date = d

    nt = len(closed_trades)
    nw = sum(1 for t in closed_trades if t.pnl > 0)
    nl = nt - nw
    wr = nw / nt * 100 if nt else 0
    total_pnl = sum(t.pnl for t in closed_trades)

    hold_days_list = [engine._td_held(t.confirm_date, t.sell_date)
                      for t in closed_trades if t.sell_date]
    avg_hold = float(np.mean(hold_days_list)) if hold_days_list else 0

    avg_win = float(np.mean([t.pnl for t in closed_trades if t.pnl > 0])) if nw else 0
    avg_loss = float(np.mean([t.pnl for t in closed_trades if t.pnl <= 0])) if nl else 0

    gp = sum(t.pnl for t in closed_trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in closed_trades if t.pnl <= 0))
    pf = gp / gl if gl > 0 else 0

    daily_rets = []
    for i in range(1, len(daily_values)):
        pv = daily_values[i - 1][1]
        cv = daily_values[i][1]
        if pv > 0:
            daily_rets.append(cv / pv - 1)
    sharpe = 0
    if daily_rets:
        rf = 0.025 / 252
        excess = np.array(daily_rets) - rf
        if np.std(excess) > 0:
            sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    bm_val = initial * (1 + 0.025 * yrs)
    bm_ret = (bm_val / initial - 1) * 100

    # 按银行统计
    from collections import defaultdict
    bank_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
    for t in closed_trades:
        b = engine.products[t.product_key].bank
        bank_stats[b]['trades'] += 1
        bank_stats[b]['pnl'] += t.pnl
        if t.pnl > 0:
            bank_stats[b]['wins'] += 1

    bank_summary = {}
    for b, s in bank_stats.items():
        bank_summary[b] = {
            'trades': s['trades'],
            'pnl': round(s['pnl'] / 1e4, 2),
            'win_rate': round(s['wins'] / s['trades'] * 100, 1) if s['trades'] else 0,
        }

    return {
        'period': f"{engine.all_sim_dates[0]} ~ {engine.all_sim_dates[-1]}",
        'trading_days': len(engine.all_sim_dates),
        'initial_capital': round(initial / 1e4, 0),
        'final_value': round(fv / 1e4, 0),
        'total_return': round(total_ret, 4),
        'ann_return': round(ann_ret, 4),
        'total_pnl': round(total_pnl / 1e4, 2),
        'benchmark_return': round(bm_ret, 4),
        'excess_return': round(total_ret - bm_ret, 4),
        'max_drawdown': round(max_dd, 4),
        'max_dd_date': dd_date,
        'sharpe': round(sharpe, 4),
        'profit_factor': round(pf, 2),
        'total_trades': nt,
        'wins': nw,
        'losses': nl,
        'win_rate': round(wr, 1),
        'avg_hold_days': round(avg_hold, 1),
        'avg_win': round(avg_win / 1e4, 2),
        'avg_loss': round(avg_loss / 1e4, 2),
        'bank_stats': bank_summary,
    }


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
        'buy_threshold': 3.5,
        'sell_threshold': 2.0,
        'ann_return': summary.get('ann_return', 0),
        'sharpe': summary.get('sharpe', 0),
        'max_drawdown': summary.get('max_drawdown', 0),
        'win_rate': summary.get('win_rate', 0),
    }]


# ══════════════════════════════════════════
# GPU 参数寻优
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


def get_patterns_snapshot():
    """从回测结果中提取释放规律快照"""
    summary = _get('summary', {})
    if not summary:
        return []
    # patterns_snapshot 在 _run_backtest 中额外存储
    snap = _get('patterns_snapshot', {})
    if not snap:
        return []
    engine = _get('engine')
    result = []
    for key, info in snap.items():
        p = engine.products.get(key) if engine else None
        bank = p.bank if p else str(key[0]) if isinstance(key, tuple) else ''
        code = p.code if p else str(key[1]) if isinstance(key, tuple) else ''
        name = (p.name[:25] if p else '')
        result.append({
            'bank': bank,
            'code': code,
            'name': name,
            'confidence': info.get('confidence', 0),
            'period_days': info.get('period_days', 0),
            'period_cv': info.get('period_cv', 1.0),
            'has_period': info.get('has_period', False),
            'top_phase': info.get('top_phase', 0),
            'phase_pvalue': info.get('phase_pvalue', 1.0),
            'n_events': info.get('n_events', 0),
            'alpha': info.get('confidence', 0),  # 用 confidence 作排序依据
        })
    result.sort(key=lambda x: -x['alpha'])
    return result
