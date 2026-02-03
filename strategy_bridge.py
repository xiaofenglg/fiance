# -*- coding: utf-8 -*-
"""
策略系统适配层 V8.0 — 连接 Flask API 与 bank_product_strategy_v6.py

线程安全地调用策略函数，缓存结果供 API 返回。
V8.0: SOPM v1 集成 — refresh_classification 自动应用 SOPM 排序
"""

import os
import re
import json
import threading
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(BASE_DIR, '策略结果_缓存.json')


# ── 日志监听器：捕获策略内部进度 ──

class _StrategyProgressHandler(logging.Handler):
    """监听 bank_product_strategy_v6 的日志，提取银行级扫描进度并更新 SSE"""

    # 扫描阶段 35→50，分3家银行：35→40→45→50
    BANK_PROGRESS = {
        '民生银行': 38, '华夏银行': 42, '中信银行': 46,
        '民生': 38, '华夏': 42, '中信': 46,
    }

    def __init__(self):
        super().__init__(logging.INFO)
        self.active = False  # 仅在运行期间激活

    def emit(self, record):
        if not self.active:
            return
        try:
            msg = record.getMessage()
            # 匹配 "扫描 [XXX银行]" 模式
            m = re.search(r'扫描\s*\[(.+?)\]', msg)
            if m:
                bank = m.group(1)
                pct = self.BANK_PROGRESS.get(bank, None)
                if pct:
                    with _lock:
                        _state['progress'] = pct
                        _state['message'] = f'扫描 {bank}...'
                return

            # 匹配 "发现 N 个实时机会"
            m = re.search(r'发现\s*(\d+)\s*个实时机会', msg)
            if m:
                count = m.group(1)
                with _lock:
                    cur = _state.get('message', '')
                    bank = cur.replace('扫描 ', '').replace('...', '') if '扫描' in cur else ''
                    _state['message'] = f'{bank} 完成，发现 {count} 个信号'
                return

            # 匹配 "数据合并: N条"
            m = re.search(r'数据合并:\s*(\d+)', msg)
            if m:
                with _lock:
                    _state['message'] = f'数据处理中（{m.group(1)}条）...'
                return

            # 匹配 "释放规律: N个产品"
            m = re.search(r'释放规律:\s*(\d+)', msg)
            if m:
                with _lock:
                    _state['progress'] = 68
                    _state['message'] = f'释放规律分析（{m.group(1)}个产品）...'
                return

            # 匹配 "持仓分析完成: N个产品"
            if '持仓分析' in msg:
                with _lock:
                    _state['progress'] = 78
                    _state['message'] = '持仓分析中...'
                return

            # 匹配 "精选推荐" 相关
            if '精选推荐' in msg and '硬筛' in msg:
                with _lock:
                    _state['progress'] = 88
                    _state['message'] = '精选推荐筛选中...'
                return

        except Exception:
            pass


_progress_handler = _StrategyProgressHandler()

# ── 全局状态 ──
_lock = threading.Lock()
_state = {
    'status': 'idle',        # idle / running / done / error
    'progress': 0,           # 0-100
    'message': '',
    'error': '',
    'started_at': None,
    'finished_at': None,
    # 缓存的结果
    'product_lib': [],
    'watch_pool': [],
    'opportunities': [],
    'recommendations': [],
    'portfolio_analysis': [],
    'prediction_data': {},
    'current_positions': [],
    'vip_channels': {},
}


def _set(key, value):
    with _lock:
        _state[key] = value


def _get(key, default=None):
    with _lock:
        return _state.get(key, default)


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
    logger.info(f"[Strategy] {progress}% — {message}")


# ── 结果持久化 ──

_CACHE_KEYS = ['product_lib', 'watch_pool', 'opportunities', 'recommendations',
               'portfolio_analysis', 'current_positions', 'vip_channels']

def _save_cache():
    """策略完成后将结果保存到 JSON，下次启动直接加载"""
    try:
        cache = {'finished_at': _get('finished_at')}
        for k in _CACHE_KEYS:
            cache[k] = _get(k, [])
        # patterns 需要预处理（原始对象不可序列化）
        cache['patterns_processed'] = get_patterns()
        with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, default=str)
        logger.info(f"[Strategy] 结果已缓存到 {_CACHE_FILE}")
    except Exception as e:
        logger.error(f"保存策略缓存失败: {e}")


def _load_cache():
    """启动时尝试从 JSON 加载上次运行结果"""
    if not os.path.exists(_CACHE_FILE):
        return
    try:
        with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        for k in _CACHE_KEYS:
            if k in cache:
                _set(k, cache[k])
        if cache.get('finished_at'):
            _set('finished_at', cache['finished_at'])
            _set('status', 'done')
            _set('message', '已加载上次运行结果')
        if cache.get('patterns_processed'):
            _set('_patterns_cache', cache['patterns_processed'])
        logger.info(f"[Strategy] 已从缓存加载上次结果 (运行于 {cache.get('finished_at', '?')})")

        # 重跑双通道分类，确保 sold_out.json 的最新状态被正确反映
        recs = _get('recommendations', [])
        if recs:
            import post_processing
            vip_channels = post_processing.classify_recommendations(recs)
            _set('vip_channels', vip_channels)
            logger.info("[Strategy] 已根据售罄状态重新分类")
    except Exception as e:
        logger.error(f"加载策略缓存失败: {e}")


# ── 策略执行（后台线程） ──

def run_strategy(force_refresh=False):
    """在后台线程中执行完整策略流水线"""
    if _get('status') == 'running':
        return False, '策略正在运行中'

    _set('status', 'running')
    _set('progress', 0)
    _set('message', '启动中...')
    _set('error', '')
    _set('started_at', datetime.now().isoformat())
    _set('finished_at', None)

    t = threading.Thread(target=_run_pipeline, args=(force_refresh,), daemon=True)
    t.start()
    return True, '已启动'


def _run_pipeline(force_refresh):
    # 挂载日志监听器，捕获策略内部进度
    strat_logger = logging.getLogger('bank_product_strategy_v6')
    _progress_handler.active = True
    strat_logger.addHandler(_progress_handler)

    try:
        # 延迟导入，避免模块级别的副作用
        import bank_product_strategy_v6 as strat

        # Step 1: 风控检查
        _emit(5, '信用利差风控检查...')
        risk_triggered, spread, spread_date, spread_basis = strat.检查信用利差风控()

        # Step 2: 构建产品库
        _emit(15, '构建高成功率产品库...')
        product_lib, watch_pool = strat.构建高成功率产品库(强制刷新=force_refresh)
        _set('product_lib', product_lib)
        _set('watch_pool', watch_pool)

        # Step 3: 前瞻预测 (释放规律学习 + 卡方检验) — V6.2: 提前到扫描之前，供新鲜度计算使用
        _emit(30, '学习释放规律并预测...')
        if not risk_triggered:
            _emit(32, '释放规律学习中 (工作日映射 + 卡方检验)...')
            prediction_data = strat.学习释放规律并预测(product_lib)
            _set('prediction_source', 'chi_square')
            _emit(45, '释放规律预测完成')
        else:
            prediction_data = {'prediction_scores': {}, 'patterns': {}, 'sorted_dates': [], 'date_to_idx': {}}
            _set('prediction_source', 'chi_square')
        _set('prediction_data', prediction_data)
        patterns = prediction_data.get('patterns', {})
        logger.info(f"[Strategy] 释放规律: {len(patterns)}个产品有规律 (传给扫描+推荐)")

        # Step 4: 扫描实时机会（传入释放规律供新鲜度计算）
        _emit(50, f'扫描实时机会（{len(product_lib)}个产品）...')
        opportunities = strat.扫描实时机会(product_lib, watch_pool, 释放规律=patterns)

        if risk_triggered:
            for o in opportunities:
                o['操作建议'] = f'!! 全部赎回（信用利差{spread}bp>{strat.信用利差阈值}bp）'

        _set('opportunities', opportunities)

        # Step 5: 加载持仓
        _emit(60, '加载持仓数据...')
        trade_records, current_positions = strat.加载持仓()
        if current_positions:
            _set('current_positions', current_positions)

        # Step 6: 持仓分析
        _emit(75, '分析持仓...')
        portfolio_analysis = []
        if current_positions:
            portfolio_analysis = strat.分析持仓(current_positions, product_lib, opportunities, 预测数据=prediction_data)
            if risk_triggered:
                for p in portfolio_analysis:
                    if p['持仓状态'] == '持有中':
                        p['持仓建议'] = f'!! 立即赎回（信用利差{spread}bp>{strat.信用利差阈值}bp）'
        _set('portfolio_analysis', portfolio_analysis)

        # Step 7: 精选推荐 + 新鲜度过滤
        _emit(85, '多维度精选推荐 + 新鲜度过滤...')
        today_str = datetime.now().strftime('%Y-%m-%d')
        is_pre_hol, pre_hol_name = strat.is_pre_holiday(today_str)
        pred_scores = prediction_data.get('prediction_scores', {})

        if not risk_triggered:
            recommendations = strat.精选推荐(
                opportunities, pred_scores, 产品库=product_lib,
                是否节前=is_pre_hol, 节前假期名=pre_hol_name,
                释放规律=patterns
            )
        else:
            recommendations = []
        _set('recommendations', recommendations)

        # Step 8: VIP Sniper 双通道分类
        _emit(92, '双通道分类...')
        import post_processing
        vip_channels = post_processing.classify_recommendations(recommendations)
        _set('vip_channels', vip_channels)

        # 完成
        _emit(100, f'完成 — 产品库{len(product_lib)}个，机会{len(opportunities)}个，推荐{len(recommendations)}个')
        _set('status', 'done')
        _set('finished_at', datetime.now().isoformat())

        # 持久化结果
        _save_cache()

    except Exception as e:
        logger.error(f"策略执行失败: {e}\n{traceback.format_exc()}")
        _set('status', 'error')
        _set('error', str(e))
        _set('finished_at', datetime.now().isoformat())
    finally:
        # 移除日志监听器
        _progress_handler.active = False
        strat_logger.removeHandler(_progress_handler)


# ── GPU 预测格式转换 ──

def _convert_gpu_predictions(gpu_preds, strat, product_lib):
    """将 gpu_predictor 输出转换为现有 prediction_data 格式

    gpu_preds: {(bank, code): {release_probs, confidence, predicted_date, score, ...}}
    需要构造: {prediction_scores: {key: {score, ...}}, patterns: {key: pattern}, ...}
    """
    prediction_scores = {}
    patterns = {}

    # 构建日期索引（从产品库中提取）
    sorted_dates = []
    date_to_idx = {}

    for key, pred in gpu_preds.items():
        prediction_scores[key] = {
            'score': pred.get('score', 0),
            'confidence': pred.get('confidence', 0),
            'predicted_date': pred.get('predicted_date', ''),
            'predicted_end': pred.get('predicted_end', ''),
            'model_type': pred.get('model_type', 'deep_learning'),
            'release_probs': pred.get('release_probs', []),
        }

    # 同时运行原始规律学习（用于模式展示，不替代预测分数）
    try:
        original_data = strat.学习释放规律并预测(product_lib)
        patterns = original_data.get('patterns', {})
        sorted_dates = original_data.get('sorted_dates', [])
        date_to_idx = original_data.get('date_to_idx', {})

        # 合并: GPU 预测分数覆盖原始分数
        orig_scores = original_data.get('prediction_scores', {})
        for key in orig_scores:
            if key not in prediction_scores:
                prediction_scores[key] = orig_scores[key]
    except Exception as e:
        logger.warning(f"[Strategy] 原始规律学习失败: {e}")

    return {
        'prediction_scores': prediction_scores,
        'patterns': patterns,
        'sorted_dates': sorted_dates,
        'date_to_idx': date_to_idx,
    }


# ── 数据访问（供 API 调用） ──

def get_summary():
    """KPI 摘要"""
    lib = _get('product_lib', [])
    opps = _get('opportunities', [])
    recs = _get('recommendations', [])
    positions = _get('current_positions', [])
    portfolio = _get('portfolio_analysis', [])

    holding = [p for p in (positions or []) if p.get('持仓状态') == '持有中']
    total_holding = sum(p.get('净持仓', 0) for p in holding)

    buy_signals = [o for o in opps if '★' in o.get('操作建议', '')]
    watchable = [o for o in opps if '☆' in o.get('操作建议', '')]

    avg_return = 0
    returns = [o.get('最新收益率%', 0) or 0 for o in opps if (o.get('最新收益率%', 0) or 0) > 0]
    if returns:
        avg_return = round(sum(returns) / len(returns), 2)

    # 银行分布
    bank_dist = {}
    for o in opps:
        b = o.get('银行', '未知')
        bank_dist[b] = bank_dist.get(b, 0) + 1

    # 获取净值数据库最新日期
    nav_latest_date = ''
    try:
        import crawl_bridge
        stats = crawl_bridge.get_db_stats()
        if stats:
            dates = [s.get('latest_date', '') for s in stats if s.get('latest_date') and '加载' not in s.get('bank', '')]
            if dates:
                nav_latest_date = max(dates)
    except Exception:
        pass

    # GPU 状态
    gpu_available = False
    try:
        from gpu_engine import is_gpu_available
        gpu_available = is_gpu_available()
    except Exception:
        pass

    return {
        'product_lib_count': len(lib),
        'watch_pool_count': len(_get('watch_pool', [])),
        'opportunity_count': len(opps),
        'buy_signal_count': len(buy_signals),
        'watch_signal_count': len(watchable),
        'recommendation_count': len(recs),
        'holding_count': len(holding),
        'total_holding_amount': round(total_holding, 2),
        'avg_return': avg_return,
        'bank_distribution': bank_dist,
        'status': _get('status'),
        'last_run': _get('finished_at'),
        'nav_data_date': nav_latest_date,
        'prediction_source': _get('prediction_source', 'chi_square'),
        'gpu_available': gpu_available,
    }


def get_top20():
    """精选推荐 Top 20"""
    return _get('recommendations', [])


def get_opportunities():
    """实时信号"""
    return _get('opportunities', [])


def get_portfolio():
    """持仓分析"""
    return _get('portfolio_analysis', [])


def get_patterns():
    """释放规律快照"""
    pred_data = _get('prediction_data', {})
    patterns = pred_data.get('patterns', {})
    pred_scores = pred_data.get('prediction_scores', {})

    # 如果没有原始 prediction_data（从缓存加载），返回预处理结果
    if not patterns:
        cached = _get('_patterns_cache', [])
        if cached:
            return cached

    result = []
    for key, pat in patterns.items():
        bank, code = key
        pred = pred_scores.get(key, {})
        weekday_names = ['周一', '周二', '周三', '周四', '周五']
        phase_names = ['月初(1-7)', '月中(8-14)', '月中下(15-21)', '月末(22-31)']

        result.append({
            'bank': bank,
            'code': code,
            'period_days': round(pat.period_days, 1),
            'has_period': pat.has_period,
            'period_cv': round(pat.period_cv, 3),
            'top_phase': phase_names[pat.top_phase] if 0 <= pat.top_phase < 4 else '',
            'phase_pvalue': round(pat.phase_pvalue, 4),
            'top_weekday': weekday_names[pat.top_weekday] if 0 <= pat.top_weekday < 5 else '',
            'weekday_pvalue': round(pat.weekday_pvalue, 4),
            'confidence': round(pat.confidence, 3),
            'n_events': pat.n_events,
            'last_release': pat.last_release_date,
            'avg_window_days': round(pat.avg_window_days, 1),
            'prediction': pred if pred else None,
        })

    result.sort(key=lambda x: -x['confidence'])
    return result


# ── VIP Sniper: 双通道访问函数 ──

def get_vip_channels():
    """返回双通道分类结果 {list_a, list_b, unclassified, stats}"""
    return _get('vip_channels', {})


def get_sold_out_list():
    """返回当前售罄产品列表"""
    import post_processing
    mgr = post_processing.get_sold_out_manager()
    return mgr.get_list()


def refresh_classification():
    """用缓存的 recommendations 重跑双通道分类（无需重跑策略）"""
    import post_processing
    recs = _get('recommendations', [])
    if not recs:
        return {'list_a': [], 'list_b': [], 'unclassified': [], 'stats': {}}
    vip_channels = post_processing.classify_recommendations(recs)
    _set('vip_channels', vip_channels)
    # 持久化到缓存文件，确保页面刷新后售罄状态不丢失
    _save_cache()
    logger.info("[VIP Sniper] refresh_classification 完成")
    return vip_channels


# ── 模块加载时尝试恢复上次结果 ──
_load_cache()
