# -*- coding: utf-8 -*-
"""
净值抓取适配层 — 连接 Flask API 与 crawl_all_banks.py

线程安全地调用抓取函数，SSE 推送进度，缓存结果供 API 返回。
支持 per-bank 独立模式选择和 per-bank 独立停止。
"""

print("[CRAWL_BRIDGE] Module loaded from:", __file__)

import os
import threading
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 全局状态 ──
_lock = threading.Lock()
_state = {
    'status': 'idle',        # idle / running / done / error / stopped
    'progress': 0,           # 0-100
    'message': '',
    'error': '',
    'started_at': None,
    'finished_at': None,
    # 缓存结果
    'results': [],           # 每个银行的抓取结果
    # per-bank 进度
    'bank_progress': {},     # {bank_key: {name, current, total, status}}
}

# 停止信号
_stop_event = threading.Event()          # 全局停止
_bank_stop_events = {}                   # {bank_key: threading.Event()} per-bank停止

# ── 数据库统计缓存 ──
_stats_cache = None
_stats_loading = False


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
            'bank_progress': {k: dict(v) for k, v in _state['bank_progress'].items()},
        }


def _emit(progress, message):
    with _lock:
        _state['progress'] = progress
        _state['message'] = message
    logger.info(f"[Crawl] {progress}% — {message}")


def _emit_bank_progress(bank_key, bank_name, current, total, status):
    """更新单个银行的进度状态（线程安全）

    注意：已标记为 stopped/done/error 的银行不会被后台线程覆盖回 running
    """
    with _lock:
        existing = _state['bank_progress'].get(bank_key, {})
        existing_status = existing.get('status')
        # 已终态(stopped/done/error)不允许被后台线程覆盖回running
        if existing_status in ('stopped', 'done', 'error') and status == 'running':
            return
        _state['bank_progress'][bank_key] = {
            'name': bank_name,
            'current': current,
            'total': total,
            'status': status,  # pending / running / done / error / stopped
        }


def get_results():
    return _get('results', [])


def is_stopped():
    """检查全局停止信号"""
    return _stop_event.is_set()


def is_bank_stopped(bank_key):
    """检查特定银行是否收到停止信号（全局或单独）"""
    if _stop_event.is_set():
        return True
    evt = _bank_stop_events.get(bank_key)
    return evt.is_set() if evt else False


def stop_crawl(bank_key=None):
    """请求停止抓取

    Args:
        bank_key: None=停止全部, str=停止指定银行
    """
    if _get('status') != 'running':
        return False, '当前没有运行中的抓取'

    if bank_key:
        # 停止单个银行
        evt = _bank_stop_events.get(bank_key)
        if evt:
            evt.set()
            # 立即在UI状态中标记为stopped（不等待后台线程响应）
            with _lock:
                bp = _state['bank_progress'].get(bank_key)
                if bp and bp.get('status') in ('running', 'pending'):
                    bp['status'] = 'stopped'
            logger.info(f"[Crawl] 收到停止请求: {bank_key}")
            return True, f'已发送停止信号: {bank_key}'
        return False, f'未找到银行: {bank_key}'
    else:
        # 停止全部
        _stop_event.set()
        # 立即将所有运行中的银行标记为stopped
        with _lock:
            for bk, bp in _state['bank_progress'].items():
                if bp.get('status') in ('running', 'pending'):
                    bp['status'] = 'stopped'
        _emit(_get('progress', 0), '正在停止...')
        logger.info("[Crawl] 收到全局停止请求")
        return True, '已发送停止信号'


# ── 数据库统计（带缓存） ──

def _load_stats_background():
    """后台加载数据库统计（含赎回费数据库信息）— 直接查询SQLite"""
    global _stats_cache, _stats_loading
    logger.info("[V3] _load_stats_background using SQLite direct query")
    try:
        import sqlite3

        # 尝试 SQLite 数据库 (aifinance.sqlite3)
        db_path = os.path.join(BASE_DIR, 'aifinance.sqlite3')
        logger.info(f"[V3] Checking db_path: {db_path}, exists: {os.path.exists(db_path)}")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 获取各银行统计
            cursor.execute("""
                SELECT
                    p.bank_name,
                    COUNT(DISTINCT p.product_code) as products,
                    COUNT(DISTINCT h.date) as dates,
                    MIN(h.date) as earliest_date,
                    MAX(h.date) as latest_date
                FROM products p
                LEFT JOIN nav_history h ON p.product_code = h.product_code
                GROUP BY p.bank_name
                ORDER BY p.bank_name
            """)
            rows = cursor.fetchall()
            conn.close()

            result = []
            for row in rows:
                result.append({
                    'bank': row[0],
                    'products': row[1] or 0,
                    'dates': row[2] or 0,
                    'earliest_date': row[3] or '',
                    'latest_date': row[4] or '',
                })
        else:
            # 回退到 NAVDatabaseExcel
            from nav_db_excel import NAVDatabaseExcel
            db = NAVDatabaseExcel()
            raw = db.get_stats()
            result = []
            for sheet_name, info in raw.items():
                result.append({
                    'bank': sheet_name,
                    'products': info.get('products', 0),
                    'dates': info.get('dates', 0),
                    'earliest_date': info.get('earliest_date', ''),
                    'latest_date': info.get('latest_date', ''),
                })

        # V13: 加载赎回费数据库统计
        fee_stats = _get_fee_db_stats()

        _stats_cache = {
            'banks': result,
            'fee_db': fee_stats,
        }
        logger.info(f"[Crawl] 数据库统计已加载: {len(result)} 家银行, "
                    f"费率数据: {fee_stats['with_fee']}/{fee_stats['total']} 产品")
    except Exception as e:
        logger.error(f"获取数据库统计失败: {e}")
        _stats_cache = {'banks': [], 'fee_db': {'total': 0, 'with_fee': 0}}
    finally:
        _stats_loading = False


def _get_fee_db_stats():
    """获取赎回费数据库统计信息"""
    try:
        import fee_engine
        fee_engine.reload_fee_database()
        db = fee_engine.get_fee_database()
        data = db.load()
        products = data.get('products', {})
        total = len(products)
        with_fee = sum(1 for v in products.values() if v.get('has_redemption_fee'))
        return {
            'total': total,
            'with_fee': with_fee,
            'coverage': f"{with_fee/total*100:.1f}%" if total > 0 else "0%",
        }
    except Exception as e:
        logger.warning(f"获取费率数据库统计失败: {e}")
        return {'total': 0, 'with_fee': 0, 'coverage': '0%'}


def get_db_stats(force_refresh=False):
    """获取净值数据库各银行的产品数/日期范围（带缓存）

    返回结构 V13:
    {
        'banks': [{bank, products, dates, earliest_date, latest_date}, ...],
        'fee_db': {total, with_fee, coverage}
    }

    force_refresh=True 时同步加载（阻塞但保证最新数据）
    """
    global _stats_cache, _stats_loading

    _empty_result = {
        'banks': [{'bank': '加载中...', 'products': 0, 'dates': 0, 'earliest_date': '', 'latest_date': ''}],
        'fee_db': {'total': 0, 'with_fee': 0, 'coverage': '0%'}
    }

    if force_refresh:
        # 同步加载确保返回最新数据
        _stats_cache = None
        _stats_loading = True
        _load_stats_background()
        if _stats_cache is not None:
            return _stats_cache
        return _empty_result

    if _stats_cache is not None:
        return _stats_cache

    if _stats_loading:
        return _empty_result

    # 首次调用：后台线程加载
    _stats_loading = True
    t = threading.Thread(target=_load_stats_background, daemon=True)
    t.start()

    # 尝试直接从文件快速读取基本信息
    parquet_file = os.path.join(BASE_DIR, "净值数据库.parquet")
    db_file = os.path.join(BASE_DIR, "净值数据库.xlsx")
    if os.path.exists(parquet_file) or os.path.exists(db_file):
        return {
            'banks': [{'bank': '数据库加载中...', 'products': 0, 'dates': 0,
                       'earliest_date': '', 'latest_date': ''}],
            'fee_db': {'total': 0, 'with_fee': 0, 'coverage': '0%'}
        }
    return {'banks': [], 'fee_db': {'total': 0, 'with_fee': 0, 'coverage': '0%'}}


def refresh_stats_cache():
    """抓取完成后刷新缓存"""
    global _stats_cache
    _stats_cache = None
    get_db_stats(force_refresh=True)


def _sync_fees_to_sqlite():
    """同步赎回费数据到SQLite — 抓取完成后自动调用"""
    try:
        import migrate_fees_to_sqlite
        logger.info("[Crawl] 同步费率数据到SQLite...")
        result = migrate_fees_to_sqlite.migrate()
        if result == 0:
            logger.info("[Crawl] 费率数据同步成功")
        else:
            logger.warning(f"[Crawl] 费率数据同步失败: return code {result}")
    except ImportError:
        logger.warning("[Crawl] migrate_fees_to_sqlite模块不可用")
    except Exception as e:
        logger.error(f"[Crawl] 费率数据同步出错: {e}")


# ── 抓取执行（后台线程） — V13 Delta Sync Engine ──

def run_crawl(banks=None, full_history=False, max_months=0, bank_modes=None):
    """在后台线程中执行净值抓取（V13: 通过crawl_master统一调度）

    Args:
        banks: 银行列表，支持6家: ['minsheng','huaxia','citic','spdb','ningyin','psbc']
               None=默认3家主银行
        full_history: True=全量抓取, False=增量(Delta Sync) — 默认值，可被bank_modes覆盖
        max_months: >0时覆盖默认区间（仅支持中信），例如24=2年
        bank_modes: {bank_key: bool} per-bank模式选择，True=全量, False=增量
                    例如: {'minsheng': False, 'ningyin': True, 'citic': False}
    """
    if _get('status') == 'running':
        return False, '抓取正在运行中'

    if not banks:
        banks = ['minsheng', 'huaxia', 'citic']

    # 初始化 per-bank 停止事件
    global _bank_stop_events
    _bank_stop_events = {bk: threading.Event() for bk in banks}
    _stop_event.clear()

    _set('status', 'running')
    _set('progress', 0)
    _set('message', '启动中...')
    _set('error', '')
    _set('results', [])
    _set('bank_progress', {})
    _set('started_at', datetime.now().isoformat())
    _set('finished_at', None)

    t = threading.Thread(
        target=_run_crawl,
        args=(banks, full_history, max_months, bank_modes),
        daemon=True
    )
    t.start()
    return True, '已启动'


def _run_crawl(banks, full_history, max_months=0, bank_modes=None):
    try:
        import crawl_master

        def _progress_cb(pct, msg):
            _emit(pct, msg)

        def _bank_progress_cb(bank_key, bank_name, current, total, status):
            _emit_bank_progress(bank_key, bank_name, current, total, status)

        results = crawl_master.crawl_all(
            banks=banks,
            full_history=full_history,
            max_months=max_months,
            bank_modes=bank_modes,
            progress_callback=_progress_cb,
            bank_progress_callback=_bank_progress_cb,
            stop_check=is_stopped,
            bank_stop_check=is_bank_stopped,
        )

        _set('results', results)

        if _stop_event.is_set():
            success_count = sum(1 for r in results if r.get('success'))
            total = len(results)
            _emit(_get('progress', 0), f'已停止 — {success_count}/{total} 家银行已完成')
            _set('status', 'stopped')
            _set('finished_at', datetime.now().isoformat())
            # 同步已采集的费率数据
            _sync_fees_to_sqlite()
            refresh_stats_cache()
            return

        success_count = sum(1 for r in results if r.get('success'))
        total = len(results)
        _emit(100, f'抓取完成 — {success_count}/{total} 家银行成功')
        _set('status', 'done')
        _set('finished_at', datetime.now().isoformat())

        # 同步费率数据到SQLite
        _sync_fees_to_sqlite()

        # 刷新统计缓存
        refresh_stats_cache()

    except Exception as e:
        logger.error(f"抓取执行失败: {e}\n{traceback.format_exc()}")
        _set('status', 'error')
        _set('error', str(e))
        _set('finished_at', datetime.now().isoformat())


# ── 模块加载时自动启动后台加载统计 ──
threading.Thread(target=_load_stats_background, daemon=True).start()
