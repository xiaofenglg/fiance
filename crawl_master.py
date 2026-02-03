# -*- coding: utf-8 -*-
"""
V13 Delta Sync Engine — 统一抓取指挥官

功能：
1. 加载Manifest一次，传递给所有爬虫
2. 支持全部6家银行（民生/华夏/中信/浦银/宁银/中邮）
3. 通过ThreadPoolExecutor并行抓取独立银行
4. 所有数据保存到Parquet（通过nav_db_excel）

被crawl_bridge.py调用，替代直接调用crawl_all_banks。
"""

import os
import sys
import time
import logging
import threading
import traceback
import concurrent.futures
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── 银行配置 ──

BANK_CONFIG = {
    'minsheng': {
        'name': '民生银行',
        'sheet': '民生银行',
        'group': 'main',       # main = crawl_all_banks 处理
    },
    'huaxia': {
        'name': '华夏银行',
        'sheet': '华夏银行',
        'group': 'main',
    },
    'citic': {
        'name': '中信银行',
        'sheet': '中信银行',
        'group': 'main',
    },
    'spdb': {
        'name': '浦银理财',
        'sheet': '浦银理财',
        'group': 'extra',
    },
    'ningyin': {
        'name': '宁银理财',
        'sheet': '宁银理财',
        'group': 'extra',
    },
    'psbc': {
        'name': '中邮理财',
        'sheet': '中邮理财',
        'group': 'extra',
    },
}


def get_all_bank_keys():
    return list(BANK_CONFIG.keys())


def get_bank_name(key):
    return BANK_CONFIG.get(key, {}).get('name', key)


# ── 各银行增量抓取函数 ──

def _crawl_main_bank(bank_key, full_history, max_months, manifest, progress_callback, stop_check=None):
    """抓取主要银行（民生/华夏/中信）— 委托给crawl_all_banks"""
    import crawl_all_banks as cab

    func_map = {
        'minsheng': 'crawl_minsheng',
        'huaxia': 'crawl_huaxia',
        'citic': 'crawl_citic',
    }
    func_name = func_map.get(bank_key)
    if not func_name or not hasattr(cab, func_name):
        return {'bank': get_bank_name(bank_key), 'success': False, 'error': f'函数不存在: {func_name}'}

    func = getattr(cab, func_name)
    kwargs = {
        'full_history': full_history,
        'progress_callback': progress_callback,
        'manifest': manifest,
        'stop_check': stop_check,
    }
    if bank_key == 'citic' and max_months > 0:
        kwargs['max_months'] = max_months

    return func(**kwargs)


def _crawl_spdb(manifest=None, progress_callback=None, stop_check=None):
    """浦银理财增量抓取"""
    result = {'bank': '浦银理财', 'success': False, 'products': 0, 'error': None}
    try:
        from spdb_wm import SPDBWealthCrawler
        from nav_db_excel import update_nav_database
        from crawl_utils import needs_update

        # 赎回费数据库
        try:
            from redemption_fee_db import (update_fee_info, has_fee_data,
                                            save_fee_db)
            has_fee_module = True
        except ImportError:
            has_fee_module = False

        if stop_check and stop_check():
            result['error'] = '用户停止'
            return result

        crawler = SPDBWealthCrawler()
        bank_m = (manifest or {}).get('浦银理财', {})

        # 获取产品列表（传入stop_check以支持分页中断）
        products = crawler.get_product_list(stop_check=stop_check)

        if stop_check and stop_check():
            result['error'] = '用户停止'
            return result
        if not products:
            result['error'] = "未获取到产品列表"
            return result

        total = len(products)
        skipped = 0
        products_for_db = []
        saved_total = 0
        BATCH_SIZE = 200

        for i, product in enumerate(products):
            if stop_check and stop_check():
                logger.info(f"[浦银] 收到停止信号，已处理 {i}/{total}")
                break

            code = product.get('product_code', '') or product.get('PRDC_CD', '')
            name = product.get('product_name', '') or product.get('PRDC_NM', '')
            code = str(code).strip()

            # Delta sync: 跳过已最新
            if bank_m and code and not needs_update(code, bank_m):
                skipped += 1
                continue

            # 赎回费提取: 新产品首次爬取时获取费用
            if has_fee_module and code and not has_fee_data('浦银理财', code):
                fee_info = crawler._extract_fee_info(product)
                if fee_info:
                    update_fee_info('浦银理财', code, fee_info)
                else:
                    update_fee_info('浦银理财', code, {
                        'has_redemption_fee': False,
                        'fee_schedule': [],
                        'fee_description': '',
                        'source': 'name_parse',
                    })

            # 抓取净值（增量模式）
            try:
                nav_data = crawler.get_nav_history(code, full_history=False)
            except Exception:
                nav_data = []

            if nav_data:
                products_for_db.append({
                    'product_code': code,
                    'product_name': name,
                    'nav_history': nav_data,
                })

            # 分批保存，防止中断丢失
            if len(products_for_db) >= BATCH_SIZE:
                stats = update_nav_database('浦银', products_for_db)
                saved_total += len(products_for_db)
                logger.info(f"[浦银] 分批保存: {saved_total} 个产品已入库")
                products_for_db = []

            if progress_callback:
                progress_callback(i + 1, total, f'浦银理财 {i+1}/{total}')

        if skipped > 0:
            logger.info(f"[浦银] Delta Sync: {total} 总产品, 跳过 {skipped}, 需更新 {total - skipped}")

        # 保存剩余
        if products_for_db:
            stats = update_nav_database('浦银', products_for_db)
            saved_total += len(products_for_db)
            logger.info(f"[浦银] 数据库更新完成: 共 {saved_total} 个产品")

        # 保存赎回费数据库
        if has_fee_module:
            save_fee_db()

        result['success'] = True
        result['products'] = saved_total

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[浦银] 抓取失败: {e}")

    return result


def _crawl_ningyin(manifest=None, progress_callback=None, stop_check=None):
    """宁银理财增量抓取"""
    result = {'bank': '宁银理财', 'success': False, 'products': 0, 'error': None}
    try:
        from crawl_ningyin_2y import NingyinCrawler
        from nav_db_excel import update_nav_database
        from crawl_utils import needs_update

        if stop_check and stop_check():
            result['error'] = '用户停止'
            return result

        crawler = NingyinCrawler()
        bank_m = (manifest or {}).get('宁银理财', {})

        products = crawler.get_product_list(stop_check=stop_check)

        if stop_check and stop_check():
            result['error'] = '用户停止'
            return result
        if not products:
            result['error'] = "未获取到产品列表"
            return result

        total = len(products)
        skipped = 0
        products_for_db = []
        saved_total = 0
        BATCH_SIZE = 200

        for i, product in enumerate(products):
            if stop_check and stop_check():
                logger.info(f"[宁银] 收到停止信号，已处理 {i}/{total}")
                break

            code = product.get('projectcode', '') or product.get('product_code', '') or product.get('code', '')
            name = product.get('projectshortname', '') or product.get('projectname', '') or product.get('product_name', '')
            code = str(code).strip()

            if bank_m and code and not needs_update(code, bank_m):
                skipped += 1
                continue

            try:
                # 增量: 只取近30天
                nav_data = crawler.get_nav_history(code, days=30)
            except Exception as e:
                logger.warning(f"[宁银] 产品 {code} 净值获取失败: {e}")
                nav_data = []

            if nav_data:
                products_for_db.append({
                    'product_code': code,
                    'product_name': name,
                    'nav_history': nav_data,
                })

            # 分批保存，防止中断丢失
            if len(products_for_db) >= BATCH_SIZE:
                stats = update_nav_database('宁银', products_for_db)
                saved_total += len(products_for_db)
                logger.info(f"[宁银] 分批保存: {saved_total} 个产品已入库")
                products_for_db = []

            if progress_callback:
                progress_callback(i + 1, total, f'宁银理财 {i+1}/{total}')

        if skipped > 0:
            logger.info(f"[宁银] Delta Sync: {total} 总产品, 跳过 {skipped}, 需更新 {total - skipped}")

        # 保存剩余
        if products_for_db:
            stats = update_nav_database('宁银', products_for_db)
            saved_total += len(products_for_db)
            logger.info(f"[宁银] 数据库更新完成: 共 {saved_total} 个产品")

        result['success'] = True
        result['products'] = saved_total

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[宁银] 抓取失败: {e}")

    return result


def _crawl_psbc(manifest=None, progress_callback=None, stop_check=None):
    """中邮理财增量抓取"""
    result = {'bank': '中邮理财', 'success': False, 'products': 0, 'error': None}
    try:
        from crawl_psbc_2y import PSBCWMCrawler, get_waf_cookies
        from nav_db_excel import update_nav_database
        from crawl_utils import needs_update

        if stop_check and stop_check():
            result['error'] = '用户停止'
            return result

        bank_m = (manifest or {}).get('中邮理财', {})

        # 中邮需要WAF cookies
        logger.info("[中邮] 获取WAF cookies...")
        waf_cookies = get_waf_cookies()
        if not waf_cookies:
            result['error'] = "无法获取WAF cookies（需要Selenium）"
            return result

        crawler = PSBCWMCrawler(waf_cookies)

        products = crawler.get_product_list(stop_check=stop_check)

        if stop_check and stop_check():
            result['error'] = '用户停止'
            return result
        if not products:
            result['error'] = "未获取到产品列表"
            return result

        total = len(products)
        skipped = 0
        products_for_db = []
        saved_total = 0
        BATCH_SIZE = 200

        for i, product in enumerate(products):
            if stop_check and stop_check():
                logger.info(f"[中邮] 收到停止信号，已处理 {i}/{total}")
                break

            code = product.get('product_code', '') or product.get('wp_code', '')
            name = product.get('product_name', '') or product.get('wp_name', '')
            code = str(code).strip()

            if bank_m and code and not needs_update(code, bank_m):
                skipped += 1
                continue

            try:
                nav_data = crawler.get_nav_history(code, max_pages=5, stop_check=stop_check)
            except Exception as e:
                logger.warning(f"[中邮] 产品 {code} 净值获取失败: {e}")
                nav_data = []

            if nav_data:
                products_for_db.append({
                    'product_code': code,
                    'product_name': name,
                    'nav_history': nav_data,
                })

            # 分批保存，防止中断丢失
            if len(products_for_db) >= BATCH_SIZE:
                stats = update_nav_database('中邮', products_for_db)
                saved_total += len(products_for_db)
                logger.info(f"[中邮] 分批保存: {saved_total} 个产品已入库")
                products_for_db = []

            if progress_callback:
                progress_callback(i + 1, total, f'中邮理财 {i+1}/{total}')

        if skipped > 0:
            logger.info(f"[中邮] Delta Sync: {total} 总产品, 跳过 {skipped}, 需更新 {total - skipped}")

        # 保存剩余
        if products_for_db:
            stats = update_nav_database('中邮', products_for_db)
            saved_total += len(products_for_db)
            logger.info(f"[中邮] 数据库更新完成: 共 {saved_total} 个产品")

        result['success'] = True
        result['products'] = saved_total

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[中邮] 抓取失败: {e}")

    return result


# ── 调度器映射 ──

_CRAWL_DISPATCH = {
    'minsheng': lambda **kw: _crawl_main_bank('minsheng', **kw),
    'huaxia':   lambda **kw: _crawl_main_bank('huaxia', **kw),
    'citic':    lambda **kw: _crawl_main_bank('citic', **kw),
    'spdb':     lambda manifest=None, progress_callback=None, stop_check=None, **_: _crawl_spdb(manifest, progress_callback, stop_check),
    'ningyin':  lambda manifest=None, progress_callback=None, stop_check=None, **_: _crawl_ningyin(manifest, progress_callback, stop_check),
    'psbc':     lambda manifest=None, progress_callback=None, stop_check=None, **_: _crawl_psbc(manifest, progress_callback, stop_check),
}


def crawl_single(bank_key, full_history=False, max_months=0, manifest=None, progress_callback=None, stop_check=None):
    """抓取单个银行"""
    dispatch = _CRAWL_DISPATCH.get(bank_key)
    if not dispatch:
        return {'bank': bank_key, 'success': False, 'error': f'未知银行: {bank_key}'}

    return dispatch(
        full_history=full_history,
        max_months=max_months,
        manifest=manifest,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )


def crawl_all(banks=None, full_history=False, max_months=0, bank_modes=None,
              progress_callback=None, bank_progress_callback=None,
              stop_check=None, bank_stop_check=None):
    """统一抓取入口 — 加载Manifest一次，并行执行各银行

    Args:
        banks: 银行key列表，None=全部6家
        full_history: 全量模式（默认值，可被bank_modes覆盖）
        max_months: 中信专用月份数
        bank_modes: {bank_key: bool} per-bank模式，True=全量, False=增量
        progress_callback: (progress_pct, message) 回调
        bank_progress_callback: (bank_key, bank_name, current, total, status) per-bank进度回调
        stop_check: callable，返回 True 表示全局应停止
        bank_stop_check: callable(bank_key)，返回 True 表示该银行应停止

    Returns:
        list[dict]: 各银行结果
    """
    from crawl_utils import get_db_manifest, manifest_summary

    if not banks:
        banks = ['minsheng', 'huaxia', 'citic']  # 默认3家主银行

    # ★ V13核心: 加载Manifest一次
    t0 = time.time()
    manifest = get_db_manifest()
    t_manifest = time.time() - t0
    logger.info(f"[Master] Manifest加载 ({t_manifest:.2f}s): {manifest_summary(manifest)}")

    if progress_callback:
        progress_callback(1, f'Manifest已加载 ({t_manifest:.1f}s)')

    total = len(banks)

    # ── 并行状态跟踪 ──
    _progress_lock = threading.Lock()
    # 每家银行的结构化进度: {bank_key: {current, total}}
    _bank_progress_data = {}
    _completed_count = 0

    def _build_progress_message():
        """构建多银行并发进度文本"""
        parts = []
        for bk in banks:
            if bk in _bank_progress_data:
                bp = _bank_progress_data[bk]
                name = get_bank_name(bk)
                if bp.get('status') == 'done':
                    parts.append(f'{name} 完成')
                elif bp.get('status') == 'error':
                    parts.append(f'{name} 失败')
                elif bp.get('total', 0) > 0:
                    parts.append(f'{name} {bp["current"]}/{bp["total"]}')
                else:
                    parts.append(f'{name} 启动中...')
        return ' | '.join(parts) if parts else '抓取中...'

    def _emit_progress():
        """基于各银行产品级进度加权计算总百分比并回调"""
        if not progress_callback:
            return
        # total_pct = 2 + sum(bank.current/bank.total for each bank) / num_banks * 93
        weighted_sum = 0
        for bk in banks:
            bp = _bank_progress_data.get(bk, {})
            bp_total = bp.get('total', 0)
            bp_current = bp.get('current', 0)
            if bp.get('status') in ('done', 'error'):
                weighted_sum += 1.0
            elif bp_total > 0:
                weighted_sum += min(bp_current / bp_total, 1.0)
        pct = int(2 + weighted_sum / total * 93)
        pct = min(pct, 95)
        progress_callback(pct, _build_progress_message())

    def _emit_bank(bank_key, bank_name, current, total_items, status):
        """发送 per-bank 结构化进度"""
        if bank_progress_callback:
            bank_progress_callback(bank_key, bank_name, current, total_items, status)

    def _make_bank_cb(bank_key, bank_name):
        """创建per-bank进度回调（线程安全）"""
        def _cb(current, total_items, msg):
            with _progress_lock:
                _bank_progress_data[bank_key] = {
                    'current': current,
                    'total': total_items,
                    'status': 'running',
                }
                _emit_bank(bank_key, bank_name, current, total_items, 'running')
                _emit_progress()
        return _cb

    def _make_bank_stop_check(bank_key):
        """创建 per-bank 停止检查函数"""
        def _check():
            if stop_check and stop_check():
                return True
            if bank_stop_check and bank_stop_check(bank_key):
                return True
            return False
        return _check

    def _crawl_one(bank_key):
        """单个银行的抓取任务（在线程池中执行）"""
        nonlocal _completed_count
        bank_name = get_bank_name(bank_key)
        bank_stop_fn = _make_bank_stop_check(bank_key)

        # 检查停止信号
        if bank_stop_fn():
            with _progress_lock:
                _completed_count += 1
                _bank_progress_data[bank_key] = {'current': 0, 'total': 0, 'status': 'stopped'}
                _emit_bank(bank_key, bank_name, 0, 0, 'stopped')
                _emit_progress()
            return {
                'bank': bank_name,
                'bank_key': bank_key,
                'success': False,
                'products': 0,
                'error': '用户停止',
            }

        with _progress_lock:
            _bank_progress_data[bank_key] = {'current': 0, 'total': 0, 'status': 'running'}
            _emit_bank(bank_key, bank_name, 0, 0, 'running')
            _emit_progress()

        bank_cb = _make_bank_cb(bank_key, bank_name)

        # 确定此银行的模式
        bank_full = full_history
        if bank_modes and bank_key in bank_modes:
            bank_full = bank_modes[bank_key]

        try:
            t_bank = time.time()
            result = crawl_single(
                bank_key,
                full_history=bank_full,
                max_months=max_months,
                manifest=manifest,
                progress_callback=bank_cb,
                stop_check=bank_stop_fn,
            )
            elapsed = time.time() - t_bank

            products = result.get('products', 0) if isinstance(result, dict) else 0
            success = result.get('success', False) if isinstance(result, dict) else False

            # 检查是否在执行过程中收到停止信号
            was_stopped = bank_stop_fn()
            if was_stopped:
                final_status = 'stopped'
            else:
                final_status = 'done' if success else 'error'
            bp = _bank_progress_data.get(bank_key, {})
            final_total = bp.get('total', products) or products

            with _progress_lock:
                _completed_count += 1
                _bank_progress_data[bank_key] = {
                    'current': final_total if success else bp.get('current', 0),
                    'total': final_total,
                    'status': final_status,
                }
                _emit_bank(bank_key, bank_name,
                           final_total if success else bp.get('current', 0),
                           final_total, final_status)
                _emit_progress()

            status = "OK" if success else "FAIL"
            logger.info(f"[Master] {bank_name}: {status} ({products} 产品, {elapsed:.1f}s)")

            return {
                'bank': bank_name,
                'bank_key': bank_key,
                'success': success,
                'products': products,
                'elapsed': round(elapsed, 1),
                'error': result.get('error') if isinstance(result, dict) else None,
            }

        except Exception as e:
            logger.error(f"[Master] {bank_name} 异常: {e}\n{traceback.format_exc()}")
            bp = _bank_progress_data.get(bank_key, {})
            with _progress_lock:
                _completed_count += 1
                _bank_progress_data[bank_key] = {
                    'current': bp.get('current', 0),
                    'total': bp.get('total', 0),
                    'status': 'error',
                }
                _emit_bank(bank_key, bank_name, bp.get('current', 0), bp.get('total', 0), 'error')
                _emit_progress()

            return {
                'bank': bank_name,
                'bank_key': bank_key,
                'success': False,
                'products': 0,
                'error': str(e),
            }

    # ── 使用 ThreadPoolExecutor 并行抓取所有银行 ──
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(banks)) as executor:
        future_map = {executor.submit(_crawl_one, bk): bk for bk in banks}

        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())

    # 按原始 banks 顺序排列结果
    order = {bk: i for i, bk in enumerate(banks)}
    results.sort(key=lambda r: order.get(r.get('bank_key', ''), 999))

    # 清理临时字段
    for r in results:
        r.pop('bank_key', None)

    if progress_callback:
        success_count = sum(1 for r in results if r.get('success'))
        progress_callback(100, f'全部完成 — {success_count}/{total} 家银行成功')

    return results


# ── CLI入口 ──

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crawl_master.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description='V13 统一抓取指挥官')
    parser.add_argument('--banks', nargs='*', help='银行列表 (minsheng huaxia citic spdb ningyin psbc)')
    parser.add_argument('--full', action='store_true', help='全量模式')
    parser.add_argument('--max-months', type=int, default=0, help='中信最大月份')
    parser.add_argument('--all', action='store_true', help='抓取全部6家银行')
    args = parser.parse_args()

    banks = args.banks
    if args.all:
        banks = get_all_bank_keys()

    def _cli_progress(pct, msg):
        print(f"  [{pct:3d}%] {msg}")

    t0 = time.time()
    results = crawl_all(
        banks=banks,
        full_history=args.full,
        max_months=args.max_months,
        progress_callback=_cli_progress,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"  V13 抓取完成 — 总耗时 {elapsed:.1f}s")
    print("=" * 60)
    for r in results:
        status = "OK" if r['success'] else f"FAIL: {r.get('error', '')}"
        print(f"  {r['bank']}: {status} ({r.get('products', 0)} 产品)")
