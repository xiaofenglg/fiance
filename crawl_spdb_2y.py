# -*- coding: utf-8 -*-
"""
浦银理财2年净值数据抓取 — 多进程×多线程 + 断点续抓

获取浦银理财个人理财产品的2年(730天)净值数据，
并增量导入净值数据库Excel。

特性:
- 多进程×多线程并发（N进程 × M线程）
- 每批完成后保存JSON检查点
- 断点续抓（跳过已有检查点的产品）
- 增量DB导入（不覆盖已有数据）

用法:
  python crawl_spdb_2y.py                    # 默认 6进程×4线程
  python crawl_spdb_2y.py --workers 8 --threads 5
  python crawl_spdb_2y.py --batch-size 200
  python crawl_spdb_2y.py --import-only      # 仅导入已有检查点
  python crawl_spdb_2y.py --clean            # 清理检查点重新抓取
"""

import sys
import os
import json
import time
import glob as glob_mod
import logging
import argparse
import threading
from datetime import datetime
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spdb_checkpoints')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('crawl_spdb_2y')


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_checkpoint_file(batch_id):
    return os.path.join(CHECKPOINT_DIR, f'batch_{batch_id:04d}.json')


def save_checkpoint(batch_id, products_data, status='done'):
    fp = get_checkpoint_file(batch_id)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump({
            'batch_id': batch_id,
            'status': status,
            'count': len(products_data),
            'timestamp': datetime.now().isoformat(),
            'products': products_data,
        }, f, ensure_ascii=False)


def load_all_checkpoints():
    all_products = []
    done_codes = set()
    pattern = os.path.join(CHECKPOINT_DIR, 'batch_*.json')
    for fp in sorted(glob_mod.glob(pattern)):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('status') == 'done':
                for p in data.get('products', []):
                    code = p.get('product_code', '')
                    if code and code not in done_codes:
                        all_products.append(p)
                        done_codes.add(code)
        except (json.JSONDecodeError, IOError):
            continue
    return all_products, done_codes


def crawl_worker(worker_id, product_batch, batch_id, num_threads, result_queue):
    """
    工作进程：多线程抓取一批产品的净值数据

    每个线程维护自己的session（thread-local），
    通过ThreadPoolExecutor实现线程级并发。
    """
    worker_logger = logging.getLogger(f'worker-{worker_id}')
    worker_logger.setLevel(logging.INFO)

    try:
        from spdb_wm import SPDBWealthCrawler

        crawler = SPDBWealthCrawler()
        results = []
        results_lock = threading.Lock()
        progress_count = [0]  # 用list包装以便在闭包中修改
        progress_lock = threading.Lock()
        total = len(product_batch)
        t0 = time.time()

        # thread-local storage for sessions
        local = threading.local()

        def get_session():
            if not hasattr(local, 'session') or local.session is None:
                local.session = crawler._create_session()
                local.count = 0
            local.count += 1
            if local.count >= 100:
                local.session.close()
                local.session = crawler._create_session()
                local.count = 0
            return local.session

        def fetch_product(product):
            product_code = product.get('product_code', '')
            product_name = product.get('product_name', '')
            if not product_code:
                return None

            session = get_session()
            try:
                nav_history = crawler.get_nav_history(
                    product_code,
                    session=session,
                    full_history=True,
                )
            except Exception as e:
                worker_logger.debug(f"产品 {product_code} 异常: {e}")
                # 重建session再试一次
                local.session.close()
                local.session = crawler._create_session()
                local.count = 0
                try:
                    nav_history = crawler.get_nav_history(
                        product_code,
                        session=local.session,
                        full_history=True,
                    )
                except Exception:
                    nav_history = []

            result = None
            if nav_history:
                result = {
                    'product_code': product_code,
                    'product_name': product_name,
                    'nav_history': nav_history,
                }
                with results_lock:
                    results.append(result)

            # 进度统计
            with progress_lock:
                progress_count[0] += 1
                count = progress_count[0]

            if count % 50 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else 0
                with results_lock:
                    valid = len(results)
                    avg_nav = (sum(len(p['nav_history']) for p in results) /
                               valid) if valid else 0
                worker_logger.info(
                    f"批次{batch_id} 进度: {count}/{total} ({count / total * 100:.1f}%), "
                    f"有效: {valid}, 平均: {avg_nav:.0f}条/产品, "
                    f"ETA: {eta / 60:.1f}分钟"
                )

            return result

        # 多线程执行
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(fetch_product, p): p
                for p in product_batch
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    worker_logger.debug(f"线程异常: {e}")

        # 保存检查点
        save_checkpoint(batch_id, results, status='done')

        elapsed = time.time() - t0
        worker_logger.info(
            f"批次{batch_id} 完成: {len(results)}/{total} 有效产品, "
            f"耗时 {elapsed / 60:.1f}分钟, "
            f"速度 {total / elapsed * 60:.0f}个/分钟"
        )

        result_queue.put({
            'batch_id': batch_id,
            'worker_id': worker_id,
            'total': total,
            'valid': len(results),
            'elapsed': elapsed,
            'status': 'done',
        })

    except Exception as e:
        worker_logger.error(f"批次{batch_id} 异常: {e}")
        if results:
            save_checkpoint(batch_id, results, status='partial')
        result_queue.put({
            'batch_id': batch_id,
            'worker_id': worker_id,
            'total': len(product_batch),
            'valid': len(results) if 'results' in dir() else 0,
            'status': 'error',
            'error': str(e),
        })


def format_nav_for_db(products_data):
    formatted = []
    for p in products_data:
        nav_list = []
        for nav in p.get('nav_history', []):
            date_val = nav.get('date', '')
            unit_nav = nav.get('unit_nav')
            if date_val and unit_nav is not None:
                date_clean = str(date_val).strip()
                if len(date_clean) == 8 and date_clean.isdigit():
                    date_clean = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:8]}"
                nav_list.append({'date': date_clean, 'unit_nav': unit_nav})
        if nav_list:
            formatted.append({
                'product_code': p['product_code'],
                'product_name': p['product_name'],
                'nav_history': nav_list,
            })
    return formatted


def import_to_db(products_data):
    from nav_db_excel import NAVDatabaseExcel

    if not products_data:
        logger.warning("无数据可导入")
        return

    db = NAVDatabaseExcel()

    stats_before = db.get_stats().get('浦银理财', {})
    logger.info(f"导入前: {stats_before.get('products', 0)} 产品, "
                f"{stats_before.get('dates', 0)} 日期, "
                f"范围 {stats_before.get('earliest_date')} ~ {stats_before.get('latest_date')}")

    formatted = format_nav_for_db(products_data)
    logger.info(f"准备导入 {len(formatted)} 个产品...")

    stats = db.update_nav('浦银', formatted)
    db.save()

    stats_after = db.get_stats().get('浦银理财', {})
    logger.info(f"导入后: {stats_after.get('products', 0)} 产品, "
                f"{stats_after.get('dates', 0)} 日期, "
                f"范围 {stats_after.get('earliest_date')} ~ {stats_after.get('latest_date')}")
    logger.info(f"新增 {stats.get('new_products', 0)} 产品, "
                f"新增 {len(stats.get('new_dates', []))} 日期, "
                f"更新 {stats.get('updated_cells', 0)} 单元格")


def main():
    parser = argparse.ArgumentParser(description='浦银理财2年净值数据抓取')
    parser.add_argument('--workers', type=int, default=6,
                        help='并行工作进程数 (默认: 6)')
    parser.add_argument('--threads', type=int, default=4,
                        help='每个进程的线程数 (默认: 4)')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='每批产品数量 (默认: 200)')
    parser.add_argument('--import-only', action='store_true',
                        help='仅导入已有检查点数据到净值数据库')
    parser.add_argument('--clean', action='store_true',
                        help='清理所有检查点文件后重新抓取')
    args = parser.parse_args()

    ensure_checkpoint_dir()

    if args.import_only:
        logger.info("=== 仅导入模式 ===")
        all_products, _ = load_all_checkpoints()
        logger.info(f"找到 {len(all_products)} 个已抓取产品")
        import_to_db(all_products)
        logger.info("=== 导入完成 ===")
        return

    if args.clean:
        import shutil
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            logger.info("已清理所有检查点")
        ensure_checkpoint_dir()

    total_concurrency = args.workers * args.threads
    logger.info(f"=== 浦银理财 730天(2年)净值抓取 ===")
    logger.info(f"配置: {args.workers}进程 × {args.threads}线程 = {total_concurrency}并发, "
                f"批次{args.batch_size}个")

    # 1. 获取产品列表
    from spdb_wm import SPDBWealthCrawler
    crawler = SPDBWealthCrawler()

    logger.info("获取个人理财产品列表...")
    products = crawler.get_product_list()
    if not products:
        logger.error("未获取到产品列表")
        return
    logger.info(f"共 {len(products)} 个产品")

    # 2. 对比数据库 + 检查点，跳过已有产品
    from nav_db_excel import NAVDatabaseExcel
    import pandas as pd

    # 2a. 从净值数据库读取已有产品代码
    db_codes = set()
    try:
        db = NAVDatabaseExcel()
        df = db.data.get('浦银理财')
        if df is not None and isinstance(df.index, pd.MultiIndex) and len(df) > 0:
            db_codes = set(str(code) for code, _ in df.index)
            logger.info(f"净值数据库已有: {len(db_codes)} 个产品 → 跳过")
        else:
            logger.info("净值数据库中暂无浦银理财数据")
    except Exception as e:
        logger.warning(f"读取净值数据库失败: {e}")

    # 2b. 从检查点读取已抓取的产品代码
    existing_products, checkpoint_codes = load_all_checkpoints()
    logger.info(f"已有检查点: {len(checkpoint_codes)} 个产品")

    # 合并跳过集合
    skip_codes = db_codes | checkpoint_codes
    remaining = [p for p in products if p.get('product_code', '') not in skip_codes]
    logger.info(f"跳过: {len(skip_codes)} (DB={len(db_codes)}, 检查点={len(checkpoint_codes)})")
    logger.info(f"待抓取: {len(remaining)} 个产品")

    if not remaining:
        logger.info("所有产品已抓取完毕，直接导入数据库")
        import_to_db(existing_products)
        logger.info("=== 完成 ===")
        return

    # 3. 分批
    batch_size = args.batch_size
    existing_batch_ids = set()
    pattern = os.path.join(CHECKPOINT_DIR, 'batch_*.json')
    for fp in glob_mod.glob(pattern):
        try:
            bid = int(os.path.basename(fp).replace('batch_', '').replace('.json', ''))
            existing_batch_ids.add(bid)
        except ValueError:
            pass
    next_batch_id = max(existing_batch_ids, default=-1) + 1

    batches = []
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        bid = next_batch_id + len(batches)
        batches.append((bid, batch))

    logger.info(f"分为 {len(batches)} 个批次, 批次ID: {batches[0][0]}~{batches[-1][0]}")

    # 4. 并行抓取
    result_queue = Queue()
    t0 = time.time()
    active_processes = []
    batch_idx = 0
    completed_batches = 0
    total_batches = len(batches)
    total_valid = 0

    while batch_idx < total_batches or active_processes:
        # 启动新进程
        while len(active_processes) < args.workers and batch_idx < total_batches:
            bid, batch = batches[batch_idx]
            worker_id = batch_idx % args.workers

            logger.info(f"启动批次{bid} ({len(batch)}个产品), "
                        f"{args.threads}线程, 工作进程{worker_id}")

            p = Process(
                target=crawl_worker,
                args=(worker_id, batch, bid, args.threads, result_queue),
                daemon=True,
            )
            p.start()
            active_processes.append((p, bid, time.time()))
            batch_idx += 1
            time.sleep(1)  # 错开启动

        # 等待结果
        if not active_processes:
            break

        try:
            result = result_queue.get(timeout=30)
            completed_batches += 1
            total_valid += result.get('valid', 0)
            elapsed_batch = result.get('elapsed', 0)
            total_elapsed = time.time() - t0

            logger.info(
                f"[{completed_batches}/{total_batches}] "
                f"批次{result['batch_id']} 完成: "
                f"{result['valid']}/{result['total']} 有效, "
                f"耗时 {elapsed_batch / 60:.1f}分钟, "
                f"总有效 {total_valid + len(done_codes)}/{len(products)}, "
                f"总耗时 {total_elapsed / 60:.1f}分钟"
            )

            # 清理已完成进程
            active_processes = [
                (p, bid, t) for p, bid, t in active_processes if p.is_alive()
            ]

        except Exception:
            # 检查死进程
            new_active = []
            for p, bid, t in active_processes:
                if p.is_alive():
                    new_active.append((p, bid, t))
                else:
                    # 检查是否已经通过queue报告过
                    pass
            active_processes = new_active

    total_elapsed = time.time() - t0
    logger.info(f"所有批次完成, 总耗时: {total_elapsed / 60:.1f}分钟")

    # 5. 汇总并导入数据库
    logger.info("汇总所有检查点数据...")
    all_products, all_codes = load_all_checkpoints()
    logger.info(f"共 {len(all_products)} 个有效产品")

    import_to_db(all_products)
    logger.info("=== 完成 ===")


if __name__ == '__main__':
    main()
