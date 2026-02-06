# -*- coding: utf-8 -*-
"""
宁银理财2年净值数据抓取 (多线程 + 断点续抓)

用法:
  python crawl_ningyin_2y.py              # 抓取全部个人理财产品 (断点续抓)
  python crawl_ningyin_2y.py --test       # 测试模式(仅抓取10个产品)
  python crawl_ningyin_2y.py --reset      # 清除断点，从头开始
  python crawl_ningyin_2y.py --workers 8  # 指定并发线程数

数据来源: https://www.wmbnb.com
API路径使用分号绕过WAF: /ningbo-web/;a/product/
"""

import sys
import os
import ssl
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# httpx 替代 requests (解决 wmbnb.com 的 SSL 兼容问题)
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s — %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('crawl_ningyin_2y')

# httpx 默认日志太多，关掉
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(PROJECT_DIR, 'ningyin_checkpoint.json')


# ============================================================================
# 宁银理财爬虫 (httpx 版, 线程安全)
# ============================================================================

class NingyinCrawler:
    """宁银理财净值爬虫

    关键技术点:
    1. 使用 httpx 而非 requests (urllib3 无法处理 wmbnb.com 的 legacy SSL)
    2. API 路径使用 ;a 分号段绕过 nginx WAF
       /ningbo-web/;a/product/xxx.json  →  Spring 正常路由到 /product/xxx
    3. 净值表接口 funddaytable.json 支持分页，需要逐页获取
    4. httpx.Client 是线程安全的，可在多线程中共享
    """

    BASE_URL = "https://www.wmbnb.com"
    API_PREFIX = "/ningbo-web/;a/product"

    REQUEST_DELAY = 0.3      # 请求间隔(秒)
    PAGE_SIZE = 100           # 产品列表每页数量
    NAV_PAGE_SIZE = 500       # 净值每页数量
    TIMEOUT = 30

    def __init__(self):
        self.client = self._create_client()

    def _create_client(self) -> httpx.Client:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT

        return httpx.Client(
            verify=ctx,
            timeout=self.TIMEOUT,
            headers={
                'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                               'AppleWebKit/537.36 (KHTML, like Gecko) '
                               'Chrome/120.0.0.0 Safari/537.36'),
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Referer': f'{self.BASE_URL}/product/index.html',
            },
            follow_redirects=True,
        )

    def _api_url(self, endpoint: str) -> str:
        return f"{self.BASE_URL}{self.API_PREFIX}/{endpoint}"

    # ------------------------------------------------------------------
    # 产品列表
    # ------------------------------------------------------------------

    def get_product_list(self, personal_only: bool = True,
                         max_products: int = 0, stop_check=None) -> List[Dict]:
        """获取产品列表

        Args:
            personal_only: True=只获取个人理财产品 (targetsale 01/03)
            max_products: >0 时最多获取 N 个产品 (用于测试)

        Returns:
            产品列表 [{projectcode, projectname, risklevel, ...}, ...]
        """
        logger.info("获取产品列表...")
        all_products = []
        seen_codes = set()

        # 个人理财: targetsale=01(纯个人) + targetsale=03(个人/机构)
        target_sales = ['01', '03'] if personal_only else ['']

        for ts in target_sales:
            page = 1
            retries = 0
            max_retries = 3
            while True:
                if stop_check and stop_check():
                    logger.info(f"[宁银] 收到停止信号，已获取 {len(all_products)} 个产品")
                    break
                if max_products > 0 and len(all_products) >= max_products:
                    break

                try:
                    params = {
                        'request_num': self.PAGE_SIZE,
                        'request_pageno': page,
                    }
                    if ts:
                        params['targetsale'] = ts

                    r = self.client.get(
                        self._api_url('list.json'),
                        params=params,
                    )
                    data = r.json()

                    if data.get('status') != 'success':
                        logger.warning(f"产品列表请求失败: {data.get('msg')}")
                        break

                    products = data.get('list', [])
                    if not products:
                        break

                    retries = 0  # 成功后重置重试计数

                    new_count = 0
                    for p in products:
                        code = p.get('projectcode', '')
                        if code and code not in seen_codes:
                            seen_codes.add(code)
                            all_products.append(p)
                            new_count += 1

                    total = data.get('total', 0)

                    if page % 10 == 0 or page == 1:
                        logger.info(
                            f"  targetsale={ts} 第{page}页: "
                            f"新增{new_count}, 累计{len(all_products)}/{total}"
                        )

                    actual_page_size = len(products)
                    if new_count == 0 or page * actual_page_size >= total:
                        break

                    page += 1
                    time.sleep(self.REQUEST_DELAY)

                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"  获取第{page}页失败 (已重试{max_retries}次): {e}")
                        break
                    wait = retries * 3
                    logger.warning(f"  获取第{page}页失败 (重试{retries}/{max_retries}, "
                                   f"等待{wait}s): {e}")
                    time.sleep(wait)

            if max_products > 0 and len(all_products) >= max_products:
                all_products = all_products[:max_products]
                break

        logger.info(f"共获取 {len(all_products)} 个产品")
        return all_products

    # ------------------------------------------------------------------
    # 净值历史
    # ------------------------------------------------------------------

    def get_nav_history(self, product_code: str, days: int = 730) -> List[Dict]:
        """获取产品净值历史 (线程安全)"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        all_nav = []
        page = 1

        while True:
            try:
                r = self.client.get(
                    self._api_url('funddaytable.json'),
                    params={
                        'code': product_code,
                        'startdate': start_date,
                        'enddate': end_date,
                        'request_num': self.NAV_PAGE_SIZE,
                        'request_pageno': page,
                    },
                )
                data = r.json()

                if data.get('status') != 'success':
                    break

                total_records = data.get('total', 0)
                items = data.get('list', [])

                if not items:
                    break

                for item in items:
                    cdate = item.get('cdate')
                    netvalue = item.get('netvalue')

                    if cdate is None or netvalue is None:
                        continue

                    try:
                        date_str = datetime.fromtimestamp(
                            cdate / 1000
                        ).strftime('%Y-%m-%d')
                    except (ValueError, OSError):
                        continue

                    try:
                        nav_float = float(netvalue)
                    except (ValueError, TypeError):
                        continue

                    all_nav.append({
                        'date': date_str,
                        'nav': nav_float,
                    })

                if page * self.NAV_PAGE_SIZE >= total_records:
                    break

                page += 1
                time.sleep(self.REQUEST_DELAY * 0.5)

            except Exception as e:
                logger.debug(f"  获取净值失败 {product_code} page={page}: {e}")
                break

        return all_nav

    def close(self):
        self.client.close()


# ============================================================================
# 断点管理
# ============================================================================

def load_checkpoint() -> Dict:
    """加载断点文件"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_checkpoint(data: Dict):
    """保存断点文件"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    """清除断点文件"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("已清除断点文件")


# ============================================================================
# 批量导入数据库
# ============================================================================

def import_batch_to_db(batch_data: List[Dict], batch_num: int):
    """将一批产品数据导入净值数据库 (SQLite)"""
    if not batch_data:
        return {}

    from nav_db_excel import update_nav_database

    logger.info(f"[批次{batch_num}] 导入 {len(batch_data)} 个产品到数据库...")

    # SQLite 数据库
    try:
        stats = update_nav_database('宁银', batch_data)
        logger.info(
            f"[批次{batch_num}] SQLite完成: "
            f"{stats.get('products_added', 0)} 新产品, "
            f"{stats.get('nav_rows_added', 0)} 净值记录"
        )
        return stats
    except Exception as e:
        logger.error(f"[批次{batch_num}] SQLite导入失败: {e}")
        return {}


# ============================================================================
# 多线程 NAV 抓取
# ============================================================================

def fetch_nav_for_product(crawler: NingyinCrawler, product: Dict,
                          delay_lock: threading.Lock) -> Optional[Dict]:
    """为单个产品抓取净值 (线程池任务)"""
    product_code = product.get('projectcode', '')
    product_name = (product.get('projectshortname', '')
                    or product.get('projectname', ''))

    # 简单限速: 在获取锁后 sleep 一小段
    with delay_lock:
        time.sleep(0.1)

    nav_history = crawler.get_nav_history(product_code, days=730)

    if nav_history:
        return {
            'product_code': product_code,
            'product_name': product_name,
            'nav_history': nav_history,
        }
    return None


# ============================================================================
# 主流程
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='宁银理财2年净值数据抓取')
    parser.add_argument('--test', action='store_true',
                        help='测试模式(仅抓取10个产品)')
    parser.add_argument('--reset', action='store_true',
                        help='清除断点，从头开始')
    parser.add_argument('--workers', type=int, default=4,
                        help='并发线程数 (默认4)')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='每批保存的产品数 (默认200)')
    args = parser.parse_args()

    if args.reset:
        clear_checkpoint()

    max_months = 24
    logger.info(f"=== 宁银理财 {max_months} 个月净值抓取 ===")
    logger.info(f"并发线程: {args.workers}, 批次大小: {args.batch_size}")

    # 加载断点
    checkpoint = load_checkpoint()
    done_codes: Set[str] = set(checkpoint.get('done_codes', []))
    cached_products = checkpoint.get('product_list', None)

    if done_codes:
        logger.info(f"从断点恢复: 已完成 {len(done_codes)} 个产品")

    crawler = NingyinCrawler()

    try:
        # 1. 获取产品列表 (优先用缓存)
        if cached_products and not args.test:
            logger.info(f"使用缓存的产品列表: {len(cached_products)} 个产品")
            products = cached_products
        else:
            max_products = 10 if args.test else 0
            products = crawler.get_product_list(
                personal_only=True,
                max_products=max_products,
            )
            if not products:
                logger.error("未获取到产品列表")
                return

        # 缓存产品列表到断点文件 (只保留必要字段)
        product_list_for_cache = [
            {
                'projectcode': p.get('projectcode', ''),
                'projectshortname': p.get('projectshortname', ''),
                'projectname': p.get('projectname', ''),
            }
            for p in products
        ]

        # 过滤掉已完成的产品
        remaining = [p for p in products
                     if p.get('projectcode', '') not in done_codes]

        total_all = len(products)
        total_remaining = len(remaining)

        logger.info(
            f"产品总数: {total_all}, "
            f"已完成: {total_all - total_remaining}, "
            f"待抓取: {total_remaining}"
        )

        if total_remaining == 0:
            logger.info("所有产品已抓取完毕，无需继续")
            clear_checkpoint()
            return

        # 2. 分批抓取净值
        batch_size = args.batch_size
        batch_num = 0
        total_products_with_nav = 0
        total_nav_records = 0
        delay_lock = threading.Lock()
        t0 = time.time()

        for batch_start in range(0, total_remaining, batch_size):
            batch_end = min(batch_start + batch_size, total_remaining)
            batch = remaining[batch_start:batch_end]
            batch_num += 1

            logger.info(
                f"[批次{batch_num}] 抓取产品 "
                f"{batch_start+1}-{batch_end}/{total_remaining} "
                f"(线程数={args.workers})"
            )

            batch_data = []
            batch_done_codes = []

            # 多线程抓取
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        fetch_nav_for_product, crawler, p, delay_lock
                    ): p
                    for p in batch
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    product = futures[future]
                    code = product.get('projectcode', '')
                    batch_done_codes.append(code)

                    try:
                        result = future.result()
                        if result:
                            batch_data.append(result)
                    except Exception as e:
                        logger.warning(f"  产品 {code} 抓取异常: {e}")

                    # 进度日志
                    if completed % 50 == 0 or completed == len(batch):
                        elapsed = time.time() - t0
                        global_done = batch_start + completed
                        rate = global_done / elapsed if elapsed > 0 else 0
                        eta = ((total_remaining - global_done) / rate
                               if rate > 0 else 0)
                        logger.info(
                            f"  进度: {global_done}/{total_remaining} "
                            f"({global_done/total_remaining*100:.1f}%), "
                            f"本批有效: {len(batch_data)}, "
                            f"ETA: {eta/60:.1f}分钟"
                        )

            # 批量导入数据库
            if batch_data:
                nav_count = sum(len(p['nav_history']) for p in batch_data)
                total_products_with_nav += len(batch_data)
                total_nav_records += nav_count
                logger.info(
                    f"[批次{batch_num}] "
                    f"{len(batch_data)} 个产品有净值, "
                    f"共 {nav_count} 条记录"
                )
                import_batch_to_db(batch_data, batch_num)

            # 更新断点
            done_codes.update(batch_done_codes)
            checkpoint_data = {
                'done_codes': list(done_codes),
                'product_list': product_list_for_cache,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_products': total_all,
                'total_done': len(done_codes),
            }
            save_checkpoint(checkpoint_data)
            logger.info(
                f"[批次{batch_num}] 断点已保存: "
                f"{len(done_codes)}/{total_all} 完成"
            )

        # 3. 完成
        elapsed_total = time.time() - t0
        logger.info(
            f"抓取完成: {total_products_with_nav} 个产品有净值数据, "
            f"共 {total_nav_records} 条净值记录, "
            f"耗时 {elapsed_total/60:.1f} 分钟"
        )

        # 全部完成后清除断点文件
        clear_checkpoint()

    finally:
        crawler.close()

    logger.info("=== 完成 ===")


if __name__ == '__main__':
    main()
