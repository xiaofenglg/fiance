# -*- coding: utf-8 -*-
"""
中邮理财2年净值数据抓取

网站: https://www.psbc-wm.com/products/index.html
API:  /pswm-api/product/search (产品列表)
      /pswm-api/product/nvlist  (净值历史)

特点:
- WAF保护 (412), 需要 undetected_chromedriver 获取cookie
- 使用cookie后可用 requests 直接调用API
- 仅抓取个人理财产品 (sale_object_indi=1)
- 多线程并发抓取净值 + 分批保存数据库

用法:
  python crawl_psbc_2y.py               # 抓取全部个人理财产品
  python crawl_psbc_2y.py --test 20     # 测试模式, 仅抓取20个产品
  python crawl_psbc_2y.py --workers 8   # 指定并发线程数
"""

import sys
import os
import time
import json
import logging
import argparse
import threading
import requests
import ssl
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crawl_psbc_2y.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger('crawl_psbc_2y')

from nav_db_excel import NAVDatabaseExcel


# ============================================================================
# SSL适配器
# ============================================================================

class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器"""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


# ============================================================================
# WAF Cookie获取
# ============================================================================

def get_waf_cookies(max_wait: int = 40) -> Optional[Dict[str, str]]:
    """
    使用 undetected_chromedriver 绕过WAF获取cookie

    Returns:
        cookie字典, 失败返回None
    """
    try:
        import undetected_chromedriver as uc
    except ImportError:
        logger.error("需要安装 undetected_chromedriver: pip install undetected-chromedriver")
        return None

    logger.info("启动浏览器绕过WAF...")
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--window-position=-2400,0')  # 移到屏幕外

    driver = None
    try:
        driver = uc.Chrome(options=options, version_main=144)
        driver.set_page_load_timeout(120)

        # 访问首页触发WAF验证
        driver.get('https://www.psbc-wm.com/')

        # 等待WAF验证完成
        for i in range(max_wait // 2):
            time.sleep(2)
            src = driver.page_source
            if len(src) > 5000 and '$_ts' not in src[:1000]:
                logger.info(f"WAF验证通过 ({(i+1)*2}秒)")
                break
        else:
            logger.warning("WAF验证超时, 尝试继续...")

        # 获取cookies
        cookies = driver.get_cookies()
        cookie_dict = {c['name']: c['value'] for c in cookies}
        logger.info(f"获取到 {len(cookie_dict)} 个cookie")
        return cookie_dict

    except Exception as e:
        logger.error(f"获取WAF cookie失败: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


# ============================================================================
# 中邮理财爬虫
# ============================================================================

class PSBCWMCrawler:
    """中邮理财产品爬虫 - 2年净值抓取 (多线程版)"""

    BASE_URL = "https://www.psbc-wm.com"
    API_BASE = "https://www.psbc-wm.com/pswm-api"

    def __init__(self, cookies: Dict[str, str] = None, num_workers: int = 6):
        self._cookies = cookies
        self._num_workers = num_workers
        # 主session用于产品列表
        self.session = self._create_session(cookies)
        # 线程本地存储, 每个线程有自己的session
        self._local = threading.local()

    def _create_session(self, cookies: Dict[str, str] = None) -> requests.Session:
        """创建HTTP会话"""
        session = requests.Session()
        adapter = LegacySSLAdapter(pool_connections=30, pool_maxsize=30)
        session.mount('https://', adapter)

        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.psbc-wm.com/products/index.html',
            'Origin': 'https://www.psbc-wm.com',
        })
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if cookies:
            for name, value in cookies.items():
                session.cookies.set(name, value, domain='www.psbc-wm.com')

        return session

    def _get_thread_session(self) -> requests.Session:
        """获取当前线程的session (线程安全)"""
        if not hasattr(self._local, 'session'):
            self._local.session = self._create_session(self._cookies)
        return self._local.session

    def _request_with_retry(self, url: str, params: dict = None,
                            max_retries: int = 3,
                            session: requests.Session = None) -> Optional[dict]:
        """带重试的GET请求 (线程安全)"""
        sess = session or self._get_thread_session()
        for retry in range(max_retries):
            try:
                resp = sess.get(url, params=params, timeout=30)

                if resp.status_code == 412:
                    logger.warning(f"WAF拦截 (412), cookie可能已过期")
                    return None

                resp.raise_for_status()
                data = resp.json()

                if data.get('state') == 'ok':
                    return data
                else:
                    logger.warning(f"API返回异常: state={data.get('state')}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (重试 {retry+1}/{max_retries}): {url}")
                time.sleep(2 * (retry + 1))
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (重试 {retry+1}/{max_retries}): {e}")
                time.sleep(2 * (retry + 1))
            except json.JSONDecodeError:
                logger.warning(f"JSON解析失败: {url}")
                return None

        return None

    # ------------------------------------------------------------------
    # 产品列表
    # ------------------------------------------------------------------

    def get_product_list(self, stop_check=None) -> List[Dict]:
        """
        获取全部个人理财产品列表

        Args:
            stop_check: callable, 返回True表示应停止

        Returns:
            产品列表 [{wp_code, wp_name, prd_risk_rate, nav, ...}, ...]
        """
        logger.info("获取个人理财产品列表...")

        all_products = []
        page_num = 1
        page_size = 10  # API最大支持每页10个
        total_page = None

        while True:
            if stop_check and stop_check():
                logger.info(f"[中邮] 收到停止信号，已获取 {len(all_products)} 个产品")
                break
            params = {
                'sale_object_indi': '1',     # 个人理财
                'sale_object_inst': '',
                'wp_type': '',
                'prd_template_code': '',
                'base_days': '',
                'pageSize': page_size,
                'pageNum': page_num,
                'sort': 'wp_establish_date',
                'order': 'desc',
                'min_subs_amount_by_indi': '',
            }

            data = self._request_with_retry(f"{self.API_BASE}/product/search", params,
                                            session=self.session)
            if not data:
                if page_num == 1:
                    logger.error("首页请求失败, cookie可能已过期")
                    return []
                break

            page_data = data.get('data', {})
            products = page_data.get('list', [])

            if not products:
                break

            all_products.extend(products)

            if total_page is None:
                total_page = page_data.get('totalPage', 1)
                total_row = page_data.get('totalRow', 0)
                logger.info(f"共 {total_row} 个个人理财产品, {total_page} 页")

            logger.info(f"第 {page_num}/{total_page} 页, 本页 {len(products)} 个, "
                       f"累计 {len(all_products)} 个")

            if page_num >= total_page:
                break

            page_num += 1
            time.sleep(0.3)

        logger.info(f"产品列表获取完成: 共 {len(all_products)} 个个人理财产品")
        return all_products

    # ------------------------------------------------------------------
    # 净值历史
    # ------------------------------------------------------------------

    def get_nav_history(self, wp_code: str, max_pages: int = 200, stop_check=None) -> List[Dict]:
        """
        获取产品的全部净值历史 (最多2年, 线程安全)

        API: /pswm-api/product/nvlist
        返回格式: [{date, unit_nav}, ...]
        """
        all_nav = []
        page_num = 1
        page_size = 100
        total_page = None
        cutoff_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
        session = self._get_thread_session()

        while True:
            if stop_check and stop_check():
                break
            params = {
                'wp_code': wp_code,
                'pageSize': page_size,
                'pageNum': page_num,
            }

            data = self._request_with_retry(
                f"{self.API_BASE}/product/nvlist", params,
                max_retries=2, session=session
            )
            if not data:
                break

            page_data = data.get('data', {})
            nav_list = page_data.get('list', [])

            if not nav_list:
                break

            if total_page is None:
                total_page = page_data.get('totalPage', 1)

            reached_cutoff = False
            for item in nav_list:
                nav_record = self._parse_nav_record(item)
                if nav_record:
                    raw_date = nav_record['date'].replace('-', '')
                    if raw_date < cutoff_date:
                        reached_cutoff = True
                        break
                    all_nav.append(nav_record)

            if reached_cutoff:
                break

            if page_num >= total_page or page_num >= max_pages:
                break

            page_num += 1
            time.sleep(0.05)

        return all_nav

    def _parse_nav_record(self, item: dict) -> Optional[Dict]:
        """
        解析单条净值记录

        API实际返回字段:
        - update_date: 日期 (YYYYMMDD)
        - nav: 单位净值
        - accumulative_nav: 累计净值
        """
        # 日期字段: 优先 update_date
        date_val = None
        for key in ['update_date', 'nav_date', 'nv_date', 'date', 'valueDate']:
            if key in item and item[key]:
                date_val = str(item[key]).strip()
                break

        # 净值字段: 优先 nav
        nav_val = None
        for key in ['nav', 'unit_nav', 'nv', 'netValue']:
            if key in item and item[key] is not None:
                try:
                    nav_val = float(item[key])
                    break
                except (ValueError, TypeError):
                    continue

        if not date_val or nav_val is None:
            return None

        # 标准化日期
        date_str = date_val.replace('-', '').replace('/', '').replace('.', '')
        if len(date_str) >= 8 and date_str[:8].isdigit():
            formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            return {'date': formatted, 'unit_nav': nav_val}

        return None


# ============================================================================
# 主流程
# ============================================================================

def _fetch_one_product(crawler: PSBCWMCrawler, product: dict) -> Optional[dict]:
    """抓取单个产品的净值 (在线程池中运行)"""
    wp_code = product.get('wp_code', '')
    wp_name = product.get('wp_name', '')
    if not wp_code:
        return None

    nav_history = crawler.get_nav_history(wp_code)
    if nav_history:
        return {
            'product_code': wp_code,
            'product_name': wp_name,
            'nav_history': nav_history,
        }
    return None


def _save_batch_to_db(batch: List[dict], batch_num: int):
    """将一批产品净值保存到数据库"""
    if not batch:
        return
    logger.info(f"[批次{batch_num}] 保存 {len(batch)} 个产品到数据库...")
    db = NAVDatabaseExcel()
    stats = db.update_nav('中邮', batch)
    db.save()

    stats_after = db.get_stats().get('中邮理财', {})
    logger.info(
        f"[批次{batch_num}] 已保存. "
        f"数据库: {stats_after.get('products', 0)} 产品, "
        f"{stats_after.get('dates', 0)} 日期, "
        f"新增 {stats.get('new_products', 0)} 产品, "
        f"更新 {stats.get('updated_cells', 0)} 单元格"
    )


def main():
    parser = argparse.ArgumentParser(description='中邮理财2年净值数据抓取')
    parser.add_argument('--test', type=int, default=0,
                       help='测试模式: 仅抓取指定数量的产品')
    parser.add_argument('--workers', type=int, default=6,
                       help='并发线程数 (默认6)')
    parser.add_argument('--batch-size', type=int, default=200,
                       help='每批保存到数据库的产品数 (默认200)')
    args = parser.parse_args()

    max_products = args.test if args.test > 0 else None
    num_workers = args.workers
    batch_size = args.batch_size
    mode_str = f"测试模式 ({max_products}个产品)" if max_products else "完整模式"

    logger.info("=" * 70)
    logger.info(f"中邮理财 2年净值数据抓取 - {mode_str}")
    logger.info(f"并发线程: {num_workers}, 批次大小: {batch_size}")
    logger.info("=" * 70)

    # 1. 获取WAF cookie
    cookies = get_waf_cookies()
    if not cookies:
        logger.error("无法获取WAF cookie, 退出")
        return

    # 2. 创建爬虫
    crawler = PSBCWMCrawler(cookies, num_workers=num_workers)

    # 3. 验证API可用性
    logger.info("验证API连接...")
    test_data = crawler._request_with_retry(
        f"{crawler.API_BASE}/product/search",
        {'sale_object_indi': '1', 'pageSize': 1, 'pageNum': 1},
        session=crawler.session
    )
    if not test_data:
        logger.error("API验证失败, cookie可能无效")
        return
    total_products = test_data.get('data', {}).get('totalRow', 0)
    logger.info(f"API验证成功, 个人理财产品总数: {total_products}")

    # 4. 获取产品列表
    products = crawler.get_product_list()
    if not products:
        logger.error("未获取到产品列表")
        return

    # 过滤掉尚未成立的产品
    today_str = datetime.now().strftime('%Y%m%d')
    established = [p for p in products if p.get('wp_establish_date', '99999999') <= today_str]
    not_established = len(products) - len(established)
    if not_established > 0:
        logger.info(f"过滤未成立产品: {not_established} 个 (成立日期晚于今天)")
    products = established

    if max_products:
        products = products[:max_products]
        logger.info(f"测试模式: 仅处理前 {max_products} 个产品")

    # 5. 多线程并发抓取 + 分批保存
    total = len(products)
    logger.info(f"开始抓取 {total} 个产品净值, {num_workers} 线程并发...")

    t0 = time.time()
    success_count = 0
    empty_count = 0
    done_count = 0
    batch_buffer = []
    batch_num = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_fetch_one_product, crawler, p): p
            for p in products
        }

        for future in as_completed(futures):
            result = future.result()
            to_save = None

            with lock:
                done_count += 1
                if result:
                    batch_buffer.append(result)
                    success_count += 1
                else:
                    empty_count += 1

                # 进度日志 (每50个)
                if done_count % 50 == 0 or done_count == total:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (total - done_count) / rate if rate > 0 else 0
                    logger.info(
                        f"进度: {done_count}/{total} ({done_count/total*100:.1f}%), "
                        f"有效: {success_count}, 空: {empty_count}, "
                        f"速度: {rate:.1f}个/秒, "
                        f"ETA: {eta/60:.1f}分钟"
                    )

                # 分批保存
                if len(batch_buffer) >= batch_size:
                    batch_num += 1
                    to_save = (batch_buffer[:], batch_num)
                    batch_buffer.clear()

            # 保存在锁外执行, 避免阻塞其他线程
            if to_save:
                _save_batch_to_db(to_save[0], to_save[1])

    # 保存最后一批
    if batch_buffer:
        batch_num += 1
        _save_batch_to_db(batch_buffer, batch_num)

    elapsed_total = time.time() - t0
    logger.info(
        f"抓取完成: {success_count}/{total} 个产品有净值数据, "
        f"耗时 {elapsed_total/60:.1f}分钟, "
        f"平均 {total/elapsed_total:.1f}个/秒"
    )

    # 最终统计
    db = NAVDatabaseExcel()
    stats_final = db.get_stats().get('中邮理财', {})
    logger.info(
        f"最终数据库: {stats_final.get('products', 0)} 产品, "
        f"{stats_final.get('dates', 0)} 日期, "
        f"范围 {stats_final.get('earliest_date', 'N/A')} ~ "
        f"{stats_final.get('latest_date', 'N/A')}"
    )
    logger.info("=== 完成 ===")


if __name__ == '__main__':
    main()
