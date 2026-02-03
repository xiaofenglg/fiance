# -*- coding: utf-8 -*-
"""
光大理财爬虫 - 多浏览器并行版

使用多个浏览器实例并行爬取，大幅提升速度
支持定期重启浏览器绕过反爬虫限速

使用方法:
    python ceb_crawler_parallel.py --workers 2 --fast              # 2个浏览器并行
    python ceb_crawler_parallel.py --workers 3 --fast --max 20     # 最多爬20个
    python ceb_crawler_parallel.py --fast --restart-every 5        # 每5个产品重启浏览器
    python ceb_crawler_parallel.py --export                        # 导出Excel

参数说明:
    --workers N       浏览器数量 (默认2)
    --fast            快速模式：爬取已有URL但未爬净值的产品
    --max N           最大爬取数量
    --restart-every N 每N个产品重启浏览器绕过反爬虫 (默认10)
    --export          导出Excel
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import pandas as pd
import time
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 线程锁
progress_lock = threading.Lock()
nav_lock = threading.Lock()


class CEBWorker:
    """单个爬虫工作器"""

    BASE_URL = "https://www.cebwm.com/wealth/grlc/index.html"

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.driver = None

    def start(self) -> bool:
        """启动浏览器"""
        try:
            print(f"[Worker {self.worker_id}] 启动浏览器...")
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-gpu')
            self.driver = uc.Chrome(options=options, use_subprocess=True)
            self.driver.set_page_load_timeout(60)
            self.driver.set_script_timeout(30)

            # 建立会话
            self.driver.get(self.BASE_URL)
            self._wait_for_page(timeout=30)
            print(f"[Worker {self.worker_id}] 浏览器就绪")
            return True
        except Exception as e:
            print(f"[Worker {self.worker_id}] 启动失败: {e}")
            return False

    def stop(self):
        """关闭浏览器"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None

    def _wait_for_page(self, min_size: int = 10000, timeout: int = 20) -> bool:
        """等待页面加载"""
        for i in range(timeout):
            try:
                size = len(self.driver.page_source)
                if size > min_size:
                    return True
            except:
                pass
            time.sleep(1)
        return False

    def _check_alive(self) -> bool:
        """检查浏览器是否响应"""
        try:
            _ = self.driver.current_url
            return True
        except:
            return False

    def restart(self) -> bool:
        """重启浏览器（绕过反爬虫限速）"""
        print(f"[Worker {self.worker_id}] 重启浏览器...")
        self.stop()
        time.sleep(3)
        return self.start()

    def crawl_product(self, code: str, name: str, detail_url: str) -> Optional[List[Dict]]:
        """爬取单个产品的净值"""
        try:
            if not self._check_alive():
                print(f"[Worker {self.worker_id}] 浏览器失效，重启...")
                self.stop()
                time.sleep(2)
                if not self.start():
                    return None

            # JS 导航到详情页
            self.driver.execute_script(f"window.location.href = '{detail_url}'")

            if not self._wait_for_page(min_size=10000, timeout=20):
                print(f"[Worker {self.worker_id}] {code}: 页面加载超时")
                return None

            # 点击产品净值标签
            try:
                nav_tab = self.driver.find_element(By.CSS_SELECTOR, ".a2")
                nav_tab.click()
                time.sleep(3)
            except:
                return None

            # 切换到 iframe
            try:
                iframe = self.driver.find_element(By.ID, "fundValueframe")
                self.driver.switch_to.frame(iframe)
                time.sleep(3)
            except:
                return None

            # 提取净值
            js_extract = """
            var result = [];
            var tables = document.querySelectorAll('table');
            for (var t = 0; t < tables.length; t++) {
                var rows = tables[t].querySelectorAll('tr');
                for (var i = 0; i < rows.length; i++) {
                    var cells = rows[i].querySelectorAll('td');
                    if (cells.length >= 2) {
                        var text0 = cells[0] ? cells[0].innerText.trim() : '';
                        var text1 = cells[1] ? cells[1].innerText.trim() : '';
                        var text2 = cells[2] ? cells[2].innerText.trim() : '';
                        if (/\\d{4}-\\d{2}-\\d{2}/.test(text0)) {
                            result.push({
                                date: text0,
                                unit_nav: parseFloat(text1) || null,
                                total_nav: parseFloat(text2) || null
                            });
                        }
                    }
                }
            }
            return result;
            """
            nav_data = self.driver.execute_script(js_extract) or []

            self.driver.switch_to.default_content()

            # 返回列表页准备下一个
            self.driver.get(self.BASE_URL)
            self._wait_for_page(timeout=15)

            return nav_data if nav_data else None

        except Exception as e:
            print(f"[Worker {self.worker_id}] {code}: 错误 - {e}")
            try:
                self.driver.switch_to.default_content()
            except:
                pass
            return None


class CEBParallelCrawler:
    """并行爬虫管理器"""

    def __init__(self, data_dir: str = ".", num_workers: int = 2):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.workers: List[CEBWorker] = []

        # 数据文件
        self.progress_file = os.path.join(data_dir, "ceb_progress.json")
        self.nav_file = os.path.join(data_dir, "ceb_nav_history.json")

        # 数据
        self.progress = {"products": {}}
        self.nav_history = {}

    def load_data(self):
        """加载已有数据"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                print(f"已加载 {len(self.progress.get('products', {}))} 个产品")
        except:
            pass

        try:
            if os.path.exists(self.nav_file):
                with open(self.nav_file, 'r', encoding='utf-8') as f:
                    self.nav_history = json.load(f)
                print(f"已加载 {len(self.nav_history)} 个产品的净值")
        except:
            pass

    def save_data(self):
        """保存数据"""
        with progress_lock:
            self.progress['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)

        with nav_lock:
            with open(self.nav_file, 'w', encoding='utf-8') as f:
                json.dump(self.nav_history, f, ensure_ascii=False, indent=2)

    def get_pending_products(self) -> List[Dict]:
        """获取待爬取的产品"""
        pending = []
        for code, info in self.progress.get('products', {}).items():
            if info.get('detail_url') and code not in self.nav_history:
                pending.append({
                    'code': code,
                    'name': info.get('name', ''),
                    'detail_url': info.get('detail_url')
                })
        return pending

    def start_workers(self) -> bool:
        """启动所有工作器"""
        print(f"\n启动 {self.num_workers} 个浏览器...")
        self.workers = []

        for i in range(self.num_workers):
            worker = CEBWorker(i + 1)
            if worker.start():
                self.workers.append(worker)
            else:
                print(f"Worker {i + 1} 启动失败")

            # 错开启动时间
            if i < self.num_workers - 1:
                time.sleep(3)

        if not self.workers:
            print("没有可用的工作器!")
            return False

        print(f"成功启动 {len(self.workers)} 个浏览器")
        return True

    def stop_workers(self):
        """停止所有工作器"""
        for worker in self.workers:
            worker.stop()
        self.workers = []

    def crawl_with_worker(self, worker: CEBWorker, product: Dict) -> tuple:
        """使用指定工作器爬取产品"""
        code = product['code']
        name = product['name']
        detail_url = product['detail_url']

        nav_data = worker.crawl_product(code, name, detail_url)

        return code, name, detail_url, nav_data

    def crawl_parallel(self, products: List[Dict], restart_every: int = 10) -> int:
        """并行爬取多个产品，定期重启浏览器绕过反爬虫"""
        success_count = 0
        total = len(products)
        processed = 0

        print(f"\n=== 开始并行爬取 {total} 个产品 (使用 {len(self.workers)} 个浏览器) ===")
        print(f"    每 {restart_every} 个产品重启浏览器绕过反爬虫\n")

        # 分批处理，每批 restart_every 个产品
        batch_size = restart_every
        batches = [products[i:i + batch_size] for i in range(0, len(products), batch_size)]

        for batch_idx, batch in enumerate(batches):
            print(f"\n--- 批次 {batch_idx + 1}/{len(batches)} ({len(batch)} 个产品) ---")
            batch_success = 0

            # 真正的并行处理：每个worker分配一部分产品
            # 将批次内的产品按worker数量分组
            worker_tasks = [[] for _ in range(len(self.workers))]
            for i, product in enumerate(batch):
                worker_tasks[i % len(self.workers)].append(product)

            # 使用线程池并行执行每个worker的任务
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                futures = []
                for worker_idx, tasks in enumerate(worker_tasks):
                    if tasks:
                        future = executor.submit(
                            self._worker_crawl_batch,
                            self.workers[worker_idx],
                            tasks
                        )
                        futures.append(future)

                # 等待所有worker完成
                for future in as_completed(futures):
                    try:
                        results = future.result(timeout=300)  # 5分钟超时
                        for code, name, nav_data in results:
                            processed += 1
                            if nav_data:
                                print(f"[OK] [{processed}/{total}] {code}: {len(nav_data)} 条净值")
                                success_count += 1
                                batch_success += 1

                                with progress_lock:
                                    if code in self.progress['products']:
                                        self.progress['products'][code]['last_nav_date'] = nav_data[0]['date']
                                        self.progress['products'][code]['crawled_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                                with nav_lock:
                                    self.nav_history[code] = {
                                        'name': name,
                                        'nav_history': nav_data
                                    }
                            else:
                                print(f"[--] [{processed}/{total}] {code}: 无净值数据")
                    except Exception as e:
                        print(f"[!!] 批次执行错误: {e}")

            # 批次结束后保存数据
            self.save_data()
            print(f"    批次完成: {batch_success}/{len(batch)} 成功")

            # 如果不是最后一批，重启所有浏览器
            if batch_idx < len(batches) - 1:
                print(f"\n>>> 重启所有浏览器绕过反爬虫...")
                for worker in self.workers:
                    worker.restart()
                print(f">>> 浏览器重启完成，继续爬取...\n")

        return success_count

    def _worker_crawl_batch(self, worker: CEBWorker, products: List[Dict]) -> List[tuple]:
        """单个worker串行爬取分配给它的产品"""
        results = []
        for product in products:
            code = product['code']
            name = product['name']
            detail_url = product['detail_url']

            try:
                nav_data = worker.crawl_product(code, name, detail_url)
                results.append((code, name, nav_data))
            except Exception as e:
                print(f"[Worker {worker.worker_id}] {code}: 错误 - {e}")
                results.append((code, name, None))

        return results

    def crawl_fast(self, max_products: int = None, restart_every: int = 10):
        """快速模式：只爬取已有URL的产品"""
        self.load_data()

        pending = self.get_pending_products()
        print(f"待爬取产品: {len(pending)} 个")

        if not pending:
            print("没有待爬取的产品")
            return

        if max_products:
            pending = pending[:max_products]

        if not self.start_workers():
            return

        try:
            success = self.crawl_parallel(pending, restart_every=restart_every)
            print(f"\n完成: 成功 {success}/{len(pending)}")
        finally:
            self.save_data()
            self.stop_workers()

    def save_to_excel(self):
        """导出到Excel"""
        self.load_data()

        if not self.progress.get('products'):
            print("没有数据可导出")
            return

        filename = os.path.join(
            self.data_dir,
            f"光大理财_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        products_data = []
        for code, info in self.progress['products'].items():
            nav_hist = self.nav_history.get(code, {}).get('nav_history', [])
            row = {
                '产品代码': code,
                '产品名称': info.get('name'),
                '风险等级': info.get('risk_level'),
                '净值条数': len(nav_hist),
                '最新净值日期': info.get('last_nav_date'),
            }
            products_data.append(row)

        df = pd.DataFrame(products_data)
        df.to_excel(filename, index=False)
        print(f"导出到: {filename}")


def main():
    parser = argparse.ArgumentParser(description='光大理财爬虫 - 并行版')
    parser.add_argument('--workers', type=int, default=2, help='浏览器数量 (默认2)')
    parser.add_argument('--fast', action='store_true', help='快速模式：只爬取已有URL的产品')
    parser.add_argument('--max', type=int, help='最大爬取数量')
    parser.add_argument('--restart-every', type=int, default=10, help='每N个产品重启浏览器 (默认10)')
    parser.add_argument('--export', action='store_true', help='导出Excel')
    args = parser.parse_args()

    crawler = CEBParallelCrawler(num_workers=args.workers)

    if args.export:
        crawler.save_to_excel()
        return

    print("=" * 50)
    print(f"光大理财爬虫 - 并行版 ({args.workers} 个浏览器)")
    print(f"每 {args.restart_every} 个产品重启浏览器")
    print("=" * 50)

    if args.fast:
        crawler.crawl_fast(max_products=args.max, restart_every=args.restart_every)
    else:
        print("请使用 --fast 模式爬取已有URL的产品")


if __name__ == "__main__":
    main()
