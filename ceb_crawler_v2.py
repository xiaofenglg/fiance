# -*- coding: utf-8 -*-
"""
光大理财爬虫 v2 - 复用浏览器会话优化版

优化点:
1. 只启动一次浏览器，复用会话访问所有详情页
2. 减少不必要的等待时间
3. 断点续爬，跳过已爬取的产品
4. 预计速度提升 5-10 倍

使用方法:
    python ceb_crawler_v2.py --test     # 测试10个产品
    python ceb_crawler_v2.py            # 完整爬取
    python ceb_crawler_v2.py --resume   # 断点续爬
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import json
import re
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class CEBCrawlerV2:
    """光大理财爬虫 v2 - 复用浏览器会话"""

    BASE_URL = "https://www.cebwm.com/wealth/grlc/index.html"

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.driver = None
        self.main_window = None

        # 数据文件路径
        self.progress_file = os.path.join(data_dir, "ceb_progress.json")
        self.nav_file = os.path.join(data_dir, "ceb_nav_history.json")

        # 运行时数据
        self.progress = self._init_progress()
        self.nav_history = {}

    def _init_progress(self) -> Dict:
        """初始化进度结构"""
        return {
            "last_update": None,
            "current_page": 1,
            "total_pages": 402,
            "total_products": 0,
            "crawled_count": 0,
            "products": {}
        }

    def _start_browser(self) -> bool:
        """启动浏览器（只启动一次）"""
        if self.driver:
            return True

        try:
            print("启动Chrome (undetected模式)...")
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-gpu')
            self.driver = uc.Chrome(options=options, use_subprocess=True)
            # 设置超时
            self.driver.set_page_load_timeout(60)
            self.driver.set_script_timeout(30)
            self.main_window = self.driver.current_window_handle
            print("浏览器启动成功")
            return True
        except Exception as e:
            print(f"浏览器启动失败: {e}")
            return False

    def _close_browser(self):
        """关闭浏览器"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            self.main_window = None

    def load_progress(self) -> bool:
        """加载已保存的进度"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                print(f"已加载进度: {len(self.progress.get('products', {}))} 个产品")
                return True
        except Exception as e:
            print(f"加载进度失败: {e}")
        return False

    def save_progress(self):
        """保存当前进度"""
        try:
            self.progress['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.progress['crawled_count'] = len(self.progress['products'])
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存进度失败: {e}")

    def load_nav_history(self) -> bool:
        """加载净值历史数据"""
        try:
            if os.path.exists(self.nav_file):
                with open(self.nav_file, 'r', encoding='utf-8') as f:
                    self.nav_history = json.load(f)
                print(f"已加载净值历史: {len(self.nav_history)} 个产品")
                return True
        except Exception as e:
            print(f"加载净值历史失败: {e}")
        return False

    def save_nav_history(self):
        """保存净值历史数据"""
        try:
            with open(self.nav_file, 'w', encoding='utf-8') as f:
                json.dump(self.nav_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存净值历史失败: {e}")

    def _wait_for_page(self, min_size: int = 10000, timeout: int = 15) -> bool:
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

    def _navigate_js(self, url: str) -> bool:
        """使用JavaScript导航"""
        try:
            self.driver.execute_script(f"window.location.href = '{url}'")
            return self._wait_for_page()
        except Exception as e:
            print(f"导航失败: {e}")
            return False

    def _close_popup_windows(self):
        """关闭所有弹窗"""
        try:
            current = self.driver.current_window_handle
            for handle in self.driver.window_handles:
                if handle != self.main_window:
                    self.driver.switch_to.window(handle)
                    self.driver.close()
            self.driver.switch_to.window(self.main_window)
        except:
            pass

    def get_page_info(self) -> Tuple[int, int]:
        """获取页面信息"""
        try:
            js_code = """
            var totalElem = document.querySelector('#totalpage1');
            var countElem = document.querySelector('.product_total_num');
            return {
                totalPages: totalElem ? parseInt(totalElem.innerText) : 402,
                totalProducts: countElem ? parseInt(countElem.innerText) : 4019
            };
            """
            info = self.driver.execute_script(js_code)
            return info.get('totalProducts', 4019), info.get('totalPages', 402)
        except:
            return 4019, 402

    def get_products_on_page(self) -> List[Dict]:
        """从当前页面提取产品基本信息"""
        js_code = """
        var products = [];
        var tables = document.querySelectorAll('table');
        for (var t = 0; t < tables.length; t++) {
            var rows = tables[t].querySelectorAll('tbody tr, tr');
            for (var i = 0; i < rows.length; i++) {
                var cells = rows[i].querySelectorAll('td');
                if (cells.length >= 10) {
                    var code = cells[1] ? cells[1].innerText.trim() : '';
                    if (code && (code.startsWith('EB') || code.startsWith('EW'))) {
                        products.push({
                            name: cells[0] ? cells[0].innerText.trim() : '',
                            code: code,
                            min_amount: cells[2] ? cells[2].innerText.trim() : '',
                            unit_nav: cells[3] ? cells[3].innerText.trim() : '',
                            total_nav: cells[4] ? cells[4].innerText.trim() : '',
                            risk_level: cells[9] ? cells[9].innerText.trim() : ''
                        });
                    }
                }
            }
        }
        return products;
        """
        try:
            return self.driver.execute_script(js_code) or []
        except Exception as e:
            print(f"提取产品失败: {e}")
            return []

    def get_product_detail_url(self, product_code: str) -> Optional[str]:
        """通过点击获取产品详情页URL（优化版）"""
        try:
            # 使用 JS 直接点击，比 ActionChains 更快
            js_click = f"""
            var tables = document.querySelectorAll('table');
            for (var t = 0; t < tables.length; t++) {{
                var rows = tables[t].querySelectorAll('tbody tr, tr');
                for (var i = 0; i < rows.length; i++) {{
                    var cells = rows[i].querySelectorAll('td');
                    if (cells.length >= 2) {{
                        var code = cells[1] ? cells[1].innerText.trim() : '';
                        if (code === '{product_code}') {{
                            var link = cells[0].querySelector('a.lb_title');
                            if (link) {{
                                link.click();
                                return true;
                            }}
                        }}
                    }}
                }}
            }}
            return false;
            """
            clicked = self.driver.execute_script(js_click)
            if not clicked:
                return None

            # 等待新窗口打开（减少等待时间）
            for _ in range(15):  # 最多等待 1.5 秒
                if len(self.driver.window_handles) > 1:
                    break
                time.sleep(0.1)

            if len(self.driver.window_handles) > 1:
                new_window = [h for h in self.driver.window_handles if h != self.main_window][0]
                self.driver.switch_to.window(new_window)

                # 等待 URL 加载
                for _ in range(20):
                    url = self.driver.current_url
                    if 'lccp14' in url:
                        break
                    time.sleep(0.1)

                url = self.driver.current_url
                detail_url = url.split('?')[0]
                self.driver.close()
                self.driver.switch_to.window(self.main_window)
                return detail_url if 'lccp14' in detail_url else None

            return None

        except Exception as e:
            print(f"获取详情URL失败: {e}")
            self._close_popup_windows()
            return None

    def get_all_urls_on_page(self) -> List[Dict]:
        """批量获取当前页所有产品的 URL（更快的方式）"""
        products_with_urls = []
        page_products = self.get_products_on_page()

        for product in page_products:
            code = product['code']

            # 跳过已有数据的产品
            if code in self.progress['products']:
                continue

            detail_url = self.get_product_detail_url(code)
            if detail_url:
                products_with_urls.append({
                    **product,
                    'detail_url': detail_url
                })
                print(f"    {code}: OK")
            else:
                print(f"    {code}: 失败")

        return products_with_urls

    def extract_nav_from_detail(self) -> List[Dict]:
        """从详情页提取净值历史（与原版一致的方法）"""
        nav_data = []

        try:
            # 点击"产品净值"标签
            try:
                nav_tab = self.driver.find_element(By.CSS_SELECTOR, ".a2")
                nav_tab.click()
                time.sleep(3)  # 与原版一致
            except Exception as e:
                print(f"    点击标签失败: {e}")
                return nav_data

            # 切换到iframe
            try:
                iframe = self.driver.find_element(By.ID, "fundValueframe")
                self.driver.switch_to.frame(iframe)
                time.sleep(3)  # 与原版一致
            except Exception as e:
                print(f"    切换iframe失败: {e}")
                return nav_data

            # 从表格提取净值
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

        except Exception as e:
            print(f"    提取净值失败: {e}")
        finally:
            try:
                self.driver.switch_to.default_content()
            except:
                pass

        return nav_data

    def _check_session(self, timeout: int = 10) -> bool:
        """检查浏览器会话是否有效（带超时）"""
        import threading
        result = [False]

        def check():
            try:
                _ = self.driver.current_url
                result[0] = True
            except:
                result[0] = False

        thread = threading.Thread(target=check)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            print("    浏览器响应超时!")
            return False
        return result[0]

    def _restart_browser_if_needed(self) -> bool:
        """如果会话失效或超时则重启浏览器"""
        if not self._check_session(timeout=10):
            print("    会话失效或超时，重启浏览器...")
            self._close_browser()
            time.sleep(2)
            if not self._start_browser():
                return False
            # 重新建立会话
            try:
                self.driver.set_page_load_timeout(30)
                self.driver.get(self.BASE_URL)
                self._wait_for_page(timeout=20)
            except Exception as e:
                print(f"    建立会话失败: {e}")
                return False
        return True

    def crawl_single_product_nav(self, detail_url: str, retry: int = 2) -> List[Dict]:
        """在同一会话中爬取单个产品的净值（带超时保护）"""
        nav_data = []

        for attempt in range(retry):
            try:
                # 检查会话（带超时检测）
                if not self._restart_browser_if_needed():
                    print("    无法恢复浏览器会话")
                    return nav_data

                # 使用 JavaScript 导航（与原版一致）
                try:
                    self.driver.execute_script(f"window.location.href = '{detail_url}'")
                except Exception as e:
                    print(f"    JS导航失败: {e}")
                    if attempt < retry - 1:
                        self._close_browser()
                        time.sleep(2)
                        continue
                    return nav_data

                # 等待页面加载
                if not self._wait_for_page(min_size=10000, timeout=20):
                    if attempt < retry - 1:
                        print(f"    页面加载超时，重试...")
                        try:
                            self.driver.get(self.BASE_URL)
                            self._wait_for_page(timeout=15)
                        except:
                            self._close_browser()
                            time.sleep(2)
                        continue
                    return nav_data

                # 提取净值
                nav_data = self.extract_nav_from_detail()

                if nav_data:
                    return nav_data

            except Exception as e:
                error_msg = str(e)
                if 'timeout' in error_msg.lower() or 'session' in error_msg.lower():
                    print(f"    超时/会话错误，重启浏览器...")
                    self._close_browser()
                    time.sleep(2)
                elif attempt < retry - 1:
                    print(f"    异常，重试: {e}")
                    try:
                        self.driver.get(self.BASE_URL)
                        self._wait_for_page(timeout=10)
                    except:
                        self._close_browser()
                        time.sleep(2)
                else:
                    print(f"    爬取失败: {e}")

        return nav_data

    def collect_product_urls(self, max_pages: int = None, start_page: int = None) -> List[Dict]:
        """Phase 1: 收集所有产品的URL（支持断点续爬）"""
        products_to_crawl = []

        # 始终从第1页开始扫描所有EB产品
        already_collected = len(self.progress.get('products', {}))
        if start_page is None:
            start_page = 1  # 始终从第1页开始，确保不遗漏任何EB产品

        total_pages = self.progress.get('total_pages', 402)

        if max_pages:
            total_pages = min(total_pages, start_page + max_pages - 1)

        print(f"\n=== Phase 1: 收集产品URL ===")
        print(f"已收集 {already_collected} 个产品，从第 {start_page} 页继续")

        # 加载列表页
        self.driver.get(self.BASE_URL)
        if not self._wait_for_page(timeout=30):
            print("列表页加载失败")
            return products_to_crawl

        # 点击"开放式"筛选条件
        print("应用筛选条件: 开放式产品...")
        try:
            self.driver.execute_script("""
                var links = document.querySelectorAll('a');
                for (var i = 0; i < links.length; i++) {
                    if (links[i].innerText.trim() === '开放式') {
                        links[i].click();
                        return true;
                    }
                }
                return false;
            """)
            time.sleep(3)
            self._wait_for_page(timeout=10)
        except Exception as e:
            print(f"开放式筛选失败: {e}")

        # 点击"5万-50万"起购金额筛选
        print("应用筛选条件: 5万-50万起购金额...")
        try:
            self.driver.execute_script("""
                var links = document.querySelectorAll('a');
                for (var i = 0; i < links.length; i++) {
                    var text = links[i].innerText.trim();
                    if (text.indexOf('5万-50万') >= 0 || text === '5万-50万(含)') {
                        links[i].click();
                        return true;
                    }
                }
                return false;
            """)
            time.sleep(3)
            self._wait_for_page(timeout=10)
        except Exception as e:
            print(f"金额筛选失败: {e}")

        # 获取总页数
        total_products, total_pages_actual = self.get_page_info()
        print(f"共 {total_products} 个产品, {total_pages_actual} 页")
        self.progress['total_products'] = total_products
        self.progress['total_pages'] = total_pages_actual

        if max_pages:
            total_pages = min(total_pages_actual, start_page + max_pages - 1)

        page = start_page
        consecutive_empty = 0  # 连续空页计数
        attempted_codes = set()  # 记录已尝试过的产品代码
        last_page_codes = set()  # 上一页的产品代码，用于验证翻页

        while page <= total_pages:
            print(f"\n--- 第 {page}/{total_pages} 页 ---")

            # 跳转到目标页
            if page > 1:
                try:
                    self.driver.execute_script(f"goPage1({page})")
                    time.sleep(3)  # 增加等待时间
                    self._wait_for_page(min_size=5000, timeout=10)
                except Exception as e:
                    print(f"  翻页失败: {e}")
                    # 如果跳转失败，刷新页面重试
                    try:
                        self.driver.get(self.BASE_URL)
                        self._wait_for_page()
                        self.driver.execute_script(f"goPage1({page})")
                        time.sleep(3)
                    except Exception as e2:
                        print(f"  重试翻页也失败: {e2}")
                        break

            # 获取当前页产品
            page_products = self.get_products_on_page()
            current_codes = set(p['code'] for p in page_products)
            print(f"提取到 {len(page_products)} 个产品")

            # 验证翻页是否成功（如果和上一页完全相同，说明翻页失败）
            if page > start_page and current_codes == last_page_codes:
                print(f"  警告: 页面内容与上一页相同，翻页可能失败，重新加载...")
                # 重新加载页面并翻页
                try:
                    self.driver.get(self.BASE_URL)
                    self._wait_for_page(timeout=20)
                    time.sleep(2)
                    self.driver.execute_script(f"goPage1({page})")
                    time.sleep(4)
                    self._wait_for_page(min_size=5000, timeout=15)
                    page_products = self.get_products_on_page()
                    current_codes = set(p['code'] for p in page_products)
                    if current_codes == last_page_codes:
                        print(f"  翻页仍然失败，跳过此页")
                        page += 1
                        continue
                except Exception as e:
                    print(f"  重新翻页失败: {e}")
                    page += 1
                    continue

            last_page_codes = current_codes

            new_on_page = 0
            open_on_page = 0
            for product in page_products:
                code = product['code']

                # 开放式产品筛选后，所有产品都是开放式的
                open_on_page += 1

                # 跳过已尝试过的产品（无论成功失败）
                if code in attempted_codes or code in self.progress['products']:
                    continue

                attempted_codes.add(code)  # 标记为已尝试

                # 获取详情URL
                detail_url = self.get_product_detail_url(code)
                if detail_url:
                    # 立即保存到进度
                    self.progress['products'][code] = {
                        'name': product['name'],
                        'min_amount': product.get('min_amount'),
                        'unit_nav': product.get('unit_nav'),
                        'total_nav': product.get('total_nav'),
                        'risk_level': product.get('risk_level'),
                        'detail_url': detail_url,
                        'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    products_to_crawl.append({
                        **product,
                        'detail_url': detail_url
                    })
                    new_on_page += 1
                    print(f"    {code}: OK")
                else:
                    print(f"    {code}: 失败（无详情链接）")

            # 每页保存进度
            self.progress['current_page'] = page
            self.save_progress()
            print(f"  本页产品: {open_on_page}, 新增: {new_on_page}, 已尝试: {len(attempted_codes)}")

            # 如果连续多页没有新产品，说明可能已收集完
            if new_on_page == 0 and open_on_page == 0:
                consecutive_empty += 1
                if consecutive_empty >= 10:
                    print("连续10页无产品，停止收集")
                    break
            else:
                consecutive_empty = 0

            page += 1

            # 每10页检查一次浏览器状态
            if page % 10 == 0:
                if not self._check_session(timeout=5):
                    print("  浏览器会话失效，重启...")
                    self._close_browser()
                    time.sleep(2)
                    if not self._start_browser():
                        print("  重启浏览器失败，停止收集")
                        break
                    self.driver.get(self.BASE_URL)
                    self._wait_for_page(timeout=20)

        return products_to_crawl

    def crawl_navs(self, products: List[Dict], save_interval: int = 5) -> int:
        """Phase 2: 爬取所有产品的净值（复用浏览器会话）"""
        print(f"\n=== Phase 2: 爬取净值数据 ({len(products)} 个产品) ===")

        success_count = 0
        start_time = time.time()
        session_valid = False

        for i, product in enumerate(products):
            code = product['code']
            name = product['name']
            detail_url = product['detail_url']

            # 跳过已爬取的
            if code in self.nav_history:
                print(f"[{i+1}/{len(products)}] {code}: 已有数据，跳过")
                continue

            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1) if i > 0 else 0
            remaining = avg_time * (len(products) - i - 1)

            print(f"\n[{i+1}/{len(products)}] {code}: {name[:20]}...")
            print(f"    URL: {detail_url}")
            print(f"    预计剩余: {remaining/60:.1f} 分钟")

            # 每5个产品或会话失效时，重新访问列表页建立会话
            if not session_valid or i % 5 == 0:
                try:
                    print("    建立会话...")
                    self.driver.get(self.BASE_URL)
                    self._wait_for_page(timeout=15)
                    session_valid = True
                except:
                    session_valid = False

            # 爬取净值
            nav_data = self.crawl_single_product_nav(detail_url)

            if nav_data:
                print(f"    净值: {len(nav_data)} 条")
                session_valid = True  # 成功说明会话有效

                # 保存数据
                self.progress['products'][code] = {
                    'name': name,
                    'min_amount': product.get('min_amount'),
                    'unit_nav': product.get('unit_nav'),
                    'total_nav': product.get('total_nav'),
                    'risk_level': product.get('risk_level'),
                    'detail_url': detail_url,
                    'last_nav_date': nav_data[0]['date'] if nav_data else None,
                    'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                self.nav_history[code] = {
                    'name': name,
                    'nav_history': nav_data
                }
                success_count += 1
            else:
                print(f"    无净值数据")
                session_valid = False  # 失败可能是会话问题

            # 定期保存
            if (i + 1) % save_interval == 0:
                print(f"\n--- 保存进度 ({i+1}/{len(products)}) ---")
                self.save_progress()
                self.save_nav_history()

        return success_count

    def get_pending_products(self) -> List[Dict]:
        """获取已有 detail_url 但未爬取净值的产品"""
        pending = []
        for code, info in self.progress.get('products', {}).items():
            detail_url = info.get('detail_url')
            if detail_url and code not in self.nav_history:
                pending.append({
                    'code': code,
                    'name': info.get('name', ''),
                    'min_amount': info.get('min_amount'),
                    'unit_nav': info.get('unit_nav'),
                    'total_nav': info.get('total_nav'),
                    'risk_level': info.get('risk_level'),
                    'detail_url': detail_url
                })
        return pending

    def crawl_pending_only(self, max_products: int = None) -> int:
        """只爬取已有 URL 但未爬取净值的产品（最快模式）"""
        self.load_progress()
        self.load_nav_history()

        pending = self.get_pending_products()
        print(f"\n发现 {len(pending)} 个待爬取产品（已有URL）")

        if not pending:
            print("没有待爬取的产品")
            return 0

        if max_products:
            pending = pending[:max_products]

        if not self._start_browser():
            return 0

        try:
            # 先访问主页建立会话
            print("建立会话...")
            self.driver.get(self.BASE_URL)
            self._wait_for_page(timeout=20)

            success = self.crawl_navs(pending)
            print(f"\n爬取完成: 成功 {success}/{len(pending)}")
            return success
        finally:
            self.save_progress()
            self.save_nav_history()
            self._close_browser()

    def crawl(self, max_products: int = None, resume: bool = True,
              fast_mode: bool = False) -> Dict:
        """主爬取流程

        Args:
            max_products: 最大爬取数量
            resume: 是否断点续爬
            fast_mode: 快速模式，只爬取已有URL的产品
        """

        # 加载已有进度
        if resume:
            self.load_progress()
            self.load_nav_history()

        # 快速模式：只爬取已有 URL 但未爬取净值的产品
        if fast_mode:
            pending = self.get_pending_products()
            if pending:
                print(f"\n快速模式: 发现 {len(pending)} 个待爬取产品")
                if max_products:
                    pending = pending[:max_products]

                if not self._start_browser():
                    return self.progress['products']

                try:
                    # 建立会话
                    self.driver.get(self.BASE_URL)
                    self._wait_for_page(timeout=20)

                    success = self.crawl_navs(pending)
                    print(f"\n快速模式完成: 成功 {success}/{len(pending)}")
                finally:
                    self.save_progress()
                    self.save_nav_history()
                    self._close_browser()

                return self.progress['products']

        # 启动浏览器（只启动一次）
        if not self._start_browser():
            return {}

        try:
            # Phase 1: 收集产品URL
            # 计算需要收集的页数
            max_pages = None
            if max_products:
                max_pages = (max_products // 10) + 2  # 每页约10个产品

            products_to_crawl = self.collect_product_urls(max_pages=max_pages)

            if max_products:
                products_to_crawl = products_to_crawl[:max_products]

            print(f"\n收集到 {len(products_to_crawl)} 个待爬取产品")

            if not products_to_crawl:
                print("没有新产品需要爬取")
                return self.progress['products']

            # Phase 2: 爬取净值（复用会话）
            success = self.crawl_navs(products_to_crawl)

            print(f"\n爬取完成: 成功 {success}/{len(products_to_crawl)}")

        finally:
            # 最终保存
            self.save_progress()
            self.save_nav_history()
            # 关闭浏览器
            self._close_browser()

        return self.progress['products']

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_products': len(self.progress['products']),
            'nav_history_count': len(self.nav_history),
            'total_pages': self.progress.get('total_pages', 402),
            'last_update': self.progress.get('last_update')
        }

    def save_to_excel(self) -> Optional[str]:
        """导出数据到Excel"""
        if not self.progress['products'] and not self.nav_history:
            print("没有数据可导出")
            return None

        filename = os.path.join(
            self.data_dir,
            f"光大理财_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 全部产品
                products_data = []
                for code, info in self.progress['products'].items():
                    nav_hist = self.nav_history.get(code, {}).get('nav_history', [])

                    latest_nav = None
                    try:
                        nav_str = info.get('unit_nav', '')
                        if nav_str and nav_str != '--':
                            latest_nav = round(float(nav_str), 4)
                    except:
                        pass

                    row = {
                        '银行': '光大理财',
                        '产品名称': info.get('name'),
                        '产品代码': code,
                        '风险等级': info.get('risk_level'),
                        '起购金额': info.get('min_amount'),
                        '最新净值日期': info.get('last_nav_date'),
                        '净值天数': len(nav_hist),
                        '最新净值': latest_nav,
                        '爬取时间': info.get('crawled_at')
                    }

                    for i, nav in enumerate(nav_hist[:15]):
                        date_str = nav.get('date', '')
                        unit_nav = nav.get('unit_nav')
                        if date_str and unit_nav is not None:
                            row[date_str] = round(float(unit_nav), 4)

                    products_data.append(row)

                if products_data:
                    df_products = pd.DataFrame(products_data)
                    df_products.to_excel(writer, sheet_name='全部产品', index=False)

                # 净值历史明细表
                nav_data = []
                for code, info in self.nav_history.items():
                    name = info.get('name', '')
                    for nav in info.get('nav_history', []):
                        unit_nav = nav.get('unit_nav')
                        total_nav = nav.get('total_nav')
                        nav_data.append({
                            '产品代码': code,
                            '产品名称': name,
                            '日期': nav.get('date'),
                            '单位净值': round(float(unit_nav), 4) if unit_nav else None,
                            '累计净值': round(float(total_nav), 4) if total_nav else None
                        })

                if nav_data:
                    df_nav = pd.DataFrame(nav_data)
                    df_nav.to_excel(writer, sheet_name='净值历史', index=False)

            print(f"数据已导出到: {filename}")
            return filename

        except Exception as e:
            print(f"导出Excel失败: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='光大理财爬虫 v2 - 复用浏览器会话优化版')
    parser.add_argument('--test', action='store_true', help='测试模式（只爬10个产品）')
    parser.add_argument('--max', type=int, help='最大爬取产品数')
    parser.add_argument('--resume', action='store_true', default=True, help='断点续爬')
    parser.add_argument('--fast', action='store_true', help='快速模式：只爬取已有URL的产品（最快）')
    parser.add_argument('--pending', action='store_true', help='只显示待爬取产品数量')
    parser.add_argument('--export', action='store_true', help='只导出数据不爬取')
    args = parser.parse_args()

    crawler = CEBCrawlerV2()

    if args.export:
        crawler.load_progress()
        crawler.load_nav_history()
        crawler.save_to_excel()
        return

    if args.pending:
        crawler.load_progress()
        crawler.load_nav_history()
        pending = crawler.get_pending_products()
        print(f"已有 URL 但未爬取净值的产品: {len(pending)} 个")
        print(f"已爬取净值的产品: {len(crawler.nav_history)} 个")
        print(f"总产品数: {len(crawler.progress.get('products', {}))} 个")
        return

    max_products = None
    if args.test:
        max_products = 10
    elif args.max:
        max_products = args.max

    print("=" * 50)
    print("光大理财爬虫 v2 - 复用浏览器会话优化版")
    if args.fast:
        print("模式: 快速模式（只爬取已有URL的产品）")
    print("=" * 50)

    products = crawler.crawl(
        max_products=max_products,
        resume=args.resume,
        fast_mode=args.fast
    )

    if products:
        print(f"\n成功爬取 {len(products)} 个产品")
        crawler.save_to_excel()

    stats = crawler.get_stats()
    print(f"\n统计: {stats}")


if __name__ == "__main__":
    main()
