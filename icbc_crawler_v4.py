# -*- coding: utf-8 -*-
"""
工银理财爬虫 v4 - CDP网络拦截 + 单会话翻页

方案改进:
1. 使用Chrome DevTools Protocol直接捕获网络响应
2. 保持单一浏览器会话，通过页面内翻页
3. 使用等待机制确保API响应完成

使用方法:
    python icbc_crawler_v4.py --test        # 测试30个产品
    python icbc_crawler_v4.py               # 完整爬取
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote


class ICBCCrawlerV4:
    """工银理财爬虫 v4 - CDP网络拦截"""

    BASE_URL = "https://wm.icbc.com.cn"
    LIST_PAGE = "/netWorthDisclosure"
    DETAIL_PAGE = "/netWorthDisclosureDetails"

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.driver = None

        # 数据文件
        self.progress_file = os.path.join(data_dir, "icbc_progress.json")
        self.nav_file = os.path.join(data_dir, "icbc_nav_history.json")

        # 运行时数据
        self.products = {}
        self.nav_history = {}
        self.all_product_list = []  # 完整产品列表

    def _start_browser(self) -> bool:
        """启动浏览器"""
        try:
            driver_path = os.path.expanduser("~\\appdata\\roaming\\undetected_chromedriver\\undetected_chromedriver.exe")
            if os.path.exists(driver_path):
                try:
                    os.remove(driver_path)
                except:
                    pass

            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            self.driver = uc.Chrome(options=options, use_subprocess=True)
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

    def load_progress(self) -> bool:
        """加载进度"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.products = json.load(f)
                print(f"已加载 {len(self.products)} 个产品")
                return True
        except Exception as e:
            print(f"加载进度失败: {e}")
        return False

    def save_progress(self):
        """保存进度"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.products, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存进度失败: {e}")

    def load_nav_history(self):
        """加载净值历史"""
        try:
            if os.path.exists(self.nav_file):
                with open(self.nav_file, 'r', encoding='utf-8') as f:
                    self.nav_history = json.load(f)
                print(f"已加载 {len(self.nav_history)} 个产品的净值历史")
        except:
            pass

    def save_nav_history(self):
        """保存净值历史"""
        try:
            with open(self.nav_file, 'w', encoding='utf-8') as f:
                json.dump(self.nav_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存净值历史失败: {e}")

    def _inject_response_interceptor(self):
        """注入响应拦截器"""
        script = """
        (function() {
            window._capturedResponses = [];
            window._currentPage = 1;

            var origFetch = window.fetch;
            window.fetch = function() {
                return origFetch.apply(this, arguments).then(function(response) {
                    if (arguments[0] && arguments[0].toString().includes('112901')) {
                        response.clone().json().then(function(data) {
                            if (data && data.rows) {
                                window._capturedResponses.push({
                                    page: window._currentPage,
                                    total: data.total,
                                    rows: data.rows,
                                    timestamp: Date.now()
                                });
                            }
                        }).catch(function() {});
                    }
                    return response;
                });
            };

            var origSend = XMLHttpRequest.prototype.send;
            XMLHttpRequest.prototype.send = function(body) {
                var xhr = this;
                xhr.addEventListener('load', function() {
                    if (xhr._url && xhr._url.includes('112901')) {
                        try {
                            var data = JSON.parse(xhr.responseText);
                            if (data && data.rows) {
                                window._capturedResponses.push({
                                    page: window._currentPage,
                                    total: data.total,
                                    rows: data.rows,
                                    timestamp: Date.now()
                                });
                            }
                        } catch(e) {}
                    }
                });
                return origSend.apply(this, arguments);
            };

            var origOpen = XMLHttpRequest.prototype.open;
            XMLHttpRequest.prototype.open = function(method, url) {
                this._url = url;
                return origOpen.apply(this, arguments);
            };
        })();
        """
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': script})

    def _get_captured_responses(self) -> List[Dict]:
        """获取已捕获的API响应"""
        try:
            return self.driver.execute_script("return window._capturedResponses || []")
        except:
            return []

    def _clear_captured_responses(self):
        """清空已捕获的响应"""
        try:
            self.driver.execute_script("window._capturedResponses = []")
        except:
            pass

    def _wait_for_api_response(self, timeout: int = 15) -> bool:
        """等待API响应"""
        start = time.time()
        while time.time() - start < timeout:
            responses = self._get_captured_responses()
            if responses:
                return True
            time.sleep(0.5)
        return False

    def _click_next_page(self) -> bool:
        """点击下一页"""
        try:
            js_next = """
            var nextBtn = document.querySelector('.el-pagination .btn-next');
            if (nextBtn && !nextBtn.disabled) {
                window._currentPage = (window._currentPage || 1) + 1;
                nextBtn.click();
                return true;
            }
            return false;
            """
            return self.driver.execute_script(js_next)
        except:
            return False

    def _get_current_page_info(self) -> Dict:
        """获取当前页面分页信息"""
        try:
            js_info = """
            var result = {current: 1, total: 0, pageSize: 10};
            var pager = document.querySelector('.el-pagination');
            if (pager) {
                var current = pager.querySelector('.el-pager li.active');
                if (current) result.current = parseInt(current.innerText) || 1;

                var total = pager.querySelector('.el-pagination__total');
                if (total) {
                    var match = total.innerText.match(/共\\s*(\\d+)\\s*条/);
                    if (match) result.total = parseInt(match[1]);
                }
            }
            return result;
            """
            return self.driver.execute_script(js_info)
        except:
            return {'current': 1, 'total': 0, 'pageSize': 10}

    def _collect_all_products(self, max_products: int = None) -> List[Dict]:
        """收集所有产品列表"""
        all_products = []
        seen_ids = set()

        # 注入拦截器
        self._inject_response_interceptor()

        # 访问列表页
        print("  访问列表页...")
        self.driver.get(f"{self.BASE_URL}{self.LIST_PAGE}")
        time.sleep(8)

        # 获取分页信息
        page_info = self._get_current_page_info()
        total = page_info.get('total', 0)
        page_size = 10
        total_pages = (total + page_size - 1) // page_size if total else 150

        print(f"  总产品数: {total}, 总页数: {total_pages}")

        # 获取第一页
        if self._wait_for_api_response():
            responses = self._get_captured_responses()
            for resp in responses:
                for row in resp.get('rows', []):
                    nid = row.get('networth_id')
                    if nid and nid not in seen_ids:
                        seen_ids.add(nid)
                        all_products.append(row)

        print(f"  第1页: {len(all_products)} 个产品")

        if max_products and len(all_products) >= max_products:
            return all_products[:max_products]

        # 遍历剩余页
        page = 1
        consecutive_empty = 0

        while page < total_pages:
            if max_products and len(all_products) >= max_products:
                break

            # 清空捕获
            self._clear_captured_responses()

            # 点击下一页
            if not self._click_next_page():
                print(f"  无法点击下一页")
                break

            page += 1
            time.sleep(3)

            # 等待API响应
            if self._wait_for_api_response(timeout=10):
                responses = self._get_captured_responses()
                added = 0
                for resp in responses:
                    for row in resp.get('rows', []):
                        nid = row.get('networth_id')
                        if nid and nid not in seen_ids:
                            seen_ids.add(nid)
                            all_products.append(row)
                            added += 1

                if added > 0:
                    consecutive_empty = 0
                    if page % 10 == 0 or added > 0:
                        print(f"  第{page}页: +{added}, 累计: {len(all_products)}")
                else:
                    consecutive_empty += 1
                    print(f"  第{page}页: 无新产品 (连续{consecutive_empty}次)")
            else:
                consecutive_empty += 1
                print(f"  第{page}页: API无响应 (连续{consecutive_empty}次)")

            if consecutive_empty >= 5:
                print(f"  连续{consecutive_empty}页无数据，停止")
                break

        return all_products

    def _extract_nav_from_detail_page(self) -> List[Dict]:
        """从详情页提取净值数据"""
        nav_data = []

        try:
            time.sleep(3)

            js_get_table = """
            var result = [];
            var rows = document.querySelectorAll('.el-table__body tr');
            for (var i = 0; i < rows.length; i++) {
                var cells = rows[i].querySelectorAll('td');
                if (cells.length >= 4) {
                    result.push({
                        sales_code: cells[0].innerText.trim(),
                        date: cells[1].innerText.trim(),
                        ext1: cells[2].innerText.trim(),
                        ext2: cells[3].innerText.trim()
                    });
                }
            }
            return result;
            """
            rows = self.driver.execute_script(js_get_table)

            for row in rows:
                date_str = row.get('date', '').replace('-', '')
                ext1 = row.get('ext1', '')
                ext2 = row.get('ext2', '')
                sales_code = row.get('sales_code', '')

                if len(date_str) >= 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                try:
                    nav_data.append({
                        'date': date_str,
                        'unit_nav': float(ext1) if ext1 else None,
                        'total_nav': float(ext2) if ext2 else None,
                        'sales_code': sales_code
                    })
                except (ValueError, TypeError):
                    pass

        except Exception as e:
            print(f"    提取净值失败: {e}")

        return nav_data

    def crawl(self, max_products: int = None, resume: bool = True) -> Dict:
        """主爬取流程"""
        if resume:
            self.load_progress()
            self.load_nav_history()

        print("\n=== 启动工银理财爬虫 v4 ===")

        if not self._start_browser():
            return {}

        try:
            # 收集产品列表
            print("\n收集产品列表...")
            self.all_product_list = self._collect_all_products(max_products=max_products)

            if not self.all_product_list:
                print("获取产品列表失败")
                return {}

            print(f"\n共获取 {len(self.all_product_list)} 个产品")

            # 保存产品列表
            with open(os.path.join(self.data_dir, "icbc_product_list.json"), 'w', encoding='utf-8') as f:
                json.dump(self.all_product_list, f, ensure_ascii=False, indent=2)

            # 遍历每个产品获取净值
            print("\n获取净值数据...")
            total_crawled = 0

            for i, product in enumerate(self.all_product_list):
                product_code = product.get('product_code', '')
                product_name = product.get('product_name', '')
                networth_id = product.get('networth_id', '')
                product_type = product.get('product_type', '03')

                if not product_code or not networth_id:
                    continue

                # 检查是否已爬取
                if product_code in self.products and resume:
                    print(f"  [{i+1}/{len(self.all_product_list)}] {product_code}: 已存在,跳过")
                    continue

                print(f"  [{i+1}/{len(self.all_product_list)}] {product_code}: {product_name[:30]}...")

                # 访问详情页
                detail_url = f"{self.BASE_URL}{self.DETAIL_PAGE}?type={product_type}&id={networth_id}&title={quote(product_name[:50])}"
                self.driver.get(detail_url)

                # 提取净值数据
                nav_data = self._extract_nav_from_detail_page()
                if nav_data:
                    self.nav_history[product_code] = {
                        'name': product_name,
                        'nav_history': nav_data
                    }
                    print(f"    净值: {len(nav_data)} 条")
                else:
                    print(f"    无净值数据")

                # 保存产品信息
                self.products[product_code] = {
                    'name': product_name,
                    'networth_id': networth_id,
                    'product_type': product_type,
                    'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                total_crawled += 1

                # 定期保存
                if total_crawled % 20 == 0:
                    self.save_progress()
                    self.save_nav_history()
                    print(f"  -- 已保存进度: {total_crawled} 个产品 --")

        finally:
            self._close_browser()

        self.save_progress()
        self.save_nav_history()

        print(f"\n爬取完成: {len(self.products)} 个产品")
        return self.products

    def save_to_excel(self) -> Optional[str]:
        """导出到Excel"""
        if not self.products:
            print("没有数据可导出")
            return None

        filename = os.path.join(
            self.data_dir,
            f"工银理财_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                products_data = []
                for code, info in self.products.items():
                    nav_hist = self.nav_history.get(code, {}).get('nav_history', [])

                    seen_dates = set()
                    unique_nav = []
                    for nav in nav_hist:
                        date = nav.get('date', '')
                        if date and date not in seen_dates:
                            seen_dates.add(date)
                            unique_nav.append(nav)

                    row = {
                        '银行': '工银理财',
                        '产品名称': info.get('name'),
                        '产品代码': code,
                        '净值天数': len(unique_nav),
                        '爬取时间': info.get('crawled_at')
                    }

                    for i, nav in enumerate(unique_nav[:15]):
                        date_str = nav.get('date', '')
                        unit_nav = nav.get('unit_nav')
                        if date_str and unit_nav is not None:
                            row[date_str] = round(float(unit_nav), 4)

                    products_data.append(row)

                if products_data:
                    df = pd.DataFrame(products_data)
                    df.to_excel(writer, sheet_name='全部产品', index=False)

            print(f"数据已导出到: {filename}")
            return filename

        except Exception as e:
            print(f"导出失败: {e}")
            return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='工银理财爬虫 v4')
    parser.add_argument('--test', action='store_true', help='测试模式(30个产品)')
    parser.add_argument('--max', type=int, help='最大产品数')
    parser.add_argument('--no-resume', action='store_true', help='不继续上次进度')

    args = parser.parse_args()

    crawler = ICBCCrawlerV4()

    max_products = args.max
    if args.test:
        max_products = 30

    products = crawler.crawl(
        max_products=max_products,
        resume=not args.no_resume
    )

    if products:
        crawler.save_to_excel()


if __name__ == "__main__":
    main()
