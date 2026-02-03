# -*- coding: utf-8 -*-
"""
光大理财爬虫 - 使用undetected-chromedriver绕过WAF

使用方法:
    python run_ceb_crawler.py --test     # 测试10个产品
    python run_ceb_crawler.py            # 完整爬取
    python run_ceb_crawler.py --list-only # 只获取产品列表(不获取历史净值)

工作原理:
    1. 启动浏览器访问列表页
    2. 点击产品链接获取详情页URL(通过弹窗获取)
    3. 关闭浏览器
    4. 启动新浏览器直接导航到详情页URL(绕过WAF)
    5. 提取净值数据
    6. 重复直到所有产品处理完毕
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class CEBCrawler:
    """光大理财爬虫 - 使用undetected-chromedriver"""

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
        """启动新浏览器实例"""
        try:
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = uc.Chrome(options=options, use_subprocess=True)
            self.main_window = self.driver.current_window_handle
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

    def connect(self) -> bool:
        """启动浏览器"""
        print("启动Chrome (undetected模式)...")
        return self._start_browser()

    def close(self):
        """关闭浏览器"""
        self._close_browser()

    def load_progress(self) -> bool:
        """加载已保存的进度"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                print(f"已加载进度: {len(self.progress.get('products', {}))} 个产品, 第{self.progress.get('current_page', 1)}页")
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
            size = len(self.driver.page_source)
            if size > min_size:
                return True
            time.sleep(1)
        return False

    def _navigate_js(self, url: str) -> bool:
        """使用JavaScript导航（绕过WAF检测）"""
        try:
            self.driver.execute_script(f"window.location.href = '{url}'")
            return self._wait_for_page()
        except Exception as e:
            print(f"导航失败: {e}")
            return False

    def get_page_info(self) -> Tuple[int, int]:
        """获取页面信息（总产品数、总页数）"""
        try:
            js_code = """
            var totalElem = document.querySelector('#totalpage1');
            var countElem = document.querySelector('.product_total_num') ||
                           document.querySelector('[id*="total"]');
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
                            asset_nav: cells[5] ? cells[5].innerText.trim() : '',
                            month_change: cells[6] ? cells[6].innerText.trim() : '',
                            year_change: cells[7] ? cells[7].innerText.trim() : '',
                            total_change: cells[8] ? cells[8].innerText.trim() : '',
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
        """通过点击获取产品详情页URL"""
        try:
            # 点击产品链接
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
                                return link;
                            }}
                        }}
                    }}
                }}
            }}
            return null;
            """
            link = self.driver.execute_script(js_click)
            if not link:
                return None

            windows_before = len(self.driver.window_handles)

            # 使用ActionChains点击
            actions = ActionChains(self.driver)
            actions.move_to_element(link)
            actions.click()
            actions.perform()

            time.sleep(3)

            # 获取新窗口的URL
            if len(self.driver.window_handles) > windows_before:
                new_window = [h for h in self.driver.window_handles if h != self.main_window][0]
                self.driver.switch_to.window(new_window)
                url = self.driver.current_url
                detail_url = url.split('?')[0]  # 去掉查询参数

                # 关闭弹窗
                self.driver.close()
                self.driver.switch_to.window(self.main_window)

                return detail_url

            return None

        except Exception as e:
            print(f"获取详情URL失败: {e}")
            # 清理可能的弹窗
            try:
                for h in self.driver.window_handles:
                    if h != self.main_window:
                        self.driver.switch_to.window(h)
                        self.driver.close()
                self.driver.switch_to.window(self.main_window)
            except:
                pass
            return None

    def extract_nav_from_detail(self) -> List[Dict]:
        """从详情页提取净值历史"""
        nav_data = []

        try:
            # 点击"产品净值"标签
            nav_tab = self.driver.find_element(By.CSS_SELECTOR, ".a2")
            nav_tab.click()
            time.sleep(3)

            # 切换到iframe
            iframe = self.driver.find_element(By.ID, "fundValueframe")
            self.driver.switch_to.frame(iframe)
            time.sleep(3)

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

            self.driver.switch_to.default_content()

        except Exception as e:
            print(f"    提取净值失败: {e}")
            try:
                self.driver.switch_to.default_content()
            except:
                pass

        return nav_data

    def crawl_product_nav(self, detail_url: str) -> List[Dict]:
        """使用新浏览器实例爬取单个产品的净值"""
        nav_data = []

        # 启动新浏览器
        if not self._start_browser():
            return nav_data

        try:
            # 先访问列表页（建立会话）
            self.driver.get(self.BASE_URL)
            if not self._wait_for_page():
                print("    列表页加载失败")
                return nav_data

            # 使用JS导航到详情页
            if not self._navigate_js(detail_url):
                print("    详情页加载失败")
                return nav_data

            # 提取净值
            nav_data = self.extract_nav_from_detail()

        finally:
            self._close_browser()

        return nav_data

    def crawl(self, max_products: int = None, start_page: int = 1,
              resume: bool = True, save_interval: int = 10) -> Dict:
        """主爬取流程"""

        # 加载已有进度
        if resume:
            self.load_progress()
            self.load_nav_history()

        # Phase 1: 收集所有产品URL（每页使用新浏览器）
        print("\n=== Phase 1: 收集产品URL ===")

        products_to_crawl = []
        page = start_page
        total_pages = 402  # 默认值

        while page <= total_pages:
            print(f"\n=== 第 {page}/{total_pages} 页 ===")

            # 每页启动新浏览器（避免WAF检测累积）
            if not self._start_browser():
                print("浏览器启动失败")
                break

            try:
                # 加载列表页
                self.driver.get(self.BASE_URL)
                if not self._wait_for_page(timeout=30):
                    print("列表页加载失败")
                    break

                # 第一页获取总页数
                if page == start_page:
                    total_products, total_pages = self.get_page_info()
                    print(f"共 {total_products} 个产品, {total_pages} 页")
                    self.progress['total_products'] = total_products
                    self.progress['total_pages'] = total_pages

                # 如果不是第一页，跳转到目标页
                if page > 1:
                    print(f"  跳转到第{page}页...")
                    self.driver.execute_script(f"goPage1({page})")
                    time.sleep(5)
                    # 验证页面跳转
                    current = self.driver.execute_script(
                        "return document.querySelector('#currentPage1') ? document.querySelector('#currentPage1').value : null"
                    )
                    if not current or int(current) != page:
                        print(f"  跳转失败，当前页: {current}，重试...")
                        self.driver.execute_script(f"goPage1({page})")
                        time.sleep(5)

                # 获取当前页产品
                page_products = self.get_products_on_page()
                print(f"提取到 {len(page_products)} 个产品")

                for product in page_products:
                    code = product['code']

                    # 检查是否已爬取
                    if code in self.progress['products'] and code in self.nav_history:
                        print(f"  {code}: 已有数据,跳过")
                        continue

                    # 获取详情URL
                    print(f"  {code}: 获取URL...")
                    detail_url = self.get_product_detail_url(code)

                    if detail_url:
                        print(f"    URL: {detail_url}")
                        products_to_crawl.append({
                            **product,
                            'detail_url': detail_url
                        })
                    else:
                        print(f"    获取URL失败")

                    # 检查限制
                    if max_products and len(products_to_crawl) >= max_products:
                        break

            finally:
                self._close_browser()

            if max_products and len(products_to_crawl) >= max_products:
                break

            page += 1

        print(f"\n收集到 {len(products_to_crawl)} 个待爬取产品")

        # Phase 2: 爬取每个产品的净值（使用新浏览器）
        print("\n=== Phase 2: 爬取净值数据 ===")

        for i, product in enumerate(products_to_crawl):
            code = product['code']
            name = product['name']
            detail_url = product['detail_url']

            print(f"\n[{i+1}/{len(products_to_crawl)}] {code}: {name}")
            print(f"    URL: {detail_url}")

            # 使用新浏览器爬取净值
            nav_data = self.crawl_product_nav(detail_url)

            if nav_data:
                print(f"    净值: {len(nav_data)} 条")

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
            else:
                print(f"    无净值数据")

            # 定期保存
            if (i + 1) % save_interval == 0:
                print(f"\n--- 保存进度 ({i+1}) ---")
                self.save_progress()
                self.save_nav_history()

        # 最终保存
        self.save_progress()
        self.save_nav_history()

        print(f"\n爬取完成: 共 {len(self.progress['products'])} 个产品")
        return self.progress['products']

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_products': len(self.progress['products']),
            'nav_history_count': len(self.nav_history),
            'current_page': self.progress.get('current_page', 1),
            'total_pages': self.progress.get('total_pages', 402),
            'last_update': self.progress.get('last_update')
        }

    def save_to_excel(self) -> Optional[str]:
        """导出数据到Excel（参考民生银行格式）"""
        if not self.progress['products'] and not self.nav_history:
            print("没有数据可导出")
            return None

        filename = os.path.join(
            self.data_dir,
            f"光大理财_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 全部产品（参考民生银行格式）
                products_data = []
                for code, info in self.progress['products'].items():
                    nav_hist = self.nav_history.get(code, {}).get('nav_history', [])

                    # 解析最新净值
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

                    # 添加历史净值列（用日期作为列名，最多15天）
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

                # 设置净值列的数字格式
                try:
                    for sheet_name in writer.sheets:
                        ws = writer.sheets[sheet_name]
                        for col_idx, col_cell in enumerate(ws[1], 1):
                            col_name = str(col_cell.value) if col_cell.value else ''
                            # 匹配净值列或日期格式列名
                            is_nav_col = ('净值' in col_name and '日期' not in col_name and '天数' not in col_name)
                            is_date_col = bool(re.match(r'^\d{4}-\d{2}-\d{2}$', col_name))
                            if is_nav_col or is_date_col:
                                for row_idx in range(2, ws.max_row + 1):
                                    cell = ws.cell(row=row_idx, column=col_idx)
                                    if cell.value is not None and isinstance(cell.value, (int, float)):
                                        cell.number_format = '0.0000'
                except Exception as e:
                    print(f"设置Excel格式警告: {e}")

            print(f"数据已导出到: {filename}")
            return filename

        except Exception as e:
            print(f"导出Excel失败: {e}")
            return None


if __name__ == "__main__":
    # 快速测试
    crawler = CEBCrawler()
    products = crawler.crawl(max_products=3)

    if products:
        print(f"\n成功爬取 {len(products)} 个产品")
        crawler.save_to_excel()
