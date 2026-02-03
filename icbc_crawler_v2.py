# -*- coding: utf-8 -*-
"""
工银理财爬虫 v2 - 使用页面导航方式

方案：直接访问详情页获取净值数据，避免API session问题
1. 从列表页获取产品信息（通过页面XHR拦截）
2. 直接访问每个产品的详情页URL
3. 从详情页HTML提取净值表格数据

使用方法:
    python icbc_crawler_v2.py --test        # 测试10个产品
    python icbc_crawler_v2.py               # 完整爬取
    python icbc_crawler_v2.py --stats       # 查看统计
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote


class ICBCCrawlerV2:
    """工银理财爬虫 v2"""

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

    def _start_browser(self) -> bool:
        """启动浏览器"""
        try:
            # 清理旧的driver文件
            import shutil
            driver_path = os.path.expanduser("~\\appdata\\roaming\\undetected_chromedriver\\undetected_chromedriver.exe")
            if os.path.exists(driver_path):
                try:
                    os.remove(driver_path)
                except:
                    pass

            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
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

    def _wait_for_page(self, timeout: int = 20) -> bool:
        """等待页面加载"""
        for i in range(timeout):
            if len(self.driver.page_source) > 10000:
                return True
            time.sleep(1)
        return False

    def _get_products_from_list_page(self) -> List[Dict]:
        """从列表页获取产品信息（点击产品获取详情页URL）"""
        products = []

        # 获取产品项
        items = self.driver.find_elements(By.CSS_SELECTOR, '.product-item')
        print(f"  找到 {len(items)} 个产品项")

        for i, item in enumerate(items):
            try:
                # 获取产品标题
                title_el = item.find_element(By.CSS_SELECTOR, '.p-title')
                title = title_el.get_attribute('title') or title_el.text

                # 解析产品代码
                code_match = re.search(r'\(产品代码[：:]\s*(\w+)\)', title)
                product_code = code_match.group(1) if code_match else f"UNKNOWN_{i}"

                # 获取日期
                date_el = item.find_element(By.CSS_SELECTOR, '.time')
                date_text = date_el.text if date_el else ''

                products.append({
                    'index': i,
                    'product_code': product_code,
                    'product_name': title,
                    'date': date_text
                })
            except Exception as e:
                print(f"    解析产品 {i} 失败: {e}")

        return products

    def _click_product_and_get_detail_url(self, index: int) -> Optional[str]:
        """点击产品并获取详情页URL"""
        try:
            items = self.driver.find_elements(By.CSS_SELECTOR, '.product-item')
            if index >= len(items):
                return None

            # 记录当前窗口
            original_window = self.driver.current_window_handle

            # 使用ActionChains点击
            actions = ActionChains(self.driver)
            actions.move_to_element(items[index]).click().perform()

            time.sleep(2)

            # 检查是否有新窗口
            windows = self.driver.window_handles
            if len(windows) > 1:
                # 切换到新窗口
                self.driver.switch_to.window(windows[-1])
                detail_url = self.driver.current_url

                # 关闭详情窗口，返回列表窗口
                self.driver.close()
                self.driver.switch_to.window(original_window)

                return detail_url

            return None
        except Exception as e:
            print(f"    点击产品 {index} 失败: {e}")
            return None

    def _extract_nav_from_detail_page(self) -> List[Dict]:
        """从详情页提取净值数据"""
        nav_data = []

        try:
            # 等待页面加载
            time.sleep(3)

            # 查找表格行
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
                date_str = row.get('date', '')
                ext1 = row.get('ext1', '')
                ext2 = row.get('ext2', '')
                sales_code = row.get('sales_code', '')

                # 格式化日期
                date_str = date_str.replace('-', '')
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

    def _get_products_from_single_page(self) -> tuple:
        """从当前页面获取产品列表和总数"""
        products = []

        # 获取API响应
        responses = self.driver.execute_script("return window._apiResponses || []")

        if not responses:
            return products, 0

        first_resp = responses[0]
        total = first_resp.get('total', 0)
        rows = first_resp.get('rows', [])

        for row in rows:
            products.append({
                'networth_id': row.get('networth_id', ''),
                'product_code': row.get('product_code', ''),
                'product_name': row.get('product_name', ''),
                'product_type': row.get('product_type', '03'),
                'show_date_time': row.get('show_date_time', '')
            })

        return products, total

    def _get_product_list_via_api(self, max_products: int = None) -> List[Dict]:
        """通过多次新建浏览器会话获取完整产品列表（避免session过期）"""
        all_products = []
        seen_codes = set()

        # XHR拦截脚本
        intercept_script = """
        (function() {
            window._apiResponses = [];
            var origSend = XMLHttpRequest.prototype.send;
            var origOpen = XMLHttpRequest.prototype.open;

            XMLHttpRequest.prototype.open = function(method, url) {
                this._url = url;
                return origOpen.apply(this, arguments);
            };

            XMLHttpRequest.prototype.send = function(body) {
                var xhr = this;
                this.addEventListener('load', function() {
                    if (xhr._url && xhr._url.includes('/clt/info/112901')) {
                        try {
                            var resp = JSON.parse(xhr.responseText);
                            if (resp.rows) {
                                window._apiResponses.push(resp);
                            }
                        } catch(e) {}
                    }
                });
                return origSend.apply(this, arguments);
            };
        })();
        """

        # 第一页
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})
        self.driver.get(f"{self.BASE_URL}{self.LIST_PAGE}")
        time.sleep(8)

        products, total = self._get_products_from_single_page()
        print(f"  首页获取: {len(products)} 个产品, total={total}")

        for p in products:
            if p['product_code'] not in seen_codes:
                seen_codes.add(p['product_code'])
                all_products.append(p)

        if max_products and len(all_products) >= max_products:
            return all_products[:max_products]

        # 计算总页数（页面默认page_size=10）
        page_size = 10
        total_pages = (total + page_size - 1) // page_size

        # 遍历剩余页（每页新建浏览器会话）
        for page in range(2, total_pages + 1):
            if max_products and len(all_products) >= max_products:
                break

            print(f"  获取第 {page}/{total_pages} 页...")

            # 关闭并重新启动浏览器
            self._close_browser()
            if not self._start_browser():
                print(f"    浏览器启动失败")
                break

            # 重新注入拦截器
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})

            # 访问列表页
            self.driver.get(f"{self.BASE_URL}{self.LIST_PAGE}")
            time.sleep(5)

            # 清空已捕获的API响应
            self.driver.execute_script("window._apiResponses = []")

            # 使用页面点击翻页（通过el-pagination）
            js_goto_page = f"""
            // 找到分页组件并跳转
            var pager = document.querySelector('.el-pagination');
            if (pager) {{
                // 尝试使用跳转输入框
                var jumper = pager.querySelector('.el-pagination__jump');
                if (jumper) {{
                    var input = jumper.querySelector('input');
                    if (input) {{
                        input.focus();
                        input.value = '';
                        input.value = '{page}';
                        input.dispatchEvent(new Event('input', {{bubbles: true}}));
                        // 模拟回车
                        var event = new KeyboardEvent('keydown', {{
                            key: 'Enter',
                            code: 'Enter',
                            keyCode: 13,
                            which: 13,
                            bubbles: true
                        }});
                        input.dispatchEvent(event);
                        return 'jump_input';
                    }}
                }}
                // 或者点击下一页按钮
                var nextBtn = pager.querySelector('.btn-next');
                if (nextBtn && !nextBtn.disabled) {{
                    nextBtn.click();
                    return 'next_click';
                }}
            }}
            return 'not_found';
            """

            try:
                result = self.driver.execute_script(js_goto_page)

                # 等待新的API响应
                time.sleep(5)

                # 获取新页面的产品
                new_products, _ = self._get_products_from_single_page()
                added = 0
                for p in new_products:
                    if p['product_code'] not in seen_codes:
                        seen_codes.add(p['product_code'])
                        all_products.append(p)
                        added += 1

                print(f"    获取 {len(new_products)} 个产品, 新增: {added}, 累计: {len(all_products)}")

                # 如果没有新产品，可能到达末尾
                if added == 0 and len(new_products) == 0:
                    print(f"    没有新产品，停止")
                    break

            except Exception as e:
                print(f"    翻页失败: {e}")
                break

        return all_products

    def crawl(self, max_products: int = None, resume: bool = True) -> Dict:
        """主爬取流程"""
        if resume:
            self.load_progress()
            self.load_nav_history()

        print("\n=== 启动工银理财爬虫 v2 ===")

        if not self._start_browser():
            return {}

        try:
            # 获取产品列表
            print("\n获取产品列表...")
            products = self._get_product_list_via_api(max_products=max_products)

            if not products:
                print("获取产品列表失败")
                return {}

            print(f"共 {len(products)} 个产品")

            # 遍历每个产品
            total_crawled = 0
            for i, product in enumerate(products):
                product_code = product.get('product_code', '')
                product_name = product.get('product_name', '')
                networth_id = product.get('networth_id', '')
                product_type = product.get('product_type', '03')

                if not product_code:
                    continue

                # 检查是否已爬取
                if product_code in self.products and resume:
                    print(f"  [{i+1}/{len(products)}] {product_code}: 已存在,跳过")
                    continue

                print(f"  [{i+1}/{len(products)}] {product_code}: {product_name[:30]}...")

                # 构建详情页URL并访问
                detail_url = f"{self.BASE_URL}{self.DETAIL_PAGE}?type={product_type}&id={networth_id}&title={quote(product_name[:50])}"
                self.driver.get(detail_url)
                time.sleep(4)

                # 提取净值数据
                nav_data = self._extract_nav_from_detail_page()
                if nav_data:
                    self.nav_history[product_code] = {
                        'name': product_name,
                        'nav_history': nav_data
                    }
                    print(f"    净值数据: {len(nav_data)} 条")
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
                if max_products and total_crawled >= max_products:
                    break

                # 定期保存
                if total_crawled % 10 == 0:
                    self.save_progress()
                    self.save_nav_history()
                    print(f"  -- 已保存进度: {total_crawled} 个产品 --")

        finally:
            self._close_browser()

        # 最终保存
        self.save_progress()
        self.save_nav_history()

        print(f"\n爬取完成: {len(self.products)} 个产品")
        return self.products

    def save_to_excel(self) -> Optional[str]:
        """导出到Excel（民生银行格式）"""
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

                    # 按日期去重（取第一个销售代码的数据）
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

                    # 添加历史净值（最近15天）
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
    parser = argparse.ArgumentParser(description='工银理财爬虫 v2')
    parser.add_argument('--test', action='store_true', help='测试模式(10个产品)')
    parser.add_argument('--max', type=int, help='最大产品数')
    parser.add_argument('--stats', action='store_true', help='显示统计')
    parser.add_argument('--no-resume', action='store_true', help='不继续上次进度')

    args = parser.parse_args()

    crawler = ICBCCrawlerV2()

    if args.stats:
        crawler.load_progress()
        crawler.load_nav_history()
        print(f"产品数: {len(crawler.products)}")
        print(f"净值历史: {len(crawler.nav_history)}")
        return

    max_products = args.max
    if args.test:
        max_products = 10

    products = crawler.crawl(
        max_products=max_products,
        resume=not args.no_resume
    )

    if products:
        crawler.save_to_excel()


if __name__ == "__main__":
    main()
