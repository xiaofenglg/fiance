# -*- coding: utf-8 -*-
"""
工银理财爬虫 v3 - 单浏览器会话多次翻页

使用方法:
    python icbc_crawler_v3.py --test        # 测试30个产品
    python icbc_crawler_v3.py               # 完整爬取
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import pandas as pd
import time
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote


class ICBCCrawlerV3:
    """工银理财爬虫 v3 - 单会话翻页"""

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
        """主爬取流程 - 从列表页直接点击每个产品"""
        if resume:
            self.load_progress()
            self.load_nav_history()

        print("\n=== 启动工银理财爬虫 v3 ===")

        if not self._start_browser():
            return {}

        try:
            print("访问列表页...")
            self.driver.get(f"{self.BASE_URL}{self.LIST_PAGE}")
            time.sleep(8)

            total_crawled = 0
            page = 1
            consecutive_empty = 0

            while True:
                if max_products and total_crawled >= max_products:
                    break

                print(f"\n=== 第 {page} 页 ===")

                # 获取当前页的产品列表
                products = self._get_current_page_products()

                if not products:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        print("连续3页没有产品，停止")
                        break
                    page += 1
                    if not self._go_to_next_page():
                        break
                    continue

                consecutive_empty = 0
                print(f"找到 {len(products)} 个产品")

                # 处理每个产品
                for i, product in enumerate(products):
                    if max_products and total_crawled >= max_products:
                        break

                    product_code = product.get('product_code', '')
                    product_name = product.get('product_name', '')
                    networth_id = product.get('networth_id', '')
                    product_type = product.get('product_type', '03')

                    if not product_code or not networth_id:
                        continue

                    # 检查是否已爬取
                    if product_code in self.products and resume:
                        print(f"  [{i+1}/{len(products)}] {product_code}: 已存在,跳过")
                        continue

                    print(f"  [{i+1}/{len(products)}] {product_code}: {product_name[:30]}...")

                    # 直接访问详情页
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

                # 返回列表页并翻页
                print("  返回列表页...")
                self.driver.get(f"{self.BASE_URL}{self.LIST_PAGE}")
                time.sleep(5)

                # 跳转到下一页
                page += 1
                if page > 150:  # 安全限制
                    break

                if not self._go_to_page(page):
                    print(f"  无法跳转到第 {page} 页，停止")
                    break

        finally:
            self._close_browser()

        self.save_progress()
        self.save_nav_history()

        print(f"\n爬取完成: {len(self.products)} 个产品")
        return self.products

    def _get_current_page_products(self) -> List[Dict]:
        """获取当前页面的产品信息"""
        products = []

        # 等待API响应
        js_wait_api = """
        return new Promise((resolve) => {
            var checkInterval = setInterval(() => {
                var items = document.querySelectorAll('.product-item');
                if (items.length > 0) {
                    clearInterval(checkInterval);
                    resolve(items.length);
                }
            }, 500);
            setTimeout(() => {
                clearInterval(checkInterval);
                resolve(0);
            }, 10000);
        });
        """
        item_count = self.driver.execute_script(js_wait_api)
        if not item_count:
            return products

        # 解析产品信息
        js_get_products = """
        var items = document.querySelectorAll('.product-item');
        var result = [];
        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            var title = item.querySelector('.p-title');
            var titleText = title ? (title.getAttribute('title') || title.innerText) : '';

            // 解析产品代码
            var codeMatch = titleText.match(/产品代码[:：]\\s*(\\w+)/);
            var code = codeMatch ? codeMatch[1] : '';

            // 从onclick或href获取networth_id
            var onclick = item.getAttribute('onclick') || '';
            var href = item.querySelector('a') ? item.querySelector('a').href : '';
            var idMatch = (onclick + href).match(/id=([\\w]+)/);

            // 如果没找到，尝试从页面数据获取
            if (!idMatch && window.__NUXT__ && window.__NUXT__.data) {
                // Vue/Nuxt数据
            }

            result.push({
                index: i,
                product_code: code,
                product_name: titleText,
                networth_id: '',  // 需要从API或点击获取
                product_type: '03'
            });
        }
        return result;
        """
        items = self.driver.execute_script(js_get_products)

        # 获取API响应中的完整数据
        js_get_api_data = """
        // 尝试从Vue组件获取数据
        var app = document.querySelector('#app');
        if (app && app.__vue__) {
            var vm = app.__vue__;
            // 递归查找列表数据
            function findData(obj, depth) {
                if (depth > 5) return null;
                for (var key in obj) {
                    if (key === 'rows' || key === 'list' || key === 'productList') {
                        if (Array.isArray(obj[key]) && obj[key].length > 0 && obj[key][0].networth_id) {
                            return obj[key];
                        }
                    }
                    if (typeof obj[key] === 'object' && obj[key] !== null) {
                        var result = findData(obj[key], depth + 1);
                        if (result) return result;
                    }
                }
                return null;
            }
            var data = findData(vm.$data || vm, 0);
            if (data) return data;
        }
        return null;
        """
        api_data = self.driver.execute_script(js_get_api_data)

        if api_data:
            # 使用API数据
            for item in api_data:
                products.append({
                    'product_code': item.get('product_code', ''),
                    'product_name': item.get('product_name', ''),
                    'networth_id': item.get('networth_id', ''),
                    'product_type': item.get('product_type', '03')
                })
        else:
            # 点击每个产品获取详情URL
            for i, item in enumerate(items):
                if item.get('product_code'):
                    # 点击产品获取networth_id
                    js_click = f"""
                    var items = document.querySelectorAll('.product-item');
                    if (items[{i}]) {{
                        items[{i}].click();
                        return true;
                    }}
                    return false;
                    """
                    clicked = self.driver.execute_script(js_click)
                    if clicked:
                        time.sleep(2)
                        # 检查新窗口
                        windows = self.driver.window_handles
                        if len(windows) > 1:
                            self.driver.switch_to.window(windows[-1])
                            url = self.driver.current_url
                            # 解析URL获取参数
                            id_match = re.search(r'id=([^&]+)', url)
                            type_match = re.search(r'type=([^&]+)', url)
                            if id_match:
                                item['networth_id'] = id_match.group(1)
                            if type_match:
                                item['product_type'] = type_match.group(1)
                            # 关闭详情窗口
                            self.driver.close()
                            self.driver.switch_to.window(windows[0])
                            time.sleep(1)

                    products.append(item)

        return products

    def _go_to_next_page(self) -> bool:
        """点击下一页"""
        try:
            js_next = """
            var nextBtn = document.querySelector('.el-pagination .btn-next');
            if (nextBtn && !nextBtn.disabled) {
                nextBtn.click();
                return true;
            }
            return false;
            """
            result = self.driver.execute_script(js_next)
            if result:
                time.sleep(5)
                return True
        except:
            pass
        return False

    def _go_to_page(self, page: int) -> bool:
        """跳转到指定页码"""
        try:
            js_goto = f"""
            var jumper = document.querySelector('.el-pagination__jump input');
            if (jumper) {{
                jumper.value = '{page}';
                jumper.dispatchEvent(new Event('input', {{bubbles: true}}));
                var event = new KeyboardEvent('keydown', {{
                    key: 'Enter',
                    code: 'Enter',
                    keyCode: 13,
                    which: 13,
                    bubbles: true
                }});
                jumper.dispatchEvent(event);
                return true;
            }}
            // 尝试点击页码
            var pager = document.querySelector('.el-pager');
            if (pager) {{
                var items = pager.querySelectorAll('li');
                for (var i = 0; i < items.length; i++) {{
                    if (items[i].innerText === '{page}') {{
                        items[i].click();
                        return true;
                    }}
                }}
            }}
            return false;
            """
            result = self.driver.execute_script(js_goto)
            if result:
                time.sleep(5)
                return True
        except:
            pass
        return False

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
    parser = argparse.ArgumentParser(description='工银理财爬虫 v3')
    parser.add_argument('--test', action='store_true', help='测试模式(30个产品)')
    parser.add_argument('--max', type=int, help='最大产品数')
    parser.add_argument('--no-resume', action='store_true', help='不继续上次进度')

    args = parser.parse_args()

    crawler = ICBCCrawlerV3()

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
