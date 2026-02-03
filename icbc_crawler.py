# -*- coding: utf-8 -*-
"""
工银理财爬虫 - 使用undetected-chromedriver

网站: https://wm.icbc.com.cn/netWorthDisclosure
API: /clt/info/112901 (产品列表)

使用方法:
    python icbc_crawler.py --test        # 测试10个产品
    python icbc_crawler.py               # 完整爬取
    python icbc_crawler.py --stats       # 查看统计
"""

import undetected_chromedriver as uc
import pandas as pd
import time
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional


class ICBCCrawler:
    """工银理财爬虫"""

    BASE_URL = "https://wm.icbc.com.cn"
    LIST_PAGE = "/netWorthDisclosure"

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.driver = None

        # 数据文件
        self.progress_file = os.path.join(data_dir, "icbc_progress.json")
        self.nav_file = os.path.join(data_dir, "icbc_nav_history.json")

        # 运行时数据
        self.products = {}
        self.nav_history = {}

        # API参数（会从页面获取）
        self.head_osnumber = None

    def _start_browser(self) -> bool:
        """启动浏览器"""
        try:
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

    def _inject_interceptor(self):
        """注入XHR拦截器来捕获osnumber"""
        intercept_script = """
        (function() {
            window._requestParams = [];
            var origSend = XMLHttpRequest.prototype.send;
            XMLHttpRequest.prototype.send = function(body) {
                if (body) {
                    try {
                        var parsed = JSON.parse(body);
                        if (parsed.head_osnumber) {
                            window._requestParams.push({
                                osnumber: parsed.head_osnumber,
                                time: new Date().toISOString()
                            });
                        }
                    } catch(e) {}
                }
                return origSend.apply(this, arguments);
            };
        })();
        """
        try:
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})
        except:
            pass

    def _get_osnumber(self) -> Optional[str]:
        """从页面XHR请求中获取osnumber"""
        try:
            params = self.driver.execute_script("return window._requestParams || []")
            for p in params:
                if p.get('osnumber'):
                    return p['osnumber']
        except:
            pass
        return None

    def get_product_list(self, page: int = 1, page_size: int = 50) -> Dict:
        """通过浏览器发送API请求获取产品列表"""
        osnumber = self.head_osnumber or ""
        js_fetch = f"""
        return await fetch('/clt/info/112901', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }},
            body: JSON.stringify({{
                "menu_id": "information_disclosure",
                "route": "/netWorthDisclosure",
                "full_route": "/netWorthDisclosure",
                "page_num": {page},
                "page_size": {page_size},
                "product_name": "",
                "head_system_id": "gtpoints",
                "head_channel_id": "0",
                "head_trans_code": "112901",
                "head_jsession_id": "",
                "batch_param": "",
                "head_osnumber": "{osnumber}"
            }})
        }}).then(r => r.json()).catch(e => ({{error: e.toString()}}));
        """
        try:
            result = self.driver.execute_script(js_fetch)
            return result if result else {}
        except Exception as e:
            print(f"API请求失败: {e}")
            return {}

    def get_product_detail(self, networth_id: str, product_type: str = "03", page: int = 1, page_size: int = 100) -> Dict:
        """获取产品详情（包含净值数据）

        注意：API需要正确的route参数才能返回数据
        - route: /netWorthDisclosureDetails
        - 需要networth_id和product_type
        """
        osnumber = self.head_osnumber or ""
        js_fetch = f"""
        return await fetch('/clt/info/112902', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }},
            body: JSON.stringify({{
                "menu_id": "information_disclosure",
                "route": "/netWorthDisclosureDetails",
                "full_route": "/netWorthDisclosureDetails?type={product_type}&id={networth_id}&title=nav",
                "networth_id": "{networth_id}",
                "product_type": "{product_type}",
                "page_num": {page},
                "page_size": {page_size},
                "head_system_id": "gtpoints",
                "head_channel_id": "0",
                "head_trans_code": "112902",
                "head_jsession_id": "",
                "batch_param": "",
                "head_osnumber": "{osnumber}"
            }})
        }}).then(r => r.json()).catch(e => ({{error: e.toString()}}));
        """
        try:
            result = self.driver.execute_script(js_fetch)
            return result if result else {}
        except Exception as e:
            print(f"获取详情失败: {e}")
            return {}

    def crawl(self, max_products: int = None, resume: bool = True) -> Dict:
        """主爬取流程"""
        if resume:
            self.load_progress()
            self.load_nav_history()

        print("\n=== 启动工银理财爬虫 ===")

        if not self._start_browser():
            return {}

        try:
            # 注入XHR拦截器（必须在页面加载前）
            self._inject_interceptor()

            # 访问页面建立session
            print("访问净值披露页...")
            self.driver.get(f"{self.BASE_URL}{self.LIST_PAGE}")
            if not self._wait_for_page():
                print("页面加载失败")
                return {}

            time.sleep(5)  # 等待XHR请求完成

            # 获取osnumber
            self.head_osnumber = self._get_osnumber()
            if self.head_osnumber:
                print(f"捕获 osnumber: {self.head_osnumber}")
            else:
                print("警告: 未捕获到osnumber，API可能无法正常工作")

            # 获取产品列表
            print("\n获取产品列表...")
            page = 1
            page_size = 50
            total_crawled = 0

            while True:
                print(f"\n第 {page} 页...")
                result = self.get_product_list(page, page_size)

                if not result or 'rows' not in result:
                    print(f"  获取失败: {result}")
                    break

                rows = result.get('rows', [])
                total = result.get('total', 0)
                print(f"  获取 {len(rows)} 个产品 (总计 {total})")

                if not rows:
                    break

                for item in rows:
                    networth_id = item.get('networth_id', '')
                    product_code = item.get('product_code', '')
                    product_name = item.get('product_name', '')

                    if not product_code:
                        continue

                    # 检查是否已爬取
                    if product_code in self.products and resume:
                        print(f"  {product_code}: 已存在,跳过")
                        continue

                    print(f"  {product_code}: {product_name[:30]}...")

                    # 获取详情
                    if networth_id:
                        product_type = item.get('product_type', '03')
                        detail = self.get_product_detail(networth_id, product_type)
                        if detail and detail.get('succ'):
                            # 提取净值数据
                            nav_data = self._extract_nav_from_detail(detail)
                            if nav_data:
                                self.nav_history[product_code] = {
                                    'name': product_name,
                                    'nav_history': nav_data
                                }
                                print(f"    净值数据: {len(nav_data)} 条")
                        elif detail.get('head_ret_msg'):
                            print(f"    获取详情失败: {detail.get('head_ret_msg')}")

                    # 保存产品信息
                    self.products[product_code] = {
                        'name': product_name,
                        'networth_id': networth_id,
                        'product_type': item.get('product_type', ''),
                        'show_date_time': item.get('show_date_time', ''),
                        'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    total_crawled += 1
                    if max_products and total_crawled >= max_products:
                        break

                    time.sleep(0.5)

                if max_products and total_crawled >= max_products:
                    break

                # 检查是否还有更多页
                if len(rows) < page_size:
                    break

                page += 1

                # 定期保存
                if page % 5 == 0:
                    self.save_progress()
                    self.save_nav_history()

        finally:
            self._close_browser()

        # 最终保存
        self.save_progress()
        self.save_nav_history()

        print(f"\n爬取完成: {len(self.products)} 个产品")
        return self.products

    def _extract_nav_from_detail(self, detail: Dict) -> List[Dict]:
        """从详情响应提取净值数据"""
        nav_data = []

        # 尝试多种可能的数据结构
        for key in ['rows', 'data', 'navList', 'netWorthList', 'list']:
            if key in detail:
                items = detail[key]
                if isinstance(items, list):
                    for item in items:
                        nav_record = self._parse_nav_record(item)
                        if nav_record:
                            nav_data.append(nav_record)

        return nav_data

    def _parse_nav_record(self, item: Dict) -> Optional[Dict]:
        """解析单条净值记录

        工银理财API返回格式:
        - date: 净值日期 (YYYYMMDD)
        - ext1: 每万份收益（元）
        - ext2: 七日年化收益率（%）
        - sales_code: 销售代码
        """
        if not isinstance(item, dict):
            return None

        # 工银理财特定字段
        date_val = item.get('date')
        ext1 = item.get('ext1')  # 每万份收益
        ext2 = item.get('ext2')  # 七日年化收益率
        sales_code = item.get('sales_code', '')

        # 如果有工银理财特定字段，使用它们
        if date_val and ext1:
            # 格式化日期 YYYYMMDD -> YYYY-MM-DD
            date_str = str(date_val)
            if len(date_str) >= 8:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            try:
                unit_nav = float(ext1)  # 每万份收益
                total_nav = float(ext2) if ext2 else None  # 七日年化收益率

                return {
                    'date': date_str,
                    'unit_nav': unit_nav,  # 每万份收益
                    'total_nav': total_nav,  # 七日年化收益率
                    'sales_code': sales_code
                }
            except (ValueError, TypeError):
                pass

        # 通用字段处理（备用）
        date_fields = ['date', 'navDate', 'valueDate', 'publishDate', 'nav_date']
        nav_fields = ['nav', 'unitNav', 'unitValue', 'netValue', 'unit_nav', 'ext1']
        total_fields = ['totalNav', 'accNav', 'cumulativeNav', 'total_nav', 'ext2']

        date_val = None
        nav_val = None
        total_val = None

        for f in date_fields:
            if f in item and item[f]:
                date_val = str(item[f])
                break

        for f in nav_fields:
            if f in item and item[f]:
                try:
                    nav_val = float(item[f])
                    break
                except:
                    pass

        for f in total_fields:
            if f in item and item[f]:
                try:
                    total_val = float(item[f])
                    break
                except:
                    pass

        if date_val and nav_val:
            # 格式化日期
            date_val = re.sub(r'[^\d]', '', date_val)
            if len(date_val) >= 8:
                date_val = f"{date_val[:4]}-{date_val[4:6]}-{date_val[6:8]}"
                return {
                    'date': date_val,
                    'unit_nav': nav_val,
                    'total_nav': total_val
                }

        return None

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
                # 产品列表
                products_data = []
                for code, info in self.products.items():
                    nav_hist = self.nav_history.get(code, {}).get('nav_history', [])

                    row = {
                        '银行': '工银理财',
                        '产品名称': info.get('name'),
                        '产品代码': code,
                        '净值天数': len(nav_hist),
                        '爬取时间': info.get('crawled_at')
                    }

                    # 添加历史净值
                    for i, nav in enumerate(nav_hist[:15]):
                        date_str = nav.get('date', '')
                        unit_nav = nav.get('unit_nav')
                        if date_str and unit_nav:
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
    parser = argparse.ArgumentParser(description='工银理财爬虫')
    parser.add_argument('--test', action='store_true', help='测试模式(10个产品)')
    parser.add_argument('--max', type=int, help='最大产品数')
    parser.add_argument('--stats', action='store_true', help='显示统计')
    parser.add_argument('--no-resume', action='store_true', help='不继续上次进度')

    args = parser.parse_args()

    crawler = ICBCCrawler()

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
