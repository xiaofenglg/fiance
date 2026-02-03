# -*- coding: utf-8 -*-
"""
光大理财产品历史净值爬虫

通过CDP连接Chrome，获取产品列表和历史净值数据
需要先以调试模式启动Chrome: chrome.exe --remote-debugging-port=9222
然后在浏览器中打开: https://www.cebwm.com/wealth/grlc/index.html

日期：2026-01-17
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import json
import os
from datetime import datetime


class CEBWMNavCrawler:
    """光大理财历史净值爬虫"""

    def __init__(self, debug_port=9222):
        self.debug_port = debug_port
        self.driver = None
        self.products = []
        self.nav_history = {}  # {产品代码: [{date, unit_nav, total_nav}, ...]}

    def connect(self):
        """连接到Chrome浏览器"""
        print("连接到Chrome浏览器...")
        try:
            options = Options()
            options.add_experimental_option('debuggerAddress', f'127.0.0.1:{self.debug_port}')
            self.driver = webdriver.Chrome(options=options)

            # 切换到光大理财页面
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                if 'cebwm' in self.driver.current_url:
                    print(f"已连接: {self.driver.title}")
                    return True

            # 如果没有找到，导航到目标页面
            print("未找到光大理财页面，正在导航...")
            self.driver.get("https://www.cebwm.com/wealth/grlc/index.html")
            time.sleep(5)

            if 'cebwm' in self.driver.current_url:
                print(f"已连接: {self.driver.title}")
                return True

            print("无法访问光大理财页面")
            return False

        except Exception as e:
            print(f"连接失败: {e}")
            print("\n请确保:")
            print("1. Chrome已以调试模式启动:")
            print('   chrome.exe --remote-debugging-port=9222')
            print("2. 在浏览器中打开:")
            print('   https://www.cebwm.com/wealth/grlc/index.html')
            return False

    def get_product_list_from_ajax(self, page=1):
        """通过AJAX获取产品列表（包含HTMLURL）"""
        try:
            # 获取pageId
            page_id = self.driver.execute_script(
                "return document.getElementById('eprotalCurrentPageId').value;"
            )

            # 调用AJAX API
            result = self.driver.execute_script(f"""
                var result = null;
                $.ajax({{
                    type: 'POST',
                    url: '/eportal/ui?struts.portlet.action=/portlet/financeProductSearchList!searchFinanceList.action&pageId={page_id}&pageKey=financeProductSearchListAction',
                    dataType: 'json',
                    async: false,
                    data: {{
                        inputQueryStr: '',
                        mjbzArr: [],
                        xsqdjeArr: [],
                        cpfxdjArr: [],
                        sfkfsArr: [],
                        pageKeyStr: 'financeProductSearchListAction',
                        currentPage: {page},
                        isGrlcStr: '1',
                        qclx: '',
                        cpFlbs: '0'
                    }},
                    success: function(data) {{
                        result = data;
                    }},
                    error: function() {{
                        result = null;
                    }}
                }});
                return result;
            """)

            return result
        except Exception as e:
            print(f"获取产品列表失败: {e}")
            return None

    def extract_page_products(self):
        """从当前页面表格提取产品数据"""
        products = []

        try:
            # 等待表格加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )

            # 获取产品表格（第3个表格，索引2）
            tables = self.driver.find_elements(By.TAG_NAME, 'table')
            if len(tables) < 3:
                print("未找到产品表格")
                return products

            product_table = tables[2]
            rows = product_table.find_elements(By.TAG_NAME, 'tr')

            for row in rows[1:]:  # 跳过表头
                try:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 10:
                        name = cells[0].text.strip()
                        code = cells[1].text.strip()

                        if not name or not code or name == '产品名称':
                            continue

                        product = {
                            '银行': '光大理财',
                            '产品名称': name,
                            '产品代码': code,
                            '起点金额': cells[2].text.strip(),
                            '单位净值': self._parse_number(cells[3].text),
                            '累计净值': self._parse_number(cells[4].text),
                            '资产净值': cells[5].text.strip(),
                            '近一月涨跌幅': self._parse_percent(cells[6].text),
                            '近一年涨跌幅': self._parse_percent(cells[7].text),
                            '成立以来涨跌幅': self._parse_percent(cells[8].text),
                            '风险等级': cells[9].text.strip() if len(cells) > 9 else '',
                        }
                        products.append(product)
                except Exception as e:
                    continue

        except TimeoutException:
            print("  表格加载超时")
        except Exception as e:
            print(f"  提取错误: {e}")

        return products

    def get_nav_history_for_product(self, product_code, product_name):
        """获取单个产品的历史净值数据"""
        nav_data = []
        main_window = self.driver.current_window_handle

        try:
            # 找到产品行并点击详情
            tables = self.driver.find_elements(By.TAG_NAME, 'table')
            if len(tables) < 3:
                return nav_data

            product_table = tables[2]
            rows = product_table.find_elements(By.TAG_NAME, 'tr')

            for row in rows[1:]:
                cells = row.find_elements(By.TAG_NAME, 'td')
                if len(cells) >= 2:
                    code = cells[1].text.strip()
                    if code == product_code:
                        # 找到详情链接
                        detail_links = row.find_elements(By.XPATH, ".//a[contains(text(), '详')]")
                        if detail_links:
                            # 使用ActionChains模拟真实点击
                            actions = ActionChains(self.driver)
                            actions.move_to_element(detail_links[0])
                            actions.pause(0.3)
                            actions.click()
                            actions.perform()

                            time.sleep(3)

                            # 检查是否打开了新窗口
                            if len(self.driver.window_handles) > 1:
                                new_window = [h for h in self.driver.window_handles if h != main_window][0]
                                self.driver.switch_to.window(new_window)

                                # 在详情页查找并点击"产品净值"
                                nav_data = self._extract_nav_from_detail_page()

                                # 关闭详情页，回到主页面
                                self.driver.close()
                                self.driver.switch_to.window(main_window)
                            else:
                                # 可能是弹窗或iframe
                                nav_data = self._extract_nav_from_modal()

                        break

        except Exception as e:
            print(f"  获取产品 {product_code} 净值历史失败: {e}")

            # 确保回到主窗口
            try:
                if len(self.driver.window_handles) > 1:
                    for handle in self.driver.window_handles:
                        if handle != main_window:
                            self.driver.switch_to.window(handle)
                            self.driver.close()
                    self.driver.switch_to.window(main_window)
            except:
                pass

        return nav_data

    def _extract_nav_from_detail_page(self):
        """从详情页提取净值数据"""
        nav_data = []

        try:
            time.sleep(2)

            # 查找并点击"产品净值"标签
            nav_tabs = self.driver.find_elements(By.XPATH, "//*[contains(text(), '产品净值')]")
            for tab in nav_tabs:
                try:
                    if tab.is_displayed():
                        self.driver.execute_script("arguments[0].click();", tab)
                        time.sleep(2)
                        break
                except:
                    pass

            # 查找净值表格
            tables = self.driver.find_elements(By.TAG_NAME, 'table')
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, 'tr')
                if len(rows) >= 5:  # 至少有几行数据
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, 'td')
                        if len(cells) >= 2:
                            # 尝试解析日期和净值
                            date_text = cells[0].text.strip()
                            if self._is_date(date_text):
                                nav_entry = {
                                    'date': date_text,
                                    'unit_nav': self._parse_number(cells[1].text) if len(cells) > 1 else None,
                                    'total_nav': self._parse_number(cells[2].text) if len(cells) > 2 else None
                                }
                                nav_data.append(nav_entry)

        except Exception as e:
            print(f"  提取详情页净值失败: {e}")

        return nav_data

    def _extract_nav_from_modal(self):
        """从弹窗/模态框提取净值数据"""
        nav_data = []

        try:
            # 查找可能的弹窗容器
            modal_selectors = [
                '.modal', '.dialog', '.popup', '.layer',
                '[class*="modal"]', '[class*="dialog"]', '[class*="popup"]',
                '.el-dialog', '.ant-modal', '#layui-layer1'
            ]

            for selector in modal_selectors:
                try:
                    modals = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for modal in modals:
                        if modal.is_displayed():
                            # 在弹窗中查找净值标签并点击
                            nav_tabs = modal.find_elements(By.XPATH, ".//*[contains(text(), '产品净值')]")
                            for tab in nav_tabs:
                                try:
                                    if tab.is_displayed():
                                        self.driver.execute_script("arguments[0].click();", tab)
                                        time.sleep(1)
                                        break
                                except:
                                    pass

                            # 提取净值表格
                            tables = modal.find_elements(By.TAG_NAME, 'table')
                            for table in tables:
                                rows = table.find_elements(By.TAG_NAME, 'tr')
                                for row in rows:
                                    cells = row.find_elements(By.TAG_NAME, 'td')
                                    if len(cells) >= 2:
                                        date_text = cells[0].text.strip()
                                        if self._is_date(date_text):
                                            nav_entry = {
                                                'date': date_text,
                                                'unit_nav': self._parse_number(cells[1].text) if len(cells) > 1 else None,
                                                'total_nav': self._parse_number(cells[2].text) if len(cells) > 2 else None
                                            }
                                            nav_data.append(nav_entry)

                            # 关闭弹窗
                            close_btns = modal.find_elements(By.CSS_SELECTOR, '.close, .ui_close, [class*="close"]')
                            for btn in close_btns:
                                try:
                                    if btn.is_displayed():
                                        btn.click()
                                        break
                                except:
                                    pass

                            if nav_data:
                                return nav_data
                except:
                    continue

        except Exception as e:
            print(f"  提取弹窗净值失败: {e}")

        return nav_data

    def _is_date(self, text):
        """检查是否是日期格式"""
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}/\d{2}/\d{2}',
            r'\d{4}\.\d{2}\.\d{2}'
        ]
        for pattern in date_patterns:
            if re.match(pattern, text):
                return True
        return False

    def _parse_number(self, text):
        """解析数字"""
        try:
            text = text.strip().replace(',', '')
            if text and text != '--' and text != '-':
                return float(text)
        except:
            pass
        return None

    def _parse_percent(self, text):
        """解析百分比"""
        try:
            text = text.strip().replace('%', '').replace(',', '')
            if text and text != '--' and text != '-':
                return float(text)
        except:
            pass
        return None

    def click_next_page(self):
        """点击下一页"""
        try:
            # 调用页面的翻页函数
            self.driver.execute_script("goPage1('next');")
            time.sleep(3)
            return True
        except:
            pass

        # 备用方法：直接点击按钮
        try:
            next_btns = self.driver.find_elements(By.CSS_SELECTOR, '.page_right')
            for btn in next_btns:
                if btn.is_displayed():
                    btn.click()
                    time.sleep(3)
                    return True
        except:
            pass

        return False

    def crawl_all(self, max_products=None, get_nav_history=True):
        """爬取所有产品"""
        if not self.connect():
            return []

        # 获取总产品数
        try:
            total_text = self.driver.find_element(By.XPATH, "//*[contains(text(), '共有')]").text
            import re
            match = re.search(r'(\d+)', total_text)
            total = int(match.group(1)) if match else 0
            print(f"\n共有 {total} 款产品")
        except:
            total = 0
            print("无法获取总产品数")

        page = 1
        consecutive_empty = 0

        while True:
            print(f"\n提取第 {page} 页...")

            page_products = self.extract_page_products()

            if page_products:
                # 去重
                existing_codes = {p['产品代码'] for p in self.products}
                new_products = [p for p in page_products if p['产品代码'] not in existing_codes]

                self.products.extend(new_products)
                print(f"  获取 {len(new_products)} 个新产品，总计: {len(self.products)}")
                consecutive_empty = 0

                # 获取历史净值（可选）
                if get_nav_history:
                    for product in new_products:
                        code = product['产品代码']
                        name = product['产品名称']
                        print(f"  获取 {code} 的历史净值...")
                        nav_data = self.get_nav_history_for_product(code, name)
                        if nav_data:
                            self.nav_history[code] = nav_data
                            print(f"    获取到 {len(nav_data)} 条净值记录")
                        else:
                            print(f"    未获取到净值数据")
            else:
                consecutive_empty += 1
                print("  本页无数据")
                if consecutive_empty >= 3:
                    print("连续3页无数据，停止")
                    break

            # 检查是否达到限制
            if max_products and len(self.products) >= max_products:
                print(f"已达到最大产品数限制: {max_products}")
                break

            if total > 0 and len(self.products) >= total:
                print("已获取全部产品")
                break

            # 翻页
            if not self.click_next_page():
                print("无法翻页，停止")
                break

            page += 1

            # 进度保存（每10页保存一次）
            if page % 10 == 0:
                self._save_progress()

        return self.products

    def _save_progress(self):
        """保存进度"""
        filename = f'cebwm_progress_{len(self.products)}.json'
        data = {
            'products': self.products,
            'nav_history': self.nav_history
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  进度已保存: {filename}")

    def save_to_excel(self, filename=None):
        """保存到Excel"""
        if not self.products:
            print("没有数据可保存")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'光大理财_{timestamp}.xlsx'

        df = pd.DataFrame(self.products)

        # 添加净值历史列
        if self.nav_history:
            # 获取所有日期
            all_dates = set()
            for code, nav_list in self.nav_history.items():
                for nav in nav_list:
                    if nav.get('date'):
                        all_dates.add(nav['date'])

            # 按日期排序
            sorted_dates = sorted(list(all_dates), reverse=True)[:20]  # 最近20天

            # 添加日期列
            for date in sorted_dates:
                col_name = f'净值_{date}'
                df[col_name] = None

            # 填充净值数据
            for idx, row in df.iterrows():
                code = row['产品代码']
                if code in self.nav_history:
                    nav_dict = {nav['date']: nav['unit_nav'] for nav in self.nav_history[code] if nav.get('date')}
                    for date in sorted_dates:
                        col_name = f'净值_{date}'
                        if date in nav_dict:
                            df.at[idx, col_name] = nav_dict[date]

        # 列顺序
        base_columns = [
            '银行', '产品名称', '产品代码', '风险等级', '起点金额',
            '单位净值', '累计净值', '资产净值',
            '近一月涨跌幅', '近一年涨跌幅', '成立以来涨跌幅'
        ]
        base_columns = [c for c in base_columns if c in df.columns]

        # 添加净值日期列
        nav_columns = [c for c in df.columns if c.startswith('净值_')]
        all_columns = base_columns + sorted(nav_columns, reverse=True)
        df = df[[c for c in all_columns if c in df.columns]]

        df.to_excel(filename, index=False)
        print(f"\n已保存到: {filename}")
        print(f"共 {len(df)} 个产品")

        return filename


def main():
    """主函数"""
    print("=" * 60)
    print("光大理财产品历史净值爬虫")
    print("=" * 60)
    print("\n使用前请确保:")
    print("1. Chrome已以调试模式启动:")
    print('   chrome.exe --remote-debugging-port=9222')
    print("2. 在浏览器中打开:")
    print('   https://www.cebwm.com/wealth/grlc/index.html')
    print()

    crawler = CEBWMNavCrawler()

    # 测试模式：只获取10个产品
    print("开始测试模式（获取10个产品）...")
    products = crawler.crawl_all(max_products=10, get_nav_history=True)

    if products:
        # 保存到Excel
        crawler.save_to_excel()

        # 显示统计
        print("\n统计信息:")
        print(f"  总产品数: {len(products)}")
        print(f"  获取净值历史的产品数: {len(crawler.nav_history)}")

        # 风险等级分布
        risk_counts = {}
        for p in products:
            risk = p.get('风险等级', '未知')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        print("  风险等级分布:")
        for risk, count in sorted(risk_counts.items()):
            print(f"    {risk}: {count}")
    else:
        print("\n未获取到数据")


if __name__ == "__main__":
    main()
