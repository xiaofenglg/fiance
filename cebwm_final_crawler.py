# -*- coding: utf-8 -*-
"""
光大理财最终版爬虫 - 通过点击获取详情页净值

通过点击产品链接进入详情页，等待加载后提取历史净值
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import json
import re
from datetime import datetime


class CEBWMFinalCrawler:
    """光大理财最终版爬虫"""

    def __init__(self, debug_port=9222):
        self.debug_port = debug_port
        self.driver = None
        self.products = []
        self.main_window = None

    def connect(self):
        print("连接Chrome...")
        try:
            options = Options()
            options.add_experimental_option('debuggerAddress', f'127.0.0.1:{self.debug_port}')
            self.driver = webdriver.Chrome(options=options)
            self.main_window = self.driver.current_window_handle
            print(f"已连接: {self.driver.current_url}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def go_to_list_page(self):
        if 'grlc/index' not in self.driver.current_url:
            self.driver.get("https://www.cebwm.com/wealth/grlc/index.html")
            time.sleep(5)

    def get_product_table(self):
        tables = self.driver.find_elements(By.TAG_NAME, 'table')
        for table in tables:
            rows = table.find_elements(By.TAG_NAME, 'tr')
            if len(rows) > 5:
                for row in rows[1:3]:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 10:
                        code = cells[1].text.strip()
                        if code.startswith('EB'):
                            return table
        return None

    def extract_product_and_nav(self, row_index):
        product = None
        nav_history = []

        try:
            table = self.get_product_table()
            if not table:
                return None, []

            rows = table.find_elements(By.TAG_NAME, 'tr')
            if row_index >= len(rows):
                return None, []

            row = rows[row_index]
            cells = row.find_elements(By.TAG_NAME, 'td')

            if len(cells) < 10:
                return None, []

            name = cells[0].text.strip()
            code = cells[1].text.strip()

            if not name or not code or not code.startswith('EB'):
                return None, []

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

            # 点击产品名称
            print(f"    点击进入详情页...")
            name_link = cells[0].find_element(By.TAG_NAME, 'a')

            windows_before = len(self.driver.window_handles)

            # 点击
            actions = ActionChains(self.driver)
            actions.move_to_element(name_link)
            actions.pause(0.5)
            actions.click()
            actions.perform()

            # 等待新窗口
            time.sleep(2)

            if len(self.driver.window_handles) > windows_before:
                new_window = [h for h in self.driver.window_handles if h != self.main_window][-1]
                self.driver.switch_to.window(new_window)

                # 等待页面加载 - 增加等待时间
                print(f"    等待页面加载...")
                time.sleep(5)

                # 等待直到页面有足够内容
                for _ in range(10):
                    page_len = len(self.driver.page_source)
                    if page_len > 5000:
                        break
                    time.sleep(1)

                page_len = len(self.driver.page_source)
                print(f"    页面长度: {page_len}")

                if page_len > 5000:
                    # 滚动到底部确保数据加载
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)

                    # 提取净值
                    nav_history = self._extract_nav_from_page()
                else:
                    print(f"    页面未完全加载")

                # 关闭详情页
                self.driver.close()
                self.driver.switch_to.window(self.main_window)
                time.sleep(1)

        except Exception as e:
            print(f"    错误: {e}")
            try:
                for h in self.driver.window_handles:
                    if h != self.main_window:
                        self.driver.switch_to.window(h)
                        self.driver.close()
                self.driver.switch_to.window(self.main_window)
            except:
                pass

        return product, nav_history

    def _extract_nav_from_page(self):
        """从详情页提取净值历史"""
        nav_history = []

        try:
            # 方法1: 查找表格
            tables = self.driver.find_elements(By.TAG_NAME, 'table')
            print(f"    找到 {len(tables)} 个表格")

            for table in tables:
                rows = table.find_elements(By.TAG_NAME, 'tr')
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 2:
                        date_text = cells[0].text.strip()
                        if re.match(r'\d{4}-\d{2}-\d{2}', date_text):
                            nav = {
                                'date': date_text,
                                'unit_nav': self._parse_number(cells[1].text),
                                'total_nav': self._parse_number(cells[2].text) if len(cells) > 2 else None
                            }
                            nav_history.append(nav)

            if nav_history:
                return nav_history

            # 方法2: 从页面文本提取
            print(f"    从页面文本提取...")
            page_text = self.driver.find_element(By.TAG_NAME, 'body').text

            # 匹配日期和净值
            pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d+\.\d+)\s+(\d+\.\d+)'
            matches = re.findall(pattern, page_text)

            for match in matches:
                nav = {
                    'date': match[0],
                    'unit_nav': float(match[1]),
                    'total_nav': float(match[2])
                }
                nav_history.append(nav)

            if nav_history:
                return nav_history

            # 方法3: 查找所有包含日期的元素
            print(f"    查找日期元素...")
            date_els = self.driver.find_elements(By.XPATH,
                "//*[contains(text(), '2026-') or contains(text(), '2025-')]")

            for el in date_els:
                text = el.text.strip()
                parts = text.split()
                if len(parts) >= 3:
                    date_match = re.match(r'\d{4}-\d{2}-\d{2}', parts[0])
                    if date_match:
                        try:
                            nav = {
                                'date': parts[0],
                                'unit_nav': float(parts[1]),
                                'total_nav': float(parts[2]) if len(parts) > 2 else None
                            }
                            nav_history.append(nav)
                        except:
                            pass

        except Exception as e:
            print(f"    提取错误: {e}")

        return nav_history

    def _parse_number(self, text):
        try:
            text = text.strip().replace(',', '')
            if text and text != '--' and text != '-':
                return float(text)
        except:
            pass
        return None

    def _parse_percent(self, text):
        try:
            text = text.strip().replace('%', '').replace(',', '')
            if text and text != '--' and text != '-':
                return float(text)
        except:
            pass
        return None

    def click_next_page(self):
        try:
            self.driver.execute_script("goPage1('next');")
            time.sleep(3)
            return True
        except:
            return False

    def crawl(self, max_products=10):
        if not self.connect():
            return []

        self.go_to_list_page()
        time.sleep(3)

        try:
            total_el = self.driver.find_element(By.XPATH, "//*[contains(text(), '共有')]")
            match = re.search(r'(\d+)', total_el.text)
            total = int(match.group(1)) if match else 0
            print(f"\n共有 {total} 款产品")
        except:
            total = 0

        page = 1

        while len(self.products) < max_products:
            print(f"\n=== 第 {page} 页 ===")

            table = self.get_product_table()
            if not table:
                print("未找到产品表格")
                break

            rows = table.find_elements(By.TAG_NAME, 'tr')
            data_rows = len(rows) - 1

            for i in range(data_rows):
                if len(self.products) >= max_products:
                    break

                row_idx = i + 1
                print(f"\n产品 {len(self.products)+1}/{max_products}:")

                product, nav_history = self.extract_product_and_nav(row_idx)

                if product:
                    if nav_history:
                        for nav in nav_history[:20]:
                            col = f"净值_{nav['date']}"
                            product[col] = nav['unit_nav']
                        print(f"    {product['产品代码']}: 获取 {len(nav_history)} 条净值")
                    else:
                        print(f"    {product['产品代码']}: 无净值数据")

                    self.products.append(product)

                # 需要重新获取表格（页面可能变化）
                self.go_to_list_page()
                time.sleep(2)

            if len(self.products) < max_products:
                if not self.click_next_page():
                    print("无法翻页")
                    break
                page += 1
                time.sleep(2)

        return self.products

    def save_to_excel(self, filename=None):
        if not self.products:
            print("没有数据")
            return None

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'光大理财_{timestamp}.xlsx'

        df = pd.DataFrame(self.products)

        base_cols = ['银行', '产品名称', '产品代码', '风险等级', '起点金额',
                     '单位净值', '累计净值', '资产净值',
                     '近一月涨跌幅', '近一年涨跌幅', '成立以来涨跌幅']
        nav_cols = sorted([c for c in df.columns if c.startswith('净值_')], reverse=True)

        all_cols = [c for c in base_cols if c in df.columns] + nav_cols
        df = df[[c for c in all_cols if c in df.columns]]

        df.to_excel(filename, index=False)
        print(f"\n保存到: {filename}")
        print(f"共 {len(df)} 个产品, {len(nav_cols)} 个净值日期")

        return filename


def main():
    print("=" * 60)
    print("光大理财最终版爬虫")
    print("=" * 60)

    crawler = CEBWMFinalCrawler()

    print("\n测试: 获取3个产品...")
    products = crawler.crawl(max_products=3)

    if products:
        crawler.save_to_excel()

        # 显示第一个产品的净值数据
        if products:
            p = products[0]
            nav_cols = [k for k in p.keys() if k.startswith('净值_')]
            if nav_cols:
                print(f"\n{p['产品代码']} 的净值数据:")
                for col in sorted(nav_cols, reverse=True)[:10]:
                    print(f"  {col}: {p[col]}")
    else:
        print("未获取到数据")


if __name__ == "__main__":
    main()
