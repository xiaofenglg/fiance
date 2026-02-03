# -*- coding: utf-8 -*-
"""
光大理财爬虫 - 通过点击获取详情页

点击产品链接进入详情页，提取历史净值数据
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


class CEBWMClickCrawler:
    """通过点击获取净值的爬虫"""

    def __init__(self, debug_port=9222):
        self.debug_port = debug_port
        self.driver = None
        self.products = []
        self.main_window = None

    def connect(self):
        """连接Chrome"""
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
        """确保在产品列表页"""
        if 'grlc/index' not in self.driver.current_url:
            self.driver.get("https://www.cebwm.com/wealth/grlc/index.html")
            time.sleep(5)

    def get_product_table(self):
        """获取产品表格"""
        tables = self.driver.find_elements(By.TAG_NAME, 'table')
        for table in tables:
            rows = table.find_elements(By.TAG_NAME, 'tr')
            if len(rows) > 5:
                # 检查是否有产品数据
                for row in rows[1:3]:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 10:
                        code = cells[1].text.strip()
                        if code.startswith('EB'):
                            return table
        return None

    def extract_product_and_nav(self, row_index):
        """提取单个产品数据和净值历史"""
        product = None
        nav_history = []

        try:
            # 重新获取表格（页面可能已刷新）
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

            # 基础产品数据
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

            # 点击产品名称进入详情页
            print(f"    点击进入详情页...")
            name_link = cells[0].find_element(By.TAG_NAME, 'a')

            # 记录当前窗口数
            windows_before = len(self.driver.window_handles)

            # 使用多种方式点击
            try:
                # 方式1: ActionChains
                actions = ActionChains(self.driver)
                actions.move_to_element(name_link)
                actions.pause(0.5)
                actions.click()
                actions.perform()
            except:
                # 方式2: JavaScript click
                self.driver.execute_script("arguments[0].click();", name_link)

            time.sleep(3)

            # 检查是否打开了新窗口
            if len(self.driver.window_handles) > windows_before:
                # 切换到新窗口
                new_window = [h for h in self.driver.window_handles if h != self.main_window][-1]
                self.driver.switch_to.window(new_window)

                print(f"    详情页: {self.driver.current_url[:80]}...")

                # 等待页面加载
                time.sleep(2)

                # 提取净值历史
                nav_history = self._extract_nav_from_page()

                # 关闭详情页
                self.driver.close()
                self.driver.switch_to.window(self.main_window)

            else:
                print(f"    未打开新窗口")

        except Exception as e:
            print(f"    错误: {e}")
            # 确保回到主窗口
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
            # 查找所有表格
            tables = self.driver.find_elements(By.TAG_NAME, 'table')

            for table in tables:
                rows = table.find_elements(By.TAG_NAME, 'tr')

                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 2:
                        date_text = cells[0].text.strip()

                        # 检查是否是日期
                        if re.match(r'\d{4}-\d{2}-\d{2}', date_text):
                            nav_entry = {
                                'date': date_text,
                                'unit_nav': self._parse_number(cells[1].text),
                                'total_nav': self._parse_number(cells[2].text) if len(cells) > 2 else None
                            }
                            nav_history.append(nav_entry)

        except Exception as e:
            print(f"    提取净值错误: {e}")

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
        """翻页"""
        try:
            self.driver.execute_script("goPage1('next');")
            time.sleep(3)
            return True
        except:
            return False

    def crawl(self, max_products=10):
        """爬取产品"""
        if not self.connect():
            return []

        self.go_to_list_page()
        time.sleep(2)

        # 获取总数
        try:
            total_el = self.driver.find_element(By.XPATH, "//*[contains(text(), '共有')]")
            match = re.search(r'(\d+)', total_el.text)
            total = int(match.group(1)) if match else 0
            print(f"\n共有 {total} 款产品")
        except:
            total = 0

        page = 1
        row_index = 1  # 从第2行开始（第1行是表头）

        while len(self.products) < max_products:
            print(f"\n=== 第 {page} 页 ===")

            table = self.get_product_table()
            if not table:
                print("未找到产品表格")
                break

            rows = table.find_elements(By.TAG_NAME, 'tr')
            data_rows = len(rows) - 1  # 减去表头

            for i in range(data_rows):
                if len(self.products) >= max_products:
                    break

                row_idx = i + 1  # 跳过表头
                print(f"\n产品 {len(self.products)+1}/{max_products}:")

                product, nav_history = self.extract_product_and_nav(row_idx)

                if product:
                    # 添加净值数据
                    if nav_history:
                        for nav in nav_history[:20]:
                            col = f"净值_{nav['date']}"
                            product[col] = nav['unit_nav']
                        print(f"    {product['产品代码']}: 获取 {len(nav_history)} 条净值")
                    else:
                        print(f"    {product['产品代码']}: 无净值数据")

                    self.products.append(product)

                time.sleep(1)

            # 翻页
            if len(self.products) < max_products:
                if not self.click_next_page():
                    print("无法翻页")
                    break
                page += 1
                time.sleep(2)

        return self.products

    def save_to_excel(self, filename=None):
        """保存到Excel"""
        if not self.products:
            print("没有数据")
            return None

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'光大理财_{timestamp}.xlsx'

        df = pd.DataFrame(self.products)

        # 列排序
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
    print("光大理财净值爬虫 - 点击模式")
    print("=" * 60)

    crawler = CEBWMClickCrawler()

    # 测试：获取5个产品
    print("\n测试: 获取5个产品的历史净值...")
    products = crawler.crawl(max_products=5)

    if products:
        crawler.save_to_excel()
    else:
        print("未获取到数据")


if __name__ == "__main__":
    main()
