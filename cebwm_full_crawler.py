# -*- coding: utf-8 -*-
"""
光大理财完整爬虫 - 获取产品列表和历史净值

URL模式: https://www.cebwm.com/wealth/lcxx/lccp14/{产品ID}/index.html

需要先以调试模式启动Chrome: chrome.exe --remote-debugging-port=9222
然后在浏览器中打开: https://www.cebwm.com/wealth/grlc/index.html

日期：2026-01-17
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import time
import json
import re
import os
from datetime import datetime


class CEBWMFullCrawler:
    """光大理财完整爬虫"""

    def __init__(self, debug_port=9222):
        self.debug_port = debug_port
        self.driver = None
        self.products = []
        self.main_window = None

    def connect(self):
        """连接到Chrome浏览器"""
        print("连接到Chrome浏览器...")
        try:
            options = Options()
            options.add_experimental_option('debuggerAddress', f'127.0.0.1:{self.debug_port}')
            self.driver = webdriver.Chrome(options=options)
            self.main_window = self.driver.current_window_handle

            # 检查当前页面
            print(f"已连接: {self.driver.current_url}")
            return True

        except Exception as e:
            print(f"连接失败: {e}")
            print("\n请确保Chrome已以调试模式启动:")
            print('   chrome.exe --remote-debugging-port=9222')
            return False

    def go_to_product_list(self):
        """导航到产品列表页"""
        list_url = "https://www.cebwm.com/wealth/grlc/index.html"
        if 'grlc/index' not in self.driver.current_url:
            print(f"导航到产品列表页...")
            self.driver.get(list_url)
            time.sleep(5)
        return 'grlc' in self.driver.current_url

    def get_total_products(self):
        """获取总产品数"""
        try:
            total_el = self.driver.find_element(By.XPATH, "//*[contains(text(), '共有')]")
            match = re.search(r'(\d+)', total_el.text)
            if match:
                return int(match.group(1))
        except:
            pass
        return 0

    def extract_products_with_urls(self):
        """从当前页面提取产品数据和详情页URL"""
        products = []

        try:
            # 等待表格加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )

            # 找到产品表格（通常是第3个表格）
            tables = self.driver.find_elements(By.TAG_NAME, 'table')

            for table in tables:
                rows = table.find_elements(By.TAG_NAME, 'tr')
                if len(rows) < 5:
                    continue

                # 检查是否是产品表格（有产品名称列）
                first_row = rows[0] if rows else None
                if first_row:
                    headers = first_row.find_elements(By.TAG_NAME, 'th')
                    header_texts = [h.text.strip() for h in headers]
                    if '产品名称' not in header_texts and '代码' not in header_texts:
                        # 检查td
                        tds = first_row.find_elements(By.TAG_NAME, 'td')
                        td_texts = [t.text.strip() for t in tds]
                        if '产品名称' not in td_texts:
                            continue

                for row in rows[1:]:  # 跳过表头
                    try:
                        cells = row.find_elements(By.TAG_NAME, 'td')
                        if len(cells) < 10:
                            continue

                        name = cells[0].text.strip()
                        code = cells[1].text.strip()

                        if not name or not code or name == '产品名称':
                            continue

                        # 获取详情页URL（从产品名称或详情链接）
                        detail_url = None

                        # 方法1: 从产品名称链接获取
                        name_links = cells[0].find_elements(By.TAG_NAME, 'a')
                        if name_links:
                            href = name_links[0].get_attribute('href')
                            if href and 'wealth/lcxx' in href:
                                detail_url = href

                        # 方法2: 从最后一列的详情链接获取
                        if not detail_url:
                            detail_links = cells[-1].find_elements(By.TAG_NAME, 'a')
                            for link in detail_links:
                                href = link.get_attribute('href')
                                if href and 'wealth/lcxx' in href:
                                    detail_url = href
                                    break

                        # 方法3: 从data属性获取
                        if not detail_url:
                            for cell in cells:
                                links = cell.find_elements(By.TAG_NAME, 'a')
                                for link in links:
                                    # 检查各种属性
                                    for attr in ['href', 'data-url', 'data-href']:
                                        val = link.get_attribute(attr)
                                        if val and 'wealth/lcxx' in val:
                                            detail_url = val
                                            break
                                    if detail_url:
                                        break
                                if detail_url:
                                    break

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
                            '详情页URL': detail_url,
                        }
                        products.append(product)

                    except Exception as e:
                        continue

                if products:  # 找到产品后跳出
                    break

        except Exception as e:
            print(f"  提取产品错误: {e}")

        return products

    def get_nav_history(self, detail_url):
        """获取产品历史净值"""
        nav_history = []

        if not detail_url:
            return nav_history

        try:
            # 打开详情页
            self.driver.execute_script(f"window.open('{detail_url}', '_blank');")
            time.sleep(3)

            # 切换到新窗口
            new_windows = [h for h in self.driver.window_handles if h != self.main_window]
            if not new_windows:
                return nav_history

            self.driver.switch_to.window(new_windows[-1])
            time.sleep(2)

            # 查找净值表格
            tables = self.driver.find_elements(By.TAG_NAME, 'table')

            for table in tables:
                rows = table.find_elements(By.TAG_NAME, 'tr')

                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 2:
                        date_text = cells[0].text.strip()

                        # 检查是否是日期格式
                        if re.match(r'\d{4}-\d{2}-\d{2}', date_text):
                            nav_entry = {
                                'date': date_text,
                                'unit_nav': self._parse_number(cells[1].text),
                                'total_nav': self._parse_number(cells[2].text) if len(cells) > 2 else None
                            }
                            nav_history.append(nav_entry)

            # 关闭详情页
            self.driver.close()
            self.driver.switch_to.window(self.main_window)

        except Exception as e:
            print(f"    获取净值历史错误: {e}")
            # 确保回到主窗口
            try:
                for handle in self.driver.window_handles:
                    if handle != self.main_window:
                        self.driver.switch_to.window(handle)
                        self.driver.close()
                self.driver.switch_to.window(self.main_window)
            except:
                pass

        return nav_history

    def click_next_page(self):
        """点击下一页"""
        try:
            # 使用JavaScript翻页
            self.driver.execute_script("goPage1('next');")
            time.sleep(3)
            return True
        except:
            pass

        try:
            next_btns = self.driver.find_elements(By.CSS_SELECTOR, '.page_right')
            for btn in next_btns:
                if btn.is_displayed():
                    self.driver.execute_script("arguments[0].click();", btn)
                    time.sleep(3)
                    return True
        except:
            pass

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

    def crawl_all(self, max_products=None, get_nav=True):
        """爬取所有产品"""
        if not self.connect():
            return []

        if not self.go_to_product_list():
            print("无法访问产品列表页")
            return []

        total = self.get_total_products()
        print(f"\n共有 {total} 款产品")

        page = 1
        consecutive_empty = 0

        while True:
            print(f"\n=== 第 {page} 页 ===")

            page_products = self.extract_products_with_urls()

            if page_products:
                # 去重
                existing_codes = {p['产品代码'] for p in self.products}
                new_products = [p for p in page_products if p['产品代码'] not in existing_codes]

                print(f"获取 {len(new_products)} 个新产品")

                # 获取历史净值
                if get_nav:
                    for i, product in enumerate(new_products):
                        code = product['产品代码']
                        url = product.get('详情页URL')

                        if url:
                            print(f"  [{i+1}/{len(new_products)}] {code} 获取净值...")
                            nav_history = self.get_nav_history(url)

                            if nav_history:
                                # 添加净值数据到产品
                                for nav in nav_history[:20]:  # 最近20天
                                    col_name = f"净值_{nav['date']}"
                                    product[col_name] = nav['unit_nav']
                                print(f"      获取到 {len(nav_history)} 条净值")
                            else:
                                print(f"      无净值数据")
                        else:
                            print(f"  [{i+1}/{len(new_products)}] {code} 无详情页URL")

                self.products.extend(new_products)
                print(f"总计: {len(self.products)} 个产品")
                consecutive_empty = 0

            else:
                consecutive_empty += 1
                print("本页无数据")
                if consecutive_empty >= 3:
                    print("连续3页无数据，停止")
                    break

            # 检查限制
            if max_products and len(self.products) >= max_products:
                print(f"已达到最大产品数: {max_products}")
                break

            if total > 0 and len(self.products) >= total:
                print("已获取全部产品")
                break

            # 翻页
            if not self.click_next_page():
                print("无法翻页，停止")
                break

            page += 1

            # 保存进度
            if page % 5 == 0:
                self._save_progress()

        return self.products

    def _save_progress(self):
        """保存进度"""
        filename = f'cebwm_progress_{len(self.products)}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)
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

        # 基础列
        base_columns = [
            '银行', '产品名称', '产品代码', '风险等级', '起点金额',
            '单位净值', '累计净值', '资产净值',
            '近一月涨跌幅', '近一年涨跌幅', '成立以来涨跌幅'
        ]

        # 净值日期列（按日期排序）
        nav_columns = sorted([c for c in df.columns if c.startswith('净值_')], reverse=True)

        # 组合列顺序
        all_columns = [c for c in base_columns if c in df.columns] + nav_columns
        df = df[[c for c in all_columns if c in df.columns]]

        df.to_excel(filename, index=False)
        print(f"\n已保存到: {filename}")
        print(f"共 {len(df)} 个产品, {len(nav_columns)} 个净值日期列")

        return filename


def main():
    """主函数"""
    print("=" * 60)
    print("光大理财完整爬虫")
    print("=" * 60)
    print("\n使用前请确保:")
    print("1. Chrome已以调试模式启动:")
    print('   chrome.exe --remote-debugging-port=9222')
    print("2. 在浏览器中打开产品列表页:")
    print('   https://www.cebwm.com/wealth/grlc/index.html')
    print()

    crawler = CEBWMFullCrawler()

    # 测试模式：获取5个产品
    print("测试模式：获取5个产品的历史净值...")
    products = crawler.crawl_all(max_products=5, get_nav=True)

    if products:
        crawler.save_to_excel()

        print("\n统计:")
        print(f"  产品数: {len(products)}")

        # 统计有净值数据的产品
        nav_count = sum(1 for p in products if any(k.startswith('净值_') for k in p.keys()))
        print(f"  有净值数据: {nav_count}")
    else:
        print("\n未获取到数据")


if __name__ == "__main__":
    main()
