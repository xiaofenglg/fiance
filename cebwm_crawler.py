# -*- coding: utf-8 -*-
"""
光大理财产品数据爬虫

通过CDP连接到已打开的Chrome浏览器，抓取光大理财产品数据
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
import pandas as pd
import time
import json
import re
import os
from datetime import datetime


class CEBWMCrawler:
    """光大理财爬虫"""

    def __init__(self, debug_port=9222):
        self.debug_port = debug_port
        self.driver = None
        self.products = []

    def connect(self):
        """连接到Chrome浏览器"""
        print("连接到Chrome浏览器...")
        options = Options()
        options.add_experimental_option('debuggerAddress', f'127.0.0.1:{self.debug_port}')
        self.driver = webdriver.Chrome(options=options)

        # 切换到光大理财页面
        for handle in self.driver.window_handles:
            self.driver.switch_to.window(handle)
            if 'cebwm' in self.driver.current_url:
                print(f"已连接: {self.driver.title}")
                return True

        print("未找到光大理财页面，请先在浏览器中打开")
        return False

    def get_total_products(self):
        """获取总产品数"""
        try:
            total_text = self.driver.find_element(By.XPATH, "//*[contains(text(), '共有')]").text
            match = re.search(r'(\d+)', total_text)
            if match:
                return int(match.group(1))
        except:
            pass
        return 0

    def extract_page_products(self):
        """提取当前页产品数据"""
        products = []

        try:
            # 等待表格加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )

            rows = self.driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')

            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 10:
                        # 清理数据
                        name = cells[0].text.strip()
                        code = cells[1].text.strip()

                        # 跳过表头或空行
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

    def _parse_number(self, text):
        """解析数字"""
        try:
            text = text.strip().replace(',', '')
            if text and text != '--':
                return float(text)
        except:
            pass
        return None

    def _parse_percent(self, text):
        """解析百分比"""
        try:
            text = text.strip().replace('%', '').replace(',', '')
            if text and text != '--':
                return float(text)
        except:
            pass
        return None

    def click_next_page(self):
        """点击下一页"""
        try:
            # 尝试多种方式找到下一页按钮
            next_selectors = [
                "//a[contains(text(), '下一页')]",
                "//a[contains(text(), '>')]",
                "//a[contains(@class, 'next')]",
                "//li[contains(@class, 'next')]/a",
                "//button[contains(text(), '下一页')]",
            ]

            for selector in next_selectors:
                try:
                    next_btn = self.driver.find_element(By.XPATH, selector)
                    if next_btn.is_displayed() and next_btn.is_enabled():
                        next_btn.click()
                        time.sleep(2)  # 等待页面加载
                        return True
                except:
                    continue

            # 尝试通过页码点击
            try:
                current_page = self.driver.find_element(By.CSS_SELECTOR, ".active, .current")
                page_num = int(current_page.text)
                next_page = self.driver.find_element(By.XPATH, f"//a[text()='{page_num + 1}']")
                next_page.click()
                time.sleep(2)
                return True
            except:
                pass

        except Exception as e:
            print(f"  翻页失败: {e}")

        return False

    def crawl_all(self, max_pages=None):
        """爬取所有产品"""
        if not self.connect():
            return []

        total = self.get_total_products()
        print(f"\n共有 {total} 款产品")

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
            else:
                consecutive_empty += 1
                print(f"  本页无数据")
                if consecutive_empty >= 3:
                    print("连续3页无数据，停止")
                    break

            # 检查是否达到限制
            if max_pages and page >= max_pages:
                print(f"已达到最大页数限制: {max_pages}")
                break

            if len(self.products) >= total:
                print("已获取全部产品")
                break

            # 翻页
            if not self.click_next_page():
                print("无法翻页，停止")
                break

            page += 1

            # 进度保存（每50页保存一次）
            if page % 50 == 0:
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

        # 列顺序
        columns = [
            '银行', '产品名称', '产品代码', '风险等级', '起点金额',
            '单位净值', '累计净值', '资产净值',
            '近一月涨跌幅', '近一年涨跌幅', '成立以来涨跌幅'
        ]
        columns = [c for c in columns if c in df.columns]
        df = df[columns]

        df.to_excel(filename, index=False)
        print(f"\n已保存到: {filename}")
        print(f"共 {len(df)} 个产品")

        return filename


def main():
    """主函数"""
    print("=" * 60)
    print("光大理财产品数据爬虫")
    print("=" * 60)
    print("\n使用前请确保:")
    print("1. Chrome已以调试模式启动:")
    print('   chrome.exe --remote-debugging-port=9222')
    print("2. 在浏览器中打开:")
    print('   https://www.cebwm.com/wealth/grlc/index.html')
    print()

    crawler = CEBWMCrawler()

    # 爬取数据
    products = crawler.crawl_all()

    if products:
        # 保存到Excel
        crawler.save_to_excel()

        # 显示统计
        print("\n统计信息:")
        print(f"  总产品数: {len(products)}")

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
