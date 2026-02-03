"""
单独爬取宁波银行理财产品
"""

import sys
sys.path.insert(0, '.')

from bank_crawler import NBBCrawler, MultiBankCrawler, BankType
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("="*60)
    print("    宁波银行理财产品采集")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 创建管理器，只注册宁波银行
    manager = MultiBankCrawler()
    manager.register_crawler(NBBCrawler())

    # 爬取
    profiles = manager.crawl_all()

    if profiles:
        filename = manager.save_results()
        print(f"\n完成！获取 {len(profiles)} 个产品")
        print(f"文件: {filename}")
    else:
        print("\n未获取到数据，可能是网络问题")


if __name__ == "__main__":
    main()
