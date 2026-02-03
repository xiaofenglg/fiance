# -*- coding: utf-8 -*-
"""
银行理财数据采集统一入口

支持银行:
- 浦发理财 (SPDB)
- 农银理财 (ABC)
- 中邮理财 (PSBC)
- 工银理财 (ICBC)

使用方法:
  python run_crawlers.py              # 爬取所有银行
  python run_crawlers.py --bank spdb  # 只爬取浦发
  python run_crawlers.py --test       # 测试模式（每银行10个产品）
  python run_crawlers.py --days 30    # 获取30天净值

日期: 2026-01-17
"""

import sys
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, '.')

from crawlers import (
    get_crawler,
    crawl_all,
    SPDBCrawler,
    ABCCrawler,
    PSBCCrawler,
    ICBCCrawler,
    CRAWLERS,
)


def main():
    parser = argparse.ArgumentParser(description='银行理财数据采集工具')
    parser.add_argument('--bank', '-b', type=str, default=None,
                        help=f'指定银行代码 ({"/".join(CRAWLERS.keys())})')
    parser.add_argument('--test', '-t', action='store_true',
                        help='测试模式（每银行10个产品）')
    parser.add_argument('--max', '-m', type=int, default=None,
                        help='每银行最大产品数')
    parser.add_argument('--days', '-d', type=int, default=15,
                        help='获取净值天数（默认15天）')
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出支持的银行')

    args = parser.parse_args()

    # 列出支持的银行
    if args.list:
        print("\n支持的银行:")
        print("-" * 40)
        for code, crawler_cls in CRAWLERS.items():
            print(f"  {code:6s} - {crawler_cls.bank_name}")
        print()
        return

    print("=" * 70)
    print("         银行理财数据采集系统")
    print("=" * 70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 确定参数
    max_products = args.max
    if args.test:
        max_products = 10
        print("模式: 测试模式（每银行10个产品）")

    nav_days = args.days
    print(f"净值天数: {nav_days}")

    if args.bank:
        # 爬取指定银行
        bank_code = args.bank.lower()
        if bank_code not in CRAWLERS:
            print(f"错误: 不支持的银行代码 '{bank_code}'")
            print(f"支持的银行: {', '.join(CRAWLERS.keys())}")
            return

        print(f"目标银行: {CRAWLERS[bank_code].bank_name}")
        print()

        crawler = get_crawler(bank_code)
        products = crawler.crawl(max_products=max_products, nav_days=nav_days)

        if products:
            filename = crawler.save_to_excel()
            crawler.display_report()
            print(f"\n数据已保存到: {filename}")
        else:
            print("\n未获取到数据。")
            print("可能的原因:")
            print("  1. 网站API端点需要手动分析")
            print("  2. 网站有反爬虫/WAF保护")
            print("  3. SSL连接问题")
            print("  4. 网站结构已变化")

    else:
        # 爬取所有银行
        print("目标: 所有银行")
        print()

        results = crawl_all(max_products=max_products, nav_days=nav_days)

        # 总结
        success_count = sum(1 for r in results.values() if r['success'])
        total_products = sum(r['count'] for r in results.values())

        print(f"\n总计: {success_count}/{len(results)} 个银行成功")
        print(f"总产品数: {total_products}")


if __name__ == "__main__":
    main()
