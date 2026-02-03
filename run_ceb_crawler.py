# -*- coding: utf-8 -*-
"""
光大理财爬虫运行入口

使用方法:
    # 测试模式 - 只爬取10个产品
    python run_ceb_crawler.py --test

    # 完整爬取
    python run_ceb_crawler.py

    # 从指定页开始
    python run_ceb_crawler.py --start-page 50

    # 爬取指定数量
    python run_ceb_crawler.py --max 100

    # 查看统计信息
    python run_ceb_crawler.py --stats

    # 导出Excel
    python run_ceb_crawler.py --export
"""

import argparse
import sys
from datetime import datetime

from ceb_crawler import CEBCrawler


def parse_args():
    parser = argparse.ArgumentParser(description='光大理财爬虫')

    parser.add_argument('--test', action='store_true',
                        help='测试模式，只爬取10个产品')
    parser.add_argument('--max', type=int, default=None,
                        help='最大爬取产品数量')
    parser.add_argument('--start-page', type=int, default=1,
                        help='起始页码')
    parser.add_argument('--no-resume', action='store_true',
                        help='不从上次进度继续，重新开始')
    parser.add_argument('--stats', action='store_true',
                        help='只显示统计信息')
    parser.add_argument('--export', action='store_true',
                        help='只导出已有数据到Excel')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='每隔多少产品保存一次进度')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='跳过确认，直接开始爬取')

    return parser.parse_args()


def show_stats(crawler: CEBCrawler):
    """显示统计信息"""
    crawler.load_progress()
    crawler.load_nav_history()

    stats = crawler.get_stats()

    print("\n" + "=" * 50)
    print("光大理财爬虫 - 统计信息")
    print("=" * 50)
    print(f"已爬取产品数: {stats['total_products']}")
    print(f"净值历史记录: {stats['nav_history_count']} 个产品")
    print(f"当前进度: 第 {stats['current_page']}/{stats['total_pages']} 页")
    print(f"最后更新时间: {stats['last_update'] or '无'}")
    print("=" * 50)

    # 显示最近爬取的产品
    if crawler.progress['products']:
        print("\n最近爬取的5个产品:")
        products = list(crawler.progress['products'].items())
        products.sort(key=lambda x: x[1].get('crawled_at', ''), reverse=True)
        for code, info in products[:5]:
            print(f"  {code}: {info.get('name', '')[:20]} "
                  f"(净值日期: {info.get('last_nav_date', '无')})")


def export_data(crawler: CEBCrawler):
    """导出数据到Excel"""
    crawler.load_progress()
    crawler.load_nav_history()

    if not crawler.progress['products']:
        print("没有数据可导出，请先运行爬虫")
        return

    filename = crawler.save_to_excel()
    if filename:
        print(f"数据已导出到: {filename}")


def run_crawler(args):
    """运行爬虫"""
    crawler = CEBCrawler()

    # 确定爬取数量
    max_products = args.max
    if args.test:
        max_products = 10
        print("\n[测试模式] 只爬取10个产品")

    # 开始爬取
    print("\n" + "=" * 60)
    print("光大理财爬虫 (undetected-chromedriver)")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"起始页码: {args.start_page}")
    print(f"最大产品数: {max_products or '不限'}")
    print(f"继续上次进度: {'否' if args.no_resume else '是'}")
    print("=" * 60)

    if not args.yes:
        input("\n按回车键开始爬取...")

    products = crawler.crawl(
        max_products=max_products,
        start_page=args.start_page,
        resume=not args.no_resume,
        save_interval=args.save_interval
    )

    if products:
        print("\n正在导出Excel...")
        crawler.save_to_excel()

        # 显示统计
        show_stats(crawler)
    else:
        print("\n未获取到数据")


def main():
    args = parse_args()

    crawler = CEBCrawler()

    if args.stats:
        show_stats(crawler)
    elif args.export:
        export_data(crawler)
    else:
        run_crawler(args)


if __name__ == "__main__":
    main()
