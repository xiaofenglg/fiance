# -*- coding: utf-8 -*-
"""
多银行费率采集统一入口

功能：
1. 调用各银行专用解析器采集费率
2. 统一存储到赎回费数据库
3. 输出统计报告

支持银行：
- 中信: 本地PDF解析 (pdfs/中信/)
- 民生: 通过 crawl_minsheng_fees.py 采集
- 宁银: TODO - 需要找到费率公告源
- 中邮: TODO - 需要绕过WAF
- 华夏: TODO - JS数据可能无费率

用法:
    python parse_bank_fees.py status          # 查看各银行费率覆盖状态
    python parse_bank_fees.py citic           # 解析中信PDF
    python parse_bank_fees.py citic --max 100 # 限量测试
    python parse_bank_fees.py minsheng        # 采集民生费率
    python parse_bank_fees.py all             # 采集所有支持的银行
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_status():
    """获取各银行费率采集状态"""
    from redemption_fee_db import get_fee_summary
    from nav_db_excel import NAVDatabaseExcel

    # 费率数据库统计
    fee_summary = get_fee_summary()

    # 净值数据库产品统计
    try:
        db = NAVDatabaseExcel()
        nav_products = {}
        for bank, df in db.data.items():
            if df is not None:
                # DataFrame index是MultiIndex(产品代码, 产品名称)
                nav_products[bank] = len(df.index.get_level_values(0).unique())
    except Exception as e:
        logger.warning(f"读取净值数据库失败: {e}")
        nav_products = {}

    print("=" * 70)
    print("              多银行费率采集状态报告")
    print("=" * 70)
    print(f"报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 按银行统计 (使用净值数据库中的实际bank名，以及费率数据库中可能存在的key)
    # 净值数据库bank names: 民生银行, 华夏银行, 中信银行, 浦银理财, 中邮理财, 宁银理财
    banks = [
        ('中信银行', '信银理财'),  # (NAV DB name, Fee DB name)
        ('民生银行', '民生银行'),
        ('宁银理财', '宁银理财'),
        ('中邮理财', '中邮理财'),
        ('华夏银行', '华夏理财'),
        ('浦银理财', '浦银理财'),
    ]

    print("银行费率覆盖:")
    print("-" * 70)
    print(f"{'银行':<12} {'净值库产品':<12} {'有费率':<10} {'覆盖率':<10} {'状态'}")
    print("-" * 70)

    for nav_bank, fee_bank in banks:
        nav_count = nav_products.get(nav_bank, 0)
        fee_stats = fee_summary['by_bank'].get(fee_bank, {})
        fee_count = fee_stats.get('total', 0)
        coverage = f"{fee_count/nav_count*100:.1f}%" if nav_count > 0 else "-"

        if fee_count > 0 and nav_count > 0:
            if fee_count / nav_count > 0.8:
                status = "[OK] 完成"
            elif fee_count / nav_count > 0.3:
                status = "[..] 进行中"
            else:
                status = "[  ] 待采集"
        elif nav_count > 0:
            status = "[  ] 待采集"
        else:
            status = "[-] 无数据"

        display_name = nav_bank if nav_bank == fee_bank else f"{nav_bank}"
        print(f"{display_name:<12} {nav_count:<12} {fee_count:<10} {coverage:<10} {status}")

    print("-" * 70)
    print(f"{'合计':<12} {sum(nav_products.values()):<12} {fee_summary['total']:<10}")
    print()

    # PDF缓存状态
    print("本地PDF缓存:")
    print("-" * 70)

    pdf_dirs = {
        '中信': os.path.join(BASE_DIR, 'pdfs', '中信'),
        '民生': os.path.join(BASE_DIR, 'pdf_archive', 'minsheng'),
    }

    for bank, pdf_dir in pdf_dirs.items():
        if os.path.exists(pdf_dir):
            pdf_count = sum(1 for f in os.listdir(pdf_dir) if f.endswith('.pdf'))
            size_mb = sum(os.path.getsize(os.path.join(pdf_dir, f))
                         for f in os.listdir(pdf_dir) if f.endswith('.pdf')) / 1024 / 1024
            print(f"  {bank}: {pdf_count} 个PDF, {size_mb:.1f} MB")
        else:
            print(f"  {bank}: (未下载)")

    print("=" * 70)

    return fee_summary


def run_citic_parser(max_count=0, force=False, debug=False):
    """运行中信PDF解析"""
    print("\n" + "=" * 60)
    print("  中信理财费率解析")
    print("=" * 60)

    cmd = [sys.executable, os.path.join(BASE_DIR, 'parse_citic_pdfs.py')]
    if max_count > 0:
        cmd.extend(['--max', str(max_count)])
    if force:
        cmd.append('--force')
    if debug:
        cmd.append('--debug')

    try:
        result = subprocess.run(cmd, cwd=BASE_DIR)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"运行中信解析器失败: {e}")
        return False


def run_minsheng_crawler(limit=0, full=False):
    """运行民生费率采集"""
    print("\n" + "=" * 60)
    print("  民生银行费率采集")
    print("=" * 60)

    cmd = [sys.executable, os.path.join(BASE_DIR, 'crawl_minsheng_fees.py')]
    if limit > 0:
        cmd.extend(['--limit', str(limit)])
    if full:
        cmd.append('--full')

    try:
        result = subprocess.run(cmd, cwd=BASE_DIR)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"运行民生采集器失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='多银行费率采集统一入口')
    parser.add_argument('command', nargs='?', default='status',
                        choices=['status', 'citic', 'minsheng', 'all'],
                        help='操作命令')
    parser.add_argument('--max', type=int, default=0,
                        help='限制处理数量（测试用）')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--full', action='store_true',
                        help='全量模式（民生用）')

    args = parser.parse_args()

    if args.command == 'status':
        get_status()

    elif args.command == 'citic':
        run_citic_parser(max_count=args.max, force=args.force, debug=args.debug)

    elif args.command == 'minsheng':
        run_minsheng_crawler(limit=args.max, full=args.full)

    elif args.command == 'all':
        print("\n" + "=" * 70)
        print("              开始全银行费率采集")
        print("=" * 70)

        # 中信
        citic_pdf_dir = os.path.join(BASE_DIR, 'pdfs', '中信')
        if os.path.exists(citic_pdf_dir):
            pdf_count = sum(1 for f in os.listdir(citic_pdf_dir) if f.endswith('.pdf'))
            if pdf_count > 0:
                print(f"\n[1/2] 中信 ({pdf_count} PDFs)")
                run_citic_parser(max_count=args.max, force=args.force)
            else:
                print(f"\n[1/2] 中信: 跳过（无PDF）")
        else:
            print(f"\n[1/2] 中信: 跳过（PDF目录不存在）")

        # 民生
        print(f"\n[2/2] 民生")
        run_minsheng_crawler(limit=args.max, full=args.full)

        # 最终状态
        print("\n")
        get_status()


if __name__ == '__main__':
    main()
