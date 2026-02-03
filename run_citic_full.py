# -*- coding: utf-8 -*-
"""运行完整的CITIC爬取 - 包含所有产品的历史净值"""
import sys
sys.path.insert(0, 'D:\\AI-FINANCE')
sys.stdout.reconfigure(encoding='utf-8')

from bank_crawler import CITICCrawler, logger
import pandas as pd
import time

# 导入净值数据库Excel管理模块
try:
    from nav_db_excel import update_nav_database
    HAS_NAV_DB = True
except ImportError:
    HAS_NAV_DB = False

def main():
    print("=" * 60)
    print("运行完整的CITIC爬取")
    print("=" * 60)

    start_time = time.time()
    crawler = CITICCrawler()

    # 1. 获取所有产品
    print("\n1. 获取产品列表...")
    products = crawler.get_product_list()
    print(f"   获取到 {len(products)} 个产品")

    if not products:
        print("   未获取到产品，退出")
        return

    # 2. 获取每个产品的历史净值
    print("\n2. 获取历史净值...")
    session = crawler._get_session()
    profiles = []
    total = len(products)

    for i, product in enumerate(products):
        product_code = product.get('product_code', '')

        # 获取历史净值
        nav_history = crawler.get_nav_history(product_code, session)

        # 计算指标
        profile = crawler.calculate_metrics(nav_history, product)
        if profile:
            profiles.append(profile)

        # 进度日志
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total - i - 1)
            print(f"   进度: {i+1}/{total}, 已获取: {len(profiles)}, 用时: {elapsed:.0f}秒, 预计剩余: {remaining:.0f}秒")

    print(f"\n   处理完成，共 {len(profiles)} 个有效产品")

    # 3. 转换为DataFrame
    print("\n3. 转换数据...")

    # 获取所有日期
    all_dates = set()
    for p in profiles:
        if p.nav_history:
            for nav in p.nav_history:
                all_dates.add(nav['date'])

    sorted_dates = sorted(all_dates, reverse=True)
    print(f"   涉及 {len(sorted_dates)} 个不同日期")

    # 构建数据
    rows = []
    for p in profiles:
        row = {
            '银行': p.bank_name,
            '产品代码': p.product_code,
            '产品名称': p.product_name,
            '产品类型': str(p.product_type.value) if p.product_type else '',
            '风险等级': p.risk_level,
            '其他信息': p.duration_text,
        }

        # 创建日期到净值的映射
        nav_map = {}
        if p.nav_history:
            for nav in p.nav_history:
                nav_map[nav['date']] = nav['unit_nav']

        # 添加每个日期的净值列
        for date in sorted_dates:
            row[date] = nav_map.get(date, '')

        rows.append(row)

    df = pd.DataFrame(rows)

    # 4. 保存
    print("\n4. 保存数据...")
    output_file = "citic_nav_history_full.xlsx"
    df.to_excel(output_file, index=False)
    print(f"   已保存到 {output_file}")

    # 5. 更新净值数据库Excel
    print("\n5. 更新净值数据库...")
    if HAS_NAV_DB:
        try:
            # 转换profiles为数据库需要的格式
            products_for_db = []
            for p in profiles:
                nav_history = []
                if hasattr(p, 'nav_history') and p.nav_history:
                    for nav in p.nav_history:
                        if isinstance(nav, dict):
                            nav_history.append({
                                'date': nav.get('date', ''),
                                'unit_nav': nav.get('unit_nav')
                            })

                if nav_history:
                    products_for_db.append({
                        'product_code': p.product_code,
                        'product_name': p.product_name,
                        'nav_history': nav_history
                    })

            if products_for_db:
                stats = update_nav_database('中信', products_for_db)
                print(f"   净值数据库更新完成: 新增 {len(stats.get('new_dates', []))} 个日期, "
                      f"更新 {stats.get('updated_cells', 0)} 个单元格")
        except Exception as e:
            print(f"   更新净值数据库失败: {e}")
    else:
        print("   净值数据库模块未安装，跳过数据库更新")

    # 6. 统计
    elapsed = time.time() - start_time
    print(f"\n6. 统计信息:")
    print(f"   总用时: {elapsed:.0f}秒")
    print(f"   产品数: {len(profiles)}")
    print(f"   日期数: {len(sorted_dates)}")

    by_category = {}
    for p in profiles:
        t = str(p.product_type.value) if p.product_type else '未知'
        by_category[t] = by_category.get(t, 0) + 1
    for t, c in by_category.items():
        print(f"   {t}: {c}")

if __name__ == "__main__":
    main()
