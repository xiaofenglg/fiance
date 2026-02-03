# -*- coding: utf-8 -*-
"""测试更新后的CITIC爬虫 - 包含历史净值"""
import sys
sys.path.insert(0, 'D:\\AI-FINANCE')
sys.stdout.reconfigure(encoding='utf-8')

from bank_crawler import CITICCrawler
import pandas as pd
import json

def main():
    print("=" * 60)
    print("测试更新后的CITIC爬虫 - 包含历史净值")
    print("=" * 60)

    crawler = CITICCrawler()

    # 1. 测试获取产品列表
    print("\n1. 测试获取产品列表...")
    products = crawler.get_product_list()
    print(f"   获取到 {len(products)} 个产品")

    if products:
        print(f"\n   样例产品:")
        p = products[0]
        print(f"   - 代码: {p.get('product_code')}")
        print(f"   - 名称: {p.get('product_name')[:40]}...")
        print(f"   - 类别: {p.get('category')}")
        print(f"   - 净值: {p.get('nav')}")
        print(f"   - 日期: {p.get('nav_date')}")

    # 2. 测试获取历史净值
    print("\n2. 测试获取历史净值...")
    if products:
        test_code = products[0].get('product_code')
        print(f"   产品代码: {test_code}")

        nav_history = crawler.get_nav_history(test_code)
        print(f"   获取到 {len(nav_history)} 条历史净值")

        if nav_history:
            print(f"\n   历史净值(近10条):")
            for i, nav in enumerate(nav_history[:10]):
                print(f"      {nav['date']}: 单位={nav['unit_nav']}, 累计={nav['total_nav']}")

    # 3. 测试完整爬取(只取前20个产品测试)
    print("\n3. 测试完整爬取(前20个产品)...")

    # 限制产品数量进行测试
    test_products = products[:20] if products else []
    profiles = []

    session = crawler._get_session()
    for i, product in enumerate(test_products):
        product_code = product.get('product_code', '')
        nav_history = crawler.get_nav_history(product_code, session)
        profile = crawler.calculate_metrics(nav_history, product)
        if profile:
            profiles.append(profile)
        print(f"   处理 {i+1}/20: {product_code} - 净值历史: {len(nav_history)}条")

    print(f"\n   获取到 {len(profiles)} 个有效产品Profile")

    # 4. 转换为DataFrame并保存
    if profiles:
        print("\n4. 保存到Excel...")

        # 获取所有日期
        all_dates = set()
        for p in profiles:
            if p.nav_history:
                for nav in p.nav_history:
                    all_dates.add(nav['date'])

        # 排序日期(最新在前)
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
            for date in sorted_dates[:10]:  # 只取最近10天
                row[date] = nav_map.get(date, '')

            rows.append(row)

        df = pd.DataFrame(rows)

        # 保存
        output_file = "citic_nav_history_test.xlsx"
        df.to_excel(output_file, index=False)
        print(f"   已保存到 {output_file}")

        # 显示预览
        print("\n5. 数据预览:")
        print(df.to_string(index=False))

    # 5. 统计
    print("\n6. 统计信息:")
    if products:
        by_category = {}
        for p in products:
            cat = p.get('category', '未知')
            by_category[cat] = by_category.get(cat, 0) + 1
        for cat, count in by_category.items():
            print(f"   {cat}: {count}")

if __name__ == "__main__":
    main()
