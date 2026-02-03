"""测试CITIC爬虫"""
import sys
sys.path.insert(0, 'D:\\AI-FINANCE')

from bank_crawler import CITICCrawler
import json

def main():
    print("=" * 60)
    print("测试CITIC爬虫")
    print("=" * 60)

    crawler = CITICCrawler()

    print("\n1. 运行完整爬取 (只调用一次get_product_list)...")
    profiles = crawler.crawl()
    print(f"   获取到 {len(profiles)} 个有效产品")

    if profiles:
        print("\n2. 样例Profile:")
        for i, p in enumerate(profiles[:5]):
            print(f"\n   产品{i+1}:")
            print(f"   - 代码: {p.product_code}")
            print(f"   - 名称: {p.product_name[:40]}...")
            print(f"   - 类型: {p.product_type}")
            print(f"   - 净值: {p.latest_nav}")
            print(f"   - 日期: {p.latest_nav_date}")
            print(f"   - 风险: {p.risk_level}")
            print(f"   - 其他: {p.duration_text}")

        # 统计
        print("\n3. 统计信息:")
        by_type = {}
        for p in profiles:
            t = str(p.product_type)
            by_type[t] = by_type.get(t, 0) + 1
        for t, c in by_type.items():
            print(f"   {t}: {c}")

        # 保存结果
        print("\n4. 保存结果...")
        import pandas as pd

        data = []
        for p in profiles:
            row = {
                '银行': p.bank_name,
                '产品代码': p.product_code,
                '产品名称': p.product_name,
                '产品类型': str(p.product_type.value) if p.product_type else '',
                '风险等级': p.risk_level,
                '净值': p.latest_nav,
                '净值日期': p.latest_nav_date,
                '其他信息': p.duration_text,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel("citic_results.xlsx", index=False)
        print(f"   已保存到 citic_results.xlsx ({len(data)} 条记录)")

    else:
        print("\n   没有获取到有效产品，检查calculate_metrics...")

        # 直接测试get_product_list和calculate_metrics
        print("\n   直接测试...")
        products = crawler.get_product_list()
        print(f"   获取到 {len(products)} 个原始产品")

        if products:
            print(f"\n   样例原始数据:")
            print(json.dumps(products[0], ensure_ascii=False, indent=2))

            # 测试calculate_metrics
            profile = crawler.calculate_metrics([], products[0])
            if profile:
                print(f"\n   calculate_metrics 成功:")
                print(f"   - 代码: {profile.product_code}")
                print(f"   - 净值: {profile.current_nav}")
            else:
                print(f"\n   calculate_metrics 返回 None")

if __name__ == "__main__":
    main()
