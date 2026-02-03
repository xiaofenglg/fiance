"""测试华夏理财净值解析修复"""
import sys
sys.path.insert(0, '.')

from bank_crawler import HuaxiaCrawler
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_nav_parsing():
    print("="*60)
    print("测试华夏理财净值解析修复")
    print("="*60)

    crawler = HuaxiaCrawler()

    # 获取产品列表（仅前5个测试）
    products = crawler.get_product_list()[:5]
    print(f"\n获取到 {len(products)} 个测试产品\n")

    for i, product in enumerate(products):
        product_code = product.get('id', '')
        excel_url = product.get('address', '')
        title = product.get('title', '')[:50]

        print(f"\n[{i+1}] {title}")
        print(f"    产品代码: {product_code}")
        print(f"    Excel地址: {excel_url}")

        if not excel_url:
            print("    -> 无Excel地址，跳过")
            continue

        # 下载并解析Excel
        nav_data = crawler.download_nav_excel(excel_url)

        if nav_data:
            print(f"    -> 成功获取 {len(nav_data)} 天净值数据:")
            for j, nav in enumerate(nav_data):
                date = nav.get('date', '')
                unit_nav = nav.get('unit_nav', '')
                total_nav = nav.get('total_nav', '')
                print(f"       {j+1}. {date}: 单位净值={unit_nav}, 累计净值={total_nav}")
        else:
            print("    -> 无法获取净值数据")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_nav_parsing()
