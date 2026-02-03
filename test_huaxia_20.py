"""测试华夏理财 - 仅20条数据"""
import sys
sys.path.insert(0, '.')

from bank_crawler import HuaxiaCrawler, MultiBankCrawler, BankType, ProductProfile
from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    print("="*60)
    print("华夏理财测试 - 20条数据")
    print("="*60)

    crawler = HuaxiaCrawler()

    # 只获取前20个产品
    all_products = crawler.get_product_list()
    products = all_products[:20]
    print(f"\n测试产品数: {len(products)}")

    profiles = []
    for i, product in enumerate(products):
        product_code = product.get('id', '')
        excel_url = product.get('address', '')

        if not excel_url:
            continue

        # 下载并解析Excel
        nav_data = crawler.download_nav_excel(excel_url)

        if nav_data and len(nav_data) >= 2:
            profile = crawler.calculate_metrics(nav_data, product)
            if profile:
                profile.score = crawler.calculate_score(profile)
                profile.signal = crawler.generate_signal(profile)
                profiles.append(profile)
                print(f"[{i+1}] {product_code}: {len(nav_data)}天净值, 1周={profile.metrics.return_1w}%, 1月={profile.metrics.return_1m}%")

    print(f"\n有效产品数: {len(profiles)}")

    # 使用MultiBankCrawler保存结果
    manager = MultiBankCrawler()
    manager.all_profiles = profiles
    filename = f'华夏理财_测试20_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

    # 手动保存以使用正确的文件名
    data = []
    for p in profiles:
        row = {
            '银行': p.bank_name,
            '产品名称': p.product_name,
            '产品代码': p.product_code,
            '风险等级': p.risk_level,
            '风险类型': p.risk_category.value,
            '产品类型': p.product_type.value,
            '可购买': '是' if p.is_buyable else '否',
            '申购状态': p.status.value,
            '持有期限': p.duration_text,
            '期限天数': p.duration_days,
            '最新净值日期': p.latest_nav_date,
            '净值天数': len(p.nav_history) if p.nav_history else 0,
            '最新净值': p.latest_nav,
            '近1周年化(%)': p.metrics.return_1w,
            '近1月年化(%)': p.metrics.return_1m,
            '近3月年化(%)': p.metrics.return_3m,
            '近6月年化(%)': p.metrics.return_6m,
            '波动率(%)': p.metrics.volatility,
            '最大回撤(%)': p.metrics.max_drawdown,
            '夏普比率': p.metrics.sharpe_ratio,
            '卡玛比率': p.metrics.calmar_ratio,
            '综合评分': p.score,
            '交易信号': p.signal.value,
            '业绩基准': p.benchmark
        }

        # 添加历史净值 - 用日期作为列名
        if p.nav_history:
            for j, nav in enumerate(p.nav_history[:15]):
                date_str = nav.get('date', '')
                unit_nav = nav.get('unit_nav')
                if unit_nav is not None:
                    unit_nav = round(float(unit_nav), 4)
                if date_str:
                    row[date_str] = unit_nav

        data.append(row)

    df = pd.DataFrame(data)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='华夏理财', index=False)

        # 设置净值列格式（包括日期列名）
        try:
            import re
            ws = writer.sheets['华夏理财']
            for col_idx, col_cell in enumerate(ws[1], 1):
                col_name = str(col_cell.value) if col_cell.value else ''
                is_nav_col = ('净值' in col_name and '日期' not in col_name)
                is_date_col = bool(re.match(r'^\d{4}-\d{2}-\d{2}$', col_name))
                if is_nav_col or is_date_col:
                    for row_idx in range(2, ws.max_row + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        if cell.value is not None and isinstance(cell.value, (int, float)):
                            cell.number_format = '0.0000'
        except Exception as e:
            print(f"格式设置: {e}")

    print(f"\n已保存: {filename}")
    return filename

if __name__ == "__main__":
    main()
