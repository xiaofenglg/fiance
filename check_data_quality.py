# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime

xls = pd.ExcelFile('净值数据库.xlsx')

print('数据库质量报告：')
print('='*60)

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
    date_cols = [c for c in df.columns if len(str(c))==10 and '-' in str(c)]

    print(f'{sheet}:')
    print(f'  产品数: {len(df)}')
    print(f'  日期数: {len(date_cols)}')

    if date_cols:
        start_date = min(date_cols)
        end_date = max(date_cols)
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        print(f'  起始日: {start_date}')
        print(f'  结束日: {end_date}')
        print(f'  跨度: {days}天 ({days/30:.1f}个月)')

        # 建议的分析参数
        if days >= 180:
            print(f'  数据质量: [优秀] 可用完整策略')
            print(f'  建议回测天数: 90天')
            print(f'  建议实时窗口: 3天')
        elif days >= 60:
            print(f'  数据质量: [良好] 可用简化策略')
            print(f'  建议回测天数: {days//2}天')
            print(f'  建议实时窗口: 2天')
        else:
            print(f'  数据质量: [不足] 仅供参考')
            print(f'  建议回测天数: {max(7, days-7)}天')
            print(f'  建议实时窗口: 1天')

    print()
