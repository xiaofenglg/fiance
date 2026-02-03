# -*- coding: utf-8 -*-
"""检查数据结构"""
import pandas as pd
import json

# 读取Excel
df = pd.read_excel('D:/AI-FINANCE/多银行理财_量化分析_20260121_014815.xlsx', sheet_name=0)

print("=" * 60)
print("数据概况")
print("=" * 60)
print(f"总行数: {len(df)}")
print(f"\n列名:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print(f"\n各银行产品数:")
bank_col = df.columns[0]  # 第一列是银行
print(df[bank_col].value_counts())

print(f"\n样例数据 (第一行):")
for col in df.columns:
    val = df[col].iloc[0]
    print(f"  {col}: {val}")
