# -*- coding: utf-8 -*-
"""合并所有银行的理财产品数据"""
import pandas as pd
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def find_latest_file(pattern, directory='.'):
    """查找最新的匹配文件"""
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_bank_data(file_path, bank_name):
    """加载银行数据"""
    if not file_path or not os.path.exists(file_path):
        print(f"   {bank_name}: 文件不存在 - {file_path}")
        return None

    try:
        df = pd.read_excel(file_path)
        print(f"   {bank_name}: {len(df)} 条记录, 列数: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"   {bank_name}: 加载失败 - {e}")
        return None

def main():
    print("=" * 60)
    print("合并所有银行理财产品数据")
    print("=" * 60)

    data_dir = "D:\\AI-FINANCE"

    # 1. 查找各银行的最新数据文件
    print("\n1. 查找数据文件...")

    # 民生理财
    minsheng_file = find_latest_file("民生理财_量化分析_*.xlsx", data_dir)
    if not minsheng_file:
        minsheng_file = find_latest_file("民生理财_*.xlsx", data_dir)
    print(f"   民生: {os.path.basename(minsheng_file) if minsheng_file else 'Not found'}")

    # 华夏理财 - 查找多银行文件或单独文件
    huaxia_file = find_latest_file("多银行理财_量化分析_*.xlsx", data_dir)
    if not huaxia_file:
        huaxia_file = find_latest_file("华夏理财_*.xlsx", data_dir)
    print(f"   华夏: {os.path.basename(huaxia_file) if huaxia_file else 'Not found'}")

    # 中信理财
    citic_file = find_latest_file("citic_nav_history_full.xlsx", data_dir)
    if not citic_file:
        citic_file = find_latest_file("citic_nav_history_*.xlsx", data_dir)
    print(f"   中信: {os.path.basename(citic_file) if citic_file else 'Not found'}")

    # 2. 加载数据
    print("\n2. 加载数据...")

    minsheng_df = load_bank_data(minsheng_file, "民生")
    huaxia_df = load_bank_data(huaxia_file, "华夏/多银行")
    citic_df = load_bank_data(citic_file, "中信")

    # 3. 标准化列名
    print("\n3. 标准化数据格式...")

    all_dfs = []

    # 处理民生数据
    if minsheng_df is not None:
        if '银行' not in minsheng_df.columns:
            minsheng_df['银行'] = '民生理财'
        all_dfs.append(minsheng_df)
        print(f"   民生: 已添加 {len(minsheng_df)} 条")

    # 处理华夏/多银行数据
    if huaxia_df is not None:
        if '银行' not in huaxia_df.columns:
            huaxia_df['银行'] = '华夏理财'
        all_dfs.append(huaxia_df)
        print(f"   华夏: 已添加 {len(huaxia_df)} 条")

    # 处理中信数据
    if citic_df is not None:
        if '银行' not in citic_df.columns:
            citic_df['银行'] = '信银理财'
        all_dfs.append(citic_df)
        print(f"   中信: 已添加 {len(citic_df)} 条")

    if not all_dfs:
        print("\n   没有找到任何数据文件!")
        return

    # 4. 合并数据
    print("\n4. 合并数据...")

    # 获取所有可能的日期列
    all_date_cols = set()
    non_date_cols = {'银行', '产品代码', '产品名称', '产品类型', '风险等级', '其他信息',
                     '净值', '净值日期', '累计净值', '起购金额', '成立日期', '持有期',
                     '下一开放日', '募集方式', '状态'}

    for df in all_dfs:
        for col in df.columns:
            if col not in non_date_cols:
                # 检查是否像日期列
                if '-' in str(col) or col.isdigit():
                    all_date_cols.add(col)

    # 排序日期列(最新在前)
    sorted_date_cols = sorted(all_date_cols, reverse=True)
    print(f"   日期列数: {len(sorted_date_cols)}")

    # 定义标准列顺序
    standard_cols = ['银行', '产品代码', '产品名称', '产品类型', '风险等级', '其他信息']

    # 合并所有DataFrame
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"   合并后总记录数: {len(merged_df)}")

    # 5. 整理列顺序
    print("\n5. 整理列顺序...")

    # 最终列顺序: 标准列 + 日期列
    final_cols = [c for c in standard_cols if c in merged_df.columns]
    final_cols.extend([c for c in sorted_date_cols if c in merged_df.columns])

    # 添加其他列
    other_cols = [c for c in merged_df.columns if c not in final_cols]
    final_cols.extend(other_cols)

    merged_df = merged_df[final_cols]

    # 6. 保存
    print("\n6. 保存数据...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{data_dir}\\全部银行理财_{timestamp}.xlsx"
    merged_df.to_excel(output_file, index=False)
    print(f"   已保存到: {output_file}")

    # 7. 统计
    print("\n7. 统计信息:")
    if '银行' in merged_df.columns:
        by_bank = merged_df.groupby('银行').size()
        for bank, count in by_bank.items():
            print(f"   {bank}: {count}")

    print(f"\n   总计: {len(merged_df)} 个产品")
    print(f"   日期列: {len([c for c in merged_df.columns if '-' in str(c)])} 天")

if __name__ == "__main__":
    main()
