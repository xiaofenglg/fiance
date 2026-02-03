# -*- coding: utf-8 -*-
"""中信银行净值数据全面分析"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'D:\AI-FINANCE')

import pandas as pd
from datetime import datetime
from nav_db_excel import NAVDatabaseExcel

print("=" * 80)
print("          中信银行 净值数据库 全面分析报告")
print("=" * 80)

# 加载数据库
db = NAVDatabaseExcel()
print(f"\n数据库文件: {db.db_file}")
print(f"已加载的Sheet: {list(db.data.keys())}")

# ============================================================
# 第一部分：中信银行日期详细分析
# ============================================================
print("\n" + "=" * 80)
print("【第一部分】中信银行 日期列详细分析")
print("=" * 80)

citic_sheet = "中信银行"
if citic_sheet in db.data:
    df_citic = db.data[citic_sheet]
    # 获取日期列
    date_cols = sorted([c for c in df_citic.columns if db._is_date_column(c)])

    print(f"\n总产品数: {len(df_citic)}")
    print(f"总日期列数: {len(date_cols)}")

    print(f"\n--- 所有 {len(date_cols)} 个日期列（排序后）---")
    print(f"{'序号':>4}  {'日期':>12}  {'有数据的产品数':>14}  {'数据覆盖率':>10}  {'与上一日期间隔':>14}")
    print("-" * 70)

    prev_date = None
    for i, d in enumerate(date_cols, 1):
        # 计算有数据的产品数
        col_data = df_citic[d]
        non_null_count = col_data.apply(lambda x: pd.notna(x) and str(x).strip() != '').sum()
        coverage = non_null_count / len(df_citic) * 100 if len(df_citic) > 0 else 0

        # 计算间隔
        gap_str = ""
        if prev_date is not None:
            d1 = datetime.strptime(prev_date, "%Y-%m-%d")
            d2 = datetime.strptime(d, "%Y-%m-%d")
            gap = (d2 - d1).days
            gap_str = f"{gap}天"
        else:
            gap_str = "-"

        prev_date = d
        print(f"{i:>4}  {d:>12}  {non_null_count:>14}  {coverage:>9.1f}%  {gap_str:>14}")

    # 间隔统计
    if len(date_cols) >= 2:
        gaps = []
        for i in range(1, len(date_cols)):
            d1 = datetime.strptime(date_cols[i-1], "%Y-%m-%d")
            d2 = datetime.strptime(date_cols[i], "%Y-%m-%d")
            gaps.append((d2 - d1).days)

        print(f"\n--- 间隔统计 ---")
        print(f"最小间隔: {min(gaps)}天")
        print(f"最大间隔: {max(gaps)}天")
        print(f"平均间隔: {sum(gaps)/len(gaps):.1f}天")
        print(f"中位数间隔: {sorted(gaps)[len(gaps)//2]}天")

        # 间隔分布
        from collections import Counter
        gap_counter = Counter(gaps)
        print(f"\n间隔分布:")
        for g in sorted(gap_counter.keys()):
            print(f"  {g}天间隔: {gap_counter[g]}次")
else:
    print(f"[错误] 未找到 '{citic_sheet}' Sheet")

# ============================================================
# 第二部分：民生银行和华夏银行对比
# ============================================================
print("\n" + "=" * 80)
print("【第二部分】对比: 民生银行 和 华夏银行 日期覆盖摘要")
print("=" * 80)

for bank in ["民生银行", "华夏银行"]:
    if bank in db.data:
        df_bank = db.data[bank]
        bank_dates = sorted([c for c in df_bank.columns if db._is_date_column(c)])

        print(f"\n--- {bank} ---")
        print(f"  产品总数: {len(df_bank)}")
        print(f"  日期总数: {len(bank_dates)}")
        if bank_dates:
            print(f"  最早日期: {bank_dates[0]}")
            print(f"  最新日期: {bank_dates[-1]}")

            if len(bank_dates) >= 2:
                gaps = []
                for i in range(1, len(bank_dates)):
                    d1 = datetime.strptime(bank_dates[i-1], "%Y-%m-%d")
                    d2 = datetime.strptime(bank_dates[i], "%Y-%m-%d")
                    gaps.append((d2 - d1).days)
                avg_gap = sum(gaps) / len(gaps)
                print(f"  平均日期间隔: {avg_gap:.1f}天")
                print(f"  最小间隔: {min(gaps)}天, 最大间隔: {max(gaps)}天")

                # 日期跨度
                span_days = (datetime.strptime(bank_dates[-1], "%Y-%m-%d") - datetime.strptime(bank_dates[0], "%Y-%m-%d")).days
                print(f"  日期跨度: {span_days}天 ({span_days/30:.1f}个月)")
    else:
        print(f"\n--- {bank} ---")
        print(f"  [未找到此Sheet]")

# ============================================================
# 第三部分：高成功率产品库中的8个中信产品
# ============================================================
print("\n" + "=" * 80)
print("【第三部分】高成功率产品库中的 8 个中信产品详细分析")
print("=" * 80)

with open(r'D:\AI-FINANCE\高成功率产品库_缓存.json', 'r', encoding='utf-8') as f:
    lib_data = json.load(f)

citic_products = [p for p in lib_data['产品库'] if p['银行'] == '中信银行']
print(f"\n中信产品数量: {len(citic_products)}")

if citic_sheet in db.data:
    df_citic = db.data[citic_sheet]

    print(f"\n{'序号':>4}  {'产品代码':>12}  {'产品名称':<35}  {'NAV点数':>8}  {'首日期':>12}  {'末日期':>12}  {'区间收益率':>10}")
    print("-" * 120)

    for i, prod in enumerate(citic_products, 1):
        code = prod['产品代码']
        name = prod['产品名称']

        # 在DataFrame中查找此产品
        nav_count = 0
        first_date = ""
        last_date = ""
        total_return = ""

        # 查找产品行
        found = False
        for idx in df_citic.index:
            if idx[0] == code:
                found = True
                # 获取所有有效净值
                nav_points = []
                for col in date_cols:
                    val = df_citic.loc[idx, col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    if pd.notna(val) and str(val).strip() != '':
                        try:
                            nav_points.append((col, float(val)))
                        except:
                            pass

                nav_count = len(nav_points)
                if nav_points:
                    nav_points.sort(key=lambda x: x[0])
                    first_date = nav_points[0][0]
                    last_date = nav_points[-1][0]
                    first_nav = nav_points[0][1]
                    last_nav = nav_points[-1][1]
                    if first_nav > 0:
                        ret = (last_nav / first_nav - 1) * 100
                        total_return = f"{ret:+.4f}%"
                break

        if not found:
            total_return = "未找到"

        # 截断名称以便显示
        display_name = name[:30] + "..." if len(name) > 30 else name
        print(f"{i:>4}  {code:>12}  {display_name:<35}  {nav_count:>8}  {first_date:>12}  {last_date:>12}  {total_return:>10}")

    # 额外：显示每个产品的净值序列概览
    print(f"\n--- 各产品净值序列详情 ---")
    for prod in citic_products:
        code = prod['产品代码']
        name = prod['产品名称']

        for idx in df_citic.index:
            if idx[0] == code:
                nav_points = []
                for col in date_cols:
                    val = df_citic.loc[idx, col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    if pd.notna(val) and str(val).strip() != '':
                        try:
                            nav_points.append((col, float(val)))
                        except:
                            pass

                if nav_points:
                    nav_points.sort(key=lambda x: x[0])
                    print(f"\n  {code} ({name[:25]}...)")
                    print(f"    数据点数: {len(nav_points)}")
                    print(f"    日期范围: {nav_points[0][0]} ~ {nav_points[-1][0]}")
                    # 显示前3个和后3个净值
                    print(f"    前3个净值: ", end="")
                    for d, v in nav_points[:3]:
                        print(f"{d}={v:.6f}  ", end="")
                    print()
                    if len(nav_points) > 3:
                        print(f"    后3个净值: ", end="")
                        for d, v in nav_points[-3:]:
                            print(f"{d}={v:.6f}  ", end="")
                        print()

                    # 年化收益计算
                    first_nav = nav_points[0][1]
                    last_nav = nav_points[-1][1]
                    d1 = datetime.strptime(nav_points[0][0], "%Y-%m-%d")
                    d2 = datetime.strptime(nav_points[-1][0], "%Y-%m-%d")
                    days = (d2 - d1).days
                    if days > 0 and first_nav > 0:
                        total_ret = last_nav / first_nav - 1
                        annual_ret = (1 + total_ret) ** (365 / days) - 1
                        print(f"    区间收益率: {total_ret*100:+.4f}%  (跨度{days}天, 年化约{annual_ret*100:.2f}%)")
                break

# ============================================================
# 第四部分：中信产品 日开 vs 非日开 分类
# ============================================================
print("\n" + "=" * 80)
print("【第四部分】中信产品 按流动性/产品名称 分类统计")
print("=" * 80)

if citic_sheet in db.data:
    df_citic = db.data[citic_sheet]

    daily_open_count = 0
    non_daily_count = 0
    daily_open_products = []
    non_daily_products = []

    for idx in df_citic.index:
        code = idx[0]
        name = idx[1]

        # 通过产品名称判断是否日开
        is_daily = "日开" in str(name)

        if is_daily:
            daily_open_count += 1
            daily_open_products.append((code, name))
        else:
            non_daily_count += 1
            non_daily_products.append((code, name))

    print(f"\n中信银行产品总数: {len(df_citic)}")
    print(f"日开型产品数: {daily_open_count} ({daily_open_count/len(df_citic)*100:.1f}%)")
    print(f"非日开型产品数: {non_daily_count} ({non_daily_count/len(df_citic)*100:.1f}%)")

    # 非日开产品中，进一步分类
    if non_daily_products:
        # 尝试分析持有期类型
        from collections import Counter
        type_counter = Counter()
        for code, name in non_daily_products:
            name_str = str(name)
            if "7天" in name_str or "周" in name_str:
                type_counter["7天持有期"] += 1
            elif "14天" in name_str or "两周" in name_str:
                type_counter["14天持有期"] += 1
            elif "30天" in name_str or "月" in name_str:
                type_counter["30天/月持有期"] += 1
            elif "90天" in name_str or "季" in name_str:
                type_counter["90天/季持有期"] += 1
            elif "封闭" in name_str:
                type_counter["封闭式"] += 1
            else:
                type_counter["其他非日开"] += 1

        print(f"\n  非日开产品分类:")
        for t, c in type_counter.most_common():
            print(f"    {t}: {c}个")

    # 示例：列出部分非日开产品
    if non_daily_products:
        print(f"\n  非日开产品示例（前10个）:")
        for code, name in non_daily_products[:10]:
            print(f"    {code}: {name}")

# ============================================================
# 第五部分：中信产品 按NAV数据点数排名 Top 20
# ============================================================
print("\n" + "=" * 80)
print("【第五部分】中信产品 按NAV数据点数排名 Top 20")
print("=" * 80)

if citic_sheet in db.data:
    df_citic = db.data[citic_sheet]
    date_cols = sorted([c for c in df_citic.columns if db._is_date_column(c)])

    product_nav_counts = []

    for idx in df_citic.index:
        code = idx[0]
        name = idx[1]

        # 计算有效净值数
        nav_count = 0
        first_date = None
        last_date = None
        first_nav = None
        last_nav = None

        for col in date_cols:
            val = df_citic.loc[idx, col]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if pd.notna(val) and str(val).strip() != '':
                try:
                    nav_val = float(val)
                    nav_count += 1
                    if first_date is None:
                        first_date = col
                        first_nav = nav_val
                    last_date = col
                    last_nav = nav_val
                except:
                    pass

        ret_str = ""
        if first_nav and last_nav and first_nav > 0 and first_date != last_date:
            ret = (last_nav / first_nav - 1) * 100
            ret_str = f"{ret:+.4f}%"

        product_nav_counts.append((code, name, nav_count, first_date or "", last_date or "", ret_str))

    # 按NAV数据点数降序排序
    product_nav_counts.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'排名':>4}  {'产品代码':>12}  {'产品名称':<40}  {'NAV点数':>8}  {'首日期':>12}  {'末日期':>12}  {'区间收益':>10}")
    print("-" * 130)

    for rank, (code, name, count, fd, ld, ret) in enumerate(product_nav_counts[:20], 1):
        display_name = name[:35] + "..." if len(name) > 35 else name
        print(f"{rank:>4}  {code:>12}  {display_name:<40}  {count:>8}  {fd:>12}  {ld:>12}  {ret:>10}")

    # 数据点数分布统计
    print(f"\n--- NAV数据点数分布统计 ---")
    counts = [x[2] for x in product_nav_counts]
    print(f"  最多数据点: {max(counts)}   最少数据点: {min(counts)}")
    print(f"  平均数据点: {sum(counts)/len(counts):.1f}")

    # 分段统计
    brackets = [(0, 0), (1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    for lo, hi in brackets:
        cnt = sum(1 for c in counts if lo <= c <= hi)
        if cnt > 0:
            print(f"  {lo}-{hi}个数据点: {cnt}个产品")
    above = sum(1 for c in counts if c > 50)
    if above > 0:
        print(f"  50+个数据点: {above}个产品")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
