# -*- coding: utf-8 -*-
"""
导入现有净值数据到统一数据库

将各银行的净值历史JSON文件导入到 nav_database 中
"""

import json
import os
from pathlib import Path
from nav_database import get_database


def import_json_nav_files():
    """导入现有的JSON净值历史文件"""
    db = get_database()

    # 查找所有净值历史JSON文件
    nav_files = {
        'huaxia_nav_history.json': '华夏银行',
        'icbc_nav_history.json': '工商银行',
        'ceb_nav_history.json': '光大银行',
        'cmbc_nav_history.json': '民生银行',
        'citic_nav_history.json': '中信银行',
    }

    base_dir = Path("D:/AI-FINANCE")
    total_imported = 0

    for filename, bank_name in nav_files.items():
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"[跳过] {filename} 不存在")
            continue

        print(f"\n[导入] {bank_name} ({filename})")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nav_data = json.load(f)

            product_count = 0
            nav_count = 0

            for product_code, product_data in nav_data.items():
                # 检测数据格式
                # 格式1: {product_code: [nav_entries...]} (华夏)
                # 格式2: {product_code: {name: ..., nav_history: [...]}} (工商/光大)
                if isinstance(product_data, list):
                    # 格式1: 直接是净值列表
                    history = product_data
                    product_name = product_code
                elif isinstance(product_data, dict):
                    # 格式2: 包含name和nav_history
                    history = product_data.get('nav_history', [])
                    product_name = product_data.get('name', product_code)
                else:
                    continue

                # 添加产品信息
                db.add_product(bank_name, product_code, {
                    'name': product_name,
                    'source': filename,
                })

                # 转换净值历史格式
                nav_history = []
                for item in history:
                    if not isinstance(item, dict):
                        continue
                    nav_entry = {
                        'date': item.get('date', item.get('nav_date', '')),
                        'nav': item.get('nav', item.get('unit_nav', '')),
                        'acc_nav': item.get('acc_nav', item.get('accumulated_nav', item.get('total_nav', ''))),
                    }
                    # 过滤空数据
                    if nav_entry['date']:
                        nav_history.append(nav_entry)

                # 增量添加净值历史
                if nav_history:
                    new_count = db.add_nav_history(bank_name, product_code, nav_history)
                    nav_count += new_count

                product_count += 1

            print(f"  产品数: {product_count}, 新增净值记录: {nav_count}")
            total_imported += nav_count

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 更新元数据
    db.update_metadata()

    print(f"\n=== 导入完成 ===")
    print(f"总计新增净值记录: {total_imported}")

    # 显示统计
    stats = db.get_statistics()
    print(f"数据库统计:")
    print(f"  总产品数: {stats['total_products']}")
    print(f"  总净值记录: {stats['total_nav_records']}")
    print(f"  各银行:")
    for bank, info in stats['banks'].items():
        print(f"    {bank}: {info['products']} 产品, {info['nav_records']} 条净值")

    return db


def import_from_crawler_results(excel_file: str = None):
    """从爬虫结果Excel导入"""
    import pandas as pd
    from pathlib import Path
    import glob

    db = get_database()

    # 如果没有指定文件，找最新的
    if excel_file is None:
        files = glob.glob("D:/AI-FINANCE/多银行理财产品_*.xlsx")
        if not files:
            print("未找到爬虫结果文件")
            return
        excel_file = max(files, key=os.path.getmtime)

    print(f"导入文件: {excel_file}")

    # 读取Excel
    df = pd.read_excel(excel_file)
    print(f"共 {len(df)} 个产品")

    # 识别净值列
    nav_cols = [c for c in df.columns if '净值' in str(c) and '日期' in str(c)]

    new_products = 0
    new_nav_records = 0

    for _, row in df.iterrows():
        bank = row.get('银行', '未知')
        code = row.get('产品代码', '')
        if not code:
            continue

        # 添加产品
        product_info = {
            'name': row.get('产品名称', ''),
            'risk_level': row.get('风险等级', ''),
            'product_type': str(row.get('产品类型', '')),
        }
        db.add_product(bank, code, product_info)
        new_products += 1

        # 提取净值历史（从列名中）
        nav_history = []
        for col in df.columns:
            if '净值_' in str(col):
                try:
                    # 列名格式: 净值_2024-01-15
                    date_part = str(col).split('_')[1]
                    nav_value = row.get(col)
                    if pd.notna(nav_value):
                        nav_history.append({
                            'date': date_part,
                            'nav': float(nav_value),
                        })
                except:
                    pass

        if nav_history:
            count = db.add_nav_history(bank, code, nav_history)
            new_nav_records += count

    db.update_metadata()

    print(f"导入完成: {new_products} 个产品, {new_nav_records} 条新净值记录")

    return db


if __name__ == "__main__":
    print("=" * 60)
    print("净值数据导入工具")
    print("=" * 60)

    # 1. 先导入现有JSON文件
    import_json_nav_files()

    # 2. 可选：导入Excel结果
    # import_from_crawler_results()
