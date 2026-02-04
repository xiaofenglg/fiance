# -*- coding: utf-8 -*-
"""
将赎回费数据库(JSON)迁移到SQLite

功能：
1. 读取赎回费数据库.json
2. 将费率数据写入aifinance.sqlite3的products表
3. SQLite自带并发安全，避免JSON损坏问题
"""

import os
import json
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DB_PATH = os.path.join(BASE_DIR, "赎回费数据库.json")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "aifinance.sqlite3")


def format_rate(value):
    """将小数费率格式化为百分比字符串"""
    if value is None:
        return ''
    return f"{value * 100:.2f}%"


def format_redemption_fee(fee_info):
    """格式化赎回费描述"""
    if fee_info is None:
        return ''
    if not fee_info.get('has_redemption_fee'):
        return '无'
    schedule = fee_info.get('fee_schedule', [])
    if not schedule:
        return '无'
    parts = []
    for entry in schedule:
        min_d = entry.get('min_days', 0)
        max_d = entry.get('max_days', 999999)
        rate = entry.get('fee_rate', 0)
        rate_pct = f"{rate * 100:.2f}%"
        if max_d >= 999999:
            parts.append(f"{min_d}天以上:{rate_pct}")
        else:
            parts.append(f"{min_d}-{max_d}天:{rate_pct}")
    return '; '.join(parts)


def migrate():
    """执行迁移 - 支持插入新产品和更新现有产品"""
    if not os.path.exists(JSON_DB_PATH):
        logger.error(f"JSON数据库不存在: {JSON_DB_PATH}")
        return 1

    # 加载JSON数据
    try:
        with open(JSON_DB_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return 1

    products = db.get('products', {})
    logger.info(f"JSON数据库中有 {len(products)} 个产品")

    # 连接SQLite
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # 确保products表存在
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_code TEXT PRIMARY KEY,
            product_name TEXT,
            bank_name TEXT,
            subscription_fee TEXT,
            purchase_fee TEXT,
            redemption_fee TEXT,
            sales_service_fee TEXT,
            custody_fee TEXT,
            management_fee TEXT,
            redemption_fee_schedule TEXT,
            has_redemption_fee INTEGER,
            fee_source TEXT
        )
    ''')

    # 添加缺失列（如果需要）
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(products)").fetchall()]

    new_cols = [
        ('redemption_fee_schedule', 'TEXT'),
        ('has_redemption_fee', 'INTEGER'),
        ('fee_source', 'TEXT'),
    ]

    for col_name, col_type in new_cols:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE products ADD COLUMN {col_name} {col_type}")
            logger.info(f"添加列: {col_name}")

    conn.commit()

    # 插入或更新数据
    inserted = 0
    updated = 0

    for key, info in products.items():
        if '|' not in key:
            continue
        bank, product_code = key.split('|', 1)

        # 检查产品是否已存在
        cursor.execute("SELECT 1 FROM products WHERE product_code = ?", (product_code,))
        exists = cursor.fetchone() is not None

        # 准备数据
        data = {
            'product_code': product_code,
            'bank_name': bank,
            'subscription_fee': format_rate(info.get('subscription_fee')),
            'purchase_fee': format_rate(info.get('purchase_fee')),
            'sales_service_fee': format_rate(info.get('sales_service_fee')),
            'custody_fee': format_rate(info.get('custody_fee')),
            'management_fee': format_rate(info.get('management_fee')),
            'redemption_fee': format_redemption_fee(info),
            'has_redemption_fee': 1 if info.get('has_redemption_fee') else 0,
            'redemption_fee_schedule': json.dumps(info.get('fee_schedule', []), ensure_ascii=False),
            'fee_source': info.get('source', ''),
        }

        if exists:
            # 更新现有产品
            sql = '''UPDATE products SET
                bank_name = ?, subscription_fee = ?, purchase_fee = ?,
                sales_service_fee = ?, custody_fee = ?, management_fee = ?,
                redemption_fee = ?, has_redemption_fee = ?,
                redemption_fee_schedule = ?, fee_source = ?
                WHERE product_code = ?'''
            cursor.execute(sql, (
                data['bank_name'], data['subscription_fee'], data['purchase_fee'],
                data['sales_service_fee'], data['custody_fee'], data['management_fee'],
                data['redemption_fee'], data['has_redemption_fee'],
                data['redemption_fee_schedule'], data['fee_source'],
                product_code
            ))
            updated += 1
        else:
            # 插入新产品
            sql = '''INSERT INTO products (
                product_code, bank_name, subscription_fee, purchase_fee,
                sales_service_fee, custody_fee, management_fee,
                redemption_fee, has_redemption_fee, redemption_fee_schedule, fee_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            cursor.execute(sql, (
                data['product_code'], data['bank_name'], data['subscription_fee'],
                data['purchase_fee'], data['sales_service_fee'], data['custody_fee'],
                data['management_fee'], data['redemption_fee'], data['has_redemption_fee'],
                data['redemption_fee_schedule'], data['fee_source']
            ))
            inserted += 1

    conn.commit()
    conn.close()

    logger.info(f"迁移完成: 新增 {inserted} 个产品, 更新 {updated} 个产品")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(migrate())
