# -*- coding: utf-8 -*-
"""
费率数据库模块 — 产品全量费率的存储、查询与管理

核心功能:
1. 存储各银行产品的基础费率 (管理费、托管费、销售费、申购费、赎回费)
2. 存储费率优惠公告信息 (优惠后费率、优惠期限)
3. 计算当前生效费率 (基础费率 + 优惠覆盖)
4. 增量更新检查 (根据公告日期跳过已最新产品)

数据来源:
- 发行公告 PDF (SAMJ_TYPE=1): 基础费率结构
- 费率优惠公告 PDF (SAMJ_TYPE=4): 费率折扣信息

存储文件: 费率数据库.json
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEE_RATE_DB_PATH = os.path.join(BASE_DIR, "费率数据库.json")

_fee_rate_cache = None


def _empty_db():
    """创建空数据库结构"""
    return {"version": 1, "products": {}}


def load_fee_rate_db():
    """加载费率数据库（带内存缓存）

    Returns:
        dict: {"version": int, "products": {key: ProductFeeData}}
    """
    global _fee_rate_cache
    if _fee_rate_cache is not None:
        return _fee_rate_cache
    if not os.path.exists(FEE_RATE_DB_PATH):
        _fee_rate_cache = _empty_db()
        return _fee_rate_cache
    try:
        with open(FEE_RATE_DB_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
        if not isinstance(db, dict) or 'products' not in db:
            db = _empty_db()
        _fee_rate_cache = db
        return db
    except Exception as e:
        logger.warning(f"加载费率数据库失败: {e}")
        _fee_rate_cache = _empty_db()
        return _fee_rate_cache


def save_fee_rate_db(db=None):
    """保存费率数据库到文件"""
    global _fee_rate_cache
    if db is None:
        db = _fee_rate_cache
    if db is None:
        return
    try:
        with open(FEE_RATE_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        _fee_rate_cache = db
    except Exception as e:
        logger.error(f"保存费率数据库失败: {e}")


def invalidate_cache():
    """清除内存缓存，强制下次从文件重新加载"""
    global _fee_rate_cache
    _fee_rate_cache = None


def _make_key(bank, product_code):
    """生成产品唯一键: '银行|产品代码'"""
    return f"{bank}|{product_code}"


def update_product_fees(bank, product_code, data):
    """更新单个产品的费率信息

    Args:
        bank: 银行名称 (如 '民生')
        product_code: 产品代码 (如 'FBAE48303S')
        data: dict, 包含以下可选字段:
            product_name: str
            base_fees: {fee_type: {rate: float, unit: str}}
            base_source: str
            base_announcement_date: str (YYYYMMDD)
            discounts: [{fee_type, original_rate, discounted_rate,
                         effective_from, effective_until}]
            discount_announcement_date: str (YYYYMMDD)
    """
    db = load_fee_rate_db()
    key = _make_key(bank, product_code)

    existing = db['products'].get(key, {})

    # 合并: 新数据覆盖旧数据，但只覆盖提供的字段
    if data.get('base_fees'):
        existing['base_fees'] = data['base_fees']
    if data.get('base_source'):
        existing['base_source'] = data['base_source']
    if data.get('base_announcement_date'):
        existing['base_announcement_date'] = data['base_announcement_date']
    if data.get('product_name'):
        existing['product_name'] = data['product_name']
    if 'discounts' in data:
        existing['discounts'] = data['discounts']
    if data.get('discount_announcement_date'):
        existing['discount_announcement_date'] = data['discount_announcement_date']

    existing['last_updated'] = datetime.now().isoformat()

    db['products'][key] = existing


def get_product_fees(bank, product_code):
    """查询产品费率信息

    Returns:
        dict | None: 产品费率数据或None(未收录)
    """
    db = load_fee_rate_db()
    key = _make_key(bank, product_code)
    return db['products'].get(key)


def get_current_effective_fees(bank, product_code):
    """计算当前生效费率 (基础费率 + 优惠覆盖)

    逻辑: 以base_fees为底，用discounts中当前有效的条目覆盖对应费率

    Returns:
        dict: {fee_type: {rate: float, unit: str, discounted: bool, original_rate: float|None}}
        如果产品未收录返回空dict
    """
    product = get_product_fees(bank, product_code)
    if not product:
        return {}

    base_fees = product.get('base_fees', {})
    discounts = product.get('discounts', [])

    # 拷贝基础费率
    effective = {}
    for fee_type, info in base_fees.items():
        effective[fee_type] = {
            'rate': info.get('rate', 0.0),
            'unit': info.get('unit', ''),
            'discounted': False,
            'original_rate': None,
        }

    # 用当前有效的优惠覆盖
    today_str = datetime.now().strftime('%Y%m%d')
    for d in discounts:
        eff_from = d.get('effective_from', '')
        eff_until = d.get('effective_until')

        # 检查是否在有效期内
        if eff_from and eff_from > today_str:
            continue  # 尚未生效
        if eff_until and eff_until < today_str:
            continue  # 已过期

        fee_type = d.get('fee_type', '')
        # 模糊匹配: "固定管理费" 匹配 "固定管理费率"
        matched_key = None
        for key in effective:
            if fee_type in key or key in fee_type:
                matched_key = key
                break

        if matched_key:
            effective[matched_key]['original_rate'] = effective[matched_key]['rate']
            effective[matched_key]['rate'] = d.get('discounted_rate', effective[matched_key]['rate'])
            effective[matched_key]['discounted'] = True
        else:
            # 优惠涉及的费率类型不在基础费率中，新增
            effective[fee_type] = {
                'rate': d.get('discounted_rate', 0.0),
                'unit': '',
                'discounted': True,
                'original_rate': d.get('original_rate'),
            }

    return effective


def needs_update(bank, product_code, latest_announce_date=None, latest_discount_date=None):
    """检查产品费率数据是否需要更新

    Args:
        bank: 银行名称
        product_code: 产品代码
        latest_announce_date: 最新发行公告日期 (YYYYMMDD) 或 None
        latest_discount_date: 最新费率优惠公告日期 (YYYYMMDD) 或 None

    Returns:
        bool: True=需要更新, False=已是最新
    """
    product = get_product_fees(bank, product_code)
    if not product:
        return True  # 无记录，需要获取

    # 检查基础费率
    if not product.get('base_fees'):
        return True  # 没有基础费率数据

    # 如果有新的发行公告日期，比较
    if latest_announce_date:
        stored_date = product.get('base_announcement_date', '')
        if latest_announce_date > stored_date:
            return True

    # 如果有新的优惠公告日期，比较
    if latest_discount_date:
        stored_discount_date = product.get('discount_announcement_date', '')
        if latest_discount_date > stored_discount_date:
            return True

    return False


def get_fee_rate_summary():
    """获取费率数据库统计摘要"""
    db = load_fee_rate_db()
    products = db.get('products', {})
    total = len(products)
    has_base = sum(1 for v in products.values() if v.get('base_fees'))
    has_discount = sum(1 for v in products.values() if v.get('discounts'))

    by_bank = {}
    for key, info in products.items():
        bank = key.split('|')[0] if '|' in key else '未知'
        if bank not in by_bank:
            by_bank[bank] = {'total': 0, 'has_base': 0, 'has_discount': 0}
        by_bank[bank]['total'] += 1
        if info.get('base_fees'):
            by_bank[bank]['has_base'] += 1
        if info.get('discounts'):
            by_bank[bank]['has_discount'] += 1

    return {
        'total': total,
        'has_base': has_base,
        'has_discount': has_discount,
        'by_bank': by_bank,
    }
