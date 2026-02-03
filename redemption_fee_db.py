# -*- coding: utf-8 -*-
"""
赎回费数据库模块 — 产品赎回费用的存储、查询与管理

核心功能:
1. 存储各银行产品的赎回费率阶梯
2. 根据持有天数查询适用费率
3. 计算赎回费用金额
4. 从产品名称正则提取费用信息

数据来源:
- api_detail: 银行API产品详情接口
- name_parse: 产品名称正则解析
- manual: 手工录入

存储文件: 赎回费数据库.json
"""

import os
import re
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEE_DB_PATH = os.path.join(BASE_DIR, "赎回费数据库.json")

# 费用相关正则模式
_FEE_PATTERNS = [
    # "(180天内收取赎回费)" — 中信常见格式
    re.compile(r'(\d+)\s*天内[收取]*赎回费'),
    # "(持有180天内赎回费0.5%)"
    re.compile(r'持有\s*(\d+)\s*天内.*?赎回费\s*([\d.]+)\s*%'),
    # "(赎回费：7天内0.5%)"
    re.compile(r'赎回费[：:]\s*(\d+)\s*天内\s*([\d.]+)\s*%'),
    # "7天内赎回费率0.5%"
    re.compile(r'(\d+)\s*天内赎回费率?\s*([\d.]+)\s*%'),
    # "(短期赎回费)" — 仅标记有费但无具体费率
    re.compile(r'短期赎回费'),
    # "(收取赎回费)"
    re.compile(r'收取赎回费'),
]

# 常见费率阶梯模板 (用于仅知道有费但不知具体费率的情况)
DEFAULT_FEE_SCHEDULE_7D = [
    {'min_days': 0, 'max_days': 7, 'fee_rate': 0.005},
    {'min_days': 7, 'max_days': 999999, 'fee_rate': 0.0},
]

DEFAULT_FEE_SCHEDULE_30D = [
    {'min_days': 0, 'max_days': 7, 'fee_rate': 0.015},
    {'min_days': 7, 'max_days': 30, 'fee_rate': 0.005},
    {'min_days': 30, 'max_days': 999999, 'fee_rate': 0.0},
]

DEFAULT_FEE_SCHEDULE_180D = [
    {'min_days': 0, 'max_days': 7, 'fee_rate': 0.015},
    {'min_days': 7, 'max_days': 30, 'fee_rate': 0.005},
    {'min_days': 30, 'max_days': 180, 'fee_rate': 0.002},
    {'min_days': 180, 'max_days': 999999, 'fee_rate': 0.0},
]


def _empty_db():
    """创建空数据库结构"""
    return {"version": 1, "products": {}}


_fee_db_cache = None


def load_fee_db():
    """加载赎回费数据库

    Returns:
        dict: {"version": int, "products": {key: ProductFeeInfo}}
    """
    global _fee_db_cache
    if _fee_db_cache is not None:
        return _fee_db_cache
    if not os.path.exists(FEE_DB_PATH):
        _fee_db_cache = _empty_db()
        return _fee_db_cache
    try:
        with open(FEE_DB_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
        if not isinstance(db, dict) or 'products' not in db:
            db = _empty_db()
        _fee_db_cache = db
        return db
    except Exception as e:
        logger.warning(f"加载赎回费数据库失败: {e}")
        _fee_db_cache = _empty_db()
        return _fee_db_cache


def save_fee_db(db=None):
    """保存赎回费数据库到文件"""
    global _fee_db_cache
    if db is None:
        db = _fee_db_cache
    if db is None:
        return
    try:
        with open(FEE_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        _fee_db_cache = db
    except Exception as e:
        logger.error(f"保存赎回费数据库失败: {e}")


def invalidate_cache():
    """清除内存缓存，强制下次从文件重新加载"""
    global _fee_db_cache
    _fee_db_cache = None


def _make_key(bank, product_code):
    """生成产品唯一键: '银行|产品代码'"""
    return f"{bank}|{product_code}"


def get_fee_info(bank, product_code):
    """获取产品费用完整信息

    Returns:
        dict | None: ProductFeeInfo 或 None(未收录)
    """
    db = load_fee_db()
    key = _make_key(bank, product_code)
    return db['products'].get(key)


def get_all_fees(bank, product_code):
    """获取产品全部费率信息（含赎回费阶梯+各项费率）

    Returns:
        dict | None: {
            'has_redemption_fee': bool,
            'fee_schedule': [...],
            'subscription_fee': float | None,
            'purchase_fee': float | None,
            'sales_service_fee': float | None,
            'custody_fee': float | None,
            'management_fee': float | None,
        }
    """
    return get_fee_info(bank, product_code)


def format_redemption_fee(fee_info):
    """将赎回费阶梯格式化为可读字符串

    Examples:
        "0-180天:1.00%; 180天以上:0.00%"
        "无"
    """
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


def format_rate(value):
    """将小数费率格式化为百分比字符串，None返回空字符串

    Examples:
        0.004 -> "0.40%"
        0.0 -> "0.00%"
        None -> ""
    """
    if value is None:
        return ''
    return f"{value * 100:.2f}%"


def get_fee_rate(bank, product_code, holding_days):
    """查询指定持有天数的赎回费率

    Args:
        bank: 银行名称 (如 '信银理财', '民生银行')
        product_code: 产品代码
        holding_days: 持有日历天数

    Returns:
        float: 费率 (0.005 表示 0.5%), 无费或未知返回 0.0
    """
    info = get_fee_info(bank, product_code)
    if info is None:
        return 0.0
    if info.get('has_redemption_fee') is False:
        return 0.0
    if info.get('has_redemption_fee') is None:
        return 0.0  # 未知状态按无费处理

    schedule = info.get('fee_schedule', [])
    if not schedule:
        return 0.0

    for entry in schedule:
        if entry['min_days'] <= holding_days < entry['max_days']:
            return entry['fee_rate']

    # 超出所有阶梯范围，返回最后一个阶梯的费率
    return schedule[-1].get('fee_rate', 0.0)


def calculate_fee_cost(bank, product_code, holding_days, amount):
    """计算赎回费用金额

    Args:
        bank: 银行名称
        product_code: 产品代码
        holding_days: 持有日历天数
        amount: 赎回金额(含盈亏)

    Returns:
        float: 赎回费用金额
    """
    rate = get_fee_rate(bank, product_code, holding_days)
    return amount * rate


def update_fee_info(bank, product_code, fee_info):
    """更新单个产品的费用信息

    Args:
        bank: 银行名称
        product_code: 产品代码
        fee_info: dict, 包含:
            has_redemption_fee: bool | None
            fee_schedule: list[{min_days, max_days, fee_rate}]
            fee_description: str
            source: str ('api_detail'|'name_parse'|'prospectus'|'manual')
            subscription_fee: float | None  (认购费率，如 0.0)
            purchase_fee: float | None      (申购费率，如 0.0)
            sales_service_fee: float | None (销售服务费率，如 0.004)
            custody_fee: float | None       (托管费率，如 0.0003)
            management_fee: float | None    (管理费率，如 0.004)
    """
    # 数据源优先级: manual > prospectus > api_detail > name_parse
    SOURCE_PRIORITY = {'manual': 4, 'prospectus': 3, 'api_detail': 2, 'name_parse': 1}

    db = load_fee_db()
    key = _make_key(bank, product_code)

    existing = db['products'].get(key)
    if existing:
        existing_priority = SOURCE_PRIORITY.get(existing.get('source', ''), 0)
        new_priority = SOURCE_PRIORITY.get(fee_info.get('source', ''), 0)
        if new_priority < existing_priority:
            return  # 低优先级数据不覆盖高优先级

    fee_info['last_updated'] = datetime.now().isoformat()
    db['products'][key] = fee_info
    _fee_db_cache.update(db)  # 更新缓存


def has_fee_data(bank, product_code):
    """检查产品是否已有费用数据"""
    db = load_fee_db()
    key = _make_key(bank, product_code)
    return key in db['products']


def parse_fee_from_name(product_name):
    """从产品名称正则提取赎回费信息

    Args:
        product_name: 产品名称字符串

    Returns:
        dict | None: ProductFeeInfo 或 None(未匹配到费用信息)
    """
    if not product_name:
        return None

    name = str(product_name)

    # 模式1: "持有N天内赎回费X%" 或 "N天内赎回费率X%"
    for pat in _FEE_PATTERNS[1:4]:  # 有具体费率的模式
        m = pat.search(name)
        if m:
            groups = m.groups()
            if len(groups) >= 2:
                days = int(groups[0])
                rate = float(groups[1]) / 100
                return {
                    'has_redemption_fee': True,
                    'fee_schedule': [
                        {'min_days': 0, 'max_days': days, 'fee_rate': rate},
                        {'min_days': days, 'max_days': 999999, 'fee_rate': 0.0},
                    ],
                    'fee_description': f'{days}天内{groups[1]}%',
                    'source': 'name_parse',
                }

    # 模式2: "N天内收取赎回费" — 有费但无具体费率
    m = _FEE_PATTERNS[0].search(name)
    if m:
        days = int(m.group(1))
        # 根据天数选择默认费率模板
        if days <= 7:
            schedule = DEFAULT_FEE_SCHEDULE_7D
        elif days <= 30:
            schedule = DEFAULT_FEE_SCHEDULE_30D
        else:
            schedule = DEFAULT_FEE_SCHEDULE_180D
        return {
            'has_redemption_fee': True,
            'fee_schedule': schedule,
            'fee_description': f'{days}天内收取赎回费(默认费率)',
            'source': 'name_parse',
        }

    # 模式3: "短期赎回费" / "收取赎回费" — 仅标记有费
    for pat in _FEE_PATTERNS[4:]:
        if pat.search(name):
            return {
                'has_redemption_fee': True,
                'fee_schedule': DEFAULT_FEE_SCHEDULE_7D,
                'fee_description': '收取赎回费(默认7天0.5%)',
                'source': 'name_parse',
            }

    return None


def batch_parse_names(products):
    """批量从产品名称解析费用信息

    Args:
        products: list of dict, 每个包含 'bank', 'product_code', 'product_name'

    Returns:
        int: 新发现有费产品数
    """
    found = 0
    for p in products:
        bank = p.get('bank', '')
        code = p.get('product_code', '')
        name = p.get('product_name', '')
        if not bank or not code:
            continue
        # 已有数据的跳过
        if has_fee_data(bank, code):
            continue
        fee_info = parse_fee_from_name(name)
        if fee_info:
            update_fee_info(bank, code, fee_info)
            found += 1
        else:
            # 名称中未找到费用信息，标记为无赎回费
            update_fee_info(bank, code, {
                'has_redemption_fee': False,
                'fee_schedule': [],
                'fee_description': '',
                'source': 'name_parse',
            })
    return found


def get_fee_summary():
    """获取费用数据库统计摘要"""
    db = load_fee_db()
    products = db.get('products', {})
    total = len(products)
    has_fee = sum(1 for v in products.values() if v.get('has_redemption_fee') is True)
    no_fee = sum(1 for v in products.values() if v.get('has_redemption_fee') is False)
    unknown = sum(1 for v in products.values() if v.get('has_redemption_fee') is None)

    # 按银行统计
    by_bank = {}
    for key, info in products.items():
        bank = key.split('|')[0] if '|' in key else '未知'
        if bank not in by_bank:
            by_bank[bank] = {'total': 0, 'has_fee': 0, 'no_fee': 0, 'unknown': 0}
        by_bank[bank]['total'] += 1
        if info.get('has_redemption_fee') is True:
            by_bank[bank]['has_fee'] += 1
        elif info.get('has_redemption_fee') is False:
            by_bank[bank]['no_fee'] += 1
        else:
            by_bank[bank]['unknown'] += 1

    return {
        'total': total,
        'has_fee': has_fee,
        'no_fee': no_fee,
        'unknown': unknown,
        'by_bank': by_bank,
    }
