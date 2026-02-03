# -*- coding: utf-8 -*-
"""
Fee Engine — V13 量化系统统一费率计算引擎

Single Source of Truth for all fee calculations across:
- Strategy module (bank_product_strategy_v6.py)
- Backtest module (backtest_pattern_v4.py)
- Post-processing module (post_processing.py)

Core Principles:
1. 管理费/托管费已体现在NAV中，无需代码计算
2. 赎回费(Redemption Fee)是唯一需要计算的费用
3. 支持阶梯费率和限时优惠

Author: V13 Quant System
Version: 1.0.0
Date: 2026-02-03
"""

import os
import json
import logging
import threading
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# 缺失费率处理
# ============================================================
def _parse_missing_default():
    raw = os.getenv("FEE_MISSING_DEFAULT", "N/A").strip().lower()
    if raw in ("n/a", "na", "", "none"):
        return None  # 不回退，标记不可用
    try:
        return float(raw)
    except ValueError:
        return None

MISSING_FEE_DEFAULT = _parse_missing_default()
_MISSING_PRODUCTS: set[str] = set()

def get_missing_fee_products() -> List[str]:
    """获取缺失赎回费的产品列表（排序后返回）"""
    return sorted(_MISSING_PRODUCTS)

# ============================================================
# 配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEE_DB_PATH = os.path.join(BASE_DIR, "赎回费数据库.json")

# 线程安全锁
_DATA_LOCK = threading.Lock()

# ============================================================
# 单例模式数据库加载
# ============================================================

class _FeeDatabase:
    """赎回费数据库单例"""

    _instance = None
    _data: Dict[str, Any] = None
    _load_time: datetime = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = None
            cls._instance._load_time = None
        return cls._instance

    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        """加载赎回费数据库（懒加载 + 缓存 + 线程安全）

        Args:
            force_reload: 强制重新加载

        Returns:
            数据库字典 {"version": int, "products": {...}}
        """
        # 快速路径：已加载且无需刷新
        if self._data is not None and not force_reload:
            return self._data

        # 线程安全加载
        with _DATA_LOCK:
            # Double-check after acquiring lock
            if self._data is not None and not force_reload:
                return self._data

            if not os.path.exists(FEE_DB_PATH):
                logger.warning(f"[FeeEngine] 赎回费数据库不存在: {FEE_DB_PATH}")
                self._data = {"version": 1, "products": {}}
                return self._data

            try:
                with open(FEE_DB_PATH, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                self._load_time = datetime.now()
                n_products = len(self._data.get('products', {}))
                n_with_fee = sum(1 for p in self._data.get('products', {}).values()
                               if p.get('has_redemption_fee'))
                logger.info(f"[FeeEngine] 数据库已加载: {n_products} 产品, {n_with_fee} 有赎回费")
                return self._data
            except Exception as e:
                logger.error(f"[FeeEngine] 加载数据库失败: {e}")
            self._data = {"version": 1, "products": {}}
            return self._data

    def get_product_fee_info(self, product_key: str) -> Optional[Dict[str, Any]]:
        """获取产品费率信息

        Args:
            product_key: 产品键，格式为 "银行|产品代码" 或 "产品代码"

        Returns:
            费率信息字典，或 None
        """
        db = self.load()
        products = db.get('products', {})

        # 直接匹配
        if product_key in products:
            return products[product_key]

        # 尝试只用产品代码匹配
        for key, info in products.items():
            if '|' in key and key.split('|')[1] == product_key:
                return info
            if key == product_key:
                return info

        return None


# 全局单例
_fee_db = _FeeDatabase()


def get_fee_database() -> _FeeDatabase:
    """获取费率数据库单例"""
    return _fee_db


def reload_fee_database():
    """强制重新加载费率数据库"""
    _fee_db.load(force_reload=True)


# ============================================================
# 核心计算函数
# ============================================================

def get_redemption_rate(product_code: str, days_held: int,
                        bank: str = None) -> float:
    """根据持有天数查询赎回费率

    支持阶梯费率逻辑，例如：
    - <7天: 1.5%
    - 7-30天: 0.1%
    - >30天: 0%

    Args:
        product_code: 产品代码
        days_held: 持有天数（日历天）
        bank: 银行名称（可选，用于构建完整产品键）

    Returns:
        赎回费率（小数形式，如 0.015 表示 1.5%）
        若无数据则返回 0
    """
    if days_held < 0:
        days_held = 0

    # 构建产品键（数据库中key格式: "民生银行|FBAE95001"）
    if bank:
        # 确保银行名称以"银行"结尾
        bank_name = bank if bank.endswith('银行') else f"{bank}银行"
        product_key = f"{bank_name}|{product_code}"
    else:
        product_key = product_code

    # 查询产品费率信息
    fee_info = _fee_db.get_product_fee_info(product_key)

    # 无数据：记录缺失，按配置决定是否回退
    if fee_info is None:
        _MISSING_PRODUCTS.add(product_key)
        if MISSING_FEE_DEFAULT is None:
            logger.warning(f"[FeeEngine] 未找到赎回费，未回退（N/A）：{product_key}")
            return None
        logger.warning(f"[FeeEngine] 未找到赎回费，使用保守默认 {MISSING_FEE_DEFAULT*100:.2f}%: {product_key}")
        return MISSING_FEE_DEFAULT

    # 无赎回费标记
    if not fee_info.get('has_redemption_fee', False):
        return 0.0

    # 获取费率阶梯
    fee_schedule = fee_info.get('fee_schedule', [])
    if not fee_schedule:
        return 0.0

    # 遍历阶梯找到适用的费率
    # 支持两种格式：
    # 格式1 (现有): {"min_days": 0, "max_days": 7, "fee_rate": 0.015}
    # 格式2 (新版): {"max_days": 7, "rate": 0.015}
    for tier in fee_schedule:
        # 获取阈值
        max_days = tier.get('max_days', float('inf'))
        min_days = tier.get('min_days', 0)

        # 获取费率（兼容两种字段名）
        rate = tier.get('fee_rate', tier.get('rate', 0.0))

        # 检查是否在此阶梯范围内
        if min_days <= days_held < max_days:
            return rate

        # 新格式：只有 max_days，表示 <max_days 时适用
        if 'min_days' not in tier and days_held < max_days:
            return rate

    # 超出所有阶梯范围，返回最后一个阶梯的费率
    last_tier = fee_schedule[-1]
    return last_tier.get('fee_rate', last_tier.get('rate', 0.0))


def calculate_net_yield(gross_annual_yield: float,
                        days_held: int,
                        product_code: str = None,
                        bank: str = None,
                        fee_rate: float = None) -> float:
    """计算扣除赎回费后的净年化收益率

    公式：净年化收益率 = 毛年化收益率 - (赎回费率 * 365 / max(days_held, 1))

    注意：不扣除管理费（已体现在NAV中），只扣除赎回费磨损

    Args:
        gross_annual_yield: 毛年化收益率（%，如 3.5 表示 3.5%）
        days_held: 持有天数（日历天）
        product_code: 产品代码（用于查询费率，可选）
        bank: 银行名称（可选）
        fee_rate: 直接指定赎回费率（如提供则跳过查询）

    Returns:
        净年化收益率（%）
    """
    if days_held <= 0:
        days_held = 1

    # 获取赎回费率
    if fee_rate is None:
        if product_code:
            fee_rate = get_redemption_rate(product_code, days_held, bank)
        else:
            fee_rate = 0.0

    # 如果没有赎回费，直接返回毛收益
    if fee_rate <= 0:
        return gross_annual_yield

    # 计算赎回费的年化影响
    # 赎回费是一次性的，需要年化：fee_rate * 365 / days_held
    fee_annual_impact = fee_rate * 365 / days_held * 100  # 转换为百分比

    # 净年化收益 = 毛年化收益 - 年化费用影响
    net_yield = gross_annual_yield - fee_annual_impact

    return net_yield


def calculate_redemption_cost(amount: float,
                              gross_pnl: float,
                              days_held: int,
                              product_code: str = None,
                              bank: str = None,
                              fee_rate: float = None) -> Tuple[float, float]:
    """计算赎回费金额（回测用）

    公式（与回测模块一致）：
        redemption_cost = (amount + gross_pnl) * fee_rate
        net_pnl = gross_pnl - redemption_cost

    Args:
        amount: 本金
        gross_pnl: 毛盈亏
        days_held: 持有天数（日历天）
        product_code: 产品代码
        bank: 银行名称
        fee_rate: 直接指定费率（可选）

    Returns:
        (redemption_cost, net_pnl) 元组
    """
    # 获取赎回费率
    if fee_rate is None:
        if product_code:
            fee_rate = get_redemption_rate(product_code, days_held, bank)
        else:
            fee_rate = 0.0

    # 赎回费基于赎回金额（本金 + 收益）
    redemption_amount = amount + gross_pnl
    redemption_cost = redemption_amount * fee_rate

    # 净盈亏
    net_pnl = gross_pnl - redemption_cost

    return redemption_cost, net_pnl


def calculate_fee_drag(days_held: int,
                       product_code: str = None,
                       bank: str = None,
                       fee_rate: float = None) -> float:
    """计算赎回费磨损率（年化）

    公式：磨损率 = 赎回费率 * (365 / 持有天数) * 100

    用于评估赎回费对收益的年化影响。

    Args:
        days_held: 预期持有天数（日历天）
        product_code: 产品代码
        bank: 银行名称
        fee_rate: 直接指定费率（可选）

    Returns:
        年化磨损率（%，如 13.04 表示 13.04%）
    """
    if days_held <= 0:
        days_held = 1

    # 获取赎回费率
    if fee_rate is None:
        if product_code:
            fee_rate = get_redemption_rate(product_code, days_held, bank)
        else:
            fee_rate = 0.0

    if fee_rate <= 0:
        return 0.0

    # 磨损率 = 费率 * (365 / 持有天数) * 100
    drag = fee_rate * (365 / days_held) * 100
    return drag


def check_liquidity_cost_veto(expected_yield: float,
                              days_held: int,
                              product_code: str = None,
                              bank: str = None,
                              fee_rate: float = None,
                              tolerance: float = 0.3) -> Tuple[bool, float, str]:
    """流动性成本一票否决检查

    判断逻辑：如果 磨损率 > (预期年化收益 * tolerance)，则否决

    Args:
        expected_yield: 预期年化收益率（%，如 4.0 表示 4.0%）
        days_held: 预期持有天数
        product_code: 产品代码
        bank: 银行名称
        fee_rate: 直接指定费率（可选）
        tolerance: 容忍阈值（默认 0.3 即 30%）

    Returns:
        (is_vetoed, drag_rate, reason) 元组
        - is_vetoed: 是否被否决
        - drag_rate: 计算出的磨损率（%）
        - reason: 否决原因（如果被否决）
    """
    # 计算磨损率
    drag = calculate_fee_drag(days_held, product_code, bank, fee_rate)

    # 计算容忍上限
    threshold = expected_yield * tolerance

    # 判断是否否决
    if drag > threshold and drag > 0:
        reason = f"赎回磨损{drag:.1f}% > 收益{int(tolerance*100)}%={threshold:.1f}%"
        return True, drag, reason

    return False, drag, ""


# ============================================================
# 便捷函数
# ============================================================

def has_redemption_fee(product_code: str, bank: str = None) -> bool:
    """检查产品是否有赎回费

    Args:
        product_code: 产品代码
        bank: 银行名称

    Returns:
        是否有赎回费
    """
    if bank:
        bank_name = bank if bank.endswith('银行') else f"{bank}银行"
        product_key = f"{bank_name}|{product_code}"
    else:
        product_key = product_code

    fee_info = _fee_db.get_product_fee_info(product_key)
    if fee_info is None:
        return False

    return fee_info.get('has_redemption_fee', False)


def get_fee_info(product_code: str, bank: str = None) -> Optional[Dict[str, Any]]:
    """获取产品完整费率信息

    Args:
        product_code: 产品代码
        bank: 银行名称

    Returns:
        费率信息字典，包含 has_redemption_fee, fee_schedule, fee_description 等
    """
    if bank:
        bank_name = bank if bank.endswith('银行') else f"{bank}银行"
        product_key = f"{bank_name}|{product_code}"
    else:
        product_key = product_code

    return _fee_db.get_product_fee_info(product_key)


def format_fee_description(product_code: str, bank: str = None) -> str:
    """格式化费率描述（用于显示）

    Args:
        product_code: 产品代码
        bank: 银行名称

    Returns:
        人类可读的费率描述字符串
    """
    fee_info = get_fee_info(product_code, bank)
    if fee_info is None:
        return "无费率数据"

    if not fee_info.get('has_redemption_fee'):
        return "无赎回费"

    fee_schedule = fee_info.get('fee_schedule', [])
    if not fee_schedule:
        return "有赎回费（具体费率未知）"

    parts = []
    for tier in fee_schedule:
        min_days = tier.get('min_days', 0)
        max_days = tier.get('max_days', float('inf'))
        rate = tier.get('fee_rate', tier.get('rate', 0))

        if max_days >= 999999:
            if rate == 0:
                parts.append(f">={min_days}天: 免费")
            else:
                parts.append(f">={min_days}天: {rate*100:.2f}%")
        elif min_days == 0:
            if rate == 0:
                parts.append(f"<{max_days}天: 免费")
            else:
                parts.append(f"<{max_days}天: {rate*100:.2f}%")
        else:
            if rate == 0:
                parts.append(f"{min_days}-{max_days}天: 免费")
            else:
                parts.append(f"{min_days}-{max_days}天: {rate*100:.2f}%")

    return ", ".join(parts)


# ============================================================
# 验证与测试
# ============================================================

def verify_consistency():
    """验证费率计算一致性（测试用）

    Returns:
        bool: 验证是否通过
    """
    # 测试参数
    test_cases = [
        # (gross_yield, days_held, fee_rate, expected_net)
        (3.5, 14, 0.005, 3.5 - 0.005 * 365 / 14 * 100),  # -9.55%
        (4.0, 30, 0.001, 4.0 - 0.001 * 365 / 30 * 100),  # 2.78%
        (3.0, 7, 0.015, 3.0 - 0.015 * 365 / 7 * 100),    # -75.21%
        (5.0, 90, 0.0, 5.0),  # 无费用
    ]

    all_passed = True
    for gross, days, rate, expected in test_cases:
        result = calculate_net_yield(gross, days, fee_rate=rate)
        diff = abs(result - expected)
        passed = diff < 0.0001
        if not passed:
            logger.error(f"[FeeEngine] 验证失败: gross={gross}, days={days}, "
                        f"rate={rate}, expected={expected:.4f}, got={result:.4f}")
            all_passed = False

    if all_passed:
        logger.info("[FeeEngine] 一致性验证通过")

    return all_passed


# ============================================================
# 模块初始化
# ============================================================

# 预加载数据库（可选，首次调用时懒加载）
# _fee_db.load()

if __name__ == '__main__':
    # 测试
    logging.basicConfig(level=logging.INFO)

    # 加载数据库
    db = _fee_db.load()
    print(f"Database loaded: {len(db.get('products', {}))} products")

    # 验证一致性
    verify_consistency()

    # 测试计算
    print("\nTest calculations:")
    print(f"  Gross 3.5%, 14 days, 0.5% fee: {calculate_net_yield(3.5, 14, fee_rate=0.005):.4f}%")
    print(f"  Gross 4.0%, 30 days, 0.1% fee: {calculate_net_yield(4.0, 30, fee_rate=0.001):.4f}%")
    print(f"  Gross 3.5%, 60 days, 0.5% fee: {calculate_net_yield(3.5, 60, fee_rate=0.005):.4f}%")
