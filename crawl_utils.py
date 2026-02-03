# -*- coding: utf-8 -*-
"""
V13 Delta Sync Engine — 状态管理与增量同步工具

功能：
1. 从Parquet数据库高效读取各产品最新日期（Manifest）
2. 提供增量计算工具（需要获取多少天？起始日期是？）
3. 所有银行爬虫共用，避免重复加载
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "净值数据库.parquet")

# 认为产品"已最新"的间隔天数（考虑T+1发布延迟+周末）
UP_TO_DATE_GAP = 1


def get_db_manifest(parquet_path=None):
    """从Parquet数据库高效读取各产品的最新日期

    只读取3列（银行/产品代码/日期），groupby取max，极快。

    Returns:
        dict: {'民生银行': {'AF123456': '2026-01-31', ...}, ...}
    """
    path = parquet_path or PARQUET_PATH
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logger.info("[Manifest] Parquet文件不存在，返回空manifest")
        return {}

    try:
        t0 = time.time()
        # 只读需要的列 — 比加载全文件快10倍+
        df = pd.read_parquet(path, columns=['银行', '产品代码', '日期'])
        # 每个(银行,产品)取最大日期
        latest = df.groupby(['银行', '产品代码'])['日期'].max().reset_index()

        manifest = {}
        for _, row in latest.iterrows():
            bank = row['银行']
            code = row['产品代码']
            last_date = row['日期']
            if bank not in manifest:
                manifest[bank] = {}
            if hasattr(last_date, 'strftime'):
                manifest[bank][code] = last_date.strftime('%Y-%m-%d')
            else:
                manifest[bank][code] = str(last_date)[:10]

        elapsed = time.time() - t0
        total_products = sum(len(v) for v in manifest.values())
        logger.info(f"[Manifest] 加载完成 ({elapsed:.2f}s): "
                    f"{len(manifest)} 银行, {total_products} 产品")
        return manifest

    except Exception as e:
        logger.error(f"[Manifest] 加载失败: {e}")
        return {}


def needs_update(product_code, bank_manifest, gap_days=UP_TO_DATE_GAP):
    """判断产品是否需要更新

    Args:
        product_code: 产品代码
        bank_manifest: 该银行的 {code: last_date_str} 字典
        gap_days: 容许间隔天数（默认1，考虑T+1发布延迟）

    Returns:
        bool: True=需要抓取, False=已最新可跳过
    """
    if not bank_manifest or product_code not in bank_manifest:
        return True  # 新产品，需要抓取

    last_date_str = bank_manifest[product_code]
    try:
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return True

    today = datetime.now().date()
    return (today - last_date).days > gap_days


def calc_days_needed(product_code, bank_manifest, default_days=180, buffer=5):
    """计算产品需要获取多少天的数据

    Args:
        product_code: 产品代码
        bank_manifest: 该银行的manifest
        default_days: 新产品默认天数
        buffer: 额外缓冲天数

    Returns:
        int: 需要获取的天数。0=无需更新
    """
    if not bank_manifest or product_code not in bank_manifest:
        return default_days

    last_date_str = bank_manifest[product_code]
    try:
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return default_days

    today = datetime.now().date()
    gap = (today - last_date).days

    if gap <= UP_TO_DATE_GAP:
        return 0  # 已最新

    return gap + buffer


def calc_start_date(product_code, bank_manifest):
    """计算增量抓取的起始日期（适用于支持startDate的API）

    Returns:
        str: 'YYYY-MM-DD' 或 'YYYYMMDD' 格式
        None: 需要完整抓取（新产品）
        'SKIP': 无需更新
    """
    if not bank_manifest or product_code not in bank_manifest:
        return None

    last_date_str = bank_manifest[product_code]
    try:
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        start = last_date + timedelta(days=1)
        today = datetime.now().date()
        if start > today:
            return 'SKIP'
        return start.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def filter_products_by_manifest(products, bank_manifest, code_field='REAL_PRD_CODE',
                                fallback_field='PRD_CODE'):
    """批量过滤产品列表，跳过已最新的产品

    Args:
        products: 产品列表（字典数组）
        bank_manifest: 该银行的manifest
        code_field: 产品代码字段名
        fallback_field: 备选产品代码字段名

    Returns:
        (to_fetch, skipped_count): 需要抓取的产品列表, 跳过数量
    """
    if not bank_manifest:
        return products, 0

    to_fetch = []
    skipped = 0

    for p in products:
        code = ''
        if isinstance(p, dict):
            code = p.get(code_field) or p.get(fallback_field, '')
        elif hasattr(p, 'product_code'):
            code = p.product_code

        code = str(code).strip()
        if code and not needs_update(code, bank_manifest):
            skipped += 1
            continue
        to_fetch.append(p)

    return to_fetch, skipped


def manifest_summary(manifest):
    """打印manifest摘要"""
    if not manifest:
        return "空（首次运行）"

    parts = []
    for bank, products in sorted(manifest.items()):
        dates = list(products.values())
        if dates:
            latest = max(dates)
            parts.append(f"{bank}: {len(products)}产品, 最新{latest}")
    return " | ".join(parts)
