# -*- coding: utf-8 -*-
"""
GPU 设备管理与通用工具

核心职责：
- 检测 GPU 可用性，提供统一的 device 对象
- CPU 自动回退，对调用方透明
- 数据加载工具：从 NAVDatabaseExcel 批量提取时序张量
- 共享的特征工程函数
"""

import os
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── PyTorch 导入（允许失败/禁用） ──

_torch_available = False
torch = None

# 环境变量可禁用 PyTorch: DISABLE_TORCH=1
if os.environ.get('DISABLE_TORCH', '').lower() in ('1', 'true', 'yes'):
    logger.info("[GPU Engine] PyTorch 已通过 DISABLE_TORCH 环境变量禁用")
else:
    try:
        import torch as _torch
        torch = _torch
        _torch_available = True
    except ImportError:
        logger.warning("[GPU Engine] PyTorch 未安装，所有计算将使用 CPU/NumPy")


# ══════════════════════════════════════════
# 设备管理
# ══════════════════════════════════════════

def get_device():
    """返回最佳可用设备 (cuda 或 cpu)"""
    if _torch_available and torch.cuda.is_available():
        return torch.device('cuda')
    if _torch_available:
        return torch.device('cpu')
    return 'cpu'  # 无 PyTorch 时返回字符串标识


def is_gpu_available() -> bool:
    """GPU 是否可用"""
    return _torch_available and torch.cuda.is_available()


def is_torch_available() -> bool:
    """PyTorch 是否可用"""
    return _torch_available


def gpu_info() -> dict:
    """GPU 硬件信息"""
    info = {
        'available': False,
        'torch_installed': _torch_available,
        'device_name': 'N/A',
        'vram_total_gb': 0,
        'vram_used_gb': 0,
        'vram_free_gb': 0,
        'cuda_version': 'N/A',
        'torch_version': 'N/A',
    }

    if not _torch_available:
        return info

    info['torch_version'] = torch.__version__

    if torch.cuda.is_available():
        info['available'] = True
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda or 'N/A'
        vram_total = torch.cuda.get_device_properties(0).total_memory
        vram_reserved = torch.cuda.memory_reserved(0)
        vram_allocated = torch.cuda.memory_allocated(0)
        info['vram_total_gb'] = round(vram_total / (1024 ** 3), 2)
        info['vram_used_gb'] = round(vram_allocated / (1024 ** 3), 2)
        info['vram_free_gb'] = round((vram_total - vram_reserved) / (1024 ** 3), 2)

    return info


# ══════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════

def _load_nav_database():
    """加载 NAVDatabaseExcel 实例"""
    from nav_db_excel import NAVDatabaseExcel
    return NAVDatabaseExcel()


def load_nav_tensors(bank_name: str = None, min_dates: int = 30) -> dict:
    """从净值数据库批量加载产品时序数据为张量

    Args:
        bank_name: 指定银行名称，None 加载全部
        min_dates: 最少数据天数，少于此的产品跳过

    Returns: {
        'returns': ndarray or Tensor [num_products, max_seq_len],  年化收益率%
        'navs': ndarray or Tensor [num_products, max_seq_len],     单位净值
        'masks': ndarray or Tensor [num_products, max_seq_len],    有效数据掩码
        'product_keys': List[(bank, code)],                        产品标识
        'product_names': List[str],                                产品名称
        'dates': List[str],                                        日期序列（全局）
    }
    """
    db = _load_nav_database()

    # 确定要加载的银行
    if bank_name:
        sheet_name = db._get_sheet_name(bank_name)
        sheets = {sheet_name: db.data[sheet_name]} if sheet_name in db.data else {}
    else:
        sheets = db.data

    # 收集所有日期
    all_dates = set()
    for sheet_name, df in sheets.items():
        for col in df.columns:
            if db._is_date_column(col):
                all_dates.add(col)
    all_dates = sorted(all_dates)

    if not all_dates:
        return _empty_tensors()

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    max_len = len(all_dates)

    # 收集产品数据 — 向量化处理
    product_keys = []
    product_names = []
    nav_chunks = []
    mask_chunks = []
    ret_chunks = []

    for sheet_name, df in sheets.items():
        bank = sheet_name
        n_rows = len(df)
        if n_rows == 0:
            continue

        # 获取日期列
        date_cols = [c for c in df.columns if db._is_date_column(c)]
        if not date_cols:
            continue

        # 批量转换: 整个 DataFrame -> 数值矩阵
        date_col_list = [col for col in date_cols if col in date_to_idx]
        col_target_indices = np.array([date_to_idx[col] for col in date_col_list])

        sub_df = df[date_col_list]
        nav_matrix = sub_df.apply(pd.to_numeric, errors='coerce').values.astype(np.float64)

        # 计算每行有效数据量
        valid_counts = np.sum(~np.isnan(nav_matrix), axis=1)
        valid_rows = valid_counts >= min_dates

        if not np.any(valid_rows):
            continue

        # 过滤有效行
        nav_matrix = nav_matrix[valid_rows]
        valid_row_indices = np.where(valid_rows)[0]
        n_valid = len(valid_row_indices)

        # 提取产品信息
        for row_idx in valid_row_indices:
            idx = df.index[row_idx]
            code = idx[0] if isinstance(idx, tuple) else str(idx)
            name = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else ''
            product_keys.append((bank, code))
            product_names.append(name)

        # 创建整个银行的 NAV 矩阵 [n_valid, max_len]
        bank_navs = np.full((n_valid, max_len), np.nan, dtype=np.float64)
        bank_navs[:, col_target_indices] = nav_matrix

        # Mask: 有效数据位置
        bank_masks = (~np.isnan(bank_navs)).astype(np.float64)

        # 计算年化收益率 — 向量化
        bank_returns = np.zeros((n_valid, max_len), dtype=np.float64)

        # 对于每个产品计算收益率（使用 numba 或纯 numpy）
        # 这里使用分块处理来加速
        for i in range(n_valid):
            valid_idx = np.where(bank_masks[i] > 0)[0]
            if len(valid_idx) > 1:
                navs = bank_navs[i, valid_idx]
                period_ret = navs[1:] / navs[:-1] - 1
                gap_days = np.maximum(np.diff(valid_idx), 1)
                ann_ret = period_ret / gap_days * 365 * 100
                bank_returns[i, valid_idx[1:]] = ann_ret

        nav_chunks.append(bank_navs)
        mask_chunks.append(bank_masks)
        ret_chunks.append(bank_returns)

        logger.debug(f"[GPU Engine] 处理 {bank}: {n_valid} 产品")

    if not product_keys:
        return _empty_tensors()

    # 合并所有银行的数据
    navs_np = np.vstack(nav_chunks).astype(np.float32)
    masks_np = np.vstack(mask_chunks).astype(np.float32)
    returns_np = np.vstack(ret_chunks).astype(np.float32)

    # 转为 PyTorch 张量（如果可用）
    if _torch_available:
        device = get_device()
        result = {
            'returns': torch.from_numpy(returns_np).to(device),
            'navs': torch.from_numpy(navs_np).to(device),
            'masks': torch.from_numpy(masks_np).to(device),
            'product_keys': product_keys,
            'product_names': product_names,
            'dates': all_dates,
        }
    else:
        result = {
            'returns': returns_np,
            'navs': navs_np,
            'masks': masks_np,
            'product_keys': product_keys,
            'product_names': product_names,
            'dates': all_dates,
        }

    logger.info(f"[GPU Engine] 加载 {len(product_keys)} 个产品, "
                f"{max_len} 个日期, device={get_device()}")
    return result


def _empty_tensors():
    """返回空的张量结构"""
    if _torch_available:
        return {
            'returns': torch.zeros(0, 0),
            'navs': torch.zeros(0, 0),
            'masks': torch.zeros(0, 0),
            'product_keys': [],
            'product_names': [],
            'dates': [],
        }
    return {
        'returns': np.zeros((0, 0), dtype=np.float32),
        'navs': np.zeros((0, 0), dtype=np.float32),
        'masks': np.zeros((0, 0), dtype=np.float32),
        'product_keys': [],
        'product_names': [],
        'dates': [],
    }


# ══════════════════════════════════════════
# 特征工程
# ══════════════════════════════════════════

def compute_features(returns, navs, masks, dates: List[str] = None):
    """计算特征矩阵 [products, dates, features]

    Features (7维):
    0: 年化收益率%
    1: 5日滚动波动率
    2: 10日动量 (MA5/MA10 - 1)
    3: 20日动量 (MA5/MA20 - 1)
    4: 月相编码 sin(2pi*day/30)
    5: 月相编码 cos(2pi*day/30)
    6: 星期编码 (0-4 归一化到 0-1)
    """
    if _torch_available and torch.is_tensor(returns):
        return _compute_features_torch(returns, navs, masks, dates)
    else:
        return _compute_features_numpy(returns, navs, masks, dates)


def _compute_features_torch(returns, navs, masks, dates):
    """PyTorch 版本的特征计算"""
    n_products, n_dates = returns.shape
    device = returns.device
    features = torch.zeros(n_products, n_dates, 7, device=device)

    # Feature 0: 年化收益率%
    features[:, :, 0] = returns

    # Feature 1: 5日滚动波动率
    for i in range(4, n_dates):
        window = returns[:, i-4:i+1] * masks[:, i-4:i+1]
        window_mask = masks[:, i-4:i+1]
        count = window_mask.sum(dim=1).clamp(min=1)
        mean = (window.sum(dim=1)) / count
        sq_diff = ((window - mean.unsqueeze(1)) ** 2) * window_mask
        var = sq_diff.sum(dim=1) / count.clamp(min=1)
        features[:, i, 1] = torch.sqrt(var)

    # Feature 2: 10日动量 (MA5/MA10 - 1)
    for i in range(9, n_dates):
        ma5_mask = masks[:, i-4:i+1]
        ma5_sum = (returns[:, i-4:i+1] * ma5_mask).sum(dim=1)
        ma5_cnt = ma5_mask.sum(dim=1).clamp(min=1)
        ma5 = ma5_sum / ma5_cnt

        ma10_mask = masks[:, i-9:i+1]
        ma10_sum = (returns[:, i-9:i+1] * ma10_mask).sum(dim=1)
        ma10_cnt = ma10_mask.sum(dim=1).clamp(min=1)
        ma10 = ma10_sum / ma10_cnt

        safe_ma10 = ma10.clamp(min=0.01)
        features[:, i, 2] = (ma5 / safe_ma10 - 1) * masks[:, i]

    # Feature 3: 20日动量 (MA5/MA20 - 1)
    for i in range(19, n_dates):
        ma5_mask = masks[:, i-4:i+1]
        ma5_sum = (returns[:, i-4:i+1] * ma5_mask).sum(dim=1)
        ma5_cnt = ma5_mask.sum(dim=1).clamp(min=1)
        ma5 = ma5_sum / ma5_cnt

        ma20_mask = masks[:, i-19:i+1]
        ma20_sum = (returns[:, i-19:i+1] * ma20_mask).sum(dim=1)
        ma20_cnt = ma20_mask.sum(dim=1).clamp(min=1)
        ma20 = ma20_sum / ma20_cnt

        safe_ma20 = ma20.clamp(min=0.01)
        features[:, i, 3] = (ma5 / safe_ma20 - 1) * masks[:, i]

    # Features 4-6: 时间编码
    if dates:
        for i, d in enumerate(dates):
            try:
                dt = datetime.strptime(d, '%Y-%m-%d')
                day = dt.day
                weekday = dt.weekday()
                features[:, i, 4] = math.sin(2 * math.pi * day / 30)
                features[:, i, 5] = math.cos(2 * math.pi * day / 30)
                features[:, i, 6] = weekday / 4.0 if weekday < 5 else 1.0
            except ValueError:
                pass

    return features


def _compute_features_numpy(returns, navs, masks, dates):
    """NumPy 版本的特征计算"""
    n_products, n_dates = returns.shape
    features = np.zeros((n_products, n_dates, 7), dtype=np.float32)

    # Feature 0: 年化收益率%
    features[:, :, 0] = returns

    # Feature 1: 5日滚动波动率
    for i in range(4, n_dates):
        window = returns[:, i-4:i+1] * masks[:, i-4:i+1]
        window_mask = masks[:, i-4:i+1]
        count = np.maximum(window_mask.sum(axis=1), 1)
        mean = window.sum(axis=1) / count
        sq_diff = ((window - mean[:, np.newaxis]) ** 2) * window_mask
        var = sq_diff.sum(axis=1) / np.maximum(count, 1)
        features[:, i, 1] = np.sqrt(var)

    # Feature 2: 10日动量
    for i in range(9, n_dates):
        ma5_mask = masks[:, i-4:i+1]
        ma5 = (returns[:, i-4:i+1] * ma5_mask).sum(axis=1) / np.maximum(ma5_mask.sum(axis=1), 1)
        ma10_mask = masks[:, i-9:i+1]
        ma10 = (returns[:, i-9:i+1] * ma10_mask).sum(axis=1) / np.maximum(ma10_mask.sum(axis=1), 1)
        safe_ma10 = np.maximum(ma10, 0.01)
        features[:, i, 2] = (ma5 / safe_ma10 - 1) * masks[:, i]

    # Feature 3: 20日动量
    for i in range(19, n_dates):
        ma5_mask = masks[:, i-4:i+1]
        ma5 = (returns[:, i-4:i+1] * ma5_mask).sum(axis=1) / np.maximum(ma5_mask.sum(axis=1), 1)
        ma20_mask = masks[:, i-19:i+1]
        ma20 = (returns[:, i-19:i+1] * ma20_mask).sum(axis=1) / np.maximum(ma20_mask.sum(axis=1), 1)
        safe_ma20 = np.maximum(ma20, 0.01)
        features[:, i, 3] = (ma5 / safe_ma20 - 1) * masks[:, i]

    # Features 4-6: 时间编码
    if dates:
        for i, d in enumerate(dates):
            try:
                dt = datetime.strptime(d, '%Y-%m-%d')
                day = dt.day
                weekday = dt.weekday()
                features[:, i, 4] = math.sin(2 * math.pi * day / 30)
                features[:, i, 5] = math.cos(2 * math.pi * day / 30)
                features[:, i, 6] = weekday / 4.0 if weekday < 5 else 1.0
            except ValueError:
                pass

    return features


# ══════════════════════════════════════════
# 辅助工具
# ══════════════════════════════════════════

def to_numpy(tensor):
    """将 Tensor 转为 numpy array"""
    if _torch_available and torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def to_tensor(array, device=None):
    """将 numpy array 转为 Tensor"""
    if not _torch_available:
        return np.asarray(array, dtype=np.float32)
    if device is None:
        device = get_device()
    if torch.is_tensor(array):
        return array.to(device)
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)
