# -*- coding: utf-8 -*-
"""
GPU Alpha Engine V2 — 深度学习增强的Alpha信号生成器 (MAX GPU 优化版)

核心改进（相比 gpu_predictor.py V1）:
1. 12维特征工程 — 全GPU向量化，零Python循环
2. 双头模型: 分类(释放概率) + 回归(预期收益幅度)
3. 3层256维LSTM + 8头Attention + 残差连接 + BatchNorm
4. Mixed Precision (AMP fp16) + 大批次 → 最大化GPU利用率
5. Walk-forward训练 + 3模型集成
6. GPU数据全程驻留显存 — pin_memory + non_blocking transfer

目标: 在 4.50% 基准上提供 +1.5% alpha，接近 6% 年化
GPU利用率目标: RTX 5090 使用率 >60%

变更记录:
- V2.0 (2026-01-31): 初始版本
- V2.1 (2026-01-31): 全面GPU优化 — 向量化特征计算，AMP混合精度，大批次
"""

import os
import logging
import math
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── PyTorch 导入 ──
_torch_ok = False
torch = None
nn = None
F = None

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    torch = _torch
    nn = _nn
    F = _F
    _torch_ok = True
except ImportError:
    pass

# ── 超参数 ──
SEQ_LEN = 60
PRED_HORIZON = 5
FEATURE_DIM = 12
HIDDEN_DIM = 128          # 简化模型(256→128), 减少过拟合
NUM_LAYERS = 2            # 减少层数(3→2), 数据量不支持深层
NUM_HEADS = 4             # 减少注意力头(8→4)
DROPOUT = 0.20            # 增加正则化(0.15→0.20)
RELEASE_THRESHOLD = 2.5
TRAIN_WINDOW = 180        # 训练窗口(天)
BATCH_SIZE = 512          # 大批次充分利用24GB VRAM
MAX_EPOCHS = 30           # 减少epoch，早停会更早触发
PATIENCE = 8
LEARNING_RATE = 1e-3      # 更高学习率，更快收敛
WEIGHT_DECAY = 1e-4
N_ENSEMBLE = 2            # 2模型集成(速度/质量平衡)
USE_AMP = True            # 混合精度训练
MIN_ACTIVE_DATES = 60     # 产品最少活跃天数(过滤不活跃产品)
MAX_TRAIN_SAMPLES = 50000 # 训练样本上限(防止后期训练过慢)


# ══════════════════════════════════════════
# 全GPU向量化 特征工程 (12维) — 零Python循环
# ══════════════════════════════════════════

def compute_enhanced_features(returns, masks, dates=None, device=None):
    """计算12维增强特征 — 全部在GPU上完成，无Python for循环"""
    use_torch = _torch_ok and torch.is_tensor(returns)
    if use_torch:
        return _compute_features_gpu(returns, masks, dates, device)
    else:
        return _compute_features_cpu(
            np.asarray(returns, dtype=np.float32),
            np.asarray(masks, dtype=np.float32),
            dates
        )


@torch.no_grad()
def _compute_features_gpu(returns, masks, dates, device):
    """全GPU向量化特征计算 — 利用unfold/conv1d/cumsum消除所有循环"""
    n_p, n_d = returns.shape
    if device is None:
        device = returns.device

    features = torch.zeros(n_p, n_d, FEATURE_DIM, device=device)
    ret_masked = returns * masks

    # ── Feature 0: 年化收益率 ──
    features[:, :, 0] = ret_masked

    # ── Feature 1: 5日滚动波动率 (unfold向量化) ──
    if n_d >= 5:
        padded_r = F.pad(ret_masked, (4, 0), value=0)
        padded_m = F.pad(masks, (4, 0), value=0)
        win_r = padded_r.unfold(1, 5, 1)      # (P, D, 5)
        win_m = padded_m.unfold(1, 5, 1)
        cnt = win_m.sum(dim=2).clamp(min=1)
        mean = (win_r * win_m).sum(dim=2) / cnt
        var = (((win_r - mean.unsqueeze(2))**2) * win_m).sum(dim=2) / cnt
        features[:, :, 1] = torch.sqrt(var) * masks

    # ── Feature 2: 10日动量 MA5/MA10 (conv1d 向量化) ──
    if n_d >= 10:
        ma5 = _rolling_mean_gpu(ret_masked, masks, 5)
        ma10 = _rolling_mean_gpu(ret_masked, masks, 10)
        safe_ma10 = ma10.clamp(min=0.01)
        momentum_10 = (ma5 / safe_ma10 - 1) * masks
        features[:, :, 2] = momentum_10

    # ── Feature 3: 20日动量 MA5/MA20 ──
    if n_d >= 20:
        if 'ma5' not in dir():
            ma5 = _rolling_mean_gpu(ret_masked, masks, 5)
        ma20 = _rolling_mean_gpu(ret_masked, masks, 20)
        safe_ma20 = ma20.clamp(min=0.01)
        momentum_20 = (ma5 / safe_ma20 - 1) * masks
        features[:, :, 3] = momentum_20

    # ── Features 4-6: 时间编码 (向量化) ──
    if dates:
        time_feats = _encode_time_features_gpu(dates, n_p, n_d, device)
        features[:, :, 4:7] = time_feats

    # ── Feature 7: 收益加速度 (差分) ──
    if n_d >= 2:
        accel = torch.zeros_like(returns)
        accel[:, 1:] = (returns[:, 1:] - returns[:, :-1]) * masks[:, 1:] * masks[:, :-1]
        features[:, :, 7] = accel

    # ── Feature 8: 跨产品相对强度 (排名百分位, 全向量化) ──
    features[:, :, 8] = _cross_product_rank_gpu(ret_masked, masks, n_p, n_d, device)

    # ── Feature 9: 局部峰值回撤 (unfold向量化) ──
    if n_d >= 20:
        padded_r2 = F.pad(ret_masked, (19, 0), value=-1e9)
        win20 = padded_r2.unfold(1, 20, 1)    # (P, D, 20)
        local_max = win20.max(dim=2).values
        safe_max = local_max.clamp(min=0.01)
        dd = ((safe_max - ret_masked) / safe_max).clamp(0, 1)
        features[:, :, 9] = dd * masks

    # ── Feature 10: 距上次释放事件天数 (cumsum技巧, 全向量化) ──
    features[:, :, 10] = _days_since_release_gpu(returns, masks, RELEASE_THRESHOLD, device)

    # ── Feature 11: 近30日释放成功率 (conv1d滑动均值) ──
    if n_d >= 30:
        release_f = ((returns > RELEASE_THRESHOLD) & (masks > 0)).float()
        features[:, :, 11] = _rolling_mean_gpu(release_f, masks, 30)

    # NaN安全: 清理所有NaN/Inf
    features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
    return features


def _rolling_mean_gpu(values, masks, window):
    """GPU向量化滚动均值 — 使用cumsum技巧"""
    n_p, n_d = values.shape
    device = values.device

    # cumsum 技巧: rolling_sum = cumsum[i] - cumsum[i-window]
    val_masked = values * masks
    cs_val = torch.cumsum(F.pad(val_masked, (1, 0), value=0), dim=1)
    cs_mask = torch.cumsum(F.pad(masks, (1, 0), value=0), dim=1)

    roll_sum = cs_val[:, window:] - cs_val[:, :-window]
    roll_cnt = cs_mask[:, window:] - cs_mask[:, :-window]
    roll_cnt = roll_cnt.clamp(min=1)
    roll_mean = roll_sum / roll_cnt

    # 前面 window-1 个位置用逐步累积
    result = torch.zeros(n_p, n_d, device=device)
    result[:, window-1:] = roll_mean
    # 填充前 window-1 位置
    for i in range(1, min(window, n_d)):
        c = cs_mask[:, i+1] - cs_mask[:, 0]
        c = c.clamp(min=1)
        result[:, i] = (cs_val[:, i+1] - cs_val[:, 0]) / c

    return result * masks


def _encode_time_features_gpu(dates, n_p, n_d, device):
    """向量化时间编码 — 预计算后广播到所有产品"""
    sins = torch.zeros(n_d, device=device)
    coss = torch.zeros(n_d, device=device)
    wdays = torch.zeros(n_d, device=device)

    for i, d in enumerate(dates):
        try:
            dt = datetime.strptime(d, '%Y-%m-%d')
            sins[i] = math.sin(2 * math.pi * dt.day / 30)
            coss[i] = math.cos(2 * math.pi * dt.day / 30)
            wdays[i] = dt.weekday() / 4.0 if dt.weekday() < 5 else 1.0
        except ValueError:
            pass

    # 广播到 (n_p, n_d, 3)
    time_feats = torch.stack([sins, coss, wdays], dim=1)  # (n_d, 3)
    return time_feats.unsqueeze(0).expand(n_p, -1, -1)    # (n_p, n_d, 3)


def _cross_product_rank_gpu(ret_masked, masks, n_p, n_d, device):
    """全向量化跨产品排名 — 使用argsort(argsort())得到排名"""
    # (P, D) → 每个日期内排名
    # 未激活的产品用 -inf 保证排在最后
    big_neg = torch.full_like(ret_masked, -1e9)
    active_vals = torch.where(masks > 0, ret_masked, big_neg)  # (P, D)

    # argsort 两次得到排名矩阵
    order = active_vals.argsort(dim=0)   # (P, D) 每列排序索引
    ranks = torch.zeros_like(ret_masked)
    # 使用 scatter 快速赋排名
    row_indices = torch.arange(n_p, device=device).unsqueeze(1).expand(-1, n_d)
    ranks.scatter_(0, order, row_indices.float())

    # 归一化到 [0, 1]
    n_valid_per_day = (masks > 0).sum(dim=0).clamp(min=1).float()  # (D,)
    result = (ranks / (n_valid_per_day.unsqueeze(0) - 1).clamp(min=1)) * masks
    return result


def _days_since_release_gpu(returns, masks, threshold, device):
    """全向量化计算距上次释放事件天数

    技巧: 对每个产品, release_events累积重置计数器
    使用 cumsum + 创新的重置技巧
    """
    n_p, n_d = returns.shape
    is_release = ((returns > threshold) & (masks > 0)).float()  # (P, D)

    # 计算距离: 使用 cummax 技巧
    # 构造递增索引
    indices = torch.arange(n_d, device=device).float().unsqueeze(0).expand(n_p, -1)  # (P, D)

    # 将释放事件位置标记
    release_idx = indices * is_release  # 非释放处为0

    # cummax 获取到当前位置的最后一个释放事件索引
    # 但0会干扰，用-inf替换
    release_idx_masked = torch.where(is_release > 0, indices, torch.tensor(-1.0, device=device))
    last_release_idx = torch.cummax(release_idx_masked, dim=1).values  # (P, D)

    # 距离 = 当前索引 - 最后释放索引
    gap = indices - last_release_idx  # 未发生过释放时，gap很大
    gap = torch.where(last_release_idx >= 0, gap, torch.tensor(30.0, device=device))

    # 归一化到 [0, 1]
    result = (gap / 30.0).clamp(0, 1) * masks
    return result


def _compute_features_cpu(returns, masks, dates):
    """CPU/NumPy 版本（回退用）"""
    n_p, n_d = returns.shape
    features = np.zeros((n_p, n_d, FEATURE_DIM), dtype=np.float32)
    ret_masked = returns * masks

    features[:, :, 0] = ret_masked

    # Feature 1: 5日波动率
    for i in range(4, n_d):
        w = ret_masked[:, i-4:i+1]
        wm = masks[:, i-4:i+1]
        cnt = np.maximum(wm.sum(axis=1), 1)
        mean = w.sum(axis=1) / cnt
        var = (((w - mean[:, None])**2) * wm).sum(axis=1) / cnt
        features[:, i, 1] = np.sqrt(var) * masks[:, i]

    # Feature 2, 3: 动量
    for i in range(9, n_d):
        ma5_m = masks[:, i-4:i+1]
        ma5 = (ret_masked[:, i-4:i+1] * ma5_m).sum(axis=1) / np.maximum(ma5_m.sum(axis=1), 1)
        ma10_m = masks[:, i-9:i+1]
        ma10 = (ret_masked[:, i-9:i+1] * ma10_m).sum(axis=1) / np.maximum(ma10_m.sum(axis=1), 1)
        features[:, i, 2] = (ma5 / np.maximum(ma10, 0.01) - 1) * masks[:, i]
    for i in range(19, n_d):
        ma5_m = masks[:, i-4:i+1]
        ma5 = (ret_masked[:, i-4:i+1] * ma5_m).sum(axis=1) / np.maximum(ma5_m.sum(axis=1), 1)
        ma20_m = masks[:, i-19:i+1]
        ma20 = (ret_masked[:, i-19:i+1] * ma20_m).sum(axis=1) / np.maximum(ma20_m.sum(axis=1), 1)
        features[:, i, 3] = (ma5 / np.maximum(ma20, 0.01) - 1) * masks[:, i]

    if dates:
        for i, d in enumerate(dates):
            try:
                dt = datetime.strptime(d, '%Y-%m-%d')
                features[:, i, 4] = math.sin(2 * math.pi * dt.day / 30)
                features[:, i, 5] = math.cos(2 * math.pi * dt.day / 30)
                features[:, i, 6] = dt.weekday() / 4.0 if dt.weekday() < 5 else 1.0
            except ValueError:
                pass

    if n_d >= 2:
        features[:, 1:, 7] = (returns[:, 1:] - returns[:, :-1]) * masks[:, 1:] * masks[:, :-1]

    # Feature 8: 跨产品排名
    for d in range(n_d):
        col = ret_masked[:, d]
        valid = masks[:, d] > 0
        nv = valid.sum()
        if nv > 1:
            ranks = np.zeros(n_p)
            order = col.argsort()
            ranks[order] = np.arange(n_p, dtype=np.float32)
            features[:, d, 8] = (ranks / max(nv - 1, 1)) * valid.astype(np.float32)

    # Feature 9: 回撤
    for d in range(20, n_d):
        wmax = (ret_masked[:, d-19:d+1]).max(axis=1)
        cur = ret_masked[:, d]
        safe = np.maximum(wmax, 0.01)
        features[:, d, 9] = np.clip((safe - cur) / safe, 0, 1) * masks[:, d]

    # Feature 10: 距上次释放
    release = (returns > RELEASE_THRESHOLD) & (masks > 0)
    for p in range(n_p):
        last = -1
        for d in range(n_d):
            if release[p, d]:
                last = d
            if last >= 0 and masks[p, d] > 0:
                features[p, d, 10] = min((d - last) / 30.0, 1.0)
            else:
                features[p, d, 10] = 1.0

    # Feature 11: 30日释放率
    rel_f = release.astype(np.float32)
    for d in range(30, n_d):
        w = rel_f[:, d-29:d+1]
        m = masks[:, d-29:d+1]
        features[:, d, 11] = (w * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1)

    return features


# ══════════════════════════════════════════
# 模型定义 (V2 — GPU优化版)
# ══════════════════════════════════════════

if _torch_ok:
    class AlphaPredictor(nn.Module):
        """增强Alpha预测模型 — 双头输出, CUDA优化

        Input:  (batch, seq_len=60, features=12)
        Output: {'release_logits': (B,5), 'return_pred': (B,5)}
        """

        def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                     num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT):
            super().__init__()

            self.input_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm 比 BatchNorm 更适合时序
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False,
            )

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.drop = nn.Dropout(dropout)

            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, PRED_HORIZON),
            )

            self.reg_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, PRED_HORIZON),
            )

        def forward(self, x, mask=None):
            B, T, Feat = x.shape

            # Input projection
            h = self.input_proj(x)

            # LSTM encoding
            lstm_out, _ = self.lstm(h)

            # Self-attention
            attn_mask = (mask == 0) if mask is not None else None
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out,
                                         key_padding_mask=attn_mask)
            out = self.norm1(lstm_out + self.drop(attn_out))

            # FFN
            out = self.norm2(out + self.drop(self.ffn(out)))

            # Last valid state
            if mask is not None:
                lengths = mask.sum(dim=1).long().clamp(min=1) - 1
                batch_idx = torch.arange(B, device=x.device)
                last_h = out[batch_idx, lengths]
            else:
                last_h = out[:, -1]

            return {
                'release_logits': self.cls_head(last_h),
                'return_pred': self.reg_head(last_h),
            }


# ══════════════════════════════════════════
# 训练器 V2 (AMP + 大批次)
# ══════════════════════════════════════════

class AlphaTrainer:
    """Walk-forward 训练器 — AMP混合精度 + 大批次 = MAX GPU"""

    def __init__(self, device=None, ensemble_idx=0):
        if not _torch_ok:
            raise RuntimeError("PyTorch 未安装")
        self.device = device or (torch.device('cuda') if torch.cuda.is_available()
                                  else torch.device('cpu'))
        self.model = AlphaPredictor().to(self.device)
        self.ensemble_idx = ensemble_idx
        self.scaler = torch.amp.GradScaler('cuda') if (USE_AMP and self.device.type == 'cuda') else None

    def _prepare_data(self, features, returns, masks, as_of_idx):
        """准备训练数据 — 全部保留在GPU上"""
        n_p, n_d = returns.shape[:2]
        train_end = as_of_idx - PRED_HORIZON
        train_start = max(0, train_end - TRAIN_WINDOW)

        if train_end - train_start < SEQ_LEN + PRED_HORIZON + 10:
            return None

        X_list, y_cls_list, y_reg_list, m_list = [], [], [], []

        # 预过滤: 只保留活跃产品（减少训练集大小, 提高GPU效率）
        total_active = masks[:, train_start:train_end].sum(dim=1)
        active_mask = total_active >= MIN_ACTIVE_DATES
        n_active = active_mask.sum().item()
        logger.info(f"[Alpha V2] 活跃产品: {n_active}/{masks.shape[0]} "
                     f"(>={MIN_ACTIVE_DATES}天数据)")

        active_features = features[active_mask]
        active_returns = returns[active_mask]
        active_masks = masks[active_mask]

        # 采样步长: 每3天取一个样本(平衡速度和覆盖)
        for t in range(train_start + SEQ_LEN, train_end - PRED_HORIZON + 1, 3):
            x_s = active_features[:, t - SEQ_LEN:t, :]
            m_s = active_masks[:, t - SEQ_LEN:t]
            fut_r = active_returns[:, t:t + PRED_HORIZON]
            fut_m = active_masks[:, t:t + PRED_HORIZON]

            y_cls = (fut_r > RELEASE_THRESHOLD).float() * fut_m
            y_reg = fut_r * fut_m

            valid = (m_s.sum(dim=1) >= SEQ_LEN // 3) & (fut_m.sum(dim=1) >= PRED_HORIZON // 2)
            if valid.sum() < 10:
                continue

            X_list.append(x_s[valid])
            y_cls_list.append(y_cls[valid])
            y_reg_list.append(y_reg[valid])
            m_list.append(m_s[valid])

        if not X_list:
            return None

        X_all = torch.cat(X_list, dim=0)
        y_cls_all = torch.cat(y_cls_list, dim=0)
        y_reg_all = torch.cat(y_reg_list, dim=0)
        m_all = torch.cat(m_list, dim=0)

        # 样本上限: 防止后期训练集过大导致超时
        n_total = X_all.size(0)
        if n_total > MAX_TRAIN_SAMPLES:
            perm = torch.randperm(n_total, device=X_all.device)[:MAX_TRAIN_SAMPLES]
            X_all = X_all[perm]
            y_cls_all = y_cls_all[perm]
            y_reg_all = y_reg_all[perm]
            m_all = m_all[perm]

        return {
            'X': X_all,
            'y_cls': y_cls_all,
            'y_reg': y_reg_all,
            'masks': m_all,
        }

    def train(self, features, returns, masks, as_of_idx, epochs=MAX_EPOCHS):
        """AMP混合精度训练 — 最大化GPU吞吐"""
        data = self._prepare_data(features, returns, masks, as_of_idx)
        if data is None:
            return {'status': 'insufficient_data'}

        X_all = data['X']
        y_cls_all = data['y_cls']
        y_reg_all = data['y_reg']
        m_all = data['masks']

        n = X_all.size(0)
        split = int(n * 0.85)

        torch.manual_seed(42 + self.ensemble_idx * 1000)
        perm = torch.randperm(n, device=self.device)
        train_idx = perm[:split]
        val_idx = perm[split:]

        logger.info(f"[Alpha V2] 训练 {split} 样本, 验证 {n-split}, "
                     f"模型#{self.ensemble_idx}, batch={BATCH_SIZE}, AMP={self.scaler is not None}")

        pos_rate = y_cls_all[train_idx].mean().item()
        # 限制pos_weight防止极端值
        pw_val = min((1 - pos_rate) / max(pos_rate, 0.05), 10.0)
        pos_weight = torch.tensor([pw_val] * PRED_HORIZON, device=self.device)

        cls_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        reg_crit = nn.SmoothL1Loss()  # Huber loss比MSE更鲁棒

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        patience_cnt = 0
        best_state = None

        self.model.train()
        use_amp = self.scaler is not None

        for epoch in range(epochs):
            ep_perm = torch.randperm(split, device=self.device)
            ep_loss = 0.0
            nb = 0

            for i in range(0, split, BATCH_SIZE):
                idx = train_idx[ep_perm[i:i + BATCH_SIZE]]
                bx, by_c, by_r, bm = X_all[idx], y_cls_all[idx], y_reg_all[idx], m_all[idx]

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        out = self.model(bx, bm)
                        loss = cls_crit(out['release_logits'], by_c) + \
                               reg_crit(out['return_pred'], by_r) * 0.02
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    out = self.model(bx, bm)
                    loss = cls_crit(out['release_logits'], by_c) + \
                           reg_crit(out['return_pred'], by_r) * 0.1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                ep_loss += loss.item()
                nb += 1

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        vout = self.model(X_all[val_idx], m_all[val_idx])
                        vloss = (cls_crit(vout['release_logits'], y_cls_all[val_idx]) +
                                 reg_crit(vout['return_pred'], y_reg_all[val_idx]) * 0.02).item()
                else:
                    vout = self.model(X_all[val_idx], m_all[val_idx])
                    vloss = (cls_crit(vout['release_logits'], y_cls_all[val_idx]) +
                             reg_crit(vout['return_pred'], y_reg_all[val_idx]) * 0.1).item()
            self.model.train()

            if vloss < best_val_loss:
                best_val_loss = vloss
                patience_cnt = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return {
            'status': 'ok',
            'train_loss': round(ep_loss / max(nb, 1), 4),
            'val_loss': round(best_val_loss, 4),
            'epochs': epoch + 1,
            'pos_rate': round(pos_rate, 4),
            'n_samples': n,
        }

    def predict(self, features, masks):
        self.model.eval()
        with torch.no_grad():
            # NaN安全: 清理输入中的NaN
            features = torch.nan_to_num(features, nan=0.0)
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    out = self.model(features, masks)
            else:
                out = self.model(features, masks)
            probs = torch.sigmoid(out['release_logits'])
            rets = out['return_pred']
            # NaN安全: 清理输出中的NaN
            probs = torch.nan_to_num(probs, nan=0.0).clamp(0, 1)
            rets = torch.nan_to_num(rets, nan=0.0).clamp(-100, 100)
            return {
                'release_probs': probs,
                'return_pred': rets,
            }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'ensemble_idx': self.ensemble_idx,
            'timestamp': datetime.now().isoformat(),
        }, path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        return True


# ══════════════════════════════════════════
# Alpha 集成引擎
# ══════════════════════════════════════════

class AlphaEngine:
    """集成Alpha引擎 — GPU数据全程驻留"""

    def __init__(self, device=None):
        if _torch_ok:
            self.device = device or (torch.device('cuda') if torch.cuda.is_available()
                                     else torch.device('cpu'))
        else:
            self.device = 'cpu'
        self.trainers = []
        self._data_cache = None
        self._last_avg_val_loss = 999.0

    def _load_data(self):
        if self._data_cache is not None:
            return self._data_cache
        from gpu_engine import load_nav_tensors
        data = load_nav_tensors()
        if len(data['product_keys']) == 0:
            return None
        self._data_cache = data
        return data

    def train_and_predict(self, as_of_date=None, progress_cb=None):
        """完整训练→预测流程 — 全GPU"""
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y-%m-%d')

        def _prog(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)
            logger.info(f"[Alpha V2] {pct}% — {msg}")

        _prog(5, '加载数据...')
        data = self._load_data()
        if data is None:
            return {}

        returns_raw = data['returns']
        masks_raw = data['masks']
        dates = data['dates']
        all_keys = data['product_keys']

        # 确保数据在GPU上
        if _torch_ok and torch.is_tensor(returns_raw):
            returns_t = returns_raw.to(self.device)
            masks_t = masks_raw.to(self.device)
        elif _torch_ok:
            returns_t = torch.from_numpy(np.asarray(returns_raw, dtype=np.float32)).to(self.device)
            masks_t = torch.from_numpy(np.asarray(masks_raw, dtype=np.float32)).to(self.device)
        else:
            returns_t = np.asarray(returns_raw, dtype=np.float32)
            masks_t = np.asarray(masks_raw, dtype=np.float32)

        past_dates = [d for d in dates if d <= as_of_date]
        if not past_dates:
            return {}
        as_of_idx = dates.index(past_dates[-1])

        if as_of_idx < SEQ_LEN + PRED_HORIZON + 30:
            return {}

        _prog(10, f'计算12维特征 (GPU向量化, {len(all_keys)}产品×{len(dates)}天)...')
        t0 = time.time()
        features = compute_enhanced_features(returns_t, masks_t, dates, self.device)
        feat_time = time.time() - t0

        if _torch_ok and self.device.type == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1024**3
            _prog(25, f'特征完成 {feat_time:.1f}s | GPU显存 {mem_used:.1f}GB')
        else:
            _prog(25, f'特征完成 {feat_time:.1f}s (CPU)')

        # 训练集成
        if _torch_ok:
            self.trainers = []
            all_probs = []
            all_rets = []
            all_val_losses = []

            for ei in range(N_ENSEMBLE):
                _prog(25 + ei * 20, f'训练模型 {ei+1}/{N_ENSEMBLE} (AMP={USE_AMP})...')
                t1 = time.time()
                model_path = os.path.join(MODEL_DIR, f'alpha_v2_e{ei}.pt')

                trainer = AlphaTrainer(device=self.device, ensemble_idx=ei)
                result = trainer.train(features, returns_t, masks_t, as_of_idx)

                train_time = time.time() - t1
                if result['status'] == 'ok':
                    all_val_losses.append(result['val_loss'])
                    trainer.save(model_path)
                    _prog(25 + (ei+1) * 20 - 5,
                          f'模型#{ei} 训练完成 {train_time:.1f}s | '
                          f'val_loss={result["val_loss"]:.4f} | '
                          f'epochs={result["epochs"]}')

                    # 批量预测（全部产品一次性）
                    start_idx = max(0, as_of_idx - SEQ_LEN + 1)
                    pred_feat = features[:, start_idx:as_of_idx + 1, :]
                    pred_mask = masks_t[:, start_idx:as_of_idx + 1]

                    actual_len = pred_feat.size(1)
                    if actual_len < SEQ_LEN:
                        pad_len = SEQ_LEN - actual_len
                        pred_feat = F.pad(pred_feat, (0, 0, pad_len, 0), value=0)
                        pred_mask = F.pad(pred_mask, (pad_len, 0), value=0)

                    pred = trainer.predict(pred_feat, pred_mask)
                    all_probs.append(pred['release_probs'].cpu().numpy())
                    all_rets.append(pred['return_pred'].cpu().numpy())

                self.trainers.append(trainer)

                # 清理中间GPU内存
                if _torch_ok and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            if not all_probs:
                _prog(80, '回退到统计方法')
                self._last_avg_val_loss = 999.0  # 统计回退时标记低质量
                if _torch_ok and torch.is_tensor(returns_t):
                    returns_np = returns_t.cpu().numpy()
                    masks_np = masks_t.cpu().numpy()
                else:
                    returns_np = returns_t
                    masks_np = masks_t
                return self._statistical_fallback(returns_np, masks_np, dates, as_of_idx, all_keys)

            avg_probs = np.mean(all_probs, axis=0)
            avg_rets = np.mean(all_rets, axis=0)

        else:
            returns_np = np.asarray(returns_raw, dtype=np.float32)
            masks_np = np.asarray(masks_raw, dtype=np.float32)
            return self._statistical_fallback(returns_np, masks_np, dates, as_of_idx, all_keys)

        # 生成alpha信号
        _prog(90, '生成alpha信号...')
        result = {}

        weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])

        for i, key in enumerate(all_keys):
            probs = avg_probs[i]
            rets = avg_rets[i]

            release_prob_5d = float(np.dot(probs, weights))
            expected_return = float(np.dot(rets, weights))

            prob_trend = float(probs[0] - probs[-1]) if len(probs) > 1 else 0
            entry_quality = float(np.clip(release_prob_5d + prob_trend * 0.5, 0, 1))
            hold_signal = float(np.mean(probs[2:]))

            if len(all_probs) > 1:
                std_across = np.std([p[i] for p in all_probs], axis=0).mean()
                confidence = float(np.clip(1 - std_across * 3, 0.1, 1.0))
            else:
                confidence = 0.5

            alpha_score = float(np.clip(
                release_prob_5d * 0.4 + entry_quality * 0.3 +
                (expected_return / 10.0) * 0.2 + hold_signal * 0.1,
                0, 1
            ))

            # NaN safety: 如果任何值是NaN，置为默认值
            if np.isnan(alpha_score) or np.isnan(release_prob_5d):
                alpha_score = 0.0
                release_prob_5d = 0.0
                expected_return = 0.0
                confidence = 0.0
                entry_quality = 0.0
                hold_signal = 0.0

            result[key] = {
                'alpha_score': round(alpha_score, 4),
                'release_prob_5d': round(release_prob_5d, 4),
                'expected_return': round(expected_return, 2),
                'confidence': round(confidence, 4),
                'entry_quality': round(entry_quality, 4),
                'hold_signal': round(hold_signal, 4),
            }

        # 记录训练质量(供backtest调整DL权重)
        avg_val_loss = float(np.mean(all_val_losses)) if all_val_losses else 999.0
        self._last_avg_val_loss = avg_val_loss

        if _torch_ok and self.device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            _prog(100, f'完成 — {len(result)}产品 | 峰值显存 {peak_mem:.1f}GB | avg_val_loss={avg_val_loss:.3f}')
        else:
            _prog(100, f'完成 — {len(result)}产品')

        return result

    def _statistical_fallback(self, returns, masks, dates, as_of_idx, all_keys):
        """CPU统计回退"""
        result = {}
        n_p, n_d = returns.shape

        for i, key in enumerate(all_keys):
            start = max(0, as_of_idx - 90)
            ret_w = returns[i, start:as_of_idx + 1]
            mask_w = masks[i, start:as_of_idx + 1]
            valid = ret_w[mask_w > 0]

            if len(valid) < 15:
                continue

            rr = float((valid > RELEASE_THRESHOLD).mean())
            recent = valid[-10:] if len(valid) >= 10 else valid
            rec_rate = float((recent > RELEASE_THRESHOLD).mean())
            trend = rec_rate - rr
            momentum = float(recent.mean() - valid.mean()) if len(recent) > 0 else 0

            alpha = float(np.clip(rr * 0.4 + rec_rate * 0.3 + max(trend, 0) * 0.2 +
                                   max(momentum / 10, 0) * 0.1, 0, 1))

            result[key] = {
                'alpha_score': round(alpha, 4),
                'release_prob_5d': round(rec_rate, 4),
                'expected_return': round(float(recent.mean()), 2),
                'confidence': 0.3,
                'entry_quality': round(float(np.clip(rec_rate + trend, 0, 1)), 4),
                'hold_signal': round(float(rr), 4),
            }

        return result


# ── 便捷接口 ──
_engine_cache = None

def get_alpha_signals(as_of_date=None, progress_cb=None) -> dict:
    """获取所有产品的alpha信号"""
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = AlphaEngine()
    return _engine_cache.train_and_predict(as_of_date, progress_cb)
