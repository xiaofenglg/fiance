# -*- coding: utf-8 -*-
"""
深度学习净值预测 — 替代卡方检验的释放事件预测器

核心目标：预测每个产品未来1-10天出现"释放事件"(收益率>2.5%)的概率
替代现有 PatternLearner 的卡方检验方法

模型架构：LSTM + Multi-head Attention (轻量版)
- Encoder: 2层 LSTM (hidden=128) 处理60天历史序列
- Attention: Multi-head self-attention 捕捉周期性
- Decoder: FC → Sigmoid 输出10天释放概率
"""

import os
import logging
import traceback
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'release_predictor.pt')

# ── 导入 GPU Engine ──
from gpu_engine import (
    get_device, is_gpu_available, is_torch_available,
    load_nav_tensors, compute_features, to_numpy, to_tensor
)

# ── PyTorch 导入 ──
_torch_ok = False
torch = None
nn = None

try:
    import torch as _torch
    import torch.nn as _nn
    torch = _torch
    nn = _nn
    _torch_ok = True
except ImportError:
    pass

# ── 超参数 ──
SEQ_LEN = 60          # 输入序列长度（天）
PRED_HORIZON = 10     # 预测未来天数
FEATURE_DIM = 7       # 特征维度
HIDDEN_DIM = 128      # LSTM 隐藏维度
NUM_LAYERS = 2        # LSTM 层数
NUM_HEADS = 4         # Attention 头数
DROPOUT = 0.2
RELEASE_THRESHOLD = 2.5  # 释放事件阈值（年化收益率%）
TRAIN_WINDOW = 180    # 训练窗口天数
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# ══════════════════════════════════════════
# 模型定义
# ══════════════════════════════════════════

if _torch_ok:
    class ReleasePredictor(nn.Module):
        """释放事件预测器

        Input:  (batch, seq_len=60, features=7)
        Output: (batch, 10) — 未来10天每天的释放概率
        """

        def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                     num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT):
            super().__init__()

            self.input_proj = nn.Linear(feature_dim, hidden_dim)

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

            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

            self.fc_out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, PRED_HORIZON),
            )

        def forward(self, x, mask=None):
            """
            Args:
                x: (batch, seq_len, feature_dim)
                mask: (batch, seq_len) — 有效数据掩码
            Returns:
                (batch, PRED_HORIZON) — 未来每天的释放 logits
            """
            # Input projection
            h = self.input_proj(x)  # (B, T, H)

            # LSTM encoding
            lstm_out, _ = self.lstm(h)  # (B, T, H)

            # Self-attention (捕捉周期模式)
            attn_mask = None
            if mask is not None:
                # key_padding_mask: True = 忽略
                attn_mask = (mask == 0)

            attn_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=attn_mask,
            )

            # Residual + LayerNorm
            out = self.norm(lstm_out + self.dropout(attn_out))

            # 取最后一个有效时间步的表示
            if mask is not None:
                # 找到每个样本最后一个有效位置
                lengths = mask.sum(dim=1).long().clamp(min=1) - 1
                batch_idx = torch.arange(out.size(0), device=out.device)
                last_hidden = out[batch_idx, lengths]
            else:
                last_hidden = out[:, -1]

            # 预测
            logits = self.fc_out(last_hidden)  # (B, PRED_HORIZON)
            return logits


# ══════════════════════════════════════════
# 训练管理器
# ══════════════════════════════════════════

class PredictorTrainer:
    """Walk-forward 训练管理器

    训练策略:
    - 滑动窗口: 用前180天数据训练，预测后10天
    - 标签: 未来10天中每天 return > 2.5% 则为1，否则为0
    - 损失函数: BCEWithLogitsLoss (类别不平衡加权)
    - 优化器: AdamW, lr=1e-3, weight_decay=1e-4
    - 早停: patience=10, monitor=val_loss
    """

    def __init__(self, device=None):
        if not _torch_ok:
            raise RuntimeError("PyTorch 未安装，无法训练模型")
        self.device = device or get_device()
        self.model = ReleasePredictor().to(self.device)

    def _prepare_data(self, features, returns, masks, as_of_idx):
        """准备训练数据 — 严格时序分割

        Args:
            features: (n_products, n_dates, 7)
            returns: (n_products, n_dates)
            masks: (n_products, n_dates)
            as_of_idx: 当前日期索引（预测截止点）

        Returns:
            train_X, train_y, train_mask, val_X, val_y, val_mask
        """
        n_products, n_dates = returns.shape[:2]

        # 训练数据截止: as_of_idx - PRED_HORIZON (防止前视偏差)
        train_end = as_of_idx - PRED_HORIZON
        train_start = max(0, train_end - TRAIN_WINDOW)

        if train_end - train_start < SEQ_LEN + PRED_HORIZON:
            return None, None, None, None, None, None

        # 构建样本
        X_list, y_list, m_list = [], [], []

        for t in range(train_start + SEQ_LEN, train_end - PRED_HORIZON + 1, 2):
            # 输入: features[:, t-SEQ_LEN:t, :]
            x_slice = features[:, t - SEQ_LEN:t, :]  # (P, SEQ_LEN, 7)
            m_slice = masks[:, t - SEQ_LEN:t]          # (P, SEQ_LEN)

            # 标签: returns[:, t:t+PRED_HORIZON] > RELEASE_THRESHOLD
            future_returns = returns[:, t:t + PRED_HORIZON]  # (P, PRED_HORIZON)
            y_slice = (future_returns > RELEASE_THRESHOLD).float()  # (P, PRED_HORIZON)

            # 只保留有足够历史数据的产品
            valid = (m_slice.sum(dim=1) >= SEQ_LEN // 2)  # 至少30天有效数据
            if valid.sum() == 0:
                continue

            X_list.append(x_slice[valid])
            y_list.append(y_slice[valid])
            m_list.append(m_slice[valid])

        if not X_list:
            return None, None, None, None, None, None

        X_all = torch.cat(X_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        m_all = torch.cat(m_list, dim=0)

        # 时序分割: 最后20%作为验证
        n = X_all.size(0)
        split = int(n * 0.8)
        train_X, val_X = X_all[:split], X_all[split:]
        train_y, val_y = y_all[:split], y_all[split:]
        train_m, val_m = m_all[:split], m_all[split:]

        return train_X, train_y, train_m, val_X, val_y, val_m

    def train(self, features, returns, masks, as_of_idx, epochs=MAX_EPOCHS) -> dict:
        """训练模型

        Returns: {'train_loss': float, 'val_loss': float, 'epochs': int, 'status': str}
        """
        data = self._prepare_data(features, returns, masks, as_of_idx)
        train_X, train_y, train_m, val_X, val_y, val_m = data

        if train_X is None:
            return {'status': 'insufficient_data', 'train_loss': 0, 'val_loss': 0, 'epochs': 0}

        logger.info(f"[Predictor] 训练样本: {train_X.size(0)}, 验证样本: {val_X.size(0)}")

        # 类别不平衡权重
        pos_rate = train_y.mean().item()
        pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 0.01)] * PRED_HORIZON,
                                  device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        n_train = train_X.size(0)

        for epoch in range(epochs):
            # Mini-batch 训练
            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_train, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                bx = train_X[idx]
                by = train_y[idx]
                bm = train_m[idx]

                logits = self.model(bx, bm)
                loss = criterion(logits, by)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_X, val_m)
                val_loss = criterion(val_logits, val_y).item()
            self.model.train()

            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"[Predictor] 早停于 epoch {epoch+1}")
                    break

        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return {
            'status': 'ok',
            'train_loss': round(avg_train_loss, 4),
            'val_loss': round(best_val_loss, 4),
            'epochs': epoch + 1,
        }

    def predict(self, features, masks):
        """预测未来10天释放概率

        Args:
            features: (n_products, seq_len, 7) — 最近 SEQ_LEN 天特征
            masks: (n_products, seq_len)

        Returns:
            (n_products, PRED_HORIZON) — 概率值 [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features, masks)
            probs = torch.sigmoid(logits)
        return probs

    def save(self, path=None):
        """保存模型"""
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }, path)
        logger.info(f"[Predictor] 模型已保存: {path}")

    def load(self, path=None):
        """加载模型"""
        path = path or MODEL_PATH
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        logger.info(f"[Predictor] 模型已加载: {path} "
                    f"(训练于 {checkpoint.get('timestamp', '?')})")
        return True


# ══════════════════════════════════════════
# 集成接口（替代 PatternLearner）
# ══════════════════════════════════════════

def predict_releases(product_keys: List[Tuple], as_of_date: str = None) -> dict:
    """主入口：为所有产品生成释放预测

    Args:
        product_keys: [(bank, code), ...] 候选产品列表
        as_of_date: 预测基准日期 (YYYY-MM-DD)，默认今天

    Returns: {
        (bank, code): {
            'release_probs': [p1, ..., p10],
            'confidence': float,
            'predicted_date': str,
            'predicted_end': str,
            'score': float,
            'model_type': 'deep_learning'
        }
    }
    """
    if as_of_date is None:
        as_of_date = datetime.now().strftime('%Y-%m-%d')

    result = {}

    # 回退到 NumPy/统计方法（如果 PyTorch 不可用）
    if not _torch_ok:
        logger.info("[Predictor] PyTorch 不可用，使用统计回退方法")
        return _predict_releases_numpy(product_keys, as_of_date)

    try:
        # 加载数据
        data = load_nav_tensors()
        if len(data['product_keys']) == 0:
            logger.warning("[Predictor] 无产品数据")
            return result

        returns = data['returns']
        navs = data['navs']
        masks = data['masks']
        dates = data['dates']
        all_keys = data['product_keys']

        # 确定 as_of_date 索引
        if as_of_date in dates:
            as_of_idx = dates.index(as_of_date)
        else:
            # 找最近的过去日期
            past_dates = [d for d in dates if d <= as_of_date]
            if not past_dates:
                logger.warning(f"[Predictor] 无 {as_of_date} 之前的数据")
                return result
            as_of_idx = dates.index(past_dates[-1])

        # 计算特征
        features = compute_features(returns, navs, masks, dates)

        # 创建训练器并尝试加载/训练模型
        device = get_device()
        trainer = PredictorTrainer(device)

        if not trainer.load():
            logger.info("[Predictor] 无保存模型，开始训练...")
            train_result = trainer.train(features, returns, masks, as_of_idx)
            logger.info(f"[Predictor] 训练结果: {train_result}")
            if train_result['status'] == 'ok':
                trainer.save()
            else:
                logger.warning("[Predictor] 训练失败，使用统计回退")
                return _predict_releases_numpy(product_keys, as_of_date)

        # 准备预测输入: 取每个产品最近 SEQ_LEN 天的特征
        start_idx = max(0, as_of_idx - SEQ_LEN + 1)
        pred_features = features[:, start_idx:as_of_idx + 1, :]
        pred_masks = masks[:, start_idx:as_of_idx + 1]

        # Pad if shorter than SEQ_LEN
        actual_len = pred_features.size(1)
        if actual_len < SEQ_LEN:
            pad_len = SEQ_LEN - actual_len
            pad_feat = torch.zeros(pred_features.size(0), pad_len, 7, device=device)
            pad_mask = torch.zeros(pred_features.size(0), pad_len, device=device)
            pred_features = torch.cat([pad_feat, pred_features], dim=1)
            pred_masks = torch.cat([pad_mask, pred_masks], dim=1)

        # 批量预测
        probs = trainer.predict(pred_features, pred_masks)  # (n_products, 10)
        probs_np = to_numpy(probs)

        # 构建结果字典
        as_of_dt = datetime.strptime(as_of_date, '%Y-%m-%d')
        key_set = set(product_keys) if product_keys else set(all_keys)

        for i, key in enumerate(all_keys):
            if key not in key_set:
                continue

            prob_seq = probs_np[i].tolist()  # [p1, ..., p10]
            max_prob_idx = int(np.argmax(prob_seq))
            max_prob = prob_seq[max_prob_idx]

            pred_date = (as_of_dt + timedelta(days=max_prob_idx + 1)).strftime('%Y-%m-%d')
            pred_end = (as_of_dt + timedelta(days=PRED_HORIZON)).strftime('%Y-%m-%d')

            # 综合评分: 加权平均（近期权重更高）
            weights = np.array([1.0 / (j + 1) for j in range(PRED_HORIZON)])
            weights /= weights.sum()
            score = float(np.dot(prob_seq, weights))

            # 置信度: 基于预测概率的集中程度
            confidence = float(max_prob * (1 + np.std(prob_seq)))
            confidence = min(confidence, 1.0)

            result[key] = {
                'release_probs': [round(p, 4) for p in prob_seq],
                'confidence': round(confidence, 4),
                'predicted_date': pred_date,
                'predicted_end': pred_end,
                'score': round(score, 4),
                'model_type': 'deep_learning',
            }

        logger.info(f"[Predictor] 深度学习预测完成: {len(result)} 个产品")

    except Exception as e:
        logger.error(f"[Predictor] 预测失败: {e}\n{traceback.format_exc()}")
        logger.info("[Predictor] 回退到统计方法")
        return _predict_releases_numpy(product_keys, as_of_date)

    return result


def _predict_releases_numpy(product_keys: List[Tuple], as_of_date: str) -> dict:
    """NumPy 统计回退方法 — 基于历史频率的简单预测"""
    result = {}

    try:
        data = load_nav_tensors()
        if len(data['product_keys']) == 0:
            return result

        returns = to_numpy(data['returns'])
        masks = to_numpy(data['masks'])
        dates = data['dates']
        all_keys = data['product_keys']

        as_of_dt = datetime.strptime(as_of_date, '%Y-%m-%d')

        # 找 as_of_date 索引
        past_dates = [d for d in dates if d <= as_of_date]
        if not past_dates:
            return result
        as_of_idx = dates.index(past_dates[-1])

        key_set = set(product_keys) if product_keys else set(all_keys)

        for i, key in enumerate(all_keys):
            if key not in key_set:
                continue

            # 取最近180天数据
            start = max(0, as_of_idx - TRAIN_WINDOW)
            ret_window = returns[i, start:as_of_idx + 1]
            mask_window = masks[i, start:as_of_idx + 1]

            # 计算释放事件频率
            valid_returns = ret_window[mask_window > 0]
            if len(valid_returns) < 10:
                continue

            release_rate = (valid_returns > RELEASE_THRESHOLD).mean()

            # 简单预测: 每天释放概率 = 历史频率
            prob_seq = [float(release_rate)] * PRED_HORIZON

            # 用最近趋势调整
            recent = ret_window[-10:]
            recent_mask = mask_window[-10:]
            recent_valid = recent[recent_mask > 0]
            if len(recent_valid) > 0:
                recent_rate = (recent_valid > RELEASE_THRESHOLD).mean()
                # 加权: 70% 历史 + 30% 近期
                for j in range(PRED_HORIZON):
                    decay = 0.3 * (1 - j / PRED_HORIZON)  # 近期影响随天数衰减
                    prob_seq[j] = float(release_rate * (1 - decay) + recent_rate * decay)

            max_prob_idx = int(np.argmax(prob_seq))
            pred_date = (as_of_dt + timedelta(days=max_prob_idx + 1)).strftime('%Y-%m-%d')
            pred_end = (as_of_dt + timedelta(days=PRED_HORIZON)).strftime('%Y-%m-%d')

            weights = np.array([1.0 / (j + 1) for j in range(PRED_HORIZON)])
            weights /= weights.sum()
            score = float(np.dot(prob_seq, weights))
            confidence = float(min(release_rate * 2, 1.0))

            result[key] = {
                'release_probs': [round(p, 4) for p in prob_seq],
                'confidence': round(confidence, 4),
                'predicted_date': pred_date,
                'predicted_end': pred_end,
                'score': round(score, 4),
                'model_type': 'statistical_fallback',
            }

    except Exception as e:
        logger.error(f"[Predictor] 统计回退也失败: {e}")

    return result
