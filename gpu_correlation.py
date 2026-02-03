# -*- coding: utf-8 -*-
"""
跨产品关联分析

核心目标：发现产品间的联动关系，改进分散化配置

方法：
- GPU 计算大规模相关性矩阵 (500x500)
- 层次聚类识别产品群组
- 主成分分析分解系统性/特异性风险
"""

import os
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

from gpu_engine import (
    get_device, is_gpu_available, is_torch_available,
    load_nav_tensors, to_numpy, to_tensor
)

_torch_ok = False
torch = None

try:
    import torch as _torch
    torch = _torch
    _torch_ok = True
except ImportError:
    pass

# ── 参数 ──
MIN_OVERLAP = 30       # 最小共同数据天数
MAX_CLUSTERS = 10      # 最大聚类数
PCA_COMPONENTS = 3     # PCA 主成分数


class CorrelationAnalyzer:
    """跨产品关联分析器"""

    def __init__(self, device=None):
        if device is None:
            device = get_device()
        self.device = device

    def analyze(self, min_overlap: int = MIN_OVERLAP,
                max_products: int = 200) -> dict:
        """计算跨产品关联分析

        Args:
            min_overlap: 最少共同数据天数
            max_products: 最多分析的产品数（取最活跃的）

        Returns: 完整分析结果字典
        """
        try:
            # 加载数据
            data = load_nav_tensors()
            if len(data['product_keys']) == 0:
                return self._empty_result()

            returns = to_numpy(data['returns'])
            masks = to_numpy(data['masks'])
            all_keys = data['product_keys']
            all_names = data['product_names']

            n_products, n_dates = returns.shape

            # 选取最活跃的产品（数据天数最多）
            data_counts = masks.sum(axis=1)
            active_idx = np.argsort(-data_counts)

            # 过滤至少有 min_overlap 天数据的产品
            active_idx = [i for i in active_idx if data_counts[i] >= min_overlap]
            active_idx = active_idx[:max_products]

            if len(active_idx) < 3:
                return self._empty_result()

            n = len(active_idx)
            logger.info(f"[Correlation] 分析 {n} 个活跃产品")

            # 提取子集
            sub_returns = returns[active_idx]
            sub_masks = masks[active_idx]
            sub_keys = [all_keys[i] for i in active_idx]
            sub_names = [all_names[i] for i in active_idx]
            labels = [f"{k[0][:2]}|{k[1][:8]}" for k in sub_keys]

            # Step 1: 计算相关矩阵
            corr_matrix = self._compute_correlation(sub_returns, sub_masks, min_overlap)

            # Step 2: 层次聚类
            clusters, n_clusters = self._hierarchical_clustering(corr_matrix)

            # Step 3: PCA
            pca_result = self._pca_analysis(sub_returns, sub_masks)

            # Step 4: 分散化评分
            div_score = self._diversification_score(corr_matrix)

            # Step 5: 配置建议
            recommendations = self._generate_recommendations(
                corr_matrix, clusters, sub_keys, sub_names, labels
            )

            # 构建聚类结果
            cluster_info = []
            for c in range(n_clusters):
                members = [i for i in range(n) if clusters[i] == c]
                if not members:
                    continue
                # 簇内平均相关性
                if len(members) > 1:
                    sub_corr = corr_matrix[np.ix_(members, members)]
                    mask = np.ones_like(sub_corr, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_corr = float(sub_corr[mask].mean())
                else:
                    avg_corr = 1.0

                cluster_info.append({
                    'id': int(c),
                    'products': [labels[i] for i in members],
                    'product_keys': [f"{sub_keys[i][0]}|{sub_keys[i][1]}" for i in members],
                    'size': len(members),
                    'avg_corr': round(avg_corr, 3),
                })

            cluster_info.sort(key=lambda x: -x['size'])

            return {
                'correlation_matrix': [[round(float(corr_matrix[i, j]), 4)
                                        for j in range(n)] for i in range(n)],
                'product_labels': labels,
                'product_keys': [f"{k[0]}|{k[1]}" for k in sub_keys],
                'clusters': cluster_info,
                'n_clusters': n_clusters,
                'n_products': n,
                'pca_explained': [round(float(v), 4) for v in pca_result['explained']],
                'systematic_risk_pct': round(pca_result['systematic_pct'], 2),
                'diversification_score': round(div_score, 3),
                'recommendations': recommendations,
            }

        except Exception as e:
            logger.error(f"[Correlation] 分析失败: {e}\n{traceback.format_exc()}")
            return self._empty_result()

    def _compute_correlation(self, returns, masks, min_overlap):
        """计算 Pearson 相关矩阵"""
        n = returns.shape[0]

        if _torch_ok and is_gpu_available():
            return self._compute_correlation_gpu(returns, masks, min_overlap)

        # CPU 版本
        corr = np.eye(n, dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                # 共同有效日期
                overlap = (masks[i] > 0) & (masks[j] > 0)
                n_overlap = overlap.sum()

                if n_overlap < min_overlap:
                    corr[i, j] = corr[j, i] = 0.0
                    continue

                ri = returns[i][overlap]
                rj = returns[j][overlap]

                std_i = ri.std()
                std_j = rj.std()

                if std_i < 1e-8 or std_j < 1e-8:
                    corr[i, j] = corr[j, i] = 0.0
                    continue

                r = np.corrcoef(ri, rj)[0, 1]
                if np.isnan(r):
                    r = 0.0

                corr[i, j] = corr[j, i] = float(r)

        return corr

    def _compute_correlation_gpu(self, returns, masks, min_overlap):
        """GPU 加速的相关矩阵计算"""
        device = self.device if not isinstance(self.device, str) else torch.device(self.device)
        n = returns.shape[0]

        ret_t = torch.from_numpy(returns).to(device)
        mask_t = torch.from_numpy(masks).to(device)

        # 标准化 (均值为0, 标准差为1)
        masked_ret = ret_t * mask_t
        counts = mask_t.sum(dim=1, keepdim=True).clamp(min=1)
        means = masked_ret.sum(dim=1, keepdim=True) / counts
        centered = (ret_t - means) * mask_t

        stds = torch.sqrt((centered ** 2).sum(dim=1, keepdim=True) / counts.clamp(min=1))
        stds = stds.clamp(min=1e-8)
        normed = centered / stds

        # 矩阵乘法得到相关系数
        overlap_matrix = mask_t @ mask_t.T  # (n, n) 共同天数
        corr_matrix = (normed @ normed.T) / overlap_matrix.clamp(min=1)

        # 不足共同天数的设为0
        corr_matrix[overlap_matrix < min_overlap] = 0
        # 对角线设为1
        corr_matrix.fill_diagonal_(1.0)
        # 限制范围
        corr_matrix = corr_matrix.clamp(-1, 1)

        return to_numpy(corr_matrix)

    def _hierarchical_clustering(self, corr_matrix):
        """层次聚类 (Ward 方法)"""
        n = corr_matrix.shape[0]
        if n < 3:
            return np.zeros(n, dtype=int), 1

        # 转为距离矩阵
        dist_matrix = 1 - np.abs(corr_matrix)
        np.fill_diagonal(dist_matrix, 0)

        # 确保对称且非负
        dist_matrix = np.maximum(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        try:
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='ward')

            # 自动选择聚类数 (轮廓系数)
            best_k = 3
            best_score = -1

            for k in range(2, min(MAX_CLUSTERS + 1, n)):
                labels = fcluster(Z, k, criterion='maxclust')
                # 简化的聚类质量评估
                score = self._cluster_quality(corr_matrix, labels, k)
                if score > best_score:
                    best_score = score
                    best_k = k

            labels = fcluster(Z, best_k, criterion='maxclust')
            # 转为0-indexed
            labels = labels - 1

            return labels, best_k

        except Exception as e:
            logger.warning(f"[Correlation] 聚类失败: {e}")
            return np.zeros(n, dtype=int), 1

    def _cluster_quality(self, corr_matrix, labels, k):
        """简化的聚类质量评估"""
        n = len(labels)
        intra_corr = 0
        inter_corr = 0
        intra_count = 0
        inter_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    intra_corr += abs(corr_matrix[i, j])
                    intra_count += 1
                else:
                    inter_corr += abs(corr_matrix[i, j])
                    inter_count += 1

        avg_intra = intra_corr / max(intra_count, 1)
        avg_inter = inter_corr / max(inter_count, 1)

        # 好的聚类: 簇内高相关, 簇间低相关
        return avg_intra - avg_inter

    def _pca_analysis(self, returns, masks):
        """PCA 主成分分析"""
        n_products, n_dates = returns.shape

        # 用有效数据填充（均值填充）
        filled = returns.copy()
        for i in range(n_products):
            valid = returns[i][masks[i] > 0]
            if len(valid) > 0:
                mean_val = valid.mean()
                filled[i][masks[i] == 0] = mean_val

        # 标准化
        means = filled.mean(axis=1, keepdims=True)
        stds = filled.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1
        normed = (filled - means) / stds

        try:
            # SVD
            U, S, Vt = np.linalg.svd(normed, full_matrices=False)
            explained_var = (S ** 2) / (S ** 2).sum()

            # 前3个主成分的方差解释比例
            n_comp = min(PCA_COMPONENTS, len(explained_var))
            explained = explained_var[:n_comp].tolist()
            systematic_pct = float(sum(explained) * 100)

            return {
                'explained': explained,
                'systematic_pct': systematic_pct,
            }
        except Exception as e:
            logger.warning(f"[Correlation] PCA 失败: {e}")
            return {'explained': [0] * PCA_COMPONENTS, 'systematic_pct': 0}

    def _diversification_score(self, corr_matrix):
        """计算分散化评分 (0-1)

        评分依据:
        - 平均绝对相关性越低 → 越分散 → 评分越高
        - 极端高相关性产品占比
        """
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.5

        # 提取上三角
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        abs_corr = np.abs(corr_matrix[mask])

        if len(abs_corr) == 0:
            return 0.5

        avg_abs_corr = abs_corr.mean()
        high_corr_pct = (abs_corr > 0.7).mean()

        # 评分: 低相关 = 高分散
        score = (1 - avg_abs_corr) * 0.7 + (1 - high_corr_pct) * 0.3
        return float(np.clip(score, 0, 1))

    def _generate_recommendations(self, corr_matrix, clusters, keys, names, labels):
        """生成配置建议"""
        recommendations = []
        n_clusters = len(set(clusters))

        # 找到产品最少的簇（可能需要增配）
        cluster_sizes = {}
        for c in set(clusters):
            members = [i for i in range(len(clusters)) if clusters[i] == c]
            cluster_sizes[c] = len(members)

        if n_clusters >= 2:
            # 最小簇 → 建议增配
            min_cluster = min(cluster_sizes, key=cluster_sizes.get)
            members = [i for i in range(len(clusters)) if clusters[i] == min_cluster]
            if members:
                recommendations.append({
                    'action': '增配',
                    'cluster': int(min_cluster),
                    'products': [labels[i] for i in members[:3]],
                    'reason': f'该组仅{len(members)}个产品，与其他组低相关，可提高分散化',
                })

            # 最大簇 → 建议精选
            max_cluster = max(cluster_sizes, key=cluster_sizes.get)
            members = [i for i in range(len(clusters)) if clusters[i] == max_cluster]
            if len(members) > 5:
                recommendations.append({
                    'action': '精选',
                    'cluster': int(max_cluster),
                    'products': [labels[i] for i in members[:3]],
                    'reason': f'该组{len(members)}个产品高度相关，建议精选代表性产品',
                })

        # 检查高相关对
        n = corr_matrix.shape[0]
        high_corr_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > 0.85:
                    high_corr_pairs.append((labels[i], labels[j], corr_matrix[i, j]))

        if high_corr_pairs:
            pair = high_corr_pairs[0]
            recommendations.append({
                'action': '替换',
                'products': [pair[0], pair[1]],
                'reason': f'相关性 {pair[2]:.2f}，同时持有分散化效果差',
            })

        return recommendations

    def _empty_result(self) -> dict:
        return {
            'correlation_matrix': [],
            'product_labels': [],
            'product_keys': [],
            'clusters': [],
            'n_clusters': 0,
            'n_products': 0,
            'pca_explained': [0, 0, 0],
            'systematic_risk_pct': 0,
            'diversification_score': 0,
            'recommendations': [],
        }
