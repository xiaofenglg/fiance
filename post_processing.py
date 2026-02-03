# -*- coding: utf-8 -*-
"""
V8.0 "Liquidity Master" — SOPM v1 + 双通道分类 + 售罄管理

SOPM (Sold-Out Probability Model) v1:
  - Name Engineering: 名称关键词代理评估流动性
  - Data Logging: 售罄事件记录到CSV供未来AI训练
  - List B Scoring: 综合收益排名 + SOPM评分重排序

双通道分类:
  List A (私行预约): VIP关键词 OR 高收益密度>70% (Perma-Hot)
  List B (捡漏交易): 新鲜度≤50% AND 加速度≥0, 按SOPM排序

售罄管理: data/sold_out.json, 7天自动过期
"""

import os
import re
import csv
import json
import time
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Thread Safety ──
_DATA_LOCK = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SOLD_OUT_PATH = os.path.join(DATA_DIR, 'sold_out.json')
SOPM_CSV_PATH = os.path.join(DATA_DIR, 'sopm_training_data.csv')
SOLD_OUT_EXPIRE_SECONDS = 7 * 24 * 3600  # 7天过期


def _safe_float(value, default=0.0):
    """Robust float conversion handling percentage strings, dashes, None, etc."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ('--', '-', 'N/A', 'n/a'):
            return default
        s = s.rstrip('%')
        try:
            return float(s)
        except (ValueError, TypeError):
            return default
    return default


# V13: 使用统一费率引擎 (Single Source of Truth)
_fee_engine_available = False
try:
    import fee_engine
    _fee_engine_available = True
    _calc_net_return = lambda gross, fee_rate, days=14: fee_engine.calculate_net_yield(gross, days, fee_rate=fee_rate)
    _check_veto = fee_engine.check_liquidity_cost_veto
    _calc_fee_drag = fee_engine.calculate_fee_drag
except ImportError:
    # 降级：内联公式
    def _calc_net_return(gross_return, fee_rate, holding_days=14):
        """[降级] 计算净收益率 — 公式：净年化 = 毛年化 - (费率 * 365 / 持有天数 * 100)"""
        if fee_rate <= 0:
            return gross_return
        if holding_days <= 0:
            holding_days = 14
        fee_annual_impact = fee_rate * 365 / holding_days * 100
        return gross_return - fee_annual_impact

    def _check_veto(expected_yield, days_held, product_code=None, bank=None, fee_rate=None, tolerance=0.3):
        """[降级] 流动性成本一票否决"""
        if fee_rate is None or fee_rate <= 0:
            return False, 0.0, ""
        drag = fee_rate * (365 / max(days_held, 1)) * 100
        threshold = expected_yield * tolerance
        if drag > threshold:
            return True, drag, f"赎回磨损{drag:.1f}% > 收益{int(tolerance*100)}%"
        return False, drag, ""

    def _calc_fee_drag(days_held, product_code=None, bank=None, fee_rate=None):
        """[降级] 计算磨损率"""
        if fee_rate is None or fee_rate <= 0:
            return 0.0
        return fee_rate * (365 / max(days_held, 1)) * 100

# VIP 关键词匹配 (List A classification)
VIP_KEYWORDS = re.compile(
    r'私银|钻石|VIP|Diamond|尊享|高净值|私人银行|贵宾',
    re.IGNORECASE
)

# SOPM CSV 列定义
SOPM_CSV_COLUMNS = ['timestamp', 'date', 'bank', 'code', 'name', 'yield',
                    'is_high_liq_name', 'is_vip_name', 'label']


# ============================================================
# SOPM v1: Sold-Out Probability Model — Name Engineering Proxy
# ============================================================

class LiquidityScorer:
    """SOPM v1 — 名称工程代理: 通过产品名称关键词估算流动性/售罄概率"""

    # 高流动性关键词 → 容易买到 (+20分)
    HIGH_LIQ_KEYWORDS = ['天天', '周周', '月月', '活钱', '宝', '薪', '通', '快线']

    # 低流动性/VIP关键词 → 容易售罄 (-50分)
    LOW_LIQ_KEYWORDS = ['私银', '专属', '高净值', '尊享', '特供', '私行', 'VIP']

    @staticmethod
    def is_high_liquidity(name):
        """名称是否含高流动性关键词"""
        name_upper = (name or '').upper()
        return any(kw.upper() in name_upper for kw in LiquidityScorer.HIGH_LIQ_KEYWORDS)

    @staticmethod
    def is_low_liquidity(name):
        """名称是否含低流动性/VIP关键词"""
        name_upper = (name or '').upper()
        return any(kw.upper() in name_upper for kw in LiquidityScorer.LOW_LIQ_KEYWORDS)

    @staticmethod
    def name_modifier(name):
        """名称工程修正分: +20(高流动性) / -50(VIP/低流动性)"""
        score = 0
        if LiquidityScorer.is_high_liquidity(name):
            score += 20
        if LiquidityScorer.is_low_liquidity(name):
            score -= 50
        return score

    @staticmethod
    def calc_sopm_score(rec, ever_sold_out=False):
        """计算 SOPM 综合评分

        Base 100 + Name Modifier + Yield Penalty + History Penalty
        """
        name = rec.get('产品名称', '') or ''
        score = 100 + LiquidityScorer.name_modifier(name)

        # Yield Penalty: 高收益往往热钱涌入, 更容易售罄
        yield_pct = _safe_float(rec.get('最新收益率%'), 0.0)
        if yield_pct > 4.0:
            score -= 10

        # History Penalty: 曾经售罄过的产品更可能再次售罄
        if ever_sold_out:
            score -= 30

        return score


# ============================================================
# 售罄管理 + SOPM数据记录
# ============================================================

class SoldOutManager:
    """管理售罄产品标记，7天自动过期，同时记录SOPM训练数据"""

    def __init__(self, path=SOLD_OUT_PATH):
        self._path = path
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._data = self._load()

    def _load(self):
        """安全读取 JSON"""
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            logger.warning(f"[SoldOut] 读取失败，重置: {self._path}")
            return {}

    def _save(self):
        """安全写入 JSON (thread-safe)"""
        try:
            with _DATA_LOCK:
                os.makedirs(os.path.dirname(self._path), exist_ok=True)
                with open(self._path, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[SoldOut] 写入失败: {e}")

    def _key(self, bank, code):
        return f"{bank}|{code}"

    @staticmethod
    def _get_ts(val):
        """兼容旧格式(float)和新格式(dict with 'ts')"""
        return val['ts'] if isinstance(val, dict) else val

    def _clean_expired(self):
        """清理过期条目 (thread-safe read-modify-write)"""
        with _DATA_LOCK:
            now = time.time()
            expired = [k for k, v in self._data.items()
                       if now - self._get_ts(v) > SOLD_OUT_EXPIRE_SECONDS]
            for k in expired:
                del self._data[k]
        if expired:
            self._save()
            logger.info(f"[SoldOut] 清理 {len(expired)} 个过期条目")

    def mark(self, bank, code, name='', yield_pct=0):
        """标记售罄，7天后过期，同时记录SOPM训练数据 (thread-safe)"""
        self._clean_expired()
        with _DATA_LOCK:
            key = self._key(bank, code)
            self._data[key] = {'ts': time.time(), 'name': name, 'yield': yield_pct}
        self._save()
        logger.info(f"[SoldOut] 标记售罄: {key}")

        # SOPM v1: 记录训练数据到CSV
        self._log_sopm_event(bank, code, name, yield_pct)

    def _log_sopm_event(self, bank, code, name, yield_pct):
        """追加售罄事件到 SOPM 训练数据 CSV (thread-safe)"""
        try:
            with _DATA_LOCK:
                os.makedirs(os.path.dirname(SOPM_CSV_PATH), exist_ok=True)
                file_exists = os.path.exists(SOPM_CSV_PATH)
                with open(SOPM_CSV_PATH, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(SOPM_CSV_COLUMNS)
                    now = datetime.now()
                    writer.writerow([
                        now.isoformat(),
                        now.strftime('%Y-%m-%d'),
                        bank,
                        code,
                        name,
                        yield_pct,
                        LiquidityScorer.is_high_liquidity(name),
                        LiquidityScorer.is_low_liquidity(name),
                        1,  # label = 1 (售罄事件)
                    ])
            logger.info(f"[SOPM] 训练数据已记录: {bank}|{code} → {SOPM_CSV_PATH}")
        except Exception as e:
            logger.error(f"[SOPM] 训练数据写入失败: {e}")

    def get_sold_out_history(self):
        """返回所有曾经售罄的 (bank, code) 集合（含当前JSON + CSV历史）"""
        history = set()
        # 当前 JSON 中的
        for key in self._data:
            parts = key.split('|', 1)
            if len(parts) == 2:
                history.add((parts[0], parts[1]))
        # CSV 历史记录
        if os.path.exists(SOPM_CSV_PATH):
            try:
                with open(SOPM_CSV_PATH, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        history.add((row.get('bank', ''), row.get('code', '')))
            except Exception:
                pass
        return history

    def unmark(self, bank, code):
        """取消售罄标记 (thread-safe)"""
        key = self._key(bank, code)
        removed = False
        with _DATA_LOCK:
            if key in self._data:
                del self._data[key]
                removed = True
        if removed:
            self._save()
            logger.info(f"[SoldOut] 取消售罄: {key}")

    def is_sold_out(self, bank, code):
        """是否售罄（自动清理过期）"""
        self._clean_expired()
        key = self._key(bank, code)
        return key in self._data

    def get_list(self):
        """返回未过期的售罄列表"""
        self._clean_expired()
        result = []
        now = time.time()
        for key, val in self._data.items():
            parts = key.split('|', 1)
            if len(parts) == 2:
                ts = self._get_ts(val)
                name = val.get('name', '') if isinstance(val, dict) else ''
                yield_pct = val.get('yield', 0) if isinstance(val, dict) else 0
                remaining_hours = max(0, (SOLD_OUT_EXPIRE_SECONDS - (now - ts)) / 3600)
                result.append({
                    'bank': parts[0],
                    'code': parts[1],
                    'name': name,
                    'yield': yield_pct,
                    'marked_at': ts,
                    'remaining_hours': round(remaining_hours, 1),
                })
        return result


# 全局单例
_sold_out_mgr = SoldOutManager()


def get_sold_out_manager():
    return _sold_out_mgr


# ============================================================
# 双通道分类 + SOPM v1 排序
# ============================================================

def _is_vip_product(rec):
    """判断是否为 VIP 产品（名称含关键词）"""
    name = rec.get('产品名称', '') or ''
    return bool(VIP_KEYWORDS.search(name))


def _is_perma_hot(rec):
    """判断是否为常热产品（高收益密度30 > 70%）"""
    density = _safe_float(rec.get('高收益密度30'), 0.0)
    return density > 0.70


def _freshness_progress_bar(rec):
    """生成新鲜度进度条文字，如 'Day 2/8'"""
    progress = _safe_float(rec.get('新鲜度进度'), 0.0)
    window = _safe_float(rec.get('预测窗口天数'), 0.0)
    if window > 0 and progress >= 0:
        current_day = max(1, round(progress * window))
        return f"Day {current_day}/{round(window)}"
    days_since = _safe_float(rec.get('信号距今天数'), 0.0)
    return f"Day {int(days_since) + 1}"


def classify_recommendations(recommendations):
    """双通道分类 + SOPM v1 智能排序

    Args:
        recommendations: 精选推荐列表 (Top 100 candidates)

    Returns:
        dict with keys: list_a, list_b, unclassified, stats
    """
    mgr = get_sold_out_manager()

    # SOPM v1: 预加载售罄历史（一次性读取，避免逐条查CSV）
    sold_out_history = mgr.get_sold_out_history()

    list_a = []
    list_b = []
    unclassified = []

    for rec in recommendations:
        bank = rec.get('银行', '')
        code = rec.get('产品代码', '')
        is_vip = _is_vip_product(rec)
        is_hot = _is_perma_hot(rec)
        is_sold = mgr.is_sold_out(bank, code)

        # 共用字段增强
        rec_out = dict(rec)
        rec_out['VIP产品'] = is_vip
        rec_out['常热产品'] = is_hot
        rec_out['售罄'] = is_sold

        # List A: 私行预约 (VIP关键词 OR Perma-Hot)
        # List A 不受售罄影响 — VIP可申请额度
        if is_vip or is_hot:
            rec_out['通道'] = 'A'
            rec_out['通道标签'] = 'Perma-Hot' if is_hot and not is_vip else ('VIP' if is_vip else 'Perma-Hot')
            list_a.append(rec_out)
            continue

        # List B: 捡漏交易
        freshness_progress = _safe_float(rec.get('新鲜度进度'), 0.0)
        acceleration = _safe_float(rec.get('收益加速度'), 0.0)

        # 条件: 新鲜度进度≤0.5 (FRESH) AND 加速度≥0 (非减速)
        is_fresh = freshness_progress <= 0.5
        is_not_decelerating = acceleration >= 0

        if is_fresh and is_not_decelerating and not is_sold:
            rec_out['通道'] = 'B'
            rec_out['通道标签'] = '捡漏'
            rec_out['新鲜度进度条'] = _freshness_progress_bar(rec)

            # SOPM v1: 计算流动性评分
            ever_sold = (bank, code) in sold_out_history
            rec_out['sopm_score'] = LiquidityScorer.calc_sopm_score(rec, ever_sold_out=ever_sold)
            rec_out['is_high_liq'] = LiquidityScorer.is_high_liquidity(rec.get('产品名称', ''))
            rec_out['is_low_liq'] = LiquidityScorer.is_low_liquidity(rec.get('产品名称', ''))

            list_b.append(rec_out)
            continue

        # 未分类（不符合任何通道条件，或已售罄的非VIP产品）
        rec_out['通道'] = ''
        rec_out['通道标签'] = '售罄' if is_sold else ''
        unclassified.append(rec_out)

    # ── SOPM v1: List B 按综合评分排序 ──
    # Final_Score = Yield_Rank_Score * 0.6 + SOPM_Score * 0.4
    # 效果: 高流动性名称的产品上浮, 曾售罄/VIP类产品下沉
    # V8.2: 使用净收益（扣赎回费后）进行排名，确保公平比较
    # V13: 添加流动性成本信息，与策略端口径一致
    if list_b:
        # V13: 计算净收益和磨损率（与策略端一致的口径）
        def _get_net_yield_and_drag(r):
            gross = _safe_float(r.get('最新收益率%'), 0.0)
            expected = _safe_float(r.get('预期年化收益%'), gross)
            fee_rate = _safe_float(r.get('赎回费费率_14天'), 0.0)
            holding_days = int(_safe_float(r.get('预期持有天数'), 14.0))
            net_yield = _calc_net_return(gross, fee_rate, holding_days)
            drag = _calc_fee_drag(holding_days, fee_rate=fee_rate)
            is_vetoed, _, reason = _check_veto(expected, holding_days, fee_rate=fee_rate, tolerance=0.3)
            return net_yield, drag, is_vetoed

        for r in list_b:
            y_net, drag, is_vetoed = _get_net_yield_and_drag(r)
            r['最新收益率%_净'] = round(y_net, 2)
            r['赎回磨损率%'] = round(drag, 2)
            r['流动性否决'] = is_vetoed

        # 收益排名分归一化
        net_yields = [r['最新收益率%_净'] for r in list_b]
        y_min, y_max = min(net_yields), max(net_yields)
        y_range = y_max - y_min if y_max > y_min else 1.0

        for r in list_b:
            y_net = r['最新收益率%_净']
            yield_rank_score = ((y_net - y_min) / y_range) * 100
            sopm = r.get('sopm_score', 100)
            r['yield_rank_score'] = round(yield_rank_score, 1)
            r['sopm_final_score'] = round(yield_rank_score * 0.6 + sopm * 0.4, 1)

        # 按 SOPM 综合分降序排列
        list_b.sort(key=lambda r: -r['sopm_final_score'])

        top_score = list_b[0].get('sopm_final_score', 0)
        bot_score = list_b[-1].get('sopm_final_score', 0) if len(list_b) > 1 else top_score
        logger.info(f"[SOPM] List B 已按综合评分重排序 "
                    f"(Top={top_score}, Bot={bot_score}, N={len(list_b)})")

    # List B 最多50个
    list_b = list_b[:50]

    stats = {
        'list_a_count': len(list_a),
        'list_b_count': len(list_b),
        'unclassified_count': len(unclassified),
        'sold_out_count': sum(1 for r in unclassified if r.get('售罄')),
        'total_input': len(recommendations),
    }

    logger.info(f"[SOPM v1] 双通道分类: A={stats['list_a_count']} B={stats['list_b_count']} "
                f"未分类={stats['unclassified_count']} 售罄={stats['sold_out_count']}")

    return {
        'list_a': list_a,
        'list_b': list_b,
        'unclassified': unclassified,
        'stats': stats,
    }
