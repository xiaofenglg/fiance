# -*- coding: utf-8 -*-
"""
银行理财产品V7策略 - 深度学习Alpha增强回测 (从V6升级)

变更记录:
- V7.0 (2026-01-31): 基于 backtest_v6.py 升级
  改动部分:
  1. 集成 gpu_alpha_v2.py 深度学习预测器
  2. Walk-forward DL Alpha: 每次重建产品库时训练GPU模型
  3. 动态买入门槛: DL高置信时降低买入阈值(3.0→2.5%)，捕捉更多释放事件
  4. 智能卖出: DL预测持续释放时延长持有期
  5. 组合Alpha评分: 传统60% + DL 40%
  6. 更激进的轮动: 替换收益下降但DL预测不佳的持仓

目标: 年化收益率 4.50% → 6.0%+ (增加 +1.5% alpha)

回测区间: 2025-06-01 ~ 2026-01-29
初始资金: 1亿元

策略参数对比 (V6 → V7):
  买入阈值: 3.0% → 动态(DL高置信时2.5%, 默认3.0%)
  卖出阈值: 2.0% → 动态(DL预测持续时1.5%, 默认2.0%)
  最大持有: 30天 → 动态(DL预测持续时延长到35天)
  Alpha权重: 纯传统 → 传统60% + DL40%
"""

import os, sys, re, warnings, logging, bisect, time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "净值数据库.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "回测报告_V7_alpha.xlsx")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_v7.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 策略参数 — V7 DL Alpha 增强版
# ============================================================
信号阈值_买入 = 3.0          # 基础买入阈值
信号阈值_买入_DL低 = 2.0     # DL高置信时更低阈值(2.5→2.0), 捕捉更多释放 [V7.1]
信号阈值_卖出 = 2.0          # 基础卖出阈值
信号阈值_卖出_DL低 = 1.0     # DL预测持续时更低卖出(1.5→1.0), 持有更久 [V7.1]
需要突破信号 = False
持有天数_评估 = 7
成功标准 = 2.5
最低成功率 = 30.0            # 降低准入门槛(35→30), 扩大选股池 [V7.1]
最少历史信号 = 2
最短持续天数 = 2
最大持有天数_评估 = 30
最长赎回天数 = 14

# Alpha评分权重 (传统因子)
W_SUCCESS = 0.50
W_AVG_RET = 0.25
W_SHARPE  = 0.15
W_PERSIST = 0.10

# DL Alpha 集成权重 [V7.1 — 自适应权重]
DL_ALPHA_WEIGHT_BASE = 0.50   # DL alpha 基础权重提高到50% [V7.1]
DL_ALPHA_WEIGHT_MIN = 0.15    # 模型质量差时最低权重 [V7.1]
DL_CONFIDENCE_THRESHOLD = 0.4  # 降低置信度门槛(0.5→0.4) [V7.1]
DL_ENTRY_BOOST = 0.5         # DL高置信时买入阈值下调幅度(%)
DL_HOLD_EXTENSION = 5        # DL预测持续释放时延长持有天数
DL_SELL_TIGHTEN = 0.5        # DL预测结束时卖出阈值上调(%)

# 回测参数
回测开始日 = '2025-06-01'
回测结束日 = '2026-01-30'
初始资金 = 100_000_000
最大持仓数 = 9
单仓基础 = 初始资金 / 9
单仓最小倍率 = 0.8           # DL低分产品仓位缩小 [CHANGED]
单仓最大倍率 = 1.2           # DL高分产品仓位放大 [CHANGED]
产品库重建间隔 = 10

# T+1 交易机制
最短持有交易日_赎回 = 4
最大持有交易日 = 30
最大持有交易日_DL = 35        # DL高置信延长 [NEW]
长假阈值_天 = 3
衰减卖出_回看天数 = 3
衰减卖出_比例 = 0.0

# DL 模型重训间隔
DL_RETRAIN_INTERVAL = 30      # 每30天重训一次DL模型(平衡速度和新鲜度) [NEW]


# ============================================================
# 工具函数 (同V6)
# ============================================================

def is_date_col(col_name):
    if not isinstance(col_name, str):
        return False
    return len(col_name) == 10 and col_name[4] == '-' and col_name[7] == '-'


def 判断产品流动性(name):
    if '日开' in name or '天天' in name or '每日' in name:
        return '日开', 0
    m = re.search(r'(\d+)\s*天持有期', name)
    if m:
        return f'{int(m.group(1))}天持有期', int(m.group(1))
    m = re.search(r'最短持有\s*(\d+)\s*天', name)
    if m:
        return f'{int(m.group(1))}天持有', int(m.group(1))
    m = re.search(r'周期\s*(\d+)\s*天', name)
    if m:
        return f'{int(m.group(1))}天周期', int(m.group(1))
    m = re.search(r'(\d+)\s*个?月', name)
    if m and '半年' not in name:
        months = int(m.group(1))
        if months <= 6:
            return f'{months}个月', months * 30
    if '周开' in name: return '周开', 7
    if '日申季赎' in name: return '季赎', 90
    if '月开' in name: return '月开', 30
    if '季开' in name or '季度开' in name: return '季开', 90
    if '半年' in name: return '半年', 180
    if '年开' in name: return '年开', 365
    if '封闭' in name: return '封闭', 999
    if '定开' in name: return '定开', 999
    m = re.search(r'(\d+)\s*Y\s*持有', name)
    if m: return f'{int(m.group(1))}年持有', int(m.group(1)) * 365
    if re.search(r'20[2-5]\d', name) and '信颐' in name: return '目标日期', 999
    return '未知', 999


# ============================================================
# 数据结构 (同V6)
# ============================================================

class ProductData:
    __slots__ = ['bank', 'code', 'name', 'liquidity', 'redeem_days',
                 'nav_dates', 'navs', 'ret_dates', 'rets',
                 'ret_date_idx', 'signals']

    def __init__(self, bank, code, name, liquidity, redeem_days):
        self.bank = bank
        self.code = code
        self.name = name
        self.liquidity = liquidity
        self.redeem_days = redeem_days
        self.nav_dates = []
        self.navs = {}
        self.ret_dates = []
        self.rets = {}
        self.ret_date_idx = {}
        self.signals = []


class Position:
    __slots__ = ['product_key', 'signal_date', 'confirm_date', 'buy_nav',
                 'amount', 'entry_return', 'alpha_score', 'dl_alpha_score',
                 'sell_date', 'sell_nav', 'pnl', 'sell_reason']

    def __init__(self, product_key, signal_date, confirm_date, buy_nav, amount,
                 entry_return=0.0, alpha_score=0.0, dl_alpha_score=0.0):
        self.product_key = product_key
        self.signal_date = signal_date
        self.confirm_date = confirm_date
        self.buy_nav = buy_nav
        self.amount = amount
        self.entry_return = entry_return
        self.alpha_score = alpha_score
        self.dl_alpha_score = dl_alpha_score    # [NEW] DL alpha
        self.sell_date = None
        self.sell_nav = None
        self.pnl = 0.0
        self.sell_reason = ''


# ============================================================
# DL Alpha Manager — Walk-forward GPU 预测器
# ============================================================

class DLAlphaManager:
    """管理DL Alpha模型的walk-forward训练和预测"""

    def __init__(self):
        self.engine = None
        self.alpha_signals = {}   # {(bank, code): {alpha_score, ...}}
        self.last_train_date = None
        self.available = False

        try:
            from gpu_alpha_v2 import AlphaEngine
            import torch
            self.engine = AlphaEngine()
            self.available = torch.cuda.is_available()
            if self.available:
                logger.info("[DL Alpha] GPU可用, 将使用深度学习Alpha增强")
            else:
                logger.info("[DL Alpha] GPU不可用, 使用统计回退")
        except ImportError as e:
            logger.warning(f"[DL Alpha] 导入失败: {e}")

    def should_retrain(self, current_date):
        """是否需要重训"""
        if not self.available and self.engine is None:
            return False
        if self.last_train_date is None:
            return True
        gap = (pd.Timestamp(current_date) - pd.Timestamp(self.last_train_date)).days
        return gap >= DL_RETRAIN_INTERVAL

    def train_and_predict(self, as_of_date):
        """Walk-forward: 训练并生成alpha信号"""
        if self.engine is None:
            return

        try:
            import gc
            t0 = time.time()
            logger.info(f"[DL Alpha] Walk-forward 训练 as_of={as_of_date}")
            self.alpha_signals = self.engine.train_and_predict(as_of_date=as_of_date)
            self.last_train_date = as_of_date
            elapsed = time.time() - t0
            logger.info(f"[DL Alpha] 训练完成: {len(self.alpha_signals)}产品, {elapsed:.1f}s")

            # 清理GPU内存，防止泄漏
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception as e:
            logger.error(f"[DL Alpha] 训练失败: {e}", exc_info=True)

    def get_alpha(self, key):
        """获取产品的DL alpha信号"""
        return self.alpha_signals.get(key, {})

    def get_alpha_score(self, key):
        """获取DL alpha评分 [0, 1]"""
        sig = self.alpha_signals.get(key, {})
        return sig.get('alpha_score', 0.0)

    def get_entry_quality(self, key):
        """获取入场质量 [0, 1]"""
        sig = self.alpha_signals.get(key, {})
        return sig.get('entry_quality', 0.0)

    def get_hold_signal(self, key):
        """获取持有信号 [0, 1]"""
        sig = self.alpha_signals.get(key, {})
        return sig.get('hold_signal', 0.0)

    def get_confidence(self, key):
        """获取模型置信度"""
        sig = self.alpha_signals.get(key, {})
        return sig.get('confidence', 0.0)

    def get_adaptive_weight(self):
        """[V7.1] 根据模型质量自适应调整DL权重
        val_loss < 2 → 高权重(50%), val_loss > 8 → 低权重(15%)
        """
        if self.engine is None:
            return DL_ALPHA_WEIGHT_MIN
        val_loss = getattr(self.engine, '_last_avg_val_loss', 999.0)
        if val_loss < 2.0:
            return DL_ALPHA_WEIGHT_BASE
        elif val_loss > 8.0:
            return DL_ALPHA_WEIGHT_MIN
        else:
            # 线性插值: loss 2→8 映射到 weight 0.50→0.15
            t = (val_loss - 2.0) / 6.0
            return DL_ALPHA_WEIGHT_BASE - t * (DL_ALPHA_WEIGHT_BASE - DL_ALPHA_WEIGHT_MIN)


# ============================================================
# 回测引擎 V7 — DL Alpha 增强
# ============================================================

class BacktestEngine:

    def __init__(self):
        self.products = {}
        self.all_sim_dates = []
        self.library = {}
        self.library_date = None
        self.positions = {}
        self.pending_buys = []
        self.pending_sells = []
        self.pending_buy_keys = set()
        self.pending_sell_keys = set()
        self.date_to_idx = {}
        self.closed_trades = []
        self.cash = 初始资金
        self.daily_values = []
        self.trade_log = []

        # [NEW] DL Alpha Manager
        self.dl_alpha = DLAlphaManager()

    def load_data(self):
        """加载净值数据 (同V6)"""
        logger.info("加载净值数据库...")
        xlsx = pd.ExcelFile(DB_PATH)
        all_dates = set()
        total = 0

        for sheet in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            if 'level_0' in df.columns:
                df = df.rename(columns={'level_0': '产品代码', 'level_1': '产品名称'})

            date_cols = sorted([c for c in df.columns if is_date_col(c)])
            if len(date_cols) < 10:
                continue
            all_dates.update(date_cols)
            count = 0

            for _, row in df.iterrows():
                code = row.get('产品代码')
                name = row.get('产品名称', '')
                if pd.isna(code):
                    continue
                code = str(code).strip()
                name = str(name).strip() if pd.notna(name) else ''

                liq_type, redeem_days = 判断产品流动性(name)
                if redeem_days > 最长赎回天数:
                    continue

                nav_points = []
                for d in date_cols:
                    try:
                        v = float(row[d])
                        if not np.isnan(v) and v > 0:
                            nav_points.append((d, v))
                    except (ValueError, TypeError):
                        pass

                if len(nav_points) < 10:
                    continue

                key = (sheet, code)
                p = ProductData(sheet, code, name, liq_type, redeem_days)
                p.nav_dates = [x[0] for x in nav_points]
                p.navs = {x[0]: x[1] for x in nav_points}

                for i in range(1, len(nav_points)):
                    d0, n0 = nav_points[i - 1]
                    d1, n1 = nav_points[i]
                    gap = (pd.Timestamp(d1) - pd.Timestamp(d0)).days
                    if 0 < gap <= 60 and n0 > 0:
                        p.rets[d1] = (n1 / n0 - 1) / gap * 365 * 100

                p.ret_dates = sorted(p.rets.keys())
                p.ret_date_idx = {d: i for i, d in enumerate(p.ret_dates)}

                if len(p.ret_dates) < 10:
                    continue

                self._precompute_signals(p)
                self.products[key] = p
                count += 1

            total += count

        self.all_sim_dates = sorted(
            d for d in all_dates if 回测开始日 <= d < 回测结束日
        )
        logger.info(f"加载完成: {total}个产品, {len(self.all_sim_dates)}个回测日")

    def _precompute_signals(self, product):
        """预计算历史信号 (同V6)"""
        rd = product.ret_dates
        rets = product.rets
        n = len(rd)
        signals = []

        for i in range(1, n - 持有天数_评估):
            if rets[rd[i]] > 信号阈值_买入 and rets[rd[i - 1]] <= 信号阈值_买入:
                hold_rets = [rets[rd[i + 1 + k]] for k in range(持有天数_评估)]
                avg_ret = float(np.mean(hold_rets))
                eval_end = rd[i + 持有天数_评估]

                persist = 0
                for j in range(i, min(i + 20, n)):
                    if rets[rd[j]] > 信号阈值_买入 * 0.8:
                        persist += 1
                    else:
                        break
                cal_days = ((pd.Timestamp(rd[i + persist - 1])
                             - pd.Timestamp(rd[i])).days + 1) if persist > 0 else 0

                signals.append({
                    'date': rd[i],
                    'eval_end': eval_end,
                    'avg_return': avg_ret,
                    'success': avg_ret > 成功标准,
                    'persist_days': cal_days,
                })

        product.signals = signals

    def build_library(self, as_of_date):
        """构建产品库 + 传统Alpha + DL Alpha 混合"""
        raw = {}
        for key, product in self.products.items():
            valid = [s for s in product.signals if s['eval_end'] < as_of_date]
            if len(valid) < 最少历史信号:
                continue
            success_n = sum(1 for s in valid if s['success'])
            rate = success_n / len(valid) * 100
            if rate < 最低成功率:
                continue
            avg_persist = float(np.mean([s['persist_days'] for s in valid]))
            if avg_persist < 最短持续天数 or avg_persist > 最大持有天数_评估:
                continue
            rets_list = [s['avg_return'] for s in valid]
            avg_ret = float(np.mean(rets_list))
            std_ret = float(np.std(rets_list)) if len(rets_list) > 1 else 1.0
            sharpe = avg_ret / max(std_ret, 0.01)
            raw[key] = {
                'success_rate': round(rate, 1),
                'signal_count': len(valid),
                'avg_return': round(avg_ret, 2),
                'avg_persist': round(avg_persist, 1),
                'ret_std': round(std_ret, 3),
                'sharpe': round(sharpe, 3),
            }

        if not raw:
            self.library = {}
            self.library_date = as_of_date
            return {}

        # 传统因子标准化
        vals = list(raw.values())
        sr_min = min(v['success_rate'] for v in vals)
        sr_max = max(v['success_rate'] for v in vals)
        ar_min = min(v['avg_return'] for v in vals)
        ar_max = max(v['avg_return'] for v in vals)
        sh_min = min(v['sharpe'] for v in vals)
        sh_max = max(v['sharpe'] for v in vals)
        ps_min = min(v['avg_persist'] for v in vals)
        ps_max = max(v['avg_persist'] for v in vals)

        def norm(x, lo, hi):
            return (x - lo) / (hi - lo) if hi > lo else 0.5

        lib = {}
        for key, info in raw.items():
            # 传统Alpha
            trad_alpha = (W_SUCCESS * norm(info['success_rate'], sr_min, sr_max) +
                          W_AVG_RET * norm(info['avg_return'], ar_min, ar_max) +
                          W_SHARPE  * norm(info['sharpe'], sh_min, sh_max) +
                          W_PERSIST * norm(info['avg_persist'], ps_min, ps_max))

            # [V7.1] DL Alpha 自适应混合
            dl_score = self.dl_alpha.get_alpha_score(key)
            dl_confidence = self.dl_alpha.get_confidence(key)
            dl_weight = self.dl_alpha.get_adaptive_weight()

            if dl_confidence > DL_CONFIDENCE_THRESHOLD and dl_score > 0:
                # 有可信的DL预测: 自适应混合权重
                combined_alpha = (1 - dl_weight) * trad_alpha + dl_weight * dl_score
            else:
                # DL不可信: 只用传统
                combined_alpha = trad_alpha

            info['trad_alpha'] = round(trad_alpha, 4)
            info['dl_alpha'] = round(dl_score, 4)
            info['dl_confidence'] = round(dl_confidence, 4)
            info['alpha'] = round(combined_alpha, 4)
            info['entry_quality'] = round(self.dl_alpha.get_entry_quality(key), 4)
            info['hold_signal'] = round(self.dl_alpha.get_hold_signal(key), 4)
            lib[key] = info

        self.library = lib
        self.library_date = as_of_date
        return lib

    def get_latest_nav(self, product, as_of_date):
        idx = bisect.bisect_right(product.nav_dates, as_of_date) - 1
        if idx >= 0:
            d = product.nav_dates[idx]
            return d, product.navs[d]
        return None, None

    def _process_pending_buys(self, date, buy_count):
        new_buys = 0
        for key, signal_date, reason in self.pending_buys:
            if key in self.positions:
                self.pending_buy_keys.discard(key)
                continue

            product = self.products[key]
            buy_nav = product.navs.get(date)
            if buy_nav is None:
                _, buy_nav = self.get_latest_nav(product, date)
            if buy_nav is None:
                self.pending_buy_keys.discard(key)
                continue

            lib_info = self.library.get(key, {})
            alpha = lib_info.get('alpha', 0.5)
            dl_alpha = lib_info.get('dl_alpha', 0.0)
            target_amount = self._compute_position_size(alpha)
            amount = min(target_amount, self.cash)
            if amount < 单仓基础 * 0.3:
                self.pending_buy_keys.discard(key)
                continue

            entry_ret = product.rets.get(signal_date, 信号阈值_买入)

            self.cash -= amount
            self.positions[key] = Position(key, signal_date, date, buy_nav, amount,
                                           entry_return=entry_ret, alpha_score=alpha,
                                           dl_alpha_score=dl_alpha)
            self.pending_buy_keys.discard(key)
            new_buys += 1
            self.trade_log.append({
                'date': date, 'action': '买入确认',
                'bank': product.bank, 'code': product.code,
                'name': product.name, 'nav': buy_nav,
                'amount': amount, 'pnl': 0,
                'hold_days': 0,
                'reason': f'T+1确认(信号{signal_date}) {reason}',
            })

        self.pending_buys = []
        return new_buys

    def _process_pending_sells(self, date, sell_count):
        new_sells = 0
        for key, signal_date, reason in self.pending_sells:
            if key not in self.positions:
                self.pending_sell_keys.discard(key)
                continue

            pos = self.positions.pop(key)
            product = self.products[key]
            sell_nav = product.navs.get(date)
            if sell_nav is None:
                _, sell_nav = self.get_latest_nav(product, date)
            if sell_nav is None:
                sell_nav = pos.buy_nav

            pos.sell_date = date
            pos.sell_nav = sell_nav
            pos.pnl = pos.amount * (sell_nav / pos.buy_nav - 1)
            pos.sell_reason = reason
            self.cash += pos.amount + pos.pnl
            self.closed_trades.append(pos)
            self.pending_sell_keys.discard(key)
            new_sells += 1

            hold_td = self._trading_days_held(pos.confirm_date, date)
            self.trade_log.append({
                'date': date, 'action': '赎回到账',
                'bank': product.bank, 'code': product.code,
                'name': product.name, 'nav': sell_nav,
                'amount': pos.amount + pos.pnl,
                'pnl': pos.pnl,
                'hold_days': hold_td,
                'reason': f'T+1到账(提交{signal_date}) {reason}',
            })

        self.pending_sells = []
        return new_sells

    def _trading_days_held(self, confirm_date, current_date):
        ci = self.date_to_idx.get(confirm_date)
        di = self.date_to_idx.get(current_date)
        if ci is not None and di is not None:
            return di - ci
        return (pd.Timestamp(current_date) - pd.Timestamp(confirm_date)).days

    def _is_pre_holiday(self, di):
        if di + 1 >= len(self.all_sim_dates):
            return True
        next_date = self.all_sim_dates[di + 1]
        gap = (pd.Timestamp(next_date) - pd.Timestamp(self.all_sim_dates[di])).days
        return gap > 长假阈值_天

    def _check_sells(self, date):
        """[ENHANCED] DL-智能卖出决策"""
        for key in list(self.positions):
            if key in self.pending_sell_keys:
                continue
            pos = self.positions[key]
            p = self.products[key]

            hold_td = self._trading_days_held(pos.confirm_date, date)
            if hold_td < 最短持有交易日_赎回:
                continue

            ret = p.rets.get(date)

            # [NEW] DL-adjusted 动态卖出阈值
            dl_hold = self.dl_alpha.get_hold_signal(key)
            dl_conf = self.dl_alpha.get_confidence(key)

            # 如果DL高置信预测持续释放 → 降低卖出阈值(更不容易卖)
            effective_sell_thresh = 信号阈值_卖出
            effective_max_hold = 最大持有交易日

            if dl_conf > DL_CONFIDENCE_THRESHOLD and dl_hold > 0.4:
                # DL预测持续释放 → 放宽卖出条件
                effective_sell_thresh = 信号阈值_卖出_DL低
                effective_max_hold = 最大持有交易日_DL
            elif dl_conf > DL_CONFIDENCE_THRESHOLD and dl_hold < 0.15:
                # DL预测释放结束 → 收紧卖出条件
                effective_sell_thresh = 信号阈值_卖出 + DL_SELL_TIGHTEN

            # ① 硬阈值卖出
            if ret is not None and ret <= effective_sell_thresh:
                self.pending_sells.append((key, date,
                    f'收益{ret:.1f}%≤{effective_sell_thresh}%(持有{hold_td}日,DL_hold={dl_hold:.2f})'))
                self.pending_sell_keys.add(key)
                continue

            # ② 最大持有卖出
            if hold_td >= effective_max_hold:
                self.pending_sells.append((key, date,
                    f'持有{hold_td}日达上限{effective_max_hold}'))
                self.pending_sell_keys.add(key)
                continue

    def _compute_position_size(self, alpha_score):
        """[ENHANCED] 基于组合Alpha的动态仓位"""
        ratio = 单仓最小倍率 + (单仓最大倍率 - 单仓最小倍率) * min(alpha_score, 1.0)
        return int(单仓基础 * ratio)

    def _find_and_queue_buys(self, date, slots, di):
        """[ENHANCED] DL Alpha 排名 + 动态买入阈值"""
        if self._is_pre_holiday(di):
            return 0

        signals = []
        for key, info in self.library.items():
            if key in self.positions or key in self.pending_buy_keys or key in self.pending_sell_keys:
                continue

            p = self.products[key]
            r = p.rets.get(date)
            if r is None:
                continue

            # [NEW] DL-adjusted 动态买入阈值
            dl_entry = info.get('entry_quality', 0)
            dl_conf = info.get('dl_confidence', 0)
            effective_buy_thresh = 信号阈值_买入

            if dl_conf > DL_CONFIDENCE_THRESHOLD and dl_entry > 0.5:
                # DL高置信预测即将释放 → 降低买入门槛 [V7.1]
                effective_buy_thresh = 信号阈值_买入_DL低

            if r <= effective_buy_thresh:
                continue

            if 需要突破信号:
                idx = p.ret_date_idx.get(date)
                if idx is None or idx == 0:
                    continue
                prev_r = p.rets[p.ret_dates[idx - 1]]
                if prev_r > effective_buy_thresh:
                    continue

            alpha = info.get('alpha', 0.5)
            signals.append((key, alpha, r, info['success_rate'], info['avg_return'],
                           dl_entry, dl_conf))

        # [ENHANCED] 排序: 组合Alpha优先, 同分按DL入场质量排
        signals.sort(key=lambda x: (-x[1], -x[5], -x[2]))

        queued = 0
        for key, alpha, ret, sr, avg_r, dl_e, dl_c in signals[:slots]:
            size = self._compute_position_size(alpha)
            if self.cash - size < -单仓基础 * 0.3:
                continue

            reason = (f'α{alpha:.2f}(DL:{dl_e:.2f}/{dl_c:.2f}) '
                     f'成功率{sr}% 收益{ret:.1f}% 仓位{size/1e4:.0f}万')
            self.pending_buys.append((key, date, reason))
            self.pending_buy_keys.add(key)
            queued += 1
            self.trade_log.append({
                'date': date, 'action': '提交申购',
                'bank': self.products[key].bank,
                'code': self.products[key].code,
                'name': self.products[key].name,
                'nav': 0, 'amount': 0, 'pnl': 0,
                'hold_days': 0,
                'reason': f'T日提交 {reason}',
            })
        return queued

    def _record_value(self, date):
        pos_val = 0
        for key, pos in self.positions.items():
            p = self.products[key]
            _, nav = self.get_latest_nav(p, date)
            if nav is not None:
                pos_val += pos.amount * (nav / pos.buy_nav)
            else:
                pos_val += pos.amount
        total = self.cash + pos_val
        if self.daily_values and self.daily_values[-1][0] == date:
            self.daily_values[-1] = (date, total, self.cash, pos_val)
        else:
            self.daily_values.append((date, total, self.cash, pos_val))

    def run(self):
        self.load_data()

        sep = '=' * 70
        print(f"\n{sep}")
        print("   V7.1 DL Alpha增强回测 (深度学习 + 传统因子 + T+1)")
        print(f"   区间: {回测开始日} ~ {回测结束日}")
        print(f"   资金: {初始资金 / 1e8:.1f}亿 | "
              f"仓位: {最大持仓数}×({单仓基础*单仓最小倍率/1e4:.0f}~{单仓基础*单仓最大倍率/1e4:.0f}万)")
        print(f"   Alpha: 自适应权重 DL {DL_ALPHA_WEIGHT_MIN*100:.0f}~{DL_ALPHA_WEIGHT_BASE*100:.0f}%")
        print(f"   DL GPU: {self.dl_alpha.available}")
        print(f"   买入: {信号阈值_买入}%(DL高置信→{信号阈值_买入_DL低}%)")
        print(f"   卖出: {信号阈值_卖出}%(DL持续→{信号阈值_卖出_DL低}%)")
        print(f"   持有: {最大持有交易日}天(DL持续→{最大持有交易日_DL}天)")
        print(f"   成功率门槛: {最低成功率}%")
        print(sep)

        self.date_to_idx = {d: i for i, d in enumerate(self.all_sim_dates)}

        # [NEW] 首次DL训练
        if self.dl_alpha.engine is not None:
            self.dl_alpha.train_and_predict(回测开始日)

        self.build_library(回测开始日)
        logger.info(f"初始产品库: {len(self.library)}个")

        n_days = len(self.all_sim_dates)
        buy_n = sell_n = 0

        for di, date in enumerate(self.all_sim_dates):
            # ① 定期重建产品库 + DL重训
            need_rebuild = (self.library_date is None or
                           (pd.Timestamp(date) - pd.Timestamp(self.library_date)).days >= 产品库重建间隔)

            if need_rebuild:
                # [NEW] Walk-forward DL 重训
                if self.dl_alpha.should_retrain(date):
                    self.dl_alpha.train_and_predict(date)

                old = len(self.library)
                self.build_library(date)
                if len(self.library) != old or di == 0:
                    # 统计DL alpha分布
                    dl_scores = [info.get('dl_alpha', 0) for info in self.library.values()]
                    dl_mean = np.mean(dl_scores) if dl_scores else 0
                    logger.info(f"  [{date}] 产品库: {len(self.library)}个 | "
                               f"DL alpha均值: {dl_mean:.3f}")

            # ② T+1确认
            nb = self._process_pending_buys(date, buy_n)
            buy_n += nb
            ns = self._process_pending_sells(date, sell_n)
            sell_n += ns

            # ③ DL-智能卖出检查
            self._check_sells(date)

            # ④ DL-智能买入
            used = len(self.positions) + len(self.pending_buys)
            slots = 最大持仓数 - used
            if slots > 0:
                self._find_and_queue_buys(date, slots, di)

            self._record_value(date)

            if (di + 1) % 50 == 0 or di == n_days - 1:
                v = self.daily_values[-1][1]
                r = (v / 初始资金 - 1) * 100
                logger.info(f"  [{date}] {di+1}/{n_days} | "
                           f"持仓{len(self.positions)} 待买{len(self.pending_buys)} | "
                           f"买{buy_n} 卖{sell_n} | {v/1e4:,.0f}万({r:+.2f}%)")

        # 强制平仓
        last = self.all_sim_dates[-1]
        for key in list(self.positions):
            pos = self.positions.pop(key)
            p = self.products[key]
            _, nav = self.get_latest_nav(p, last)
            if nav is None:
                nav = pos.buy_nav
            pos.sell_date = last
            pos.sell_nav = nav
            pos.pnl = pos.amount * (nav / pos.buy_nav - 1)
            pos.sell_reason = '回测结束平仓'
            self.cash += pos.amount + pos.pnl
            self.closed_trades.append(pos)
        self._record_value(last)

        self._print_summary()

    def _monthly_returns(self):
        monthly = {}
        for date, value, _, _ in self.daily_values:
            monthly[date[:7]] = value
        result = []
        prev = 初始资金
        for m in sorted(monthly):
            cur = monthly[m]
            result.append((m, (cur / prev - 1) * 100))
            prev = cur
        return result

    def _print_summary(self):
        sep = '=' * 70
        if not self.daily_values:
            print("无回测数据")
            return

        fv = self.daily_values[-1][1]
        total_ret = (fv / 初始资金 - 1) * 100
        t0 = pd.Timestamp(self.all_sim_dates[0])
        t1 = pd.Timestamp(self.all_sim_dates[-1])
        yrs = (t1 - t0).days / 365
        ann_ret = ((fv / 初始资金) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0

        peak = 初始资金
        max_dd = 0
        dd_date = ''
        for d, v, _, _ in self.daily_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
                dd_date = d

        nt = len(self.closed_trades)
        nw = sum(1 for t in self.closed_trades if t.pnl > 0)
        nl = nt - nw
        wr = nw / nt * 100 if nt else 0
        total_pnl = sum(t.pnl for t in self.closed_trades)

        hold_days_list = [self._trading_days_held(t.confirm_date, t.sell_date)
                          for t in self.closed_trades]
        avg_hold = float(np.mean(hold_days_list)) if hold_days_list else 0

        daily_rets = []
        for i in range(1, len(self.daily_values)):
            pv = self.daily_values[i - 1][1]
            cv = self.daily_values[i][1]
            if pv > 0:
                daily_rets.append(cv / pv - 1)
        sharpe = 0
        if daily_rets:
            rf = 0.025 / 252
            excess = np.array(daily_rets) - rf
            if np.std(excess) > 0:
                sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

        bm_val = 初始资金 * (1 + 0.025 * yrs)

        print(f"\n{sep}")
        print("      V7.1 DL Alpha增强 回测结果")
        print(sep)
        print(f"  回测区间:    {self.all_sim_dates[0]} ~ {self.all_sim_dates[-1]}")
        print(f"  交易日数:    {len(self.all_sim_dates)}")
        print(f"  初始资金:    {初始资金/1e4:,.0f}万")
        print(f"  最终市值:    {fv/1e4:,.0f}万")
        print(f"  总收益率:    {total_ret:+.4f}%")
        print(f"  年化收益率:  {ann_ret:+.4f}%")
        print(f"  夏普比率:    {sharpe:.4f}")
        print(f"  最大回撤:    {max_dd:.4f}% ({dd_date})")
        print(f"  总交易次数:  {nt}")
        print(f"  胜率:        {wr:.1f}%")
        print(f"  平均持有:    {avg_hold:.1f}交易日")
        print(f"  总盈亏:      {total_pnl/1e4:+,.2f}万")
        print(f"  基准(2.5%):  {(bm_val/初始资金-1)*100:+.4f}%")
        print(f"  超额收益:    {(total_ret-(bm_val/初始资金-1)*100):+.4f}%")
        print(f"  DL Alpha:    {'GPU' if self.dl_alpha.available else '统计回退'}")

        print(f"\n  【月度收益率】")
        for m, r in self._monthly_returns():
            bar = '+' * max(0, int(r * 20)) if r >= 0 else '-' * min(20, int(abs(r) * 20))
            print(f"  {m}: {r:+.4f}%  {bar}")

        # 按银行统计
        bank_stats = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
        for t in self.closed_trades:
            b = self.products[t.product_key].bank
            bank_stats[b]['n'] += 1
            bank_stats[b]['pnl'] += t.pnl
            if t.pnl > 0:
                bank_stats[b]['wins'] += 1
        print(f"\n  【按银行统计】")
        for b, s in sorted(bank_stats.items()):
            wr2 = s['wins'] / s['n'] * 100 if s['n'] else 0
            print(f"  {b}: {s['n']}笔 | 盈亏{s['pnl']/1e4:+,.2f}万 | 胜率{wr2:.0f}%")

        print(sep)


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    try:
        engine = BacktestEngine()
        engine.run()
    except Exception as e:
        logger.error(f"回测异常终止: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
