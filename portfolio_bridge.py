# -*- coding: utf-8 -*-
"""
持仓管理适配层 — 连接 Flask API 与 我的持仓_模板.xlsx

提供：交易录入、交易历史、产品补全、OCR 识别、持仓收益计算。
"""

import os
import logging
import traceback
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_PATH = os.path.join(BASE_DIR, "我的持仓_模板.xlsx")
NAV_DB_PATH = os.path.join(BASE_DIR, "净值数据库.xlsx")

# 净值数据库缓存（加载耗时约60秒，缓存后秒级响应）
_nav_db_cache = None
_nav_db_mtime = None
_nav_db_lock = threading.Lock()

def _get_nav_db():
    """获取净值数据库实例，带文件修改时间缓存 + 线程锁"""
    global _nav_db_cache, _nav_db_mtime
    try:
        current_mtime = os.path.getmtime(NAV_DB_PATH) if os.path.exists(NAV_DB_PATH) else None
    except OSError:
        current_mtime = None

    if _nav_db_cache is not None and current_mtime == _nav_db_mtime:
        return _nav_db_cache

    with _nav_db_lock:
        # 双重检查：可能在等锁期间其他线程已加载完毕
        if _nav_db_cache is not None and current_mtime == _nav_db_mtime:
            return _nav_db_cache
        from nav_db_excel import NAVDatabaseExcel
        logger.info("加载净值数据库（首次或文件已更新）...")
        _nav_db_cache = NAVDatabaseExcel()
        _nav_db_mtime = current_mtime
        logger.info("净值数据库加载完成")
    return _nav_db_cache


# ══════════════════════════════════════════
# 交易记录 CRUD
# ══════════════════════════════════════════

def get_trade_history() -> List[Dict]:
    """读取全部交易记录"""
    if not os.path.exists(PORTFOLIO_PATH):
        return []
    try:
        df = pd.read_excel(PORTFOLIO_PATH)
        required = {'银行', '产品代码', '产品名称', '交易', '金额', '日期'}
        if not required.issubset(set(df.columns)):
            return []
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
        records = df.to_dict('records')
        return records
    except Exception as e:
        logger.error(f"读取交易记录失败: {e}")
        return []


def add_trade(bank: str, product_code: str, product_name: str,
              trade_type: str, amount: float, date: str) -> Dict:
    """添加单笔交易"""
    try:
        new_row = pd.DataFrame([{
            '银行': bank,
            '产品代码': product_code,
            '产品名称': product_name,
            '交易': trade_type,
            '金额': float(amount),
            '日期': date,
        }])

        if os.path.exists(PORTFOLIO_PATH):
            df = pd.read_excel(PORTFOLIO_PATH)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_excel(PORTFOLIO_PATH, index=False)
        logger.info(f"添加交易: {bank} {product_code} {trade_type} {amount}")
        return {'ok': True, 'message': '交易已录入'}
    except Exception as e:
        logger.error(f"添加交易失败: {e}")
        return {'ok': False, 'message': str(e)}


def add_trades_batch(trades: List[Dict]) -> Dict:
    """批量添加交易（OCR 识别结果）"""
    try:
        new_rows = []
        for t in trades:
            new_rows.append({
                '银行': t.get('bank', ''),
                '产品代码': t.get('product_code', ''),
                '产品名称': t.get('product_name', ''),
                '交易': t.get('trade_type', '买入'),
                '金额': float(t.get('amount', 0)),
                '日期': t.get('date', ''),
            })

        new_df = pd.DataFrame(new_rows)

        if os.path.exists(PORTFOLIO_PATH):
            df = pd.read_excel(PORTFOLIO_PATH)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df

        df.to_excel(PORTFOLIO_PATH, index=False)
        logger.info(f"批量添加交易: {len(new_rows)} 笔")
        return {'ok': True, 'message': f'已录入 {len(new_rows)} 笔交易', 'count': len(new_rows)}
    except Exception as e:
        logger.error(f"批量添加失败: {e}")
        return {'ok': False, 'message': str(e)}


# ══════════════════════════════════════════
# 产品自动补全
# ══════════════════════════════════════════

def get_product_suggestions(query: str, limit: int = 20) -> List[Dict]:
    """根据关键字搜索产品（从净值数据库中）"""
    if not query or len(query) < 1:
        return []

    try:
        db = _get_nav_db()
        results = []
        query_lower = query.lower()

        for sheet_name, df in db.data.items():
            if not isinstance(df.index, pd.MultiIndex):
                continue
            for idx in df.index:
                code, name = idx[0], idx[1]
                if (query_lower in str(code).lower() or
                        query_lower in str(name).lower()):
                    results.append({
                        'bank': sheet_name,
                        'product_code': code,
                        'product_name': name,
                    })
                    if len(results) >= limit:
                        return results
        return results
    except Exception as e:
        logger.error(f"产品搜索失败: {e}")
        return []


def get_product_name(bank: str, code: str) -> str:
    """根据银行和产品代码查询产品名称"""
    try:
        db = _get_nav_db()
        sheet_name = db._get_sheet_name(bank)
        if sheet_name not in db.data:
            return ''
        df = db.data[sheet_name]
        if not isinstance(df.index, pd.MultiIndex):
            return ''
        for idx in df.index:
            if idx[0] == code:
                return idx[1]
        return ''
    except Exception:
        return ''


# ══════════════════════════════════════════
# OCR 识别（简化实现 — 提取表格文本）
# ══════════════════════════════════════════

def parse_portfolio_image(image_bytes: bytes) -> Dict:
    """OCR 识别持仓截图

    尝试使用 pytesseract；若不可用则返回提示信息。
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')

        # 尝试从文本中解析表格行
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        trades = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                trades.append({
                    'bank': parts[0] if len(parts) > 0 else '',
                    'product_code': parts[1] if len(parts) > 1 else '',
                    'product_name': parts[2] if len(parts) > 2 else '',
                    'trade_type': '买入',
                    'amount': parts[3] if len(parts) > 3 else '0',
                    'date': parts[4] if len(parts) > 4 else datetime.now().strftime('%Y-%m-%d'),
                    'confidence': 0.6,
                })

        return {
            'ok': True,
            'raw_text': text,
            'trades': trades,
            'message': f'识别到 {len(trades)} 条可能的交易记录',
        }

    except ImportError:
        return {
            'ok': False,
            'message': 'OCR 功能需要安装 pytesseract 和 Pillow：pip install pytesseract Pillow',
            'trades': [],
        }
    except Exception as e:
        logger.error(f"OCR 识别失败: {e}")
        return {
            'ok': False,
            'message': f'识别失败: {str(e)}',
            'trades': [],
        }


# ══════════════════════════════════════════
# 核心：持仓收益计算
# ══════════════════════════════════════════

def get_holding_returns() -> Dict:
    """计算持仓收益明细 + 时间序列

    从交易记录 + 净值数据库，逐日计算：
    - 每个持有产品的：份额、市值、每日收益、累计收益、收益率
    - 组合汇总
    - 每日时间序列

    Returns:
        {
            'holdings': [...],         # 产品级持仓明细
            'summary': {...},          # 组合汇总 KPI
            'daily_series': [...],     # 每日时间序列
        }
    """
    try:
        # 1. 读取交易记录
        if not os.path.exists(PORTFOLIO_PATH):
            return {'holdings': [], 'summary': _empty_summary(), 'daily_series': []}

        df = pd.read_excel(PORTFOLIO_PATH)
        required = {'银行', '产品代码', '产品名称', '交易', '金额', '日期'}
        if not required.issubset(set(df.columns)):
            return {'holdings': [], 'summary': _empty_summary(), 'daily_series': []}

        df['日期'] = pd.to_datetime(df['日期'])
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
        df = df[df['交易'].isin(['买入', '卖出'])].copy()
        df = df.sort_values('日期')

        if df.empty:
            return {'holdings': [], 'summary': _empty_summary(), 'daily_series': []}

        # 2. 加载净值数据库（使用缓存避免重复加载）
        nav_db = _get_nav_db()

        # 3. 加载策略数据（用于生成持仓建议）
        strategy_ctx = _load_strategy_context()

        # 4. 按产品分组计算
        holdings = []
        all_daily = {}  # date_str -> {cost, market_value, daily_pnl}

        grouped = df.groupby(['银行', '产品代码', '产品名称'])

        for (bank, code, name), trades in grouped:
            result = _calc_product_holding(bank, code, name, trades, nav_db, all_daily, strategy_ctx)
            if result:
                holdings.append(result)

        # 4. 汇总 KPI
        total_cost = sum(h['cost'] for h in holdings)
        total_market_value = sum(h['market_value'] for h in holdings)
        total_pnl = total_market_value - total_cost
        total_return_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        # 计算累计年化收益率（基于最早买入日到今天的天数）
        active_holdings = [h for h in holdings if h['status'] == '持有中']
        max_holding_days = max((h['holding_days'] for h in active_holdings), default=0)
        total_ann_return_pct = (total_return_pct / max_holding_days * 365) if max_holding_days > 0 else 0

        # 今日收益 = 所有持有中产品的日收益之和
        today_pnl = sum(h.get('daily_return', 0) for h in active_holdings)

        summary = {
            'total_cost': round(total_cost, 2),
            'total_market_value': round(total_market_value, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 4),
            'total_ann_return_pct': round(total_ann_return_pct, 4),
            'today_pnl': round(today_pnl, 2),
            'holding_count': len(active_holdings),
            'product_count': len(holdings),
        }

        # 5. 构建每日时间序列（含逐日收益率、年化收益率）
        _finalize_daily_series(all_daily)
        daily_series = []
        if all_daily:
            sorted_dates = sorted(all_daily.keys())
            first_date_str = sorted_dates[0] if sorted_dates else None
            prev_cum_pnl = None
            prev_cost = None
            for date_str in sorted_dates:
                d = all_daily[date_str]
                mv = d.get('market_value', 0)
                cost = d.get('cost', 0)
                cum_pnl = d.get('cumulative_pnl', 0)
                cum_ret_pct = d.get('cumulative_return_pct', 0)

                # 逐日收益(元) = 当日累计收益 - 前日累计收益（稳定，不受产品覆盖差异影响）
                daily_pnl = (cum_pnl - prev_cum_pnl) if prev_cum_pnl is not None else 0
                # 日收益率% = 日收益 / 前日成本 * 100
                daily_return_pct = (daily_pnl / prev_cost * 100) if prev_cost and prev_cost > 0 else 0
                # 日年化收益率%
                daily_ann_return_pct = daily_return_pct * 365

                # 累计年化收益率%：按首日到当日的持仓天数折算
                if first_date_str:
                    days_held = (pd.Timestamp(date_str) - pd.Timestamp(first_date_str)).days
                    cumulative_ann_return_pct = (cum_ret_pct / days_held * 365) if days_held > 0 else 0
                else:
                    cumulative_ann_return_pct = 0

                daily_series.append({
                    'date': date_str,
                    'cost': round(float(cost), 2),
                    'market_value': round(float(mv), 2),
                    'daily_pnl': round(float(daily_pnl), 2),
                    'daily_return_pct': round(float(daily_return_pct), 4),
                    'daily_ann_return_pct': round(float(daily_ann_return_pct), 4),
                    'cumulative_pnl': round(float(cum_pnl), 2),
                    'cumulative_return_pct': round(float(cum_ret_pct), 4),
                    'cumulative_ann_return_pct': round(float(cumulative_ann_return_pct), 4),
                })
                prev_cum_pnl = cum_pnl
                prev_cost = cost

        return {
            'holdings': holdings,
            'summary': summary,
            'daily_series': daily_series,
        }

    except Exception as e:
        logger.error(f"持仓收益计算失败: {e}\n{traceback.format_exc()}")
        return {'holdings': [], 'summary': _empty_summary(), 'daily_series': []}


def _empty_summary():
    return {
        'total_cost': 0,
        'total_market_value': 0,
        'total_pnl': 0,
        'total_return_pct': 0,
        'total_ann_return_pct': 0,
        'holding_count': 0,
        'product_count': 0,
    }


def _calc_product_holding(bank, code, name, trades_df, nav_db, all_daily, strategy_ctx=None):
    """计算单个产品的持仓收益"""
    try:
        # 获取该产品的净值数据
        sheet_name = nav_db._get_sheet_name(bank)
        nav_data = {}  # date_str -> nav_value

        if sheet_name in nav_db.data:
            df_nav = nav_db.data[sheet_name]
            if isinstance(df_nav.index, pd.MultiIndex):
                # 先对索引排序避免 lexsort 性能警告
                if not df_nav.index.is_monotonic_increasing:
                    df_nav = df_nav.sort_index()
                for idx in df_nav.index:
                    if idx[0] == code:
                        row = df_nav.loc[[idx]]  # 用列表索引返回DataFrame避免scalar access问题
                        for col in row.columns:
                            if nav_db._is_date_column(col):
                                val = row[col].iloc[0]
                                if pd.notna(val) and str(val).strip():
                                    try:
                                        nav_data[col] = float(val)
                                    except (ValueError, TypeError):
                                        pass
                        break

        # 计算交易汇总
        buys = trades_df[trades_df['交易'] == '买入']
        sells = trades_df[trades_df['交易'] == '卖出']

        total_buy = buys['金额'].sum()
        total_sell = sells['金额'].sum()
        net_cost = total_buy - total_sell

        first_buy_date = buys['日期'].min() if not buys.empty else None
        last_trade_date = trades_df['日期'].max()

        # 状态判断
        status = '持有中' if net_cost > 0 else '已清仓'

        # 如果有净值数据，计算基于净值的收益
        market_value = net_cost  # 默认
        latest_nav = None
        buy_nav = None
        daily_return = 0
        cumulative_return_pct = 0
        ann_return_pct = 0
        holding_days = 0

        if first_buy_date is not None:
            holding_days = (datetime.now() - pd.Timestamp(first_buy_date).to_pydatetime()).days

        if nav_data and net_cost > 0:
            sorted_dates = sorted(nav_data.keys())

            # 找到买入日附近的净值作为基准
            if first_buy_date is not None:
                buy_date_str = first_buy_date.strftime('%Y-%m-%d')
                buy_nav = _find_nearest_nav(nav_data, buy_date_str, sorted_dates)

            # 最新净值
            if sorted_dates:
                latest_nav = nav_data[sorted_dates[-1]]

            # 计算收益
            if buy_nav and latest_nav and buy_nav > 0:
                nav_return = (latest_nav / buy_nav - 1)
                market_value = net_cost * (1 + nav_return)
                cumulative_return_pct = nav_return * 100

                # 年化收益率
                if holding_days > 0:
                    ann_return_pct = (cumulative_return_pct / holding_days * 365)

                # 日收益（最近两天净值差）
                if len(sorted_dates) >= 2:
                    prev_nav = nav_data[sorted_dates[-2]]
                    if prev_nav and prev_nav > 0:
                        daily_return = net_cost * (latest_nav / prev_nav - 1)

            # 填充每日时间序列
            if buy_nav and buy_nav > 0 and first_buy_date:
                buy_date_str = first_buy_date.strftime('%Y-%m-%d')
                running_cost = 0
                for d in sorted_dates:
                    if d < buy_date_str:
                        continue
                    nav_val = nav_data[d]
                    if nav_val is None or nav_val <= 0:
                        continue

                    # 计算截至该日的累计买入/卖出
                    trades_before = trades_df[trades_df['日期'] <= pd.Timestamp(d)]
                    cost_at_date = (trades_before[trades_before['交易'] == '买入']['金额'].sum() -
                                    trades_before[trades_before['交易'] == '卖出']['金额'].sum())
                    if cost_at_date <= 0:
                        continue

                    mv_at_date = cost_at_date * (nav_val / buy_nav)
                    pnl_at_date = mv_at_date - cost_at_date

                    if d not in all_daily:
                        all_daily[d] = {'cost': 0, 'market_value': 0, 'daily_pnl': 0,
                                        'cumulative_pnl': 0, 'cumulative_return_pct': 0}
                    all_daily[d]['cost'] += cost_at_date
                    all_daily[d]['market_value'] += mv_at_date
                    all_daily[d]['cumulative_pnl'] += pnl_at_date

        # 检查是否在产品库中
        if strategy_ctx is None:
            strategy_ctx = {}
        in_lib = code in strategy_ctx.get('product_lib', {})

        # 持仓建议（基于策略数据）
        advice = _generate_advice(code, status, cumulative_return_pct,
                                  ann_return_pct, holding_days, in_lib, strategy_ctx)

        return {
            'bank': bank,
            'product_code': code,
            'product_name': name if name else '',
            'status': status,
            'cost': round(net_cost, 2),
            'market_value': round(market_value, 2),
            'cumulative_pnl': round(market_value - net_cost, 2),
            'cumulative_return_pct': round(cumulative_return_pct, 4),
            'holding_days': holding_days,
            'ann_return_pct': round(ann_return_pct, 4),
            'daily_return': round(daily_return, 2),
            'in_lib': in_lib,
            'advice': advice,
            'latest_nav': round(latest_nav, 6) if latest_nav else None,
            'buy_nav': round(buy_nav, 6) if buy_nav else None,
        }

    except Exception as e:
        logger.error(f"计算产品持仓失败 {bank}/{code}: {e}")
        return None


def _load_strategy_context() -> Dict:
    """加载策略引擎数据，用于生成持仓建议"""
    ctx = {'opportunities': {}, 'top20': {}, 'patterns': {}, 'product_lib': {}}
    try:
        import strategy_bridge
        # 实时信号 — 按产品代码索引
        for opp in strategy_bridge.get_opportunities():
            code = opp.get('产品代码', '')
            if code:
                ctx['opportunities'][code] = opp
        # 精选推荐 — 按产品代码索引
        for rec in strategy_bridge.get_top20():
            code = rec.get('产品代码', '')
            if code:
                ctx['top20'][code] = rec
        # 释放规律 — 按产品代码索引
        for pat in strategy_bridge.get_patterns():
            code = pat.get('code', '')
            if code:
                ctx['patterns'][code] = pat
        # 产品库 — 按产品代码索引
        lib = strategy_bridge._get('product_lib', [])
        if lib:
            for p in lib:
                code = p.get('产品代码', '')
                if code:
                    ctx['product_lib'][code] = p
    except Exception as e:
        logger.warning(f"加载策略数据失败（建议功能降级）: {e}")
    return ctx


def _generate_advice(code, status, cumulative_return_pct, ann_return_pct,
                     holding_days, in_lib, strategy_ctx) -> str:
    """基于策略数据生成持仓建议

    优先级：实时信号 > 精选推荐 > 释放规律 > 收益率兜底
    """
    if status != '持有中':
        return '已清仓'

    opp = strategy_ctx.get('opportunities', {}).get(code)
    rec = strategy_ctx.get('top20', {}).get(code)
    pat = strategy_ctx.get('patterns', {}).get(code)

    # ── 1. 有实时信号 ──
    if opp:
        action = opp.get('操作建议', '')
        if '★★★' in action:
            return f'买入 ★★★ 强烈推荐加仓（今日新信号，成功率{opp.get("历史成功率%", 0):.0f}%）'
        elif '★★' in action:
            days = opp.get('信号距今天数', 0)
            return f'买入 ★★ 推荐加仓（信号第{days}天，成功率{opp.get("历史成功率%", 0):.0f}%）'
        elif '★' in action:
            days = opp.get('信号距今天数', 0)
            return f'买入 ★ 可加仓（信号第{days}天，仍在高收益窗口）'
        elif '关注' in action or '☆' in action:
            return '持有 — 信号接近买入门槛，继续观察'
        elif '观望' in action or '已结束' in action:
            return '卖出 — 高收益窗口已结束，建议止盈'

    # ── 2. 在精选推荐中 ──
    if rec:
        score = rec.get('综合得分', 0)
        success = rec.get('历史成功率%', 0)
        # 有预测释放日
        pred_date = rec.get('预测释放日', '')
        if pred_date:
            conf = rec.get('预测置信度', 0)
            return f'持有 — 精选推荐（得分{score:.1f}），预测{pred_date}释放（置信{conf:.0%}）'
        return f'持有 — 精选推荐Top20（得分{score:.1f}，成功率{success:.0f}%）'

    # ── 3. 有释放规律 ──
    if pat and pat.get('confidence', 0) > 0.3:
        conf = pat['confidence']
        pred = pat.get('prediction', {})
        if pred and pred.get('predicted_date'):
            td = pred.get('td_until', 0)
            if 0 < td <= 10:
                return f'持有 — 预测{td}天后释放（置信{conf:.0%}，{pat.get("top_phase", "")}）'
        if pat.get('has_period'):
            period = pat.get('period_days', 0)
            return f'持有 — 有{period:.0f}天周期规律（置信{conf:.0%}），等待下次释放'
        return f'持有 — 检测到释放规律（置信{conf:.0%}），建议继续观察'

    # ── 4. 在产品库中但无信号 ──
    if in_lib:
        return '持有 — 高成功率产品，等待下次买入信号'

    # ── 5. 兜底：基于收益率 ──
    if cumulative_return_pct > 3:
        return '卖出 — 非库产品，收益较好可止盈'
    elif cumulative_return_pct > 0:
        return '观望 — 非库产品，小幅盈利中'
    elif cumulative_return_pct > -1:
        return '观望 — 非库产品，微亏耐心等待'
    else:
        return '卖出 — 非库产品，持续亏损建议赎回'


def _find_nearest_nav(nav_data, target_date, sorted_dates):
    """查找最接近目标日期的净值"""
    # 先找精确匹配
    if target_date in nav_data:
        return nav_data[target_date]

    # 找之后最近的
    for d in sorted_dates:
        if d >= target_date:
            return nav_data[d]

    # 找之前最近的
    for d in reversed(sorted_dates):
        if d <= target_date:
            return nav_data[d]

    return None


# 计算组合的每日累计收益率
def _finalize_daily_series(all_daily):
    """为汇总的每日数据计算累计收益率"""
    for date_str in sorted(all_daily.keys()):
        d = all_daily[date_str]
        if d['cost'] > 0:
            d['cumulative_return_pct'] = (d['cumulative_pnl'] / d['cost']) * 100
        else:
            d['cumulative_return_pct'] = 0
