# -*- coding: utf-8 -*-
"""
共享新鲜度模块 — 回测 (backtest_pattern_v4) 和推荐系统 (bank_product_strategy_v6) 共用

改参数只需改这里，两边自动生效。
"""

# ===== 信号新鲜度参数 =====
信号新鲜度_最大进度 = 0.5       # 正常窗口: 仅允许前50%进入
信号新鲜度_连续窗口阈值 = 30    # avg_window > 此值 = "连续释放"
信号新鲜度_连续最大进度 = 0.1   # 连续释放: 仅前10%
收益加速度_窗口 = 5             # velocity回溯交易日
收益加速度_权重 = 0.10          # velocity在composite中的权重
信号新鲜度_启用 = True          # 主开关


def calc_signal_freshness(ret_dates, rets, obs_date, obs_idx, pattern, threshold):
    """统一的信号新鲜度计算函数

    Args:
        ret_dates: 排序后的收益日期列表
        rets:      {date_str: float} 或支持 .get(date, 0) 的收益率映射
        obs_date:  观察日 (信号日的前一天 / 最新有效日期)
        obs_idx:   obs_date 在 ret_dates 中的索引 (int), 若为 None 则自动查找
        pattern:   ReleasePattern 对象 (可为 None)
        threshold: 连续高收益天数的判断阈值 (通常 = 释放识别阈值 * 0.5)

    Returns:
        dict 或 None
    """
    if obs_date is None or not ret_dates:
        return None

    # 解析索引
    ret_idx = obs_idx
    if ret_idx is None:
        try:
            ret_idx = ret_dates.index(obs_date)
        except ValueError:
            return None

    # 向前扫描连续高收益天数 (有无 pattern 都需要)
    days_since_start = 0
    for i in range(ret_idx, -1, -1):
        d = ret_dates[i]
        r = rets.get(d, 0) if isinstance(rets, dict) else rets.get(d, 0)
        if r > threshold:
            days_since_start += 1
        else:
            break

    # ── Scenario B: 无 pattern → "Flash Opportunity" 规则 ──
    # 假设默认窗口5天, 仅允许 Day 0-3 进入 (刚出现的尖峰)
    if pattern is None:
        default_window = 5
        is_fresh = days_since_start <= 3
        status = "FLASH" if is_fresh else "STALE"
        progress_ratio = days_since_start / default_window
        return {
            'is_fresh': is_fresh,
            'progress_ratio': progress_ratio,
            'days_since_start': days_since_start,
            'predicted_window': default_window,
            'yield_velocity': 0.0,
            'freshness_tag': f"Day {days_since_start} - {status}",
            'has_pattern': False,
        }

    # ── Scenario A: 有 pattern → 正常进度计算 ──
    predicted_window = pattern.avg_window_days if pattern.avg_window_days > 0 else 3.0
    progress_ratio = days_since_start / predicted_window if predicted_window > 0 else 0.0

    # 收益加速度
    yield_velocity = 0.0
    lookback = 收益加速度_窗口
    if ret_idx >= lookback:
        r_now = rets.get(obs_date, 0) if isinstance(rets, dict) else rets.get(obs_date, 0)
        r_past = rets.get(ret_dates[ret_idx - lookback], 0) if isinstance(rets, dict) else rets.get(ret_dates[ret_idx - lookback], 0)
        yield_velocity = r_now - r_past

    # 判断 is_fresh
    if predicted_window > 信号新鲜度_连续窗口阈值:
        max_progress = 信号新鲜度_连续最大进度
    else:
        max_progress = 信号新鲜度_最大进度

    is_fresh = progress_ratio <= max_progress

    # 生成标签
    window_int = max(int(round(predicted_window)), 1)
    status = "FRESH" if is_fresh else "STALE"
    freshness_tag = f"Day {days_since_start}/{window_int} - {status}"

    return {
        'is_fresh': is_fresh,
        'progress_ratio': progress_ratio,
        'days_since_start': days_since_start,
        'predicted_window': predicted_window,
        'yield_velocity': yield_velocity,
        'freshness_tag': freshness_tag,
        'has_pattern': True,
    }
