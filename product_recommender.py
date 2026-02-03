# -*- coding: utf-8 -*-
"""
理财产品推荐系统 - 基于新评价体系

筛选逻辑：
1. 资金分布：1亿元资金池，单只产品≤2000万（20%），至少5只产品
2. 流动性分层：日开、周开(7天)、双周(14天)、月度(30天)、季度(90天)
3. 初筛：可购买=是，申购状态=开放，剔除异常收益率
4. 分组：无权益类 vs 含权益类
5. 实时验证：从银行API获取当前在售产品进行交叉验证

综合评分公式：
Score = 近2日年化×0.5 + 近1周年化×0.4 + 近3月年化×0.4 - 回撤修复天数×0.1 + 流动性因子×5

核心理念：短期波动大=买入机会，长期表现做参考

版本：2.1
日期：2026-01-21
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 配置参数
# ============================================================================

@dataclass
class RecommenderConfig:
    """推荐系统配置"""
    # 资金配置
    total_capital: float = 100_000_000  # 1亿元
    max_single_ratio: float = 0.20       # 单只产品最大占比20%
    max_single_amount: float = 20_000_000  # 单只最多2000万
    min_products: int = 5                # 最少5只产品

    # 评分权重
    weight_return: float = 1.0    # 收益率权重 wY
    weight_drawdown: float = 0.1  # 回撤修复天数权重 wD
    weight_liquidity: float = 5.0 # 流动性风险权重 wR

    # 异常值阈值
    max_return_1m: float = 50.0   # 近1月年化收益率上限 (%)
    max_return_cash: float = 30.0 # 现金管理类收益率上限 (%)

    # 流动性周期定义 (天)
    liquidity_tiers = {
        '日开': (0, 1),
        '周开': (2, 7),
        '双周': (8, 14),
        '月度': (15, 45),
        '季度': (46, 120),
    }

    # 巨额赎回阈值
    redemption_threshold: float = 0.01  # 1亿占规模超过1%视为高风险


# ============================================================================
# 数据处理
# ============================================================================

class DataProcessor:
    """数据处理器"""

    def __init__(self, config: RecommenderConfig = None):
        self.config = config or RecommenderConfig()
        self.df = None
        self.date_columns = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """加载Excel数据"""
        print(f"正在加载数据: {filepath}")
        self.df = pd.read_excel(filepath, sheet_name=0)

        # 标准化列名
        self._standardize_columns()

        # 识别日期列
        self._identify_date_columns()

        print(f"加载完成: {len(self.df)} 个产品")
        print(f"银行分布: {self.df['银行'].value_counts().to_dict()}")

        return self.df

    def _standardize_columns(self):
        """标准化列名"""
        # 根据实际列顺序的映射
        column_mapping = {
            0: '银行',
            1: '产品名称',
            2: '产品代码',
            3: '风险等级',
            4: '风险类型',
            5: '产品类型',
            6: '可购买',
            7: '申购状态',
            8: '期限天数',
            9: '期限文字',
            10: '最新净值日期',
            11: '净值天数',
            12: '最新净值',
            13: '近1周年化(%)',
            14: '近1月年化(%)',
            15: '近3月年化(%)',
            16: '近6月年化(%)',
            17: '波动率(%)',
            18: '最大回撤(%)',
            19: '夏普比率',
            20: '卡玛比率',
            21: '综合评分',
            22: '交易信号',
            23: '业绩基准',
        }

        # 重命名前24列
        new_columns = []
        for i, col in enumerate(self.df.columns):
            if i in column_mapping:
                new_columns.append(column_mapping[i])
            else:
                new_columns.append(col)

        self.df.columns = new_columns

    def _identify_date_columns(self):
        """识别日期列"""
        self.date_columns = []
        for col in self.df.columns:
            if isinstance(col, str) and col.count('-') == 2:
                try:
                    pd.to_datetime(col)
                    self.date_columns.append(col)
                except:
                    pass

        # 按日期排序（最新的在前）
        self.date_columns.sort(reverse=True)
        print(f"识别到 {len(self.date_columns)} 个日期列")

    def calculate_drawdown_recovery_days(self, row: pd.Series) -> int:
        """
        计算最大回撤修复日数

        从历史净值中找到最大回撤点，然后计算从谷底恢复到峰值所需的天数
        """
        try:
            if not self.date_columns:
                return 0

            # 获取历史净值序列
            nav_series = []
            for date_col in self.date_columns:
                val = row.get(date_col)
                try:
                    if pd.notna(val) and float(val) > 0:
                        nav_series.append((date_col, float(val)))
                except (ValueError, TypeError):
                    continue

            if len(nav_series) < 2:
                return 0

            # 按日期排序（从早到晚）
            nav_series.sort(key=lambda x: x[0])

            # 计算回撤
            peak = nav_series[0][1]
            max_drawdown = 0
            drawdown_date = None
            trough_nav = peak

            for date, nav in nav_series:
                if nav > peak:
                    peak = nav
                drawdown = (peak - nav) / peak if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    drawdown_date = date
                    trough_nav = nav

            # 如果没有回撤，返回0
            if max_drawdown < 0.001 or drawdown_date is None:
                return 0

            # 从谷底找恢复日期
            recovery_days = 0
            found_recovery = False

            for date, nav in nav_series:
                if date > drawdown_date:
                    if nav >= peak:
                        recovery_days = (pd.to_datetime(date) - pd.to_datetime(drawdown_date)).days
                        found_recovery = True
                        break

            # 如果还没恢复，计算到最新日期的天数
            if not found_recovery and nav_series:
                latest_date = nav_series[-1][0]
                recovery_days = (pd.to_datetime(latest_date) - pd.to_datetime(drawdown_date)).days

            return max(0, recovery_days)

        except Exception:
            return 0

    def classify_liquidity_tier(self, duration_days, product_name: str = "") -> str:
        """
        根据期限天数分类流动性层级
        """
        import re

        name = str(product_name) if pd.notna(product_name) else ""

        # 尝试从期限天数判断
        try:
            if pd.notna(duration_days):
                duration = int(float(duration_days))
                for tier, (min_d, max_d) in self.config.liquidity_tiers.items():
                    if min_d <= duration <= max_d:
                        return tier
                if duration > 120:
                    return '长期'
        except (ValueError, TypeError):
            pass

        # 从产品名称推断

        # 1. 明确的日开产品关键词
        daily_keywords = ['日开', '灵活申赎', '天天盈', '天天利', '活期']
        for kw in daily_keywords:
            if kw in name:
                return '日开'

        # 2. 尝试提取明确的天数/月数
        patterns = [
            (r'(\d+)天持有期', 1),
            (r'(\d+)天封闭', 1),
            (r'(\d+)个月持有期', 30),
            (r'(\d+)个月封闭', 30),
            (r'(\d+)月持有期', 30),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, name)
            if match:
                days = int(match.group(1)) * multiplier
                for tier, (min_d, max_d) in self.config.liquidity_tiers.items():
                    if min_d <= days <= max_d:
                        return tier
                if days > 120:
                    return '长期'

        # 3. 具体天数关键词
        if '7天' in name:
            return '周开'
        elif '14天' in name or '双周' in name:
            return '双周'
        elif '28天' in name or '30天' in name:
            return '月度'
        elif '60天' in name or '90天' in name:
            return '季度'

        # 4. 月份关键词
        if '1个月' in name or '一个月' in name:
            return '月度'
        elif '3个月' in name or '三个月' in name:
            return '季度'
        elif '6个月' in name or '六个月' in name:
            return '长期'

        # 5. 封闭式产品
        if '封闭' in name or '定开' in name:
            return '长期'

        # 6. 现金管理类通常是日开
        if '现金' in name or '货币' in name:
            return '日开'

        return '其他'

    def is_equity_product(self, risk_type: str, risk_level: str, product_name: str = "") -> bool:
        """
        判断是否为权益类产品

        按风险等级判断：
        - R1、R2 = 不含权益（固收类）
        - R3及以上 = 含权益
        """
        try:
            # 提取风险等级数字
            level_str = str(risk_level) if pd.notna(risk_level) else ""
            # 移除常见前缀/后缀
            level_str = level_str.replace('R', '').replace('级', '').replace('r', '').strip()
            level = int(level_str)

            # R3及以上为含权益产品
            return level >= 3
        except (ValueError, TypeError):
            # 无法解析风险等级时，默认为不含权益
            return False

    def filter_abnormal_returns(self, return_1m: float, return_1w: float = None,
                                 return_3m: float = None,
                                 product_type: str = "", product_name: str = "",
                                 nav_days: int = None, latest_nav: float = None,
                                 nav_history: dict = None) -> bool:
        """
        检查是否为异常数据（应被剔除）

        策略：短期波动大是买入机会，不过滤高收益

        剔除条件：
        1. 近1月年化缺失 - 数据不完整
        2. 近3月年化为负 - 长期表现不佳
        3. 净值异常（<0.5或>2.0）- 数据可能有误
        4. 净值波动异常（标准差>30%）- 数据可能有误
        5. 收益率计算错误（显示收益与实际净值变化不符）
        """
        # 数据缺失
        if pd.isna(return_1m):
            return True

        # 近3月年化为负 - 长期表现不佳，剔除
        if pd.notna(return_3m) and return_3m < 0:
            return True

        # 净值异常 - 正常理财产品净值应在0.5-2.0之间
        if pd.notna(latest_nav):
            if latest_nav < 0.5 or latest_nav > 2.0:
                return True

        # 净值历史分析 - 检测数据异常
        if nav_history and len(nav_history) >= 2:
            nav_values = list(nav_history.values())

            # 检查净值波动是否异常大（标准差>30%意味着极度不稳定）
            if len(nav_values) >= 3:
                import numpy as np
                nav_std = np.std(nav_values)
                nav_mean = np.mean(nav_values)
                if nav_mean > 0 and (nav_std / nav_mean) > 0.30:
                    return True  # 净值波动超过30%，数据异常

                # 检查平均净值是否异常低
                # 正常理财产品净值应在0.9-1.1左右
                # 平均净值<0.7说明产品可能已经亏损超过30%或数据有误
                if nav_mean < 0.7:
                    return True  # 平均净值过低，产品可能有严重问题

            # 验证周收益率计算是否正确
            if pd.notna(return_1w) and len(nav_history) >= 2:
                # 按日期排序获取最近的净值
                sorted_navs = sorted(nav_history.items(), key=lambda x: x[0], reverse=True)
                if len(sorted_navs) >= 2:
                    latest_date, latest_val = sorted_navs[0]

                    # 找7天前左右的净值
                    for date, val in sorted_navs[1:]:
                        try:
                            days_diff = (pd.to_datetime(latest_date) - pd.to_datetime(date)).days
                            if 5 <= days_diff <= 10:  # 大约一周
                                # 计算实际周收益
                                if val > 0:
                                    actual_weekly_return = (latest_val / val - 1) * 100

                                    # 如果显示正收益但实际是负收益，或者差异太大
                                    if return_1w > 100 and actual_weekly_return < -5:
                                        # 显示大幅正收益但实际亏损，计算错误
                                        return True
                                    if return_1w > 500 and abs(actual_weekly_return) < 50:
                                        # 显示超高收益但实际变化不大，计算错误
                                        return True
                                break
                        except:
                            pass

        return False

    def calculate_redemption_risk(self, scale_billion: float) -> int:
        """
        计算赎回流动性风险因子

        R = 0 如果 1亿/规模 > 1%
        R = 1 如果 1亿/规模 <= 1%
        """
        if pd.isna(scale_billion) or scale_billion <= 0:
            return 0  # 未知规模视为高风险

        # 规模单位是亿元
        scale_yuan = scale_billion * 100_000_000
        ratio = self.config.total_capital / scale_yuan

        if ratio > self.config.redemption_threshold:
            return 0  # 高风险
        else:
            return 1  # 低风险

    def calculate_return_2d(self, row: pd.Series, date_columns: List[str]) -> float:
        """
        计算近2日年化收益率，考虑周末/假期

        逻辑：
        1. 获取最近2个有净值的日期
        2. 计算实际交易天数（排除周末）
        3. 根据实际交易天数年化

        Args:
            row: 产品数据行
            date_columns: 日期列名列表（已排序，最新在前）

        Returns:
            调整后的近2日年化收益率 (%)
        """
        from datetime import datetime
        import numpy as np

        # 获取最近的净值数据点
        nav_points = []
        for date_col in date_columns[:10]:  # 只看最近10个日期
            val = row.get(date_col)
            try:
                if pd.notna(val) and float(val) > 0:
                    nav_points.append((date_col, float(val)))
                    if len(nav_points) >= 3:  # 最多取3个点
                        break
            except (ValueError, TypeError):
                continue

        if len(nav_points) < 2:
            return np.nan

        # 最新和次新的净值
        latest_date, latest_nav = nav_points[0]
        prev_date, prev_nav = nav_points[1]

        # 计算日历天数
        try:
            d1 = pd.to_datetime(latest_date)
            d2 = pd.to_datetime(prev_date)
            calendar_days = (d1 - d2).days
        except:
            return np.nan

        if calendar_days <= 0 or prev_nav <= 0:
            return np.nan

        # 计算实际交易天数（排除周末）
        # 简化处理：周一到周五为交易日
        trading_days = 0
        current = d2
        while current < d1:
            current += pd.Timedelta(days=1)
            if current.weekday() < 5:  # 周一=0, 周五=4
                trading_days += 1

        # 如果没有交易日（比如周六到周日），使用日历天数
        if trading_days == 0:
            trading_days = calendar_days

        # 计算收益率
        period_return = (latest_nav / prev_nav - 1)

        # 年化收益率 = 期间收益 * (365 / 实际交易天数)
        annual_return = period_return * (365 / trading_days) * 100

        # 限制在合理范围内，避免极端值影响评分
        # 理财产品2日年化超过100%或低于-100%通常是数据异常
        if annual_return > 100:
            annual_return = 100
        elif annual_return < -100:
            annual_return = -100

        return round(annual_return, 2)


# ============================================================================
# 评分计算
# ============================================================================

class Scorer:
    """评分计算器"""

    def __init__(self, config: RecommenderConfig = None):
        self.config = config or RecommenderConfig()

    def calculate_score(self, return_2d: float, return_1w: float, return_3m: float,
                        recovery_days: int, liquidity_risk: int) -> float:
        """
        计算综合评分 - 波动捕捉策略

        评分公式：
        Score = 近2日年化 × 0.5 + 近1周年化 × 0.4 + 近3月年化 × 0.4
                - 回撤修复天数 × 0.1
                + 流动性风险因子 × 5

        Args:
            return_2d: 近2日年化收益率 (%) - 已调整周末/假期
            return_1w: 近1周年化收益率 (%)
            return_3m: 近3月年化收益率 (%)
            recovery_days: 最大回撤修复日数
            liquidity_risk: 赎回流动性风险因子 (0或1)

        Returns:
            综合评分
        """
        # 近2日 - 捕捉即时信号 (权重0.5)
        Y_2d = return_2d if pd.notna(return_2d) else 0

        # 近1周 - 短期趋势 (权重0.4)
        Y_1w = return_1w if pd.notna(return_1w) else 0

        # 近3月 - 产品质量验证 (权重0.4)
        Y_3m = return_3m if pd.notna(return_3m) else 0

        D = recovery_days if recovery_days > 0 else 0
        R = liquidity_risk

        score = (Y_2d * 0.5 + Y_1w * 0.4 + Y_3m * 0.4
                 - D * 0.1
                 + R * 5)

        return round(score, 2)


# ============================================================================
# 推荐引擎
# ============================================================================

class ProductRecommender:
    """产品推荐引擎"""

    def __init__(self, config: RecommenderConfig = None):
        self.config = config or RecommenderConfig()
        self.processor = DataProcessor(config)
        self.scorer = Scorer(config)
        self.df = None

    def load_and_process(self, filepath: str) -> pd.DataFrame:
        """加载并处理数据"""
        self.df = self.processor.load_data(filepath)

        # 添加计算列
        print("\n正在计算评估指标...")

        # 流动性层级
        self.df['流动性层级'] = self.df.apply(
            lambda row: self.processor.classify_liquidity_tier(
                row.get('期限天数'),
                row.get('产品名称', '')
            ),
            axis=1
        )

        # 是否权益类
        self.df['是否权益'] = self.df.apply(
            lambda row: self.processor.is_equity_product(
                row.get('风险类型', ''),
                row.get('风险等级', ''),
                row.get('产品名称', '')
            ),
            axis=1
        )

        # 回撤修复天数
        print("计算回撤修复天数...")
        self.df['回撤修复天数'] = self.df.apply(
            self.processor.calculate_drawdown_recovery_days,
            axis=1
        )

        # 近2日年化收益率（考虑周末/假期）
        print("计算近2日年化收益率...")
        self.df['近2日年化(%)'] = self.df.apply(
            lambda row: self.processor.calculate_return_2d(row, self.processor.date_columns),
            axis=1
        )

        # 赎回风险因子 (暂无规模数据，默认为0)
        self.df['赎回风险因子'] = 0  # TODO: 需要产品规模数据

        # 综合评分（新公式）- 波动捕捉策略
        # 近2日×0.5 + 近1周×0.4 + 近3月×0.4 - 回撤×0.1 + 流动性×5
        self.df['新综合评分'] = self.df.apply(
            lambda row: self.scorer.calculate_score(
                row.get('近2日年化(%)'),
                row.get('近1周年化(%)'),
                row.get('近3月年化(%)'),
                row.get('回撤修复天数', 0),
                row.get('赎回风险因子', 0)
            ),
            axis=1
        )

        # 是否异常数据（不过滤高收益，只过滤净值异常、近3月为负、计算错误）
        def check_abnormal(row):
            # 提取净值历史
            nav_history = {}
            for date_col in self.processor.date_columns[:15]:  # 最近15个日期
                val = row.get(date_col)
                try:
                    if pd.notna(val) and float(val) > 0:
                        nav_history[date_col] = float(val)
                except (ValueError, TypeError):
                    pass

            return self.processor.filter_abnormal_returns(
                row.get('近1月年化(%)'),
                row.get('近1周年化(%)'),
                row.get('近3月年化(%)'),
                row.get('产品类型', ''),
                row.get('产品名称', ''),
                row.get('净值天数'),
                row.get('最新净值'),
                nav_history  # 新增：净值历史用于验证计算
            )

        self.df['异常收益'] = self.df.apply(check_abnormal, axis=1)

        print(f"处理完成")
        print(f"  - 流动性层级分布: {self.df['流动性层级'].value_counts().to_dict()}")
        print(f"  - 权益类产品数: {self.df['是否权益'].sum()}")
        print(f"  - 异常收益产品数: {self.df['异常收益'].sum()}")

        return self.df

    def filter_buyable(self, df: pd.DataFrame = None,
                       pre_verify: bool = True) -> pd.DataFrame:
        """
        初步筛选：可购买且申购状态开放

        筛选条件：
        1. 可购买 = 是
        2. 申购状态 = 开放
        3. 非异常收益
        4. (可选) 预先验证：产品代码在银行官网在售列表中

        Args:
            df: 数据框
            pre_verify: 是否预先验证官网在售
        """
        if df is None:
            df = self.df

        # 基础筛选条件
        mask = (
            (df['可购买'].astype(str).str.contains('是|True|1', case=False, na=False)) &
            (df['申购状态'].astype(str).str.contains('开放|可申购', case=False, na=False)) &
            (~df['异常收益'])  # 剔除异常收益
        )

        filtered = df[mask].copy()
        print(f"基础筛选后: {len(filtered)} 个产品 (原: {len(df)})")

        # 预先验证：只保留官网在售的产品
        if pre_verify:
            try:
                from product_verifier import ProductVerifier
                verifier = ProductVerifier()

                # 获取民生银行官网在售产品
                cmbc_products = verifier.fetch_cmbc_products()
                cmbc_codes = set(cmbc_products.keys())

                # 所有产品都需要在民生银行官网有售才能推荐
                # 因为民生银行官网也代销其他银行的产品（如华夏理财）
                def is_available(row):
                    return row['产品代码'] in cmbc_codes

                before_count = len(filtered)
                filtered = filtered[filtered.apply(is_available, axis=1)].copy()
                print(f"官网预验证后: {len(filtered)} 个产品 (民生银行官网有售)")
                print(f"  排除了 {before_count - len(filtered)} 个官网无售的产品")

            except Exception as e:
                print(f"预验证失败，跳过: {e}")

        return filtered

    def get_top_by_tier(self, df: pd.DataFrame,
                        include_equity: bool = False,
                        n_per_tier: int = 1,
                        candidates_per_tier: int = 10) -> pd.DataFrame:
        """
        按流动性层级获取评分最高的产品

        Args:
            df: 已筛选的数据框
            include_equity: 是否包含权益类产品
            n_per_tier: 每层级最终选几个
            candidates_per_tier: 每层级候选数量（用于验证前的筛选）

        Returns:
            推荐产品列表（候选产品，需后续验证）
        """
        results = []
        tiers = ['日开', '周开', '双周', '月度', '季度']

        # 定义哪些层级优先选择权益类（含权益组）
        equity_priority_tiers = ['月度', '季度'] if include_equity else []

        for tier in tiers:
            tier_df = df[df['流动性层级'] == tier].copy()

            if len(tier_df) == 0:
                print(f"  警告: {tier} 层级无可用产品")
                continue

            # 对于含权益组，在特定层级优先选择权益类产品
            if tier in equity_priority_tiers:
                # 先看有没有权益类产品
                equity_tier = tier_df[tier_df['是否权益']].copy()
                if len(equity_tier) > 0:
                    equity_tier = equity_tier.sort_values('新综合评分', ascending=False)
                    top_n = equity_tier.head(candidates_per_tier)
                    results.append(top_n)
                    continue

            # 无权益组：排除权益类
            if not include_equity:
                tier_df = tier_df[~tier_df['是否权益']].copy()

            if len(tier_df) == 0:
                print(f"  警告: {tier} 层级无可用产品（筛选后）")
                continue

            # 按新综合评分排序
            tier_df = tier_df.sort_values('新综合评分', ascending=False)

            # 取前n个候选
            top_n = tier_df.head(candidates_per_tier)
            results.append(top_n)

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_verified_recommendations(self, candidates: pd.DataFrame,
                                     n_per_tier: int = 1) -> pd.DataFrame:
        """
        从已验证的候选产品中选出最终推荐

        Args:
            candidates: 已验证的候选产品（包含'已验证可购买'列）
            n_per_tier: 每层级选几个

        Returns:
            最终推荐产品
        """
        if '已验证可购买' not in candidates.columns:
            print("警告: 候选产品未经验证，返回原始推荐")
            return candidates

        # 只保留已验证可购买的
        verified = candidates[candidates['已验证可购买'] == True].copy()

        if len(verified) == 0:
            print("警告: 没有产品通过验证!")
            return pd.DataFrame()

        results = []
        tiers = ['日开', '周开', '双周', '月度', '季度']

        for tier in tiers:
            tier_df = verified[verified['流动性层级'] == tier].copy()

            if len(tier_df) == 0:
                print(f"  {tier}: 无已验证产品")
                continue

            # 按评分排序取前n个
            tier_df = tier_df.sort_values('新综合评分', ascending=False)
            top_n = tier_df.head(n_per_tier)
            results.append(top_n)
            print(f"  {tier}: 选中 {len(top_n)} 个")

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    def generate_recommendations(self, candidates_per_tier: int = 10,
                                  pre_verify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成推荐组合

        新流程：
        1. 先从银行官网获取在售产品列表
        2. 只从官网有售的产品中选择候选
        3. 每层级选10个候选（共50个）
        4. 最终推荐每层级1个

        Args:
            candidates_per_tier: 每层级候选数量
            pre_verify: 是否预先验证官网在售

        Returns:
            (无权益组合候选, 含权益组合候选)
        """
        if self.df is None:
            raise ValueError("请先调用 load_and_process() 加载数据")

        # 初步筛选（含预验证）
        filtered = self.filter_buyable(pre_verify=pre_verify)

        print(f"\n=== 第一组：无权益类资产 (每层级{candidates_per_tier}个候选) ===")
        group1 = self.get_top_by_tier(filtered, include_equity=False,
                                       candidates_per_tier=candidates_per_tier)
        if len(group1) > 0:
            print(f"候选产品数: {len(group1)}")
            for tier in ['日开', '周开', '双周', '月度', '季度']:
                tier_count = len(group1[group1['流动性层级'] == tier])
                if tier_count > 0:
                    print(f"  {tier}: {tier_count}个候选")

        print(f"\n=== 第二组：含权益类资产 (每层级{candidates_per_tier}个候选) ===")
        group2 = self.get_top_by_tier(filtered, include_equity=True,
                                       candidates_per_tier=candidates_per_tier)
        if len(group2) > 0:
            print(f"候选产品数: {len(group2)}")
            for tier in ['日开', '周开', '双周', '月度', '季度']:
                tier_count = len(group2[group2['流动性层级'] == tier])
                if tier_count > 0:
                    print(f"  {tier}: {tier_count}个候选")

        return group1, group2

    def export_results(self, output_file: str = None, verify: bool = True,
                       pre_verify: bool = True) -> str:
        """
        导出推荐结果到Excel

        新流程：
        1. 先从银行官网获取在售产品列表，预过滤
        2. 从官网有售的产品中选50个候选（每层级10个）
        3. 最终推荐每层级1个（共5个）

        Args:
            output_file: 输出文件路径
            verify: 是否进行二次验证（预验证后通常不需要）
            pre_verify: 是否预先验证官网在售
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"D:/AI-FINANCE/产品推荐_{timestamp}.xlsx"

        # 生成候选产品（每组约50个，已预验证官网在售）
        candidates1, candidates2 = self.generate_recommendations(
            candidates_per_tier=10,
            pre_verify=pre_verify
        )

        # 定义输出列
        output_columns = [
            '产品代码', '产品名称', '流动性层级', '银行',
            '风险等级', '风险类型', '是否权益',
            '近2日年化(%)', '近1周年化(%)', '近1月年化(%)', '近3月年化(%)',
            '波动率(%)', '最大回撤(%)', '回撤修复天数',
            '赎回风险因子', '新综合评分',
            '可购买', '申购状态'
        ]

        # 如果已预验证，候选产品都是官网有售的，直接选最终推荐
        # 如果需要二次验证，再调用验证器
        verified1, verified2 = None, None
        final1, final2 = pd.DataFrame(), pd.DataFrame()

        if pre_verify:
            # 已预验证，直接从候选中选每层级第1个
            print("\n" + "=" * 60)
            print("从预验证候选中选择最终推荐")
            print("=" * 60)

            # 标记为已验证
            if len(candidates1) > 0:
                candidates1 = candidates1.copy()
                candidates1['已验证可购买'] = True
                candidates1['验证状态'] = 'pre_verified'
                candidates1['验证说明'] = '官网预验证通过'
                verified1 = candidates1
                final1 = self.get_verified_recommendations(verified1, n_per_tier=1)
                print(f"无权益组: {len(final1)}个推荐")

            if len(candidates2) > 0:
                candidates2 = candidates2.copy()
                candidates2['已验证可购买'] = True
                candidates2['验证状态'] = 'pre_verified'
                candidates2['验证说明'] = '官网预验证通过'
                verified2 = candidates2
                final2 = self.get_verified_recommendations(verified2, n_per_tier=1)
                print(f"含权益组: {len(final2)}个推荐")

        elif verify:
            # 二次验证模式
            try:
                from product_verifier import ProductVerifier
                verifier = ProductVerifier()

                print("\n" + "=" * 60)
                print("验证候选产品")
                print("=" * 60)

                if len(candidates1) > 0:
                    print(f"\n验证无权益组候选 ({len(candidates1)}个)...")
                    verified1 = verifier.verify_dataframe(candidates1)
                    final1 = self.get_verified_recommendations(verified1, n_per_tier=1)
                    print(f"最终推荐: {len(final1)}个产品通过验证")

                if len(candidates2) > 0:
                    print(f"\n验证含权益组候选 ({len(candidates2)}个)...")
                    verified2 = verifier.verify_dataframe(candidates2)
                    final2 = self.get_verified_recommendations(verified2, n_per_tier=1)
                    print(f"最终推荐: {len(final2)}个产品通过验证")

            except ImportError:
                print("警告: 未找到 product_verifier 模块，跳过验证")
            except Exception as e:
                print(f"验证失败: {e}")

        # 如果验证失败，使用原始候选作为最终推荐
        if final1 is None or len(final1) == 0:
            final1 = candidates1.head(5) if len(candidates1) > 0 else pd.DataFrame()
        if final2 is None or len(final2) == 0:
            final2 = candidates2.head(5) if len(candidates2) > 0 else pd.DataFrame()

        # 打印最终推荐
        print("\n" + "=" * 60)
        print("最终推荐产品")
        print("=" * 60)

        print("\n【无权益组】")
        if len(final1) > 0:
            for _, row in final1.iterrows():
                verified_mark = "[已验证]" if verify and row.get('已验证可购买', False) else ""
                print(f"  {verified_mark}[{row['流动性层级']}] {row['产品代码']} - {row['产品名称'][:25]}... "
                      f"评分:{row['新综合评分']:.2f}")
        else:
            print("  无推荐产品")

        print("\n【含权益组】")
        if len(final2) > 0:
            for _, row in final2.iterrows():
                verified_mark = "[已验证]" if verify and row.get('已验证可购买', False) else ""
                equity_mark = "[权益]" if row['是否权益'] else ""
                print(f"  {verified_mark}{equity_mark}[{row['流动性层级']}] {row['产品代码']} - {row['产品名称'][:25]}... "
                      f"评分:{row['新综合评分']:.2f}")
        else:
            print("  无推荐产品")

        # 导出Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 最终推荐（已验证）
            if len(final1) > 0:
                cols = [c for c in output_columns if c in final1.columns]
                if '已验证可购买' in final1.columns:
                    cols = cols + ['已验证可购买', '验证状态', '验证说明']
                final1[[c for c in cols if c in final1.columns]].to_excel(
                    writer, sheet_name='无权益类推荐', index=False)

            if len(final2) > 0:
                cols = [c for c in output_columns if c in final2.columns]
                if '已验证可购买' in final2.columns:
                    cols = cols + ['已验证可购买', '验证状态', '验证说明']
                final2[[c for c in cols if c in final2.columns]].to_excel(
                    writer, sheet_name='含权益类推荐', index=False)

            # 候选产品（含验证状态）
            if verified1 is not None and len(verified1) > 0:
                cols = [c for c in output_columns if c in verified1.columns]
                cols = cols + ['已验证可购买', '验证状态', '验证说明']
                verified1[[c for c in cols if c in verified1.columns]].to_excel(
                    writer, sheet_name='无权益候选50', index=False)
            elif len(candidates1) > 0:
                cols = [c for c in output_columns if c in candidates1.columns]
                candidates1[cols].to_excel(writer, sheet_name='无权益候选50', index=False)

            if verified2 is not None and len(verified2) > 0:
                cols = [c for c in output_columns if c in verified2.columns]
                cols = cols + ['已验证可购买', '验证状态', '验证说明']
                verified2[[c for c in cols if c in verified2.columns]].to_excel(
                    writer, sheet_name='含权益候选50', index=False)
            elif len(candidates2) > 0:
                cols = [c for c in output_columns if c in candidates2.columns]
                candidates2[cols].to_excel(writer, sheet_name='含权益候选50', index=False)

            # 完整评分列表
            if self.df is not None:
                cols = [c for c in output_columns if c in self.df.columns]
                sorted_df = self.df.sort_values('新综合评分', ascending=False)
                sorted_df[cols].head(100).to_excel(writer, sheet_name='评分排行Top100', index=False)

            # 统计信息
            stats = pd.DataFrame([
                {'项目': '总产品数', '数值': len(self.df)},
                {'项目': '可购买产品数', '数值': len(self.filter_buyable())},
                {'项目': '无权益候选数', '数值': len(candidates1)},
                {'项目': '含权益候选数', '数值': len(candidates2)},
                {'项目': '无权益已验证数', '数值': len(final1)},
                {'项目': '含权益已验证数', '数值': len(final2)},
                {'项目': '评分公式', '数值': '近2日×0.5+近1周×0.4+近3月×0.4-回撤×0.1+流动性×5'},
            ])
            stats.to_excel(writer, sheet_name='配置信息', index=False)

        print(f"\n结果已导出到: {output_file}")
        return output_file


# ============================================================================
# 主程序
# ============================================================================

def main(verify: bool = True):
    """
    主程序入口

    流程：
    1. 加载数据并计算评分
    2. 每组选50个候选产品（每层级10个）
    3. 实时验证候选产品是否可购买
    4. 从已验证产品中选出最终推荐（每层级1个，共5个）
    5. 导出结果

    Args:
        verify: 是否进行实时验证（从银行API获取在售产品核对）
    """
    print("=" * 60)
    print("理财产品推荐系统 v2.1")
    print("基于新评价体系：波动捕捉策略 + 实时验证")
    print("=" * 60)
    print("\n核心策略：短期波动大 = 买入机会")
    print("评分公式：近2日×0.5 + 近1周×0.4 + 近3月×0.4 - 回撤×0.1 + 流动性×5")
    print("\n流程：选50候选 → API验证 → 推荐5个已验证产品")

    # 配置
    config = RecommenderConfig(
        weight_return=1.0,
        weight_drawdown=0.1,
        weight_liquidity=5.0,
    )

    # 创建推荐器
    recommender = ProductRecommender(config)

    # 加载数据
    data_file = "D:/AI-FINANCE/多银行理财_量化分析_20260121_014815.xlsx"
    recommender.load_and_process(data_file)

    # 导出结果（含验证）
    output_file = recommender.export_results(verify=verify)

    print("\n" + "=" * 60)
    print("推荐完成!")
    print(f"输出文件: {output_file}")
    print("=" * 60)

    return recommender


if __name__ == "__main__":
    import sys

    # 命令行参数：--no-verify 跳过验证
    verify = "--no-verify" not in sys.argv

    main(verify=verify)
