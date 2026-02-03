# -*- coding: utf-8 -*-
"""
理财产品评价筛选体系

设计目标：
1. 平衡流动性和收益率
2. 考虑日开/周开/双周/月度/季度的组合搭配
3. 单产品限额2000万（避免大额赎回）
4. 筛选两组Top5：不含权益 vs 含权益

评价标准：
- 流动性评分 (30%): 根据开放周期评分
- 收益率评分 (40%): 近1月/3月年化收益率
- 稳定性评分 (20%): 最大回撤、波动率
- 规模适配评分 (10%): 是否适合2000万买入
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class ProductEvaluator:
    """理财产品评价器"""

    # 流动性评分配置（开放周期 -> 分数）
    LIQUIDITY_SCORES = {
        '日开': 100,
        '每日': 100,
        '日日': 100,
        'T+0': 100,
        'T+1': 95,
        '周开': 85,
        '每周': 85,
        '双周': 70,
        '半月': 70,
        '月开': 55,
        '每月': 55,
        '月度': 55,
        '季开': 40,
        '每季': 40,
        '季度': 40,
        '半年': 25,
        '年开': 15,
        '封闭': 0,
    }

    # 权重配置
    WEIGHTS = {
        'liquidity': 0.30,      # 流动性
        'return': 0.40,         # 收益率
        'stability': 0.20,      # 稳定性
        'size_fit': 0.10,       # 规模适配
    }

    # 单产品最大买入金额（万元）
    MAX_SINGLE_INVESTMENT = 2000

    def __init__(self, df: pd.DataFrame):
        """
        初始化评价器

        Args:
            df: 产品数据DataFrame，需包含以下列：
                - 产品代码, 产品名称, 银行
                - 产品类型, 开放周期/运作模式
                - 近1月年化(%), 近3月年化(%), 近6月年化(%)
                - 最大回撤(%), 波动率
                - 风险等级, 是否含权益
        """
        self.df = df.copy()
        self._preprocess()

    def _preprocess(self):
        """数据预处理"""
        # 标准化列名
        self.df.columns = [str(c).strip() for c in self.df.columns]

        # 确保数值列为数值类型
        numeric_cols = ['近1月年化(%)', '近3月年化(%)', '近6月年化(%)',
                       '最大回撤(%)', '波动率', '年化波动率(%)']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 识别开放周期
        self.df['开放周期'] = self.df.apply(self._detect_open_period, axis=1)

        # 识别是否含权益
        self.df['含权益'] = self.df.apply(self._detect_equity, axis=1)

    def _detect_open_period(self, row) -> str:
        """识别产品开放周期"""
        # 尝试从多个列获取开放周期信息
        period_cols = ['开放周期', '运作模式', '产品类型', '产品名称']

        for col in period_cols:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).lower()

                if any(k in text for k in ['日开', '每日', '日日', 't+0', 't+1', '现金管理']):
                    return '日开'
                elif any(k in text for k in ['周开', '每周', '7天']):
                    return '周开'
                elif any(k in text for k in ['双周', '14天', '半月']):
                    return '双周'
                elif any(k in text for k in ['月开', '每月', '月度', '30天', '1个月']):
                    return '月开'
                elif any(k in text for k in ['季开', '每季', '季度', '90天', '3个月']):
                    return '季开'
                elif any(k in text for k in ['半年', '6个月', '180天']):
                    return '半年'
                elif any(k in text for k in ['年开', '12个月', '365天']):
                    return '年开'
                elif any(k in text for k in ['封闭']):
                    return '封闭'

        return '未知'

    def _detect_equity(self, row) -> bool:
        """识别是否含权益资产"""
        # 尝试从多个列获取权益信息
        equity_cols = ['风险分类', '产品类型', '产品名称', '投资范围']

        for col in equity_cols:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).lower()

                if any(k in text for k in ['权益', '混合', '股票', '指数', '量化']):
                    return True
                elif any(k in text for k in ['固收', '固定收益', '债券', '货币']):
                    return False

        # 默认根据风险等级判断
        risk_col = '风险等级'
        if risk_col in row.index and pd.notna(row[risk_col]):
            risk = str(row[risk_col])
            if any(k in risk for k in ['高', '较高', 'R4', 'R5', '4级', '5级']):
                return True

        return False

    def calc_liquidity_score(self, row) -> float:
        """计算流动性评分"""
        period = row.get('开放周期', '未知')
        return self.LIQUIDITY_SCORES.get(period, 30)  # 默认30分

    def calc_return_score(self, row) -> float:
        """计算收益率评分（基于排名百分位）"""
        # 使用近1月年化收益率
        ret = row.get('近1月年化(%)', np.nan)
        if pd.isna(ret):
            ret = row.get('近3月年化(%)', np.nan)
        if pd.isna(ret):
            return 50  # 无数据返回中等分数

        # 计算在同类产品中的百分位排名
        all_returns = self.df['近1月年化(%)'].dropna()
        if len(all_returns) == 0:
            return 50

        percentile = (all_returns < ret).sum() / len(all_returns) * 100
        return min(100, max(0, percentile))

    def calc_stability_score(self, row) -> float:
        """计算稳定性评分"""
        score = 100

        # 最大回撤扣分
        max_dd = row.get('最大回撤(%)', 0)
        if pd.notna(max_dd) and max_dd < 0:
            score -= min(50, abs(max_dd) * 10)  # 每1%回撤扣10分，最多扣50分

        # 波动率扣分
        vol = row.get('波动率', row.get('年化波动率(%)', 0))
        if pd.notna(vol) and vol > 0:
            score -= min(30, vol * 5)  # 每1%波动扣5分，最多扣30分

        return max(0, score)

    def calc_size_fit_score(self, row) -> float:
        """计算规模适配评分"""
        # 如果有产品规模信息，检查2000万是否会触发大额赎回
        # 假设大额赎回阈值为产品规模的10%
        # 目前简化为固定分数，实际应根据产品规模计算
        return 80  # 默认给80分

    def calc_total_score(self, row) -> float:
        """计算综合评分"""
        scores = {
            'liquidity': self.calc_liquidity_score(row),
            'return': self.calc_return_score(row),
            'stability': self.calc_stability_score(row),
            'size_fit': self.calc_size_fit_score(row),
        }

        total = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        return round(total, 2)

    def evaluate_all(self) -> pd.DataFrame:
        """评估所有产品"""
        # 计算各项评分
        self.df['流动性评分'] = self.df.apply(self.calc_liquidity_score, axis=1)
        self.df['收益率评分'] = self.df.apply(self.calc_return_score, axis=1)
        self.df['稳定性评分'] = self.df.apply(self.calc_stability_score, axis=1)
        self.df['规模适配评分'] = self.df.apply(self.calc_size_fit_score, axis=1)
        self.df['综合评分'] = self.df.apply(self.calc_total_score, axis=1)

        return self.df

    def get_top_products(self, n: int = 5,
                         include_equity: bool = False,
                         min_liquidity_score: float = 40) -> pd.DataFrame:
        """
        获取Top N推荐产品

        Args:
            n: 返回产品数量
            include_equity: 是否包含权益类
            min_liquidity_score: 最低流动性评分要求

        Returns:
            筛选后的Top N产品
        """
        if '综合评分' not in self.df.columns:
            self.evaluate_all()

        # 筛选条件
        mask = self.df['流动性评分'] >= min_liquidity_score

        if not include_equity:
            mask &= ~self.df['含权益']
        else:
            mask &= self.df['含权益']

        filtered = self.df[mask].copy()

        # 按综合评分排序
        filtered = filtered.sort_values('综合评分', ascending=False)

        return filtered.head(n)

    def get_portfolio_recommendation(self) -> Dict:
        """
        获取组合推荐

        返回两组Top5：
        1. 不含权益组
        2. 含权益组

        每组按流动性搭配：日开+周开+月开的组合
        """
        if '综合评分' not in self.df.columns:
            self.evaluate_all()

        result = {
            '不含权益Top5': self.get_top_products(n=5, include_equity=False),
            '含权益Top5': self.get_top_products(n=5, include_equity=True),
        }

        # 添加流动性搭配建议
        for key, df in result.items():
            if len(df) > 0:
                periods = df['开放周期'].value_counts()
                result[f'{key}_流动性分布'] = periods.to_dict()

        return result

    def export_to_excel(self, filename: str = None) -> str:
        """导出评估结果到Excel"""
        if filename is None:
            filename = f"D:/AI-FINANCE/理财评估结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        if '综合评分' not in self.df.columns:
            self.evaluate_all()

        # 获取推荐
        recommendations = self.get_portfolio_recommendation()

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 全部产品评分
            self.df.to_excel(writer, sheet_name='全部产品评分', index=False)

            # 不含权益Top5
            if len(recommendations['不含权益Top5']) > 0:
                recommendations['不含权益Top5'].to_excel(
                    writer, sheet_name='不含权益Top5', index=False)

            # 含权益Top5
            if len(recommendations['含权益Top5']) > 0:
                recommendations['含权益Top5'].to_excel(
                    writer, sheet_name='含权益Top5', index=False)

            # 评价标准说明
            standards = pd.DataFrame({
                '评价维度': ['流动性评分', '收益率评分', '稳定性评分', '规模适配评分'],
                '权重': ['30%', '40%', '20%', '10%'],
                '计算方法': [
                    '日开100分, 周开85分, 双周70分, 月开55分, 季开40分',
                    '近1月年化收益率在所有产品中的百分位排名',
                    '100分起，最大回撤每1%扣10分(最多50分)，波动率每1%扣5分(最多30分)',
                    '根据产品规模判断2000万是否会触发大额赎回，默认80分'
                ],
                '说明': [
                    '流动性越高分数越高，便于资金调配',
                    '收益率越高排名越靠前，分数越高',
                    '回撤和波动越小越稳定，分数越高',
                    '产品规模越大，单笔2000万占比越小，越不易触发大额赎回'
                ]
            })
            standards.to_excel(writer, sheet_name='评价标准说明', index=False)

            # 流动性评分表
            liquidity_df = pd.DataFrame([
                {'开放周期': k, '评分': v}
                for k, v in sorted(self.LIQUIDITY_SCORES.items(), key=lambda x: -x[1])
            ])
            liquidity_df.to_excel(writer, sheet_name='流动性评分表', index=False)

        print(f"评估结果已导出到: {filename}")
        return filename


def analyze_banks_data(filepath: str) -> str:
    """
    分析银行理财数据并导出评估结果

    Args:
        filepath: 数据文件路径（Excel）

    Returns:
        导出的评估结果文件路径
    """
    print(f"读取数据: {filepath}")
    df = pd.read_excel(filepath)
    print(f"共 {len(df)} 个产品")

    evaluator = ProductEvaluator(df)
    evaluator.evaluate_all()

    # 导出结果
    output_file = evaluator.export_to_excel()

    # 打印推荐
    recommendations = evaluator.get_portfolio_recommendation()

    print("\n=== 不含权益Top5 ===")
    if len(recommendations['不含权益Top5']) > 0:
        cols = ['银行', '产品名称', '开放周期', '近1月年化(%)', '综合评分']
        cols = [c for c in cols if c in recommendations['不含权益Top5'].columns]
        print(recommendations['不含权益Top5'][cols].to_string())
    else:
        print("无符合条件的产品")

    print("\n=== 含权益Top5 ===")
    if len(recommendations['含权益Top5']) > 0:
        cols = ['银行', '产品名称', '开放周期', '近1月年化(%)', '综合评分']
        cols = [c for c in cols if c in recommendations['含权益Top5'].columns]
        print(recommendations['含权益Top5'][cols].to_string())
    else:
        print("无符合条件的产品")

    return output_file


if __name__ == "__main__":
    # 测试：使用最新的多银行数据
    import glob

    # 找最新的数据文件
    files = glob.glob("D:/AI-FINANCE/*银行*.xlsx") + glob.glob("D:/AI-FINANCE/*理财*.xlsx")
    if files:
        latest = max(files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
        print(f"使用数据文件: {latest}")
        analyze_banks_data(latest)
    else:
        print("未找到数据文件")
