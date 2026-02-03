# -*- coding: utf-8 -*-
"""
合并中信、民生、华夏银行的理财数据
按照民生理财的格式和规则输出标准Excel

日期：2026-01-17
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置编码
import sys
sys.stdout.reconfigure(encoding='utf-8')


class BankDataMerger:
    """银行数据合并器"""

    # 标准列顺序（按民生理财格式）
    STANDARD_COLUMNS = [
        '银行',
        '产品名称',
        '产品代码',
        '风险等级',
        '风险类型',
        '产品类型',
        '可购买',
        '申购状态',
        '持有期限',
        '期限天数',
        '最新净值日期',
        '净值天数',
        '最新净值',
        '近1周年化(%)',
        '近1月年化(%)',
        '近3月年化(%)',
        '近6月年化(%)',
        '波动率(%)',
        '最大回撤(%)',
        '夏普比率',
        '卡玛比率',
        '综合评分',
        '交易信号',
        '业绩基准',
    ]

    def __init__(self, data_dir="D:\\AI-FINANCE"):
        self.data_dir = data_dir
        self.all_data = []

    def load_minsheng(self):
        """加载民生理财数据"""
        print("\n[民生理财] 加载数据...")

        # 查找最新的民生理财量化分析文件
        files = [f for f in os.listdir(self.data_dir)
                 if f.startswith('民生理财_量化分析_') and f.endswith('.xlsx')]
        if not files:
            print("   未找到民生理财数据文件")
            return None

        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)))
        file_path = os.path.join(self.data_dir, latest_file)

        df = pd.read_excel(file_path)

        # 添加银行标识
        if '银行' not in df.columns:
            df['银行'] = '民生理财'

        print(f"   文件: {latest_file}")
        print(f"   产品数: {len(df)}")

        return df

    def load_huaxia(self):
        """加载华夏理财数据"""
        print("\n[华夏理财] 加载数据...")

        # 从多银行文件中提取华夏数据
        files = [f for f in os.listdir(self.data_dir)
                 if f.startswith('多银行理财_量化分析_') and f.endswith('.xlsx')]

        if files:
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)))
            file_path = os.path.join(self.data_dir, latest_file)

            df = pd.read_excel(file_path, sheet_name='全部产品')

            # 筛选华夏理财
            if '银行' in df.columns:
                df = df[df['银行'] == '华夏理财'].copy()

            if len(df) > 0:
                print(f"   文件: {latest_file}")
                print(f"   产品数: {len(df)}")
                return df

        # 备选：单独的华夏文件
        huaxia_files = [f for f in os.listdir(self.data_dir)
                        if f.startswith('华夏理财_') and f.endswith('.xlsx')]
        if huaxia_files:
            latest_file = max(huaxia_files, key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)))
            file_path = os.path.join(self.data_dir, latest_file)
            df = pd.read_excel(file_path)
            if '银行' not in df.columns:
                df['银行'] = '华夏理财'
            print(f"   文件: {latest_file}")
            print(f"   产品数: {len(df)}")
            return df

        print("   未找到华夏理财数据文件")
        return None

    def load_citic(self):
        """加载中信理财数据并计算量化指标"""
        print("\n[信银理财] 加载数据...")

        file_path = os.path.join(self.data_dir, 'citic_nav_history_full.xlsx')
        if not os.path.exists(file_path):
            print("   未找到中信理财数据文件")
            return None

        df = pd.read_excel(file_path)
        print(f"   文件: citic_nav_history_full.xlsx")
        print(f"   原始产品数: {len(df)}")

        # 识别日期列
        date_cols = [col for col in df.columns if self._is_date_column(col)]
        date_cols = sorted(date_cols, reverse=True)  # 最新日期在前

        print(f"   净值日期数: {len(date_cols)}")

        # 计算量化指标
        print("   计算量化指标...")
        df = self._calculate_citic_metrics(df, date_cols)

        # 确保银行标识
        df['银行'] = '信银理财'

        print(f"   有效产品数: {len(df)}")
        return df

    def _is_date_column(self, col):
        """判断是否是日期列"""
        col_str = str(col)
        # 匹配 YYYY-MM-DD 格式
        if len(col_str) == 10 and col_str[4] == '-' and col_str[7] == '-':
            try:
                datetime.strptime(col_str, '%Y-%m-%d')
                return True
            except:
                pass
        return False

    def _calculate_citic_metrics(self, df, date_cols):
        """为中信数据计算量化指标"""

        results = []

        for idx, row in df.iterrows():
            # 提取净值序列
            nav_series = []
            for col in date_cols:
                val = row.get(col)
                if pd.notna(val) and val != '' and val != 0:
                    try:
                        nav_series.append({
                            'date': str(col),
                            'nav': float(val)
                        })
                    except:
                        pass

            # 计算指标
            metrics = self._compute_metrics(nav_series)

            # 构建结果行
            result = {
                '银行': '信银理财',
                '产品名称': row.get('产品名称', ''),
                '产品代码': row.get('产品代码', ''),
                '风险等级': row.get('风险等级', ''),
                '风险类型': '固收类',  # 默认
                '产品类型': row.get('产品类型', '其他'),
                '可购买': '是',  # 默认
                '申购状态': '开放',
                '持有期限': '',
                '期限天数': None,
                '最新净值日期': metrics.get('latest_date', ''),
                '净值天数': metrics.get('nav_count', 0),
                '最新净值': metrics.get('latest_nav'),
                '近1周年化(%)': metrics.get('return_1w'),
                '近1月年化(%)': metrics.get('return_1m'),
                '近3月年化(%)': metrics.get('return_3m'),
                '近6月年化(%)': metrics.get('return_6m'),
                '波动率(%)': metrics.get('volatility'),
                '最大回撤(%)': metrics.get('max_drawdown'),
                '夏普比率': metrics.get('sharpe'),
                '卡玛比率': metrics.get('calmar'),
                '综合评分': metrics.get('score'),
                '交易信号': metrics.get('signal', '观望'),
                '业绩基准': row.get('其他信息', ''),
            }

            # 添加历史净值列（最多15天）
            for i, nav_item in enumerate(nav_series[:15]):
                result[nav_item['date']] = nav_item['nav']

            results.append(result)

        return pd.DataFrame(results)

    def _compute_metrics(self, nav_series):
        """计算量化指标"""
        if not nav_series or len(nav_series) < 2:
            return {'nav_count': len(nav_series) if nav_series else 0}

        # 按日期排序（最新在前）
        nav_series = sorted(nav_series, key=lambda x: x['date'], reverse=True)

        latest_nav = nav_series[0]['nav']
        latest_date = nav_series[0]['date']
        nav_count = len(nav_series)

        # 计算日收益率序列
        returns = []
        for i in range(len(nav_series) - 1):
            if nav_series[i+1]['nav'] > 0:
                daily_return = (nav_series[i]['nav'] / nav_series[i+1]['nav']) - 1
                returns.append(daily_return)

        if not returns:
            return {
                'latest_nav': latest_nav,
                'latest_date': latest_date,
                'nav_count': nav_count
            }

        returns = np.array(returns)

        # 年化收益率计算
        def annualized_return(days):
            if len(nav_series) <= days:
                return None
            start_nav = nav_series[min(days, len(nav_series)-1)]['nav']
            if start_nav <= 0:
                return None
            total_return = (latest_nav / start_nav) - 1
            return round(total_return * 365 / days * 100, 2)

        return_1w = annualized_return(7)
        return_1m = annualized_return(30)
        return_3m = annualized_return(90)
        return_6m = annualized_return(180)

        # 波动率（年化）
        if len(returns) >= 5:
            volatility = round(np.std(returns) * np.sqrt(252) * 100, 2)
        else:
            volatility = None

        # 最大回撤
        navs = [item['nav'] for item in nav_series]
        max_drawdown = self._calculate_max_drawdown(navs)

        # 夏普比率（假设无风险利率2%）
        if volatility and volatility > 0 and return_1m is not None:
            risk_free_rate = 2.0
            sharpe = round((return_1m - risk_free_rate) / volatility, 2)
        else:
            sharpe = None

        # 卡玛比率
        if max_drawdown and max_drawdown < 0 and return_1m is not None:
            calmar = round(return_1m / abs(max_drawdown), 2)
        else:
            calmar = None

        # 综合评分
        score = self._calculate_score(return_1m, volatility, max_drawdown, sharpe)

        # 交易信号
        signal = self._generate_signal(score, return_1m, max_drawdown)

        return {
            'latest_nav': round(latest_nav, 4),
            'latest_date': latest_date,
            'nav_count': nav_count,
            'return_1w': return_1w,
            'return_1m': return_1m,
            'return_3m': return_3m,
            'return_6m': return_6m,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'calmar': calmar,
            'score': score,
            'signal': signal
        }

    def _calculate_max_drawdown(self, navs):
        """计算最大回撤"""
        if not navs or len(navs) < 2:
            return None

        peak = navs[0]
        max_dd = 0

        for nav in navs:
            if nav > peak:
                peak = nav
            dd = (nav - peak) / peak
            if dd < max_dd:
                max_dd = dd

        return round(max_dd * 100, 2) if max_dd < 0 else 0

    def _calculate_score(self, return_1m, volatility, max_drawdown, sharpe):
        """计算综合评分"""
        score = 50  # 基础分

        # 收益加分（最多30分）
        if return_1m is not None:
            if return_1m >= 5:
                score += 30
            elif return_1m >= 3:
                score += 20
            elif return_1m >= 2:
                score += 10
            elif return_1m >= 0:
                score += 5
            else:
                score -= 10

        # 波动率（低波动加分，最多15分）
        if volatility is not None:
            if volatility < 1:
                score += 15
            elif volatility < 3:
                score += 10
            elif volatility < 5:
                score += 5
            elif volatility > 10:
                score -= 10

        # 最大回撤（小回撤加分，最多15分）
        if max_drawdown is not None:
            if max_drawdown > -0.5:
                score += 15
            elif max_drawdown > -1:
                score += 10
            elif max_drawdown > -2:
                score += 5
            elif max_drawdown < -5:
                score -= 10

        # 夏普比率加分（最多10分）
        if sharpe is not None:
            if sharpe > 2:
                score += 10
            elif sharpe > 1:
                score += 5
            elif sharpe < 0:
                score -= 5

        return round(min(max(score, 0), 100), 2)

    def _generate_signal(self, score, return_1m, max_drawdown):
        """生成交易信号"""
        if score is None:
            return '观望'

        if score >= 80 and return_1m and return_1m >= 3:
            return '强烈买入'
        elif score >= 70:
            return '买入'
        elif score >= 50:
            return '持有'
        elif score >= 30:
            return '观望'
        else:
            return '回避'

    def merge_all(self):
        """合并所有银行数据"""
        print("=" * 60)
        print("合并三家银行理财数据（民生格式）")
        print("=" * 60)

        # 加载各银行数据
        df_minsheng = self.load_minsheng()
        df_huaxia = self.load_huaxia()
        df_citic = self.load_citic()

        # 收集数据
        all_dfs = []
        if df_minsheng is not None and len(df_minsheng) > 0:
            all_dfs.append(df_minsheng)
        if df_huaxia is not None and len(df_huaxia) > 0:
            all_dfs.append(df_huaxia)
        if df_citic is not None and len(df_citic) > 0:
            all_dfs.append(df_citic)

        if not all_dfs:
            print("\n没有找到任何数据!")
            return None

        # 合并
        print("\n[合并] 合并数据...")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"   合并后总数: {len(merged_df)}")

        # 统计
        print("\n[统计] 各银行产品数:")
        for bank, count in merged_df['银行'].value_counts().items():
            print(f"   {bank}: {count}")

        return merged_df

    def save_to_excel(self, df, filename=None):
        """保存为标准格式Excel"""
        if df is None or len(df) == 0:
            print("没有数据可保存")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'三银行理财合并_{timestamp}.xlsx'

        output_path = os.path.join(self.data_dir, filename)

        print(f"\n[保存] 生成Excel文件...")

        # 整理列顺序
        # 首先确定标准列
        standard_cols = [c for c in self.STANDARD_COLUMNS if c in df.columns]

        # 然后是日期列（净值历史）
        date_cols = sorted([c for c in df.columns if self._is_date_column(c)], reverse=True)

        # 其他列
        other_cols = [c for c in df.columns if c not in standard_cols and c not in date_cols]

        # 最终列顺序
        final_cols = standard_cols + date_cols[:15]  # 最多15个日期列
        df_output = df[final_cols].copy()

        # 写入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 全部产品
            df_output.to_excel(writer, sheet_name='全部产品', index=False)

            # 按银行分sheet
            for bank in df_output['银行'].unique():
                bank_df = df_output[df_output['银行'] == bank]
                sheet_name = bank[:31]
                bank_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 买入信号
            if '交易信号' in df_output.columns:
                buy_df = df_output[df_output['交易信号'].isin(['强烈买入', '买入'])]
                if '综合评分' in buy_df.columns:
                    buy_df = buy_df.sort_values('综合评分', ascending=False)
                buy_df.to_excel(writer, sheet_name='买入信号', index=False)

            # 高评分产品
            if '综合评分' in df_output.columns:
                high_score_df = df_output[df_output['综合评分'] >= 70]
                high_score_df = high_score_df.sort_values('综合评分', ascending=False)
                high_score_df.to_excel(writer, sheet_name='高评分产品', index=False)

            # 设置数字格式
            self._format_excel(writer)

        print(f"   已保存: {filename}")
        print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

        return output_path

    def _format_excel(self, writer):
        """设置Excel格式"""
        try:
            import re
            for sheet_name in writer.sheets:
                ws = writer.sheets[sheet_name]
                for col_idx, col_cell in enumerate(ws[1], 1):
                    col_name = str(col_cell.value) if col_cell.value else ''
                    # 净值列和日期列设置4位小数
                    is_nav_col = ('净值' in col_name and '日期' not in col_name)
                    is_date_col = bool(re.match(r'^\d{4}-\d{2}-\d{2}$', col_name))
                    if is_nav_col or is_date_col:
                        for row_idx in range(2, ws.max_row + 1):
                            cell = ws.cell(row=row_idx, column=col_idx)
                            if cell.value is not None and isinstance(cell.value, (int, float)):
                                cell.number_format = '0.0000'
        except Exception as e:
            print(f"   格式设置跳过: {e}")


def main():
    """主函数"""
    merger = BankDataMerger()

    # 合并数据
    merged_df = merger.merge_all()

    if merged_df is not None:
        # 保存
        output_file = merger.save_to_excel(merged_df)

        print("\n" + "=" * 60)
        print("完成!")
        print("=" * 60)

        if output_file:
            print(f"\n输出文件: {output_file}")

            # 显示汇总
            print("\n产品汇总:")
            for bank, count in merged_df['银行'].value_counts().items():
                print(f"  {bank}: {count} 个产品")
            print(f"  总计: {len(merged_df)} 个产品")

            # 信号分布
            if '交易信号' in merged_df.columns:
                print("\n交易信号分布:")
                for signal, count in merged_df['交易信号'].value_counts().items():
                    print(f"  {signal}: {count}")


if __name__ == "__main__":
    main()
