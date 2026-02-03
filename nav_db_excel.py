# -*- coding: utf-8 -*-
"""
净值数据库Excel管理模块

功能：
1. 统一管理民生、华夏、中信三家银行的净值数据
2. 每个银行一个Sheet
3. 增量更新：每次只追加新日期的净值
4. 日期按日历日排序（从旧到新）

Excel结构：
- 行头：产品代码 | 产品名称
- 列头：日期（2024-01-01, 2024-01-02, ...）
- 数据：单位净值
"""

import os
import time
import shutil
import threading
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# 保留最近N个自动备份
MAX_AUTO_BACKUPS = 10

# 数据库文件路径
NAV_DB_FILE = os.path.join(os.path.dirname(__file__), "净值数据库.xlsx")
PARQUET_DB_FILE = os.path.join(os.path.dirname(__file__), "净值数据库.parquet")

# 费率列名常量（保持与赎回费数据库字段对应）
FEE_COLUMNS = ['认购费', '申购费', '赎回费', '销售服务费', '托管费', '管理费']


class NAVDatabaseExcel:
    """净值数据库Excel管理器"""

    # 银行名称映射
    BANK_SHEETS = {
        "民生": "民生银行",
        "华夏": "华夏银行",
        "中信": "中信银行",
        "民生理财": "民生银行",
        "华夏理财": "华夏银行",
        "信银理财": "中信银行",
        "浦银": "浦银理财",
        "浦银理财": "浦银理财",
        "浦发": "浦银理财",
        "浦发理财": "浦银理财",
        "中邮": "中邮理财",
        "中邮理财": "中邮理财",
        "邮储": "中邮理财",
        "宁银": "宁银理财",
        "宁银理财": "宁银理财",
        "宁波": "宁银理财",
        "宁波银行": "宁银理财",
    }

    def __init__(self, db_file: str = None):
        """
        初始化数据库管理器

        Args:
            db_file: Excel文件路径，默认使用 净值数据库.xlsx
        """
        self.db_file = db_file or NAV_DB_FILE
        self.data: Dict[str, pd.DataFrame] = {}  # sheet_name -> DataFrame
        self._load_ok = False          # 加载是否成功（防止加载失败后误保存）
        self._orig_stats = {}          # 加载时各sheet的行数/列数快照（完整性校验基准）
        self._load_database()

    def _get_sheet_name(self, bank_name: str) -> str:
        """获取标准化的Sheet名称"""
        return self.BANK_SHEETS.get(bank_name, bank_name)

    def _load_database(self):
        """加载现有数据库（优先Parquet，回退Excel）

        加载成功时设置 _load_ok=True 并记录各 sheet 的行数/列数快照。
        加载失败时 _load_ok=False，后续 save() 将被阻止以防数据丢失。
        """
        parquet_file = self.db_file.replace('.xlsx', '.parquet')

        if os.path.exists(parquet_file) and os.path.getsize(parquet_file) > 0:
            self._load_from_parquet(parquet_file)
        elif os.path.exists(self.db_file) and os.path.getsize(self.db_file) > 0:
            self._load_from_excel()
        else:
            # 文件不存在是合法的首次运行场景
            logger.info(f"[NAV数据库] 数据库文件不存在，将创建新文件")
            self.data = {}
            self._load_ok = True   # 允许创建
            self._orig_stats = {}

    def _load_from_parquet(self, parquet_file):
        """从Parquet文件加载数据库（快速模式）"""
        try:
            t0 = time.time()
            df = pd.read_parquet(parquet_file)

            # 检测费率列是否存在
            fee_cols_present = [c for c in FEE_COLUMNS if c in df.columns]

            self.data = {}
            for bank, group in df.groupby('银行'):
                # Pivot: 长表 -> 宽表（费率列纳入index避免被当作value）
                pivot_index = ['产品代码', '产品名称'] + fee_cols_present
                wide = group.pivot_table(
                    index=pivot_index,
                    columns='日期',
                    values='净值',
                    aggfunc='first'
                )
                # 日期列转为字符串格式
                wide.columns = [c.strftime('%Y-%m-%d') if hasattr(c, 'strftime') else str(c)
                                for c in wide.columns]
                # 排序日期列
                date_cols = sorted(wide.columns)
                wide = wide[date_cols]
                # 值转为字符串（与Excel加载一致）
                wide = wide.astype(str).replace('nan', pd.NA)

                # 费率列从MultiIndex中提取为普通列，保留产品代码+名称作为MultiIndex
                if fee_cols_present:
                    wide = wide.reset_index()
                    # 费率列中的nan还原为空字符串
                    for fc in fee_cols_present:
                        if fc in wide.columns:
                            wide[fc] = wide[fc].astype(str).replace('nan', '').replace('<NA>', '')
                    wide = wide.set_index(['产品代码', '产品名称'])

                self.data[bank] = wide

            # ★ 记录加载快照 — 用于保存前完整性校验
            self._orig_stats = {
                name: {'rows': len(df_), 'cols': len(df_.columns)}
                for name, df_ in self.data.items()
            }
            self._load_ok = True
            elapsed = time.time() - t0
            logger.info(f"[NAV数据库] Parquet加载完成 ({elapsed:.2f}s)，共 {len(self.data)} 个银行  "
                        f"{', '.join(f'{k}={v['rows']}行' for k,v in self._orig_stats.items())}")
        except Exception as e:
            logger.error(f"[NAV数据库] !! Parquet加载失败: {e}  —— 将阻止保存以防数据丢失")
            self.data = {}
            self._load_ok = False
            self._orig_stats = {}

    def _load_from_excel(self):
        """从Excel文件加载数据库（Legacy模式）"""
        try:
            xlsx = pd.ExcelFile(self.db_file)
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name, dtype=str)
                rename_map = {}
                if 'level_0' in df.columns and '产品代码' not in df.columns:
                    rename_map['level_0'] = '产品代码'
                if 'level_1' in df.columns and '产品名称' not in df.columns:
                    rename_map['level_1'] = '产品名称'
                if rename_map:
                    df = df.rename(columns=rename_map)
                if '产品代码' in df.columns and '产品名称' in df.columns:
                    df = df.set_index(['产品代码', '产品名称'])
                    df = df.sort_index()
                self.data[sheet_name] = df

            # ★ 记录加载快照 — 用于保存前完整性校验
            self._orig_stats = {
                name: {'rows': len(df), 'cols': len(df.columns)}
                for name, df in self.data.items()
            }
            self._load_ok = True
            logger.info(f"[NAV数据库] Excel加载完成，共 {len(self.data)} 个银行  "
                        f"{', '.join(f'{k}={v['rows']}行' for k,v in self._orig_stats.items())}")
        except Exception as e:
            logger.error(f"[NAV数据库] !! Excel加载失败: {e}  —— 将阻止保存以防数据丢失")
            self.data = {}
            self._load_ok = False
            self._orig_stats = {}

    # ------------------------------------------------------------------
    # 数据保护: 备份 + 完整性校验 + 保存
    # ------------------------------------------------------------------

    def _create_backup(self):
        """保存前自动备份原文件（带时间戳），保留最近 MAX_AUTO_BACKUPS 个"""
        # 确定要备份的文件（优先parquet）
        parquet_file = self.db_file.replace('.xlsx', '.parquet')
        if os.path.exists(parquet_file) and os.path.getsize(parquet_file) > 0:
            file_to_backup = parquet_file
            ext = '.parquet'
        elif os.path.exists(self.db_file) and os.path.getsize(self.db_file) > 0:
            file_to_backup = self.db_file
            ext = '.xlsx'
        else:
            return  # 无原文件，首次创建，无需备份

        backup_dir = os.path.join(os.path.dirname(self.db_file), "nav_db_backups")
        os.makedirs(backup_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.basename(self.db_file).replace('.xlsx', '')
        backup_path = os.path.join(backup_dir, f"{base}_{ts}{ext}")

        try:
            shutil.copy2(file_to_backup, backup_path)
            logger.info(f"[NAV数据库] 自动备份 -> {backup_path}")
        except Exception as e:
            logger.warning(f"[NAV数据库] 备份失败: {e}（继续保存）")

        # 清理旧备份，只保留最新 MAX_AUTO_BACKUPS 个（含.xlsx和.parquet）
        try:
            backups = sorted(
                [f for f in os.listdir(backup_dir)
                 if f.startswith(base) and (f.endswith('.xlsx') or f.endswith('.parquet'))],
                reverse=True
            )
            for old in backups[MAX_AUTO_BACKUPS:]:
                os.remove(os.path.join(backup_dir, old))
        except Exception:
            pass

    def _verify_before_save(self):
        """保存前完整性校验

        检查项:
        1. _load_ok 标志 — 加载失败后禁止保存
        2. sheet 数量不能减少
        3. 每个已有 sheet 的行数不能大幅缩水（容忍0行→有行，不容忍有行→0行）
        4. 每个已有 sheet 的日期列数不能减少

        Returns:
            (ok, reason): ok=True 允许保存; ok=False 附带拒绝原因
        """
        if not self._load_ok:
            return False, "数据库加载失败，禁止保存（防止覆盖丢失数据）"

        if not self._orig_stats:
            return True, ""  # 首次创建，无需对比

        # 检查 sheet 数量
        for orig_name in self._orig_stats:
            if orig_name not in self.data:
                return False, f"原有 sheet '{orig_name}' 在内存中消失，禁止保存"

        # 逐 sheet 检查
        for name, orig in self._orig_stats.items():
            df = self.data[name]
            cur_rows = len(df)
            cur_cols = len(df.columns)
            orig_rows = orig['rows']
            orig_cols = orig['cols']

            # 行数: 不允许从有数据变成空
            if orig_rows > 0 and cur_rows == 0:
                return False, f"sheet '{name}' 行数从 {orig_rows} 变成 0，禁止保存"

            # 行数: 不允许缩水超过50%
            if orig_rows > 10 and cur_rows < orig_rows * 0.5:
                return False, (f"sheet '{name}' 行数从 {orig_rows} 降至 {cur_rows} "
                               f"(缩水>{50}%)，禁止保存")

            # 日期列数: 不允许减少
            if orig_cols > 0 and cur_cols < orig_cols:
                return False, (f"sheet '{name}' 列数从 {orig_cols} 降至 {cur_cols}，"
                               f"存在丢列风险，禁止保存")

        return True, ""

    def _save_database(self):
        """保存数据库到Parquet（带备份 + 完整性校验）"""
        # ★ 第1关: 完整性校验
        ok, reason = self._verify_before_save()
        if not ok:
            logger.error(f"[NAV数据库] !! 保存被阻止: {reason}")
            # 降级: 保存到应急文件，不覆盖原数据库
            emergency = self.db_file.replace('.xlsx', f'_EMERGENCY_{datetime.now():%H%M%S}.parquet')
            try:
                self._write_parquet(emergency)
                logger.error(f"[NAV数据库] 数据已保存到应急文件: {emergency}（原数据库未被修改）")
            except Exception as e2:
                logger.error(f"[NAV数据库] 应急保存也失败: {e2}")
            return

        # ★ 第2关: 备份原文件
        self._create_backup()

        # ★ 第3关: 写入Parquet
        parquet_path = self.db_file.replace('.xlsx', '.parquet')
        try:
            t0 = time.time()
            self._write_parquet(parquet_path)
            elapsed = time.time() - t0
            logger.info(f"[NAV数据库] 已保存到 {parquet_path} ({elapsed:.2f}s)")
        except PermissionError:
            fallback = self.db_file.replace('.xlsx', f'_backup_{datetime.now():%H%M%S}.parquet')
            try:
                self._write_parquet(fallback)
                logger.warning(f"[NAV数据库] 原文件被占用，已保存到 {fallback}")
            except Exception as e2:
                logger.error(f"[NAV数据库] 保存失败: {e2}")

    def _write_excel(self, path):
        """实际写入 Excel 的底层方法（Legacy，仅用于向后兼容）"""
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for sheet_name, df in self.data.items():
                if isinstance(df.index, pd.MultiIndex):
                    df_to_save = df.reset_index()
                else:
                    df_to_save = df
                df_to_save.to_excel(writer, sheet_name=sheet_name, index=False)

    def _write_parquet(self, path):
        """实际写入 Parquet 的底层方法（快速：~0.5s vs Excel ~240s）"""
        all_rows = []
        for sheet_name, df in self.data.items():
            if isinstance(df.index, pd.MultiIndex):
                df_flat = df.reset_index()
            else:
                df_flat = df.copy()

            # 找出日期列
            date_cols = [c for c in df_flat.columns if self._is_date_column(c)]
            if not date_cols:
                continue

            # 构建id_vars: 基础列 + 存在的费率列
            id_vars = ['产品代码', '产品名称']
            fee_cols_present = [c for c in FEE_COLUMNS if c in df_flat.columns]
            id_vars += fee_cols_present

            # Melt: 宽表 -> 长表
            melted = df_flat.melt(
                id_vars=id_vars,
                value_vars=date_cols,
                var_name='日期',
                value_name='净值'
            )

            # 去掉空值
            melted = melted.dropna(subset=['净值'])
            melted = melted[melted['净值'].astype(str).str.strip() != '']

            # 添加银行列
            melted['银行'] = sheet_name

            # 转换类型
            melted['净值'] = pd.to_numeric(melted['净值'], errors='coerce')
            melted['日期'] = pd.to_datetime(melted['日期'], format='%Y-%m-%d', errors='coerce')
            melted = melted.dropna(subset=['净值', '日期'])

            all_rows.append(melted)

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            # 基础列 + 存在的费率列 + 日期 + 净值
            base_cols = ['银行', '产品代码', '产品名称']
            fee_cols_in_combined = [c for c in FEE_COLUMNS if c in combined.columns]
            output_cols = base_cols + fee_cols_in_combined + ['日期', '净值']
            combined = combined[output_cols]
            combined = combined.drop_duplicates(subset=['银行', '产品代码', '日期'])
            combined = combined.sort_values(['银行', '产品代码', '日期']).reset_index(drop=True)
            combined.to_parquet(path, compression='snappy', index=False)
        else:
            # 创建空Parquet（保持schema一致）
            empty = pd.DataFrame(columns=['银行', '产品代码', '产品名称', '日期', '净值'])
            empty['日期'] = pd.to_datetime(empty['日期'])
            empty['净值'] = pd.to_numeric(empty['净值'])
            empty.to_parquet(path, compression='snappy', index=False)

    def update_nav(self, bank_name: str, products: List[Dict]) -> Dict:
        """
        更新某银行的净值数据（增量更新）

        Args:
            bank_name: 银行名称（民生/华夏/中信）
            products: 产品列表，每个产品需包含：
                - product_code: 产品代码
                - product_name: 产品名称
                - nav_history: 净值历史列表 [{date: "2024-01-01", unit_nav: 1.0234}, ...]

        Returns:
            更新统计 {total_products, new_products, new_dates, updated_cells}
        """
        sheet_name = self._get_sheet_name(bank_name)

        # 获取或创建DataFrame
        if sheet_name in self.data:
            df = self.data[sheet_name]
            # 如果不是MultiIndex，尝试设置
            if not isinstance(df.index, pd.MultiIndex):
                # 兼容 level_0/level_1 列名
                if 'level_0' in df.columns and '产品代码' not in df.columns:
                    df = df.rename(columns={'level_0': '产品代码', 'level_1': '产品名称'})
                if '产品代码' in df.columns and '产品名称' in df.columns:
                    # 去除产品代码为空的重复行（历史遗留数据修复）
                    df = df.dropna(subset=['产品代码'])
                    df = df.set_index(['产品代码', '产品名称'])
                    df = df.sort_index()  # 排序索引，避免 lexsort 错误
        else:
            df = pd.DataFrame()
            df.index = pd.MultiIndex.from_tuples([], names=['产品代码', '产品名称'])

        stats = {
            'total_products': len(products),
            'new_products': 0,
            'new_dates': set(),
            'updated_cells': 0,
        }

        # 收集所有日期
        all_dates = set(df.columns.tolist()) if len(df.columns) > 0 else set()

        # 先收集所有新日期，一次性添加列（避免 DataFrame fragmentation）
        new_dates_to_add = set()
        for product in products:
            for nav_item in product.get('nav_history', []):
                date = self._normalize_date(nav_item.get('date', ''))
                if date and date not in all_dates:
                    new_dates_to_add.add(date)

        if new_dates_to_add:
            new_cols_df = pd.DataFrame(index=df.index, columns=sorted(new_dates_to_add))
            df = pd.concat([df, new_cols_df], axis=1)
            all_dates.update(new_dates_to_add)
            stats['new_dates'] = new_dates_to_add

        for product in products:
            product_code = str(product.get('product_code', '')).strip()
            product_name = str(product.get('product_name', '')).strip()
            nav_history = product.get('nav_history', [])

            if not product_code or not nav_history:
                continue

            # 产品索引
            idx = (product_code, product_name)

            # 检查是否是新产品
            if idx not in df.index:
                stats['new_products'] += 1
                new_idx = pd.MultiIndex.from_tuples([idx], names=['产品代码', '产品名称'])
                new_row = pd.DataFrame(index=new_idx, columns=df.columns)
                df = pd.concat([df, new_row])
                df = df.sort_index()  # 保持索引有序

            # 更新净值数据
            for nav_item in nav_history:
                date = nav_item.get('date', '')
                unit_nav = nav_item.get('unit_nav')

                if not date or unit_nav is None:
                    continue

                date = self._normalize_date(date)
                if not date or date not in df.columns:
                    continue

                # 使用 loc 代替 at，避免 MultiIndex 的 scalar access 错误
                try:
                    current_value = df.loc[idx, date]
                    if isinstance(current_value, pd.Series):
                        current_value = current_value.iloc[0]
                except (KeyError, IndexError):
                    current_value = None

                if pd.isna(current_value) or current_value is None or str(current_value).strip() == '':
                    try:
                        df.loc[idx, date] = str(unit_nav)
                        stats['updated_cells'] += 1
                    except Exception:
                        pass

        # 按日期排序列（从旧到新）
        date_cols = sorted([c for c in df.columns if self._is_date_column(c)])
        other_cols = [c for c in df.columns if not self._is_date_column(c)]
        df = df[other_cols + date_cols]

        # 保存回数据字典
        self.data[sheet_name] = df

        # 转换set为list以便返回
        stats['new_dates'] = list(stats['new_dates'])

        logger.info(f"[NAV数据库] {sheet_name}: 更新 {stats['total_products']} 个产品, "
                   f"新增 {stats['new_products']} 个产品, "
                   f"新增 {len(stats['new_dates'])} 个日期, "
                   f"更新 {stats['updated_cells']} 个单元格")

        return stats

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """标准化日期格式为 YYYY-MM-DD"""
        if not date_str:
            return None

        date_str = str(date_str).strip()

        # 已经是标准格式
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str

        # YYYYMMDD 格式
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # 尝试解析其他格式
        for fmt in ['%Y/%m/%d', '%Y.%m.%d', '%Y%m%d', '%Y-%m-%d']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except:
                continue

        return None

    def _is_date_column(self, col_name: str) -> bool:
        """判断是否是日期列"""
        if not isinstance(col_name, str):
            return False
        # 匹配 YYYY-MM-DD 格式
        if len(col_name) == 10 and col_name[4] == '-' and col_name[7] == '-':
            try:
                datetime.strptime(col_name, '%Y-%m-%d')
                return True
            except:
                return False
        return False

    def save(self):
        """保存数据库"""
        self._save_database()

    def get_product_nav(self, bank_name: str, product_code: str) -> Optional[Dict]:
        """
        获取某产品的所有净值数据

        Returns:
            {product_code, product_name, nav_history: [{date, unit_nav}, ...]}
        """
        sheet_name = self._get_sheet_name(bank_name)

        if sheet_name not in self.data:
            return None

        df = self.data[sheet_name]

        # 在索引中查找产品
        for idx in df.index:
            if idx[0] == product_code:
                product_name = idx[1]
                nav_history = []

                for col in df.columns:
                    if self._is_date_column(col):
                        value = df.at[idx, col]
                        if pd.notna(value) and str(value).strip():
                            nav_history.append({
                                'date': col,
                                'unit_nav': float(value)
                            })

                # 按日期排序（从新到旧）
                nav_history.sort(key=lambda x: x['date'], reverse=True)

                return {
                    'product_code': product_code,
                    'product_name': product_name,
                    'nav_history': nav_history
                }

        return None

    def get_all_dates(self, bank_name: str) -> List[str]:
        """获取某银行所有的净值日期"""
        sheet_name = self._get_sheet_name(bank_name)

        if sheet_name not in self.data:
            return []

        df = self.data[sheet_name]
        return sorted([c for c in df.columns if self._is_date_column(c)])

    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        stats = {}
        for sheet_name, df in self.data.items():
            date_cols = [c for c in df.columns if self._is_date_column(c)]
            stats[sheet_name] = {
                'products': len(df),
                'dates': len(date_cols),
                'earliest_date': min(date_cols) if date_cols else None,
                'latest_date': max(date_cols) if date_cols else None,
            }
        return stats


def inject_fee_columns(nav_db: NAVDatabaseExcel) -> Dict:
    """将赎回费数据库中的费率信息注入到净值数据库的各银行DataFrame中

    为每个银行的每个产品添加费率列:
    - 认购费: "0.00%" 或 ""
    - 申购费: "0.00%" 或 ""
    - 赎回费: "0-180天:1.00%; 180天以上:0.00%" 或 "无" 或 ""
    - 销售服务费: "0.40%" 或 ""
    - 托管费: "0.03%" 或 ""
    - 管理费: "0.20%" 或 ""

    Args:
        nav_db: NAVDatabaseExcel实例（已加载数据）

    Returns:
        dict: 统计 {bank: {total, injected}}
    """
    from redemption_fee_db import (
        load_fee_db, get_fee_info, format_redemption_fee, format_rate
    )

    load_fee_db()
    stats = {}

    for sheet_name, df in nav_db.data.items():
        if df.empty:
            continue

        bank = sheet_name  # sheet名就是银行名

        # 确保有MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            index_tuples = df.index.tolist()
        else:
            # 尝试从列构建
            if '产品代码' in df.columns and '产品名称' in df.columns:
                index_tuples = list(zip(df['产品代码'], df['产品名称']))
            else:
                stats[bank] = {'total': len(df), 'injected': 0}
                continue

        total = len(index_tuples)
        injected = 0

        # 初始化费率列为空字符串
        for col in FEE_COLUMNS:
            if isinstance(df.index, pd.MultiIndex):
                df[col] = ''
            else:
                df[col] = ''

        for idx_tuple in index_tuples:
            product_code = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            fee_info = get_fee_info(bank, str(product_code))

            if fee_info is None:
                continue

            injected += 1
            try:
                df.loc[idx_tuple, '认购费'] = format_rate(fee_info.get('subscription_fee'))
                df.loc[idx_tuple, '申购费'] = format_rate(fee_info.get('purchase_fee'))
                df.loc[idx_tuple, '赎回费'] = format_redemption_fee(fee_info)
                df.loc[idx_tuple, '销售服务费'] = format_rate(fee_info.get('sales_service_fee'))
                df.loc[idx_tuple, '托管费'] = format_rate(fee_info.get('custody_fee'))
                df.loc[idx_tuple, '管理费'] = format_rate(fee_info.get('management_fee'))
            except Exception as e:
                logger.warning(f"[费率注入] {bank}/{product_code} 写入失败: {e}")

        stats[bank] = {'total': total, 'injected': injected}
        logger.info(f"[费率注入] {bank}: {injected}/{total} 个产品已注入费率列")

    return stats


class NavDBReader:
    """净值数据库读取适配器（Parquet优先，回退Excel）

    提供与 pd.ExcelFile 类似的接口，用于替换策略/回测中的Excel读取逻辑。
    读取后的DataFrame与 pd.read_excel(dtype=str) 格式一致。

    Usage:
        reader = NavDBReader()
        for sheet in reader.sheet_names:
            df = reader.read_sheet(sheet)
    """

    def __init__(self, db_file=None):
        xlsx_path = db_file or NAV_DB_FILE
        parquet_path = xlsx_path.replace('.xlsx', '.parquet')
        self._sheets = {}

        if os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
            self._load_parquet(parquet_path)
        elif os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 0:
            self._load_excel(xlsx_path)

    def _load_parquet(self, path):
        t0 = time.time()
        df = pd.read_parquet(path)

        # 检测费率列是否存在
        fee_cols_present = [c for c in FEE_COLUMNS if c in df.columns]

        for bank, group in df.groupby('银行'):
            # pivot时将费率列纳入index，避免被当作value
            pivot_index = ['产品代码', '产品名称'] + fee_cols_present
            wide = group.pivot_table(
                index=pivot_index,
                columns='日期', values='净值', aggfunc='first'
            ).reset_index()
            # 日期列名转为字符串
            new_cols = []
            for c in wide.columns:
                if hasattr(c, 'strftime'):
                    new_cols.append(c.strftime('%Y-%m-%d'))
                else:
                    new_cols.append(str(c))
            wide.columns = new_cols
            # 全部转为字符串（匹配 pd.read_excel(dtype=str) 行为）
            wide = wide.astype(str)
            # NaN净值还原为空字符串
            meta_cols = {'产品代码', '产品名称'} | set(FEE_COLUMNS)
            date_cols = [c for c in wide.columns if c not in meta_cols]
            for col in date_cols:
                wide[col] = wide[col].replace('nan', '')
            # 费率列中的nan也还原为空字符串
            for col in fee_cols_present:
                if col in wide.columns:
                    wide[col] = wide[col].replace('nan', '')
            self._sheets[bank] = wide
        elapsed = time.time() - t0
        logger.info(f"[NavDBReader] Parquet加载完成 ({elapsed:.2f}s), {len(self._sheets)} 个银行")

    def _load_excel(self, path):
        xlsx = pd.ExcelFile(path)
        for sheet in xlsx.sheet_names:
            self._sheets[sheet] = pd.read_excel(xlsx, sheet_name=sheet, dtype=str)
        logger.info(f"[NavDBReader] Excel加载完成 (Legacy), {len(self._sheets)} 个银行")

    @property
    def sheet_names(self):
        return list(self._sheets.keys())

    def read_sheet(self, name):
        """读取指定银行的DataFrame（宽格式，dtype=str兼容）"""
        return self._sheets.get(name, pd.DataFrame())


_db_write_lock = threading.Lock()


def update_nav_database(bank_name: str, profiles: List, db_file: str = None) -> Dict:
    """
    便捷函数：从爬虫结果更新净值数据库

    Args:
        bank_name: 银行名称
        profiles: 产品Profile列表（从爬虫返回）
        db_file: 可选的数据库文件路径

    Returns:
        更新统计
    """
    # 转换profiles为标准格式（无需加锁，纯内存操作）
    products = []
    for profile in profiles:
        # 支持不同的数据结构
        if hasattr(profile, 'product_code'):
            # ProductProfile 对象
            product = {
                'product_code': profile.product_code,
                'product_name': profile.product_name,
                'nav_history': []
            }

            # 获取nav_history
            if hasattr(profile, 'nav_history') and profile.nav_history:
                for nav in profile.nav_history:
                    if isinstance(nav, dict):
                        product['nav_history'].append({
                            'date': nav.get('date') or nav.get('ISS_DATE', ''),
                            'unit_nav': nav.get('unit_nav') or nav.get('NAV')
                        })
        elif isinstance(profile, dict):
            # 字典格式
            product = {
                'product_code': profile.get('product_code') or profile.get('REAL_PRD_CODE', ''),
                'product_name': profile.get('product_name') or profile.get('PRD_NAME', ''),
                'nav_history': profile.get('nav_history', [])
            }
        else:
            continue

        if product['product_code'] and product['nav_history']:
            products.append(product)

    # 加锁保护 load→update→save（读-改-写不能并发）
    with _db_write_lock:
        db = NAVDatabaseExcel(db_file)
        stats = db.update_nav(bank_name, products)
        db.save()

    return stats


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 测试
    db = NAVDatabaseExcel()

    # 模拟数据
    test_products = [
        {
            'product_code': 'TEST001',
            'product_name': '测试产品1',
            'nav_history': [
                {'date': '2024-01-20', 'unit_nav': 1.0234},
                {'date': '2024-01-21', 'unit_nav': 1.0256},
                {'date': '2024-01-22', 'unit_nav': 1.0278},
            ]
        },
        {
            'product_code': 'TEST002',
            'product_name': '测试产品2',
            'nav_history': [
                {'date': '2024-01-20', 'unit_nav': 1.1000},
                {'date': '2024-01-21', 'unit_nav': 1.1050},
            ]
        }
    ]

    # 更新数据库
    stats = db.update_nav('民生', test_products)
    print(f"更新统计: {stats}")

    # 保存
    db.save()

    # 查看统计
    print(f"数据库统计: {db.get_stats()}")
