# -*- coding: utf-8 -*-
"""
理财产品净值数据库

设计目标：
1. 持久化存储所有历史净值数据
2. 支持增量更新（只添加新数据）
3. 长期积累形成完整的历史数据
4. 提供统一的数据查询接口

数据结构：
- 产品基础信息: products.json
- 净值历史数据: nav_history/{bank}_{product_code}.json
- 元数据: metadata.json (最后更新时间等)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path


class NAVDatabase:
    """净值数据库"""

    def __init__(self, data_dir: str = "D:/AI-FINANCE/nav_db"):
        """
        初始化数据库

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.products_file = self.data_dir / "products.json"
        self.metadata_file = self.data_dir / "metadata.json"
        self.nav_dir = self.data_dir / "nav_history"

        # 创建目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.nav_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.products = self._load_json(self.products_file, {})
        self.metadata = self._load_json(self.metadata_file, {
            "created_at": datetime.now().isoformat(),
            "last_update": None,
            "banks": {},
        })

    def _load_json(self, filepath: Path, default: dict) -> dict:
        """加载JSON文件"""
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return default
        return default

    def _save_json(self, filepath: Path, data: dict):
        """保存JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _get_nav_file(self, bank: str, product_code: str) -> Path:
        """获取产品净值文件路径"""
        # 清理文件名中的非法字符
        safe_code = "".join(c for c in product_code if c.isalnum() or c in '-_')
        return self.nav_dir / f"{bank}_{safe_code}.json"

    def add_product(self, bank: str, product_code: str, product_info: dict):
        """
        添加或更新产品基础信息

        Args:
            bank: 银行名称
            product_code: 产品代码
            product_info: 产品信息字典
        """
        key = f"{bank}_{product_code}"
        if key not in self.products:
            self.products[key] = {
                "bank": bank,
                "code": product_code,
                "created_at": datetime.now().isoformat(),
            }

        # 更新产品信息
        self.products[key].update(product_info)
        self.products[key]["updated_at"] = datetime.now().isoformat()

        # 保存
        self._save_json(self.products_file, self.products)

    def add_nav_history(self, bank: str, product_code: str,
                        nav_history: List[Dict]) -> int:
        """
        添加净值历史数据（增量更新）

        Args:
            bank: 银行名称
            product_code: 产品代码
            nav_history: 净值历史列表，每条记录需包含 'date' 和 'nav' 字段

        Returns:
            新增的记录数
        """
        nav_file = self._get_nav_file(bank, product_code)

        # 加载现有数据
        existing = self._load_json(nav_file, {"history": []})
        existing_dates = set(h.get('date') for h in existing.get('history', []))

        # 增量添加新数据
        new_count = 0
        for nav in nav_history:
            date = nav.get('date')
            if date and date not in existing_dates:
                existing['history'].append(nav)
                existing_dates.add(date)
                new_count += 1

        # 按日期排序（最新的在前）
        existing['history'].sort(key=lambda x: x.get('date', ''), reverse=True)

        # 更新元数据
        existing['bank'] = bank
        existing['product_code'] = product_code
        existing['updated_at'] = datetime.now().isoformat()
        existing['total_records'] = len(existing['history'])

        if existing['history']:
            existing['latest_date'] = existing['history'][0].get('date')
            existing['earliest_date'] = existing['history'][-1].get('date')

        # 保存
        self._save_json(nav_file, existing)

        return new_count

    def get_nav_history(self, bank: str, product_code: str,
                        start_date: str = None,
                        end_date: str = None) -> List[Dict]:
        """
        获取产品净值历史

        Args:
            bank: 银行名称
            product_code: 产品代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            净值历史列表
        """
        nav_file = self._get_nav_file(bank, product_code)
        data = self._load_json(nav_file, {"history": []})
        history = data.get('history', [])

        # 日期筛选
        if start_date or end_date:
            filtered = []
            for h in history:
                date = h.get('date', '')
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                filtered.append(h)
            return filtered

        return history

    def get_product_info(self, bank: str, product_code: str) -> Optional[Dict]:
        """获取产品基础信息"""
        key = f"{bank}_{product_code}"
        return self.products.get(key)

    def get_all_products(self, bank: str = None) -> List[Dict]:
        """
        获取所有产品列表

        Args:
            bank: 可选，筛选特定银行

        Returns:
            产品列表
        """
        products = list(self.products.values())
        if bank:
            products = [p for p in products if p.get('bank') == bank]
        return products

    def get_latest_nav(self, bank: str, product_code: str) -> Optional[Dict]:
        """获取最新净值"""
        history = self.get_nav_history(bank, product_code)
        return history[0] if history else None

    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        stats = {
            "total_products": len(self.products),
            "banks": {},
            "total_nav_records": 0,
            "last_update": self.metadata.get("last_update"),
        }

        # 统计各银行产品数
        for product in self.products.values():
            bank = product.get('bank', '未知')
            if bank not in stats['banks']:
                stats['banks'][bank] = {"products": 0, "nav_records": 0}
            stats['banks'][bank]['products'] += 1

        # 统计净值记录数
        for nav_file in self.nav_dir.glob("*.json"):
            data = self._load_json(nav_file, {})
            records = len(data.get('history', []))
            stats['total_nav_records'] += records

            # 提取银行名称
            bank = data.get('bank', '未知')
            if bank in stats['banks']:
                stats['banks'][bank]['nav_records'] += records

        return stats

    def update_metadata(self, bank: str = None):
        """更新元数据"""
        self.metadata['last_update'] = datetime.now().isoformat()
        if bank:
            if 'banks' not in self.metadata:
                self.metadata['banks'] = {}
            self.metadata['banks'][bank] = {
                "last_update": datetime.now().isoformat()
            }
        self._save_json(self.metadata_file, self.metadata)

    def export_to_excel(self, output_file: str = None,
                        bank: str = None,
                        include_nav_history: bool = False) -> str:
        """
        导出数据到Excel

        Args:
            output_file: 输出文件路径
            bank: 筛选特定银行
            include_nav_history: 是否包含完整净值历史

        Returns:
            输出文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"D:/AI-FINANCE/净值数据库导出_{timestamp}.xlsx"

        products = self.get_all_products(bank)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 产品列表
            if products:
                df_products = pd.DataFrame(products)
                df_products.to_excel(writer, sheet_name='产品列表', index=False)

            # 最新净值汇总
            latest_navs = []
            for p in products:
                bank_name = p.get('bank')
                code = p.get('code')
                latest = self.get_latest_nav(bank_name, code)
                if latest:
                    latest_navs.append({
                        '银行': bank_name,
                        '产品代码': code,
                        '产品名称': p.get('name', ''),
                        **latest
                    })

            if latest_navs:
                df_latest = pd.DataFrame(latest_navs)
                df_latest.to_excel(writer, sheet_name='最新净值', index=False)

            # 统计信息
            stats = self.get_statistics()
            df_stats = pd.DataFrame([
                {'指标': '总产品数', '数值': stats['total_products']},
                {'指标': '总净值记录数', '数值': stats['total_nav_records']},
                {'指标': '最后更新时间', '数值': stats['last_update']},
            ])
            df_stats.to_excel(writer, sheet_name='统计信息', index=False)

            # 各银行统计
            bank_stats = []
            for bank_name, info in stats.get('banks', {}).items():
                bank_stats.append({
                    '银行': bank_name,
                    '产品数': info.get('products', 0),
                    '净值记录数': info.get('nav_records', 0),
                })
            if bank_stats:
                df_banks = pd.DataFrame(bank_stats)
                df_banks.to_excel(writer, sheet_name='各银行统计', index=False)

        print(f"数据已导出到: {output_file}")
        return output_file

    def import_from_crawler_result(self, profiles: List, bank: str = None):
        """
        从爬虫结果导入数据

        Args:
            profiles: 爬虫返回的ProductProfile列表
            bank: 银行名称（可选）
        """
        new_products = 0
        new_nav_records = 0

        for profile in profiles:
            # 获取产品信息
            if hasattr(profile, 'bank'):
                bank_name = profile.bank.value if hasattr(profile.bank, 'value') else str(profile.bank)
            else:
                bank_name = bank or '未知'

            product_code = getattr(profile, 'code', '') or getattr(profile, 'product_code', '')
            if not product_code:
                continue

            # 添加产品基础信息
            product_info = {
                'name': getattr(profile, 'name', ''),
                'risk_level': getattr(profile, 'risk_level', ''),
                'product_type': str(getattr(profile, 'product_type', '')),
                'status': str(getattr(profile, 'status', '')),
            }
            self.add_product(bank_name, product_code, product_info)
            new_products += 1

            # 添加净值历史
            nav_history = getattr(profile, 'nav_history', [])
            if nav_history:
                count = self.add_nav_history(bank_name, product_code, nav_history)
                new_nav_records += count

        # 更新元数据
        self.update_metadata(bank)

        print(f"导入完成: {new_products} 个产品, {new_nav_records} 条新净值记录")
        return new_products, new_nav_records


# 全局数据库实例
_db_instance = None


def get_database(data_dir: str = "D:/AI-FINANCE/nav_db") -> NAVDatabase:
    """获取数据库单例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = NAVDatabase(data_dir)
    return _db_instance


if __name__ == "__main__":
    # 测试
    db = get_database()

    print("=== 数据库统计 ===")
    stats = db.get_statistics()
    print(f"总产品数: {stats['total_products']}")
    print(f"总净值记录: {stats['total_nav_records']}")
    print(f"各银行: {stats['banks']}")
