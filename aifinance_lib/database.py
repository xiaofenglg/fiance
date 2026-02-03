# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
from typing import List, Dict, Optional
import logging
import threading

logger = logging.getLogger(__name__)

# 全局数据库文件名
DB_FILE = "aifinance_lib/aifinance.sqlite3"
FEE_COLUMNS = ['认购费', '申购费', '赎回费', '销售服务费', '托管费', '管理费']
FEE_COLUMNS_DB = ['subscription_fee', 'purchase_fee', 'redemption_fee', 'sales_service_fee', 'custody_fee', 'management_fee']


class Database:
    """
    统一的数据库访问层 (DAL)，使用 SQLite。
    该类是线程安全的。
    """
    _local = threading.local()

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file

    def get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接，如果不存在则创建。"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            logger.info(f"为线程 {threading.get_ident()} 创建新的数据库连接。")
        return self._local.conn

    def close_conn(self):
        """关闭当前线程的数据库连接。"""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
            logger.info(f"关闭线程 {threading.get_ident()} 的数据库连接。")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_conn()

    def _create_tables(self):
        """创建数据库表结构（如果不存在）。"""
        conn = self.get_conn()
        with conn:
            # 创建产品表
            conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_code TEXT PRIMARY KEY,
                product_name TEXT,
                bank_name TEXT,
                subscription_fee TEXT,
                purchase_fee TEXT,
                redemption_fee TEXT,
                sales_service_fee TEXT,
                custody_fee TEXT,
                management_fee TEXT
            )
            """)
            # 创建净值历史表
            conn.execute("""
            CREATE TABLE IF NOT EXISTS nav_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_code TEXT NOT NULL,
                date DATE NOT NULL,
                nav REAL NOT NULL,
                FOREIGN KEY (product_code) REFERENCES products (product_code),
                UNIQUE (product_code, date)
            )
            """)
            # 创建索引以加速查询
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nav_product_date ON nav_history (product_code, date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_products_bank ON products (bank_name);")
        logger.info("数据库表结构初始化/验证完成。")

    def update_products_and_nav(self, bank_name: str, products_data: List[Dict]):
        """
        批量更新或插入产品及其净值历史。
        这是一个原子操作，使用事务来确保数据一致性。
        """
        conn = self.get_conn()
        
        products_to_upsert = []
        navs_to_insert = []

        for p_data in products_data:
            if not p_data.get('product_code'):
                continue
            
            # 准备产品数据
            product_tuple = (
                p_data['product_code'],
                p_data.get('product_name', ''),
                bank_name,
                p_data.get('subscription_fee', ''),
                p_data.get('purchase_fee', ''),
                p_data.get('redemption_fee', ''),
                p_data.get('sales_service_fee', ''),
                p_data.get('custody_fee', ''),
                p_data.get('management_fee', '')
            )
            products_to_upsert.append(product_tuple)

            # 准备净值数据
            for nav_item in p_data.get('nav_history', []):
                if nav_item.get('date') and nav_item.get('unit_nav') is not None:
                    nav_tuple = (
                        p_data['product_code'],
                        nav_item['date'],
                        nav_item['unit_nav']
                    )
                    navs_to_insert.append(nav_tuple)

        if not products_to_upsert and not navs_to_insert:
            return

        with conn:
            # 使用 INSERT OR REPLACE 来插入或更新产品信息
            conn.executemany("""
                INSERT OR REPLACE INTO products (
                    product_code, product_name, bank_name,
                    subscription_fee, purchase_fee, redemption_fee,
                    sales_service_fee, custody_fee, management_fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, products_to_upsert)

            # 使用 INSERT OR IGNORE 来插入新的净值，忽略已存在的记录
            conn.executemany("""
                INSERT OR IGNORE INTO nav_history (product_code, date, nav)
                VALUES (?, ?, ?)
            """, navs_to_insert)
        
        logger.info(f"银行 '{bank_name}': 处理了 {len(products_to_upsert)} 个产品, {len(navs_to_insert)} 条净值记录。")

    def get_nav_wide_format(self, bank_name: str) -> pd.DataFrame:
        """
        为指定银行获取宽格式的净值DataFrame，以兼容旧的分析脚本。
        """
        conn = self.get_conn()
        
        query = """
        SELECT
            p.product_code AS "产品代码",
            p.product_name AS "产品名称",
            h.date AS "日期",
            h.nav AS "净值",
            p.subscription_fee AS "认购费",
            p.purchase_fee AS "申购费",
            p.redemption_fee AS "赎回费",
            p.sales_service_fee AS "销售服务费",
            p.custody_fee AS "托管费",
            p.management_fee AS "管理费"
        FROM products p
        JOIN nav_history h ON p.product_code = h.product_code
        WHERE p.bank_name = ?
        """
        
        df_long = pd.read_sql_query(query, conn, params=(bank_name,))
        
        if df_long.empty:
            return pd.DataFrame()

        # 转换日期格式
        df_long['日期'] = pd.to_datetime(df_long['日期']).dt.strftime('%Y-%m-%d')
        
        # 定义主键和费率列
        id_vars = ['产品代码', '产品名称'] + FEE_COLUMNS
        
        # Pivot: 长表 -> 宽表
        df_wide = df_long.pivot_table(
            index=id_vars,
            columns='日期',
            values='净值',
            aggfunc='first'
        ).reset_index()
        
        # 排序日期列
        meta_cols = set(id_vars)
        date_cols = sorted([c for c in df_wide.columns if c not in meta_cols])
        
        final_cols = id_vars + date_cols
        df_wide = df_wide[final_cols]
        
        logger.info(f"为银行 '{bank_name}' 生成了包含 {len(df_wide)} 个产品和 {len(date_cols)} 个日期的宽表。")
        return df_wide

    def get_bank_names(self) -> List[str]:
        """获取数据库中所有唯一的银行名称列表。"""
        conn = self.get_conn()
        try:
            cursor = conn.execute("SELECT DISTINCT bank_name FROM products ORDER BY bank_name")
            bank_names = [row[0] for row in cursor.fetchall()]
            logger.info(f"从数据库中获取到 {len(bank_names)} 个银行名称。")
            return bank_names
        except sqlite3.Error as e:
            logger.error(f"查询银行名称列表时出错: {e}")
            return []

# 在第一次使用时，确保表已创建
# 这不是一个完美的单例，但在多线程脚本中是安全的
_db_init_lock = threading.Lock()
_db_initialized = False

def initialize_database():
    """全局初始化数据库，只执行一次。"""
    global _db_initialized
    with _db_init_lock:
        if not _db_initialized:
            with Database() as db:
                db._create_tables()
            _db_initialized = True

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 初始化
    initialize_database()
    
    # 使用上下文管理器确保连接被关闭
    with Database() as db:
        # 模拟数据
        mock_products = [
            {
                'product_code': 'DB_TEST001',
                'product_name': '数据库测试产品1',
                'nav_history': [
                    {'date': '2025-01-01', 'unit_nav': 1.0},
                    {'date': '2025-01-02', 'unit_nav': 1.01},
                ],
                'redemption_fee': '0-7天:1.5%'
            },
            {
                'product_code': 'DB_TEST002',
                'product_name': '数据库测试产品2',
                'nav_history': [
                    {'date': '2025-01-01', 'unit_nav': 2.0},
                ]
            }
        ]
        
        # 测试更新
        db.update_products_and_nav("中信银行", mock_products)
        
        # 测试读取
        df_wide = db.get_nav_wide_format("中信银行")
        print("从数据库读取的宽表数据:")
        print(df_wide.head())

        # 验证数据
        test_product = df_wide[df_wide['产品代码'] == 'DB_TEST001']
        if not test_product.empty:
            print("\n验证 DB_TEST001 数据:")
            print(test_product[['产品代码', '产品名称', '2025-01-01', '2025-01-02', '赎回费']])
