"""
单独爬取华夏理财产品
"""

import sys
sys.path.insert(0, '.')

from bank_crawler import HuaxiaCrawler, MultiBankCrawler, BankType
from datetime import datetime
import logging

# 导入净值数据库Excel管理模块
try:
    from nav_db_excel import update_nav_database
    HAS_NAV_DB = True
except ImportError:
    HAS_NAV_DB = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def update_nav_db(profiles):
    """更新净值数据库Excel"""
    if not HAS_NAV_DB:
        logger.warning("净值数据库模块未安装，跳过数据库更新")
        return

    try:
        logger.info("正在更新净值数据库...")
        # 转换profiles为数据库需要的格式
        products_for_db = []
        for p in profiles:
            nav_history = []
            if hasattr(p, 'nav_history') and p.nav_history:
                for nav in p.nav_history:
                    if isinstance(nav, dict):
                        nav_history.append({
                            'date': nav.get('date', ''),
                            'unit_nav': nav.get('unit_nav')
                        })

            if nav_history:
                products_for_db.append({
                    'product_code': p.product_code,
                    'product_name': p.product_name,
                    'nav_history': nav_history
                })

        if products_for_db:
            stats = update_nav_database('华夏', products_for_db)
            logger.info(f"净值数据库更新完成: 新增 {len(stats.get('new_dates', []))} 个日期, "
                       f"更新 {stats.get('updated_cells', 0)} 个单元格")
    except Exception as e:
        logger.warning(f"更新净值数据库失败: {e}")


def main():
    print("="*60)
    print("    华夏理财产品采集")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 创建管理器，只注册华夏理财
    manager = MultiBankCrawler()
    manager.register_crawler(HuaxiaCrawler())

    # 爬取
    profiles = manager.crawl_all()

    if profiles:
        filename = manager.save_results()
        print(f"\n完成！获取 {len(profiles)} 个产品")
        print(f"文件: {filename}")

        # 更新净值数据库Excel
        update_nav_db(profiles)
    else:
        print("\n未获取到数据，可能是网络问题")


if __name__ == "__main__":
    main()
