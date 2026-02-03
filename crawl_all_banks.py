# -*- coding: utf-8 -*-
"""
三银行理财产品统一抓取程序

功能：
1. 抓取民生、华夏、中信三家银行的理财产品净值
2. 每家银行抓取完成后自动更新净值数据库Excel
3. 支持选择性抓取（通过命令行参数）
4. 支持完整/增量模式：
   - 首次运行用 --full 完整爬取（获取最大历史深度）
   - 后续运行默认增量模式（仅获取最近数据，追加到已有数据库）
5. 绝不删除已有净值数据，只追加新日期和新产品

使用方法：
    python crawl_all_banks.py                    # 增量抓取全部三家银行
    python crawl_all_banks.py --full             # 首次完整抓取（获取最大历史）
    python crawl_all_banks.py --minsheng         # 只抓取民生（增量）
    python crawl_all_banks.py --citic --full     # 完整抓取中信
    python crawl_all_banks.py --minsheng --huaxia  # 抓取民生和华夏
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime

# 确保路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_all_banks.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入净值数据库模块
try:
    from nav_db_excel import update_nav_database, NAVDatabaseExcel
    HAS_NAV_DB = True
except ImportError:
    HAS_NAV_DB = False
    logger.warning("净值数据库模块未找到，将跳过数据库更新")

# V13: Delta Sync
try:
    from crawl_utils import filter_products_by_manifest, needs_update, manifest_summary
except ImportError:
    filter_products_by_manifest = None

# 赎回费数据库
try:
    from redemption_fee_db import (load_fee_db, save_fee_db, update_fee_info,
                                    has_fee_data, parse_fee_from_name)
    HAS_FEE_DB = True
except ImportError:
    HAS_FEE_DB = False

# 费率数据库 (全量费率: 管理费/托管费/销售费/申购费/赎回费 + 优惠)
try:
    from fee_rate_db import (load_fee_rate_db, save_fee_rate_db,
                             update_product_fees, needs_update as fee_needs_update)
    HAS_FEE_RATE_DB = True
except ImportError:
    HAS_FEE_RATE_DB = False


def show_database_status():
    """显示当前数据库覆盖情况，帮助判断是否需要完整爬取"""
    if not HAS_NAV_DB:
        print("  数据库模块未加载，跳过状态检查")
        return {}

    try:
        db = NAVDatabaseExcel()
        stats = db.get_stats()

        print("\n" + "-" * 70)
        print("  当前净值数据库状态:")
        print("-" * 70)
        print(f"  {'银行':<12} {'产品数':<10} {'日期数':<10} {'最早日期':<14} {'最新日期':<14}")
        print(f"  {'-'*58}")

        for sheet, info in stats.items():
            earliest = info.get('earliest_date', 'N/A') or 'N/A'
            latest = info.get('latest_date', 'N/A') or 'N/A'
            print(f"  {sheet:<12} {info['products']:<10} {info['dates']:<10} {earliest:<14} {latest:<14}")

        # 判断数据覆盖率
        warnings = []
        for sheet, info in stats.items():
            if info['dates'] < 30:
                warnings.append(f"  ** {sheet}: 仅 {info['dates']} 个日期，数据偏少，建议用 --full 完整爬取")

        if not stats:
            warnings.append("  ** 数据库为空，建议用 --full 首次完整爬取")

        if warnings:
            print()
            for w in warnings:
                print(w)

        print("-" * 70)
        return stats

    except Exception as e:
        logger.warning(f"读取数据库状态失败: {e}")
        return {}


def crawl_minsheng(full_history: bool = False, progress_callback=None, manifest=None, stop_check=None) -> dict:
    """抓取民生银行

    Args:
        full_history: True=完整爬取(3年历史), False=增量(180天)
        progress_callback: 可选回调 (current, total, message) 用于报告进度
        manifest: V13 delta sync manifest {银行: {产品代码: last_date}}
        stop_check: callable, 返回True表示应停止
    """
    mode_str = "完整" if full_history else "增量"
    print("\n" + "="*70)
    print(f"              民生银行理财产品抓取 [{mode_str}模式]")
    print("="*70)

    result = {
        'bank': '民生银行',
        'success': False,
        'products': 0,
        'profiles': [],
        'error': None
    }

    try:
        # 获取前先检查停止
        if stop_check and stop_check():
            logger.info("[民生] 收到停止信号，跳过抓取")
            result['error'] = '用户停止'
            return result

        from minsheng import CMBCWealthCrawler

        crawler = CMBCWealthCrawler()

        # 获取产品列表
        products = crawler.get_product_list()

        # 获取产品列表后再检查停止（可能在获取期间收到停止信号）
        if stop_check and stop_check():
            logger.info("[民生] 收到停止信号，跳过产品处理")
            result['error'] = '用户停止'
            return result
        if not products:
            result['error'] = "未获取到产品列表"
            return result

        # V13: Delta Sync — 跳过已最新的产品
        total_before = len(products)
        if manifest and not full_history and filter_products_by_manifest:
            bank_m = manifest.get('民生银行', {})
            products, skipped = filter_products_by_manifest(
                products, bank_m, code_field='REAL_PRD_CODE', fallback_field='PRD_CODE')
            if skipped > 0:
                logger.info(f"[民生] Delta Sync: {total_before} 总产品, "
                           f"跳过 {skipped} (已最新), 需更新 {len(products)}")
                if progress_callback:
                    progress_callback(0, 1, f'民生银行 跳过{skipped}个已最新产品')
            if not products:
                logger.info(f"[民生] 所有产品已是最新，无需抓取")
                result['success'] = True
                return result

        logger.info(f"[民生] 需抓取 {len(products)} 个产品，{mode_str}模式开始分析...")

        # 分批处理
        batch_size = crawler.SESSION_NAV_LIMIT
        batches = [products[i:i+batch_size] for i in range(0, len(products), batch_size)]

        all_profiles = []
        saved_total = 0
        for i, batch in enumerate(batches):
            if stop_check and stop_check():
                logger.info(f"[民生] 收到停止信号，已处理 {i}/{len(batches)} 批次")
                break
            logger.info(f"[民生] 处理批次 {i+1}/{len(batches)}...")
            if progress_callback:
                progress_callback(i, len(batches), f'民生银行 批次 {i+1}/{len(batches)}')
            batch_results = crawler._process_batch(batch, i, full_history=full_history)
            batch_profiles = []
            for profile, reason in batch_results:
                if profile:
                    all_profiles.append(profile)
                    batch_profiles.append(profile)

            # 每批次完成后立即保存到数据库，防止中断丢失
            if HAS_NAV_DB and batch_profiles:
                try:
                    products_for_db = []
                    for p in batch_profiles:
                        nav_history = []
                        for nav in p.nav_history:
                            if nav.get('NAV'):
                                date_str = crawler.format_date(nav.get('ISS_DATE'))
                                nav_history.append({
                                    'date': date_str,
                                    'unit_nav': nav.get('NAV')
                                })
                        if nav_history:
                            products_for_db.append({
                                'product_code': p.product_code,
                                'product_name': p.product_name,
                                'nav_history': nav_history
                            })
                    if products_for_db:
                        stats = update_nav_database('民生', products_for_db)
                        saved_total += len(products_for_db)
                        logger.info(f"[民生] 批次{i+1}保存: {saved_total} 个产品已入库")
                except Exception as e:
                    logger.warning(f"[民生] 批次{i+1}数据库保存失败: {e}")

        # 赎回费提取: 从产品名称解析费用信息 (旧逻辑，保持兼容)
        if HAS_FEE_DB:
            fee_count = 0
            for p in all_profiles:
                if has_fee_data('民生银行', p.product_code):
                    continue
                fee_info = crawler._extract_fee_info(
                    {'PRD_NAME': p.product_name}, {})
                if fee_info:
                    update_fee_info('民生银行', p.product_code, fee_info)
                    if fee_info.get('has_redemption_fee'):
                        fee_count += 1
                else:
                    update_fee_info('民生银行', p.product_code, {
                        'has_redemption_fee': False,
                        'fee_schedule': [],
                        'fee_description': '',
                        'source': 'name_parse',
                    })
            save_fee_db()
            if fee_count:
                logger.info(f"[民生] 赎回费: 发现 {fee_count} 个有赎回费的产品")

        # 全量费率采集: 发行公告 + 费率优惠公告 PDF
        if HAS_FEE_RATE_DB:
            fee_rate_new = 0
            fee_rate_updated = 0
            fee_rate_skipped = 0
            fee_rate_failed = 0
            fee_batch_size = crawler.SESSION_NAV_LIMIT

            # 按批次处理，每批使用独立会话
            fee_products = list(all_profiles)
            fee_batches = [fee_products[i:i+fee_batch_size]
                           for i in range(0, len(fee_products), fee_batch_size)]

            logger.info(f"[民生] 费率采集: {len(fee_products)} 个产品, "
                        f"{len(fee_batches)} 批")

            for bi, fee_batch in enumerate(fee_batches):
                if stop_check and stop_check():
                    logger.info(f"[民生] 费率采集收到停止信号")
                    break

                fee_session = crawler._create_session()
                for p in fee_batch:
                    code = p.product_code

                    # 获取公告日期并检查是否需要更新
                    try:
                        announcements = crawler.get_fee_announcements(
                            code, fee_session)
                    except Exception:
                        announcements = {'issuance': None, 'discount': None}

                    issuance_date = None
                    discount_date = None
                    if announcements.get('issuance'):
                        issuance_date = announcements['issuance'].get('date')
                    if announcements.get('discount'):
                        discount_date = announcements['discount'].get('date')

                    if not fee_needs_update('民生', code,
                                            issuance_date, discount_date):
                        fee_rate_skipped += 1
                        continue

                    # 需要更新: 下载并解析 PDF
                    try:
                        fee_data = crawler.scrape_product_fees(
                            code, fee_session)
                        if fee_data:
                            fee_data['product_name'] = p.product_name
                            update_product_fees('民生', code, fee_data)
                            if fee_data.get('base_fees'):
                                fee_rate_new += 1
                            if fee_data.get('discounts'):
                                fee_rate_updated += 1
                        else:
                            fee_rate_failed += 1
                    except Exception as e:
                        logger.debug(f"[民生] 费率采集失败 {code}: {e}")
                        fee_rate_failed += 1

                    time.sleep(0.2)

                fee_session.close()

                if progress_callback:
                    progress_callback(bi + 1, len(fee_batches),
                                      f'民生银行 费率采集 {bi+1}/{len(fee_batches)}')

            save_fee_rate_db()
            logger.info(f"[民生] 费率采集完成: 新增基础费率 {fee_rate_new}, "
                        f"新增优惠 {fee_rate_updated}, "
                        f"跳过(已最新) {fee_rate_skipped}, "
                        f"失败 {fee_rate_failed}")

        result['profiles'] = all_profiles
        result['products'] = len(all_profiles)
        result['success'] = True

        logger.info(f"[民生] 抓取完成，共 {len(all_profiles)} 个有效产品, {saved_total} 个已入库")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[民生] 抓取失败: {e}")

    return result


def crawl_huaxia(full_history: bool = False, progress_callback=None, manifest=None, stop_check=None) -> dict:
    """抓取华夏银行

    Args:
        full_history: True=完整爬取, False=增量
        progress_callback: 可选回调 (current, total, message) 用于报告进度
        manifest: V13 delta sync manifest
        stop_check: callable, 返回True表示应停止
    """
    mode_str = "完整" if full_history else "增量"
    print("\n" + "="*70)
    print(f"              华夏银行理财产品抓取 [{mode_str}模式]")
    print("="*70)

    result = {
        'bank': '华夏银行',
        'success': False,
        'products': 0,
        'profiles': [],
        'error': None
    }

    try:
        # 获取前先检查停止
        if stop_check and stop_check():
            logger.info("[华夏] 收到停止信号，跳过抓取")
            result['error'] = '用户停止'
            return result

        from bank_crawler import HuaxiaCrawler
        import re

        crawler = HuaxiaCrawler()

        # 获取产品列表
        products = crawler.get_product_list()

        # 获取产品列表后再检查停止
        if stop_check and stop_check():
            logger.info("[华夏] 收到停止信号，跳过产品处理")
            result['error'] = '用户停止'
            return result
        if not products:
            result['error'] = "未获取到产品列表"
            return result

        # V13: Delta Sync — 跳过已最新的产品
        total_before = len(products)
        if manifest and not full_history and filter_products_by_manifest:
            bank_m = manifest.get('华夏银行', {})
            products, skipped = filter_products_by_manifest(
                products, bank_m, code_field='id', fallback_field='id')
            if skipped > 0:
                logger.info(f"[华夏] Delta Sync: {total_before} 总产品, "
                           f"跳过 {skipped} (已最新), 需更新 {len(products)}")
            if not products:
                logger.info(f"[华夏] 所有产品已是最新，无需抓取")
                result['success'] = True
                return result

        logger.info(f"[华夏] 需抓取 {len(products)} 个产品，{mode_str}模式开始处理...")

        # 处理每个产品
        all_profiles = []
        products_for_db = []
        total = len(products)
        processed = 0
        saved_total = 0
        skipped_no_path = 0
        skipped_no_nav = 0
        BATCH_SIZE = 200

        for i, product in enumerate(products):
            if stop_check and stop_check():
                logger.info(f"[华夏] 收到停止信号，已处理 {i}/{total}")
                break

            excel_url = product.get('address', '') or product.get('path', '')
            if not excel_url:
                skipped_no_path += 1
                continue

            # 下载并解析Excel获取净值数据
            nav_data = crawler.download_nav_excel(excel_url)
            if not nav_data:
                skipped_no_nav += 1
                continue

            # 合并历史数据
            product_code = product.get('id', '')
            product_title = product.get('title', '')

            # 从标题提取产品名称
            product_name = product_title
            name_match = re.search(r'^(.+?)\[', product_title)
            if name_match:
                product_name = name_match.group(1).strip()

            merged_nav = crawler._merge_nav_data(product_code, nav_data)

            # 计算指标
            profile = crawler.calculate_metrics(merged_nav, product)
            if profile:
                all_profiles.append(profile)

            # 直接准备数据库更新数据（即使 profile 为 None 也保存净值）
            if merged_nav:
                nav_history = []
                for nav in merged_nav:
                    if isinstance(nav, dict) and nav.get('unit_nav'):
                        nav_history.append({
                            'date': nav.get('date', ''),
                            'unit_nav': nav.get('unit_nav')
                        })
                if nav_history:
                    products_for_db.append({
                        'product_code': product_code,
                        'product_name': product_name,
                        'nav_history': nav_history
                    })
                    processed += 1

            # 分批保存，防止中断丢失
            if HAS_NAV_DB and len(products_for_db) >= BATCH_SIZE:
                try:
                    stats = update_nav_database('华夏', products_for_db)
                    saved_total += len(products_for_db)
                    logger.info(f"[华夏] 分批保存: {saved_total} 个产品已入库")
                    products_for_db = []
                except Exception as e:
                    logger.warning(f"[华夏] 分批保存失败: {e}")

            if (i + 1) % 20 == 0 or i + 1 == total:
                logger.info(f"[华夏] 进度: {i+1}/{total}, 已处理: {processed}")
                if progress_callback:
                    progress_callback(i + 1, total, f'华夏银行 {i+1}/{total} 产品')

        # 保存净值历史到JSON
        crawler._save_nav_storage()

        # 赎回费提取: 华夏银行无详情API，仅从产品名称解析
        if HAS_FEE_DB:
            for p in products:
                p_code = p.get('id', '')
                p_name = p.get('title', '')
                if p_code and not has_fee_data('华夏银行', p_code):
                    fee_info = parse_fee_from_name(p_name)
                    if fee_info:
                        update_fee_info('华夏银行', p_code, fee_info)
                    else:
                        update_fee_info('华夏银行', p_code, {
                            'has_redemption_fee': None,  # 华夏无API，标记为未知
                            'fee_schedule': [],
                            'fee_description': '',
                            'source': 'name_parse',
                        })
            save_fee_db()

        result['profiles'] = all_profiles
        result['products'] = processed
        result['success'] = True

        logger.info(f"[华夏] 抓取完成，共 {processed} 个有效产品")
        logger.info(f"[华夏] 跳过: 无路径 {skipped_no_path}, 无净值 {skipped_no_nav}")

        # 保存剩余
        if HAS_NAV_DB and products_for_db:
            try:
                stats = update_nav_database('华夏', products_for_db)
                saved_total += len(products_for_db)
                logger.info(f"[华夏] 数据库更新完成: 共 {saved_total} 个产品")
            except Exception as e:
                logger.warning(f"[华夏] 数据库更新失败: {e}")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[华夏] 抓取失败: {e}")
        import traceback
        traceback.print_exc()

    return result


def crawl_citic(full_history: bool = False, max_months: int = 0, progress_callback=None, manifest=None, stop_check=None) -> dict:
    """抓取中信银行

    Args:
        full_history: True=完整爬取(12个月历史), False=增量(1个月)
        max_months: >0时覆盖默认区间，例如24=请求24个月(2年)历史
        progress_callback: 可选回调 (current, total, message) 用于报告进度
        manifest: V13 delta sync manifest
        stop_check: callable, 返回True表示应停止
    """
    mode_str = f"{max_months}个月" if max_months > 0 else ("完整" if full_history else "增量")
    print("\n" + "="*70)
    print(f"              中信银行理财产品抓取 [{mode_str}模式]")
    print("="*70)

    result = {
        'bank': '中信银行',
        'success': False,
        'products': 0,
        'profiles': [],
        'error': None
    }

    try:
        # 获取前先检查停止
        if stop_check and stop_check():
            logger.info("[中信] 收到停止信号，跳过抓取")
            result['error'] = '用户停止'
            return result

        from bank_crawler import CITICCrawler

        crawler = CITICCrawler()

        # 获取产品列表（使用Selenium）
        products = crawler.get_product_list()

        # 获取产品列表后再检查停止（Selenium可能阻塞数分钟）
        if stop_check and stop_check():
            logger.info("[中信] 收到停止信号，跳过产品处理")
            result['error'] = '用户停止'
            return result
        if not products:
            result['error'] = "未获取到产品列表"
            return result

        # V13: Delta Sync — 跳过已最新的产品
        total_before = len(products)
        if manifest and not full_history and max_months == 0 and filter_products_by_manifest:
            bank_m = manifest.get('中信银行', {})
            products, skipped = filter_products_by_manifest(
                products, bank_m, code_field='product_code', fallback_field='product_code')
            if skipped > 0:
                logger.info(f"[中信] Delta Sync: {total_before} 总产品, "
                           f"跳过 {skipped} (已最新), 需更新 {len(products)}")
            if not products:
                logger.info(f"[中信] 所有产品已是最新，无需抓取")
                result['success'] = True
                return result

        logger.info(f"[中信] 需抓取 {len(products)} 个产品，{mode_str}模式开始获取历史净值...")

        # 获取每个产品的历史净值
        session = crawler._get_session()
        all_profiles = []
        products_for_db = []
        saved_total = 0
        total = len(products)
        BATCH_SIZE = 200

        for i, product in enumerate(products):
            if stop_check and stop_check():
                logger.info(f"[中信] 收到停止信号，已处理 {i}/{total}")
                break

            product_code = product.get('product_code', '')
            product_name = product.get('product_name', '')

            # 赎回费提取: 中信产品名常含 "(180天内收取赎回费)"
            if HAS_FEE_DB and not has_fee_data('中信银行', product_code):
                fee_info = crawler._extract_fee_info({}, product_name)
                if fee_info:
                    update_fee_info('中信银行', product_code, fee_info)
                else:
                    update_fee_info('中信银行', product_code, {
                        'has_redemption_fee': False,
                        'fee_schedule': [],
                        'fee_description': '',
                        'source': 'name_parse',
                    })

            # 获取历史净值
            nav_history = crawler.get_nav_history(product_code, session,
                                                   full_history=full_history,
                                                   max_months=max_months)

            # 计算指标
            profile = crawler.calculate_metrics(nav_history, product)
            if profile:
                all_profiles.append(profile)
                # 准备数据库数据
                db_nav = []
                if hasattr(profile, 'nav_history') and profile.nav_history:
                    for nav in profile.nav_history:
                        if isinstance(nav, dict) and nav.get('unit_nav'):
                            db_nav.append({
                                'date': nav.get('date', ''),
                                'unit_nav': nav.get('unit_nav')
                            })
                if db_nav:
                    products_for_db.append({
                        'product_code': profile.product_code,
                        'product_name': profile.product_name,
                        'nav_history': db_nav
                    })

            # 分批保存，防止中断丢失
            if HAS_NAV_DB and len(products_for_db) >= BATCH_SIZE:
                try:
                    stats = update_nav_database('中信', products_for_db)
                    saved_total += len(products_for_db)
                    logger.info(f"[中信] 分批保存: {saved_total} 个产品已入库")
                    products_for_db = []
                except Exception as e:
                    logger.warning(f"[中信] 分批保存失败: {e}")

            if (i + 1) % 10 == 0 or i + 1 == total:
                logger.info(f"[中信] 进度: {i+1}/{total}, 已处理: {len(all_profiles)}")
                if progress_callback:
                    progress_callback(i + 1, total, f'中信银行 {i+1}/{total} 产品')

        # 保存剩余
        if HAS_NAV_DB and products_for_db:
            try:
                stats = update_nav_database('中信', products_for_db)
                saved_total += len(products_for_db)
                logger.info(f"[中信] 数据库更新完成: 共 {saved_total} 个产品")
            except Exception as e:
                logger.warning(f"[中信] 数据库更新失败: {e}")

        # 保存赎回费数据库
        if HAS_FEE_DB:
            save_fee_db()

        # 赎回费: 从产品说明书提取（更准确）
        if HAS_FEE_DB:
            try:
                from crawl_citic_prospectus import crawl_all_prospectuses
                prospectus_stats = crawl_all_prospectuses(only_active=True)
                logger.info(f"[中信] 说明书费率: {prospectus_stats}")
            except ImportError:
                logger.debug("[中信] crawl_citic_prospectus 模块不可用，跳过说明书费率爬取")
            except Exception as e:
                logger.warning(f"[中信] 说明书费率爬取失败: {e}")

        result['profiles'] = all_profiles
        result['products'] = len(all_profiles)
        result['success'] = True

        logger.info(f"[中信] 抓取完成，共 {len(all_profiles)} 个有效产品, {saved_total} 个已入库")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[中信] 抓取失败: {e}")

    return result


def print_summary(results: list, elapsed: float, full_mode: bool):
    """打印汇总报告"""
    mode_str = "完整爬取" if full_mode else "增量更新"
    print("\n" + "="*70)
    print(f"                 抓取汇总报告 [{mode_str}]")
    print("="*70)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed:.0f} 秒")
    print()

    print(f"{'银行':<12} {'状态':<10} {'产品数':<10} {'备注'}")
    print("-" * 60)

    total_products = 0
    for r in results:
        status = "成功" if r['success'] else "失败"
        note = r['error'] if r['error'] else ""
        print(f"{r['bank']:<12} {status:<10} {r['products']:<10} {note}")
        total_products += r['products']

    print("-" * 60)
    print(f"{'合计':<12} {'':<10} {total_products:<10}")

    # 显示数据库统计（更新后）
    if HAS_NAV_DB:
        try:
            db = NAVDatabaseExcel()
            stats = db.get_stats()
            print("\n净值数据库统计（更新后）:")
            print(f"  {'银行':<12} {'产品数':<10} {'日期数':<10} {'最早日期':<14} {'最新日期':<14}")
            print(f"  {'-'*58}")
            for sheet, info in stats.items():
                earliest = info.get('earliest_date', 'N/A') or 'N/A'
                latest = info.get('latest_date', 'N/A') or 'N/A'
                print(f"  {sheet:<12} {info['products']:<10} {info['dates']:<10} {earliest:<14} {latest:<14}")
        except:
            pass

    print()
    print("  说明: 净值数据库采用增量追加策略，已有数据不会被删除。")
    print("  如需获取更长历史数据，请使用 --full 参数运行。")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='三银行理财产品统一抓取程序')
    parser.add_argument('--minsheng', '-m', action='store_true', help='抓取民生银行')
    parser.add_argument('--huaxia', '-x', action='store_true', help='抓取华夏银行')
    parser.add_argument('--citic', '-c', action='store_true', help='抓取中信银行')
    parser.add_argument('--full', '-f', action='store_true',
                        help='完整爬取模式（首次运行推荐，获取最大历史深度）')

    args = parser.parse_args()

    # 如果没有指定任何银行，则抓取全部
    crawl_all = not (args.minsheng or args.huaxia or args.citic)
    full_mode = args.full

    mode_str = "完整爬取" if full_mode else "增量更新"

    print("="*70)
    print("           三银行理财产品统一抓取程序")
    print("="*70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {mode_str}")

    if full_mode:
        print("  民生: 请求3年历史 (pageSize=1125)")
        print("  中信: 请求12个月历史 (queryUnit=12)")
        print("  华夏: 保留全部历史 (NAV_DAYS=9999)")
    else:
        print("  民生: 请求180天历史 (增量)")
        print("  中信: 请求1个月历史 (增量)")
        print("  华夏: 增量追加新日期")

    if crawl_all:
        print(f"抓取范围: 全部（民生、华夏、中信）")
    else:
        banks = []
        if args.minsheng: banks.append("民生")
        if args.huaxia: banks.append("华夏")
        if args.citic: banks.append("中信")
        print(f"抓取范围: {', '.join(banks)}")

    print(f"数据库文件: 净值数据库.xlsx")
    print(f"数据保护: 已有净值数据不会被删除，仅追加新数据")

    # 显示当前数据库状态
    show_database_status()

    print()
    start_time = time.time()
    results = []

    # 依次抓取
    if crawl_all or args.minsheng:
        results.append(crawl_minsheng(full_history=full_mode))

    if crawl_all or args.huaxia:
        results.append(crawl_huaxia(full_history=full_mode))

    if crawl_all or args.citic:
        results.append(crawl_citic(full_history=full_mode))

    elapsed = time.time() - start_time

    # 打印汇总
    print_summary(results, elapsed, full_mode)

    return 0 if all(r['success'] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
