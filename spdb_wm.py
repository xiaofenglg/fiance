# -*- coding: utf-8 -*-
"""
浦银理财产品数据采集与量化分析系统
网站: https://www.spdb-wm.com/financialProducts/

功能:
1. 全维度产品画像采集（申购状态、持有期限、风险等级等）
2. 量化指标计算（近1周/1月/3月/6月年化、最大回撤、夏普比率、卡玛比率）
3. 多因子评分模型（固收类/权益类分别评分）
4. T+0 动态信号生成系统
5. 核心-卫星投资组合构建
6. 支持2年(730天)完整净值历史获取

参照: minsheng.py 结构
版本: 1.0
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
import logging
import json
import ssl
import urllib3
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 导入净值数据库Excel管理模块
try:
    from nav_db_excel import update_nav_database
    HAS_NAV_DB = True
except ImportError:
    HAS_NAV_DB = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spdb_wm_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型定义
# ============================================================================

class ProductStatus(Enum):
    """产品申购状态"""
    OPEN = "开放"
    CLOSED = "封闭"
    SUSPENDED = "暂停"
    UNKNOWN = "未知"


class ProductType(Enum):
    """产品类型"""
    HOLDING_PERIOD = "持有期"
    SCHEDULED_OPEN = "定开"
    CLOSED_END = "封闭式"
    CASH_MGMT = "现金管理"
    OTHER = "其他"


class RiskCategory(Enum):
    """风险分类"""
    FIXED_INCOME = "固收类"  # R1-R3
    EQUITY = "权益类"  # R4-R5


class SignalType(Enum):
    """交易信号类型"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    WATCH = "观望"
    AVOID = "回避"


@dataclass
class QuantMetrics:
    """量化指标"""
    return_1w: Optional[float] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_6m: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    excess_return_1m: Optional[float] = None


@dataclass
class ProductProfile:
    """产品全维度画像"""
    product_code: str = ""
    product_name: str = ""
    risk_level: str = ""
    risk_category: RiskCategory = RiskCategory.FIXED_INCOME
    product_type: ProductType = ProductType.OTHER
    status: ProductStatus = ProductStatus.UNKNOWN
    status_text: str = ""
    is_buyable: bool = False
    duration_days: Optional[int] = None
    duration_text: str = ""
    benchmark: str = ""
    benchmark_value: Optional[float] = None
    latest_nav: float = 1.0
    latest_nav_date: str = ""
    days_from_today: int = 0
    metrics: QuantMetrics = field(default_factory=QuantMetrics)
    score: float = 0.0
    signal: SignalType = SignalType.WATCH
    nav_history: List[Dict] = field(default_factory=list)


# ============================================================================
# 配置参数
# ============================================================================

class Config:
    """系统配置"""
    RISK_FREE_RATE = 1.5
    FIXED_INCOME_WEIGHTS = {
        'return_3m': 0.6,
        'return_1w': 0.3,
        'volatility': -0.1
    }
    EQUITY_WEIGHTS = {
        'sharpe_ratio': 0.5,
        'calmar_ratio': 0.3,
        'excess_return_1m': 0.2
    }
    CORE_POSITION_RATIO = (0.6, 0.8)
    SATELLITE_POSITION_RATIO = (0.2, 0.4)
    STOP_LOSS_THRESHOLD = 0.95


# ============================================================================
# SSL适配器
# ============================================================================

class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器，支持旧版SSL协商

    注意: urllib3 2.x + requests 组合中，requests 的 verify 参数会导致
    SSL context 被重建。必须同时覆写 send() 和 proxy_manager_for() 来保留我们的 context。
    """

    def __init__(self, *args, **kwargs):
        self.ssl_context = create_urllib3_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        self.ssl_context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs['ssl_context'] = self.ssl_context
        return super().init_poolmanager(connections, maxsize, block, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs['ssl_context'] = self.ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        return super().send(request, stream=stream, timeout=timeout,
                            verify=False, cert=cert, proxies=proxies)


# ============================================================================
# 量化计算引擎
# ============================================================================

class QuantEngine:
    """量化计算引擎"""

    @staticmethod
    def calculate_annualized_return(nav_series: List[float], days: int) -> Optional[float]:
        """计算年化收益率 (nav_series: 从新到旧)"""
        if len(nav_series) < 2 or days <= 0:
            return None
        try:
            start_nav = nav_series[-1]
            end_nav = nav_series[0]
            if start_nav <= 0:
                return None
            period_return = (end_nav - start_nav) / start_nav
            annualized = period_return * 365 / days * 100
            return round(annualized, 2)
        except:
            return None

    @staticmethod
    def calculate_volatility(daily_returns: List[float]) -> Optional[float]:
        """计算年化波动率"""
        if len(daily_returns) < 5:
            return None
        try:
            std = np.std(daily_returns, ddof=1)
            annualized_vol = std * np.sqrt(252)
            return round(annualized_vol, 2)
        except:
            return None

    @staticmethod
    def calculate_max_drawdown(nav_series: List[float]) -> Optional[float]:
        """计算最大回撤 (nav_series: 从新到旧)"""
        if len(nav_series) < 2:
            return None
        try:
            navs = list(reversed(nav_series))
            peak = navs[0]
            max_dd = 0
            for nav in navs:
                if nav > peak:
                    peak = nav
                dd = (peak - nav) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            return round(max_dd, 2)
        except:
            return None

    @staticmethod
    def calculate_sharpe_ratio(annualized_return: Optional[float],
                               volatility: Optional[float],
                               risk_free_rate: float = Config.RISK_FREE_RATE) -> Optional[float]:
        if annualized_return is None or volatility is None or volatility == 0:
            return None
        try:
            sharpe = (annualized_return - risk_free_rate) / volatility
            return round(sharpe, 2)
        except:
            return None

    @staticmethod
    def calculate_calmar_ratio(annualized_return: Optional[float],
                               max_drawdown: Optional[float]) -> Optional[float]:
        if annualized_return is None or max_drawdown is None or max_drawdown == 0:
            return None
        try:
            calmar = annualized_return / max_drawdown
            return round(calmar, 2)
        except:
            return None


# ============================================================================
# 主爬虫类
# ============================================================================

class SPDBWealthCrawler:
    """浦银理财产品爬虫 - 专业量化版"""

    SESSION_NAV_LIMIT = 150  # 会话级NAV查询限制

    BASE_URL = "https://www.spdb-wm.com"

    # 发现的统一API端点: POST /api/search, 不同 chlid 区分数据类型
    # chlid=1002: 产品列表  (字段: PRDC_CD, PRDC_NM, RISK_GRADE, SLL_OBJC, PRDC_STT, ...)
    # chlid=1003: 净值历史  (字段: REAL_PRD_CODE, ISS_DATE, NAV, TOT_NAV)
    # chlid=1006: 最新净值  (字段: REAL_PRD_CODE, ISS_DATE, NAV, TOT_NAV)
    API_SEARCH = "/api/search"

    def __init__(self):
        """初始化爬虫"""
        self.session = self._create_session()
        self.quant = QuantEngine()
        self.historical_scores: Dict[str, List[float]] = {}
        self._nav_query_count = 0

    def _create_session(self) -> requests.Session:
        """创建新的HTTP会话"""
        session = requests.Session()
        adapter = LegacySSLAdapter(pool_connections=30, pool_maxsize=30)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Origin': 'https://www.spdb-wm.com',
            'Referer': 'https://www.spdb-wm.com/financialProducts/',
        })
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return session

    def _rotate_session_if_needed(self):
        """检查是否需要轮换会话"""
        self._nav_query_count += 1
        if self._nav_query_count >= self.SESSION_NAV_LIMIT:
            logger.debug(f"已达到 {self.SESSION_NAV_LIMIT} 次NAV查询，轮换会话...")
            self.session.close()
            self.session = self._create_session()
            self._nav_query_count = 0
            time.sleep(0.5)

    def _api_search(self, chlid: int, searchword: str = "",
                    page: int = 1, size: int = 99999,
                    session: requests.Session = None,
                    max_retries: int = 3,
                    timeout: int = None) -> Optional[Dict]:
        """统一API调用: POST /api/search

        Args:
            chlid: 频道ID (1002=产品列表, 1003=净值历史, 1006=最新净值)
            searchword: 搜索条件 (如 "(REAL_PRD_CODE = '2501251842')")
            page: 页码
            size: 每页大小
            session: 可选指定会话
            timeout: 请求超时秒数，默认根据size自动调整
        """
        use_session = session or self.session
        url = f"{self.BASE_URL}{self.API_SEARCH}"
        payload = {
            "chlid": chlid,
            "cutsize": 150,
            "dynexpr": [],
            "dynidx": 1,
            "extopt": [],
            "orderby": "",
            "page": page,
            "size": size,
            "searchword": searchword,
        }

        # 大请求自动增加超时（净值查询用较短超时避免SSL挂起浪费时间）
        if timeout is None:
            timeout = 60 if size > 5000 else 30 if size > 500 else 8

        for retry in range(max_retries):
            try:
                resp = use_session.post(url, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("data", {})
                elif resp.status_code == 429:
                    # 限流：等待较长时间
                    wait = 10 * (retry + 1)
                    logger.warning(f"API限流(429), 等待{wait}秒...")
                    time.sleep(wait)
                else:
                    logger.warning(f"API返回状态码 {resp.status_code} (chlid={chlid}, retry {retry+1}/{max_retries})")
            except Exception as e:
                logger.warning(f"API请求异常 (chlid={chlid}, retry {retry+1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    wait = 3 * (retry + 1)
                    time.sleep(wait)
                    # 连接失败时重建session
                    if 'SSL' in str(e) or 'Connection' in str(e) or 'Timeout' in str(e):
                        use_session = self._create_session()

        return None

    # ========================================================================
    # 数据采集 - 产品列表
    # ========================================================================

    def get_product_list(self, max_retries: int = 3, stop_check=None) -> List[Dict]:
        """获取个人理财产品列表（通过 /api/search chlid=1002，分页获取）"""
        logger.info("开始获取浦银理财个人产品列表...")

        # 先尝试一次获取全部，失败则分页
        result = self._api_search(chlid=1002, size=99999, max_retries=max_retries)
        if result and result.get("content"):
            raw_items = result["content"]
            total = result.get("totalElements", len(raw_items))
        else:
            # 轮换会话后分页获取
            logger.info("全量请求失败，轮换会话后改用分页获取...")
            self.session.close()
            self.session = self._create_session()
            time.sleep(2)

            raw_items = []
            page = 1
            page_size = 500
            consecutive_empty = 0
            while True:
                if stop_check and stop_check():
                    logger.info(f"[浦银] 收到停止信号，已获取 {len(raw_items)} 个产品")
                    break
                result = self._api_search(
                    chlid=1002, page=page, size=page_size,
                    max_retries=max_retries
                )
                if not result or not result.get("content"):
                    consecutive_empty += 1
                    if consecutive_empty >= 2 or page == 1:
                        # 第1页就失败或连续2次空响应，再尝试一次新会话
                        if page == 1 and consecutive_empty == 1:
                            logger.warning("分页首页也失败，重建会话再试...")
                            self.session.close()
                            self.session = self._create_session()
                            time.sleep(5)
                            continue
                        break
                    break
                consecutive_empty = 0
                items = result["content"]
                raw_items.extend(items)
                total_elements = result.get("totalElements", 0)
                logger.info(f"分页 {page}: 获取 {len(items)} 条, "
                            f"累计 {len(raw_items)}/{total_elements}")
                if len(raw_items) >= total_elements or len(items) < page_size:
                    break
                page += 1
                time.sleep(0.5)

            if not raw_items:
                logger.error("API未返回产品数据")
                return []
            total = len(raw_items)
        logger.info(f"API返回 {total} 条原始数据")

        # 过滤: 仅保留个人理财（对私）+ 募集/存续状态
        all_products = []
        seen_codes = set()

        for item in raw_items:
            sll_objc = item.get("SLL_OBJC", "")
            prdc_stt = item.get("PRDC_STT", "")

            # 只要个人理财产品（对私 或 对公,对私）
            if "对私" not in sll_objc:
                continue

            # 只要募集或存续状态
            if "募集" not in prdc_stt and "存续" not in prdc_stt:
                continue

            # 过滤份额为0的产品
            shr = item.get("SHR", "")
            if shr and shr != "0" and shr != "0E-8":
                pass
            else:
                continue

            # 系列产品去重逻辑：保留子产品，跳过有子产品的父产品
            prdc_cd = item.get("PRDC_CD", "")
            fml_prdc_cd = item.get("FML_PRDC_CD", "")
            fml_idnt = item.get("FML_PRDC_IDNT", "")
            fml_idnt_2 = item.get("FML_PRDC_IDNT_2", "")

            if fml_idnt == "是" and fml_idnt_2 != "N":
                continue  # 有子产品的父级，跳过

            if prdc_cd in seen_codes:
                continue
            seen_codes.add(prdc_cd)

            product = self._normalize_product(item)
            if product:
                all_products.append(product)

        logger.info(f"过滤后获取到 {len(all_products)} 个个人理财产品")
        return all_products

    def _normalize_product(self, item: Dict) -> Optional[Dict]:
        """标准化浦银理财产品数据结构"""
        if not isinstance(item, dict):
            return None

        prdc_cd = item.get("PRDC_CD", "")
        if not prdc_cd:
            return None

        prdc_nm = item.get("PRDC_NM", "")
        risk_grade = item.get("RISK_GRADE", "")
        prdc_stt = item.get("PRDC_STT", "")
        prdc_typ = item.get("PRDC_TYP", "")
        prdc_frm = item.get("PRDC_FRM", "")
        rs_mthd = item.get("RS_MTHD", "")
        term_type = item.get("TERM_TYPE", "")
        sll_objc = item.get("SLL_OBJC", "")

        # 风险等级标准化: "低风险" -> "R1", "较低风险" -> "R2" ...
        risk_map = {
            "低风险": "R1", "较低风险": "R2", "中等风险": "R3",
            "较高风险": "R4", "高风险": "R5",
            "中低风险": "R2", "中风险": "R3", "中高风险": "R4",
        }
        risk_level = risk_map.get(risk_grade, risk_grade)

        return {
            'product_code': prdc_cd,
            'product_name': prdc_nm,
            'risk_level': risk_level,
            'risk_grade_text': risk_grade,
            'status_text': prdc_stt,
            'product_type_text': prdc_typ,
            'product_form': prdc_frm,
            'term_type': term_type,
            'rs_method': rs_mthd,
            'sll_objc': sll_objc,
            'benchmark': '',
            '_raw': item,
        }

    def _extract_fee_info(self, raw_item: Dict) -> Optional[Dict]:
        """从产品原始数据中提取赎回费信息

        Args:
            raw_item: _normalize_product 返回的dict (含 _raw 字段)

        Returns:
            dict: ProductFeeInfo 或 None
        """
        from redemption_fee_db import parse_fee_from_name

        raw = raw_item.get('_raw', raw_item) if isinstance(raw_item, dict) else {}

        # 搜索 _raw 中的费用相关字段
        for fee_key in ['SH_FEE', 'REDEEM_FEE', 'SELL_FEE', 'FEE_RATE',
                        'BACK_FEE', 'BUY_FEE', 'SH_FEE_RATE']:
            fee_val = raw.get(fee_key)
            if fee_val is not None and fee_val != '' and fee_val != 0:
                try:
                    rate = float(fee_val)
                    if rate > 1:
                        rate = rate / 100
                    if rate > 0:
                        return {
                            'has_redemption_fee': True,
                            'fee_schedule': [
                                {'min_days': 0, 'max_days': 7, 'fee_rate': rate},
                                {'min_days': 7, 'max_days': 999999, 'fee_rate': 0.0},
                            ],
                            'fee_description': f'API费率{rate*100:.2f}%',
                            'source': 'api_detail',
                        }
                except (ValueError, TypeError):
                    pass

        # 搜索描述字段中的赎回费信息
        for desc_key in ['PRDC_NM', 'PRDC_DESC', 'PRDC_FRM', 'FEE_DESC']:
            desc_val = raw.get(desc_key, '')
            if isinstance(desc_val, str) and '赎回费' in desc_val:
                parsed = parse_fee_from_name(desc_val)
                if parsed:
                    parsed['source'] = 'api_detail'
                    return parsed

        # Fallback: 从产品名称解析
        name = raw_item.get('product_name', '') or raw.get('PRDC_NM', '')
        if name:
            return parse_fee_from_name(name)

        return None

    # ========================================================================
    # 数据采集 - 净值历史
    # ========================================================================

    def get_nav_history(self, product_code: str, days: int = 180,
                        max_retries: int = 3,
                        session: requests.Session = None,
                        full_history: bool = False) -> List[Dict]:
        """
        获取产品历史净值（通过 /api/search chlid=1003）

        Args:
            product_code: 产品代码 (PRDC_CD)
            days: 增量模式请求天数（默认180天量化分析用）
            max_retries: 最大重试次数
            session: 可选的指定会话（批量处理时使用）
            full_history: True=请求全部数据(size=99999), False=使用days参数
        """
        use_session = session or self.session
        searchword = f"(REAL_PRD_CODE = '{product_code}')"
        size = 99999 if full_history else min(days + 30, 99999)

        result = self._api_search(
            chlid=1003,
            searchword=searchword,
            page=1,
            size=size,
            session=use_session,
            max_retries=max_retries,
        )

        if not result or not result.get("content"):
            return []

        nav_list = []
        for item in result["content"]:
            iss_date = item.get("ISS_DATE", "")
            nav_val = item.get("NAV")
            tot_nav = item.get("TOT_NAV")

            if not iss_date or nav_val is None:
                continue

            # NAV值清洗 (处理 "0E-8" 等特殊值)
            try:
                nav_float = float(nav_val)
            except (ValueError, TypeError):
                continue

            if nav_float == 0:
                continue

            try:
                tot_float = float(tot_nav) if tot_nav and tot_nav != "0E-8" else nav_float
            except (ValueError, TypeError):
                tot_float = nav_float

            # 日期格式: YYYYMMDD -> YYYY-MM-DD
            date_clean = str(iss_date).strip()
            if len(date_clean) == 8 and date_clean.isdigit():
                date_formatted = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:8]}"
            elif len(date_clean) == 10 and '-' in date_clean:
                date_formatted = date_clean
            else:
                continue

            nav_list.append({
                'date': date_formatted,
                'unit_nav': nav_float,
                'total_nav': tot_float,
            })

        return nav_list

    # ========================================================================
    # 数据解析
    # ========================================================================

    def parse_date(self, date_str) -> Optional[datetime]:
        """解析日期字符串"""
        if date_str is None:
            return None
        date_str = str(date_str).strip()
        date_str = ''.join(filter(str.isdigit, date_str))
        if len(date_str) == 8:
            try:
                return datetime.strptime(date_str, "%Y%m%d")
            except:
                return None
        return None

    def format_date(self, date_str, full: bool = True) -> str:
        """格式化日期"""
        if date_str is None:
            return ""
        date_str = str(date_str).strip()

        # 已经是标准格式
        if len(date_str) == 10 and '-' in date_str:
            return date_str if full else date_str[5:]

        date_str = ''.join(filter(str.isdigit, date_str))
        if len(date_str) == 8:
            if full:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                return f"{date_str[4:6]}-{date_str[6:8]}"
        return str(date_str)

    def parse_risk_category(self, risk_level: str) -> RiskCategory:
        """解析风险类别"""
        if not risk_level:
            return RiskCategory.FIXED_INCOME
        risk_str = str(risk_level).upper()
        if 'R4' in risk_str or 'R5' in risk_str or '4' in risk_str or '5' in risk_str:
            return RiskCategory.EQUITY
        return RiskCategory.FIXED_INCOME

    def parse_status(self, product_data: Dict) -> Tuple[ProductStatus, str]:
        """解析申购状态"""
        status_text = product_data.get('status_text', '')
        name = product_data.get('product_name', '')

        if '封闭' in name and '持有' not in name:
            return ProductStatus.CLOSED, "封闭期产品"

        if any(kw in status_text for kw in ['可购买', '开放', '在售', '申购中']):
            return ProductStatus.OPEN, "开放申购"

        raw = product_data.get('_raw', {})
        status_val = raw.get('status') or raw.get('saleStatus') or raw.get('productStatus')
        if status_val is not None:
            s = str(status_val).strip()
            if s in ['0', '1', '4', 'open', 'OPEN', 'ON_SALE']:
                return ProductStatus.OPEN, "开放申购"
            if s in ['2', '3', 'closed', 'CLOSED', 'SUSPEND']:
                return ProductStatus.SUSPENDED, "暂停申购"

        # 默认视为开放（浦银理财产品大多数个人产品是开放的）
        return ProductStatus.OPEN, "默认开放"

    def parse_product_type(self, product_name: str) -> Tuple[ProductType, bool]:
        """解析产品类型，判断是否可购买"""
        name = product_name

        if '持有' in name or re.search(r'\d+天', name):
            if '封闭' not in name:
                return ProductType.HOLDING_PERIOD, True

        if any(kw in name for kw in ['周周盈', '天天盈', '日日盈', '现金', '货币', '活钱']):
            return ProductType.CASH_MGMT, True

        if any(kw in name for kw in ['定开', '季开', '月开', '年开', '季季盈', '月月盈']):
            return ProductType.SCHEDULED_OPEN, False

        if '封闭' in name:
            return ProductType.CLOSED_END, False

        return ProductType.OTHER, True

    def parse_duration(self, product_name: str) -> Tuple[Optional[int], str]:
        """从产品名称解析持有期限"""
        patterns = [
            (r'(\d+)\s*天', lambda m: int(m.group(1))),
            (r'(\d+)\s*日', lambda m: int(m.group(1))),
            (r'(\d+)\s*个月', lambda m: int(m.group(1)) * 30),
            (r'半年', lambda m: 180),
            (r'一年', lambda m: 365),
            (r'季度|季季盈|季盈', lambda m: 90),
            (r'月月盈|月盈', lambda m: 30),
            (r'周周盈|周盈', lambda m: 7),
        ]

        for pattern, extractor in patterns:
            match = re.search(pattern, product_name)
            if match:
                try:
                    days = extractor(match)
                    return days, match.group(0)
                except:
                    continue

        return None, "未知"

    def parse_benchmark(self, benchmark_str: str) -> Optional[float]:
        """解析业绩比较基准"""
        if not benchmark_str:
            return None
        match = re.search(r'(\d+\.?\d*)\s*%', benchmark_str)
        if match:
            return float(match.group(1))
        match = re.search(r'(\d+\.?\d*)\s*[~～]\s*(\d+\.?\d*)', benchmark_str)
        if match:
            return (float(match.group(1)) + float(match.group(2))) / 2
        return None

    # ========================================================================
    # 量化指标计算
    # ========================================================================

    def calculate_metrics(self, nav_data: List[Dict], product_data: Dict) -> Optional[ProductProfile]:
        """计算全部量化指标"""
        if not nav_data or len(nav_data) < 2:
            return None

        # 按日期降序排列（最新在前）
        nav_sorted = sorted(nav_data, key=lambda x: str(x.get('date', '')), reverse=True)

        profile = ProductProfile()
        profile.product_code = product_data.get('product_code', '')
        profile.product_name = product_data.get('product_name', '')
        profile.risk_level = product_data.get('risk_level', '')
        profile.risk_category = self.parse_risk_category(profile.risk_level)
        profile.benchmark = product_data.get('benchmark', '')
        profile.benchmark_value = self.parse_benchmark(profile.benchmark)

        # 产品类型
        profile.product_type, type_buyable = self.parse_product_type(profile.product_name)

        # 状态
        profile.status, profile.status_text = self.parse_status(product_data)

        # 期限
        profile.duration_days, profile.duration_text = self.parse_duration(profile.product_name)

        # 可购买判断
        profile.is_buyable = (profile.status == ProductStatus.OPEN) and type_buyable

        # 提取净值和日期序列
        navs = []
        dates = []
        for item in nav_sorted:
            nav_val = item.get('unit_nav')
            date_str = item.get('date', '')
            if nav_val is not None and date_str:
                try:
                    navs.append(float(nav_val))
                    dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                    dates.append(dt)
                except:
                    continue

        if len(navs) < 2:
            return None

        # 最新净值
        profile.latest_nav = navs[0]
        profile.latest_nav_date = nav_sorted[0].get('date', '')[:10]
        profile.days_from_today = (datetime.now() - dates[0]).days if dates else 9999

        # 保留完整净值历史
        profile.nav_history = nav_sorted

        # 计算各期限收益率
        today = datetime.now()

        week_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 7]
        if len(week_navs) >= 2:
            profile.metrics.return_1w = self.quant.calculate_annualized_return(week_navs, min(7, len(week_navs)))

        month_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 30]
        if len(month_navs) >= 2:
            profile.metrics.return_1m = self.quant.calculate_annualized_return(month_navs, min(30, len(month_navs)))

        quarter_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 90]
        if len(quarter_navs) >= 2:
            profile.metrics.return_3m = self.quant.calculate_annualized_return(quarter_navs, min(90, len(quarter_navs)))

        half_year_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 180]
        if len(half_year_navs) >= 2:
            profile.metrics.return_6m = self.quant.calculate_annualized_return(half_year_navs, min(180, len(half_year_navs)))

        # 日收益率（波动率计算用）
        daily_returns = []
        for i in range(len(navs) - 1):
            if navs[i + 1] != 0:
                ret = (navs[i] - navs[i + 1]) / navs[i + 1] * 100
                daily_returns.append(ret)

        profile.metrics.volatility = self.quant.calculate_volatility(daily_returns)
        profile.metrics.max_drawdown = self.quant.calculate_max_drawdown(
            half_year_navs if half_year_navs else navs)

        # 风险调整收益
        return_for_sharpe = profile.metrics.return_3m or profile.metrics.return_1m
        profile.metrics.sharpe_ratio = self.quant.calculate_sharpe_ratio(
            return_for_sharpe, profile.metrics.volatility)
        profile.metrics.calmar_ratio = self.quant.calculate_calmar_ratio(
            return_for_sharpe, profile.metrics.max_drawdown)

        if profile.metrics.return_1m is not None:
            profile.metrics.excess_return_1m = profile.metrics.return_1m - Config.RISK_FREE_RATE

        return profile

    # ========================================================================
    # 多因子评分
    # ========================================================================

    def calculate_score(self, profile: ProductProfile) -> float:
        """多因子评分"""
        m = profile.metrics

        if profile.risk_category == RiskCategory.FIXED_INCOME:
            score = 0.0
            w = Config.FIXED_INCOME_WEIGHTS
            if m.return_3m is not None:
                score += w['return_3m'] * m.return_3m
            elif m.return_1m is not None:
                score += w['return_3m'] * m.return_1m
            if m.return_1w is not None:
                score += w['return_1w'] * m.return_1w
            if m.volatility is not None:
                score += w['volatility'] * m.volatility
        else:
            score = 0.0
            w = Config.EQUITY_WEIGHTS
            if m.sharpe_ratio is not None:
                score += w['sharpe_ratio'] * max(-5, min(5, m.sharpe_ratio)) * 10
            if m.calmar_ratio is not None:
                score += w['calmar_ratio'] * max(-5, min(5, m.calmar_ratio)) * 10
            if m.excess_return_1m is not None:
                score += w['excess_return_1m'] * m.excess_return_1m

        return round(score, 2)

    def generate_signal(self, profile: ProductProfile) -> SignalType:
        """生成交易信号"""
        if not profile.is_buyable:
            return SignalType.AVOID
        if profile.latest_nav < Config.STOP_LOSS_THRESHOLD:
            return SignalType.AVOID
        if profile.days_from_today > 7:
            return SignalType.WATCH

        score = profile.score

        if profile.risk_category == RiskCategory.FIXED_INCOME:
            if score >= 3.0:
                return SignalType.STRONG_BUY
            elif score >= 2.0:
                return SignalType.BUY
            elif score >= 1.0:
                return SignalType.HOLD
            elif score >= 0:
                return SignalType.WATCH
            else:
                return SignalType.AVOID
        else:
            if score >= 4.0:
                return SignalType.STRONG_BUY
            elif score >= 2.5:
                return SignalType.BUY
            elif score >= 1.0:
                return SignalType.HOLD
            elif score >= 0:
                return SignalType.WATCH
            else:
                return SignalType.AVOID

    # ========================================================================
    # 投资组合构建
    # ========================================================================

    def build_portfolio(self, profiles: List[ProductProfile]) -> Dict:
        """核心-卫星投资组合"""
        buyable = [p for p in profiles if p.is_buyable]
        fixed_income = [p for p in buyable if p.risk_category == RiskCategory.FIXED_INCOME]
        equity = [p for p in buyable if p.risk_category == RiskCategory.EQUITY]

        # 核心仓位：优先持有期产品
        fixed_income.sort(key=lambda x: (
            0 if x.product_type == ProductType.HOLDING_PERIOD else
            1 if x.product_type == ProductType.CASH_MGMT else 2,
            -x.score
        ))

        core_recs = []
        for p in fixed_income[:5]:
            if p.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                core_recs.append({
                    'code': p.product_code,
                    'name': p.product_name,
                    'risk': p.risk_level,
                    'product_type': p.product_type.value,
                    'score': p.score,
                    'signal': p.signal.value,
                    'return_3m': p.metrics.return_3m,
                    'volatility': p.metrics.volatility,
                    'duration': p.duration_text,
                })

        # 卫星仓位
        equity.sort(key=lambda x: x.score, reverse=True)
        satellite_recs = []
        for p in equity[:3]:
            if p.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                satellite_recs.append({
                    'code': p.product_code,
                    'name': p.product_name,
                    'risk': p.risk_level,
                    'score': p.score,
                    'signal': p.signal.value,
                    'sharpe': p.metrics.sharpe_ratio,
                    'calmar': p.metrics.calmar_ratio,
                    'max_drawdown': p.metrics.max_drawdown,
                })

        return {
            'core': {
                'description': '核心仓位 - 固收类 (建议占比60%-80%)',
                'target_return': '3%-4%年化',
                'recommendations': core_recs,
            },
            'satellite': {
                'description': '卫星仓位 - 权益类 (建议占比20%-40%)',
                'target_return': '波动较大，可能>10%也可能亏损',
                'stop_loss': f'建议止损线: 净值跌破{Config.STOP_LOSS_THRESHOLD}即赎回',
                'recommendations': satellite_recs,
            },
        }

    # ========================================================================
    # 批次处理
    # ========================================================================

    def process_single_product(self, product: Dict) -> Tuple[Optional[ProductProfile], str]:
        """处理单个产品"""
        code = product.get('product_code', '')
        name = product.get('product_name', '')[:30]

        if not code:
            return None, f"无产品代码: {name}"

        nav_data = self.get_nav_history(code, days=180)
        if not nav_data:
            return None, f"无NAV数据: {name}"

        profile = self.calculate_metrics(nav_data, product)
        if not profile:
            return None, f"NAV数据不足: {name} (仅{len(nav_data)}条)"

        profile.score = self.calculate_score(profile)
        profile.signal = self.generate_signal(profile)
        return profile, "成功"

    def _process_batch(self, products_batch: List[Dict], batch_num: int,
                       full_history: bool = False) -> List[Tuple[Optional[ProductProfile], str]]:
        """处理一批产品（独立会话）"""
        session = self._create_session()
        results = []

        for i, product in enumerate(products_batch):
            code = product.get('product_code', '')
            name = product.get('product_name', '')[:30]

            if not code:
                results.append((None, f"无产品代码: {name}"))
                continue

            nav_data = self.get_nav_history(
                code, days=180, session=session, full_history=full_history)
            if not nav_data:
                results.append((None, f"无NAV数据: {name}"))
                continue

            profile = self.calculate_metrics(nav_data, product)
            if not profile:
                results.append((None, f"NAV数据不足: {name} (仅{len(nav_data)}条)"))
                continue

            profile.score = self.calculate_score(profile)
            profile.signal = self.generate_signal(profile)
            results.append((profile, "成功"))

            if (i + 1) % 50 == 0:
                time.sleep(0.2)

        session.close()
        return results

    # ========================================================================
    # 主流程
    # ========================================================================

    def crawl_all_products(self, max_workers: int = 10, limit: int = None) -> Optional[str]:
        """爬取并分析所有产品"""
        logger.info("=" * 70)
        logger.info("浦银理财产品量化分析系统 v1.0")
        logger.info("=" * 70)

        start_time = time.time()

        # 1. 获取产品列表
        products = self.get_product_list()
        if not products:
            logger.error("未获取到产品列表")
            return None

        if limit:
            products = products[:limit]
            logger.info(f"测试模式: 处理前 {limit} 个产品")

        # 2. 分批处理
        batch_size = self.SESSION_NAV_LIMIT
        batches = [products[i:i + batch_size] for i in range(0, len(products), batch_size)]

        logger.info(f"开始分析 {len(products)} 个产品，分 {len(batches)} 批 (每批{batch_size}个)...")

        all_profiles: List[ProductProfile] = []
        skip_reasons: Dict[str, int] = {}
        failed = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
            futures = {executor.submit(self._process_batch, batch, i): i
                       for i, batch in enumerate(batches)}

            for future in as_completed(futures):
                batch_num = futures[future]
                try:
                    batch_results = future.result()
                    for profile, reason in batch_results:
                        if profile:
                            all_profiles.append(profile)
                        else:
                            reason_key = reason.split(':')[0] if ':' in reason else reason
                            skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1

                    logger.info(f"批次 {batch_num + 1}/{len(batches)} 完成，累计成功: {len(all_profiles)}")
                except Exception as e:
                    logger.error(f"批次 {batch_num + 1} 处理失败: {e}")
                    failed += batch_size

        elapsed = time.time() - start_time
        logger.info(f"分析完成，耗时: {elapsed:.1f}秒")
        logger.info(f"成功: {len(all_profiles)}, 跳过: {len(products) - len(all_profiles) - failed}, 失败: {failed}")

        if skip_reasons:
            logger.info("跳过产品统计:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  - {reason}: {count}个")

        if not all_profiles:
            logger.warning("未获取到有效数据")
            return None

        # 3. 排序
        all_profiles.sort(key=lambda x: (x.days_from_today, -x.score))

        # 4. 投资组合
        portfolio = self.build_portfolio(all_profiles)

        # 5. 保存
        filename = self.save_results(all_profiles, portfolio)

        # 6. 净值数据库更新
        if HAS_NAV_DB:
            try:
                logger.info("正在更新净值数据库...")
                products_for_db = []
                for p in all_profiles:
                    products_for_db.append({
                        'product_code': p.product_code,
                        'product_name': p.product_name,
                        'nav_history': [
                            {
                                'date': self.format_date(nav.get('date')),
                                'unit_nav': nav.get('unit_nav'),
                            }
                            for nav in p.nav_history if nav.get('unit_nav')
                        ]
                    })
                stats = update_nav_database('浦银', products_for_db)
                logger.info(f"净值数据库更新完成: 新增 {len(stats.get('new_dates', []))} 个日期, "
                            f"更新 {stats.get('updated_cells', 0)} 个单元格")
            except Exception as e:
                logger.warning(f"更新净值数据库失败: {e}")

        # 7. 报告
        self.display_report(all_profiles, portfolio)

        return filename

    # ========================================================================
    # 结果输出
    # ========================================================================

    def save_results(self, profiles: List[ProductProfile], portfolio: Dict) -> str:
        """保存分析结果到Excel（6个Sheet）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'浦银理财_量化分析_{timestamp}.xlsx'

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: 全部产品
            all_data = []
            for p in profiles:
                all_data.append({
                    '产品名称': p.product_name,
                    '产品代码': p.product_code,
                    '风险等级': p.risk_level,
                    '风险类型': p.risk_category.value,
                    '产品类型': p.product_type.value,
                    '可购买': '是' if p.is_buyable else '否',
                    '申购状态': p.status.value,
                    '持有期限': p.duration_text,
                    '持有天数': p.duration_days,
                    '最新净值日期': p.latest_nav_date,
                    '距今天数': p.days_from_today,
                    '最新净值': p.latest_nav,
                    '近1周年化(%)': p.metrics.return_1w,
                    '近1月年化(%)': p.metrics.return_1m,
                    '近3月年化(%)': p.metrics.return_3m,
                    '近6月年化(%)': p.metrics.return_6m,
                    '波动率(%)': p.metrics.volatility,
                    '最大回撤(%)': p.metrics.max_drawdown,
                    '夏普比率': p.metrics.sharpe_ratio,
                    '卡玛比率': p.metrics.calmar_ratio,
                    '综合评分': p.score,
                    '交易信号': p.signal.value,
                    '业绩基准': p.benchmark,
                })
            df_all = pd.DataFrame(all_data)
            df_all.to_excel(writer, sheet_name='全部产品', index=False)

            # Sheet 2: 固收类TOP20
            fixed_income = [p for p in profiles if p.risk_category == RiskCategory.FIXED_INCOME]
            fixed_income.sort(key=lambda x: x.score, reverse=True)
            fixed_data = []
            for i, p in enumerate(fixed_income[:20], 1):
                fixed_data.append({
                    '排名': i,
                    '产品名称': p.product_name,
                    '产品代码': p.product_code,
                    '风险等级': p.risk_level,
                    '产品类型': p.product_type.value,
                    '可购买': '是' if p.is_buyable else '否',
                    '持有期限': p.duration_text,
                    '近3月年化(%)': p.metrics.return_3m,
                    '近1周年化(%)': p.metrics.return_1w,
                    '波动率(%)': p.metrics.volatility,
                    '综合评分': p.score,
                    '交易信号': p.signal.value,
                })
            df_fixed = pd.DataFrame(fixed_data)
            df_fixed.to_excel(writer, sheet_name='固收类TOP20', index=False)

            # Sheet 3: 权益类TOP20
            equity = [p for p in profiles if p.risk_category == RiskCategory.EQUITY]
            equity.sort(key=lambda x: x.score, reverse=True)
            equity_data = []
            for i, p in enumerate(equity[:20], 1):
                equity_data.append({
                    '排名': i,
                    '产品名称': p.product_name,
                    '产品代码': p.product_code,
                    '风险等级': p.risk_level,
                    '申购状态': p.status.value,
                    '夏普比率': p.metrics.sharpe_ratio,
                    '卡玛比率': p.metrics.calmar_ratio,
                    '最大回撤(%)': p.metrics.max_drawdown,
                    '近1月年化(%)': p.metrics.return_1m,
                    '综合评分': p.score,
                    '交易信号': p.signal.value,
                })
            df_equity = pd.DataFrame(equity_data)
            df_equity.to_excel(writer, sheet_name='权益类TOP20', index=False)

            # Sheet 4: 投资组合推荐
            portfolio_data = []
            for rec in portfolio['core']['recommendations']:
                portfolio_data.append({
                    '仓位类型': '核心仓位(60%-80%)',
                    '产品名称': rec['name'],
                    '产品代码': rec['code'],
                    '风险等级': rec['risk'],
                    '综合评分': rec['score'],
                    '交易信号': rec['signal'],
                    '近3月年化(%)': rec.get('return_3m'),
                    '波动率(%)': rec.get('volatility'),
                    '持有期限': rec.get('duration'),
                })
            for rec in portfolio['satellite']['recommendations']:
                portfolio_data.append({
                    '仓位类型': '卫星仓位(20%-40%)',
                    '产品名称': rec['name'],
                    '产品代码': rec['code'],
                    '风险等级': rec['risk'],
                    '综合评分': rec['score'],
                    '交易信号': rec['signal'],
                    '夏普比率': rec.get('sharpe'),
                    '卡玛比率': rec.get('calmar'),
                    '最大回撤(%)': rec.get('max_drawdown'),
                })
            df_portfolio = pd.DataFrame(portfolio_data)
            df_portfolio.to_excel(writer, sheet_name='投资组合推荐', index=False)

            # Sheet 5: 买入信号
            buy_signals = [p for p in profiles
                           if p.signal in [SignalType.STRONG_BUY, SignalType.BUY]]
            buy_signals.sort(key=lambda x: x.score, reverse=True)
            buy_data = []
            for p in buy_signals[:30]:
                buy_data.append({
                    '产品名称': p.product_name,
                    '产品代码': p.product_code,
                    '风险等级': p.risk_level,
                    '风险类型': p.risk_category.value,
                    '交易信号': p.signal.value,
                    '综合评分': p.score,
                    '持有期限': p.duration_text,
                    '近3月年化(%)': p.metrics.return_3m,
                    '夏普比率': p.metrics.sharpe_ratio,
                    '最大回撤(%)': p.metrics.max_drawdown,
                })
            df_buy = pd.DataFrame(buy_data)
            df_buy.to_excel(writer, sheet_name='买入信号', index=False)

            # Sheet 6: 可购买持有期产品
            holding_period = [p for p in profiles
                              if p.is_buyable and p.product_type == ProductType.HOLDING_PERIOD]
            holding_period.sort(key=lambda x: x.score, reverse=True)
            holding_data = []
            for i, p in enumerate(holding_period, 1):
                holding_data.append({
                    '排名': i,
                    '产品名称': p.product_name,
                    '产品代码': p.product_code,
                    '风险等级': p.risk_level,
                    '持有期限': p.duration_text,
                    '近1周年化(%)': p.metrics.return_1w,
                    '近1月年化(%)': p.metrics.return_1m,
                    '近3月年化(%)': p.metrics.return_3m,
                    '波动率(%)': p.metrics.volatility,
                    '最大回撤(%)': p.metrics.max_drawdown,
                    '夏普比率': p.metrics.sharpe_ratio,
                    '综合评分': p.score,
                    '交易信号': p.signal.value,
                    '业绩基准': p.benchmark,
                })
            df_holding = pd.DataFrame(holding_data)
            df_holding.to_excel(writer, sheet_name='可购买持有期产品', index=False)

        logger.info(f"\n分析结果已保存到: {filename}")

        # latest版本
        latest_file = '浦银理财_量化分析_latest.xlsx'
        try:
            with pd.ExcelWriter(latest_file, engine='openpyxl') as writer:
                df_holding.to_excel(writer, sheet_name='可购买持有期产品', index=False)
                df_all.to_excel(writer, sheet_name='全部产品', index=False)
                df_fixed.to_excel(writer, sheet_name='固收类TOP20', index=False)
                df_equity.to_excel(writer, sheet_name='权益类TOP20', index=False)
                df_portfolio.to_excel(writer, sheet_name='投资组合推荐', index=False)
                df_buy.to_excel(writer, sheet_name='买入信号', index=False)
            logger.info(f"最新数据已保存到: {latest_file}")
        except PermissionError:
            logger.warning(f"无法保存到 {latest_file}（文件可能被占用）")

        return filename

    def display_report(self, profiles: List[ProductProfile], portfolio: Dict):
        """显示分析报告"""
        print("\n" + "=" * 80)
        print("                    浦银理财产品量化分析报告")
        print("=" * 80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"产品总数: {len(profiles)}")

        buyable_count = sum(1 for p in profiles if p.is_buyable)
        fixed_count = sum(1 for p in profiles if p.risk_category == RiskCategory.FIXED_INCOME)
        equity_count = sum(1 for p in profiles if p.risk_category == RiskCategory.EQUITY)
        holding_count = sum(1 for p in profiles
                            if p.is_buyable and p.product_type == ProductType.HOLDING_PERIOD)
        buy_signals = sum(1 for p in profiles
                          if p.signal in [SignalType.STRONG_BUY, SignalType.BUY])

        print(f"可购买产品: {buyable_count} (其中持有期产品: {holding_count})")
        print(f"固收类: {fixed_count} | 权益类: {equity_count}")
        print(f"买入信号数: {buy_signals}")

        today = datetime.now()
        if today.weekday() >= 3:
            print("\n" + "-" * 80)
            print("注意: 当前接近周末，建议谨慎申购（资金周末可能无法计息）")
            print("-" * 80)

        # 可购买持有期产品TOP10
        print("\n" + "=" * 80)
        print("              可购买持有期产品TOP10 (随时可申购)")
        print("=" * 80)

        holding = [p for p in profiles
                   if p.is_buyable and p.product_type == ProductType.HOLDING_PERIOD]
        holding.sort(key=lambda x: x.score, reverse=True)

        if holding:
            header = f"{'#':^3} | {'产品名称':^30} | {'期限':^8} | {'3月年化':^8} | {'评分':^6} | {'信号':^8}"
            print(header)
            print("-" * 80)
            for i, p in enumerate(holding[:10], 1):
                name = p.product_name[:28] if len(p.product_name) > 28 else p.product_name
                duration = p.duration_text[:6] if p.duration_text else "N/A"
                ret_3m = f"{p.metrics.return_3m:+.2f}%" if p.metrics.return_3m else "N/A"
                score = f"{p.score:.2f}"
                signal = p.signal.value
                print(f"{i:^3} | {name:30s} | {duration:^8} | {ret_3m:^8} | {score:^6} | {signal:^8}")
        else:
            print("  暂无可购买的持有期产品")

        # 固收类TOP10
        print("\n" + "=" * 80)
        print("                    固收类TOP10 (R1/R2/R3)")
        print("评分公式: 0.6*近3月年化 + 0.3*近1周年化 - 0.1*波动率")
        print("=" * 80)

        fixed = [p for p in profiles if p.risk_category == RiskCategory.FIXED_INCOME]
        fixed.sort(key=lambda x: x.score, reverse=True)

        header = f"{'#':^3} | {'产品名称':^28} | {'状态':^6} | {'3月年化':^8} | {'评分':^6} | {'信号':^8}"
        print(header)
        print("-" * 80)
        for i, p in enumerate(fixed[:10], 1):
            name = p.product_name[:26] if len(p.product_name) > 26 else p.product_name
            status = p.status.value[:4]
            ret_3m = f"{p.metrics.return_3m:+.2f}%" if p.metrics.return_3m else "N/A"
            score = f"{p.score:.2f}"
            signal = p.signal.value
            print(f"{i:^3} | {name:28s} | {status:^6} | {ret_3m:^8} | {score:^6} | {signal:^8}")

        # 权益类TOP10
        print("\n" + "=" * 80)
        print("                    权益类TOP10 (R4/R5)")
        print("评分公式: 0.5*夏普比率 + 0.3*卡玛比率 + 0.2*超额收益")
        print("=" * 80)

        equity = [p for p in profiles if p.risk_category == RiskCategory.EQUITY]
        equity.sort(key=lambda x: x.score, reverse=True)

        header = f"{'#':^3} | {'产品名称':^28} | {'夏普':^6} | {'卡玛':^6} | {'回撤':^6} | {'评分':^6} | {'信号':^8}"
        print(header)
        print("-" * 80)
        for i, p in enumerate(equity[:10], 1):
            name = p.product_name[:26] if len(p.product_name) > 26 else p.product_name
            sharpe = f"{p.metrics.sharpe_ratio:.2f}" if p.metrics.sharpe_ratio else "N/A"
            calmar = f"{p.metrics.calmar_ratio:.2f}" if p.metrics.calmar_ratio else "N/A"
            mdd = f"{p.metrics.max_drawdown:.1f}%" if p.metrics.max_drawdown else "N/A"
            score = f"{p.score:.2f}"
            signal = p.signal.value
            print(f"{i:^3} | {name:28s} | {sharpe:^6} | {calmar:^6} | {mdd:^6} | {score:^6} | {signal:^8}")

        # 投资组合推荐
        print("\n" + "=" * 80)
        print("                    投资组合推荐 (核心-卫星策略)")
        print("=" * 80)

        print("\n[核心仓位 60%-80%] - 固收类持有期产品（随时可购买）")
        print("-" * 60)
        if portfolio['core']['recommendations']:
            for rec in portfolio['core']['recommendations'][:3]:
                ptype = rec.get('product_type', '未知')
                print(f"  {rec['signal']:8} | {rec['name'][:30]}")
                print(f"           类型: {ptype} | 近3月年化: {rec.get('return_3m', 'N/A')}% | "
                      f"期限: {rec.get('duration', 'N/A')}")
        else:
            print("  暂无符合条件的推荐")

        print(f"\n[卫星仓位 20%-40%] - 权益类增强产品")
        print(f"  止损规则: 净值跌破 {Config.STOP_LOSS_THRESHOLD} 即赎回")
        print("-" * 60)
        if portfolio['satellite']['recommendations']:
            for rec in portfolio['satellite']['recommendations'][:3]:
                print(f"  {rec['signal']:8} | {rec['name'][:30]}")
                print(f"           夏普: {rec.get('sharpe', 'N/A')} | "
                      f"最大回撤: {rec.get('max_drawdown', 'N/A')}%")
        else:
            print("  暂无符合条件的推荐")

        print("\n" + "=" * 80)
        print("提示: 详细数据请查看Excel文件中的各分析Sheet")
        print("=" * 80)


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("        浦银理财产品量化分析系统 v1.0")
    print("=" * 70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("核心功能:")
    print("  - 全维度产品画像（申购状态、持有期限、风险等级）")
    print("  - 量化指标计算（近1周/1月/3月/6月年化、夏普比率、卡玛比率、最大回撤）")
    print("  - 多因子评分模型（固收类/权益类分别评分）")
    print("  - T+0 动态信号生成（买入/持有/观望/回避）")
    print("  - 核心-卫星投资组合推荐")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            crawler = SPDBWealthCrawler()
            crawler.crawl_all_products(limit=50)
        elif sys.argv[1] == '--help':
            print("用法:")
            print("  python spdb_wm.py              # 立即执行完整分析")
            print("  python spdb_wm.py --test       # 测试模式（50个产品）")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        crawler = SPDBWealthCrawler()
        result = crawler.crawl_all_products()

        if result:
            print(f"\n分析完成！文件: {result}")
        else:
            print("\n分析失败，请查看日志")


if __name__ == "__main__":
    main()
