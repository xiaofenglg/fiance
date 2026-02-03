"""
多银行理财产品数据采集与量化分析系统

支持银行：
1. 民生银行 (CMBC) - 完整API支持
2. 宁波银行 (NBB) - API支持
3. 华夏理财 (Huaxia) - JS数据解析
4. 信银理财 (CITIC) - API支持
5. 浦发银行 (SPDB) - 待开发
6. 光大银行 (CEB) - WAF保护，需要特殊处理

版本：1.0
日期：2026-01-16
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging
import json
import ssl
import urllib3
import re
import io
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# PDF解析
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    logging.warning("PyPDF2未安装，宁波银行净值解析功能受限")

# Selenium支持 (CITIC爬虫需要)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    logging.warning("Selenium未安装，信银理财爬虫功能受限")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bank_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型
# ============================================================================

class BankType(Enum):
    """银行类型"""
    CMBC = "民生银行"
    NBB = "宁波银行"
    HUAXIA = "华夏理财"
    CITIC = "信银理财"
    SPDB = "浦发银行"
    CEB = "光大银行"


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
    FIXED_INCOME = "固收类"
    EQUITY = "权益类"


class SignalType(Enum):
    """交易信号"""
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
    """产品画像"""
    # 银行信息
    bank: BankType = BankType.CMBC
    bank_name: str = ""

    # 基础信息
    product_code: str = ""
    product_name: str = ""
    risk_level: str = ""
    risk_category: RiskCategory = RiskCategory.FIXED_INCOME

    # 产品类型
    product_type: ProductType = ProductType.OTHER

    # 申购状态
    status: ProductStatus = ProductStatus.UNKNOWN
    status_text: str = ""
    is_buyable: bool = False

    # 期限信息
    duration_days: Optional[int] = None
    duration_text: str = ""

    # 业绩基准
    benchmark: str = ""
    benchmark_value: Optional[float] = None

    # 净值信息
    latest_nav: float = 1.0
    latest_nav_date: str = ""
    days_from_today: int = 0

    # 量化指标
    metrics: QuantMetrics = field(default_factory=QuantMetrics)

    # 评分
    score: float = 0.0
    signal: SignalType = SignalType.WATCH

    # 历史净值
    nav_history: List[Dict] = field(default_factory=list)


# ============================================================================
# 配置
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

    STOP_LOSS_THRESHOLD = 0.95
    SESSION_NAV_LIMIT = 150


# ============================================================================
# SSL适配器
# ============================================================================

class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


# ============================================================================
# 量化引擎
# ============================================================================

class QuantEngine:
    """量化计算引擎"""

    @staticmethod
    def calculate_annualized_return(nav_series: List[float], days: int) -> Optional[float]:
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
# 银行爬虫基类
# ============================================================================

class BaseBankCrawler(ABC):
    """银行爬虫基类"""

    bank_type: BankType = BankType.CMBC
    bank_name: str = "基类"

    def __init__(self):
        self.session = self._create_session()
        self.quant = QuantEngine()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        adapter = LegacySSLAdapter(pool_connections=30, pool_maxsize=30)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        })
        session.verify = False
        urllib3.disable_warnings()
        return session

    @abstractmethod
    def get_product_list(self) -> List[Dict]:
        """获取产品列表"""
        pass

    @abstractmethod
    def get_nav_history(self, product_code: str, session: requests.Session = None) -> List[Dict]:
        """获取净值历史"""
        pass

    def parse_date(self, date_str) -> Optional[datetime]:
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

    def format_date(self, date_str) -> str:
        if date_str is None:
            return ""
        date_str = str(date_str).strip()
        date_str = ''.join(filter(str.isdigit, date_str))
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return str(date_str)

    def parse_risk_category(self, risk_level: str) -> RiskCategory:
        if not risk_level:
            return RiskCategory.FIXED_INCOME
        risk_str = str(risk_level).upper()
        if 'R4' in risk_str or 'R5' in risk_str or '4' in risk_str or '5' in risk_str:
            return RiskCategory.EQUITY
        return RiskCategory.FIXED_INCOME

    def parse_product_type(self, product_name: str) -> Tuple[ProductType, bool]:
        if '持有' in product_name or re.search(r'\d+天', product_name):
            if '封闭' not in product_name:
                return ProductType.HOLDING_PERIOD, True
        if any(kw in product_name for kw in ['周周盈', '天天盈', '日日盈', '现金', '货币']):
            return ProductType.CASH_MGMT, True
        if any(kw in product_name for kw in ['定开', '季开', '月开', '年开', '季季盈', '月月盈']):
            return ProductType.SCHEDULED_OPEN, False
        if '封闭' in product_name:
            return ProductType.CLOSED_END, False
        return ProductType.OTHER, True

    def calculate_metrics(self, nav_data: List[Dict], product_info: Dict) -> Optional[ProductProfile]:
        """计算量化指标"""
        if not nav_data or len(nav_data) < 2:
            return None

        # 子类需要实现具体的字段映射
        profile = ProductProfile()
        profile.bank = self.bank_type
        profile.bank_name = self.bank_name

        return profile

    def calculate_score(self, profile: ProductProfile) -> float:
        """计算评分"""
        m = profile.metrics

        if profile.risk_category == RiskCategory.FIXED_INCOME:
            score = 0.0
            weights = Config.FIXED_INCOME_WEIGHTS

            if m.return_3m is not None:
                score += weights['return_3m'] * m.return_3m
            elif m.return_1m is not None:
                score += weights['return_3m'] * m.return_1m

            if m.return_1w is not None:
                score += weights['return_1w'] * m.return_1w

            if m.volatility is not None:
                score += weights['volatility'] * m.volatility
        else:
            score = 0.0
            weights = Config.EQUITY_WEIGHTS

            if m.sharpe_ratio is not None:
                normalized_sharpe = max(-5, min(5, m.sharpe_ratio)) * 10
                score += weights['sharpe_ratio'] * normalized_sharpe

            if m.calmar_ratio is not None:
                normalized_calmar = max(-5, min(5, m.calmar_ratio)) * 10
                score += weights['calmar_ratio'] * normalized_calmar

            if m.excess_return_1m is not None:
                score += weights['excess_return_1m'] * m.excess_return_1m

        return round(score, 2)

    def generate_signal(self, profile: ProductProfile) -> SignalType:
        """生成信号"""
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


# ============================================================================
# 民生银行爬虫
# ============================================================================

class CMBCCrawler(BaseBankCrawler):
    """民生银行理财爬虫"""

    bank_type = BankType.CMBC
    bank_name = "民生银行"

    def __init__(self):
        super().__init__()
        self.api_base = "https://www.cmbcwm.com.cn/gw/po_web"
        self.session.headers.update({
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://www.cmbcwm.com.cn',
            'Referer': 'https://www.cmbcwm.com.cn/grlc/index.htm'
        })

    def get_product_list(self) -> List[Dict]:
        logger.info(f"[{self.bank_name}] 获取产品列表...")
        all_products = []
        page = 1

        while True:
            try:
                url = f"{self.api_base}/BTAProductQry"
                payload = {"pageNo": page, "pageSize": 100, "code_or_name": ""}
                response = self.session.post(url, data=payload, timeout=30)
                data = response.json()

                if data.get('list'):
                    products = data['list']
                    all_products.extend(products)
                    total_size = data.get('totalSize', 0)
                    total_pages = (total_size + 99) // 100

                    if page >= total_pages:
                        break
                    page += 1
                    time.sleep(0.3)
                else:
                    break
            except Exception as e:
                logger.error(f"[{self.bank_name}] 获取第{page}页失败: {e}")
                break

        logger.info(f"[{self.bank_name}] 共获取 {len(all_products)} 个产品")
        return all_products

    def get_nav_history(self, product_code: str, session: requests.Session = None) -> List[Dict]:
        use_session = session or self.session
        try:
            url = f"{self.api_base}/BTADailyQry"
            payload = {
                "chart_type": 1,
                "real_prd_code": product_code,
                "begin_date": "",
                "end_date": "",
                "pageNo": 1,
                "pageSize": 210
            }
            response = use_session.post(url, data=payload, timeout=20)
            data = response.json()
            return data.get('btaDailyAddFieldList') or data.get('list') or []
        except:
            return []

    def parse_status(self, product_data: Dict) -> Tuple[ProductStatus, str]:
        status = product_data.get('STATUS')
        end_date = str(product_data.get('END_DATE', ''))
        name = product_data.get('PRD_NAME', '')

        if '封闭' in name and '持有' not in name:
            return ProductStatus.CLOSED, "封闭期产品"

        if end_date == '20991231':
            if status in [0, 4, '0', '4']:
                return ProductStatus.OPEN, "开放申购"

        if end_date.isdigit() and len(end_date) == 8:
            today = datetime.now().strftime('%Y%m%d')
            if end_date < today:
                return ProductStatus.CLOSED, "已到期"
            else:
                if status in [4, '4']:
                    return ProductStatus.OPEN, "开放申购"

        if status is not None:
            status_str = str(status).strip()
            if status_str in ['0', '4']:
                return ProductStatus.OPEN, "开放申购"
            if status_str in ['1', '2', '3']:
                return ProductStatus.SUSPENDED, "暂停/募集中"

        return ProductStatus.UNKNOWN, "状态未知"

    def calculate_metrics(self, nav_data: List[Dict], product_info: Dict) -> Optional[ProductProfile]:
        if not nav_data or len(nav_data) < 2:
            return None

        nav_sorted = sorted(nav_data, key=lambda x: str(x.get('ISS_DATE', '')), reverse=True)

        profile = ProductProfile()
        profile.bank = self.bank_type
        profile.bank_name = self.bank_name
        profile.product_code = product_info.get('REAL_PRD_CODE') or product_info.get('PRD_CODE', '')
        profile.product_name = product_info.get('PRD_NAME', '')
        profile.risk_level = str(product_info.get('RISK_LEVEL', ''))
        profile.risk_category = self.parse_risk_category(profile.risk_level)
        profile.benchmark = product_info.get('BENCHMARK_CUSTO', '')

        profile.product_type, type_buyable = self.parse_product_type(profile.product_name)
        profile.status, profile.status_text = self.parse_status(product_info)
        profile.is_buyable = (profile.status == ProductStatus.OPEN) and type_buyable

        # 提取净值
        navs = []
        dates = []
        for item in nav_sorted:
            nav = item.get('NAV')
            date = self.parse_date(item.get('ISS_DATE'))
            if nav and date:
                try:
                    navs.append(float(nav))
                    dates.append(date)
                except:
                    continue

        if len(navs) < 2:
            return None

        profile.latest_nav = navs[0]
        profile.latest_nav_date = self.format_date(nav_sorted[0].get('ISS_DATE'))
        profile.days_from_today = (datetime.now() - dates[0]).days if dates else 9999
        profile.nav_history = nav_sorted[:30]

        # 计算收益率
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

        # 波动率和回撤
        daily_returns = []
        for i in range(len(navs) - 1):
            if navs[i+1] != 0:
                ret = (navs[i] - navs[i+1]) / navs[i+1] * 100
                daily_returns.append(ret)

        profile.metrics.volatility = self.quant.calculate_volatility(daily_returns)
        profile.metrics.max_drawdown = self.quant.calculate_max_drawdown(half_year_navs if half_year_navs else navs)

        return_for_sharpe = profile.metrics.return_3m or profile.metrics.return_1m
        profile.metrics.sharpe_ratio = self.quant.calculate_sharpe_ratio(return_for_sharpe, profile.metrics.volatility)
        profile.metrics.calmar_ratio = self.quant.calculate_calmar_ratio(return_for_sharpe, profile.metrics.max_drawdown)

        if profile.metrics.return_1m is not None:
            profile.metrics.excess_return_1m = profile.metrics.return_1m - Config.RISK_FREE_RATE

        return profile


# ============================================================================
# 宁波银行爬虫
# ============================================================================

class NBBCrawler(BaseBankCrawler):
    """宁波银行理财爬虫 - 支持PDF净值解析"""

    bank_type = BankType.NBB
    bank_name = "宁波银行"

    # 配置
    REQUEST_DELAY = 0.5
    MAX_RETRIES = 3
    TIMEOUT = 60
    NAV_DAYS = 15  # 获取多少天的净值数据

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.wmbnb.com"
        self.api_base = "https://www.wmbnb.com/ningbo-web"
        self.nav_cache = {}  # 缓存已下载的净值数据 {product_code: [{date, nav}, ...]}

    def _create_session(self) -> requests.Session:
        """创建针对宁波银行优化的session"""
        session = requests.Session()
        adapter = LegacySSLAdapter(pool_connections=10, pool_maxsize=10)
        session.mount('https://', adapter)
        session.mount('http://', adapter)

        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Content-Type': 'application/json;charset=UTF-8',
            'Origin': 'https://www.wmbnb.com',
            'Referer': 'https://www.wmbnb.com/product/index.html',
            'X-Requested-With': 'XMLHttpRequest',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        })
        session.verify = False
        urllib3.disable_warnings()
        return session

    def _init_session(self):
        """初始化session"""
        for retry in range(self.MAX_RETRIES):
            try:
                self.session = self._create_session()
                resp = self.session.get(
                    f"{self.base_url}/product/index.html",
                    timeout=self.TIMEOUT
                )
                if resp.status_code == 200:
                    logger.info(f"[{self.bank_name}] Session初始化成功")
                    return True
            except Exception as e:
                logger.warning(f"[{self.bank_name}] Session初始化失败(重试{retry+1}/{self.MAX_RETRIES}): {e}")
                time.sleep(2 * (retry + 1))
        return False

    def get_product_list(self) -> List[Dict]:
        """获取产品列表"""
        logger.info(f"[{self.bank_name}] 获取产品列表...")

        if not self._init_session():
            logger.error(f"[{self.bank_name}] 无法连接到服务器")
            return []

        all_products = []
        page = 1
        seen_codes = set()

        while True:
            try:
                url = f"{self.api_base}/product/list.json"
                payload = {
                    "curPage": page,
                    "pageCount": 100,
                    "productTapValue": "",
                    "conductTapValue": "",
                    "profucatTapValue": "",
                }

                response = self.session.post(url, json=payload, timeout=self.TIMEOUT)
                data = response.json()

                if data.get('status') == 'success' and data.get('list'):
                    products = data['list']
                    new_products = 0

                    for p in products:
                        code = p.get('projectcode')
                        if code and code not in seen_codes:
                            seen_codes.add(code)
                            all_products.append(p)
                            new_products += 1

                    total = data.get('total', 0)
                    logger.info(f"[{self.bank_name}] 第 {page} 页: {len(products)} 个产品, 新增 {new_products}")

                    # 如果没有新产品，说明数据重复了
                    if new_products == 0:
                        break

                    page += 1
                    time.sleep(self.REQUEST_DELAY)

                    # 安全限制
                    if page > 300:
                        break
                else:
                    break

            except Exception as e:
                logger.error(f"[{self.bank_name}] 获取第{page}页失败: {e}")
                break

        logger.info(f"[{self.bank_name}] 共获取 {len(all_products)} 个产品")
        return all_products

    def get_attachment_list(self, max_pages: int = 50) -> Dict[str, List[Dict]]:
        """获取净值公告附件列表，按产品代码分组

        获取多页数据以覆盖更多天数的净值记录
        """
        logger.info(f"[{self.bank_name}] 获取净值公告列表（最多{max_pages}页）...")

        by_code = {}
        page = 1
        total_fetched = 0

        while page <= max_pages:
            try:
                url = f"{self.api_base}/product/attachmentlist.json"
                payload = {
                    "curPage": page,
                    "pageCount": 100,
                    "request_num": 100,
                    "request_pageno": page,
                }

                response = self.session.post(url, json=payload, timeout=self.TIMEOUT)
                data = response.json()

                if data.get('status') == 'success':
                    attachments = data.get('list', [])

                    if not attachments:
                        break

                    new_count = 0
                    for att in attachments:
                        url_path = att.get('url', '')
                        match = re.search(r'/report/([A-Z0-9]+)_[^_]+_(\d{8})', url_path)
                        if match:
                            code = match.group(1)
                            date = match.group(2)

                            if code not in by_code:
                                by_code[code] = []

                            # 检查是否已有该日期的数据
                            existing_dates = {a['date'] for a in by_code[code]}
                            if date not in existing_dates:
                                by_code[code].append({
                                    'url': url_path,
                                    'date': date,
                                    'title': att.get('title', '')
                                })
                                new_count += 1

                    total_fetched += len(attachments)

                    # 如果没有新数据，停止获取
                    if new_count == 0:
                        logger.info(f"[{self.bank_name}] 第{page}页无新数据，停止获取")
                        break

                    if page % 10 == 0:
                        logger.info(f"[{self.bank_name}] 已获取第{page}页，累计{len(by_code)}个产品，{total_fetched}条记录")

                    page += 1
                    time.sleep(self.REQUEST_DELAY)
                else:
                    break

            except Exception as e:
                logger.warning(f"[{self.bank_name}] 获取第{page}页失败: {e}")
                break

        # 统计
        total_records = sum(len(v) for v in by_code.values())
        avg_days = total_records / len(by_code) if by_code else 0
        logger.info(f"[{self.bank_name}] 获取完成: {len(by_code)}个产品，平均{avg_days:.1f}天数据")

        return by_code

    def download_and_parse_pdf(self, pdf_url: str) -> Optional[Dict]:
        """下载并解析PDF获取净值数据"""
        if not HAS_PYPDF2:
            return None

        try:
            full_url = f"{self.base_url}{pdf_url}"
            resp = self.session.get(full_url, timeout=30)

            if resp.status_code == 200:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(resp.content))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        # 从URL中提取日期
                        date_match = re.search(r'_(\d{8})', pdf_url)
                        date = date_match.group(1) if date_match else None

                        # 从PDF文本中提取净值
                        # 格式: 单位净值 累计净值
                        nav_match = re.search(r'(\d+\.\d{4,6})\s+(\d+\.\d{4,6})', text)
                        if nav_match:
                            return {
                                'date': date,
                                'unit_nav': float(nav_match.group(1)),
                                'total_nav': float(nav_match.group(2))
                            }
        except Exception as e:
            logger.debug(f"[{self.bank_name}] PDF解析失败: {e}")

        return None

    def get_nav_history(self, product_code: str, session: requests.Session = None) -> List[Dict]:
        """获取产品净值历史 - 从PDF解析"""
        # 检查缓存
        if product_code in self.nav_cache:
            return self.nav_cache[product_code]

        nav_list = []

        # 获取附件列表
        attachments = self.get_attachment_list()
        product_attachments = attachments.get(product_code, [])

        if not product_attachments:
            return nav_list

        # 只获取最近N天的数据
        product_attachments = sorted(product_attachments, key=lambda x: x['date'], reverse=True)
        product_attachments = product_attachments[:self.NAV_DAYS]

        for att in product_attachments:
            nav_data = self.download_and_parse_pdf(att['url'])
            if nav_data:
                nav_list.append(nav_data)
                time.sleep(0.3)  # 请求间隔

        # 缓存结果
        self.nav_cache[product_code] = nav_list
        return nav_list

    def parse_status(self, product_data: Dict) -> Tuple[ProductStatus, str]:
        """解析产品状态"""
        status = product_data.get('curstateDesc') or product_data.get('projectstate')

        if status:
            if '在售' in str(status) or '开放' in str(status):
                return ProductStatus.OPEN, "开放申购"
            elif '封闭' in str(status) or '募集' in str(status):
                return ProductStatus.CLOSED, "封闭期"

        name = product_data.get('projectname', '')
        if '封闭' in name and '持有' not in name:
            return ProductStatus.CLOSED, "封闭期产品"

        return ProductStatus.OPEN, "开放申购"

    def calculate_metrics(self, nav_data: List[Dict], product_info: Dict) -> Optional[ProductProfile]:
        """计算宁波银行产品量化指标"""
        profile = ProductProfile()
        profile.bank = self.bank_type
        profile.bank_name = self.bank_name
        profile.product_code = product_info.get('projectcode') or product_info.get('id', '')
        profile.product_name = product_info.get('projectname') or product_info.get('projectshortname', '')

        # 解析风险等级
        risk = product_info.get('risklevel') or product_info.get('riskgrade', '')
        profile.risk_level = str(risk)
        profile.risk_category = self.parse_risk_category(profile.risk_level)

        # 解析产品类型
        profile.product_type, type_buyable = self.parse_product_type(profile.product_name)

        # 解析状态
        profile.status, profile.status_text = self.parse_status(product_info)
        profile.is_buyable = (profile.status == ProductStatus.OPEN) and type_buyable

        # 业绩基准
        profile.benchmark = product_info.get('benchmarkincome') or product_info.get('expectyield', '')

        # 如果有NAV数据，计算量化指标
        if nav_data and len(nav_data) >= 1:
            # 按日期排序（最新在前）
            nav_sorted = sorted(nav_data, key=lambda x: x.get('date', ''), reverse=True)

            # 提取净值和日期
            navs = []
            dates = []
            for item in nav_sorted:
                nav = item.get('unit_nav')
                date_str = item.get('date')
                if nav and date_str:
                    navs.append(nav)
                    try:
                        dates.append(datetime.strptime(date_str, '%Y%m%d'))
                    except:
                        pass

            if navs and dates:
                profile.latest_nav = navs[0]
                profile.latest_nav_date = dates[0].strftime('%Y-%m-%d')
                profile.days_from_today = (datetime.now() - dates[0]).days

                # 保存历史数据
                profile.nav_history = nav_sorted[:30]

                # 计算收益率（需要至少2个数据点）
                if len(navs) >= 2:
                    today = datetime.now()

                    week_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 7]
                    if len(week_navs) >= 2:
                        profile.metrics.return_1w = self.quant.calculate_annualized_return(week_navs, min(7, len(week_navs)))

                    month_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 30]
                    if len(month_navs) >= 2:
                        profile.metrics.return_1m = self.quant.calculate_annualized_return(month_navs, min(30, len(month_navs)))

                    # 波动率
                    daily_returns = []
                    for i in range(len(navs) - 1):
                        if navs[i+1] != 0:
                            ret = (navs[i] - navs[i+1]) / navs[i+1] * 100
                            daily_returns.append(ret)

                    profile.metrics.volatility = self.quant.calculate_volatility(daily_returns)
                    profile.metrics.max_drawdown = self.quant.calculate_max_drawdown(navs)

                    # 夏普比率
                    return_for_sharpe = profile.metrics.return_1m or profile.metrics.return_1w
                    profile.metrics.sharpe_ratio = self.quant.calculate_sharpe_ratio(
                        return_for_sharpe, profile.metrics.volatility
                    )

        else:
            # 没有NAV数据，使用产品基本信息
            profile.latest_nav = 1.0
            profile.latest_nav_date = ""
            profile.days_from_today = 0
            profile.metrics = QuantMetrics()

        return profile


# ============================================================================
# 华夏理财爬虫
# ============================================================================

class HuaxiaCrawler(BaseBankCrawler):
    """华夏理财爬虫

    数据来源: https://www.hxwm.com.cn
    - 产品净值报告列表: /common/js/gmjzbgData.js
    - 净值数据: Excel文件 (包含最近9天历史净值)

    增量更新机制:
    - 首次运行: 下载所有产品的Excel，提取9天净值数据
    - 再次运行: 对比已有数据的日期，只补充新日期的净值
    """

    bank_type = BankType.HUAXIA
    bank_name = "华夏理财"
    NAV_DAYS = 9999  # 保存尽量多的净值数据（不截断历史）
    NAV_STORAGE_FILE = "huaxia_nav_history.json"  # 净值历史存储文件

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.hxwm.com.cn"
        self.nav_storage: Dict[str, List[Dict]] = {}  # 产品代码 -> 净值历史
        self._load_nav_storage()

    def _load_nav_storage(self):
        """加载已保存的净值历史"""
        storage_path = os.path.join(os.path.dirname(__file__), self.NAV_STORAGE_FILE)
        if os.path.exists(storage_path):
            try:
                with open(storage_path, 'r', encoding='utf-8') as f:
                    self.nav_storage = json.load(f)
                logger.info(f"[{self.bank_name}] 已加载 {len(self.nav_storage)} 个产品的历史净值")
            except Exception as e:
                logger.warning(f"[{self.bank_name}] 加载净值历史失败: {e}")
                self.nav_storage = {}

    def _save_nav_storage(self):
        """保存净值历史到文件"""
        storage_path = os.path.join(os.path.dirname(__file__), self.NAV_STORAGE_FILE)
        try:
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.nav_storage, f, ensure_ascii=False, indent=2)
            logger.info(f"[{self.bank_name}] 已保存 {len(self.nav_storage)} 个产品的净值历史")
        except Exception as e:
            logger.error(f"[{self.bank_name}] 保存净值历史失败: {e}")

    def _merge_nav_data(self, product_code: str, new_nav_data: List[Dict]) -> List[Dict]:
        """合并新旧净值数据，去重并按日期排序"""
        existing = self.nav_storage.get(product_code, [])

        # 用字典去重，key为日期
        nav_dict = {}
        for nav in existing:
            date = nav.get('date', '')
            if date:
                nav_dict[date] = nav

        # 添加新数据
        new_count = 0
        for nav in new_nav_data:
            date = nav.get('date', '')
            if date and date not in nav_dict:
                nav_dict[date] = nav
                new_count += 1

        # 按日期倒序排序，只保留最近NAV_DAYS天
        merged = sorted(nav_dict.values(), key=lambda x: x.get('date', ''), reverse=True)
        merged = merged[:self.NAV_DAYS]

        # 更新存储
        self.nav_storage[product_code] = merged

        if new_count > 0:
            logger.debug(f"[{self.bank_name}] {product_code}: 新增 {new_count} 天净值数据")

        return merged

    def get_product_list(self) -> List[Dict]:
        """获取产品净值报告列表，按产品ID分组返回最新的报告"""
        logger.info(f"[{self.bank_name}] 获取产品列表...")

        try:
            # 从JS文件获取净值报告数据
            url = f"{self.base_url}/common/js/gmjzbgData.js"
            response = self.session.get(url, timeout=30)
            content = response.content.decode('utf-8', errors='ignore')

            # 解析JS中的JSON数组
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                # 清理尾部逗号
                json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

                try:
                    all_reports = json.loads(json_str)
                    logger.info(f"[{self.bank_name}] 共获取 {len(all_reports)} 条净值报告")

                    # 按产品ID分组，只保留每个产品最新的报告
                    products_by_id = {}
                    for report in all_reports:
                        prod_id = report.get('id', '')
                        if not prod_id:
                            continue

                        # 如果这个产品ID还没有记录，或者当前报告更新
                        if prod_id not in products_by_id:
                            products_by_id[prod_id] = report
                        else:
                            existing_date = products_by_id[prod_id].get('date', 0)
                            current_date = report.get('date', 0)
                            if current_date > existing_date:
                                products_by_id[prod_id] = report

                    products = list(products_by_id.values())
                    logger.info(f"[{self.bank_name}] 去重后共 {len(products)} 个产品")
                    return products
                except json.JSONDecodeError as e:
                    logger.error(f"[{self.bank_name}] JSON解析失败: {e}")
        except Exception as e:
            logger.error(f"[{self.bank_name}] 获取产品列表失败: {e}")

        return []

    def get_nav_history(self, product_code: str, session: requests.Session = None) -> List[Dict]:
        """从Excel文件获取NAV历史数据"""
        # 华夏理财的NAV数据直接包含在Excel文件中
        # 此方法由download_nav_excel替代
        return []

    def download_nav_excel(self, excel_url: str) -> List[Dict]:
        """下载并解析净值Excel文件

        Excel结构:
        - 第0行: 产品名称
        - 第1行: 表头 (净值日期, 单位净值, 累计单位净值, 资产净值)
        - 第2行起: 净值数据（行数不固定，新产品可能只有1天，老产品可能有9天或更多）
        """
        nav_data = []

        try:
            full_url = f"{self.base_url}{excel_url}"
            response = self.session.get(full_url, timeout=30)

            if response.status_code != 200:
                logger.debug(f"[{self.bank_name}] Excel下载失败: {response.status_code}")
                return nav_data

            # 使用pandas读取Excel
            df = pd.read_excel(io.BytesIO(response.content), header=None)

            if len(df) < 3:
                return nav_data

            # 从第2行开始读取所有有效数据行（不限制行数）
            for idx in range(2, len(df)):
                row = df.iloc[idx]
                try:
                    date_val = row[0]
                    unit_nav = row[1]

                    # 如果日期为空，说明已经读完所有数据
                    if pd.isna(date_val):
                        break

                    # 如果单位净值为空，跳过该行
                    if pd.isna(unit_nav):
                        continue

                    # 转换日期格式
                    if isinstance(date_val, str):
                        date_str = date_val.replace('/', '-').strip()
                    else:
                        try:
                            date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
                        except:
                            continue

                    # 验证日期格式
                    if not date_str or len(date_str) < 8:
                        continue

                    # 转换净值（保留4位小数精度）
                    try:
                        nav_float = float(unit_nav)
                    except (ValueError, TypeError):
                        continue

                    # 累计净值（可选，可能为空或'-'）
                    total_nav_float = None
                    if len(row) > 2 and pd.notna(row[2]):
                        try:
                            total_nav_val = row[2]
                            if isinstance(total_nav_val, (int, float)):
                                total_nav_float = float(total_nav_val)
                            elif isinstance(total_nav_val, str) and total_nav_val.strip() not in ['', '-']:
                                total_nav_float = float(total_nav_val)
                        except (ValueError, TypeError):
                            pass

                    # 如果没有累计净值，使用单位净值
                    if total_nav_float is None:
                        total_nav_float = nav_float

                    nav_data.append({
                        'date': date_str,
                        'unit_nav': round(nav_float, 4),
                        'total_nav': round(total_nav_float, 4)
                    })

                except Exception as e:
                    logger.debug(f"[{self.bank_name}] 解析行数据失败(行{idx}): {e}")
                    continue

        except Exception as e:
            logger.debug(f"[{self.bank_name}] Excel解析失败: {e}")

        return nav_data

    def calculate_metrics(self, nav_data: List[Dict], product_info: Dict) -> Optional[ProductProfile]:
        """计算产品指标 - 与民生理财计算逻辑保持一致"""
        # 至少需要2条净值数据才能计算收益率
        if not nav_data or len(nav_data) < 2:
            return None

        try:
            # 从product_info获取基本信息
            product_code = product_info.get('id', '')
            product_title = product_info.get('title', '')

            # 从标题提取产品名称 (格式: "产品名称[代码]净值报告")
            product_name = product_title
            name_match = re.search(r'^(.+?)\[', product_title)
            if name_match:
                product_name = name_match.group(1).strip()

            # 创建产品档案
            profile = ProductProfile()
            profile.bank = BankType.HUAXIA
            profile.bank_name = self.bank_name
            profile.product_code = product_code
            profile.product_name = product_name

            # 解析风险等级和风险类型
            if '权益' in product_name or '股票' in product_name or 'R4' in product_name or 'R5' in product_name:
                profile.risk_level = 'PR4'
                profile.risk_category = RiskCategory.EQUITY
            elif '混合' in product_name:
                profile.risk_level = 'PR3'
                profile.risk_category = RiskCategory.FIXED_INCOME
            else:
                profile.risk_level = 'PR2'
                profile.risk_category = RiskCategory.FIXED_INCOME

            # 解析产品类型和可购买状态
            profile.product_type, type_buyable = self.parse_product_type(product_name)

            # 提取持有期限
            duration_match = re.search(r'(\d+)天', product_name)
            if duration_match:
                profile.duration_days = int(duration_match.group(1))
                profile.duration_text = f"{profile.duration_days}天"

            # 解析状态（华夏理财默认为开放状态，封闭式产品除外）
            if '封闭' in product_name and '持有' not in product_name:
                profile.status = ProductStatus.CLOSED
                profile.status_text = "封闭期"
                profile.is_buyable = False
            else:
                profile.status = ProductStatus.OPEN
                profile.status_text = "开放申购"
                profile.is_buyable = type_buyable

            # 按日期排序净值数据（最新在前）
            nav_sorted = sorted(nav_data, key=lambda x: x.get('date', ''), reverse=True)

            # 提取净值和日期
            navs = []
            dates = []
            for item in nav_sorted:
                nav = item.get('unit_nav')
                date_str = item.get('date', '')
                if nav and date_str:
                    try:
                        navs.append(float(nav))
                        dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    except:
                        continue

            if len(navs) < 2:
                return None

            # 基本净值信息
            profile.latest_nav = navs[0]
            profile.latest_nav_date = dates[0].strftime('%Y-%m-%d')
            profile.days_from_today = (datetime.now() - dates[0]).days
            profile.nav_history = nav_sorted[:30]

            # ========== 计算收益率（与民生理财逻辑一致）==========
            today = datetime.now()

            # 近1周年化收益率
            week_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 7]
            if len(week_navs) >= 2:
                profile.metrics.return_1w = self.quant.calculate_annualized_return(week_navs, min(7, len(week_navs)))

            # 近1月年化收益率
            month_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 30]
            if len(month_navs) >= 2:
                profile.metrics.return_1m = self.quant.calculate_annualized_return(month_navs, min(30, len(month_navs)))

            # 近3月年化收益率
            quarter_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 90]
            if len(quarter_navs) >= 2:
                profile.metrics.return_3m = self.quant.calculate_annualized_return(quarter_navs, min(90, len(quarter_navs)))

            # 近6月年化收益率
            half_year_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 180]
            if len(half_year_navs) >= 2:
                profile.metrics.return_6m = self.quant.calculate_annualized_return(half_year_navs, min(180, len(half_year_navs)))

            # ========== 计算波动率和回撤 ==========
            daily_returns = []
            for i in range(len(navs) - 1):
                if navs[i + 1] != 0:
                    ret = (navs[i] - navs[i + 1]) / navs[i + 1] * 100
                    daily_returns.append(ret)

            profile.metrics.volatility = self.quant.calculate_volatility(daily_returns)
            profile.metrics.max_drawdown = self.quant.calculate_max_drawdown(half_year_navs if half_year_navs else navs)

            # ========== 计算夏普比率和卡玛比率 ==========
            return_for_sharpe = profile.metrics.return_3m or profile.metrics.return_1m
            profile.metrics.sharpe_ratio = self.quant.calculate_sharpe_ratio(return_for_sharpe, profile.metrics.volatility)
            profile.metrics.calmar_ratio = self.quant.calculate_calmar_ratio(return_for_sharpe, profile.metrics.max_drawdown)

            # 超额收益
            if profile.metrics.return_1m is not None:
                profile.metrics.excess_return_1m = profile.metrics.return_1m - Config.RISK_FREE_RATE

            return profile

        except Exception as e:
            logger.debug(f"[{self.bank_name}] 计算指标失败: {e}")
            return None


# ============================================================================
# 信银理财爬虫
# ============================================================================

class CITICCrawler(BaseBankCrawler):
    """信银理财爬虫 - 使用Selenium获取产品列表,API获取历史净值"""

    bank_type = BankType.CITIC
    bank_name = "信银理财"

    # API配置
    API_BASE = "https://wechat.citic-wealth.com/cms.product/api/custom/productInfo"

    # 产品类型映射
    PRODUCT_TYPE_MAP = {
        "每日购": 2,
        "定开": 3,
    }

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.citic-wealth.com"
        self.target_categories = ["每日购", "定开"]
        self._session = None
        self.driver = None

    def _get_session(self) -> requests.Session:
        """获取配置好SSL的session"""
        if self._session is None:
            self._session = requests.Session()
            self._session.mount('https://', LegacySSLAdapter())
        return self._session

    def _setup_driver(self):
        """设置Chrome驱动"""
        if not HAS_SELENIUM:
            raise ImportError("Selenium未安装，无法使用信银理财爬虫")

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')

        self.driver = webdriver.Chrome(options=options)
        return self.driver

    def _wait_loading_gone(self, timeout=30):
        """等待 loading 遮罩层消失"""
        try:
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            WebDriverWait(self.driver, timeout).until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, '.loading.screen-loading'))
            )
        except Exception:
            time.sleep(3)  # 兜底等待

    def _scroll_to_load_all(self, max_scrolls=200):
        """滚动加载所有产品"""
        last_count = 0
        scroll_count = 0
        no_change_count = 0  # 连续无变化计数
        max_no_change = 5    # 连续5次无变化才退出

        while scroll_count < max_scrolls:
            # 尝试滚动列表容器或body
            self.driver.execute_script("""
                var listContainer = document.querySelector('.product-list-container, .list-container, .content-container');
                if (listContainer) {
                    listContainer.scrollTop = listContainer.scrollHeight;
                }
                window.scrollTo(0, document.body.scrollHeight);
            """)
            time.sleep(1.5)  # 增加等待时间让内容加载

            products = self.driver.find_elements(By.CSS_SELECTOR, '.item-productCode')
            current_count = len(products)

            if current_count == last_count:
                no_change_count += 1
                if no_change_count >= max_no_change:
                    break
            else:
                no_change_count = 0  # 重置计数

            last_count = current_count
            scroll_count += 1

            if scroll_count % 20 == 0:
                logger.info(f"[{self.bank_name}] 滚动 {scroll_count}: {current_count} 个产品")

        return last_count

    def _extract_products_from_page(self, category_name: str) -> List[Dict]:
        """从当前页面提取产品数据"""
        products = []
        time.sleep(2)

        # 获取表头
        headers = self.driver.find_elements(By.CSS_SELECTOR, '.list-item-title-right .content')
        header_texts = [h.text.strip() for h in headers]

        # 获取产品名称和代码
        product_names = self.driver.find_elements(By.CSS_SELECTOR, '.item-productName')
        product_codes = self.driver.find_elements(By.CSS_SELECTOR, '.item-productCode div')
        right_rows = self.driver.find_elements(By.CSS_SELECTOR, '.list-item-right.list-Height70')

        for i, (name_el, code_el) in enumerate(zip(product_names, product_codes)):
            try:
                product = {
                    'product_name': name_el.text.strip(),
                    'product_code': code_el.text.strip(),
                    'category': category_name,
                }

                if i < len(right_rows):
                    row = right_rows[i]
                    cells = row.find_elements(By.CSS_SELECTOR, '.list-item-content-right')

                    for j, cell in enumerate(cells):
                        cell_text = cell.text.strip()
                        if j < len(header_texts):
                            header = header_texts[j]
                            if '单位净值' in header:
                                if '\n' in cell_text:
                                    parts = cell_text.split('\n')
                                    product['nav'] = parts[0]
                                    if len(parts) > 1:
                                        product['nav_date'] = parts[1]
                                else:
                                    product['nav'] = cell_text
                            elif '累计净值' in header:
                                if '\n' in cell_text:
                                    parts = cell_text.split('\n')
                                    product['total_nav'] = parts[0]
                                else:
                                    product['total_nav'] = cell_text
                            elif '起购金额' in header:
                                product['min_amount'] = cell_text
                            elif '风险等级' in header:
                                product['risk_level'] = cell_text
                            elif '成立日' in header:
                                product['setup_date'] = cell_text
                            elif '募集方式' in header:
                                product['offering_type'] = cell_text
                            elif '持有期' in header:
                                product['holding_period'] = cell_text
                            elif '开放日' in header:
                                product['next_open_date'] = cell_text

                if product.get('product_code'):
                    products.append(product)

            except Exception as e:
                continue

        return products

    def _parse_risk_level(self, risk_level) -> str:
        """解析风险等级"""
        if risk_level is None:
            return ''
        risk_map = {
            1: 'PR1-低风险',
            2: 'PR2-中低风险',
            3: 'PR3-中风险',
            4: 'PR4-中高风险',
            5: 'PR5-高风险',
        }
        if isinstance(risk_level, int):
            return risk_map.get(risk_level, f'PR{risk_level}')
        return str(risk_level)

    def get_product_list(self) -> List[Dict]:
        """获取产品列表 - 使用Selenium抓取"""
        logger.info(f"[{self.bank_name}] 获取产品列表...")

        if not HAS_SELENIUM:
            logger.error(f"[{self.bank_name}] Selenium未安装")
            return []

        all_products = []

        try:
            self._setup_driver()

            logger.info(f"[{self.bank_name}] 访问页面...")
            self.driver.get(f"{self.base_url}/yymk/lccs/")
            time.sleep(5)

            # 切换到iframe
            iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
            if not iframes:
                logger.error(f"[{self.bank_name}] 未找到iframe")
                return []

            self.driver.switch_to.frame(iframes[0])
            time.sleep(3)
            # 等待 loading 遮罩消失
            self._wait_loading_gone()

            # 遍历目标分类
            for category in self.target_categories:
                logger.info(f"[{self.bank_name}] 切换到分类: {category}")

                try:
                    self._wait_loading_gone()
                    tabs = self.driver.find_elements(By.CSS_SELECTOR, '.typeCategory, .typeCategory-select')
                    for tab in tabs:
                        if category in tab.text:
                            try:
                                tab.click()
                            except Exception:
                                # loading 遮罩可能仍在，用 JS 点击绕过
                                self.driver.execute_script("arguments[0].click();", tab)
                            time.sleep(3)
                            self._wait_loading_gone()
                            break

                    logger.info(f"[{self.bank_name}] [{category}] 滚动加载产品...")
                    total = self._scroll_to_load_all()
                    logger.info(f"[{self.bank_name}] [{category}] 共加载 {total} 个产品")

                    products = self._extract_products_from_page(category)
                    all_products.extend(products)
                    logger.info(f"[{self.bank_name}] [{category}] 提取到 {len(products)} 个产品")

                except Exception as e:
                    logger.error(f"[{self.bank_name}] [{category}] 错误: {e}")

            logger.info(f"[{self.bank_name}] 共获取 {len(all_products)} 个产品")

        except Exception as e:
            logger.error(f"[{self.bank_name}] 获取产品列表失败: {e}")

        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

        return all_products

    def get_product_detail(self, product_code: str, prod_type: int = 2,
                           session: requests.Session = None) -> Dict:
        """获取产品详情（含费用信息）

        Args:
            product_code: 产品代码
            prod_type: 产品类型 (2=每日购, 3=定开)
            session: 请求会话
        """
        if session is None:
            session = self._get_session()
        url = f"{self.API_BASE}/getTAProductDetail"
        params = {'prodCode': product_code, 'prodType': prod_type}
        try:
            resp = session.get(url, params=params, timeout=30, verify=False)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"[{self.bank_name}] 获取产品详情失败 {product_code}: {e}")
        return {}

    def _extract_fee_info(self, detail_data: Dict, product_name: str = '') -> Optional[Dict]:
        """从getTAProductDetail响应提取赎回费信息

        Args:
            detail_data: getTAProductDetail API返回的数据
            product_name: 产品名称(fallback解析)

        Returns:
            dict: ProductFeeInfo 或 None
        """
        from redemption_fee_db import parse_fee_from_name

        # 从API详情中提取费用字段
        if detail_data:
            data_obj = detail_data.get('data', detail_data)
            if isinstance(data_obj, dict):
                # 搜索常见费用字段名
                for fee_key in ['redeemFee', 'redemptionFee', 'redeemFeeRate',
                                'sellFee', 'backFee', 'feeRate']:
                    fee_val = data_obj.get(fee_key)
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

                # 搜索描述字段
                for desc_key in ['productDesc', 'prodDesc', 'remark',
                                 'feeDesc', 'redeemDesc', 'productName']:
                    desc_val = data_obj.get(desc_key, '')
                    if isinstance(desc_val, str) and '赎回费' in desc_val:
                        parsed = parse_fee_from_name(desc_val)
                        if parsed:
                            parsed['source'] = 'api_detail'
                            return parsed

        # Fallback: 从产品名称解析 (中信产品名常含 "(180天内收取赎回费)")
        if product_name:
            return parse_fee_from_name(product_name)

        return None

    def get_nav_history(self, product_code: str, session: requests.Session = None,
                        full_history: bool = False, max_months: int = 0) -> List[Dict]:
        """获取产品历史净值

        Args:
            product_code: 产品代码
            session: 请求会话
            full_history: True=尽量获取全部历史(多次请求1/3/6/12月), False=仅近1月
            max_months: >0时覆盖默认区间，例如24=请求24个月(2年)历史
                       注意: queryUnit 仅支持1/3，超过3个月使用 startDate/endDate 方式
        """
        if session is None:
            session = self._get_session()

        url = f"{self.API_BASE}/getTAProductNav"
        all_nav_dict = {}  # date -> nav_item, 用于去重

        # max_months > 3 时使用 startDate/endDate 方式（queryUnit仅支持1/3）
        if max_months > 3:
            from dateutil.relativedelta import relativedelta
            end_date = datetime.now()
            start_date = end_date - relativedelta(months=max_months)
            params = {
                'prodCode': product_code,
                'startDate': start_date.strftime('%Y%m%d'),
                'endDate': end_date.strftime('%Y%m%d'),
            }
            try:
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('code') == '0000' and data.get('data'):
                        nav_list = data['data'].get('productNavList', [])
                        for item in nav_list:
                            nav_date = item.get('navDate', '')
                            if nav_date and len(nav_date) == 8:
                                nav_date = f"{nav_date[:4]}-{nav_date[4:6]}-{nav_date[6:8]}"
                            nav_value = item.get('nav')
                            total_nav = item.get('totalNav')
                            if nav_date and nav_value is not None:
                                all_nav_dict[nav_date] = {
                                    'date': nav_date,
                                    'unit_nav': nav_value,
                                    'total_nav': total_nav if total_nav is not None else nav_value,
                                }
            except Exception as e:
                logger.debug(f"[{self.bank_name}] 获取历史净值失败 {product_code} (date_range): {e}")

            return list(all_nav_dict.values())

        # queryUnit 模式（1/3个月）
        if full_history:
            query_units = [3, 1]
        else:
            query_units = [1]

        for unit in query_units:
            params = {
                'prodCode': product_code,
                'queryUnit': unit,
            }

            try:
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    continue

                data = resp.json()
                if data.get('code') != '0000':
                    continue

                result = data.get('data', {})
                if not result:
                    continue
                nav_list = result.get('productNavList', [])

                for item in nav_list:
                    nav_date = item.get('navDate', '')
                    if nav_date and len(nav_date) == 8:
                        nav_date = f"{nav_date[:4]}-{nav_date[4:6]}-{nav_date[6:8]}"

                    nav_value = item.get('nav')
                    total_nav = item.get('totalNav')

                    if nav_date and nav_value is not None:
                        if nav_date not in all_nav_dict:
                            all_nav_dict[nav_date] = {
                                'date': nav_date,
                                'unit_nav': nav_value,
                                'total_nav': total_nav if total_nav is not None else nav_value,
                            }

                # 如果已经拿到了最长区间的数据，后续短区间不再请求
                if nav_list:
                    break

            except Exception as e:
                logger.debug(f"[{self.bank_name}] 获取历史净值失败 {product_code} (unit={unit}): {e}")
                continue

        return list(all_nav_dict.values())

    def calculate_metrics(self, nav_data: List[Dict], product_info: Dict) -> Optional[ProductProfile]:
        """计算产品指标"""
        try:
            profile = ProductProfile(
                bank=self.bank_type,
                bank_name=self.bank_name,
                product_code=product_info.get('product_code', ''),
                product_name=product_info.get('product_name', ''),
            )

            # 设置基本信息
            profile.status = ProductStatus.OPEN
            profile.is_buyable = True
            profile.risk_level = product_info.get('risk_level', '')

            # 解析产品类型
            category = product_info.get('category', '')
            if '每日购' in category:
                profile.product_type = ProductType.HOLDING_PERIOD
            elif '定开' in category:
                profile.product_type = ProductType.SCHEDULED_OPEN
            else:
                profile.product_type = ProductType.OTHER

            # 设置净值信息
            nav = product_info.get('nav')
            if nav:
                try:
                    profile.latest_nav = float(nav)
                except:
                    profile.latest_nav = 1.0

            nav_date = product_info.get('nav_date', '')
            profile.latest_nav_date = nav_date

            # 计算距今天数
            if nav_date:
                try:
                    nav_dt = datetime.strptime(nav_date, '%Y-%m-%d')
                    profile.days_from_today = (datetime.now() - nav_dt).days
                except:
                    pass

            # 存储历史净值
            if nav_data:
                profile.nav_history = nav_data
                # 更新最新净值信息(如果历史数据更新)
                if nav_data and nav_data[0].get('date'):
                    profile.latest_nav_date = nav_data[0]['date']
                    profile.latest_nav = nav_data[0].get('unit_nav', profile.latest_nav)
            else:
                # 没有历史数据时,用当前数据
                total_nav = product_info.get('total_nav')
                total_nav_float = None
                if total_nav:
                    try:
                        total_nav_float = float(total_nav)
                    except:
                        pass

                profile.nav_history = [{
                    'date': nav_date,
                    'unit_nav': profile.latest_nav,
                    'total_nav': total_nav_float or profile.latest_nav,
                }]

            # 存储额外信息到duration_text字段
            extra_info = []
            if product_info.get('min_amount'):
                extra_info.append(f"起购:{product_info.get('min_amount')}")
            profile.duration_text = '; '.join(extra_info) if extra_info else ''

            return profile

        except Exception as e:
            logger.debug(f"[{self.bank_name}] 计算指标失败: {e}")
            return None

    def crawl(self, max_workers: int = 5) -> List[ProductProfile]:
        """执行爬取 - 包含历史净值获取"""
        profiles = []

        try:
            # 获取产品列表
            products = self.get_product_list()
            logger.info(f"[{self.bank_name}] 已获取 {len(products)} 个产品")

            # 获取每个产品的历史净值并计算指标
            session = self._get_session()
            total = len(products)

            for i, product in enumerate(products):
                product_code = product.get('product_code', '')

                # 获取历史净值
                nav_history = self.get_nav_history(product_code, session)

                # 计算指标
                profile = self.calculate_metrics(nav_history, product)
                if profile:
                    profiles.append(profile)

                # 进度日志
                if (i + 1) % 100 == 0:
                    logger.info(f"[{self.bank_name}] 进度: {i+1}/{total}, 已处理: {len(profiles)}")

            logger.info(f"[{self.bank_name}] 处理完成，共 {len(profiles)} 个有效产品")

        except Exception as e:
            logger.error(f"[{self.bank_name}] 爬取失败: {e}")

        return profiles


# ============================================================================
# 任务状态管理（断点续传支持）
# ============================================================================

class TaskState:
    """任务状态管理器 - 支持断点续传"""

    STATE_FILE = "task_state.json"

    def __init__(self):
        self.state = {
            'task_id': None,
            'start_time': None,
            'last_update': None,
            'status': 'idle',  # idle, running, paused, completed
            'banks_completed': [],  # 已完成的银行列表
            'banks_pending': [],    # 待处理的银行列表
            'current_bank': None,   # 当前正在处理的银行
            'current_progress': {}, # 当前银行的进度 {total, processed, profiles_count}
            'saved_profiles': [],   # 已保存的产品数据文件
            'total_profiles': 0,    # 总产品数量
        }
        self._load_state()

    def _load_state(self):
        """加载任务状态"""
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
                    logger.info(f"[任务状态] 已加载任务状态: {self.state['status']}")
            except Exception as e:
                logger.warning(f"[任务状态] 加载状态失败: {e}")

    def _save_state(self):
        """保存任务状态"""
        self.state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[任务状态] 保存状态失败: {e}")

    def has_unfinished_task(self) -> bool:
        """检查是否有未完成的任务"""
        return self.state['status'] in ['running', 'paused']

    def get_resume_info(self) -> dict:
        """获取恢复任务的信息"""
        return {
            'task_id': self.state['task_id'],
            'banks_completed': self.state['banks_completed'],
            'banks_pending': self.state['banks_pending'],
            'current_bank': self.state['current_bank'],
            'current_progress': self.state['current_progress'],
            'total_profiles': self.state['total_profiles'],
            'start_time': self.state['start_time'],
        }

    def start_new_task(self, banks: List[str]):
        """开始新任务"""
        self.state = {
            'task_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_update': None,
            'status': 'running',
            'banks_completed': [],
            'banks_pending': banks.copy(),
            'current_bank': None,
            'current_progress': {},
            'saved_profiles': [],
            'total_profiles': 0,
        }
        self._save_state()
        logger.info(f"[任务状态] 开始新任务: {self.state['task_id']}, 银行: {banks}")

    def resume_task(self):
        """恢复任务"""
        self.state['status'] = 'running'
        self._save_state()
        logger.info(f"[任务状态] 恢复任务: {self.state['task_id']}")

    def start_bank(self, bank_name: str):
        """开始处理某银行"""
        self.state['current_bank'] = bank_name
        self.state['current_progress'] = {'total': 0, 'processed': 0, 'profiles_count': 0}
        if bank_name in self.state['banks_pending']:
            self.state['banks_pending'].remove(bank_name)
        self._save_state()
        logger.info(f"[任务状态] 开始处理银行: {bank_name}")

    def update_progress(self, total: int, processed: int, profiles_count: int):
        """更新当前银行的进度"""
        self.state['current_progress'] = {
            'total': total,
            'processed': processed,
            'profiles_count': profiles_count
        }
        # 每处理50个产品保存一次状态
        if processed % 50 == 0:
            self._save_state()

    def complete_bank(self, bank_name: str, profiles_count: int):
        """完成某银行的处理"""
        if bank_name not in self.state['banks_completed']:
            self.state['banks_completed'].append(bank_name)
        self.state['current_bank'] = None
        self.state['current_progress'] = {}
        self.state['total_profiles'] += profiles_count
        self._save_state()
        logger.info(f"[任务状态] 完成银行: {bank_name}, 产品数: {profiles_count}")

    def save_intermediate_results(self, filename: str):
        """记录中间结果文件"""
        if filename not in self.state['saved_profiles']:
            self.state['saved_profiles'].append(filename)
        self._save_state()

    def complete_task(self):
        """完成任务"""
        self.state['status'] = 'completed'
        self._save_state()
        logger.info(f"[任务状态] 任务完成: {self.state['task_id']}, 总产品数: {self.state['total_profiles']}")

    def clear_state(self):
        """清除任务状态"""
        self.state = {
            'task_id': None,
            'start_time': None,
            'last_update': None,
            'status': 'idle',
            'banks_completed': [],
            'banks_pending': [],
            'current_bank': None,
            'current_progress': {},
            'saved_profiles': [],
            'total_profiles': 0,
        }
        if os.path.exists(self.STATE_FILE):
            os.remove(self.STATE_FILE)
        logger.info("[任务状态] 已清除任务状态")


# ============================================================================
# 多银行爬虫管理器
# ============================================================================

class MultiBankCrawler:
    """多银行爬虫管理器 - 支持断点续传"""

    def __init__(self):
        self.crawlers: Dict[BankType, BaseBankCrawler] = {}
        self.all_profiles: List[ProductProfile] = []
        self.task_state = TaskState()

    def register_crawler(self, crawler: BaseBankCrawler):
        """注册银行爬虫"""
        self.crawlers[crawler.bank_type] = crawler
        logger.info(f"已注册爬虫: {crawler.bank_name}")

    def _format_nav_date(self, date_val) -> str:
        """格式化净值日期（兼容不同银行的日期格式）"""
        if not date_val:
            return ''
        date_str = str(date_val).strip()
        # 移除非数字字符
        date_str = ''.join(filter(str.isdigit, date_str))
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return str(date_val)

    def crawl_bank(self, bank_type: BankType, max_workers: int = 10) -> List[ProductProfile]:
        """爬取单个银行"""
        if bank_type not in self.crawlers:
            logger.error(f"未注册银行爬虫: {bank_type}")
            return []

        crawler = self.crawlers[bank_type]
        logger.info(f"\n{'='*60}")
        logger.info(f"开始爬取: {crawler.bank_name}")
        logger.info(f"{'='*60}")

        # 获取产品列表
        products = crawler.get_product_list()
        if not products:
            logger.warning(f"[{crawler.bank_name}] 未获取到产品")
            return []

        profiles = []
        skip_reasons = {}

        # 根据银行类型选择处理方式
        if bank_type == BankType.CMBC:
            # 民生银行：需要分批获取NAV数据
            profiles = self._crawl_with_nav(crawler, products, max_workers, skip_reasons)
        else:
            # 其他银行：直接从产品列表创建Profile
            profiles = self._crawl_direct(crawler, products, skip_reasons)

        logger.info(f"[{crawler.bank_name}] 分析完成: {len(profiles)} 个产品")
        if skip_reasons:
            for reason, count in skip_reasons.items():
                logger.info(f"  - {reason}: {count}个")

        return profiles

    def _crawl_with_nav(self, crawler: BaseBankCrawler, products: List[Dict],
                        max_workers: int, skip_reasons: Dict) -> List[ProductProfile]:
        """需要获取NAV数据的爬取方式（民生银行）"""
        batch_size = Config.SESSION_NAV_LIMIT
        batches = [products[i:i+batch_size] for i in range(0, len(products), batch_size)]

        profiles = []

        for batch_idx, batch in enumerate(batches):
            session = crawler._create_session()

            for product in batch:
                code = product.get('REAL_PRD_CODE') or product.get('PRD_CODE') or product.get('id', '')
                if not code:
                    continue

                nav_data = crawler.get_nav_history(code, session)
                if not nav_data:
                    skip_reasons['无NAV数据'] = skip_reasons.get('无NAV数据', 0) + 1
                    continue

                profile = crawler.calculate_metrics(nav_data, product)
                if not profile:
                    skip_reasons['数据不足'] = skip_reasons.get('数据不足', 0) + 1
                    continue

                profile.score = crawler.calculate_score(profile)
                profile.signal = crawler.generate_signal(profile)
                profiles.append(profile)

            logger.info(f"[{crawler.bank_name}] 批次 {batch_idx+1}/{len(batches)} 完成，累计: {len(profiles)}")
            session.close()

        return profiles

    def _crawl_direct(self, crawler: BaseBankCrawler, products: List[Dict],
                      skip_reasons: Dict) -> List[ProductProfile]:
        """直接从产品列表创建Profile（宁波银行等）- 支持NAV获取"""
        profiles = []

        # 对于宁波银行，从附件列表获取产品和NAV数据
        if isinstance(crawler, NBBCrawler):
            logger.info(f"[{crawler.bank_name}] 获取净值公告附件列表...")
            attachments = crawler.get_attachment_list()
            logger.info(f"[{crawler.bank_name}] 找到 {len(attachments)} 个产品的净值公告")

            # 从附件中提取产品并获取NAV
            for i, (code, att_list) in enumerate(attachments.items()):
                try:
                    # 下载PDF获取NAV数据
                    nav_data = []
                    att_list = sorted(att_list, key=lambda x: x['date'], reverse=True)
                    att_list = att_list[:crawler.NAV_DAYS]

                    for att in att_list:
                        nav = crawler.download_and_parse_pdf(att['url'])
                        if nav:
                            nav_data.append(nav)
                        time.sleep(0.2)

                    if nav_data:
                        # 从附件标题提取产品名称
                        product_name = att_list[0].get('title', '').split('_')[0] if att_list else code

                        # 创建虚拟产品信息
                        product_info = {
                            'projectcode': code,
                            'projectname': product_name,
                            'risklevel': 'PR2',  # 默认中低风险
                            'curstateDesc': '在售',
                        }

                        profile = crawler.calculate_metrics(nav_data, product_info)
                        if profile:
                            profile.score = crawler.calculate_score(profile)
                            profile.signal = crawler.generate_signal(profile)
                            profiles.append(profile)
                            logger.info(f"[{crawler.bank_name}] {code}: 获取到 {len(nav_data)} 天净值数据")

                    if (i + 1) % 5 == 0:
                        logger.info(f"[{crawler.bank_name}] 进度: {i+1}/{len(attachments)}, 已分析: {len(profiles)}")

                except Exception as e:
                    skip_reasons['异常'] = skip_reasons.get('异常', 0) + 1
                    logger.debug(f"[{crawler.bank_name}] 产品处理异常: {e}")

            return profiles

        # 华夏理财：从Excel文件获取NAV数据（增量更新）
        if isinstance(crawler, HuaxiaCrawler):
            logger.info(f"[{crawler.bank_name}] 开始下载解析净值Excel文件（增量更新模式）...")
            total = len(products)
            new_nav_count = 0

            for i, product in enumerate(products):
                try:
                    product_code = product.get('id', '')
                    excel_url = product.get('address', '')
                    if not excel_url or not product_code:
                        skip_reasons['无Excel地址'] = skip_reasons.get('无Excel地址', 0) + 1
                        continue

                    # 下载并解析Excel获取新NAV数据
                    new_nav_data = crawler.download_nav_excel(excel_url)

                    if new_nav_data:
                        # 合并新旧数据（增量更新）
                        merged_nav = crawler._merge_nav_data(product_code, new_nav_data)
                        new_nav_count += len(new_nav_data)

                        # 使用合并后的数据计算指标
                        profile = crawler.calculate_metrics(merged_nav, product)
                        if profile:
                            profile.score = crawler.calculate_score(profile)
                            profile.signal = crawler.generate_signal(profile)
                            profiles.append(profile)
                    else:
                        # 尝试使用已有的历史数据
                        existing_nav = crawler.nav_storage.get(product_code, [])
                        if existing_nav:
                            profile = crawler.calculate_metrics(existing_nav, product)
                            if profile:
                                profile.score = crawler.calculate_score(profile)
                                profile.signal = crawler.generate_signal(profile)
                                profiles.append(profile)
                        else:
                            skip_reasons['无NAV数据'] = skip_reasons.get('无NAV数据', 0) + 1

                    # 控制请求频率
                    time.sleep(0.1)

                    if (i + 1) % 50 == 0:
                        logger.info(f"[{crawler.bank_name}] 进度: {i+1}/{total}, 已分析: {len(profiles)}")

                except Exception as e:
                    skip_reasons['异常'] = skip_reasons.get('异常', 0) + 1
                    logger.debug(f"[{crawler.bank_name}] 产品处理异常: {e}")

            # 保存净值历史到文件
            crawler._save_nav_storage()
            logger.info(f"[{crawler.bank_name}] 本次新增 {new_nav_count} 条净值数据")

            return profiles

        # 其他银行的处理逻辑
        for i, product in enumerate(products):
            try:
                profile = crawler.calculate_metrics([], product)
                if profile:
                    profile.score = crawler.calculate_score(profile)
                    profile.signal = crawler.generate_signal(profile)
                    profiles.append(profile)
                else:
                    skip_reasons['解析失败'] = skip_reasons.get('解析失败', 0) + 1

                if (i + 1) % 10 == 0:
                    logger.info(f"[{crawler.bank_name}] 进度: {i+1}/{len(products)}, 已分析: {len(profiles)}")

            except Exception as e:
                skip_reasons['异常'] = skip_reasons.get('异常', 0) + 1

        return profiles

    def crawl_all(self, max_workers: int = 10, resume: bool = False) -> List[ProductProfile]:
        """爬取所有银行 - 支持断点续传

        Args:
            max_workers: 最大工作线程数
            resume: 是否恢复上次未完成的任务
        """
        logger.info("="*70)
        logger.info("多银行理财产品量化分析系统")
        logger.info("="*70)

        start_time = time.time()
        self.all_profiles = []

        # 获取所有注册的银行名称
        all_banks = [crawler.bank_name for crawler in self.crawlers.values()]

        # 检查是否恢复任务
        if resume and self.task_state.has_unfinished_task():
            resume_info = self.task_state.get_resume_info()
            logger.info(f"[断点续传] 恢复任务: {resume_info['task_id']}")
            logger.info(f"[断点续传] 已完成银行: {resume_info['banks_completed']}")
            logger.info(f"[断点续传] 待处理银行: {resume_info['banks_pending']}")

            # 跳过已完成的银行
            banks_to_process = [b for b in all_banks if b not in resume_info['banks_completed']]
            self.task_state.resume_task()
        else:
            banks_to_process = all_banks
            self.task_state.start_new_task(all_banks)

        # 处理每个银行
        for bank_type, crawler in self.crawlers.items():
            if crawler.bank_name not in banks_to_process:
                logger.info(f"[{crawler.bank_name}] 已完成，跳过")
                continue

            try:
                self.task_state.start_bank(crawler.bank_name)
                profiles = self.crawl_bank(bank_type, max_workers)
                self.all_profiles.extend(profiles)
                self.task_state.complete_bank(crawler.bank_name, len(profiles))

                # 每完成一个银行，保存中间结果
                if self.all_profiles:
                    intermediate_file = self._save_intermediate_results()
                    self.task_state.save_intermediate_results(intermediate_file)

            except Exception as e:
                logger.error(f"[{crawler.bank_name}] 爬取失败: {e}")
                # 发生错误时保存当前状态，以便恢复
                self.task_state._save_state()

        elapsed = time.time() - start_time
        logger.info(f"\n全部完成，耗时: {elapsed:.1f}秒")
        logger.info(f"共获取 {len(self.all_profiles)} 个产品")

        # 标记任务完成
        self.task_state.complete_task()

        return self.all_profiles

    def _save_intermediate_results(self) -> str:
        """保存中间结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'中间结果_{timestamp}.json'
        try:
            # 只保存基本信息，不保存完整的净值历史
            data = []
            for p in self.all_profiles:
                data.append({
                    'bank_name': p.bank_name,
                    'product_code': p.product_code,
                    'product_name': p.product_name,
                    'risk_level': p.risk_level,
                    'score': p.score,
                    'signal': p.signal.value if p.signal else None,
                })
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[断点续传] 中间结果已保存: {filename}")
            return filename
        except Exception as e:
            logger.error(f"[断点续传] 保存中间结果失败: {e}")
            return ""

    def check_and_resume(self) -> bool:
        """检查是否有未完成的任务需要恢复

        Returns:
            bool: 是否有未完成的任务
        """
        if self.task_state.has_unfinished_task():
            resume_info = self.task_state.get_resume_info()
            print("\n" + "="*60)
            print("  检测到未完成的任务！")
            print("="*60)
            print(f"  任务ID: {resume_info['task_id']}")
            print(f"  开始时间: {resume_info['start_time']}")
            print(f"  已完成银行: {resume_info['banks_completed']}")
            print(f"  待处理银行: {resume_info['banks_pending']}")
            print(f"  已爬取产品数: {resume_info['total_profiles']}")
            print("="*60)
            return True
        return False

    def save_results(self, profiles: List[ProductProfile] = None) -> str:
        """保存结果"""
        if profiles is None:
            profiles = self.all_profiles

        if not profiles:
            logger.warning("无数据可保存")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'多银行理财_量化分析_{timestamp}.xlsx'

        # 转换为DataFrame
        data = []
        for p in profiles:
            # 基础列（与民生理财保持一致的顺序）
            row = {
                '银行': p.bank_name,
                '产品名称': p.product_name,
                '产品代码': p.product_code,
                '风险等级': p.risk_level,
                '风险类型': p.risk_category.value,
                '产品类型': p.product_type.value,
                '可购买': '是' if p.is_buyable else '否',
                '申购状态': p.status.value,
                '持有期限': p.duration_text,
                '期限天数': p.duration_days,
                '最新净值日期': p.latest_nav_date,
                '净值天数': len(p.nav_history) if p.nav_history else 0,
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
                '业绩基准': p.benchmark
            }

            # 添加历史净值列（最多15天）- 直接用日期作为列名
            if p.nav_history:
                max_nav_cols = min(len(p.nav_history), 15)
                for i, nav in enumerate(p.nav_history[:max_nav_cols]):
                    # 兼容不同银行的字段名：华夏用date/unit_nav，民生用ISS_DATE/NAV
                    date_str = nav.get('date') or self._format_nav_date(nav.get('ISS_DATE', ''))
                    unit_nav = nav.get('unit_nav') or nav.get('NAV')
                    # 确保净值精度为4位小数
                    if unit_nav is not None:
                        unit_nav = round(float(unit_nav), 4)
                    # 用日期作为列名，净值作为值
                    if date_str:
                        row[date_str] = unit_nav

            data.append(row)

        df = pd.DataFrame(data)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 全部产品
            df.to_excel(writer, sheet_name='全部产品', index=False)

            # 按银行分组
            for bank_name in df['银行'].unique():
                bank_df = df[df['银行'] == bank_name]
                sheet_name = bank_name[:31]  # Excel sheet名称限制
                bank_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 买入信号
            buy_df = df[df['交易信号'].isin(['强烈买入', '买入'])]
            buy_df = buy_df.sort_values('综合评分', ascending=False)
            buy_df.to_excel(writer, sheet_name='买入信号', index=False)

            # 可购买持有期产品
            holding_df = df[(df['可购买'] == '是') & (df['产品类型'] == '持有期')]
            holding_df = holding_df.sort_values('综合评分', ascending=False)
            holding_df.to_excel(writer, sheet_name='可购买持有期产品', index=False)

            # 设置净值列的数字格式为4位小数
            try:
                import re
                for sheet_name in writer.sheets:
                    ws = writer.sheets[sheet_name]
                    # 找到所有净值列并设置格式（包括"最新净值"和日期列如"2026-01-15"）
                    for col_idx, col_cell in enumerate(ws[1], 1):
                        col_name = str(col_cell.value) if col_cell.value else ''
                        # 匹配：包含"净值"但不包含"日期"的列，或者日期格式的列名(YYYY-MM-DD)
                        is_nav_col = ('净值' in col_name and '日期' not in col_name)
                        is_date_col = bool(re.match(r'^\d{4}-\d{2}-\d{2}$', col_name))
                        if is_nav_col or is_date_col:
                            for row_idx in range(2, ws.max_row + 1):
                                cell = ws.cell(row=row_idx, column=col_idx)
                                if cell.value is not None and isinstance(cell.value, (int, float)):
                                    cell.number_format = '0.0000'
            except Exception as e:
                logger.debug(f"设置Excel格式失败: {e}")

        logger.info(f"结果已保存到: {filename}")
        return filename


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 支持断点续传"""
    import sys

    print("="*70)
    print("        多银行理财产品量化分析系统 v1.1")
    print("        (支持断点续传)")
    print("="*70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("支持的银行:")
    print("  - 民生银行 (完整量化分析)")
    print("  - 宁波银行 (产品列表)")
    print()

    # 创建管理器
    manager = MultiBankCrawler()

    # 注册已实现的爬虫
    manager.register_crawler(CMBCCrawler())  # 民生银行 - 完整实现
    manager.register_crawler(NBBCrawler())   # 宁波银行 - 产品列表
    # manager.register_crawler(HuaxiaCrawler())  # 华夏理财 - JS数据
    # manager.register_crawler(CITICCrawler())   # 信银理财 - 需要API

    # 检查命令行参数
    resume_mode = '--resume' in sys.argv or '-r' in sys.argv
    clear_state = '--clear' in sys.argv or '-c' in sys.argv

    # 清除状态
    if clear_state:
        manager.task_state.clear_state()
        print("已清除任务状态")
        return

    # 检查是否有未完成的任务
    if manager.check_and_resume():
        if resume_mode:
            print("\n自动恢复模式启动...")
            resume = True
        else:
            print("\n选项:")
            print("  [R] 恢复上次任务")
            print("  [N] 开始新任务")
            print("  [Q] 退出")
            choice = input("\n请选择 (R/N/Q): ").strip().upper()
            if choice == 'Q':
                print("已退出")
                return
            elif choice == 'R':
                resume = True
            else:
                resume = False
                manager.task_state.clear_state()
    else:
        resume = False

    # 爬取所有银行
    profiles = manager.crawl_all(resume=resume)

    # 保存结果
    if profiles:
        filename = manager.save_results()
        print(f"\n分析完成！文件: {filename}")

        # 显示统计
        print("\n" + "="*60)
        print("统计信息")
        print("="*60)
        bank_counts = {}
        for p in profiles:
            bank_counts[p.bank_name] = bank_counts.get(p.bank_name, 0) + 1
        for bank, count in bank_counts.items():
            print(f"  {bank}: {count} 个产品")
        print(f"  总计: {len(profiles)} 个产品")
    else:
        print("\n未获取到数据")


def auto_resume():
    """自动恢复任务（用于开机启动脚本）"""
    print("="*70)
    print("  自动恢复任务模式")
    print("="*70)

    manager = MultiBankCrawler()
    manager.register_crawler(CMBCCrawler())
    manager.register_crawler(NBBCrawler())

    if manager.check_and_resume():
        print("\n检测到未完成任务，自动恢复...")
        profiles = manager.crawl_all(resume=True)
        if profiles:
            filename = manager.save_results()
            print(f"\n恢复完成！文件: {filename}")
        return True
    else:
        print("\n没有未完成的任务")
        return False


if __name__ == "__main__":
    main()
