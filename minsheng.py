"""
民生理财产品数据采集与量化分析系统
网站：http://www.cmbcwm.com.cn/grlc/index.htm

功能：
1. 全维度产品画像采集（申购状态、持有期限、风险等级等）
2. 量化指标计算（近1月/3月年化、最大回撤、夏普比率、卡玛比率）
3. 多因子评分模型（固收类/权益类分别评分）
4. T+0 动态信号生成系统
5. 核心-卫星投资组合构建

版本：3.0 - 专业量化版
日期：2026-01-16
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import io
import logging
import json
import schedule
import ssl
import urllib3
import re
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

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
        logging.FileHandler('cmbc_crawler.log', encoding='utf-8'),
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
    HOLDING_PERIOD = "持有期"  # 持有期产品，随时可购买
    SCHEDULED_OPEN = "定开"    # 定期开放，需等待开放期
    CLOSED_END = "封闭式"      # 封闭式产品
    CASH_MGMT = "现金管理"     # 现金管理类（T+0或T+1）
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
    # 收益指标
    return_1w: Optional[float] = None  # 近1周年化
    return_1m: Optional[float] = None  # 近1月年化
    return_3m: Optional[float] = None  # 近3月年化
    return_6m: Optional[float] = None  # 近6月年化

    # 风险指标
    volatility: Optional[float] = None  # 年化波动率
    max_drawdown: Optional[float] = None  # 最大回撤

    # 风险调整收益
    sharpe_ratio: Optional[float] = None  # 夏普比率
    calmar_ratio: Optional[float] = None  # 卡玛比率

    # 超额收益（相对无风险利率）
    excess_return_1m: Optional[float] = None


@dataclass
class ProductProfile:
    """产品全维度画像"""
    # 基础信息
    product_code: str = ""
    product_name: str = ""
    risk_level: str = ""  # R1-R5
    risk_category: RiskCategory = RiskCategory.FIXED_INCOME

    # 产品类型
    product_type: ProductType = ProductType.OTHER

    # 申购状态（一票否决指标）
    status: ProductStatus = ProductStatus.UNKNOWN
    status_text: str = ""
    is_buyable: bool = False  # 是否当前可购买

    # 期限信息
    duration_days: Optional[int] = None  # 持有期限（天）
    duration_text: str = ""  # 原始文本

    # 业绩基准
    benchmark: str = ""
    benchmark_value: Optional[float] = None  # 解析出的基准值

    # 净值信息
    latest_nav: float = 1.0
    latest_nav_date: str = ""
    days_from_today: int = 0

    # 量化指标
    metrics: QuantMetrics = field(default_factory=QuantMetrics)

    # 评分
    score: float = 0.0
    signal: SignalType = SignalType.WATCH

    # 历史净值数据（用于计算）
    nav_history: List[Dict] = field(default_factory=list)


# ============================================================================
# 配置参数
# ============================================================================

class Config:
    """系统配置"""
    # 无风险利率（用于夏普比率计算，当前约1.5%）
    RISK_FREE_RATE = 1.5

    # 评分权重 - 固收类 (R1/R2/R3)
    FIXED_INCOME_WEIGHTS = {
        'return_3m': 0.6,   # 近3月年化权重
        'return_1w': 0.3,   # 近1周年化权重（动量因子）
        'volatility': -0.1  # 波动率惩罚
    }

    # 评分权重 - 权益类 (R4/R5)
    EQUITY_WEIGHTS = {
        'sharpe_ratio': 0.5,      # 夏普比率
        'calmar_ratio': 0.3,      # 卡玛比率
        'excess_return_1m': 0.2   # 近1月超额收益
    }

    # 期限分类阈值（天）
    DURATION_THRESHOLDS = {
        'ultra_short': 7,      # 超短期
        'short': 30,           # 短期
        'medium': 180,         # 中期
        'long': 365            # 长期
    }

    # 组合配置
    CORE_POSITION_RATIO = (0.6, 0.8)  # 核心仓位比例范围
    SATELLITE_POSITION_RATIO = (0.2, 0.4)  # 卫星仓位比例范围

    # 止损线（净值）
    STOP_LOSS_THRESHOLD = 0.95

    # 周末避险规则
    WEEKEND_AVOID_DAYS = [3, 4, 5, 6]  # 周四15点后至周日不申购（0=周一）


# ============================================================================
# SSL适配器
# ============================================================================

class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器，支持旧版SSL协商"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


# ============================================================================
# 量化计算引擎
# ============================================================================

class QuantEngine:
    """量化计算引擎"""

    @staticmethod
    def calculate_annualized_return(nav_series: List[float], days: int) -> Optional[float]:
        """
        计算年化收益率
        nav_series: 净值序列（从新到旧）
        days: 实际自然日天数
        """
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
        """
        计算年化波动率
        daily_returns: 日收益率序列（百分比）
        """
        if len(daily_returns) < 5:
            return None
        try:
            std = np.std(daily_returns, ddof=1)
            annualized_vol = std * np.sqrt(252)  # 年化
            return round(annualized_vol, 2)
        except:
            return None

    @staticmethod
    def calculate_max_drawdown(nav_series: List[float]) -> Optional[float]:
        """
        计算最大回撤
        nav_series: 净值序列（从新到旧，需要反转为时间顺序）
        """
        if len(nav_series) < 2:
            return None
        try:
            # 反转为时间正序
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
        """
        计算夏普比率
        sharpe = (年化收益 - 无风险利率) / 年化波动率
        """
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
        """
        计算卡玛比率
        calmar = 年化收益 / 最大回撤
        """
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

class CMBCWealthCrawler:
    """民生理财产品爬虫 - 专业量化版"""

    # 会话级NAV查询限制（银行API限制约200次/会话）
    SESSION_NAV_LIMIT = 150  # 保守设置为150，留有余量

    def __init__(self):
        """初始化爬虫"""
        self.base_url = "https://www.cmbcwm.com.cn"
        self.api_base = "https://www.cmbcwm.com.cn/gw/po_web"
        self.session = self._create_session()

        # 量化引擎
        self.quant = QuantEngine()

        # 历史数据缓存（用于检测信号突变）
        self.historical_scores: Dict[str, List[float]] = {}

        # NAV查询计数器（用于会话轮换）
        self._nav_query_count = 0
        self._session_lock = None  # 将在并发时初始化

    def _create_session(self) -> requests.Session:
        """创建新的HTTP会话"""
        session = requests.Session()

        # 安装自定义SSL适配器
        adapter = LegacySSLAdapter(pool_connections=30, pool_maxsize=30)
        session.mount('https://', adapter)

        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://www.cmbcwm.com.cn',
            'Referer': 'https://www.cmbcwm.com.cn/grlc/index.htm'
        })
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        return session

    def _rotate_session_if_needed(self):
        """检查是否需要轮换会话（突破API限制）"""
        self._nav_query_count += 1
        if self._nav_query_count >= self.SESSION_NAV_LIMIT:
            logger.debug(f"已达到 {self.SESSION_NAV_LIMIT} 次NAV查询，轮换会话...")
            self.session.close()
            self.session = self._create_session()
            self._nav_query_count = 0
            time.sleep(0.5)  # 短暂延迟确保新会话生效

    # ========================================================================
    # 数据采集
    # ========================================================================

    def _warmup_session(self):
        """访问首页获取cookies，防止API因无cookie而拒绝服务"""
        try:
            resp = self.session.get(
                f"{self.base_url}/grlc/index.htm", timeout=15, allow_redirects=True)
            logger.info(f"会话预热: status={resp.status_code}, cookies={len(self.session.cookies)}")
        except Exception as e:
            logger.warning(f"会话预热失败: {e}")

    def get_product_list(self, page_size: int = 100, max_retries: int = 3) -> List[Dict]:
        """获取产品列表（全量采集）"""
        logger.info("开始获取产品列表...")

        # 预热：先访问首页获取cookies
        self._warmup_session()

        all_products = []
        page = 1
        total_pages = None
        consecutive_failures = 0
        empty_responses = 0  # 追踪空响应次数

        while True:
            success = False
            for retry in range(max_retries):
                try:
                    url = f"{self.api_base}/BTAProductQry"
                    payload = {
                        "pageNo": page,
                        "pageSize": page_size,
                        "code_or_name": ""
                    }

                    response = self.session.post(url, data=payload, timeout=30)
                    response.raise_for_status()
                    data = response.json()

                    if data.get('list'):
                        products = data['list']
                        all_products.extend(products)

                        total_size = data.get('totalSize', 0)
                        if total_pages is None:
                            total_pages = (total_size + page_size - 1) // page_size
                            logger.info(f"共 {total_size} 款产品，{total_pages} 页")

                        logger.info(f"获取第 {page}/{total_pages} 页，本页 {len(products)} 个产品")
                        success = True
                        consecutive_failures = 0
                        empty_responses = 0
                        break
                    else:
                        # API返回了有效响应但没有数据
                        resp_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
                        logger.warning(f"第{page}页API返回空数据 (retry {retry+1}/{max_retries}), "
                                       f"响应字段: {resp_keys}")
                        if retry < max_retries - 1:
                            # 轮换会话重试
                            self.session.close()
                            self.session = self._create_session()
                            time.sleep(3 * (retry + 1))
                            continue
                        empty_responses += 1
                        break

                except Exception as e:
                    logger.warning(f"第{page}页请求异常 (retry {retry+1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        time.sleep(2 * (retry + 1))
                    else:
                        consecutive_failures += 1

            if not success:
                # 第1页就没有数据，说明API不可用，无需继续翻页
                if page == 1 and empty_responses > 0:
                    logger.error(f"首页即无数据，API可能不可用，停止获取")
                    break
                if consecutive_failures >= 3 or empty_responses >= 2:
                    break
                page += 1
                if total_pages and page > total_pages:
                    break
                continue

            if total_pages and page >= total_pages:
                break

            page += 1
            time.sleep(0.3)

        logger.info(f"共获取 {len(all_products)} 个产品")
        return all_products

    def get_nav_history(self, product_code: str, days: int = 180, max_retries: int = 2,
                        session: requests.Session = None,
                        full_history: bool = False) -> List[Dict]:
        """
        获取产品历史净值

        Args:
            days: 增量模式请求天数（默认180天）
            full_history: True=请求最大历史(1095天/3年), False=使用days参数
            session: 可选，指定使用的会话（用于批量处理时避免会话冲突）
        """
        use_session = session or self.session
        request_days = 1095 if full_history else days  # 完整模式请求3年数据

        for retry in range(max_retries):
            try:
                url = f"{self.api_base}/BTADailyQry"
                payload = {
                    "chart_type": 1,
                    "real_prd_code": product_code,
                    "begin_date": "",
                    "end_date": "",
                    "pageNo": 1,
                    "pageSize": request_days + 30  # 多取一些确保数据完整
                }

                response = use_session.post(url, data=payload, timeout=20)
                response.raise_for_status()
                data = response.json()

                nav_list = data.get('btaDailyAddFieldList') or data.get('list') or []
                return nav_list

            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(1)
        return []

    def get_product_detail(self, product_code: str) -> Dict:
        """获取产品详情（申购状态、持有期限等）"""
        try:
            url = f"{self.api_base}/BTAProductDetail"
            payload = {"prd_code": product_code}

            response = self.session.post(url, data=payload, timeout=15)
            response.raise_for_status()
            return response.json()
        except:
            return {}

    def _extract_fee_info(self, product_data: Dict, detail_data: Dict,
                         session: requests.Session = None) -> Optional[Dict]:
        """从发行公告 PDF 或 API 响应中提取赎回费信息

        优先使用 PDF 公告解析，失败后回退到旧逻辑 (API字段 / 名称解析)。

        Args:
            product_data: 产品列表中的原始数据
            detail_data: BTAProductDetail API返回的详情数据
            session: HTTP 会话 (用于 PDF 下载)

        Returns:
            dict: ProductFeeInfo 或 None
        """
        from redemption_fee_db import parse_fee_from_name

        # ── 新策略: 尝试从发行公告 PDF 提取赎回费 ──
        product_code = (product_data.get('REAL_PRD_CODE')
                        or product_data.get('PRD_CODE', ''))
        if product_code and HAS_PDFPLUMBER:
            try:
                fee_data = self.scrape_product_fees(product_code, session)
                if fee_data and fee_data.get('base_fees'):
                    redemption = fee_data['base_fees'].get('赎回费率')
                    if redemption is not None:
                        rate = redemption.get('rate', 0.0)
                        if rate > 0:
                            return {
                                'has_redemption_fee': True,
                                'fee_schedule': [
                                    {'min_days': 0, 'max_days': 7, 'fee_rate': rate},
                                    {'min_days': 7, 'max_days': 999999, 'fee_rate': 0.0},
                                ],
                                'fee_description': f'发行公告赎回费率{rate*100:.4f}%',
                                'source': 'prospectus',
                            }
                        else:
                            return {
                                'has_redemption_fee': False,
                                'fee_schedule': [],
                                'fee_description': '发行公告赎回费率0%',
                                'source': 'prospectus',
                            }
            except Exception as e:
                logger.debug(f"PDF 费率提取失败 ({product_code}), 回退旧逻辑: {e}")

        # ── 旧策略 Fallback: API 字段 + 名称解析 ──
        if detail_data:
            data_obj = detail_data.get('data', detail_data)
            if isinstance(data_obj, dict):
                for fee_key in ['redeemFee', 'redeem_fee', 'SH_FEE', 'REDEEM_FEE',
                                'redemptionFee', 'fee_rate', 'feeRate', 'sellFee',
                                'SELL_FEE', 'backFee', 'BACK_FEE']:
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

                for desc_key in ['PRD_DESC', 'prd_desc', 'feeDesc', 'fee_desc',
                                 'redeemDesc', 'productDesc', 'remark']:
                    desc_val = data_obj.get(desc_key)
                    if isinstance(desc_val, str) and '赎回费' in desc_val:
                        parsed = parse_fee_from_name(desc_val)
                        if parsed:
                            parsed['source'] = 'api_detail'
                            return parsed

        # Fallback: 从产品名称解析
        name = product_data.get('PRD_NAME', '')
        if name:
            return parse_fee_from_name(name)

        return None

    # ========================================================================
    # 费率公告 PDF 采集与解析
    # ========================================================================

    def get_fee_announcements(self, product_code: str,
                              session: requests.Session = None) -> Dict:
        """获取产品的费率相关公告列表

        调用 BTAFileQry API:
        - SAMJ_TYPE=1: 发行公告 (基础费率)
        - SAMJ_TYPE=4: 其他公告 (筛选"费率优惠")

        Returns:
            {
                'issuance': {'date': str, 'filename': str, 'url': str} | None,
                'discount': {'date': str, 'filename': str, 'url': str} | None,
            }
        """
        use_session = session or self.session
        result = {'issuance': None, 'discount': None}

        for samj_type, key, filter_kw in [
            (1, 'issuance', None),
            (4, 'discount', '费率优惠'),
        ]:
            try:
                url = f"{self.api_base}/BTAFileQry"
                payload = {
                    'SAMJ_TYPE': samj_type,
                    'real_prd_code': product_code,
                    'pageNo': 1,
                    'pageSize': 20,
                }
                resp = use_session.post(url, data=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()

                new_url = data.get('newUrl', '')
                old_url = data.get('oldUrl', '')
                items = data.get('list', [])

                if not items:
                    continue

                # 筛选并按日期降序排列
                candidates = []
                for item in items:
                    name = item.get('K_INFNAME', '')
                    if filter_kw and filter_kw not in name:
                        continue
                    filename = item.get('K_FILENAME', '')
                    date = item.get('BUSINESS_DATE', '')
                    urlflag = str(item.get('URLFLAG', '0'))
                    base = new_url if urlflag == '1' else old_url
                    pdf_url = base + filename if base and filename else ''
                    candidates.append({
                        'date': date,
                        'filename': filename,
                        'url': pdf_url,
                        'name': name,
                    })

                if candidates:
                    candidates.sort(key=lambda x: x.get('date', ''), reverse=True)
                    result[key] = candidates[0]

            except Exception as e:
                logger.debug(f"获取公告失败 (type={samj_type}, code={product_code}): {e}")

        return result

    # PDF本地存储目录
    PDF_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'pdf_archive', 'minsheng')

    def _download_pdf(self, url: str,
                      session: requests.Session = None,
                      save_local: bool = True) -> Optional[bytes]:
        """下载 PDF 文件，保存到本地，并返回字节内容

        Args:
            url: PDF 完整 URL
            session: HTTP 会话
            save_local: 是否保存到本地 (默认True)

        Returns:
            bytes | None
        """
        if not url:
            return None

        # 检查本地缓存
        local_path = self._get_pdf_local_path(url)
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    content = f.read()
                if len(content) >= 500:
                    logger.debug(f"PDF 从本地缓存读取: {local_path}")
                    return content
            except Exception:
                pass

        # 下载
        use_session = session or self.session
        for retry in range(3):
            try:
                resp = use_session.get(url, timeout=30)
                resp.raise_for_status()
                if len(resp.content) < 500:
                    logger.debug(f"PDF 过小({len(resp.content)}B), 可能无效: {url}")
                    return None

                # 保存到本地
                if save_local and local_path:
                    self._save_pdf_local(local_path, resp.content)

                return resp.content
            except Exception as e:
                if retry < 2:
                    time.sleep(1 * (retry + 1))
                else:
                    logger.debug(f"PDF 下载失败 ({url}): {e}")
        return None

    def _get_pdf_local_path(self, url: str) -> Optional[str]:
        """根据URL生成本地存储路径

        目录结构: pdf_archive/minsheng/{产品代码}/{文件名}
        """
        if not url:
            return None
        try:
            # 从URL提取文件名
            parsed = urlparse(url)
            filename = unquote(os.path.basename(parsed.path))
            if not filename.endswith('.pdf'):
                filename += '.pdf'

            # 尝试从文件名提取产品代码 (如 FBAE48303发行公告20250213.pdf)
            code_match = re.match(r'^([A-Z0-9]+)', filename)
            product_code = code_match.group(1) if code_match else 'unknown'

            subdir = os.path.join(self.PDF_STORAGE_DIR, product_code)
            return os.path.join(subdir, filename)
        except Exception:
            return None

    def _save_pdf_local(self, local_path: str, content: bytes):
        """保存PDF到本地"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(content)
            logger.debug(f"PDF 已保存: {local_path}")
        except Exception as e:
            logger.debug(f"PDF 保存失败 ({local_path}): {e}")

    def _parse_issuance_fee_table(self, pdf_bytes: bytes) -> Dict:
        """从发行公告 PDF 解析基础费率

        支持两种格式:
        1. 表格格式 (较新产品): [产品子代码, 固定管理费率, 托管费率, 销售费率, 认购/申购费率, 赎回费率]
        2. 文本格式 (较老产品): "固定管理费率: 0.1%/年" 等文本行

        Returns:
            {fee_type: {rate: float, unit: str}}  空 dict 表示解析失败
        """
        if not HAS_PDFPLUMBER or not pdf_bytes:
            return {}

        result = {}

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_text = ''
                all_tables = []
                for page in pdf.pages:
                    text = page.extract_text() or ''
                    all_text += text + '\n'
                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)

                # 策略1: 表格提取
                if all_tables:
                    result = self._parse_fee_from_tables(all_tables)

                # 策略2: 文本提取 (fallback)
                if not result and all_text:
                    result = self._parse_fee_from_text(all_text)

        except Exception as e:
            logger.debug(f"发行公告 PDF 解析失败: {e}")

        return result

    def _parse_fee_from_tables(self, tables: List) -> Dict:
        """从表格中提取费率

        表格常见列: [产品子代码, 固定管理费率, 托管费率, 销售费率, 认购/申购费率, 赎回费率]
        """
        result = {}
        # 目标列名映射
        col_mapping = {
            '固定管理费率': '固定管理费率',
            '管理费率': '固定管理费率',
            '管理费': '固定管理费率',
            '托管费率': '托管费率',
            '托管费': '托管费率',
            '销售费率': '销售费率',
            '销售服务费率': '销售费率',
            '销售服务费': '销售费率',
            '销售费': '销售费率',
            '认购费率': '认购申购费率',
            '申购费率': '认购申购费率',
            '认购/申购费率': '认购申购费率',
            '认购/申购费': '认购申购费率',
            '赎回费率': '赎回费率',
            '赎回费': '赎回费率',
        }

        for table in tables:
            if not table or len(table) < 2:
                continue

            # 找到表头行
            header_row = None
            header_idx = -1
            for i, row in enumerate(table):
                if not row:
                    continue
                row_text = ' '.join(str(c or '') for c in row)
                # 表头至少包含"费率"关键词
                if '费率' in row_text or '费' in row_text:
                    fee_col_count = sum(1 for c in row
                                       if c and any(k in str(c) for k in ['管理费', '托管费', '销售费', '赎回费', '申购费']))
                    if fee_col_count >= 2:
                        header_row = row
                        header_idx = i
                        break

            if header_row is None or header_idx >= len(table) - 1:
                continue

            # 建立列索引
            col_indices = {}
            for j, cell in enumerate(header_row):
                cell_str = str(cell or '').strip().replace('\n', '')
                for key, mapped in col_mapping.items():
                    if key in cell_str:
                        col_indices[mapped] = j
                        break

            if len(col_indices) < 2:
                continue

            # 从数据行提取费率 (取第一个数据行)
            for data_row in table[header_idx + 1:]:
                if not data_row:
                    continue
                row_has_data = False
                for fee_type, col_idx in col_indices.items():
                    if col_idx < len(data_row):
                        cell_val = str(data_row[col_idx] or '').strip()
                        parsed = self._parse_rate_value(cell_val)
                        if parsed is not None:
                            result[fee_type] = parsed
                            row_has_data = True
                if row_has_data:
                    break  # 第一个有效数据行即可

            if result:
                break  # 已从某个表格提取到数据

        return result

    def _parse_fee_from_text(self, text: str) -> Dict:
        """从文本中提取费率

        匹配模式: "固定管理费率: 0.10%/年" 或 "固定管理费率为年化0.10%"
        """
        result = {}

        patterns = [
            (r'固定管理费率[：:为\s]*(?:年化\s*)?([\d.]+)\s*%\s*/?\s*(年|笔)?', '固定管理费率'),
            (r'管理费率[：:为\s]*(?:年化\s*)?([\d.]+)\s*%\s*/?\s*(年|笔)?', '固定管理费率'),
            (r'托管费率[：:为\s]*(?:年化\s*)?([\d.]+)\s*%\s*/?\s*(年|笔)?', '托管费率'),
            (r'销售(?:服务)?费率[：:为\s]*(?:年化\s*)?([\d.]+)\s*%\s*/?\s*(年|笔)?', '销售费率'),
            (r'(?:认购|申购)[/／]?(?:认购|申购)?费率[：:为\s]*([\d.]+)\s*%\s*/?\s*(年|笔)?', '认购申购费率'),
            (r'赎回费率[：:为\s]*([\d.]+)\s*%\s*/?\s*(年|笔)?', '赎回费率'),
        ]

        for pattern, fee_type in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    rate_pct = float(match.group(1))  # 百分比值 如 0.10 表示 0.10%
                    rate = rate_pct / 100  # 转为小数 0.001
                    unit = match.group(2) if match.lastindex >= 2 and match.group(2) else '年'
                    result[fee_type] = {'rate': rate, 'unit': unit}
                except (ValueError, IndexError):
                    continue

        return result

    @staticmethod
    def _parse_rate_value(cell_val: str) -> Optional[Dict]:
        """解析单个费率单元格值

        费率表中的纯数字一律视为百分比值 (表头标注 "%/年" 或 "%"):
        - "0.10%/年" → {rate: 0.001, unit: "年"}
        - "0.10"     → 0.10% → {rate: 0.001, unit: "年"}
        - "0.00"     → 0% → {rate: 0.0, unit: "笔"}

        Returns:
            {rate: float, unit: str} | None
        """
        if not cell_val:
            return None

        cell_val = cell_val.strip().replace('\n', '').replace(' ', '')

        # 匹配百分比格式: "0.10%/年"
        m = re.match(r'^([\d.]+)\s*%\s*/?\s*(年|笔)?$', cell_val)
        if m:
            rate_pct = float(m.group(1))
            rate = rate_pct / 100
            unit = m.group(2) or ('笔' if rate == 0 else '年')
            return {'rate': rate, 'unit': unit}

        # 纯数字: 费率表中的数字一律为百分比值 (如 "0.10" = 0.10%)
        m = re.match(r'^([\d.]+)$', cell_val)
        if m:
            rate_pct = float(m.group(1))
            rate = rate_pct / 100
            unit = '笔' if rate == 0 else '年'
            return {'rate': rate, 'unit': unit}

        return None

    def _parse_discount_announcement(self, pdf_bytes: bytes) -> List[Dict]:
        """从费率优惠公告 PDF 解析优惠信息

        表格格式: [费率明细, 产品代码, 合同费率, 优惠后费率, 优惠起始时间, 优惠截止时间]

        Returns:
            [{fee_type, original_rate, discounted_rate, effective_from, effective_until}]
        """
        if not HAS_PDFPLUMBER or not pdf_bytes:
            return []

        results = []

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_tables = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)

                for table in all_tables:
                    if not table or len(table) < 2:
                        continue

                    # 找表头
                    header_idx = -1
                    col_fee_detail = -1
                    col_contract = -1
                    col_discount = -1
                    col_start = -1
                    col_end = -1

                    for i, row in enumerate(table):
                        if not row:
                            continue
                        row_text = ' '.join(str(c or '') for c in row)
                        if '费率' in row_text and ('优惠' in row_text or '合同' in row_text):
                            for j, cell in enumerate(row):
                                c = str(cell or '').strip()
                                if '费率明细' in c or '费率类型' in c or '费用类型' in c:
                                    col_fee_detail = j
                                elif '合同费率' in c or '原费率' in c:
                                    col_contract = j
                                elif '优惠后' in c or '优惠费率' in c or '调整后' in c:
                                    col_discount = j
                                elif '起始' in c or '开始' in c or '起' == c:
                                    col_start = j
                                elif '截止' in c or '结束' in c or '止' == c:
                                    col_end = j
                            header_idx = i
                            break

                    if header_idx < 0:
                        continue

                    # 解析数据行
                    for data_row in table[header_idx + 1:]:
                        if not data_row:
                            continue

                        fee_type = ''
                        original_rate = None
                        discounted_rate = None
                        eff_from = None
                        eff_until = None

                        if col_fee_detail >= 0 and col_fee_detail < len(data_row):
                            fee_type = str(data_row[col_fee_detail] or '').strip()
                        if col_contract >= 0 and col_contract < len(data_row):
                            original_rate = self._parse_pct_to_float(
                                str(data_row[col_contract] or ''))
                        if col_discount >= 0 and col_discount < len(data_row):
                            discounted_rate = self._parse_pct_to_float(
                                str(data_row[col_discount] or ''))
                        if col_start >= 0 and col_start < len(data_row):
                            eff_from = self._parse_chinese_date(
                                str(data_row[col_start] or ''))
                        if col_end >= 0 and col_end < len(data_row):
                            eff_until = self._parse_chinese_date(
                                str(data_row[col_end] or ''))

                        if fee_type and discounted_rate is not None:
                            results.append({
                                'fee_type': fee_type,
                                'original_rate': original_rate,
                                'discounted_rate': discounted_rate,
                                'effective_from': eff_from,
                                'effective_until': eff_until,
                            })

                    if results:
                        break  # 已从某个表格提取到数据

        except Exception as e:
            logger.debug(f"费率优惠公告 PDF 解析失败: {e}")

        return results

    @staticmethod
    def _parse_pct_to_float(text: str) -> Optional[float]:
        """将百分比文本转为小数: "0.50%" → 0.005"""
        if not text:
            return None
        text = text.strip().replace('\n', '').replace(' ', '')
        m = re.search(r'([\d.]+)\s*%', text)
        if m:
            return float(m.group(1)) / 100
        # 纯数字: 判断是否已是小数
        m = re.match(r'^([\d.]+)$', text)
        if m:
            val = float(m.group(1))
            return val if val < 1 else val / 100
        return None

    @staticmethod
    def _parse_chinese_date(text: str) -> Optional[str]:
        """将中文日期解析为 YYYYMMDD 格式

        处理: "2025年11月1日（含）" → "20251101"
              "2025/11/01" → "20251101"
              "另行公告" → None
        """
        if not text:
            return None
        text = text.strip()

        if '另行公告' in text or '待定' in text or '长期' in text:
            return None

        # "2025年11月1日"
        m = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text)
        if m:
            return f"{int(m.group(1)):04d}{int(m.group(2)):02d}{int(m.group(3)):02d}"

        # "2025/11/01" 或 "2025-11-01"
        m = re.search(r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', text)
        if m:
            return f"{int(m.group(1)):04d}{int(m.group(2)):02d}{int(m.group(3)):02d}"

        # 纯数字 "20251101"
        digits = re.sub(r'\D', '', text)
        if len(digits) == 8:
            return digits

        return None

    def scrape_product_fees(self, product_code: str,
                            session: requests.Session = None) -> Optional[Dict]:
        """高层方法: 采集单个产品的完整费率数据

        流程: 获取公告列表 → 下载 PDF → 解析 → 返回结构化数据

        Returns:
            {
                'product_code': str,
                'base_fees': {fee_type: {rate, unit}},
                'base_source': '发行公告',
                'base_announcement_date': str,
                'discounts': [{fee_type, original_rate, discounted_rate, effective_from, effective_until}],
                'discount_announcement_date': str,
            }
            失败时返回 None
        """
        use_session = session or self.session

        # 1. 获取公告列表
        announcements = self.get_fee_announcements(product_code, use_session)

        result = {'product_code': product_code}
        has_data = False

        # 2. 解析发行公告 (基础费率)
        issuance = announcements.get('issuance')
        if issuance and issuance.get('url'):
            pdf_bytes = self._download_pdf(issuance['url'], use_session)
            if pdf_bytes:
                base_fees = self._parse_issuance_fee_table(pdf_bytes)
                if base_fees:
                    result['base_fees'] = base_fees
                    result['base_source'] = '发行公告'
                    result['base_announcement_date'] = issuance.get('date', '')
                    has_data = True

        # 3. 解析费率优惠公告
        discount = announcements.get('discount')
        if discount and discount.get('url'):
            pdf_bytes = self._download_pdf(discount['url'], use_session)
            if pdf_bytes:
                discounts = self._parse_discount_announcement(pdf_bytes)
                if discounts:
                    result['discounts'] = discounts
                    result['discount_announcement_date'] = discount.get('date', '')
                    has_data = True

        return result if has_data else None

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
        date_str = ''.join(filter(str.isdigit, date_str))
        if len(date_str) == 8:
            if full:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                return f"{date_str[4:6]}-{date_str[6:8]}"
        return str(date_str)

    def parse_status(self, product_data: Dict, detail_data: Dict) -> Tuple[ProductStatus, str]:
        """
        解析申购状态
        关键字段: STATUS + END_DATE (民生银行API)

        经验证：
        - STATUS=4 + END_DATE=20991231 = 永续产品，正常运作中，可购买
        - STATUS=0 + END_DATE=20991231 = 永续产品，正常运作中，可购买
        - STATUS=4 + 具体END_DATE = 有期限产品，需检查是否过期
        """
        status = product_data.get('STATUS')
        end_date = str(product_data.get('END_DATE', ''))

        # 检查产品名称中的关键词（一票否决）
        name = product_data.get('PRD_NAME', '')
        if '封闭' in name and '持有' not in name:
            return ProductStatus.CLOSED, "封闭期产品"

        # 永续产品（END_DATE=20991231）默认可购买
        if end_date == '20991231':
            if status in [0, 4, '0', '4']:
                return ProductStatus.OPEN, "开放申购"

        # 有期限产品，检查是否过期
        if end_date.isdigit() and len(end_date) == 8:
            from datetime import datetime
            today = datetime.now().strftime('%Y%m%d')
            if end_date < today:
                return ProductStatus.CLOSED, "已到期"
            else:
                # 未到期，检查STATUS
                if status in [4, '4']:
                    return ProductStatus.OPEN, "开放申购"

        # STATUS字段判断
        if status is not None:
            status_str = str(status).strip()
            if status_str == '4':
                return ProductStatus.OPEN, "开放申购"
            if status_str == '0':
                return ProductStatus.OPEN, "开放申购"  # STATUS=0也是可用状态
            if status_str in ['1', '2', '3']:
                return ProductStatus.SUSPENDED, "暂停/募集中"

        return ProductStatus.UNKNOWN, "状态未知"

    def parse_duration(self, product_data: Dict, detail_data: Dict) -> Tuple[Optional[int], str]:
        """
        解析持有期限
        文本挖掘: "180天"、"最低持有7天"、"半年"、"季度"等
        """
        # 合并所有文本字段
        text_fields = ['PRD_NAME', 'PRD_DESC', 'HOLD_PERIOD', 'MIN_HOLD_DAYS',
                       'DURATION', 'prdName', 'prdDesc']

        all_text = ""
        for field in text_fields:
            value = product_data.get(field) or detail_data.get(field)
            if value:
                all_text += str(value) + " "

        # 直接提取数字天数
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
            match = re.search(pattern, all_text)
            if match:
                try:
                    days = extractor(match)
                    return days, match.group(0)
                except:
                    continue

        # 从MIN_HOLD_DAYS字段直接获取
        min_hold = product_data.get('MIN_HOLD_DAYS') or detail_data.get('MIN_HOLD_DAYS')
        if min_hold:
            try:
                days = int(min_hold)
                return days, f"{days}天"
            except:
                pass

        return None, "未知"

    def parse_benchmark(self, benchmark_str: str) -> Optional[float]:
        """解析业绩比较基准，提取数值"""
        if not benchmark_str:
            return None

        # 匹配百分比数值
        patterns = [
            r'(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*～\s*(\d+\.?\d*)',  # 取中间值
        ]

        match = re.search(patterns[0], benchmark_str)
        if match:
            return float(match.group(1))

        match = re.search(patterns[1], benchmark_str)
        if match:
            return (float(match.group(1)) + float(match.group(2))) / 2

        return None

    def parse_risk_category(self, risk_level: str) -> RiskCategory:
        """解析风险类别"""
        if not risk_level:
            return RiskCategory.FIXED_INCOME

        risk_str = str(risk_level).upper()

        # R4/R5 为权益类
        if 'R4' in risk_str or 'R5' in risk_str or '4' in risk_str or '5' in risk_str:
            return RiskCategory.EQUITY

        # R1/R2/R3 为固收类
        return RiskCategory.FIXED_INCOME

    def parse_product_type(self, product_name: str) -> Tuple[ProductType, bool]:
        """
        解析产品类型，判断是否可购买
        返回: (产品类型, 是否当前可购买)
        """
        name = product_name.upper()

        # 持有期产品：名称中含"持有"或具体天数持有期 - 随时可购买
        if '持有' in product_name or re.search(r'\d+天', product_name):
            if '封闭' not in product_name:
                return ProductType.HOLDING_PERIOD, True

        # 现金管理类：周周盈、天天盈等 - 随时可购买
        if any(kw in product_name for kw in ['周周盈', '天天盈', '日日盈', '现金', '货币']):
            return ProductType.CASH_MGMT, True

        # 定期开放：定开、季开、月开等 - 需等待开放期
        if any(kw in product_name for kw in ['定开', '季开', '月开', '年开', '季季盈', '月月盈']):
            return ProductType.SCHEDULED_OPEN, False  # 需等待开放期

        # 封闭式产品
        if '封闭' in product_name:
            return ProductType.CLOSED_END, False

        return ProductType.OTHER, True  # 默认可购买

    # ========================================================================
    # 量化指标计算
    # ========================================================================

    def calculate_metrics(self, nav_data: List[Dict], product_info: Dict) -> Optional[ProductProfile]:
        """计算全部量化指标"""
        if not nav_data or len(nav_data) < 2:
            return None

        # 按日期降序排序（最新在前）
        nav_sorted = sorted(nav_data, key=lambda x: str(x.get('ISS_DATE', '')), reverse=True)

        # 创建产品画像
        profile = ProductProfile()
        profile.product_code = product_info.get('REAL_PRD_CODE') or product_info.get('PRD_CODE', '')
        profile.product_name = product_info.get('PRD_NAME', '')
        profile.risk_level = product_info.get('RISK_LEVEL', '')
        profile.risk_category = self.parse_risk_category(profile.risk_level)
        profile.benchmark = product_info.get('BENCHMARK_CUSTO', '')
        profile.benchmark_value = self.parse_benchmark(profile.benchmark)

        # 解析产品类型和可购买状态
        profile.product_type, type_buyable = self.parse_product_type(profile.product_name)

        # 解析状态和期限（直接从产品列表数据解析，避免调用详情API以提升性能）
        profile.status, profile.status_text = self.parse_status(product_info, {})
        profile.duration_days, profile.duration_text = self.parse_duration(product_info, {})

        # 综合判断是否可购买：状态开放 + 产品类型允许
        profile.is_buyable = (profile.status == ProductStatus.OPEN) and type_buyable

        # 提取净值序列和日期序列
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

        # 最新净值信息
        profile.latest_nav = navs[0]
        profile.latest_nav_date = self.format_date(nav_sorted[0].get('ISS_DATE'))
        profile.days_from_today = (datetime.now() - dates[0]).days if dates else 9999

        # 保存历史数据（保存全部用于数据库存储）
        profile.nav_history = nav_sorted  # 保存全部净值历史

        # ====================================================================
        # 计算各期限收益率
        # ====================================================================
        today = datetime.now()

        # 近1周年化
        week_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 7]
        week_days = min(7, len(week_navs))
        if len(week_navs) >= 2:
            profile.metrics.return_1w = self.quant.calculate_annualized_return(week_navs, week_days)

        # 近1月年化
        month_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 30]
        month_days = min(30, len(month_navs))
        if len(month_navs) >= 2:
            profile.metrics.return_1m = self.quant.calculate_annualized_return(month_navs, month_days)

        # 近3月年化
        quarter_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 90]
        quarter_days = min(90, len(quarter_navs))
        if len(quarter_navs) >= 2:
            profile.metrics.return_3m = self.quant.calculate_annualized_return(quarter_navs, quarter_days)

        # 近6月年化
        half_year_navs = [n for n, d in zip(navs, dates) if (today - d).days <= 180]
        half_year_days = min(180, len(half_year_navs))
        if len(half_year_navs) >= 2:
            profile.metrics.return_6m = self.quant.calculate_annualized_return(half_year_navs, half_year_days)

        # ====================================================================
        # 计算风险指标
        # ====================================================================

        # 日收益率序列（用于波动率计算）
        daily_returns = []
        for i in range(len(navs) - 1):
            if navs[i+1] != 0:
                ret = (navs[i] - navs[i+1]) / navs[i+1] * 100
                daily_returns.append(ret)

        # 波动率
        profile.metrics.volatility = self.quant.calculate_volatility(daily_returns)

        # 最大回撤（使用近6月数据）
        profile.metrics.max_drawdown = self.quant.calculate_max_drawdown(half_year_navs if half_year_navs else navs)

        # ====================================================================
        # 计算风险调整收益
        # ====================================================================

        # 使用近3月年化计算夏普比率
        return_for_sharpe = profile.metrics.return_3m or profile.metrics.return_1m
        profile.metrics.sharpe_ratio = self.quant.calculate_sharpe_ratio(
            return_for_sharpe,
            profile.metrics.volatility
        )

        # 卡玛比率
        profile.metrics.calmar_ratio = self.quant.calculate_calmar_ratio(
            return_for_sharpe,
            profile.metrics.max_drawdown
        )

        # 超额收益
        if profile.metrics.return_1m is not None:
            profile.metrics.excess_return_1m = profile.metrics.return_1m - Config.RISK_FREE_RATE

        return profile

    # ========================================================================
    # 多因子评分
    # ========================================================================

    def calculate_score(self, profile: ProductProfile) -> float:
        """
        计算多因子评分
        固收类: Score = 0.6 × 近3月年化 + 0.3 × 近1周年化 - 0.1 × 波动率
        权益类: Score = 0.5 × 夏普比率 + 0.3 × 卡玛比率 + 0.2 × 近1月超额收益
        """
        m = profile.metrics

        if profile.risk_category == RiskCategory.FIXED_INCOME:
            # 固收类评分
            score = 0.0
            weights = Config.FIXED_INCOME_WEIGHTS

            if m.return_3m is not None:
                score += weights['return_3m'] * m.return_3m
            elif m.return_1m is not None:
                score += weights['return_3m'] * m.return_1m  # 降级使用1月数据

            if m.return_1w is not None:
                score += weights['return_1w'] * m.return_1w

            if m.volatility is not None:
                score += weights['volatility'] * m.volatility  # 负权重

        else:
            # 权益类评分
            score = 0.0
            weights = Config.EQUITY_WEIGHTS

            if m.sharpe_ratio is not None:
                # 夏普比率归一化到合理范围
                normalized_sharpe = max(-5, min(5, m.sharpe_ratio)) * 10
                score += weights['sharpe_ratio'] * normalized_sharpe

            if m.calmar_ratio is not None:
                # 卡玛比率归一化
                normalized_calmar = max(-5, min(5, m.calmar_ratio)) * 10
                score += weights['calmar_ratio'] * normalized_calmar

            if m.excess_return_1m is not None:
                score += weights['excess_return_1m'] * m.excess_return_1m

        return round(score, 2)

    def generate_signal(self, profile: ProductProfile) -> SignalType:
        """
        生成交易信号
        综合考虑: 申购状态、产品类型、评分、风险指标
        注：周末避险规则改为报告提示，不影响信号生成
        """
        # 一票否决：不可购买（状态封闭/暂停，或定开产品不在开放期）
        if not profile.is_buyable:
            return SignalType.AVOID

        # 一票否决：净值跌破止损线
        if profile.latest_nav < Config.STOP_LOSS_THRESHOLD:
            return SignalType.AVOID

        # 一票否决：数据过旧（超过7天未更新）
        if profile.days_from_today > 7:
            return SignalType.WATCH

        # 基于评分生成信号（周末避险改为报告中提示）
        score = profile.score

        if profile.risk_category == RiskCategory.FIXED_INCOME:
            # 固收类阈值
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
            # 权益类阈值（更严格）
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
        """
        构建核心-卫星投资组合
        核心仓位 (60%-80%): R2/R3 固收类持有期产品
        卫星仓位 (20%-40%): R4/R5 权益类
        """
        # 过滤可购买产品（is_buyable=True，优先持有期产品）
        buyable = [p for p in profiles if p.is_buyable]

        # 分类
        fixed_income = [p for p in buyable if p.risk_category == RiskCategory.FIXED_INCOME]
        equity = [p for p in buyable if p.risk_category == RiskCategory.EQUITY]

        # 按评分排序
        fixed_income.sort(key=lambda x: x.score, reverse=True)
        equity.sort(key=lambda x: x.score, reverse=True)

        # 核心仓位推荐（取前5名，优先持有期产品）
        # 优先排序：持有期产品 > 现金管理 > 其他
        fixed_income.sort(key=lambda x: (
            0 if x.product_type == ProductType.HOLDING_PERIOD else
            1 if x.product_type == ProductType.CASH_MGMT else 2,
            -x.score
        ))

        core_recommendations = []
        for p in fixed_income[:5]:
            if p.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                core_recommendations.append({
                    'code': p.product_code,
                    'name': p.product_name,
                    'risk': p.risk_level,
                    'product_type': p.product_type.value,
                    'score': p.score,
                    'signal': p.signal.value,
                    'return_3m': p.metrics.return_3m,
                    'volatility': p.metrics.volatility,
                    'duration': p.duration_text
                })

        # 卫星仓位推荐（取前3名）
        satellite_recommendations = []
        for p in equity[:3]:
            if p.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                satellite_recommendations.append({
                    'code': p.product_code,
                    'name': p.product_name,
                    'risk': p.risk_level,
                    'score': p.score,
                    'signal': p.signal.value,
                    'sharpe': p.metrics.sharpe_ratio,
                    'calmar': p.metrics.calmar_ratio,
                    'max_drawdown': p.metrics.max_drawdown
                })

        return {
            'core': {
                'description': '核心仓位 - 固收类 (建议占比60%-80%)',
                'target_return': '3%-4%年化',
                'recommendations': core_recommendations
            },
            'satellite': {
                'description': '卫星仓位 - 权益类 (建议占比20%-40%)',
                'target_return': '波动较大，可能>10%也可能亏损',
                'stop_loss': f'建议止损线: 净值跌破{Config.STOP_LOSS_THRESHOLD}即赎回',
                'recommendations': satellite_recommendations
            }
        }

    def match_fund_duration(self, profiles: List[ProductProfile],
                            fund_duration_days: int) -> List[ProductProfile]:
        """
        资金期限匹配
        根据资金使用周期匹配合适期限的产品
        """
        matched = []
        for p in profiles:
            if p.status != ProductStatus.OPEN:
                continue

            if p.duration_days is None:
                # 未知期限，保守处理
                if fund_duration_days >= 30:
                    matched.append(p)
            elif p.duration_days <= fund_duration_days:
                matched.append(p)

        return sorted(matched, key=lambda x: x.score, reverse=True)

    # ========================================================================
    # 数据处理流程
    # ========================================================================

    def process_single_product(self, product: Dict) -> Tuple[Optional[ProductProfile], str]:
        """
        处理单个产品
        返回: (profile, skip_reason)
            - profile: 产品画像（成功时）
            - skip_reason: 跳过原因（失败时）
        """
        code = product.get('REAL_PRD_CODE') or product.get('PRD_CODE', '')
        name = product.get('PRD_NAME', '')[:30]

        if not code:
            return None, f"无产品代码: {name}"

        # 获取历史净值
        nav_data = self.get_nav_history(code, days=180)
        if not nav_data:
            return None, f"无NAV数据: {name}"

        # 计算量化指标
        profile = self.calculate_metrics(nav_data, product)
        if not profile:
            return None, f"NAV数据不足: {name} (仅{len(nav_data)}条)"

        # 计算评分
        profile.score = self.calculate_score(profile)

        # 生成信号
        profile.signal = self.generate_signal(profile)

        return profile, "成功"

    def _process_batch(self, products_batch: List[Dict], batch_num: int,
                       full_history: bool = False) -> List[Tuple[Optional[ProductProfile], str]]:
        """
        处理一批产品（使用独立会话，避免API限制）
        每批最多处理 SESSION_NAV_LIMIT 个产品

        Args:
            full_history: True=请求完整历史(3年), False=仅180天
        """
        session = self._create_session()
        results = []

        for i, product in enumerate(products_batch):
            code = product.get('REAL_PRD_CODE') or product.get('PRD_CODE', '')
            name = product.get('PRD_NAME', '')[:30]

            if not code:
                results.append((None, f"无产品代码: {name}"))
                continue

            # 使用本批次专属会话获取NAV数据
            nav_data = self.get_nav_history(code, days=180, session=session,
                                            full_history=full_history)
            if not nav_data:
                results.append((None, f"无NAV数据: {name}"))
                continue

            # 计算量化指标
            profile = self.calculate_metrics(nav_data, product)
            if not profile:
                results.append((None, f"NAV数据不足: {name} (仅{len(nav_data)}条)"))
                continue

            # 计算评分和信号
            profile.score = self.calculate_score(profile)
            profile.signal = self.generate_signal(profile)

            results.append((profile, "成功"))

            # 适当延迟避免请求过快
            if (i + 1) % 50 == 0:
                time.sleep(0.2)

        session.close()
        return results

    def crawl_all_products(self, max_workers: int = 10, limit: int = None) -> Optional[str]:
        """爬取并分析所有产品（使用批次处理突破API限制）"""
        logger.info("="*70)
        logger.info("民生理财产品量化分析系统 v3.0")
        logger.info("="*70)

        start_time = time.time()

        # 1. 获取产品列表
        products = self.get_product_list()
        if not products:
            logger.error("未获取到产品列表")
            return None

        if limit:
            products = products[:limit]
            logger.info(f"测试模式：处理前 {limit} 个产品")

        # 2. 分批处理（每批 SESSION_NAV_LIMIT 个，使用独立会话突破API限制）
        batch_size = self.SESSION_NAV_LIMIT
        batches = [products[i:i+batch_size] for i in range(0, len(products), batch_size)]

        logger.info(f"开始分析 {len(products)} 个产品，分 {len(batches)} 批处理（每批{batch_size}个）...")

        all_profiles: List[ProductProfile] = []
        skip_reasons: Dict[str, int] = {}
        failed = 0
        completed = 0

        # 并发处理各批次（每批使用独立会话）
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
            futures = {executor.submit(self._process_batch, batch, i): i
                      for i, batch in enumerate(batches)}

            for future in as_completed(futures):
                batch_num = futures[future]
                try:
                    batch_results = future.result()
                    for profile, reason in batch_results:
                        completed += 1
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

        # 显示跳过原因统计
        if skip_reasons:
            logger.info("跳过产品统计:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  - {reason}: {count}个")

        if not all_profiles:
            logger.warning("未获取到有效数据")
            return None

        # 3. 排序（按评分从高到低）
        all_profiles.sort(key=lambda x: (x.days_from_today, -x.score))

        # 4. 构建投资组合
        portfolio = self.build_portfolio(all_profiles)

        # 5. 保存结果
        filename = self.save_results(all_profiles, portfolio)

        # 6. 更新净值数据库Excel
        if HAS_NAV_DB:
            try:
                logger.info("正在更新净值数据库...")
                # 转换profiles为数据库需要的格式
                products_for_db = []
                for p in all_profiles:
                    products_for_db.append({
                        'product_code': p.product_code,
                        'product_name': p.product_name,
                        'nav_history': [
                            {
                                'date': self.format_date(nav.get('ISS_DATE')),
                                'unit_nav': nav.get('NAV')
                            }
                            for nav in p.nav_history if nav.get('NAV')
                        ]
                    })
                stats = update_nav_database('民生', products_for_db)
                logger.info(f"净值数据库更新完成: 新增 {len(stats.get('new_dates', []))} 个日期, "
                           f"更新 {stats.get('updated_cells', 0)} 个单元格")
            except Exception as e:
                logger.warning(f"更新净值数据库失败: {e}")
        else:
            logger.warning("净值数据库模块未安装，跳过数据库更新")

        # 7. 显示报告
        self.display_report(all_profiles, portfolio)

        return filename

    # ========================================================================
    # 结果输出
    # ========================================================================

    def save_results(self, profiles: List[ProductProfile], portfolio: Dict) -> str:
        """保存分析结果到Excel（多Sheet）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'民生理财_量化分析_{timestamp}.xlsx'

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: 全部产品评分
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
                    '业绩基准': p.benchmark
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
                    '交易信号': p.signal.value
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
                    '交易信号': p.signal.value
                })

            df_equity = pd.DataFrame(equity_data)
            df_equity.to_excel(writer, sheet_name='权益类TOP20', index=False)

            # Sheet 4: 投资组合推荐
            portfolio_data = []

            # 核心仓位
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
                    '持有期限': rec.get('duration')
                })

            # 卫星仓位
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
                    '最大回撤(%)': rec.get('max_drawdown')
                })

            df_portfolio = pd.DataFrame(portfolio_data)
            df_portfolio.to_excel(writer, sheet_name='投资组合推荐', index=False)

            # Sheet 5: 买入信号产品
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
                    '最大回撤(%)': p.metrics.max_drawdown
                })

            df_buy = pd.DataFrame(buy_data)
            df_buy.to_excel(writer, sheet_name='买入信号', index=False)

            # Sheet 6: 可购买的持有期产品（用户最关注）
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
                    '业绩基准': p.benchmark
                })

            df_holding = pd.DataFrame(holding_data)
            df_holding.to_excel(writer, sheet_name='可购买持有期产品', index=False)

        logger.info(f"\n分析结果已保存到: {filename}")

        # 同时保存latest版本
        latest_file = '民生理财_量化分析_latest.xlsx'
        try:
            with pd.ExcelWriter(latest_file, engine='openpyxl') as writer:
                df_holding.to_excel(writer, sheet_name='可购买持有期产品', index=False)  # 用户最关注，放第一个
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
        print("\n" + "="*80)
        print("                    民生理财产品量化分析报告")
        print("="*80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"产品总数: {len(profiles)}")

        # 统计信息
        open_count = sum(1 for p in profiles if p.status == ProductStatus.OPEN)
        fixed_count = sum(1 for p in profiles if p.risk_category == RiskCategory.FIXED_INCOME)
        equity_count = sum(1 for p in profiles if p.risk_category == RiskCategory.EQUITY)

        buyable_count = sum(1 for p in profiles if p.is_buyable)
        holding_count = sum(1 for p in profiles if p.is_buyable and p.product_type == ProductType.HOLDING_PERIOD)
        buy_signals = sum(1 for p in profiles if p.signal in [SignalType.STRONG_BUY, SignalType.BUY])

        print(f"可购买产品: {buyable_count} (其中持有期产品: {holding_count})")
        print(f"固收类: {fixed_count} | 权益类: {equity_count}")
        print(f"买入信号数: {buy_signals}")

        # 周末避险提示
        today = datetime.now()
        weekday = today.weekday()
        if weekday >= 3:  # 周四及以后
            print("\n" + "-"*80)
            print("注意: 当前接近周末，建议谨慎申购（资金周末可能无法计息）")
            print("-"*80)

        # ====================================================================
        # 可购买持有期产品TOP10（用户最关注）
        # ====================================================================
        print("\n" + "="*80)
        print("              可购买持有期产品TOP10 (随时可申购)")
        print("="*80)

        holding = [p for p in profiles if p.is_buyable and p.product_type == ProductType.HOLDING_PERIOD]
        holding.sort(key=lambda x: x.score, reverse=True)

        if holding:
            header = f"{'#':^3} | {'产品名称':^30} | {'期限':^8} | {'3月年化':^8} | {'评分':^6} | {'信号':^8}"
            print(header)
            print("-"*80)

            for i, p in enumerate(holding[:10], 1):
                name = p.product_name[:28] if len(p.product_name) > 28 else p.product_name
                duration = p.duration_text[:6] if p.duration_text else "N/A"
                ret_3m = f"{p.metrics.return_3m:+.2f}%" if p.metrics.return_3m else "N/A"
                score = f"{p.score:.2f}"
                signal = p.signal.value

                print(f"{i:^3} | {name:30s} | {duration:^8} | {ret_3m:^8} | {score:^6} | {signal:^8}")
        else:
            print("  暂无可购买的持有期产品")

        # ====================================================================
        # 固收类TOP10
        # ====================================================================
        print("\n" + "="*80)
        print("                    固收类TOP10 (R1/R2/R3)")
        print("评分公式: 0.6×近3月年化 + 0.3×近1周年化 - 0.1×波动率")
        print("="*80)

        fixed = [p for p in profiles if p.risk_category == RiskCategory.FIXED_INCOME]
        fixed.sort(key=lambda x: x.score, reverse=True)

        header = f"{'#':^3} | {'产品名称':^28} | {'状态':^6} | {'3月年化':^8} | {'评分':^6} | {'信号':^8}"
        print(header)
        print("-"*80)

        for i, p in enumerate(fixed[:10], 1):
            name = p.product_name[:26] if len(p.product_name) > 26 else p.product_name
            status = p.status.value[:4]
            ret_3m = f"{p.metrics.return_3m:+.2f}%" if p.metrics.return_3m else "N/A"
            score = f"{p.score:.2f}"
            signal = p.signal.value

            print(f"{i:^3} | {name:28s} | {status:^6} | {ret_3m:^8} | {score:^6} | {signal:^8}")

        # ====================================================================
        # 权益类TOP10
        # ====================================================================
        print("\n" + "="*80)
        print("                    权益类TOP10 (R4/R5)")
        print("评分公式: 0.5×夏普比率 + 0.3×卡玛比率 + 0.2×超额收益")
        print("="*80)

        equity = [p for p in profiles if p.risk_category == RiskCategory.EQUITY]
        equity.sort(key=lambda x: x.score, reverse=True)

        header = f"{'#':^3} | {'产品名称':^28} | {'夏普':^6} | {'卡玛':^6} | {'回撤':^6} | {'评分':^6} | {'信号':^8}"
        print(header)
        print("-"*80)

        for i, p in enumerate(equity[:10], 1):
            name = p.product_name[:26] if len(p.product_name) > 26 else p.product_name
            sharpe = f"{p.metrics.sharpe_ratio:.2f}" if p.metrics.sharpe_ratio else "N/A"
            calmar = f"{p.metrics.calmar_ratio:.2f}" if p.metrics.calmar_ratio else "N/A"
            mdd = f"{p.metrics.max_drawdown:.1f}%" if p.metrics.max_drawdown else "N/A"
            score = f"{p.score:.2f}"
            signal = p.signal.value

            print(f"{i:^3} | {name:28s} | {sharpe:^6} | {calmar:^6} | {mdd:^6} | {score:^6} | {signal:^8}")

        # ====================================================================
        # 投资组合推荐
        # ====================================================================
        print("\n" + "="*80)
        print("                    投资组合推荐 (核心-卫星策略)")
        print("="*80)

        print("\n[核心仓位 60%-80%] - 固收类持有期产品（随时可购买）")
        print("-"*60)
        if portfolio['core']['recommendations']:
            for rec in portfolio['core']['recommendations'][:3]:
                ptype = rec.get('product_type', '未知')
                print(f"  {rec['signal']:8} | {rec['name'][:30]}")
                print(f"           类型: {ptype} | 近3月年化: {rec.get('return_3m', 'N/A')}% | 期限: {rec.get('duration', 'N/A')}")
        else:
            print("  暂无符合条件的推荐")

        print("\n[卫星仓位 20%-40%] - 权益类增强产品")
        print(f"  止损规则: 净值跌破 {Config.STOP_LOSS_THRESHOLD} 即赎回")
        print("-"*60)
        if portfolio['satellite']['recommendations']:
            for rec in portfolio['satellite']['recommendations'][:3]:
                print(f"  {rec['signal']:8} | {rec['name'][:30]}")
                print(f"           夏普: {rec.get('sharpe', 'N/A')} | 最大回撤: {rec.get('max_drawdown', 'N/A')}%")
        else:
            print("  暂无符合条件的推荐")

        print("\n" + "="*80)
        print("提示: 详细数据请查看Excel文件中的各分析Sheet")
        print("="*80)


# ============================================================================
# 定时任务
# ============================================================================

def run_daily_task():
    """每日定时任务"""
    logger.info("\n" + "#"*60)
    logger.info(f"执行每日量化分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("#"*60)

    crawler = CMBCWealthCrawler()
    result = crawler.crawl_all_products()

    if result:
        logger.info(f"分析完成: {result}")
    else:
        logger.error("分析失败")


def schedule_daily_task(hour: int = 9, minute: int = 0):
    """设置每日定时任务"""
    schedule_time = f"{hour:02d}:{minute:02d}"
    schedule.every().day.at(schedule_time).do(run_daily_task)

    logger.info(f"定时任务已设置：每天 {schedule_time} 执行")
    logger.info("按 Ctrl+C 停止")

    while True:
        schedule.run_pending()
        time.sleep(60)


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """主函数"""
    print("="*70)
    print("        民生理财产品量化分析系统 v3.0 - 专业量化版")
    print("="*70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("核心功能:")
    print("  - 全维度产品画像（申购状态、持有期限、风险等级）")
    print("  - 量化指标计算（近1月/3月年化、夏普比率、卡玛比率、最大回撤）")
    print("  - 多因子评分模型（固收类/权益类分别评分）")
    print("  - T+0 动态信号生成（买入/持有/观望/回避）")
    print("  - 核心-卫星投资组合推荐")
    print()

    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--schedule':
            hour = int(sys.argv[2]) if len(sys.argv) > 2 else 9
            minute = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            schedule_daily_task(hour, minute)
        elif sys.argv[1] == '--test':
            crawler = CMBCWealthCrawler()
            crawler.crawl_all_products(limit=50)
        elif sys.argv[1] == '--help':
            print("用法:")
            print("  python minsheng.py              # 立即执行完整分析")
            print("  python minsheng.py --test       # 测试模式（50个产品）")
            print("  python minsheng.py --schedule [hour] [minute]  # 定时任务")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        crawler = CMBCWealthCrawler()
        result = crawler.crawl_all_products()

        if result:
            print(f"\n分析完成！文件: {result}")
        else:
            print("\n分析失败，请查看日志")


if __name__ == "__main__":
    main()
