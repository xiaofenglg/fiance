# -*- coding: utf-8 -*-
"""
中信理财说明书PDF本地解析器

从 pdfs/中信/ 目录读取已下载的PDF，提取费率信息，写入赎回费数据库。

解析策略:
1. 优先表格解析（结构化数据更可靠）
2. 回退文本正则解析
3. 失败时记录原文供人工检查

使用方法:
    python parse_citic_pdfs.py              # 解析全部
    python parse_citic_pdfs.py --max 100    # 限量测试
    python parse_citic_pdfs.py --force      # 强制重新解析
    python parse_citic_pdfs.py --debug      # 调试模式，打印解析细节
"""

import os
import sys
import re
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parse_citic_pdfs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.error("pdfplumber未安装，请运行: pip install pdfplumber")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '中信')
INDEX_FILE = os.path.join(PDF_DIR, 'index.json')
FAILED_LOG = os.path.join(BASE_DIR, 'parse_failed.json')  # 解析失败记录

sys.path.insert(0, BASE_DIR)
from redemption_fee_db import load_fee_db, save_fee_db, update_fee_info, get_fee_info

# ============================================================
# 常量与正则
# ============================================================

# 产品代码正则
PRODUCT_CODE_RE = re.compile(r'\b([A-Z]{2}\d{5,8}[A-Z]?)\b')

# 费率列名映射（表格解析用）
FEE_COLUMN_MAPPING = {
    '固定管理费': 'management_fee',
    '固定管理费率': 'management_fee',
    '管理费': 'management_fee',
    '管理费率': 'management_fee',
    '托管费': 'custody_fee',
    '托管费率': 'custody_fee',
    '销售费': 'sales_service_fee',
    '销售费率': 'sales_service_fee',
    '销售服务费': 'sales_service_fee',
    '销售服务费率': 'sales_service_fee',
    '认购费': 'subscription_fee',
    '认购费率': 'subscription_fee',
    '申购费': 'purchase_fee',
    '申购费率': 'purchase_fee',
    '赎回费': 'redemption_fee',
    '赎回费率': 'redemption_fee',
}

# 费率行关键词（用于定位表格中的费率行）
FEE_ROW_KEYWORDS = ['管理费', '托管费', '销售', '认购', '申购', '赎回', '费率']


# ============================================================
# 1. PDF文本与表格提取
# ============================================================

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """从PDF提取文本和表格

    Returns:
        {
            'text': str,           # 全文文本
            'tables': List[List],  # 所有表格
            'pages': int           # 页数
        }
    """
    if pdfplumber is None:
        return {'text': '', 'tables': [], 'pages': 0}

    try:
        full_text = ''
        all_tables = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 提取文本
                text = page.extract_text()
                if text:
                    full_text += text + '\n'

                # 提取表格
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)

            return {
                'text': full_text,
                'tables': all_tables,
                'pages': len(pdf.pages)
            }
    except Exception as e:
        logger.debug(f"PDF提取失败 {pdf_path}: {e}")
        return {'text': '', 'tables': [], 'pages': 0}


# ============================================================
# 2. 产品代码提取
# ============================================================

def extract_product_code(text: str) -> Optional[str]:
    """从文本提取产品代码"""
    if not text:
        return None

    # 先找"产品代码"附近的代码
    patterns = [
        r'(?:产品代码|产品编码|登记编码)[：:\s]*([A-Z]{2}\d{5,8}[A-Z]?)',
        r'(?:代码|编码)[：:\s]*([A-Z]{2}\d{5,8}[A-Z]?)',
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    # 退化: 全文第一个匹配
    m = PRODUCT_CODE_RE.search(text)
    if m:
        return m.group(1)

    return None


# ============================================================
# 3. 费用段落定位
# ============================================================

def extract_fee_section(text: str) -> str:
    """从全文中定位费用相关段落"""
    if not text:
        return ''

    # 尝试定位费用章节
    patterns = [
        # "（四）产品费用" 或 "四、产品费用"
        re.compile(
            r'[（(]?\s*[四五六七八九十\d]+\s*[）)]?\s*[、.]?\s*(?:产品)?费用'
            r'.*?(?=[（(]\s*[四五六七八九十\d]+\s*[）)]|[四五六七八九十]{1,2}\s*[、]|第[四五六七八九十]+[章节部分]|\Z)',
            re.DOTALL
        ),
        # "费用说明" 段
        re.compile(
            r'费用(?:说明|概述|结构|条款).*?(?=第[四五六七八九十]+[章节部分]|[四五六七八九十]{1,2}\s*[、]|\Z)',
            re.DOTALL
        ),
    ]

    for pat in patterns:
        m = pat.search(text)
        if m:
            section = m.group(0)
            if len(section) > 100:
                return section

    # 退化: 搜索包含多个费用关键词的区域
    keywords = ['管理费', '托管费', '销售服务费']
    positions = []
    for kw in keywords:
        idx = text.find(kw)
        if idx >= 0:
            positions.append(idx)

    if positions:
        start = max(0, min(positions) - 200)
        end = min(len(text), max(positions) + 1500)
        return text[start:end]

    # 最后退化: 取前4000字
    return text[:4000]


# ============================================================
# 4. 表格解析（策略1 - 优先）
# ============================================================

def parse_fees_from_tables(tables: List[List]) -> Dict[str, Any]:
    """从表格中解析费率信息

    Returns:
        {
            'management_fee': float | None,
            'custody_fee': float | None,
            'sales_service_fee': float | None,
            'subscription_fee': float | None,
            'purchase_fee': float | None,
            'redemption_fee': float | None,  # 简单费率
            'redemption_schedule': List | None,  # 阶梯费率
            'source': 'table'
        }
    """
    result = {
        'management_fee': None,
        'custody_fee': None,
        'sales_service_fee': None,
        'subscription_fee': None,
        'purchase_fee': None,
        'redemption_fee': None,
        'redemption_schedule': None,
        'source': 'table'
    }

    for table in tables:
        if not table or len(table) < 2:
            continue

        # 尝试解析为费率表
        parsed = _parse_single_fee_table(table)
        if parsed:
            # 合并结果（非None值覆盖）
            for key, val in parsed.items():
                if val is not None and result.get(key) is None:
                    result[key] = val

    # 检查是否有任何有效数据
    has_data = any(v is not None for k, v in result.items() if k not in ('source', 'redemption_schedule'))
    return result if has_data else {}


def _parse_single_fee_table(table: List[List]) -> Dict[str, Any]:
    """解析单个费率表格"""
    result = {}

    # 检查是否是费率相关表格
    table_text = ' '.join(str(cell) for row in table for cell in row if cell)
    if not any(kw in table_text for kw in FEE_ROW_KEYWORDS):
        return {}

    # 策略A: 两列表格 (费用类型 | 费率)
    if all(len(row) >= 2 for row in table if row):
        for row in table:
            if not row or len(row) < 2:
                continue

            cell0 = str(row[0] or '').strip()
            cell1 = str(row[1] or '').strip()

            # 查找费用类型
            for keyword, field in FEE_COLUMN_MAPPING.items():
                if keyword in cell0:
                    rate = _parse_rate_value(cell1)
                    if rate is not None:
                        result[field] = rate
                    break

    # 策略B: 多列表格（表头在第一行）
    if not result and len(table) >= 2:
        header = table[0]
        if header:
            col_mapping = {}
            for i, cell in enumerate(header):
                cell_str = str(cell or '').strip()
                for keyword, field in FEE_COLUMN_MAPPING.items():
                    if keyword in cell_str:
                        col_mapping[i] = field
                        break

            # 解析数据行
            for row in table[1:]:
                if not row:
                    continue
                for col_idx, field in col_mapping.items():
                    if col_idx < len(row):
                        rate = _parse_rate_value(str(row[col_idx] or ''))
                        if rate is not None and result.get(field) is None:
                            result[field] = rate

    return result


def _parse_rate_value(text: str) -> Optional[float]:
    """解析费率值

    支持格式:
    - "0.40%/年" → 0.004
    - "0.40%" → 0.004
    - "【0.40%】" → 0.004
    - "0" 或 "0%" → 0.0
    - "0.40" (纯数字，视为百分比) → 0.004
    """
    if not text:
        return None

    text = text.strip()

    # 移除【】
    text = text.replace('【', '').replace('】', '')

    # 匹配百分比格式: "0.40%"
    m = re.search(r'([\d.]+)\s*%', text)
    if m:
        try:
            return float(m.group(1)) / 100
        except ValueError:
            pass

    # 匹配纯数字（视为百分比值）
    # 注意：费率通常 < 2%，所以纯数字 < 2 才视为百分比值
    m = re.match(r'^([\d.]+)$', text.strip())
    if m:
        try:
            val = float(m.group(1))
            # 小于2的数字视为百分比值（如0.40 = 0.40%，1.5 = 1.5%）
            # 大于2的数字可能是其他含义，不处理
            if val < 2:
                return val / 100
        except ValueError:
            pass

    # "0" 或空值
    if text in ('0', '0.0', '0.00', '-', '—', '无'):
        return 0.0

    return None


# ============================================================
# 5. 文本正则解析（策略2 - 回退）
# ============================================================

def parse_fees_from_text(text: str) -> Dict[str, Any]:
    """从文本中用正则解析费率信息"""
    result = {
        'management_fee': None,
        'custody_fee': None,
        'sales_service_fee': None,
        'subscription_fee': None,
        'purchase_fee': None,
        'redemption_fee': None,
        'redemption_schedule': None,
        'source': 'text'
    }

    if not text:
        return {}

    # 定位费用段落
    fee_section = extract_fee_section(text)

    # 解析各类费率
    result['management_fee'] = _parse_rate_from_text(fee_section, r'(?:固定)?管理费')
    result['custody_fee'] = _parse_rate_from_text(fee_section, r'托管费')
    result['sales_service_fee'] = _parse_rate_from_text(fee_section, r'销售(?:服务)?费')
    result['subscription_fee'] = _parse_simple_fee_from_text(fee_section, r'认购费')
    result['purchase_fee'] = _parse_simple_fee_from_text(fee_section, r'申购费')

    # 解析赎回费（可能是阶梯）
    redemption_schedule = _parse_redemption_fee_schedule(fee_section)
    if redemption_schedule:
        result['redemption_schedule'] = redemption_schedule
    else:
        result['redemption_fee'] = _parse_simple_fee_from_text(fee_section, r'赎回费')

    # 检查是否有任何有效数据
    has_data = any(v is not None for k, v in result.items() if k not in ('source', 'redemption_schedule'))
    has_data = has_data or (result.get('redemption_schedule') is not None)

    return result if has_data else {}


def _parse_rate_from_text(text: str, fee_pattern: str) -> Optional[float]:
    """从文本解析年化费率"""
    # 格式1: "费率【0.40%】/年"
    m = re.search(rf'{fee_pattern}[：:\s]*费率[【\s]*([\d.]+)\s*%', text)
    if m:
        return float(m.group(1)) / 100

    # 格式2: "管理费率0.40%/年"
    m = re.search(rf'{fee_pattern}[率为：:\s]*[【\s]*([\d.]+)\s*%', text)
    if m:
        return float(m.group(1)) / 100

    # 格式3: "管理费：0.40%"
    m = re.search(rf'{fee_pattern}[：:\s]*([\d.]+)\s*%', text)
    if m:
        return float(m.group(1)) / 100

    return None


def _parse_simple_fee_from_text(text: str, fee_pattern: str) -> Optional[float]:
    """从文本解析简单费率（认购费、申购费等）"""
    # 同上格式
    rate = _parse_rate_from_text(text, fee_pattern)
    if rate is not None:
        return rate

    # 不收取
    if re.search(rf'不收取{fee_pattern}|{fee_pattern}[为：:\s]*0|免收{fee_pattern}|{fee_pattern}[为：:\s]*无', text):
        return 0.0

    return None


def _parse_redemption_fee_schedule(text: str) -> Optional[List[Dict]]:
    """解析赎回费阶梯结构"""
    # 明确无赎回费
    if re.search(r'不收取赎回费|赎回费[率为：:\s]*0[\.0]*\s*%|无赎回费|免收赎回费|赎回费[为：:\s]*无', text):
        return []

    thresholds = {}

    # 模式1: "持有不满N天...赎回费率X%"
    for m in re.finditer(
        r'持有(?:期限)?不满\s*(\d+)\s*(?:天|个自然日|日|个工作日)[^%]*?赎回费率?\s*([\d.]+)\s*%',
        text, re.DOTALL
    ):
        days, rate = int(m.group(1)), float(m.group(2)) / 100
        thresholds[days] = ('under', rate)

    # 模式2: "持有满N天...赎回费率X%"
    for m in re.finditer(
        r'持有(?:期限)?满\s*(\d+)\s*(?:天|个自然日|日|个工作日)[^%]*?赎回费率?\s*([\d.]+)\s*%',
        text, re.DOTALL
    ):
        days, rate = int(m.group(1)), float(m.group(2)) / 100
        if days not in thresholds:
            thresholds[days] = ('over', rate)
        else:
            thresholds[(days, 'over')] = ('over', rate)

    # 模式3: "N天以内...赎回费率X%"
    for m in re.finditer(
        r'(\d+)\s*(?:天|日|个自然日)以内[^%]*?赎回费率?\s*([\d.]+)\s*%',
        text, re.DOTALL
    ):
        days, rate = int(m.group(1)), float(m.group(2)) / 100
        if days not in thresholds:
            thresholds[days] = ('under', rate)

    # 模式4: "N天以上...赎回费率X%"
    for m in re.finditer(
        r'(\d+)\s*(?:天|日|个自然日)\s*[（(]?\s*含?\s*[）)]?\s*以上[^%]*?赎回费率?\s*([\d.]+)\s*%',
        text, re.DOTALL
    ):
        days, rate = int(m.group(1)), float(m.group(2)) / 100
        if days not in thresholds:
            thresholds[days] = ('over', rate)

    # 模式5: "少于/不足N天...赎回费率X%"
    for m in re.finditer(
        r'(?:持有|连续持有)?\s*(?:少于|不足|不到)\s*(\d+)\s*(?:天|日)[^%]*?赎回费率?\s*([\d.]+)\s*%',
        text, re.DOTALL
    ):
        days, rate = int(m.group(1)), float(m.group(2)) / 100
        if days not in thresholds:
            thresholds[days] = ('under', rate)

    if not thresholds:
        return None

    # 构建阶梯
    return _build_schedule_from_thresholds(thresholds)


def _build_schedule_from_thresholds(thresholds: Dict) -> List[Dict]:
    """从解析到的阈值构建完整的费率阶梯"""
    # 收集所有边界天数
    boundaries = set()
    entries = []

    for key, (typ, rate) in thresholds.items():
        days = key if isinstance(key, int) else key[0]
        boundaries.add(days)
        entries.append((days, typ, rate))

    boundaries = sorted(boundaries)

    if not boundaries:
        return []

    schedule = []
    prev_day = 0

    for i, day in enumerate(boundaries):
        under_rate = None
        over_rate = None

        for d, typ, rate in entries:
            if d == day:
                if typ == 'under':
                    under_rate = rate
                elif typ == 'over':
                    over_rate = rate

        if under_rate is not None:
            schedule.append({
                'min_days': prev_day,
                'max_days': day,
                'fee_rate': under_rate,
            })
            prev_day = day

        if over_rate is not None and i == len(boundaries) - 1:
            schedule.append({
                'min_days': day,
                'max_days': 999999,
                'fee_rate': over_rate,
            })

    # 补充最后阶梯（如果没有明确的"满N天"条目）
    if schedule and schedule[-1]['max_days'] != 999999:
        has_over = any(typ == 'over' for _, typ, _ in entries)
        if not has_over:
            schedule.append({
                'min_days': schedule[-1]['max_days'],
                'max_days': 999999,
                'fee_rate': 0.0,
            })

    return schedule


# ============================================================
# 6. 主解析函数
# ============================================================

def parse_single_pdf(pdf_path: str, debug: bool = False) -> Dict[str, Any]:
    """解析单个PDF文件

    Returns:
        {
            'product_code': str | None,
            'fees': {...},
            'parse_method': 'table' | 'text' | 'failed',
            'fee_section': str,  # 原文摘录（供调试）
        }
    """
    # 提取PDF内容
    content = extract_pdf_content(pdf_path)

    if not content['text'] and not content['tables']:
        return {'product_code': None, 'fees': {}, 'parse_method': 'failed', 'fee_section': ''}

    # 提取产品代码
    product_code = extract_product_code(content['text'])

    # 提取费用段落（用于调试和回退）
    fee_section = extract_fee_section(content['text'])

    if debug:
        logger.info(f"产品代码: {product_code}")
        logger.info(f"表格数量: {len(content['tables'])}")
        logger.info(f"费用段落长度: {len(fee_section)}")

    # 策略1: 优先表格解析
    fees = parse_fees_from_tables(content['tables'])

    if fees:
        if debug:
            logger.info(f"表格解析成功: {fees}")
        return {
            'product_code': product_code,
            'fees': fees,
            'parse_method': 'table',
            'fee_section': fee_section[:2000]
        }

    # 策略2: 回退文本解析
    fees = parse_fees_from_text(content['text'])

    if fees:
        if debug:
            logger.info(f"文本解析成功: {fees}")
        return {
            'product_code': product_code,
            'fees': fees,
            'parse_method': 'text',
            'fee_section': fee_section[:2000]
        }

    # 解析失败
    if debug:
        logger.warning(f"解析失败，费用段落: {fee_section[:500]}")

    return {
        'product_code': product_code,
        'fees': {},
        'parse_method': 'failed',
        'fee_section': fee_section[:2000]
    }


# ============================================================
# 7. 批量处理
# ============================================================

def load_index() -> Dict:
    """加载PDF索引"""
    if not os.path.exists(INDEX_FILE):
        logger.error(f"索引文件不存在: {INDEX_FILE}")
        logger.error("请先运行 download_citic_pdfs.py 下载PDF")
        return {}

    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_nav_products() -> Dict[str, str]:
    """加载NAV产品列表（用于匹配）"""
    try:
        from nav_db_excel import NavDBReader
        reader = NavDBReader()
        nav_products = {}

        if '中信银行' in reader.sheet_names:
            df = reader.read_sheet('中信银行')
            if '产品代码' in df.columns and '产品名称' in df.columns:
                for _, row in df[['产品代码', '产品名称']].drop_duplicates().iterrows():
                    code = str(row['产品代码']).strip()
                    name = str(row['产品名称']).strip()
                    if code and code != 'nan':
                        nav_products[code] = name

        logger.info(f"[NAV] 加载 {len(nav_products)} 个中信产品")
        return nav_products
    except Exception as e:
        logger.warning(f"[NAV] 加载失败: {e}")
        return {}


def match_product_code(parsed_code: Optional[str], title: str, nav_products: Dict[str, str]) -> Optional[str]:
    """匹配产品代码到NAV数据库

    匹配策略:
    1. PDF产品代码直接匹配（如 AF252548）
    2. 产品名称精确匹配（含产品号）
    3. 产品系列+编号匹配（如 "安盈象固收日开37号"）
    """
    # 策略1: 直接代码匹配
    if parsed_code and parsed_code in nav_products:
        return parsed_code

    # 清理PDF标题
    clean_title = re.sub(r'[-—]\s*产品说明书.*$|[-—]\s*\d{8}$', '', title).strip()
    clean_title = re.sub(r'^信银理财\s*', '', clean_title).strip()
    clean_title = re.sub(r'理财产品$', '', clean_title).strip()

    if not clean_title:
        return None

    # 策略2: 精确名称匹配
    for code, name in nav_products.items():
        # 清理NAV名称
        clean_name = re.sub(r'[（(][^）)]*[）)]$', '', name).strip()
        if clean_title == clean_name:
            return code

    # 策略3: 提取产品系列和编号，精确匹配
    # 例如: "安盈象固收日开37号" -> 系列="安盈象固收日开", 编号="37"
    m = re.match(r'^(.+?)(\d+)号?(?:理财产品)?$', clean_title)
    if m:
        series = m.group(1).strip()
        num = m.group(2)

        for code, name in nav_products.items():
            clean_name = re.sub(r'[（(][^）)]*[）)]$', '', name).strip()
            m2 = re.match(r'^(.+?)(\d+)号?', clean_name)
            if m2:
                nav_series = m2.group(1).strip()
                nav_num = m2.group(2)
                # 系列和编号都匹配
                if series == nav_series and num == nav_num:
                    return code

    # 策略4: 宽松系列匹配（仅当系列名足够长且唯一时）
    if m:
        series = m.group(1).strip()
        if len(series) >= 6:  # 系列名至少6字
            matches = []
            for code, name in nav_products.items():
                if series in name:
                    matches.append(code)
            # 仅当唯一匹配时返回
            if len(matches) == 1:
                return matches[0]

    return None  # 无法匹配


def process_all_pdfs(max_count: Optional[int] = None, force: bool = False, debug: bool = False):
    """批量处理所有PDF"""

    # 加载索引
    index = load_index()
    if not index:
        return

    # 加载NAV产品
    nav_products = load_nav_products()

    # 加载费率数据库
    load_fee_db()

    # 统计
    stats = {
        'total': 0,
        'parsed_table': 0,
        'parsed_text': 0,
        'matched': 0,
        'has_redemption': 0,
        'skipped': 0,
        'failed': 0,
    }

    failed_records = []  # 解析失败记录

    doc_ids = list(index.keys())
    total = len(doc_ids) if max_count is None else min(len(doc_ids), max_count)

    logger.info(f"开始解析 {total} 个PDF文件...")

    for i, doc_id in enumerate(doc_ids[:total]):
        stats['total'] += 1
        info = index[doc_id]
        title = info.get('title', '')

        pdf_path = os.path.join(PDF_DIR, f"{doc_id}.pdf")

        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            stats['failed'] += 1
            continue

        # 解析PDF
        result = parse_single_pdf(pdf_path, debug=debug)

        if result['parse_method'] == 'failed':
            stats['failed'] += 1
            failed_records.append({
                'doc_id': doc_id,
                'title': title,
                'fee_section': result.get('fee_section', '')[:500]
            })
            continue

        if result['parse_method'] == 'table':
            stats['parsed_table'] += 1
        else:
            stats['parsed_text'] += 1

        # 匹配产品代码
        matched_code = match_product_code(result['product_code'], title, nav_products)

        if matched_code and matched_code in nav_products:
            stats['matched'] += 1

            # 检查是否已有数据
            if not force:
                existing = get_fee_info('中信银行', matched_code)
                if existing and existing.get('source') == 'prospectus':
                    stats['skipped'] += 1
                    continue

            # 构建费率信息
            fees = result['fees']
            redemption_schedule = fees.get('redemption_schedule', [])
            if not redemption_schedule and fees.get('redemption_fee') == 0.0:
                redemption_schedule = []

            fee_info = {
                'has_redemption_fee': bool(redemption_schedule),
                'fee_schedule': redemption_schedule or [],
                'fee_description': result.get('fee_section', ''),
                'source': 'prospectus',
                'prospectus_title': title,
                'prospectus_date': info.get('date', ''),
                'subscription_fee': fees.get('subscription_fee'),
                'purchase_fee': fees.get('purchase_fee'),
                'sales_service_fee': fees.get('sales_service_fee'),
                'custody_fee': fees.get('custody_fee'),
                'management_fee': fees.get('management_fee'),
                'parse_method': result['parse_method'],
            }

            update_fee_info('中信银行', matched_code, fee_info)

            if fee_info['has_redemption_fee']:
                stats['has_redemption'] += 1

        # 进度
        if (i + 1) % 200 == 0:
            save_fee_db()
            logger.info(
                f"[进度] {i+1}/{total} - "
                f"表格:{stats['parsed_table']} 文本:{stats['parsed_text']} "
                f"匹配:{stats['matched']} 失败:{stats['failed']}"
            )

    # 最终保存
    save_fee_db()

    # 保存失败记录
    if failed_records:
        with open(FAILED_LOG, 'w', encoding='utf-8') as f:
            json.dump(failed_records, f, ensure_ascii=False, indent=2)
        logger.info(f"解析失败记录已保存到: {FAILED_LOG}")

    # 打印统计
    print("\n" + "=" * 60)
    print("  中信理财说明书PDF解析完成")
    print("=" * 60)
    print(f"  总PDF数:        {stats['total']}")
    print(f"  表格解析成功:   {stats['parsed_table']}")
    print(f"  文本解析成功:   {stats['parsed_text']}")
    print(f"  匹配NAV产品:    {stats['matched']}")
    print(f"  有赎回费:       {stats['has_redemption']}")
    print(f"  跳过(已有数据): {stats['skipped']}")
    print(f"  解析失败:       {stats['failed']}")
    print("=" * 60)

    return stats


# ============================================================
# CLI入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='中信理财说明书PDF本地解析器')
    parser.add_argument('--max', type=int, default=None, help='限制解析数量')
    parser.add_argument('--force', action='store_true', help='强制重新解析已有数据的产品')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--test', type=str, default=None, help='测试单个PDF文件')
    args = parser.parse_args()

    if pdfplumber is None:
        print("错误: pdfplumber未安装，请运行: pip install pdfplumber")
        return 1

    print("=" * 60)
    print("  中信理财说明书PDF本地解析器")
    print("=" * 60)

    # 测试单个文件
    if args.test:
        if os.path.exists(args.test):
            result = parse_single_pdf(args.test, debug=True)
            print(f"\n解析结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"文件不存在: {args.test}")
        return 0

    # 批量处理
    process_all_pdfs(max_count=args.max, force=args.force, debug=args.debug)

    return 0


if __name__ == '__main__':
    sys.exit(main())
