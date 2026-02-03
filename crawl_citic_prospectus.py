# -*- coding: utf-8 -*-
"""
中信理财产品说明书爬虫 — 从PDF说明书中提取赎回费数据

功能:
1. 从中信理财信息披露页抓取产品说明书列表
2. 下载PDF说明书，提取费用条款
3. 解析赎回费阶梯结构
4. 匹配到NAV数据库中的产品代码
5. 写入赎回费数据库

API:
- 列表: POST https://www.citic-wealth.com/was5/web/search?channelid=204182&page=N&searchword=
- PDF:  GET  https://www.citic-wealth.com/was5/web/document?columnname=file_path_new&multino=1&downloadtype=open&channelid={url字段}

使用方法:
    python crawl_citic_prospectus.py                        # 爬取活跃产品
    python crawl_citic_prospectus.py --keyword "睿享同行"   # 按关键词过滤
    python crawl_citic_prospectus.py --max 10               # 限量调试
    python crawl_citic_prospectus.py --force                # 强制重新处理
"""

import os
import re
import sys
import json
import time
import random
import logging
import argparse
from io import BytesIO
from datetime import datetime
from collections import defaultdict

import requests

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from redemption_fee_db import (
    load_fee_db, save_fee_db, update_fee_info, has_fee_data, get_fee_info
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_citic_prospectus.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(BASE_DIR, 'prospectus_progress.json')

SEARCH_URL = 'https://www.citic-wealth.com/was5/web/search'
DOCUMENT_URL = 'https://www.citic-wealth.com/was5/web/document'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://www.citic-wealth.com/',
}

# 产品代码正则: 2个大写字母 + 5-8位数字 + 可选大写字母
PRODUCT_CODE_RE = re.compile(r'\b([A-Z]{2}\d{5,8}[A-Z]?)\b')

# 说明书标题中提取产品名的正则 — 去掉后缀
TITLE_CLEAN_RE = re.compile(
    r'[-—]?\s*产品说明书.*$|[-—]?\s*\d{8}\s*$|[-—]?\s*说明书.*$',
    re.IGNORECASE
)


# ============================================================
# 1A: 列表爬取
# ============================================================

def fetch_prospectus_list(keyword='', max_pages=None, delay_range=(0.3, 0.5)):
    """分页爬取产品说明书列表

    POST /was5/web/search?channelid=204182&page=N&searchword={keyword}

    API返回JSON:
        {"msg":"success", "data":[{"title":..., "url":..., "date":..., "xpSequenceNo":...}],
         "page":"1", "count":"8478", "countpage":"848"}

    Returns:
        list[dict]: [{'title': ..., 'url': ..., 'date': ..., 'doc_id': ...}, ...]
    """
    all_items = []
    page = 1
    total_pages = None
    session = requests.Session()
    session.headers.update(HEADERS)

    while True:
        if max_pages and page > max_pages:
            break
        if total_pages is not None and page > total_pages:
            break

        params = {
            'channelid': '204182',
            'page': str(page),
            'searchword': keyword,
        }

        try:
            resp = session.post(SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"[列表] 第{page}页请求失败: {e}")
            break

        # 解析JSON响应
        try:
            payload = json.loads(resp.text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[列表] 第{page}页JSON解析失败: {e}")
            break

        if payload.get('msg') != 'success':
            logger.warning(f"[列表] 第{page}页返回非success: {payload.get('msg')}")
            break

        data = payload.get('data', [])
        if not data:
            logger.info(f"[列表] 第{page}页无数据，爬取结束")
            break

        # 首页获取总页数
        if total_pages is None:
            try:
                total_pages = int(payload.get('countpage', 0))
                total_count = int(payload.get('count', 0))
                logger.info(f"[列表] 总记录数: {total_count}, 总页数: {total_pages}")
            except (ValueError, TypeError):
                pass

        for item in data:
            title = item.get('title', '').strip()
            url_field = item.get('url', '').strip()
            date_str = item.get('date', '').strip()
            doc_id = item.get('xpSequenceNo', '') or _extract_doc_id(url_field)

            if not title:
                continue

            all_items.append({
                'title': title,
                'url': url_field,
                'date': date_str,
                'doc_id': str(doc_id),
            })

        if page % 50 == 0:
            logger.info(f"[列表] 已爬取 {page} 页, 累计 {len(all_items)} 条")

        page += 1
        time.sleep(random.uniform(*delay_range))

    logger.info(f"[列表] 爬取完成: 共 {len(all_items)} 条说明书记录, {page - 1} 页")
    return all_items


def _extract_doc_id(url_field):
    """从URL字段提取文档ID"""
    # 格式: "243547&searchword=id=XXXX" 或 "channelid=XXXX"
    m = re.search(r'id=(\d+)', url_field)
    if m:
        return m.group(1)
    # 备选: 直接用URL字段的数字部分
    m = re.search(r'(\d{4,})', url_field)
    if m:
        return m.group(1)
    return url_field


def deduplicate_by_product(items):
    """按产品名分组，每个产品仅保留最新一份说明书

    Args:
        items: fetch_prospectus_list() 的结果

    Returns:
        list[dict]: 去重后的列表
    """
    by_product = defaultdict(list)

    for item in items:
        # 从标题提取产品名 — 去掉"产品说明书"后缀和日期
        product_name = TITLE_CLEAN_RE.sub('', item['title']).strip()
        # 去掉信银理财前缀（如果有）
        product_name = re.sub(r'^信银理财\s*', '', product_name).strip()
        item['_product_name'] = product_name
        by_product[product_name].append(item)

    result = []
    for name, group in by_product.items():
        # 按日期降序排列，取最新
        group.sort(key=lambda x: x.get('date', ''), reverse=True)
        result.append(group[0])

    logger.info(f"[去重] {len(items)} 条 → {len(result)} 个不同产品")
    return result


# ============================================================
# 1B: PDF下载与解析
# ============================================================

PDF_CACHE_DIR = os.path.join(BASE_DIR, 'pdfs', '中信')


def download_and_parse_pdf(doc_url_field, session=None, doc_id=None):
    """下载PDF并提取费用条款（支持本地缓存）

    Args:
        doc_url_field: 说明书列表中的url字段，用于构造PDF下载URL
        session: 可复用的requests.Session
        doc_id: 文档ID，用于本地缓存文件名

    Returns:
        dict: {
            'product_code': str | None,
            'product_name': str,
            'fees': {...},
            'fee_description': str,
            'raw_text': str,
        }
        or None if failed
    """
    if pdfplumber is None:
        logger.error("pdfplumber未安装，无法解析PDF。请运行: pip install pdfplumber")
        return None

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)

    # 确保缓存目录存在
    os.makedirs(PDF_CACHE_DIR, exist_ok=True)

    # 本地缓存文件路径
    cache_file = None
    if doc_id:
        cache_file = os.path.join(PDF_CACHE_DIR, f"{doc_id}.pdf")

    pdf_content = None

    # 优先从本地缓存读取
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                pdf_content = f.read()
            if len(pdf_content) < 1000:
                logger.warning(f"[PDF] 缓存文件过小，重新下载: {cache_file}")
                pdf_content = None
        except Exception as e:
            logger.warning(f"[PDF] 读取缓存失败: {e}")
            pdf_content = None

    # 缓存不存在或无效，从网络下载
    if pdf_content is None:
        pdf_url = f"{DOCUMENT_URL}?columnname=file_path_new&multino=1&downloadtype=open&channelid={doc_url_field}"
        try:
            resp = session.get(pdf_url, timeout=30)
            resp.raise_for_status()
            if len(resp.content) < 1000:
                logger.warning(f"[PDF] 内容过小({len(resp.content)}字节)，可能不是PDF")
                return None
            pdf_content = resp.content

            # 保存到本地缓存
            if cache_file:
                try:
                    with open(cache_file, 'wb') as f:
                        f.write(pdf_content)
                except Exception as e:
                    logger.warning(f"[PDF] 保存缓存失败: {e}")

        except Exception as e:
            logger.warning(f"[PDF] 下载失败: {e}")
            return None

    # 用pdfplumber解析
    try:
        pdf_bytes = BytesIO(pdf_content)
        full_text = ''
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + '\n'
    except Exception as e:
        logger.warning(f"[PDF] 解析失败: {e}")
        return None

    if not full_text.strip():
        logger.warning("[PDF] 提取文本为空")
        return None

    # 提取产品代码
    product_code = _extract_product_code(full_text)

    # 提取费用条款段
    fee_section = _extract_fee_section(full_text)

    # 解析各类费用
    fees = {}
    fee_description = ''

    if fee_section:
        fees['赎回费'] = parse_redemption_fee(fee_section)
        fees['认购费'] = _parse_simple_fee(fee_section, '认购费')
        fees['申购费'] = _parse_simple_fee(fee_section, '申购费')
        fees['销售服务费'] = _parse_rate_fee(fee_section, '销售服务费')
        fees['托管费'] = _parse_rate_fee(fee_section, '托管费')
        fees['管理费'] = _parse_rate_fee(fee_section, '(?:固定)?管理费')
        fee_description = fee_section[:2000]  # 保留原文摘录
    else:
        # 在全文中搜索赎回费相关内容
        fees['赎回费'] = parse_redemption_fee(full_text)

    return {
        'product_code': product_code,
        'product_name': '',  # 由调用方从标题设定
        'fees': fees,
        'fee_description': fee_description,
        'raw_text': fee_section or '',
    }


def _extract_product_code(text):
    """从PDF正文提取产品代码"""
    # 搜索"产品代码"、"产品编码"附近的代码
    context_pattern = re.compile(
        r'(?:产品代码|产品编码|登记编码)[：:\s]*([A-Z]{2}\d{5,8}[A-Z]?)',
        re.IGNORECASE
    )
    m = context_pattern.search(text)
    if m:
        return m.group(1)

    # 退化搜索: 全文中第一个符合格式的代码
    m = PRODUCT_CODE_RE.search(text)
    if m:
        return m.group(1)

    return None


def _extract_fee_section(text):
    """从全文中定位费用条款段落

    搜索"费用"章节，提取从"费用"标题到下一章节标题之间的文本。
    """
    # 尝试定位费用章节
    fee_section_patterns = [
        # "（四）产品费用" 或 "四、产品费用"
        re.compile(r'[（(]?\s*[四五六七八九十\d]+\s*[）)]?\s*[、.]?\s*(?:产品)?费用.*?(?=[（(]\s*[四五六七八九十\d]+\s*[）)]|[四五六七八九十]{1,2}\s*[、]|第[四五六七八九十]+[章节部分]|\Z)', re.DOTALL),
        # "费用说明" 段
        re.compile(r'费用(?:说明|概述|结构).*?(?=第[四五六七八九十]+[章节部分]|[四五六七八九十]{1,2}\s*[、]|\Z)', re.DOTALL),
        # 宽泛: 从"认购费"或"赎回费"开始，提取一大段
        re.compile(r'(?:认购费|申购费|赎回费|管理费|托管费|销售服务费).{0,3000}?(?:赎回费|管理费|托管费).{0,1500}', re.DOTALL),
    ]

    for pat in fee_section_patterns:
        m = pat.search(text)
        if m:
            section = m.group(0)
            if len(section) > 100:  # 确保有足够内容
                return section

    # 退化: 搜索包含"赎回费"的较大上下文
    idx = text.find('赎回费')
    if idx >= 0:
        start = max(0, idx - 500)
        end = min(len(text), idx + 2000)
        return text[start:end]

    return None


def _parse_simple_fee(text, fee_name):
    """解析简单费用（如认购费、申购费）

    兼容格式:
    - "认购费率0.00%"
    - "认购费 0%"
    - "认购费：费率【0.00%】"
    """
    # 先尝试"费率【X%】"格式 (如: 认购费：费率【0.00%】)
    pattern_bracket = re.compile(
        rf'{fee_name}[：:\s]*费率[【\s]*([\d.]+)\s*%'
    )
    m = pattern_bracket.search(text)
    if m:
        return float(m.group(1)) / 100

    # 再尝试"费率X%"格式 (如: 认购费率0.00%)
    pattern_plain = re.compile(
        rf'{fee_name}[率为：:\s]*[【\s]*([\d.]+)\s*%'
    )
    m = pattern_plain.search(text)
    if m:
        return float(m.group(1)) / 100

    # "不收取认购费"
    if re.search(rf'不收取{fee_name}|{fee_name}为?\s*0|免收{fee_name}', text):
        return 0.0

    return None


def _parse_rate_fee(text, fee_name):
    """解析年化费率（如管理费、托管费、销售服务费）

    兼容格式:
    - "固定管理费率0.30%/年"
    - "托管费率 0.03%"
    - "销售服务费：费率【0.40%】/年。"
    - "固定管理费：费率【0.20%】/年。"
    """
    # 先尝试"费率【X%】"格式 (如: 销售服务费：费率【0.40%】/年)
    pattern_bracket = re.compile(
        rf'{fee_name}[：:\s]*费率[【\s]*([\d.]+)\s*%'
    )
    m = pattern_bracket.search(text)
    if m:
        return float(m.group(1)) / 100

    # 再尝试"费率X%"格式 (如: 固定管理费率0.30%/年)
    pattern_plain = re.compile(
        rf'{fee_name}[率为：:\s]*[【\s]*([\d.]+)\s*%'
    )
    m = pattern_plain.search(text)
    if m:
        return float(m.group(1)) / 100
    return None


# ============================================================
# 1C: 赎回费解析
# ============================================================

def parse_redemption_fee(text):
    """从费用条款文本解析赎回费阶梯

    Returns:
        list[dict]: [{'min_days': N, 'max_days': M, 'fee_rate': X}, ...]
        空列表表示无赎回费
    """
    if not text:
        return []

    # 检查: 明确无赎回费
    no_fee_patterns = [
        r'不收取赎回费',
        r'赎回费[率为：:\s]*0[\.0]*\s*%',
        r'赎回费[率为：:\s]*无',
        r'无赎回费',
        r'免收赎回费',
    ]
    for pat in no_fee_patterns:
        if re.search(pat, text):
            return []

    schedule = []

    # 模式1: "持有不满N天...赎回费率X%"
    pattern_under = re.compile(
        r'持有(?:期限)?不满\s*(\d+)\s*(?:天|个自然日|日|个工作日)'
        r'[^%]*?赎回费率?\s*([\d.]+)\s*%',
        re.DOTALL
    )
    # 模式2: "持有满N天...赎回费率X%"
    pattern_over = re.compile(
        r'持有(?:期限)?满\s*(\d+)\s*(?:天|个自然日|日|个工作日)'
        r'[^%]*?赎回费率?\s*([\d.]+)\s*%',
        re.DOTALL
    )
    # 模式3: "持有少于N天/日...赎回费率X%"
    pattern_less = re.compile(
        r'(?:持有|连续持有)\s*(?:少于|不足|不到)\s*(\d+)\s*(?:天|个自然日|日|个工作日)'
        r'[^%]*?赎回费率?\s*([\d.]+)\s*%',
        re.DOTALL
    )
    # 模式4: "N天以内...赎回费率X%"
    pattern_within = re.compile(
        r'(\d+)\s*(?:天|日|个自然日)以内'
        r'[^%]*?赎回费率?\s*([\d.]+)\s*%',
        re.DOTALL
    )
    # 模式5: "N天（含）以上...赎回费率X%"
    pattern_above = re.compile(
        r'(\d+)\s*(?:天|日|个自然日)\s*[（(]?\s*含?\s*[）)]?\s*以上'
        r'[^%]*?赎回费率?\s*([\d.]+)\s*%',
        re.DOTALL
    )

    # 模式6: 表格格式 "N日≤持有期限<M天：X%" 或 "N天≤持有天数＜M天：X%"
    # 支持全角和半角符号
    pattern_table_range = re.compile(
        r'(\d+)\s*[日天]\s*[≤<＜]\s*持有[期限天数]+\s*[<＜≤]\s*(\d+)\s*[日天]\s*[：:]\s*([\d.]+)\s*%'
    )
    # 模式7: 表格格式 "持有期限≥N天：X%" 或 "持有天数>=N天：X%"
    pattern_table_above = re.compile(
        r'持有[期限天数]+\s*[≥>＞]=?\s*(\d+)\s*[日天]\s*[：:]\s*([\d.]+)\s*%'
    )
    # 模式8: 表格格式 "持有期限<N天：X%" 或 "持有天数＜N天：X%"
    pattern_table_under = re.compile(
        r'持有[期限天数]+\s*[<＜]\s*(\d+)\s*[日天]\s*[：:]\s*([\d.]+)\s*%'
    )

    # 收集所有匹配的阈值
    thresholds = {}  # {days: (type, rate)}

    for m in pattern_under.finditer(text):
        days = int(m.group(1))
        rate = float(m.group(2)) / 100
        thresholds[days] = ('under', rate)

    for m in pattern_over.finditer(text):
        days = int(m.group(1))
        rate = float(m.group(2)) / 100
        if days not in thresholds:
            thresholds[days] = ('over', rate)
        else:
            # 同一天数有under和over，over是独立阶梯
            thresholds[(days, 'over')] = ('over', rate)

    for m in pattern_less.finditer(text):
        days = int(m.group(1))
        rate = float(m.group(2)) / 100
        thresholds[days] = ('under', rate)

    for m in pattern_within.finditer(text):
        days = int(m.group(1))
        rate = float(m.group(2)) / 100
        thresholds[days] = ('under', rate)

    for m in pattern_above.finditer(text):
        days = int(m.group(1))
        rate = float(m.group(2)) / 100
        if days not in thresholds:
            thresholds[days] = ('over', rate)
        else:
            thresholds[(days, 'over')] = ('over', rate)

    # 模式6: 表格格式 "N日≤持有期限<M天：X%"
    has_range_match = False
    for m in pattern_table_range.finditer(text):
        min_days = int(m.group(1))
        max_days = int(m.group(2))
        rate = float(m.group(3)) / 100
        schedule.append({'min_days': min_days, 'max_days': max_days, 'fee_rate': rate})
        has_range_match = True

    # 模式7: 表格格式 "持有期限≥N天：X%"
    for m in pattern_table_above.finditer(text):
        days = int(m.group(1))
        rate = float(m.group(2)) / 100
        schedule.append({'min_days': days, 'max_days': 999999, 'fee_rate': rate})

    # 模式8: 表格格式 "持有期限<N天：X%" - 仅当没有范围匹配时使用
    # 因为范围匹配会把 "持有天数＜7" 当作子串误匹配
    if not has_range_match:
        for m in pattern_table_under.finditer(text):
            days = int(m.group(1))
            rate = float(m.group(2)) / 100
            schedule.append({'min_days': 0, 'max_days': days, 'fee_rate': rate})

    # 如果通过表格格式找到了费率，直接返回
    if schedule:
        # 去重：使用(min_days, max_days)作为key
        seen = {}
        for entry in schedule:
            key = (entry['min_days'], entry['max_days'])
            if key not in seen:
                seen[key] = entry
        schedule = list(seen.values())
        # 按min_days排序
        schedule.sort(key=lambda x: x['min_days'])
        return schedule

    if not thresholds:
        # 短期惩罚条款检测: "连续持有少于7日...赎回费"
        short_penalty = re.search(
            r'(?:连续)?持有\s*(?:少于|不足|不满)\s*(\d+)\s*(?:天|日)',
            text
        )
        if short_penalty and '赎回费' in text:
            days = int(short_penalty.group(1))
            # 有赎回费但费率未明确，使用默认
            return [
                {'min_days': 0, 'max_days': days, 'fee_rate': 0.015},
                {'min_days': days, 'max_days': 999999, 'fee_rate': 0.0},
            ]
        return []

    # 构建阶梯结构
    schedule = _build_schedule_from_thresholds(thresholds)
    return schedule


def _build_schedule_from_thresholds(thresholds):
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
        # 查找"不满day天"的费率 (under)
        under_rate = None
        over_rate = None
        for d, typ, rate in entries:
            if d == day and typ == 'under':
                under_rate = rate
            if d == day and typ == 'over':
                over_rate = rate

        if under_rate is not None:
            schedule.append({
                'min_days': prev_day,
                'max_days': day,
                'fee_rate': under_rate,
            })
            prev_day = day

        if over_rate is not None and i == len(boundaries) - 1:
            # 最后一个边界的"满N天"
            schedule.append({
                'min_days': day,
                'max_days': 999999,
                'fee_rate': over_rate,
            })

    # 如果没有明确的"满最后一个阈值天"条目，补充0费率
    if schedule and schedule[-1]['max_days'] != 999999:
        last_max = schedule[-1]['max_days']
        # 检查有没有over条目
        has_over = any(typ == 'over' for _, typ, _ in entries)
        if not has_over:
            schedule.append({
                'min_days': last_max,
                'max_days': 999999,
                'fee_rate': 0.0,
            })

    return schedule


# ============================================================
# 1D: 产品代码匹配
# ============================================================

def match_product_code(title, pdf_text, nav_products):
    """将说明书匹配到NAV数据库中的产品代码

    Args:
        title: 说明书标题
        pdf_text: PDF中提取的文本（可为空）
        nav_products: dict, {product_code: product_name} 从NAV数据库读取

    Returns:
        str | None: 匹配到的产品代码
    """
    if not nav_products:
        return None

    # 策略1: 从PDF正文中提取的产品代码直接匹配
    if pdf_text:
        code = _extract_product_code(pdf_text)
        if code and code in nav_products:
            return code

    # 策略2: 标题名称子串匹配
    # 清洗标题: 去掉"产品说明书"、日期等后缀
    clean_title = TITLE_CLEAN_RE.sub('', title).strip()
    clean_title = re.sub(r'^信银理财\s*', '', clean_title).strip()

    if not clean_title:
        return None

    # 精确子串匹配
    for code, name in nav_products.items():
        if clean_title in name or name in clean_title:
            return code

    # 模糊匹配: 去掉常见变体后再匹配
    # 去掉"理财产品"、"产品"、空格、括号内容
    def normalize(s):
        s = re.sub(r'[（(][^）)]*[）)]', '', s)
        s = re.sub(r'理财产品|产品|理财', '', s)
        s = re.sub(r'\s+', '', s)
        return s

    norm_title = normalize(clean_title)
    if len(norm_title) < 4:
        return None

    best_match = None
    best_score = 0
    for code, name in nav_products.items():
        norm_name = normalize(name)
        # 计算最长公共子串比例
        score = _lcs_ratio(norm_title, norm_name)
        if score > best_score and score > 0.6:
            best_score = score
            best_match = code

    return best_match


def _lcs_ratio(s1, s2):
    """计算两个字符串的最长公共子串占较短串的比例"""
    if not s1 or not s2:
        return 0.0
    shorter = min(len(s1), len(s2))
    if shorter == 0:
        return 0.0

    # 简化: 用连续公共子序列
    max_len = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            k = 0
            while (i + k < len(s1) and j + k < len(s2)
                   and s1[i + k] == s2[j + k]):
                k += 1
            if k > max_len:
                max_len = k

    return max_len / shorter


def _load_nav_products():
    """从NAV数据库加载中信银行产品列表

    Returns:
        dict: {product_code: product_name}
    """
    try:
        from nav_db_excel import NavDBReader
        reader = NavDBReader()
        if '中信银行' not in reader.sheet_names:
            logger.warning("[NAV] 中信银行 sheet不存在")
            return {}
        df = reader.read_sheet('中信银行')
        products = {}
        if '产品代码' in df.columns and '产品名称' in df.columns:
            for _, row in df[['产品代码', '产品名称']].drop_duplicates().iterrows():
                code = str(row['产品代码']).strip()
                name = str(row['产品名称']).strip()
                if code and code != 'nan':
                    products[code] = name
        logger.info(f"[NAV] 加载中信银行 {len(products)} 个产品")
        return products
    except Exception as e:
        logger.warning(f"[NAV] 加载产品列表失败: {e}")
        return {}


# ============================================================
# 1E: 主函数
# ============================================================

def crawl_all_prospectuses(
    only_active=True,
    force_update=False,
    max_items=None,
    delay=0.5,
    keyword='',
    max_pages=None,
):
    """主入口: 爬取说明书 → 解析费率 → 写入赎回费数据库

    Args:
        only_active: 仅处理NAV数据库中有活跃数据的产品
        force_update: 强制重新处理已有费率的产品
        max_items: 限制处理的PDF数量（调试用）
        delay: 请求间隔秒数
        keyword: 搜索关键词过滤
        max_pages: 限制爬取的列表页数

    Returns:
        dict: 统计信息
    """
    stats = {
        'total_listed': 0,
        'after_dedup': 0,
        'processed': 0,
        'pdf_success': 0,
        'pdf_failed': 0,
        'matched': 0,
        'has_redemption_fee': 0,
        'no_redemption_fee': 0,
        'skipped_existing': 0,
        'updated': 0,
    }

    # 加载NAV数据库产品列表
    nav_products = _load_nav_products() if only_active else {}

    # 加载进度文件
    progress = _load_progress()

    # 加载赎回费数据库（确保缓存初始化）
    load_fee_db()

    # 1. 爬取说明书列表
    logger.info(f"[主流程] 开始爬取说明书列表 (keyword='{keyword}')")
    all_items = fetch_prospectus_list(keyword=keyword, max_pages=max_pages)
    stats['total_listed'] = len(all_items)

    # 2. 按产品去重，取最新
    items = deduplicate_by_product(all_items)
    stats['after_dedup'] = len(items)

    # 3. 如果仅处理活跃产品，优先排序NAV库中存在的产品
    if only_active and nav_products:
        def _priority(item):
            title = item.get('_product_name', item.get('title', ''))
            # 检查标题是否包含NAV库中的产品名
            for name in nav_products.values():
                if name in title or title in name:
                    return 0  # 高优先级
            return 1  # 低优先级
        items.sort(key=_priority)

    # 4. 逐一处理
    session = requests.Session()
    session.headers.update(HEADERS)

    processed_ids = set(progress.get('processed_doc_ids', []))

    for i, item in enumerate(items):
        if max_items and stats['processed'] >= max_items:
            logger.info(f"[主流程] 达到处理上限 {max_items}，停止")
            break

        doc_id = item.get('doc_id', '')
        title = item.get('title', '')

        # 断点续传: 跳过已处理
        if doc_id and doc_id in processed_ids:
            stats['skipped_existing'] += 1
            continue

        # 检查是否已有prospectus数据且不强制更新
        product_name_from_title = item.get('_product_name', '')
        if not force_update and product_name_from_title:
            # 尝试从标题匹配产品代码，检查是否已有数据
            matched_code = match_product_code(title, '', nav_products)
            if matched_code:
                existing = get_fee_info('中信银行', matched_code)
                if existing and existing.get('source') == 'prospectus':
                    stats['skipped_existing'] += 1
                    continue

        stats['processed'] += 1
        logger.info(f"[处理 {stats['processed']}] {title}")

        # 下载并解析PDF
        result = download_and_parse_pdf(item['url'], session=session, doc_id=doc_id)

        if result is None:
            stats['pdf_failed'] += 1
            # 记录已处理（即使失败），避免重复下载
            if doc_id:
                processed_ids.add(doc_id)
            time.sleep(delay)
            continue

        stats['pdf_success'] += 1
        result['product_name'] = product_name_from_title or title

        # 匹配产品代码
        matched_code = result.get('product_code')
        if matched_code and matched_code in nav_products:
            pass  # PDF中的代码直接命中
        else:
            matched_code = match_product_code(
                title, result.get('raw_text', ''), nav_products
            )

        if matched_code:
            stats['matched'] += 1
        elif only_active:
            # 仅处理活跃产品时，未匹配的跳过写入
            if doc_id:
                processed_ids.add(doc_id)
            time.sleep(delay)
            continue

        # 解析赎回费
        fees = result.get('fees', {})
        redemption_fees = fees.get('赎回费', [])

        fee_info = {
            'has_redemption_fee': len(redemption_fees) > 0,
            'fee_schedule': redemption_fees,
            'fee_description': result.get('fee_description', ''),
            'source': 'prospectus',
            'prospectus_title': title,
            'prospectus_date': item.get('date', ''),
            # 扩展费率字段
            'subscription_fee': fees.get('认购费'),
            'purchase_fee': fees.get('申购费'),
            'sales_service_fee': fees.get('销售服务费'),
            'custody_fee': fees.get('托管费'),
            'management_fee': fees.get('管理费'),
        }

        if fee_info['has_redemption_fee']:
            stats['has_redemption_fee'] += 1
        else:
            stats['no_redemption_fee'] += 1

        # 写入赎回费数据库
        product_code = matched_code or result.get('product_code', '') or doc_id
        if product_code:
            update_fee_info('中信银行', product_code, fee_info)
            stats['updated'] += 1

        # 记录已处理
        if doc_id:
            processed_ids.add(doc_id)

        # 定期保存进度（每200个产品存储一次，与民生对齐）
        if stats['processed'] % 200 == 0:
            _save_progress({
                'processed_doc_ids': list(processed_ids),
                'stats': stats,
                'last_time': datetime.now().isoformat(),
            })
            save_fee_db()
            logger.info(f"[进度] 已处理 {stats['processed']}, "
                        f"PDF成功 {stats['pdf_success']}, "
                        f"匹配 {stats['matched']}, "
                        f"有赎回费 {stats['has_redemption_fee']}")

        time.sleep(delay)

    # 最终保存
    _save_progress({
        'processed_doc_ids': list(processed_ids),
        'stats': stats,
        'last_time': datetime.now().isoformat(),
    })
    save_fee_db()

    _print_stats(stats)
    return stats


def _print_stats(stats):
    """打印统计报告"""
    print("\n" + "=" * 60)
    print("      中信理财说明书爬虫 — 统计报告")
    print("=" * 60)
    print(f"  说明书总数:       {stats['total_listed']}")
    print(f"  去重后产品数:     {stats['after_dedup']}")
    print(f"  本次处理:         {stats['processed']}")
    print(f"  PDF解析成功:      {stats['pdf_success']}")
    print(f"  PDF解析失败:      {stats['pdf_failed']}")
    print(f"  匹配到NAV产品:    {stats['matched']}")
    print(f"  有赎回费:         {stats['has_redemption_fee']}")
    print(f"  无赎回费:         {stats['no_redemption_fee']}")
    print(f"  跳过(已有数据):   {stats['skipped_existing']}")
    print(f"  写入费率数据库:   {stats['updated']}")
    print("=" * 60)


# ============================================================
# 1F: 进度追踪
# ============================================================

def _load_progress():
    """加载进度文件"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_progress(data):
    """保存进度文件"""
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存进度失败: {e}")


# ============================================================
# CLI入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='中信理财产品说明书爬虫 — 赎回费数据采集')
    parser.add_argument('--keyword', type=str, default='',
                        help='搜索关键词过滤 (如 "睿享同行")')
    parser.add_argument('--max', type=int, default=None,
                        help='限制处理PDF数量（调试用）')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='限制爬取列表页数')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理已有费率的产品')
    parser.add_argument('--all', action='store_true',
                        help='处理所有产品（不限于NAV数据库中的活跃产品）')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='请求间隔秒数 (默认0.5)')
    args = parser.parse_args()

    print("=" * 60)
    print("  中信理财产品说明书爬虫 — 赎回费数据采集")
    print("=" * 60)

    if pdfplumber is None:
        print("\n错误: pdfplumber未安装")
        print("请运行: pip install pdfplumber")
        sys.exit(1)

    stats = crawl_all_prospectuses(
        only_active=not args.all,
        force_update=args.force,
        max_items=args.max,
        delay=args.delay,
        keyword=args.keyword,
        max_pages=args.max_pages,
    )

    return 0 if stats.get('pdf_success', 0) > 0 or stats.get('skipped_existing', 0) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
