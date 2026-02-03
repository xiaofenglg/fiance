# -*- coding: utf-8 -*-
"""
中信理财说明书PDF批量下载器

只负责下载，不解析。下载到 pdfs/中信/{doc_id}.pdf
支持断点续传（已存在的文件跳过）

使用方法:
    python download_citic_pdfs.py              # 下载全部
    python download_citic_pdfs.py --max 100    # 限量测试
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from datetime import datetime

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_citic_pdfs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '中信')
PROGRESS_FILE = os.path.join(BASE_DIR, 'download_citic_progress.json')

SEARCH_URL = 'https://www.citic-wealth.com/was5/web/search'
DOCUMENT_URL = 'https://www.citic-wealth.com/was5/web/document'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://www.citic-wealth.com/',
}


def fetch_prospectus_list(keyword='', max_pages=None):
    """获取说明书列表"""
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
            payload = json.loads(resp.text)
        except Exception as e:
            logger.warning(f"[列表] 第{page}页请求失败: {e}")
            break

        if payload.get('msg') != 'success':
            break

        data = payload.get('data', [])
        if not data:
            break

        if total_pages is None:
            total_pages = int(payload.get('countpage', 0))
            total_count = int(payload.get('count', 0))
            logger.info(f"[列表] 总记录数: {total_count}, 总页数: {total_pages}")

        for item in data:
            doc_id = item.get('xpSequenceNo', '')
            if not doc_id:
                continue
            all_items.append({
                'doc_id': str(doc_id),
                'title': item.get('title', '').strip(),
                'url': item.get('url', '').strip(),
                'date': item.get('date', '').strip(),
            })

        if page % 50 == 0:
            logger.info(f"[列表] 已爬取 {page} 页, 累计 {len(all_items)} 条")

        page += 1
        time.sleep(random.uniform(0.2, 0.4))

    logger.info(f"[列表] 爬取完成: 共 {len(all_items)} 条")
    return all_items


def download_pdf(item, session):
    """下载单个PDF，返回是否成功"""
    doc_id = item['doc_id']
    url_field = item['url']

    pdf_path = os.path.join(PDF_DIR, f"{doc_id}.pdf")

    # 已存在则跳过
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
        return 'skip'

    pdf_url = f"{DOCUMENT_URL}?columnname=file_path_new&multino=1&downloadtype=open&channelid={url_field}"

    try:
        resp = session.get(pdf_url, timeout=30)
        resp.raise_for_status()

        if len(resp.content) < 1000:
            return 'fail'

        with open(pdf_path, 'wb') as f:
            f.write(resp.content)

        return 'ok'
    except Exception as e:
        logger.debug(f"[下载] {doc_id} 失败: {e}")
        return 'fail'


def save_index(items):
    """保存索引文件（doc_id -> 元信息映射）"""
    index_path = os.path.join(PDF_DIR, 'index.json')
    index = {}
    for item in items:
        index[item['doc_id']] = {
            'title': item['title'],
            'date': item['date'],
            'url': item['url'],
        }
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logger.info(f"[索引] 已保存 {len(index)} 条记录到 {index_path}")


def main():
    parser = argparse.ArgumentParser(description='中信理财说明书PDF批量下载')
    parser.add_argument('--max', type=int, default=None, help='限制下载数量')
    parser.add_argument('--keyword', type=str, default='', help='搜索关键词')
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)

    print("=" * 60)
    print("  中信理财说明书PDF批量下载器")
    print("=" * 60)

    # 1. 获取列表
    items = fetch_prospectus_list(keyword=args.keyword)
    if not items:
        print("未获取到任何说明书记录")
        return 1

    # 保存索引
    save_index(items)

    # 2. 批量下载
    session = requests.Session()
    session.headers.update(HEADERS)

    stats = {'ok': 0, 'skip': 0, 'fail': 0}
    total = len(items) if args.max is None else min(len(items), args.max)

    for i, item in enumerate(items[:total]):
        result = download_pdf(item, session)
        stats[result] += 1

        if (i + 1) % 100 == 0:
            logger.info(f"[进度] {i+1}/{total} - 成功:{stats['ok']} 跳过:{stats['skip']} 失败:{stats['fail']}")

        if result == 'ok':
            time.sleep(random.uniform(0.3, 0.6))

    print("\n" + "=" * 60)
    print(f"  下载完成")
    print(f"  成功: {stats['ok']}")
    print(f"  跳过(已存在): {stats['skip']}")
    print(f"  失败: {stats['fail']}")
    print(f"  PDF目录: {PDF_DIR}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
