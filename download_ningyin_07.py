# -*- coding: utf-8 -*-
"""
宁银理财销售文件下载器 - 仅下载_07_格式的说明书

这些文件包含完整的费率信息。
预估约2000个文件。

用法:
    python download_ningyin_07.py              # 下载全部
    python download_ningyin_07.py --max 100    # 限量测试
"""

import os
import sys
import ssl
import time
import json
import logging
import argparse
from typing import Dict, List

import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_ningyin_07.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '宁银')
INDEX_FILE = os.path.join(PDF_DIR, 'sales_index.json')

BASE_URL = "https://www.wmbnb.com"
API_PREFIX = "/ningbo-web/;a"


def create_client():
    """创建httpx客户端"""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
    except AttributeError:
        pass

    return httpx.Client(
        verify=ctx,
        timeout=60,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': f'{BASE_URL}/info/funddetails/index.html',
        },
        follow_redirects=True,
    )


def scan_sales_documents(client: httpx.Client, max_pages: int = 5000) -> List[Dict]:
    """扫描全部_07_格式的销售文件

    Returns:
        [{title, url}, ...]
    """
    logger.info("扫描全部销售文件 (_07_ 格式)...")

    all_docs = []
    page = 1
    total = 0

    while page <= max_pages:
        try:
            r = client.get(
                f"{BASE_URL}{API_PREFIX}/product/attachmentlist.json",
                params={'request_num': 100, 'request_pageno': page}
            )

            if r.status_code != 200:
                break

            data = r.json()
            if data.get('status') != 'success':
                break

            docs = data.get('list', [])
            if not docs:
                break

            total = data.get('total', 0)

            # 筛选 _07_ 格式的销售文件
            for doc in docs:
                url = doc.get('url', '')
                title = doc.get('title', '')
                if '_07_' in url:
                    all_docs.append({
                        'title': title,
                        'url': url,
                    })

            if page % 200 == 0:
                logger.info(f"扫描第{page}页, 已找到{len(all_docs)}个销售文件, "
                           f"已扫描{page*100}/{total}")

            # 检查是否扫描完毕
            if page * 100 >= total:
                break

            page += 1
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"获取第{page}页失败: {e}")
            break

    logger.info(f"扫描完成: 共找到 {len(all_docs)} 个销售文件")
    return all_docs


def download_pdf(client: httpx.Client, doc: Dict, pdf_dir: str) -> str:
    """下载PDF文件

    Returns:
        'ok' | 'skip' | 'fail'
    """
    url = doc.get('url', '')
    if not url:
        return 'fail'

    filename = url.split('/')[-1] if '/' in url else f"{hash(url)}.pdf"
    filepath = os.path.join(pdf_dir, filename)

    # 已存在则跳过
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return 'skip'

    full_url = f"{BASE_URL}{url}" if url.startswith('/') else url

    try:
        r = client.get(full_url, timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return 'ok'
        return 'fail'
    except Exception as e:
        logger.debug(f"下载失败: {e}")
        return 'fail'


def main():
    parser = argparse.ArgumentParser(description='宁银理财销售文件下载器')
    parser.add_argument('--max', type=int, default=0, help='限制下载数量')
    parser.add_argument('--scan-only', action='store_true', help='仅扫描不下载')
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)

    print("=" * 60)
    print("  宁银理财销售文件下载器 (_07_ 格式)")
    print("=" * 60)

    client = create_client()

    # 扫描销售文件
    docs = scan_sales_documents(client)

    if not docs:
        print("未找到任何销售文件")
        client.close()
        return 1

    # 保存索引
    index = {}
    for doc in docs:
        url = doc.get('url', '')
        filename = url.split('/')[-1] if '/' in url else ''
        if filename:
            index[filename] = {
                'title': doc.get('title', ''),
                'url': url,
            }

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存索引: {len(index)} 条 -> {INDEX_FILE}")

    if args.scan_only:
        print(f"\n扫描完成: {len(docs)} 个销售文件")
        client.close()
        return 0

    # 下载PDF
    total = len(docs) if args.max == 0 else min(len(docs), args.max)
    stats = {'ok': 0, 'skip': 0, 'fail': 0}

    logger.info(f"开始下载 {total} 个文件...")

    for i, doc in enumerate(docs[:total]):
        result = download_pdf(client, doc, PDF_DIR)
        stats[result] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"下载进度: {i+1}/{total} - 新下载:{stats['ok']} "
                       f"跳过:{stats['skip']} 失败:{stats['fail']}")

        if result == 'ok':
            time.sleep(0.2)

    client.close()

    print("\n" + "=" * 60)
    print(f"  下载完成")
    print(f"  新下载: {stats['ok']}")
    print(f"  跳过(已存在): {stats['skip']}")
    print(f"  失败: {stats['fail']}")
    print(f"  索引文件: {INDEX_FILE}")
    print(f"  PDF目录: {PDF_DIR}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
