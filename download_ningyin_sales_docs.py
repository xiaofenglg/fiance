# -*- coding: utf-8 -*-
"""
宁银理财销售文件下载器

从 https://www.wmbnb.com/info/funddetails/index.html 下载产品销售文件(含费率信息)

用法:
    python download_ningyin_sales_docs.py              # 下载全部
    python download_ningyin_sales_docs.py --max 100    # 限量测试
"""

import os
import sys
import ssl
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_ningyin_sales_docs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '宁银')
INDEX_FILE = os.path.join(PDF_DIR, 'index.json')

BASE_URL = "https://www.wmbnb.com"


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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': f'{BASE_URL}/info/funddetails/index.html',
        },
        follow_redirects=True,
    )


def get_cookies_via_selenium() -> Dict[str, str]:
    """使用Selenium获取cookies"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        logger.error("需要安装selenium: pip install selenium")
        return {}

    logger.info("启动浏览器获取cookies...")

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)

        # 访问信息披露页面
        driver.get(f"{BASE_URL}/info/funddetails/index.html")
        time.sleep(3)

        # 获取cookies
        cookies = {c['name']: c['value'] for c in driver.get_cookies()}
        logger.info(f"获取到 {len(cookies)} 个cookies")
        return cookies

    except Exception as e:
        logger.error(f"Selenium获取cookies失败: {e}")
        return {}
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def get_document_types(client: httpx.Client) -> List[Dict]:
    """获取文档类型字典"""
    logger.info("获取文档类型...")

    # 尝试两种API路径
    endpoints = [
        '/ningbo-web/;a/product/dictionary.json',
        '/ningbo-web/product/dictionary.json',
    ]

    for ep in endpoints:
        try:
            r = client.get(f"{BASE_URL}{ep}", params={'dictname': 'atta_type'})
            if r.status_code == 200:
                data = r.json()
                if data.get('status') == 'success':
                    types = data.get('list', [])
                    logger.info(f"获取到 {len(types)} 个文档类型")
                    return types
        except Exception as e:
            logger.debug(f"尝试 {ep} 失败: {e}")

    return []


def get_sales_documents(client: httpx.Client, atta_type: str = None,
                        max_pages: int = 5000) -> List[Dict]:
    """获取销售文件列表"""
    logger.info("获取销售文件列表...")

    all_docs = []
    page = 1

    # 尝试两种API路径
    api_path = '/ningbo-web/;a/product/attachmentlist.json'

    while page <= max_pages:
        params = {
            'request_num': 100,
            'request_pageno': page,
        }
        if atta_type:
            params['atta_type'] = atta_type

        try:
            r = client.get(f"{BASE_URL}{api_path}", params=params)
            if r.status_code != 200:
                break

            data = r.json()
            if data.get('status') != 'success':
                break

            docs = data.get('list', [])
            if not docs:
                break

            # 筛选销售文件（包含说明书、发行公告等）
            for doc in docs:
                title = doc.get('title', '')
                url = doc.get('url', '')

                # 筛选条件：包含说明书、风险揭示书、销售等关键词
                if any(kw in title for kw in ['说明书', '风险揭示', '销售', '发行', '募集']):
                    all_docs.append(doc)

            total = data.get('total', 0)
            logger.info(f"第{page}页: 本页{len(docs)}个, 筛选后累计{len(all_docs)}个, 总{total}")

            if page * 100 >= total:
                break

            page += 1
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"获取第{page}页失败: {e}")
            break

    return all_docs


def download_pdf(client: httpx.Client, doc: Dict, pdf_dir: str) -> str:
    """下载单个PDF文件

    Returns:
        'ok' | 'skip' | 'fail'
    """
    url = doc.get('url', '')
    title = doc.get('title', '')

    if not url:
        return 'fail'

    # 生成文件名
    # URL格式: /report/ZGN2430042B_xxx_20260203.pdf
    filename = url.split('/')[-1] if '/' in url else f"{hash(url)}.pdf"
    filepath = os.path.join(pdf_dir, filename)

    # 已存在则跳过
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return 'skip'

    # 下载
    full_url = f"{BASE_URL}{url}" if url.startswith('/') else url

    try:
        r = client.get(full_url, timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return 'ok'
        else:
            return 'fail'
    except Exception as e:
        logger.debug(f"下载失败 {title}: {e}")
        return 'fail'


def save_index(docs: List[Dict], index_file: str):
    """保存索引文件"""
    index = {}
    for doc in docs:
        url = doc.get('url', '')
        filename = url.split('/')[-1] if '/' in url else ''
        if filename:
            index[filename] = {
                'title': doc.get('title', ''),
                'url': url,
                'date': doc.get('date', ''),
            }

    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存索引: {len(index)} 条")


def main():
    parser = argparse.ArgumentParser(description='宁银理财销售文件下载器')
    parser.add_argument('--max', type=int, default=0, help='限制下载数量')
    parser.add_argument('--use-selenium', action='store_true', help='使用Selenium获取cookies')
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)

    print("=" * 60)
    print("  宁银理财销售文件下载器")
    print("=" * 60)

    # 创建客户端
    client = create_client()

    # 如果需要，用Selenium获取cookies
    if args.use_selenium:
        cookies = get_cookies_via_selenium()
        if cookies:
            for name, value in cookies.items():
                client.cookies.set(name, value, domain='www.wmbnb.com')

    # 获取文档类型
    doc_types = get_document_types(client)
    if doc_types:
        for t in doc_types:
            logger.info(f"  类型: {t}")

    # 获取销售文件列表
    docs = get_sales_documents(client)

    if not docs:
        logger.warning("未获取到销售文件，尝试获取所有附件...")
        # 回退：获取所有附件中的说明书
        docs = get_sales_documents(client, atta_type=None)

    if not docs:
        print("未获取到任何销售文件")
        client.close()
        return 1

    # 保存索引
    save_index(docs, INDEX_FILE)

    # 下载PDF
    total = len(docs) if args.max == 0 else min(len(docs), args.max)
    stats = {'ok': 0, 'skip': 0, 'fail': 0}

    logger.info(f"开始下载 {total} 个文件...")

    for i, doc in enumerate(docs[:total]):
        result = download_pdf(client, doc, PDF_DIR)
        stats[result] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"进度: {i+1}/{total} - 成功:{stats['ok']} 跳过:{stats['skip']} 失败:{stats['fail']}")

        if result == 'ok':
            time.sleep(0.3)

    client.close()

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
