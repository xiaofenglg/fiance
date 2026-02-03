# -*- coding: utf-8 -*-
"""
多银行费率PDF下载器

支持银行:
- 宁银: 产品说明书/发行公告
- 中邮: 产品说明书 (需WAF bypass)
- 华夏: 产品说明书
- 浦银: 产品说明书

用法:
    python download_bank_pdfs.py ningyin          # 下载宁银
    python download_bank_pdfs.py psbc             # 下载中邮
    python download_bank_pdfs.py huaxia           # 下载华夏
    python download_bank_pdfs.py all              # 下载全部
    python download_bank_pdfs.py discover         # 探索API端点
"""

import os
import sys
import ssl
import time
import json
import logging
import argparse
import requests
from datetime import datetime
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_bank_pdfs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 宁银理财
# ============================================================

class NingyinDownloader:
    """宁银理财PDF下载器"""

    BASE_URL = "https://www.wmbnb.com"
    API_PREFIX = "/ningbo-web/;a"
    PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '宁银')

    def __init__(self):
        self.client = self._create_client()
        os.makedirs(self.PDF_DIR, exist_ok=True)

    def _create_client(self):
        """创建httpx客户端处理SSL问题"""
        if httpx is None:
            logger.error("需要安装httpx: pip install httpx")
            return None

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        except AttributeError:
            pass

        return httpx.Client(
            verify=ctx,
            timeout=30,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': f'{self.BASE_URL}/product/index.html',
            },
            follow_redirects=True,
        )

    def discover_endpoints(self):
        """探索可用的API端点"""
        logger.info("[宁银] 探索API端点...")

        endpoints = [
            # 产品相关
            '/product/list.json',
            '/product/detail.json',
            '/product/attachmentlist.json',
            # 公告相关
            '/announcement/list.json',
            '/notice/list.json',
            '/disclosure/list.json',
            '/report/list.json',
            # 说明书相关
            '/prospectus/list.json',
            '/productdoc/list.json',
        ]

        found = []
        for endpoint in endpoints:
            url = f"{self.BASE_URL}{self.API_PREFIX}{endpoint}"
            try:
                r = self.client.get(url, params={'request_num': 1, 'request_pageno': 1})
                if r.status_code == 200:
                    try:
                        data = r.json()
                        if data.get('status') == 'success':
                            found.append((endpoint, 'OK', data.get('total', '?')))
                            logger.info(f"  [OK] {endpoint} - total: {data.get('total', '?')}")
                        else:
                            found.append((endpoint, 'FAIL', data.get('msg', '')))
                    except:
                        found.append((endpoint, 'NOT_JSON', ''))
                else:
                    found.append((endpoint, str(r.status_code), ''))
            except Exception as e:
                found.append((endpoint, 'ERROR', str(e)[:50]))

        return found

    def get_product_list(self, max_products=0) -> List[Dict]:
        """获取产品列表"""
        logger.info("[宁银] 获取产品列表...")

        all_products = []
        page = 1

        while True:
            try:
                r = self.client.get(
                    f"{self.BASE_URL}{self.API_PREFIX}/product/list.json",
                    params={'request_num': 100, 'request_pageno': page}
                )
                data = r.json()

                if data.get('status') != 'success':
                    break

                products = data.get('list', [])
                if not products:
                    break

                all_products.extend(products)

                total = data.get('total', 0)
                logger.info(f"  第{page}页: {len(products)}个, 累计{len(all_products)}/{total}")

                if len(all_products) >= total:
                    break
                if max_products > 0 and len(all_products) >= max_products:
                    break

                page += 1
                time.sleep(0.3)

            except Exception as e:
                logger.error(f"获取第{page}页失败: {e}")
                break

        return all_products

    def search_prospectus_documents(self) -> List[Dict]:
        """搜索产品说明书文档

        尝试从网站的信息披露/公告栏目获取产品说明书
        """
        logger.info("[宁银] 搜索产品说明书...")

        # 尝试不同的搜索方式
        search_endpoints = [
            # 附件列表 - 尝试不同的类型参数
            ('/product/attachmentlist.json', {'type': '1'}),  # 可能是说明书类型
            ('/product/attachmentlist.json', {'type': '2'}),
            ('/product/attachmentlist.json', {'doctype': 'prospectus'}),
            ('/product/attachmentlist.json', {'category': '说明书'}),
        ]

        found_docs = []
        for endpoint, extra_params in search_endpoints:
            try:
                params = {'request_num': 10, 'request_pageno': 1}
                params.update(extra_params)

                r = self.client.get(
                    f"{self.BASE_URL}{self.API_PREFIX}{endpoint}",
                    params=params
                )
                data = r.json()

                if data.get('status') == 'success':
                    docs = data.get('list', [])
                    for doc in docs:
                        title = doc.get('title', '')
                        url = doc.get('url', '')
                        # 检查是否是说明书
                        if '说明书' in title or 'prospectus' in url.lower():
                            found_docs.append(doc)
                            logger.info(f"  找到: {title}")
            except Exception as e:
                logger.debug(f"搜索失败 {endpoint}: {e}")

        return found_docs

    def close(self):
        if self.client:
            self.client.close()


# ============================================================
# 中邮理财
# ============================================================

class PSBCDownloader:
    """中邮理财PDF下载器 (需要WAF bypass)"""

    BASE_URL = "https://www.psbc-wm.com"
    API_BASE = "https://www.psbc-wm.com/pswm-api"
    PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '中邮')

    def __init__(self, cookies: Dict[str, str] = None):
        self.cookies = cookies
        self.session = self._create_session()
        os.makedirs(self.PDF_DIR, exist_ok=True)

    def _create_session(self):
        """创建HTTP会话"""
        import urllib3
        urllib3.disable_warnings()

        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': f'{self.BASE_URL}/products/index.html',
        })
        session.verify = False

        if self.cookies:
            for name, value in self.cookies.items():
                session.cookies.set(name, value)

        return session

    def get_waf_cookies(self) -> Optional[Dict[str, str]]:
        """使用undetected_chromedriver获取WAF cookies"""
        try:
            import undetected_chromedriver as uc
        except ImportError:
            logger.error("需要安装: pip install undetected-chromedriver")
            return None

        logger.info("[中邮] 启动浏览器绕过WAF...")

        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-position=-2400,0')

        driver = None
        try:
            driver = uc.Chrome(options=options)
            driver.set_page_load_timeout(60)
            driver.get(self.BASE_URL)

            # 等待WAF验证
            for i in range(20):
                time.sleep(2)
                if len(driver.page_source) > 5000:
                    logger.info(f"WAF验证通过 ({(i+1)*2}秒)")
                    break

            cookies = {c['name']: c['value'] for c in driver.get_cookies()}
            logger.info(f"获取到 {len(cookies)} 个cookie")
            return cookies

        except Exception as e:
            logger.error(f"WAF bypass失败: {e}")
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def discover_endpoints(self):
        """探索API端点"""
        logger.info("[中邮] 探索API端点...")

        if not self.cookies:
            logger.warning("无cookies，尝试获取...")
            self.cookies = self.get_waf_cookies()
            if self.cookies:
                self.session = self._create_session()

        endpoints = [
            '/product/search',
            '/product/detail',
            '/product/nvlist',
            '/product/announcement',
            '/product/prospectus',
            '/announcement/list',
            '/disclosure/list',
        ]

        found = []
        for endpoint in endpoints:
            url = f"{self.API_BASE}{endpoint}"
            try:
                r = self.session.get(url, timeout=10)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        found.append((endpoint, data.get('state', 'unknown'), ''))
                        logger.info(f"  [OK] {endpoint}")
                    except:
                        found.append((endpoint, 'NOT_JSON', ''))
                elif r.status_code == 412:
                    found.append((endpoint, 'WAF_BLOCKED', ''))
                    logger.warning(f"  [WAF] {endpoint}")
                else:
                    found.append((endpoint, str(r.status_code), ''))
            except Exception as e:
                found.append((endpoint, 'ERROR', str(e)[:30]))

        return found


# ============================================================
# 华夏理财
# ============================================================

class HuaxiaDownloader:
    """华夏理财PDF下载器"""

    BASE_URL = "https://www.hxwm.com.cn"
    PDF_DIR = os.path.join(BASE_DIR, 'pdfs', '华夏')

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        os.makedirs(self.PDF_DIR, exist_ok=True)

    def discover_endpoints(self):
        """探索API端点"""
        logger.info("[华夏] 探索数据源...")

        # 华夏使用JS文件作为数据源
        js_files = [
            '/common/js/gmjzbgData.js',      # 净值报告
            '/common/js/productData.js',     # 产品数据?
            '/common/js/announcementData.js', # 公告?
            '/common/js/prospectusData.js',  # 说明书?
        ]

        found = []
        for js_file in js_files:
            url = f"{self.BASE_URL}{js_file}"
            try:
                r = self.session.get(url, timeout=10)
                if r.status_code == 200:
                    content = r.text[:500]
                    if '[' in content or '{' in content:
                        found.append((js_file, 'OK', f'{len(r.text)} bytes'))
                        logger.info(f"  [OK] {js_file} - {len(r.text)} bytes")
                    else:
                        found.append((js_file, 'NO_DATA', ''))
                else:
                    found.append((js_file, str(r.status_code), ''))
            except Exception as e:
                found.append((js_file, 'ERROR', str(e)[:30]))

        # 也尝试网页端点
        html_pages = [
            '/xxpl/index.html',      # 信息披露
            '/cpgg/index.html',      # 产品公告
            '/fxgg/index.html',      # 发行公告
        ]

        for page in html_pages:
            url = f"{self.BASE_URL}{page}"
            try:
                r = self.session.get(url, timeout=10)
                if r.status_code == 200:
                    found.append((page, 'OK', f'{len(r.text)} bytes'))
                    logger.info(f"  [OK] {page}")
                else:
                    found.append((page, str(r.status_code), ''))
            except Exception as e:
                found.append((page, 'ERROR', str(e)[:30]))

        return found


# ============================================================
# 主程序
# ============================================================

def discover_all():
    """探索所有银行的API端点"""
    print("=" * 70)
    print("           多银行费率数据源探索")
    print("=" * 70)

    results = {}

    # 宁银
    print("\n[宁银理财]")
    try:
        dl = NingyinDownloader()
        results['宁银'] = dl.discover_endpoints()

        # 额外搜索说明书
        docs = dl.search_prospectus_documents()
        if docs:
            print(f"  找到 {len(docs)} 个说明书文档")
        dl.close()
    except Exception as e:
        print(f"  错误: {e}")

    # 华夏
    print("\n[华夏理财]")
    try:
        dl = HuaxiaDownloader()
        results['华夏'] = dl.discover_endpoints()
    except Exception as e:
        print(f"  错误: {e}")

    # 中邮 (需要WAF bypass，跳过自动探索)
    print("\n[中邮理财]")
    print("  需要WAF bypass，使用 --psbc 单独运行")

    print("\n" + "=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description='多银行费率PDF下载器')
    parser.add_argument('command', nargs='?', default='discover',
                        choices=['discover', 'ningyin', 'psbc', 'huaxia', 'all'],
                        help='操作命令')
    parser.add_argument('--max', type=int, default=0, help='最大下载数')

    args = parser.parse_args()

    if args.command == 'discover':
        discover_all()

    elif args.command == 'ningyin':
        dl = NingyinDownloader()
        products = dl.get_product_list(max_products=args.max or 10)
        print(f"获取到 {len(products)} 个产品")
        # TODO: 下载说明书
        dl.close()

    elif args.command == 'psbc':
        dl = PSBCDownloader()
        cookies = dl.get_waf_cookies()
        if cookies:
            dl = PSBCDownloader(cookies)
            dl.discover_endpoints()

    elif args.command == 'huaxia':
        dl = HuaxiaDownloader()
        dl.discover_endpoints()


if __name__ == '__main__':
    main()
