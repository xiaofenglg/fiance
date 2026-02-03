# -*- coding: utf-8 -*-
"""
浦银理财 API端点探测脚本

使用 Selenium + Chrome DevTools Protocol 拦截网络请求，
自动发现 https://www.spdb-wm.com/financialProducts/ 的API端点。

用法: python spdb_api_discovery.py
"""

import sys
import os
import json
import time
import re
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('spdb_api_discovery')

# 目标网站
TARGET_URL = "https://www.spdb-wm.com/financialProducts/"
BASE_DOMAIN = "spdb-wm.com"


def discover_with_selenium_wire():
    """使用 selenium-wire 拦截网络请求（优先方案）"""
    try:
        from seleniumwire import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        logger.warning("selenium-wire 未安装，尝试 pip install selenium-wire")
        return None

    logger.info("=== 使用 selenium-wire 拦截网络请求 ===")

    options = Options()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

    sw_options = {
        'disable_encoding': True,
        'verify_ssl': False,
    }

    driver = None
    discovered = []

    try:
        driver = webdriver.Chrome(options=options, seleniumwire_options=sw_options)
        driver.set_page_load_timeout(30)

        # 1. 访问主页
        logger.info(f"导航到 {TARGET_URL}")
        driver.get(TARGET_URL)
        time.sleep(5)

        # 收集初始请求
        _collect_requests(driver, discovered, "页面加载")

        # 2. 尝试点击"个人理财"标签
        try:
            tabs = driver.find_elements(By.CSS_SELECTOR,
                'a, button, div[class*="tab"], span[class*="tab"], '
                'li[class*="tab"], div[class*="nav"], li[class*="nav"]')
            for tab in tabs:
                text = tab.text.strip()
                if '个人' in text or '理财' in text:
                    logger.info(f"点击标签: '{text}'")
                    del driver.requests  # 清除旧请求
                    tab.click()
                    time.sleep(3)
                    _collect_requests(driver, discovered, f"点击'{text}'")
                    break
        except Exception as e:
            logger.debug(f"点击标签失败: {e}")

        # 3. 尝试翻页
        try:
            next_buttons = driver.find_elements(By.CSS_SELECTOR,
                'button.btn-next, a.next, li.btn-next, '
                'button[class*="next"], a[class*="next"], '
                '.el-pagination button:last-child')
            for btn in next_buttons:
                if btn.is_displayed() and btn.is_enabled():
                    logger.info("点击下一页按钮")
                    del driver.requests
                    btn.click()
                    time.sleep(3)
                    _collect_requests(driver, discovered, "翻页")
                    break
        except Exception as e:
            logger.debug(f"翻页失败: {e}")

        # 4. 尝试点击产品详情
        try:
            links = driver.find_elements(By.CSS_SELECTOR,
                'a[href*="product"], a[href*="detail"], '
                'div[class*="product"] a, tr a, td a')
            for link in links[:3]:
                if link.is_displayed():
                    href = link.get_attribute('href') or ''
                    text = link.text.strip()[:30]
                    logger.info(f"点击产品链接: '{text}' -> {href}")
                    del driver.requests
                    link.click()
                    time.sleep(3)
                    _collect_requests(driver, discovered, f"产品详情'{text}'")
                    driver.back()
                    time.sleep(2)
                    break
        except Exception as e:
            logger.debug(f"产品详情点击失败: {e}")

        # 5. 分析页面中的JS文件
        _analyze_js_sources(driver, discovered)

    except Exception as e:
        logger.error(f"selenium-wire 探测失败: {e}")
    finally:
        if driver:
            driver.quit()

    return discovered


def discover_with_cdp():
    """使用 Chrome DevTools Protocol 拦截网络请求（备用方案）"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
    except ImportError:
        logger.warning("selenium 未安装")
        return None

    logger.info("=== 使用 CDP 拦截网络请求 ===")

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    driver = None
    discovered = []

    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(60)

        # 启用网络事件
        driver.execute_cdp_cmd('Network.enable', {})

        logger.info(f"导航到 {TARGET_URL}")
        driver.get(TARGET_URL)
        time.sleep(5)

        # 从性能日志中提取网络请求
        logs = driver.get_log('performance')
        for entry in logs:
            try:
                msg = json.loads(entry['message'])['message']
                method = msg.get('method', '')

                if method == 'Network.requestWillBeSent':
                    params = msg.get('params', {})
                    request = params.get('request', {})
                    url = request.get('url', '')
                    req_method = request.get('method', 'GET')
                    post_data = request.get('postData', '')
                    resource_type = params.get('type', '')

                    if resource_type in ('XHR', 'Fetch') or '/api/' in url:
                        discovered.append({
                            'url': url,
                            'method': req_method,
                            'post_data': post_data,
                            'source': 'CDP-页面加载',
                            'type': resource_type,
                        })

                elif method == 'Network.responseReceived':
                    params = msg.get('params', {})
                    response = params.get('response', {})
                    url = response.get('url', '')
                    status = response.get('status', 0)
                    mime = response.get('mimeType', '')

                    if 'json' in mime or '/api/' in url:
                        # 尝试获取响应体
                        request_id = params.get('requestId')
                        try:
                            body = driver.execute_cdp_cmd(
                                'Network.getResponseBody', {'requestId': request_id}
                            )
                            resp_body = body.get('body', '')[:500]
                        except:
                            resp_body = ''

                        discovered.append({
                            'url': url,
                            'status': status,
                            'mime': mime,
                            'response_preview': resp_body,
                            'source': 'CDP-响应',
                        })
            except:
                pass

        # 尝试点击交互元素
        try:
            tabs = driver.find_elements(By.CSS_SELECTOR,
                'a, button, div[class*="tab"], span[class*="tab"]')
            for tab in tabs:
                text = tab.text.strip()
                if '个人' in text:
                    logger.info(f"点击: '{text}'")
                    tab.click()
                    time.sleep(3)

                    new_logs = driver.get_log('performance')
                    for entry in new_logs:
                        try:
                            msg = json.loads(entry['message'])['message']
                            if msg.get('method') == 'Network.requestWillBeSent':
                                params = msg.get('params', {})
                                request = params.get('request', {})
                                url = request.get('url', '')
                                if '/api/' in url or 'product' in url.lower():
                                    discovered.append({
                                        'url': url,
                                        'method': request.get('method', 'GET'),
                                        'post_data': request.get('postData', ''),
                                        'source': f"CDP-点击'{text}'",
                                    })
                        except:
                            pass
                    break
        except:
            pass

        # 分析JS源码
        _analyze_js_sources(driver, discovered)

    except Exception as e:
        logger.error(f"CDP 探测失败: {e}")
    finally:
        if driver:
            driver.quit()

    return discovered


def _collect_requests(driver, discovered, source_label):
    """从 selenium-wire driver 收集拦截到的请求"""
    for req in driver.requests:
        url = req.url
        # 过滤静态资源
        if any(ext in url for ext in ['.css', '.png', '.jpg', '.gif', '.ico', '.svg', '.woff', '.ttf']):
            continue

        entry = {
            'url': url,
            'method': req.method,
            'source': source_label,
        }

        # 请求体
        if req.body:
            try:
                body_str = req.body.decode('utf-8') if isinstance(req.body, bytes) else str(req.body)
                entry['request_body'] = body_str[:500]
            except:
                pass

        # 响应信息
        if req.response:
            entry['status'] = req.response.status_code
            content_type = req.response.headers.get('Content-Type', '')
            entry['content_type'] = content_type

            if 'json' in content_type or 'javascript' not in content_type:
                try:
                    body = req.response.body
                    if body:
                        body_str = body.decode('utf-8') if isinstance(body, bytes) else str(body)
                        entry['response_preview'] = body_str[:500]
                except:
                    pass

        # 只保留有价值的API请求
        if (('/api/' in url or 'product' in url.lower() or 'nav' in url.lower() or
             'financial' in url.lower()) and
            BASE_DOMAIN in url):
            discovered.append(entry)

        # 也保留JSON响应
        elif req.response and 'json' in (req.response.headers.get('Content-Type', '')):
            if BASE_DOMAIN in url:
                discovered.append(entry)


def _analyze_js_sources(driver, discovered):
    """分析页面中加载的JS文件，寻找API端点模式"""
    logger.info("分析页面JS源码中的API模式...")

    try:
        from selenium.webdriver.common.by import By

        scripts = driver.find_elements(By.TAG_NAME, 'script')
        for script in scripts:
            src = script.get_attribute('src') or ''

            # 内联脚本
            if not src:
                text = script.get_attribute('innerHTML') or ''
                _extract_api_patterns_from_text(text, discovered, '内联脚本')
                continue

            # 外部JS文件（只分析同域名的）
            if BASE_DOMAIN not in src:
                continue

            # chunk/app/vendor JS文件通常包含API定义
            if any(keyword in src for keyword in ['app.', 'chunk', 'vendor', 'main', 'index']):
                logger.info(f"分析JS: {src}")
                try:
                    # 直接通过driver获取JS内容，避免SSL问题
                    js_content = driver.execute_script(
                        f"return fetch('{src}').then(r => r.text()).catch(() => '')"
                    )
                    if js_content:
                        _extract_api_patterns_from_text(js_content, discovered, f'JS:{src.split("/")[-1]}')
                except:
                    pass
    except Exception as e:
        logger.debug(f"JS分析失败: {e}")


def _extract_api_patterns_from_text(text, discovered, source_label):
    """从文本中提取API端点模式"""
    patterns = [
        (r'["\'](/api/[^"\']+)["\']', 'API路径'),
        (r'baseURL\s*[=:]\s*["\']([^"\']+)["\']', 'baseURL'),
        (r'axios\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']', 'axios调用'),
        (r'fetch\s*\(\s*["\']([^"\']+)["\']', 'fetch调用'),
        (r'url\s*[=:]\s*["\']([^"\']*(?:product|nav|financial|list)[^"\']*)["\']', 'URL赋值'),
        (r'["\']([^"\']*(?:/product/|/nav/|/financial)[^"\']*)["\']', '路径模式'),
    ]

    for pattern, pattern_name in patterns:
        matches = re.findall(pattern, text[:100000])  # 限制搜索范围
        for match in matches:
            if len(match) > 5 and not match.endswith('.js') and not match.endswith('.css'):
                discovered.append({
                    'url': match,
                    'source': f'{source_label} ({pattern_name})',
                    'type': 'js_pattern',
                })


def discover_with_requests():
    """纯requests方式探测常见API端点（最终兜底）"""
    import requests
    import ssl
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util.ssl_ import create_urllib3_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    logger.info("=== 使用 requests 探测常见API端点 ===")

    class LegacySSLAdapter(HTTPAdapter):
        """自定义SSL适配器，解决legacy renegotiation问题"""
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

    session = requests.Session()
    adapter = LegacySSLAdapter(pool_connections=10, pool_maxsize=10)
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': TARGET_URL,
        'Origin': 'https://www.spdb-wm.com',
    })
    session.verify = False

    discovered = []

    # 常见的API端点模式
    candidate_endpoints = [
        # GET 方式
        ('/api/product/list', 'GET', None),
        ('/api/products', 'GET', None),
        ('/api/financial/products', 'GET', None),
        ('/api/v1/product/list', 'GET', None),
        ('/api/v1/products', 'GET', None),
        ('/financialProducts/api/list', 'GET', None),
        ('/financialProducts/api/products', 'GET', None),
        ('/wealth/api/product/list', 'GET', None),
        ('/gw/product/list', 'GET', None),
        ('/gw/po_web/product/list', 'GET', None),
        # POST 方式
        ('/api/product/list', 'POST', {"pageNo": 1, "pageSize": 10}),
        ('/api/products', 'POST', {"pageNo": 1, "pageSize": 10}),
        ('/api/product/query', 'POST', {"pageNo": 1, "pageSize": 10, "type": "personal"}),
        ('/api/financial/query', 'POST', {"page": 1, "size": 10}),
        ('/api/v1/product/list', 'POST', {"pageNo": 1, "pageSize": 10}),
        ('/api/v1/product/query', 'POST', {"pageNo": 1, "pageSize": 10}),
    ]

    # 先获取主页分析JS中的端点
    try:
        resp = session.get(TARGET_URL, timeout=15)
        if resp.status_code == 200:
            logger.info(f"主页访问成功 (状态: {resp.status_code}, 长度: {len(resp.text)})")

            # 提取JS中的API路径
            api_matches = re.findall(r'["\'](/[^"\']*(?:api|product|nav|query)[^"\']*)["\']', resp.text)
            for match in api_matches:
                if len(match) > 3:
                    discovered.append({
                        'url': match,
                        'source': '主页HTML正则',
                        'type': 'pattern',
                    })
                    # 添加到候选列表
                    candidate_endpoints.append((match, 'GET', None))
                    candidate_endpoints.append((match, 'POST', {"pageNo": 1, "pageSize": 10}))

            # 查找JS文件链接
            js_files = re.findall(r'src=["\']([^"\']+\.js[^"\']*)["\']', resp.text)
            for js_file in js_files:
                if BASE_DOMAIN in js_file or js_file.startswith('/'):
                    js_url = js_file if js_file.startswith('http') else f'https://www.spdb-wm.com{js_file}'
                    try:
                        js_resp = session.get(js_url, timeout=10)
                        if js_resp.status_code == 200:
                            js_apis = re.findall(
                                r'["\'](/[^"\']*(?:api|product|nav|query|list)[^"\']*)["\']',
                                js_resp.text[:200000]
                            )
                            for api in js_apis:
                                if len(api) > 3 and not api.endswith(('.js', '.css', '.html')):
                                    discovered.append({
                                        'url': api,
                                        'source': f'JS文件:{js_url.split("/")[-1]}',
                                        'type': 'js_pattern',
                                    })
                                    candidate_endpoints.append((api, 'GET', None))
                                    candidate_endpoints.append((api, 'POST', {"pageNo": 1, "pageSize": 10}))
                    except:
                        pass
    except Exception as e:
        logger.warning(f"主页访问失败: {e}")

    # 逐个测试候选端点
    tested = set()
    for endpoint, method, payload in candidate_endpoints:
        key = f"{method}:{endpoint}"
        if key in tested:
            continue
        tested.add(key)

        url = f"https://www.spdb-wm.com{endpoint}" if endpoint.startswith('/') else endpoint

        try:
            if method == 'GET':
                resp = session.get(url, params={"pageNo": 1, "pageSize": 10}, timeout=8)
            else:
                resp = session.post(url, json=payload, timeout=8)

            entry = {
                'url': endpoint,
                'method': method,
                'status': resp.status_code,
                'content_type': resp.headers.get('Content-Type', ''),
                'source': 'requests探测',
            }

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    entry['response_preview'] = json.dumps(data, ensure_ascii=False)[:500]
                    entry['is_valid_api'] = True
                    logger.info(f"[有效] {method} {endpoint} -> {json.dumps(data, ensure_ascii=False)[:200]}")
                except:
                    entry['response_preview'] = resp.text[:200]
                    entry['is_valid_api'] = False
            elif resp.status_code in (301, 302, 403, 404, 405):
                entry['is_valid_api'] = False
            else:
                entry['is_valid_api'] = False

            discovered.append(entry)
        except Exception as e:
            logger.debug(f"  {method} {endpoint} -> 请求异常: {e}")

    return discovered


def print_report(discovered):
    """打印探测报告"""
    print("\n" + "=" * 80)
    print("        浦银理财 API端点探测报告")
    print("=" * 80)
    print(f"探测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: {TARGET_URL}")
    print(f"发现条目: {len(discovered)}")

    # 按类型分组
    valid_apis = [d for d in discovered if d.get('is_valid_api')]
    js_patterns = [d for d in discovered if d.get('type') == 'js_pattern']
    xhr_requests = [d for d in discovered if d.get('type') in ('XHR', 'Fetch')]

    print(f"\n有效API端点: {len(valid_apis)}")
    print(f"JS中发现的模式: {len(js_patterns)}")
    print(f"XHR/Fetch请求: {len(xhr_requests)}")

    if valid_apis:
        print("\n--- 有效API端点 ---")
        for api in valid_apis:
            print(f"  [{api.get('method', '?')}] {api['url']}")
            if api.get('response_preview'):
                print(f"      响应: {api['response_preview'][:200]}")

    if js_patterns:
        # 去重
        unique = list({d['url'] for d in js_patterns})
        print(f"\n--- JS中发现的API模式 (去重后 {len(unique)} 个) ---")
        for url in sorted(unique):
            print(f"  {url}")

    if xhr_requests:
        print(f"\n--- XHR/Fetch请求 ---")
        for req in xhr_requests:
            print(f"  [{req.get('method', '?')}] {req['url']}")
            if req.get('request_body'):
                print(f"      请求体: {req['request_body'][:200]}")
            if req.get('response_preview'):
                print(f"      响应: {req['response_preview'][:200]}")

    # 输出所有发现
    if discovered:
        print(f"\n--- 全部发现 ({len(discovered)} 条) ---")
        seen = set()
        for d in discovered:
            url = d.get('url', '')
            key = f"{d.get('method', '')}:{url}"
            if key in seen:
                continue
            seen.add(key)
            source = d.get('source', '')
            status = d.get('status', '')
            print(f"  {d.get('method', '?'):6} {url:60} [{source}] {status}")

    print("\n" + "=" * 80)

    # 保存到文件
    output_file = f'spdb_api_discovery_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(discovered, f, ensure_ascii=False, indent=2, default=str)
    print(f"详细结果已保存到: {output_file}")


def main():
    logger.info("=" * 60)
    logger.info("浦银理财 API端点探测工具")
    logger.info("=" * 60)

    all_discovered = []

    # 方案1: selenium-wire
    result = discover_with_selenium_wire()
    if result:
        all_discovered.extend(result)
        logger.info(f"selenium-wire 发现 {len(result)} 条")
    else:
        # 方案2: CDP
        result = discover_with_cdp()
        if result:
            all_discovered.extend(result)
            logger.info(f"CDP 发现 {len(result)} 条")

    # 方案3: 纯requests探测（始终执行）
    result = discover_with_requests()
    if result:
        all_discovered.extend(result)
        logger.info(f"requests 发现 {len(result)} 条")

    # 打印报告
    print_report(all_discovered)


if __name__ == '__main__':
    main()
