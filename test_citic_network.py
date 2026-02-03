# -*- coding: utf-8 -*-
"""捕获中信理财网络请求,找到历史净值API"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import json
import time
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

def setup_driver_with_logging():
    """设置带网络日志的Chrome"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')

    # 启用性能日志
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    return webdriver.Chrome(options=options)

def get_network_logs(driver):
    """获取网络请求日志"""
    logs = driver.get_log('performance')
    requests = []

    for log in logs:
        try:
            message = json.loads(log['message'])
            method = message.get('message', {}).get('method', '')

            if method == 'Network.requestWillBeSent':
                params = message['message']['params']
                url = params.get('request', {}).get('url', '')
                if 'wechat.citic-wealth.com' in url or 'api' in url.lower():
                    requests.append({
                        'url': url,
                        'method': params.get('request', {}).get('method', 'GET'),
                    })

            elif method == 'Network.responseReceived':
                params = message['message']['params']
                url = params.get('response', {}).get('url', '')
                status = params.get('response', {}).get('status', 0)
                if 'wechat.citic-wealth.com' in url or 'api' in url.lower():
                    # 找到对应的request并更新
                    for req in requests:
                        if req['url'] == url:
                            req['status'] = status
                            break

        except Exception as e:
            pass

    return requests

def main():
    print("=" * 60)
    print("捕获中信理财网络请求")
    print("=" * 60)

    driver = None

    try:
        print("\n1. 启动Chrome(带网络日志)...")
        driver = setup_driver_with_logging()

        print("\n2. 访问列表页...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(5)

        # 获取初始网络请求
        print("\n3. 初始页面加载的API请求:")
        requests = get_network_logs(driver)
        for req in requests:
            print(f"   {req['method']} {req['url']}")

        # 切换到iframe
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if iframes:
            driver.switch_to.frame(iframes[0])
            time.sleep(3)

        # 点击"每日购"
        print("\n4. 切换到每日购...")
        tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory, .typeCategory-select')
        for tab in tabs:
            if '每日购' in tab.text:
                tab.click()
                time.sleep(3)
                break

        # 获取切换后的请求
        print("\n5. 切换分类后的API请求:")
        requests = get_network_logs(driver)
        for req in requests[-10:]:
            print(f"   {req['method']} {req['url']}")

        # 点击第一个产品
        print("\n6. 点击产品查看详情...")
        product_names = driver.find_elements(By.CSS_SELECTOR, '.item-productName')
        if product_names:
            first_product = product_names[0]
            product_name = first_product.text.strip()
            print(f"   产品: {product_name[:40]}...")

            # 获取产品代码
            product_codes = driver.find_elements(By.CSS_SELECTOR, '.item-productCode div')
            product_code = product_codes[0].text.strip() if product_codes else ''
            print(f"   代码: {product_code}")

            # 清空日志再点击
            driver.get_log('performance')  # 清空

            first_product.click()
            time.sleep(5)

            # 获取点击后的请求
            print("\n7. 点击产品后的API请求:")
            requests = get_network_logs(driver)
            for req in requests:
                url = req['url']
                if 'nav' in url.lower() or 'history' in url.lower() or 'detail' in url.lower() or 'product' in url.lower():
                    print(f"   ** {req['method']} {url}")
                else:
                    print(f"   {req['method']} {url[:100]}")

            # 打印页面当前内容
            print("\n8. 详情页面内容分析...")
            body = driver.find_element(By.TAG_NAME, 'body')
            body_text = body.text

            # 查找净值相关内容
            lines = body_text.split('\n')
            nav_lines = []
            for i, line in enumerate(lines):
                if re.search(r'1\.\d{4}', line) or '净值' in line or re.match(r'\d{4}-\d{2}-\d{2}', line):
                    nav_lines.append(line)

            print(f"   找到 {len(nav_lines)} 行净值相关内容:")
            for line in nav_lines[:30]:
                print(f"      {line}")

            # 尝试直接请求API获取历史净值
            print("\n9. 尝试直接请求历史净值API...")
            import requests as http_requests
            import ssl
            import urllib3
            urllib3.disable_warnings()

            class LegacySSLAdapter(http_requests.adapters.HTTPAdapter):
                def init_poolmanager(self, *args, **kwargs):
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    ctx.options |= 0x4
                    kwargs['ssl_context'] = ctx
                    return super().init_poolmanager(*args, **kwargs)

            session = http_requests.Session()
            session.mount('https://', LegacySSLAdapter())

            # 尝试不同的API格式
            api_urls = [
                f"https://wechat.citic-wealth.com/cms.product/api/productInfo/getNavHistory?productCode={product_code}",
                f"https://wechat.citic-wealth.com/cms.product/api/productInfo/getNavList?productCode={product_code}",
                f"https://wechat.citic-wealth.com/cms.product/api/productInfo/productDetail?productCode={product_code}",
                f"https://wechat.citic-wealth.com/cms.product/api/fundInfo/getNavHistory?fundCode={product_code}",
                f"https://wechat.citic-wealth.com/cms.product/api/fundInfo/navHistory?fundCode={product_code}",
                f"https://wechat.citic-wealth.com/cms.product/api/fundInfo/detail?fundCode={product_code}",
                f"https://wechat.citic-wealth.com/cms.product/api/productNavList?productCode={product_code}",
            ]

            for api_url in api_urls:
                try:
                    resp = session.get(api_url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('success') or data.get('data'):
                            print(f"\n   *** 找到有效API: {api_url}")
                            print(f"   响应: {json.dumps(data, ensure_ascii=False, indent=2)[:1000]}")
                            break
                    else:
                        print(f"   {api_url} -> {resp.status_code}")
                except Exception as e:
                    print(f"   {api_url} -> Error: {str(e)[:50]}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if driver:
            driver.quit()
            print("\n浏览器已关闭")

if __name__ == "__main__":
    main()
