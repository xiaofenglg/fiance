"""测试获取中信理财产品历史净值"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
import re

def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=options)

def get_product_detail_url(product_code):
    """获取产品详情页URL"""
    # 尝试不同的URL格式
    return f"https://www.citic-wealth.com/yymk/lccs/product?code={product_code}"

def main():
    print("=" * 60)
    print("测试获取中信理财产品历史净值")
    print("=" * 60)

    driver = None

    try:
        print("\n1. 启动Chrome...")
        driver = setup_driver()

        # 先访问列表页获取一个产品代码
        print("\n2. 访问列表页获取产品代码...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(5)

        # 切换到iframe
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if iframes:
            driver.switch_to.frame(iframes[0])
            time.sleep(3)

        # 点击"每日购"
        tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory, .typeCategory-select')
        for tab in tabs:
            if '每日购' in tab.text:
                tab.click()
                time.sleep(3)
                break

        # 获取第一个产品代码
        product_codes = driver.find_elements(By.CSS_SELECTOR, '.item-productCode div')
        if not product_codes:
            print("   未找到产品代码!")
            return

        first_code = product_codes[0].text.strip()
        print(f"   找到产品代码: {first_code}")

        # 检查是否有产品详情链接
        print("\n3. 查找产品详情链接...")

        # 方法1: 查找可点击的产品名称
        product_names = driver.find_elements(By.CSS_SELECTOR, '.item-productName')
        if product_names:
            print(f"   找到 {len(product_names)} 个产品名称元素")
            # 检查是否有onclick或href
            first_name = product_names[0]
            onclick = first_name.get_attribute('onclick')
            print(f"   onclick: {onclick}")

            # 尝试点击产品名称
            print("   尝试点击产品名称...")
            try:
                first_name.click()
                time.sleep(3)

                # 检查是否跳转到详情页
                current_url = driver.current_url
                print(f"   当前URL: {current_url}")

                # 检查页面内容
                page_source = driver.page_source
                if '净值' in page_source and '历史' in page_source:
                    print("   页面包含'净值'和'历史'关键词")

                # 查找历史净值相关元素
                nav_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '历史净值') or contains(text(), '净值走势')]")
                print(f"   找到 {len(nav_elements)} 个历史净值相关元素")

                # 查找表格数据
                tables = driver.find_elements(By.TAG_NAME, 'table')
                print(f"   找到 {len(tables)} 个表格")

                # 打印页面部分内容
                print("\n4. 页面内容分析...")
                body = driver.find_element(By.TAG_NAME, 'body')
                body_text = body.text[:2000]
                print(body_text)

            except Exception as e:
                print(f"   点击失败: {e}")

        # 方法2: 查找所有链接
        print("\n5. 查找所有链接...")
        links = driver.find_elements(By.TAG_NAME, 'a')
        for link in links[:10]:
            href = link.get_attribute('href')
            text = link.text.strip()[:50]
            if href and ('product' in href or 'detail' in href):
                print(f"   链接: {href} - {text}")

        # 方法3: 直接访问可能的详情页URL
        print("\n6. 尝试直接访问详情页...")
        driver.switch_to.default_content()  # 退出iframe

        detail_urls = [
            f"https://www.citic-wealth.com/yymk/lccs/detail?code={first_code}",
            f"https://www.citic-wealth.com/yymk/lccs/product/{first_code}",
            f"https://www.citic-wealth.com/yymk/lccs/?code={first_code}",
            f"https://www.citic-wealth.com/product/detail?code={first_code}",
        ]

        for url in detail_urls:
            print(f"\n   尝试: {url}")
            try:
                driver.get(url)
                time.sleep(3)

                # 检查页面
                body = driver.find_element(By.TAG_NAME, 'body')
                body_text = body.text[:500]
                if body_text.strip() and '404' not in body_text:
                    print(f"   页面有内容:")
                    print(body_text[:300])

                    # 检查是否有iframe
                    iframes = driver.find_elements(By.TAG_NAME, 'iframe')
                    if iframes:
                        print(f"   找到 {len(iframes)} 个iframe")
                        driver.switch_to.frame(iframes[0])
                        time.sleep(2)
                        body = driver.find_element(By.TAG_NAME, 'body')
                        print("   iframe内容:")
                        print(body.text[:500])
                        driver.switch_to.default_content()
            except Exception as e:
                print(f"   错误: {e}")

        # 方法4: 检查API
        print("\n7. 检查可能的历史净值API...")
        api_urls = [
            f"https://wechat.citic-wealth.com/cms.product/api/productInfo/navHistory?productCode={first_code}",
            f"https://wechat.citic-wealth.com/cms.product/api/productInfo/detail?productCode={first_code}",
            f"https://wechat.citic-wealth.com/cms.product/api/fundInfo/navList?fundCode={first_code}",
        ]

        import requests
        import ssl
        import urllib3
        urllib3.disable_warnings()

        # 创建支持旧版SSL的session
        class LegacySSLAdapter(requests.adapters.HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
                kwargs['ssl_context'] = ctx
                return super().init_poolmanager(*args, **kwargs)

        session = requests.Session()
        session.mount('https://', LegacySSLAdapter())

        for api_url in api_urls:
            print(f"\n   API: {api_url}")
            try:
                resp = session.get(api_url, timeout=10)
                print(f"   状态: {resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"   响应: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
            except Exception as e:
                print(f"   错误: {e}")

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
