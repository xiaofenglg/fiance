"""使用Selenium抓取中信理财数据"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import json
import time

def setup_driver():
    """设置Chrome驱动"""
    options = Options()
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    # 启用网络日志以捕获API请求
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    driver = webdriver.Chrome(options=options)
    return driver

def capture_network_requests(driver):
    """捕获网络请求"""
    logs = driver.get_log('performance')
    api_responses = []

    for log in logs:
        try:
            message = json.loads(log['message'])
            method = message.get('message', {}).get('method', '')

            if method == 'Network.responseReceived':
                response = message['message']['params']['response']
                url = response.get('url', '')

                # 只关注产品相关的API
                if 'cms.product' in url or 'productInfo' in url or 'fundList' in url:
                    api_responses.append({
                        'url': url,
                        'status': response.get('status'),
                        'mimeType': response.get('mimeType'),
                    })
        except:
            continue

    return api_responses

def main():
    print("="*60)
    print("使用Selenium抓取中信理财数据")
    print("="*60)

    driver = None
    try:
        print("\n1. 启动Chrome...")
        driver = setup_driver()

        print("\n2. 访问中信理财页面...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(3)

        print(f"   当前URL: {driver.current_url}")
        print(f"   页面标题: {driver.title}")

        # 等待iframe加载
        print("\n3. 等待iframe加载...")
        time.sleep(5)

        # 查找iframe
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        print(f"   找到 {len(iframes)} 个iframe")

        for i, iframe in enumerate(iframes):
            src = iframe.get_attribute('src')
            print(f"   iframe {i}: {src}")

        # 切换到iframe
        if iframes:
            print("\n4. 切换到产品列表iframe...")
            driver.switch_to.frame(iframes[0])
            time.sleep(3)

            # 获取iframe内容
            page_source = driver.page_source
            print(f"   iframe内容长度: {len(page_source)}")

            # 保存用于分析
            with open("citic_iframe.html", "w", encoding="utf-8") as f:
                f.write(page_source)
            print("   已保存到 citic_iframe.html")

            # 查找产品列表
            print("\n5. 查找产品元素...")
            try:
                # 尝试查找产品卡片
                products = driver.find_elements(By.CSS_SELECTOR, '.product-item, .fund-item, .card, [class*=product]')
                print(f"   找到 {len(products)} 个产品元素")

                for p in products[:5]:
                    print(f"   - {p.text[:100]}...")
            except Exception as e:
                print(f"   查找产品失败: {e}")

        # 捕获API请求
        print("\n6. 捕获API请求...")
        api_requests = capture_network_requests(driver)
        print(f"   捕获到 {len(api_requests)} 个产品相关API请求:")
        for req in api_requests[:10]:
            print(f"   - {req['url'][:80]}...")

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
