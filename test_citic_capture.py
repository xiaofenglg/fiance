"""使用Selenium捕获中信理财实际API调用"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import json
import time

def setup_driver():
    """设置Chrome驱动"""
    options = Options()
    # 不使用headless以便观察
    # options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')

    # 启用网络日志
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    driver = webdriver.Chrome(options=options)
    return driver

def capture_api_calls(driver, description=""):
    """捕获API调用"""
    logs = driver.get_log('performance')
    api_calls = []

    for log in logs:
        try:
            message = json.loads(log['message'])
            method = message.get('message', {}).get('method', '')

            if method == 'Network.requestWillBeSent':
                request = message['message']['params'].get('request', {})
                url = request.get('url', '')

                if 'api' in url.lower() or 'product' in url.lower():
                    api_calls.append({
                        'method': request.get('method'),
                        'url': url,
                        'headers': request.get('headers', {}),
                        'postData': request.get('postData', ''),
                    })

            elif method == 'Network.responseReceived':
                response = message['message']['params'].get('response', {})
                url = response.get('url', '')
                if 'api' in url.lower() or 'product' in url.lower():
                    # 尝试获取响应体
                    request_id = message['message']['params'].get('requestId')
                    try:
                        body = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                        if body:
                            api_calls.append({
                                'type': 'response',
                                'url': url,
                                'body': body.get('body', '')[:500]
                            })
                    except:
                        pass

        except:
            continue

    if api_calls:
        print(f"\n[{description}] 捕获到 {len(api_calls)} 个API调用:")
        for call in api_calls[:10]:
            if call.get('type') == 'response':
                print(f"  响应: {call['url'][:80]}")
                print(f"    内容: {call['body'][:200]}...")
            else:
                print(f"  {call['method']} {call['url'][:80]}")
                if call.get('postData'):
                    print(f"    数据: {call['postData'][:200]}")

    return api_calls

def main():
    print("=" * 60)
    print("使用Selenium捕获中信理财API调用")
    print("=" * 60)

    driver = None
    try:
        print("\n1. 启动Chrome...")
        driver = setup_driver()

        # 启用CDP网络监听
        driver.execute_cdp_cmd('Network.enable', {})

        print("\n2. 访问中信理财页面...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(5)

        print(f"   当前URL: {driver.current_url}")

        # 查找并切换到iframe
        print("\n3. 切换到iframe...")
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if iframes:
            driver.switch_to.frame(iframes[0])
            time.sleep(3)

        # 捕获初始API调用
        capture_api_calls(driver, "初始加载")

        # 清除日志
        driver.get_log('performance')

        # 点击"每日购"标签
        print("\n4. 点击'每日购'标签...")
        try:
            tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory')
            for tab in tabs:
                if '每日购' in tab.text:
                    tab.click()
                    time.sleep(3)
                    break

            capture_api_calls(driver, "每日购")
        except Exception as e:
            print(f"   点击每日购失败: {e}")

        # 清除日志
        driver.get_log('performance')

        # 点击"定开"标签
        print("\n5. 点击'定开'标签...")
        try:
            tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory')
            for tab in tabs:
                if '定开' in tab.text:
                    tab.click()
                    time.sleep(3)
                    break

            capture_api_calls(driver, "定开")
        except Exception as e:
            print(f"   点击定开失败: {e}")

        # 提取当前页面的产品数据
        print("\n6. 提取页面产品数据...")
        try:
            # 获取产品名称
            names = driver.find_elements(By.CSS_SELECTOR, '.item-productName')
            codes = driver.find_elements(By.CSS_SELECTOR, '.item-productCode')

            print(f"   找到 {len(names)} 个产品")
            for i, (name, code) in enumerate(zip(names[:10], codes[:10])):
                print(f"   - {code.text}: {name.text[:40]}...")

        except Exception as e:
            print(f"   提取产品失败: {e}")

        # 保存当前页面源码
        page_source = driver.page_source
        with open("citic_current.html", "w", encoding="utf-8") as f:
            f.write(page_source)
        print("\n   已保存当前页面到 citic_current.html")

        input("\n按Enter键关闭浏览器...")

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
