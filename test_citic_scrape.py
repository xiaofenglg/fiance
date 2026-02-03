"""使用Selenium抓取中信理财产品数据"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import json
import time
import re

def setup_driver():
    """设置Chrome驱动"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)
    return driver

def extract_products(driver, category_name):
    """从当前页面提取产品数据"""
    products = []

    try:
        # 等待产品列表加载
        time.sleep(2)

        # 获取所有产品行
        product_names = driver.find_elements(By.CSS_SELECTOR, '.item-productName')
        product_codes = driver.find_elements(By.CSS_SELECTOR, '.item-productCode')
        product_rows = driver.find_elements(By.CSS_SELECTOR, '.list-item-right')

        print(f"      找到 {len(product_names)} 个产品名称, {len(product_rows)} 个数据行")

        for i, (name_el, code_el) in enumerate(zip(product_names, product_codes)):
            try:
                product = {
                    'name': name_el.text.strip(),
                    'code': code_el.text.strip(),
                    'category': category_name,
                }

                # 获取对应的数据行
                if i < len(product_rows):
                    row = product_rows[i]
                    cells = row.find_elements(By.CSS_SELECTOR, '.list-item-content-right')
                    if len(cells) >= 5:
                        product['type'] = cells[0].text.strip()
                        product['min_amount'] = cells[1].text.strip()
                        product['risk_level'] = cells[2].text.strip()
                        # cells[3] is "查看" button for benchmark
                        product['offering_type'] = cells[4].text.strip()

                if product['code']:
                    products.append(product)
            except Exception as e:
                continue

    except Exception as e:
        print(f"      提取产品失败: {e}")

    return products

def scroll_to_load_all(driver, max_scrolls=50):
    """滚动加载所有产品"""
    last_count = 0
    scroll_count = 0

    while scroll_count < max_scrolls:
        # 滚动到底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        # 检查产品数量
        products = driver.find_elements(By.CSS_SELECTOR, '.item-productCode')
        current_count = len(products)

        if current_count == last_count:
            # 没有新产品加载
            break

        last_count = current_count
        scroll_count += 1
        print(f"      滚动 {scroll_count}: {current_count} 个产品")

    return last_count

def main():
    print("=" * 60)
    print("使用Selenium抓取中信理财产品数据")
    print("=" * 60)

    driver = None
    all_products = []

    try:
        print("\n1. 启动Chrome...")
        driver = setup_driver()

        print("\n2. 访问中信理财页面...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(5)

        print(f"   当前URL: {driver.current_url}")
        print(f"   页面标题: {driver.title}")

        # 切换到iframe
        print("\n3. 切换到iframe...")
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if not iframes:
            print("   未找到iframe!")
            return

        driver.switch_to.frame(iframes[0])
        time.sleep(3)

        # 获取分类标签
        print("\n4. 获取产品分类...")
        categories = ["募集期", "每日购", "定开", "货币", "封闭"]

        # 只抓取"每日购"和"定开"
        target_categories = ["每日购", "定开"]

        for category in target_categories:
            print(f"\n   [{category}] 切换分类...")

            # 点击分类标签
            try:
                tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory, .typeCategory-select')
                for tab in tabs:
                    if category in tab.text:
                        tab.click()
                        time.sleep(3)
                        break

                # 滚动加载所有产品
                print(f"   [{category}] 滚动加载...")
                total = scroll_to_load_all(driver)

                # 提取产品数据
                print(f"   [{category}] 提取数据...")
                products = extract_products(driver, category)
                all_products.extend(products)
                print(f"   [{category}] 提取到 {len(products)} 个产品")

            except Exception as e:
                print(f"   [{category}] 错误: {e}")

        # 保存结果
        print(f"\n5. 保存结果...")
        print(f"   总共抓取 {len(all_products)} 个产品")

        # 保存为JSON
        with open("citic_products.json", "w", encoding="utf-8") as f:
            json.dump(all_products, f, ensure_ascii=False, indent=2)
        print("   已保存到 citic_products.json")

        # 显示样例
        print("\n6. 产品样例:")
        for i, p in enumerate(all_products[:10]):
            print(f"   {i+1}. {p.get('code')}: {p.get('name')[:30]}... ({p.get('category')})")

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
