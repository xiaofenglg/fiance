"""测试抓取20条中信理财数据"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=options)

def extract_products(driver, category, limit=10):
    """提取产品数据"""
    products = []
    time.sleep(2)

    # 获取表头
    headers = driver.find_elements(By.CSS_SELECTOR, '.list-item-title-right .content')
    header_texts = [h.text.strip() for h in headers]
    print(f"   表头: {header_texts}")

    # 获取产品
    names = driver.find_elements(By.CSS_SELECTOR, '.item-productName')
    codes = driver.find_elements(By.CSS_SELECTOR, '.item-productCode div')
    rows = driver.find_elements(By.CSS_SELECTOR, '.list-item-right.list-Height70')

    for i in range(min(limit, len(names))):
        try:
            product = {
                '产品名称': names[i].text.strip(),
                '产品代码': codes[i].text.strip() if i < len(codes) else '',
                '产品类别': category,
            }

            if i < len(rows):
                cells = rows[i].find_elements(By.CSS_SELECTOR, '.list-item-content-right')
                for j, cell in enumerate(cells):
                    text = cell.text.strip()
                    if j < len(header_texts):
                        header = header_texts[j]
                        if '单位净值' in header:
                            if '\n' in text:
                                parts = text.split('\n')
                                product['单位净值'] = parts[0]
                                product['净值日期'] = parts[1] if len(parts) > 1 else ''
                            else:
                                product['单位净值'] = text
                        elif '累计净值' in header:
                            if '\n' in text:
                                product['累计净值'] = text.split('\n')[0]
                            else:
                                product['累计净值'] = text
                        elif '起购金额' in header:
                            product['起购金额'] = text
                        elif '风险等级' in header:
                            product['风险等级'] = text
                        elif '成立日' in header:
                            product['成立日期'] = text
                        elif '募集方式' in header:
                            product['募集方式'] = text
                        elif '持有期' in header:
                            product['持有期'] = text
                        elif '开放日' in header:
                            product['下一开放日'] = text

            products.append(product)
        except Exception as e:
            print(f"   提取第{i+1}个产品失败: {e}")

    return products

def main():
    print("=" * 60)
    print("测试抓取20条中信理财数据")
    print("=" * 60)

    driver = None
    all_products = []

    try:
        driver = setup_driver()

        print("\n1. 访问页面...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(5)

        # 切换到iframe
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if iframes:
            driver.switch_to.frame(iframes[0])
            time.sleep(3)

        # 抓取每日购 (10条)
        print("\n2. 抓取每日购产品 (10条)...")
        tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory, .typeCategory-select')
        for tab in tabs:
            if '每日购' in tab.text:
                tab.click()
                time.sleep(3)
                break
        products = extract_products(driver, '每日购', 10)
        all_products.extend(products)
        print(f"   获取到 {len(products)} 条")

        # 抓取定开 (10条)
        print("\n3. 抓取定开产品 (10条)...")
        tabs = driver.find_elements(By.CSS_SELECTOR, '.typeCategory, .typeCategory-select')
        for tab in tabs:
            if '定开' in tab.text:
                tab.click()
                time.sleep(3)
                break
        products = extract_products(driver, '定开', 10)
        all_products.extend(products)
        print(f"   获取到 {len(products)} 条")

        # 保存到Excel
        print(f"\n4. 保存到Excel...")
        df = pd.DataFrame(all_products)

        # 调整列顺序
        cols = ['产品代码', '产品名称', '产品类别', '单位净值', '累计净值', '净值日期',
                '起购金额', '风险等级', '成立日期', '持有期', '下一开放日', '募集方式']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        # 保存
        df.to_excel("citic_test_20.xlsx", index=False)
        print(f"   已保存到 citic_test_20.xlsx ({len(all_products)} 条记录)")

        # 显示预览
        print("\n5. 数据预览:")
        print(df.to_string(index=False))

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
