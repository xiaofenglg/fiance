"""改进的中信理财抓取脚本"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
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

def extract_products_improved(driver, category_name):
    """改进的产品数据提取"""
    products = []
    time.sleep(2)

    # 获取产品名称和代码
    product_names = driver.find_elements(By.CSS_SELECTOR, '.item-productName')
    product_codes = driver.find_elements(By.CSS_SELECTOR, '.item-productCode div')

    # 获取右侧数据容器
    right_rows = driver.find_elements(By.CSS_SELECTOR, '.list-item-right.list-Height70')

    print(f"      名称: {len(product_names)}, 代码: {len(product_codes)}, 数据行: {len(right_rows)}")

    # 先获取表头确定列结构
    headers = driver.find_elements(By.CSS_SELECTOR, '.list-item-title-right .content')
    header_texts = [h.text.strip() for h in headers]
    print(f"      表头: {header_texts}")

    for i, (name_el, code_el) in enumerate(zip(product_names, product_codes)):
        try:
            product = {
                'name': name_el.text.strip(),
                'code': code_el.text.strip(),
                'category': category_name,
            }

            # 获取对应的数据行
            if i < len(right_rows):
                row = right_rows[i]
                cells = row.find_elements(By.CSS_SELECTOR, '.list-item-content-right')

                # 根据表头动态解析
                for j, cell in enumerate(cells):
                    cell_text = cell.text.strip()
                    if j < len(header_texts):
                        header = header_texts[j]
                        if '产品类型' in header:
                            product['product_type'] = cell_text
                        elif '起购金额' in header:
                            product['min_amount'] = cell_text
                        elif '风险等级' in header:
                            product['risk_level'] = cell_text
                        elif '业绩' in header or '基准' in header:
                            product['benchmark'] = cell_text
                        elif '募集' in header:
                            product['offering_type'] = cell_text
                        elif '净值' in header and '日期' not in header:
                            # 可能包含净值和日期
                            if '\n' in cell_text:
                                parts = cell_text.split('\n')
                                product['nav'] = parts[0]
                                if len(parts) > 1:
                                    product['nav_date'] = parts[1]
                            else:
                                product['nav'] = cell_text
                        elif '净值日期' in header or '日期' in header:
                            product['nav_date'] = cell_text
                        elif '成立日' in header:
                            product['setup_date'] = cell_text
                    else:
                        # 未知列
                        product[f'col_{j}'] = cell_text

            if product['code']:
                products.append(product)
        except Exception as e:
            continue

    return products

def scroll_to_load_all(driver, max_scrolls=100):
    """滚动加载所有产品"""
    last_count = 0
    scroll_count = 0

    while scroll_count < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.8)

        products = driver.find_elements(By.CSS_SELECTOR, '.item-productCode')
        current_count = len(products)

        if current_count == last_count:
            break

        last_count = current_count
        scroll_count += 1

        if scroll_count % 10 == 0:
            print(f"      滚动 {scroll_count}: {current_count} 个产品")

    return last_count

def main():
    print("=" * 60)
    print("改进的中信理财抓取脚本")
    print("=" * 60)

    driver = None
    all_products = []

    try:
        print("\n1. 启动Chrome...")
        driver = setup_driver()

        print("\n2. 访问中信理财页面...")
        driver.get("https://www.citic-wealth.com/yymk/lccs/")
        time.sleep(5)

        # 切换到iframe
        print("\n3. 切换到iframe...")
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if not iframes:
            print("   未找到iframe!")
            return
        driver.switch_to.frame(iframes[0])
        time.sleep(3)

        # 只抓取"每日购"和"定开"
        target_categories = ["每日购", "定开"]

        for category in target_categories:
            print(f"\n   [{category}] 切换分类...")

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
                print(f"   [{category}] 共加载 {total} 个产品")

                # 提取产品数据
                print(f"   [{category}] 提取数据...")
                products = extract_products_improved(driver, category)
                all_products.extend(products)
                print(f"   [{category}] 提取到 {len(products)} 个产品")

            except Exception as e:
                print(f"   [{category}] 错误: {e}")

        # 保存结果
        print(f"\n4. 保存结果...")
        print(f"   总共抓取 {len(all_products)} 个产品")

        with open("citic_products_v2.json", "w", encoding="utf-8") as f:
            json.dump(all_products, f, ensure_ascii=False, indent=2)
        print("   已保存到 citic_products_v2.json")

        # 显示样例 - 查看数据结构
        print("\n5. 数据结构示例:")
        if all_products:
            print(json.dumps(all_products[0], ensure_ascii=False, indent=2))

        # 统计
        print("\n6. 统计信息:")
        by_category = {}
        for p in all_products:
            cat = p.get('category', '未知')
            by_category[cat] = by_category.get(cat, 0) + 1
        for cat, count in by_category.items():
            print(f"   {cat}: {count}")

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
