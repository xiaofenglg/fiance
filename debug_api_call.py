# -*- coding: utf-8 -*-
"""直接调用产品数据API"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

print("打开光大理财...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")

# 等待页面加载
for i in range(10):
    size = len(driver.page_source)
    if size > 10000:
        break
    time.sleep(1)

print(f"页面加载完成: {len(driver.page_source)} bytes")

# 使用页面内的_$jy函数解码字符串获取API URL
print("\n解码API URL...")
js_decode = """
try {
    // 获取pageId
    var pageIdElem = document.getElementById('eprotalCurrentPageId') ||
                     document.querySelector('[id*="PageId"]');
    var pageId = pageIdElem ? pageIdElem.value : '';

    // 尝试解码API URL
    var urlPart1 = typeof _$jy === 'function' ?
        _$jy("YhCYaM6x03KaBwK9dMKe9wKS21KAeWsEaJuWbQDmvRV29HA") : '';
    var urlPart2 = typeof _$jy === 'function' ?
        _$jy("XxvrBQvpfMcyTQbaehCmaQce6RKm7wDf73CJapvxBMvpr36pNMCgfQUY2Qv2URKm7wDf73CJaVoxBRoYOwoWXQDenwDS2wq") : '';

    return {
        pageId: pageId,
        urlPart1: urlPart1,
        urlPart2: urlPart2,
        fullUrl: urlPart1 + pageId + urlPart2
    };
} catch(e) {
    return {error: e.toString()};
}
"""

api_info = driver.execute_script(js_decode)
print(f"API信息: {json.dumps(api_info, indent=2)}")

# 如果成功获取API URL，尝试直接调用
if api_info.get('fullUrl'):
    print(f"\n尝试调用API: {api_info['fullUrl']}")

    # 使用页面的jQuery发起请求
    js_api_call = """
    return new Promise((resolve) => {
        $.ajax({
            type: 'POST',
            url: arguments[0],
            dataType: 'json',
            data: {
                inputQueryStr: '',
                mjbzArr: [],
                xsqdjeArr: [],
                cpfxdjArr: [],
                sfkfsArr: [],
                pageKeyStr: 'grlc',
                currentPage: 1,
                isGrlcStr: '1',
                qclx: '',
                cpFlbs: ''
            },
            success: function(data) {
                resolve({success: true, data: data});
            },
            error: function(xhr, status, error) {
                resolve({success: false, error: error, status: status});
            }
        });
    });
    """

    driver.set_script_timeout(30)
    try:
        result = driver.execute_async_script(js_api_call + """
            var callback = arguments[arguments.length - 1];
            (""" + js_api_call.replace("return new Promise((resolve) => {", "").rstrip("}") + """).then(callback);
        """, api_info['fullUrl'])

        print(f"\nAPI响应: {json.dumps(result, indent=2, default=str)[:2000]}")

        if result.get('success') and result.get('data'):
            data = result['data']
            if 'resultList' in data:
                products = data['resultList']
                print(f"\n获取到 {len(products)} 个产品")

                for p in products[:3]:
                    print(f"\n产品: {p.get('CPCODE')} - {p.get('CPNAME')}")
                    print(f"  HTMLURL: {p.get('HTMLURL')}")

                    # 尝试在同一窗口导航到第一个产品
                    if p.get('HTMLURL'):
                        detail_url = "https://www.cebwm.com" + p.get('HTMLURL')
                        print(f"\n导航到: {detail_url}")
                        driver.get(detail_url)

                        for i in range(15):
                            size = len(driver.page_source)
                            print(f"{i+1}秒: {size} bytes")
                            if size > 10000:
                                break
                            time.sleep(1)

                        if len(driver.page_source) > 10000:
                            print("\n成功加载详情页!")

                            # 保存
                            with open("D:/AI-FINANCE/detail_via_api.html", "w", encoding="utf-8") as f:
                                f.write(driver.page_source)

                            # 点击净值标签
                            try:
                                nav_tab = driver.find_element(By.CSS_SELECTOR, ".a2")
                                nav_tab.click()
                                time.sleep(3)

                                # 切换到iframe
                                iframe = driver.find_element(By.ID, "fundValueframe")
                                print(f"iframe src: {iframe.get_attribute('src')}")
                                driver.switch_to.frame(iframe)

                                time.sleep(3)

                                # 提取净值数据
                                body_text = driver.execute_script("return document.body.innerText")
                                print(f"\niframe内容长度: {len(body_text)}")

                                import re
                                nav_data = re.findall(r'(\d{4}-\d{2}-\d{2})\s+([\d.]+)', body_text)
                                if nav_data:
                                    print(f"找到净值数据: {nav_data[:5]}")
                                else:
                                    print("未找到净值数据，打印前500字符:")
                                    print(body_text[:500])

                            except Exception as e:
                                print(f"详情页处理失败: {e}")

                        break
    except Exception as e:
        print(f"API调用失败: {e}")

# 备选方案：直接发起XHR请求获取数据
print("\n\n备选方案：监听网络请求...")
driver.get("https://www.cebwm.com/wealth/grlc/index.html")
time.sleep(5)

# 使用performance API获取网络请求
js_network = """
var entries = performance.getEntriesByType('resource');
var apiCalls = entries.filter(e =>
    e.name.includes('eportal') ||
    e.name.includes('api') ||
    e.name.includes('json')
);
return apiCalls.map(e => ({
    name: e.name,
    type: e.initiatorType,
    duration: e.duration
}));
"""

network = driver.execute_script(js_network)
print("网络请求:")
for req in network[:10]:
    print(f"  {req['name'][:100]}...")

print("\n完成")
driver.quit()
