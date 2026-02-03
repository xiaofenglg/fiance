# -*- coding: utf-8 -*-
"""测试工银理财API端点"""

import undetected_chromedriver as uc
import time
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

# 先访问主页建立session
print("访问主页建立session...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")
time.sleep(5)

# 测试API端点
api_endpoints = [
    ("https://wm.icbc.com.cn/clt/solr/101003", "solr搜索"),
    ("https://wm.icbc.com.cn/clt/prod/114001", "产品列表"),
    ("https://wm.icbc.com.cn/clt/info/000001", "信息1"),
    ("https://wm.icbc.com.cn/clt/info/112901", "信息2"),
]

print("\n=== 测试API端点 ===")
for url, desc in api_endpoints:
    print(f"\n测试: {desc}")
    print(f"URL: {url}")

    try:
        # 使用fetch获取数据
        js_fetch = f"""
        return await fetch('{url}', {{
            method: 'GET',
            headers: {{'Accept': 'application/json'}}
        }}).then(r => r.text()).catch(e => e.toString());
        """
        result = driver.execute_script(js_fetch)

        if result:
            print(f"响应长度: {len(result)} 字符")
            # 尝试解析JSON
            try:
                data = json.loads(result)
                print(f"JSON解析成功")
                if isinstance(data, dict):
                    print(f"  键: {list(data.keys())[:10]}")
                    # 查找列表数据
                    for key in ['data', 'list', 'rows', 'content', 'result', 'records']:
                        if key in data:
                            items = data[key]
                            if isinstance(items, list):
                                print(f"  {key}: {len(items)} 项")
                                if items and len(items) > 0:
                                    print(f"  第一项键: {list(items[0].keys()) if isinstance(items[0], dict) else type(items[0])}")
                                    print(f"  第一项: {str(items[0])[:500]}")
                            break
                elif isinstance(data, list):
                    print(f"  列表: {len(data)} 项")
            except json.JSONDecodeError:
                print(f"  非JSON响应")
                print(f"  内容预览: {result[:300]}")
        else:
            print("  无响应")

    except Exception as e:
        print(f"  错误: {e}")

# 尝试POST请求
print("\n\n=== 测试POST请求 ===")
post_endpoints = [
    ("https://wm.icbc.com.cn/clt/solr/101003", {"page": 1, "pageSize": 10}),
    ("https://wm.icbc.com.cn/clt/prod/114001", {"pageNo": 1, "pageSize": 10}),
]

for url, payload in post_endpoints:
    print(f"\nPOST: {url}")
    print(f"Payload: {payload}")

    try:
        js_post = f"""
        return await fetch('{url}', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }},
            body: JSON.stringify({json.dumps(payload)})
        }}).then(r => r.text()).catch(e => e.toString());
        """
        result = driver.execute_script(js_post)

        if result:
            print(f"响应长度: {len(result)} 字符")
            try:
                data = json.loads(result)
                print(f"JSON解析成功, 键: {list(data.keys())[:10] if isinstance(data, dict) else 'list'}")
                if isinstance(data, dict):
                    for key in ['data', 'list', 'rows', 'content', 'result', 'records', 'body']:
                        if key in data:
                            items = data[key]
                            if isinstance(items, list) and len(items) > 0:
                                print(f"  {key}: {len(items)} 项")
                                print(f"  第一项: {str(items[0])[:500]}")
                                break
                            elif isinstance(items, dict):
                                print(f"  {key}: {list(items.keys())}")
            except:
                print(f"  内容预览: {result[:500]}")
    except Exception as e:
        print(f"  错误: {e}")

print("\n完成")
driver.quit()
