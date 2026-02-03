# -*- coding: utf-8 -*-
"""检查工银理财API完整响应"""

import undetected_chromedriver as uc
import time
import json

print("启动Chrome...")
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
driver = uc.Chrome(options=options, use_subprocess=True)

# 注入XHR拦截器来捕获请求参数
intercept_script = """
(function() {
    window._requestParams = [];
    var origSend = XMLHttpRequest.prototype.send;

    XMLHttpRequest.prototype.send = function(body) {
        if (body) {
            try {
                var parsed = JSON.parse(body);
                if (parsed.head_osnumber) {
                    window._requestParams.push({
                        osnumber: parsed.head_osnumber,
                        trans_code: parsed.head_trans_code,
                        time: new Date().toISOString()
                    });
                }
            } catch(e) {}
        }
        return origSend.apply(this, arguments);
    };
})();
"""

driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': intercept_script})

print("\n打开净值披露页...")
driver.get("https://wm.icbc.com.cn/netWorthDisclosure")

print("等待页面加载...")
time.sleep(8)

# 获取捕获的osnumber
params = driver.execute_script("return window._requestParams || []")
osnumber = None
for p in params:
    if p.get('osnumber'):
        osnumber = p['osnumber']
        break

if not osnumber:
    print("未捕获到osnumber")
    driver.quit()
    exit()

print(f"捕获到 osnumber: {osnumber}")

# 获取完整的产品列表响应
js_fetch = f"""
return await fetch('/clt/info/112901', {{
    method: 'POST',
    headers: {{
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }},
    body: JSON.stringify({{
        "menu_id": "information_disclosure",
        "route": "/netWorthDisclosure",
        "full_route": "/netWorthDisclosure",
        "page_num": 1,
        "page_size": 10,
        "product_name": "",
        "head_system_id": "gtpoints",
        "head_channel_id": "0",
        "head_trans_code": "112901",
        "head_jsession_id": "",
        "batch_param": "",
        "head_osnumber": "{osnumber}"
    }})
}}).then(r => r.json()).catch(e => ({{error: e.toString()}}));
"""

result = driver.execute_script(js_fetch)

print(f"\n=== API 112901 完整响应 ===")
print(f"succ: {result.get('succ')}")
print(f"total: {result.get('total')}")
print(f"响应keys: {list(result.keys())}")

if result.get('rows'):
    rows = result['rows']
    print(f"\n共 {len(rows)} 行")

    # 打印第一行的所有字段
    if rows:
        first = rows[0]
        print(f"\n第一行所有字段:")
        for key, val in first.items():
            val_str = str(val)[:100]
            print(f"  {key}: {val_str}")

        # 查看是否有净值相关字段
        nav_fields = ['nav', 'networth', 'unit_nav', 'unit_value', 'total_nav',
                      'cumulative', 'value', 'price', 'jz', 'ljjz']
        print(f"\n搜索净值相关字段:")
        found = False
        for field in first.keys():
            field_lower = field.lower()
            for nf in nav_fields:
                if nf in field_lower:
                    print(f"  找到: {field} = {first[field]}")
                    found = True
        if not found:
            print("  未找到明显的净值字段")

# 检查其他可能的API端点
print("\n\n=== 测试其他API端点 ===")

# 尝试112903
js_fetch_903 = f"""
return await fetch('/clt/info/112903', {{
    method: 'POST',
    headers: {{
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }},
    body: JSON.stringify({{
        "menu_id": "information_disclosure",
        "route": "/netWorthDisclosure",
        "full_route": "/netWorthDisclosure",
        "head_system_id": "gtpoints",
        "head_channel_id": "0",
        "head_trans_code": "112903",
        "head_jsession_id": "",
        "batch_param": "",
        "head_osnumber": "{osnumber}"
    }})
}}).then(r => r.json()).catch(e => ({{error: e.toString()}}));
"""
result_903 = driver.execute_script(js_fetch_903)
print(f"\n112903: {result_903}")

# 尝试其他格式的112902
if result.get('rows') and result['rows']:
    first_row = result['rows'][0]
    networth_id = first_row.get('networth_id')
    product_code = first_row.get('product_code')

    print(f"\n\n=== 用不同参数测试112902 ===")
    print(f"使用 networth_id: {networth_id}")
    print(f"使用 product_code: {product_code}")

    # 尝试用product_code
    js_detail = f"""
    return await fetch('/clt/info/112902', {{
        method: 'POST',
        headers: {{
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }},
        body: JSON.stringify({{
            "product_code": "{product_code}",
            "menu_id": "information_disclosure",
            "route": "/netWorthDisclosure",
            "full_route": "/netWorthDisclosure",
            "head_system_id": "gtpoints",
            "head_channel_id": "0",
            "head_trans_code": "112902",
            "head_jsession_id": "",
            "batch_param": "",
            "head_osnumber": "{osnumber}"
        }})
    }}).then(r => r.json()).catch(e => ({{error: e.toString()}}));
    """
    detail = driver.execute_script(js_detail)
    print(f"用product_code: {json.dumps(detail, ensure_ascii=False)[:500]}")

print("\n完成")
driver.quit()
