# -*- coding: utf-8 -*-
"""捕获工银理财的osnumber参数"""

import undetected_chromedriver as uc
import time
import json
import re

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

print("等待页面加载和API请求...")
time.sleep(8)

# 获取捕获的osnumber
params = driver.execute_script("return window._requestParams || []")
print(f"\n捕获到的请求参数: {len(params)}")
for p in params:
    print(f"  osnumber: {p['osnumber']}")
    print(f"  trans_code: {p['trans_code']}")

# 获取第一个有效的osnumber
osnumber = None
for p in params:
    if p.get('osnumber'):
        osnumber = p['osnumber']
        break

if osnumber:
    print(f"\n=== 使用捕获的osnumber测试API ===")
    print(f"osnumber: {osnumber}")

    # 测试API
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
    print(f"\n响应:")
    print(f"  succ: {result.get('succ')}")
    print(f"  total: {result.get('total')}")

    if result.get('rows'):
        rows = result['rows']
        print(f"  rows: {len(rows)}")
        if rows:
            first = rows[0]
            print(f"\n  第一个产品:")
            for key in ['product_code', 'product_name', 'networth_id', 'show_date_time']:
                if key in first:
                    val = str(first[key])[:50]
                    print(f"    {key}: {val}")

            # 测试获取详情
            networth_id = first.get('networth_id')
            if networth_id:
                print(f"\n=== 测试获取产品详情 ===")
                print(f"networth_id: {networth_id}")

                # 尝试112902端点
                js_detail = f"""
                return await fetch('/clt/info/112902', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }},
                    body: JSON.stringify({{
                        "networth_id": "{networth_id}",
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
                print(f"\n详情响应:")
                print(f"  succ: {detail.get('succ')}")
                print(f"  keys: {list(detail.keys()) if isinstance(detail, dict) else 'N/A'}")

                # 打印详情内容
                detail_str = json.dumps(detail, ensure_ascii=False, indent=2)[:1500]
                print(f"  内容:\n{detail_str}")
else:
    print("\n未捕获到osnumber")

print("\n完成")
driver.quit()
