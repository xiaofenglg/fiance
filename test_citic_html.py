"""分析中信理财HTML页面结构"""
import requests
import re
import urllib3
urllib3.disable_warnings()

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
})
session.verify = False

base_url = "https://www.citic-wealth.com"

# 1. 获取主页面
print("="*60)
print("1. 获取净值查询页面HTML")
print("="*60)
resp = session.get(f"{base_url}/yymk/lccs/", timeout=30)
html = resp.text
print(f"页面长度: {len(html)} 字符")

# 保存HTML用于分析
with open("citic_page.html", "w", encoding="utf-8") as f:
    f.write(html)
print("已保存到 citic_page.html")

# 2. 查找产品相关链接
print("\n" + "="*60)
print("2. 查找产品相关链接")
print("="*60)
# 查找所有href链接
links = re.findall(r'href=["\']([^"\']+)["\']', html)
for link in links:
    if any(kw in link.lower() for kw in ['product', 'detail', 'xq', 'mrg', 'dk']):
        print(f"  {link}")

# 3. 查找数据脚本
print("\n" + "="*60)
print("3. 查找页面中的数据")
print("="*60)
# 查找JSON数据
json_matches = re.findall(r'var\s+(\w+)\s*=\s*(\[[\s\S]*?\]);', html)
for name, data in json_matches[:5]:
    print(f"变量 {name}: {data[:200]}...")

# 查找产品名称
product_names = re.findall(r'["\']([^"\']*(?:理财|固收|稳健|天天|日日)[^"\']*)["\']', html)
print(f"\n找到 {len(product_names)} 个可能的产品名称:")
for name in set(product_names)[:10]:
    print(f"  {name}")

# 4. 查找分类标签
print("\n" + "="*60)
print("4. 查找分类标签和按钮")
print("="*60)
tabs = re.findall(r'class=["\'][^"\']*(?:tab|nav|menu)[^"\']*["\'][^>]*>([^<]+)<', html, re.I)
for tab in set(tabs)[:10]:
    if tab.strip():
        print(f"  {tab.strip()}")

# 5. 查找onclick事件
print("\n" + "="*60)
print("5. 查找onclick事件")
print("="*60)
onclicks = re.findall(r'onclick=["\']([^"\']+)["\']', html)
for oc in set(onclicks)[:10]:
    print(f"  {oc}")

print("\n完成!")
