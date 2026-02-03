# -*- coding: utf-8 -*-
"""
产品验证器 - 实时验证推荐产品是否可购买

通过调用银行API获取当前在售产品列表，
与推荐的产品进行交叉验证，确保推荐的产品确实可以购买。

版本：1.0
日期：2026-01-21
"""

import requests
import pandas as pd
import time
import ssl
import urllib3
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# 禁用SSL警告
urllib3.disable_warnings()


class LegacySSLAdapter(HTTPAdapter):
    """自定义SSL适配器"""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


@dataclass
class VerificationResult:
    """验证结果"""
    product_code: str
    product_name: str
    is_verified: bool
    api_status: str  # 'open', 'closed', 'not_found'
    api_status_text: str
    bank_name: str


class ProductVerifier:
    """产品验证器 - 支持多银行"""

    def __init__(self):
        self.session = self._create_session()
        self.cached_products = {}  # 缓存各银行的在售产品

    def _create_session(self) -> requests.Session:
        """创建HTTP Session"""
        session = requests.Session()
        adapter = LegacySSLAdapter(pool_connections=30, pool_maxsize=30)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        })
        session.verify = False
        return session

    # =========================================================================
    # 民生银行 (CMBC官网 - 非理财子公司)
    # =========================================================================

    def fetch_cmbc_products(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        从民生银行官网API获取当前在售产品列表

        重要：使用民生银行零售官网(www.cmbc.com.cn)而非理财子公司(www.cmbcwm.com.cn)
        因为理财子公司的产品不一定在银行端销售

        Returns:
            dict: {product_code: product_info}
        """
        cache_key = 'cmbc'

        if not force_refresh and cache_key in self.cached_products:
            print(f"使用缓存的民生银行官网产品列表 ({len(self.cached_products[cache_key])} 个)")
            return self.cached_products[cache_key]

        print("正在从民生银行官网API获取当前在售产品列表...")
        print("  API: https://www.cmbc.com.cn/gw/po_web/QryAllPrdListOnmarket.do")

        # 先访问页面获取cookie
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/json;charset=UTF-8',
            'Referer': 'https://www.cmbc.com.cn/cmbc/gryw/lc/zscp/index.html',
        })

        try:
            self.session.get('https://www.cmbc.com.cn/cmbc/gryw/lc/zscp/index.html',
                             verify=False, timeout=30)
        except:
            pass

        url = 'https://www.cmbc.com.cn/gw/po_web/QryAllPrdListOnmarket.do'
        all_products = {}
        page = 1
        page_size = 100
        max_pages = 50

        while page <= max_pages:
            try:
                params = {
                    'pageNo': page,
                    'pageSize': page_size,
                    'prdAttr': 'C',
                    'label': 'all'
                }

                response = self.session.post(url, json=params, timeout=30, verify=False)
                data = response.json()

                if data.get('returnCode', {}).get('code') != 'AAAAAAA':
                    print(f"  API返回错误: {data.get('returnCode', {})}")
                    break

                prd_list = data.get('prdList', [{}])[0].get('list', [])
                total = data.get('prdList', [{}])[0].get('listSize', 0)

                if not prd_list:
                    break

                for p in prd_list:
                    code = p.get('prdCode', '')
                    if code:
                        all_products[code] = {
                            'code': code,
                            'name': p.get('prdName', ''),
                            'status': p.get('status'),
                            'prd_status': p.get('prdStatus'),
                            'risk_level': str(p.get('riskLevel', '')),
                            'risk_level_name': p.get('riskLevelName', ''),
                            'prd_type_name': p.get('prdTypeName', ''),
                            'live_time': p.get('liveTime'),
                            'usable_amt': p.get('usableAmt'),
                            'raw': p
                        }

                total_pages = (total + page_size - 1) // page_size
                print(f"  获取第{page}/{total_pages}页... ({len(prd_list)}个产品)")

                if len(all_products) >= total:
                    break

                page += 1
                time.sleep(0.2)

            except Exception as e:
                print(f"  获取第{page}页失败: {e}")
                break

        print(f"民生银行官网: 共获取 {len(all_products)} 个在售产品")
        self.cached_products[cache_key] = all_products
        return all_products

    def _parse_cmbc_status(self, product_info: Dict) -> Tuple[bool, str]:
        """
        解析民生银行产品状态

        Returns:
            (is_open, status_text)
        """
        status = product_info.get('status')
        prd_status = product_info.get('prd_status')
        name = product_info.get('name', '')

        # 产品在官网列表中就说明可以购买
        # status=1 表示在售
        if status == '1' or status == 1:
            return True, "在售"

        if prd_status == '1' or prd_status == 1:
            return True, "在售"

        return True, "在官网列表中"

    # =========================================================================
    # 信银理财 (CITIC)
    # =========================================================================

    def fetch_citic_products(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        从信银理财获取当前在售产品列表

        Note: 信银理财使用不同的API结构
        """
        cache_key = 'citic'

        if not force_refresh and cache_key in self.cached_products:
            print(f"使用缓存的信银理财产品列表 ({len(self.cached_products[cache_key])} 个)")
            return self.cached_products[cache_key]

        print("正在从信银理财API获取当前在售产品列表...")

        # 信银理财API
        api_url = "https://www.citicwm.com/citiccard/api/product/public/wmsale/list"

        all_products = {}
        page = 1

        while True:
            try:
                payload = {
                    "pageNo": page,
                    "pageSize": 50,
                    "sortType": "1",
                    "isNewest": "1"
                }
                response = self.session.post(api_url, json=payload, timeout=30)
                data = response.json()

                if data.get('code') == '0' and data.get('data', {}).get('list'):
                    products = data['data']['list']

                    for p in products:
                        code = p.get('productNo', '')
                        if code:
                            all_products[code] = {
                                'code': code,
                                'name': p.get('productName', ''),
                                'status': p.get('productStatus'),
                                'risk_level': str(p.get('riskLevel', '')),
                                'raw': p
                            }

                    total = data['data'].get('total', 0)
                    total_pages = (total + 49) // 50

                    print(f"  获取第{page}/{total_pages}页... ({len(products)}个产品)")

                    if page >= total_pages:
                        break
                    page += 1
                    time.sleep(0.3)
                else:
                    break

            except Exception as e:
                print(f"  获取第{page}页失败: {e}")
                break

        print(f"信银理财: 共获取 {len(all_products)} 个产品")
        self.cached_products[cache_key] = all_products
        return all_products

    # =========================================================================
    # 华夏理财 (Huaxia)
    # =========================================================================

    def fetch_huaxia_products(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        从华夏理财获取当前在售产品列表
        """
        cache_key = 'huaxia'

        if not force_refresh and cache_key in self.cached_products:
            print(f"使用缓存的华夏理财产品列表 ({len(self.cached_products[cache_key])} 个)")
            return self.cached_products[cache_key]

        print("正在从华夏理财API获取当前在售产品列表...")

        # 华夏理财API
        api_url = "https://www.huaxiafp.com/api/jz/getJzList"

        all_products = {}
        page = 1

        while True:
            try:
                payload = {
                    "pageNum": page,
                    "pageSize": 100,
                    "productType": "",
                    "riskLevel": "",
                    "termType": ""
                }
                response = self.session.post(api_url, json=payload, timeout=30)
                data = response.json()

                if data.get('code') == 200 and data.get('data', {}).get('list'):
                    products = data['data']['list']

                    for p in products:
                        code = p.get('cpbm', '') or p.get('productCode', '')
                        if code:
                            all_products[code] = {
                                'code': code,
                                'name': p.get('cpmc', '') or p.get('productName', ''),
                                'status': p.get('cpzt', ''),
                                'risk_level': str(p.get('fxdj', '')),
                                'raw': p
                            }

                    total = data['data'].get('total', 0)
                    total_pages = (total + 99) // 100

                    print(f"  获取第{page}/{total_pages}页... ({len(products)}个产品)")

                    if page >= total_pages or len(products) == 0:
                        break
                    page += 1
                    time.sleep(0.3)
                else:
                    break

            except Exception as e:
                print(f"  获取第{page}页失败: {e}")
                break

        print(f"华夏理财: 共获取 {len(all_products)} 个产品")
        self.cached_products[cache_key] = all_products
        return all_products

    # =========================================================================
    # 验证功能
    # =========================================================================

    def verify_products(self, product_codes: List[str],
                        bank_name: str = "民生银行") -> List[VerificationResult]:
        """
        验证产品是否可购买

        Args:
            product_codes: 产品代码列表
            bank_name: 银行名称

        Returns:
            验证结果列表
        """
        # 根据银行获取在售产品
        if '民生' in bank_name:
            api_products = self.fetch_cmbc_products()
            parse_func = self._parse_cmbc_status
        elif '信银' in bank_name or '中信' in bank_name:
            api_products = self.fetch_citic_products()
            parse_func = lambda p: (p.get('status') == '1', '在售' if p.get('status') == '1' else '停售')
        elif '华夏' in bank_name:
            api_products = self.fetch_huaxia_products()
            parse_func = lambda p: (p.get('status') == '在售', p.get('status', '未知'))
        else:
            print(f"警告: 不支持的银行 {bank_name}")
            return []

        results = []
        for code in product_codes:
            if code in api_products:
                product_info = api_products[code]
                is_open, status_text = parse_func(product_info)
                results.append(VerificationResult(
                    product_code=code,
                    product_name=product_info.get('name', ''),
                    is_verified=is_open,
                    api_status='open' if is_open else 'closed',
                    api_status_text=status_text,
                    bank_name=bank_name
                ))
            else:
                results.append(VerificationResult(
                    product_code=code,
                    product_name='',
                    is_verified=False,
                    api_status='not_found',
                    api_status_text='API中未找到此产品',
                    bank_name=bank_name
                ))

        return results

    def verify_dataframe(self, df: pd.DataFrame,
                         code_column: str = '产品代码',
                         bank_column: str = '银行') -> pd.DataFrame:
        """
        验证DataFrame中的所有产品

        Args:
            df: 包含产品信息的DataFrame
            code_column: 产品代码列名
            bank_column: 银行列名

        Returns:
            添加了验证结果的DataFrame
        """
        print("\n" + "=" * 60)
        print("产品验证")
        print("=" * 60)

        # 按银行分组验证
        banks = df[bank_column].unique()
        all_results = {}

        for bank in banks:
            bank_df = df[df[bank_column] == bank]
            codes = bank_df[code_column].tolist()

            print(f"\n验证 {bank} ({len(codes)} 个产品)...")
            results = self.verify_products(codes, bank)

            for r in results:
                all_results[r.product_code] = r

        # 添加验证结果列
        df = df.copy()
        df['验证状态'] = df[code_column].apply(
            lambda x: all_results.get(x, VerificationResult(x, '', False, 'unknown', '未验证', '')).api_status
        )
        df['验证说明'] = df[code_column].apply(
            lambda x: all_results.get(x, VerificationResult(x, '', False, 'unknown', '未验证', '')).api_status_text
        )
        df['已验证可购买'] = df[code_column].apply(
            lambda x: all_results.get(x, VerificationResult(x, '', False, 'unknown', '未验证', '')).is_verified
        )

        # 统计验证结果
        verified_count = df['已验证可购买'].sum()
        total_count = len(df)
        print(f"\n验证结果: {verified_count}/{total_count} 个产品确认可购买")

        return df


def verify_recommendations(input_file: str, output_file: str = None):
    """
    验证推荐结果中的产品

    Args:
        input_file: 推荐结果Excel文件
        output_file: 输出文件路径（可选）
    """
    print("=" * 60)
    print("理财产品推荐验证系统")
    print("=" * 60)

    verifier = ProductVerifier()

    # 读取推荐结果
    print(f"\n读取推荐结果: {input_file}")

    # 读取所有sheet
    xl = pd.ExcelFile(input_file)
    sheets = xl.sheet_names
    print(f"发现 {len(sheets)} 个工作表: {sheets}")

    verified_sheets = {}

    for sheet_name in sheets:
        if sheet_name in ['配置信息']:
            continue

        print(f"\n处理工作表: {sheet_name}")
        df = pd.read_excel(input_file, sheet_name=sheet_name)

        if '产品代码' in df.columns and '银行' in df.columns:
            df = verifier.verify_dataframe(df)
            verified_sheets[sheet_name] = df
        else:
            print(f"  跳过 (无产品代码或银行列)")
            verified_sheets[sheet_name] = df

    # 输出验证后的结果
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"D:/AI-FINANCE/产品推荐_已验证_{timestamp}.xlsx"

    print(f"\n保存验证结果到: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in verified_sheets.items():
            # 只保留已验证可购买的产品
            if '已验证可购买' in df.columns:
                verified_df = df[df['已验证可购买'] == True].copy()
                all_df = df.copy()

                # 已验证可购买的产品
                verified_df.to_excel(writer, sheet_name=f"{sheet_name[:15]}_已验证", index=False)
                # 所有产品（含验证状态）
                all_df.to_excel(writer, sheet_name=sheet_name[:20], index=False)
            else:
                df.to_excel(writer, sheet_name=sheet_name[:20], index=False)

    print(f"\n验证完成!")
    print(f"输出文件: {output_file}")

    return output_file


if __name__ == "__main__":
    # 测试验证功能
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 默认使用最新的推荐结果
        import glob
        files = glob.glob("D:/AI-FINANCE/产品推荐_*.xlsx")
        if files:
            input_file = max(files, key=lambda x: x)
            print(f"自动选择最新文件: {input_file}")
        else:
            print("未找到推荐结果文件，请先运行 product_recommender.py")
            sys.exit(1)

    verify_recommendations(input_file)
