# -*- coding: utf-8 -*-
import json
from collections import Counter

d = json.load(open('D:/AI-FINANCE/中间结果_20260121_014814.json', 'r', encoding='utf-8'))
banks = Counter([p.get('bank_name','') for p in d])

print('各银行产品数:')
for bank, count in banks.most_common():
    print(f'  {bank}: {count}')
