# -*- coding: utf-8 -*-
"""
银行理财策略 Web Dashboard — Flask 服务器 V12

启动:  python web_app.py
访问:  http://localhost:8080
"""

import os
import sys

# ★ 修复 Windows 控制台中文乱码：强制 stdout/stderr 使用 UTF-8
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    for _stream in ('stdout', 'stderr'):
        _s = getattr(sys, _stream, None)
        if _s and hasattr(_s, 'reconfigure'):
            _s.reconfigure(encoding='utf-8', errors='replace')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# ★ 修复 OpenSSL 3.0+ 不支持 legacy renegotiation 的问题（民生银行网站需要）
_openssl_conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openssl_legacy.cnf')
if os.path.exists(_openssl_conf) and 'OPENSSL_CONF' not in os.environ:
    os.environ['OPENSSL_CONF'] = _openssl_conf
import json
import time
import logging
from flask import Flask, render_template, jsonify, request, Response

# 确保项目根目录在 sys.path 中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import strategy_bridge
import backtest_bridge
import crawl_bridge
import portfolio_bridge

# ── Flask 应用 ──
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
try:
    app.json.ensure_ascii = False          # Flask >= 2.2
except AttributeError:
    app.config['JSON_AS_ASCII'] = False    # Flask < 2.2 fallback
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 上传限制
app.config['TEMPLATES_AUTO_RELOAD'] = True

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s — %(message)s'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s — %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'web_app.log'), encoding='utf-8'),
        _console_handler,
    ]
)
logger = logging.getLogger('web_app')


# ── 页面 ──

@app.route('/')
def index():
    # JS 文件自动版本号: 文件修改时间戳, 改文件后浏览器自动刷新缓存
    js_path = os.path.join(BASE_DIR, 'static', 'js', 'dashboard-v2.js')
    js_ver = int(os.path.getmtime(js_path)) if os.path.exists(js_path) else 0
    return render_template('index-v2.html', js_ver=js_ver)


# ════════════════════════════════════════════
# 策略 API（保持不变）
# ════════════════════════════════════════════

@app.route('/api/strategy/summary')
def api_strategy_summary():
    return jsonify(strategy_bridge.get_summary())


@app.route('/api/strategy/top20')
def api_strategy_top20():
    return jsonify(strategy_bridge.get_top20())


@app.route('/api/strategy/opportunities')
def api_strategy_opportunities():
    return jsonify(strategy_bridge.get_opportunities())


@app.route('/api/strategy/portfolio')
def api_strategy_portfolio():
    return jsonify(strategy_bridge.get_portfolio())


@app.route('/api/strategy/patterns')
def api_strategy_patterns():
    return jsonify(strategy_bridge.get_patterns())


@app.route('/api/strategy/run', methods=['POST'])
def api_strategy_run():
    force = request.json.get('force_refresh', False) if request.is_json else False
    ok, msg = strategy_bridge.run_strategy(force_refresh=force)
    return jsonify({'ok': ok, 'message': msg})


@app.route('/api/strategy/status')
def api_strategy_status():
    """SSE — 实时推送策略执行进度"""
    def generate():
        prev = ''
        while True:
            st = strategy_bridge.get_status()
            payload = json.dumps(st, ensure_ascii=False)
            if payload != prev:
                yield f"data: {payload}\n\n"
                prev = payload
            if st['status'] in ('done', 'error', 'idle'):
                yield f"data: {payload}\n\n"
                break
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ════════════════════════════════════════════
# VIP Sniper: 双通道 + 售罄管理 API
# ════════════════════════════════════════════

@app.route('/api/strategy/vip-channels')
def api_strategy_vip_channels():
    """双通道分类结果 {list_a, list_b, unclassified, stats}"""
    return jsonify(strategy_bridge.get_vip_channels())


@app.route('/api/strategy/sold-out')
def api_strategy_sold_out():
    """当前售罄产品列表"""
    return jsonify(strategy_bridge.get_sold_out_list())


@app.route('/api/strategy/sold-out/mark', methods=['POST'])
def api_strategy_sold_out_mark():
    """标记售罄 → 记录SOPM训练数据 → 立即重分类(SOPM排序)"""
    data = request.json if request.is_json else {}
    bank = data.get('bank', '')
    code = data.get('code', '')
    if not bank or not code:
        return jsonify({'ok': False, 'message': '缺少 bank 或 code'})

    # SOPM v1: 查找产品名称和收益率（用于训练数据记录）
    name = data.get('name', '')
    yield_pct = data.get('yield', 0)
    if not name:
        recs = strategy_bridge._get('recommendations', [])
        for r in recs:
            if r.get('银行') == bank and r.get('产品代码') == code:
                name = r.get('产品名称', '')
                yield_pct = r.get('最新收益率%', 0) or 0
                break

    import post_processing
    mgr = post_processing.get_sold_out_manager()
    mgr.mark(bank, code, name=name, yield_pct=yield_pct)
    # Hot-Reload: 立即重分类（使用SOPM v1排序）
    channels = strategy_bridge.refresh_classification()
    return jsonify({'ok': True, 'message': f'已标记售罄: {bank}|{code}',
                    'stats': channels.get('stats', {})})


@app.route('/api/strategy/sold-out/unmark', methods=['POST'])
def api_strategy_sold_out_unmark():
    """取消售罄 → 立即重分类"""
    data = request.json if request.is_json else {}
    bank = data.get('bank', '')
    code = data.get('code', '')
    if not bank or not code:
        return jsonify({'ok': False, 'message': '缺少 bank 或 code'})

    import post_processing
    mgr = post_processing.get_sold_out_manager()
    mgr.unmark(bank, code)
    # Hot-Reload: 立即重分类
    channels = strategy_bridge.refresh_classification()
    return jsonify({'ok': True, 'message': f'已恢复: {bank}|{code}',
                    'stats': channels.get('stats', {})})


# ════════════════════════════════════════════
# 回测 API（保持不变）
# ════════════════════════════════════════════

@app.route('/api/backtest/results')
def api_backtest_results():
    return jsonify(backtest_bridge.get_results())


@app.route('/api/backtest/nav')
def api_backtest_nav():
    return jsonify(backtest_bridge.get_nav())


@app.route('/api/backtest/trades')
def api_backtest_trades():
    return jsonify(backtest_bridge.get_trades())


@app.route('/api/backtest/sweep')
def api_backtest_sweep():
    return jsonify(backtest_bridge.get_sweep())


@app.route('/api/backtest/patterns')
def api_backtest_patterns():
    return jsonify(backtest_bridge.get_patterns_snapshot())


@app.route('/api/backtest/run', methods=['POST'])
def api_backtest_run():
    params = request.json if request.is_json else None
    ok, msg = backtest_bridge.run_backtest(params)
    return jsonify({'ok': ok, 'message': msg})


@app.route('/api/backtest/status')
def api_backtest_status():
    """SSE — 实时推送回测执行进度"""
    def generate():
        prev = ''
        while True:
            st = backtest_bridge.get_status()
            payload = json.dumps(st, ensure_ascii=False)
            if payload != prev:
                yield f"data: {payload}\n\n"
                prev = payload
            if st['status'] in ('done', 'error', 'idle'):
                yield f"data: {payload}\n\n"
                break
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ════════════════════════════════════════════
# 净值抓取 API（新增）
# ════════════════════════════════════════════

@app.route('/api/crawl/stats')
def api_crawl_stats():
    """净值数据库状态"""
    force = request.args.get('refresh', '').lower() in ('1', 'true')
    return jsonify(crawl_bridge.get_db_stats(force_refresh=force))


@app.route('/api/crawl/banks')
def api_crawl_banks():
    """返回所有可用银行列表"""
    from crawl_master import BANK_CONFIG
    return jsonify([
        {'key': k, 'name': v['name'], 'group': v.get('group', 'extra')}
        for k, v in BANK_CONFIG.items()
    ])


@app.route('/api/crawl/run', methods=['POST'])
def api_crawl_run():
    """启动抓取 {banks:[], full:bool, max_months:int, bank_modes:{key:bool}}"""
    data = request.json if request.is_json else {}
    banks = data.get('banks', None)
    full = data.get('full', False)
    max_months = int(data.get('max_months', 0))
    bank_modes = data.get('bank_modes', None)  # {bank_key: True/False}
    ok, msg = crawl_bridge.run_crawl(
        banks=banks, full_history=full, max_months=max_months, bank_modes=bank_modes)
    return jsonify({'ok': ok, 'message': msg})


@app.route('/api/crawl/stop', methods=['POST'])
def api_crawl_stop():
    """停止抓取 {bank: key} 或 {} 停止全部"""
    data = request.json if request.is_json else {}
    bank_key = data.get('bank', None)
    ok, msg = crawl_bridge.stop_crawl(bank_key=bank_key)
    return jsonify({'ok': ok, 'message': msg})


@app.route('/api/crawl/status')
def api_crawl_status():
    """SSE — 抓取进度"""
    def generate():
        prev = ''
        while True:
            st = crawl_bridge.get_status()
            payload = json.dumps(st, ensure_ascii=False)
            if payload != prev:
                yield f"data: {payload}\n\n"
                prev = payload
            if st['status'] in ('done', 'error', 'idle', 'stopped'):
                yield f"data: {payload}\n\n"
                break
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/crawl/results')
def api_crawl_results():
    """抓取结果"""
    return jsonify(crawl_bridge.get_results())


# ════════════════════════════════════════════
# 持仓管理 API（新增）
# ════════════════════════════════════════════

@app.route('/api/portfolio/trades')
def api_portfolio_trades():
    """交易记录列表"""
    return jsonify(portfolio_bridge.get_trade_history())


@app.route('/api/portfolio/add', methods=['POST'])
def api_portfolio_add():
    """添加单笔交易"""
    data = request.json if request.is_json else {}
    result = portfolio_bridge.add_trade(
        bank=data.get('bank', ''),
        product_code=data.get('product_code', ''),
        product_name=data.get('product_name', ''),
        trade_type=data.get('trade_type', '买入'),
        amount=data.get('amount', 0),
        date=data.get('date', ''),
    )
    return jsonify(result)


@app.route('/api/portfolio/add-batch', methods=['POST'])
def api_portfolio_add_batch():
    """批量添加交易（OCR 识别结果）"""
    trades = request.json if request.is_json else []
    if not isinstance(trades, list):
        trades = []
    result = portfolio_bridge.add_trades_batch(trades)
    return jsonify(result)


@app.route('/api/portfolio/suggestions')
def api_portfolio_suggestions():
    """产品自动补全"""
    q = request.args.get('q', '')
    return jsonify(portfolio_bridge.get_product_suggestions(q))


@app.route('/api/portfolio/product-name')
def api_portfolio_product_name():
    """查询产品名"""
    bank = request.args.get('bank', '')
    code = request.args.get('code', '')
    name = portfolio_bridge.get_product_name(bank, code)
    return jsonify({'name': name})


@app.route('/api/portfolio/ocr', methods=['POST'])
def api_portfolio_ocr():
    """上传图片 OCR 识别"""
    if 'image' not in request.files:
        return jsonify({'ok': False, 'message': '未收到图片', 'trades': []})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'ok': False, 'message': '文件名为空', 'trades': []})

    image_bytes = file.read()
    result = portfolio_bridge.parse_portfolio_image(image_bytes)
    return jsonify(result)


@app.route('/api/portfolio/holding-returns')
def api_portfolio_holding_returns():
    """持仓每日收益数据"""
    return jsonify(portfolio_bridge.get_holding_returns())


# ════════════════════════════════════════════
# GPU 智能分析 API
# ════════════════════════════════════════════

@app.route('/api/gpu/info')
def api_gpu_info():
    """GPU 硬件信息"""
    try:
        from gpu_engine import gpu_info
        return jsonify(gpu_info())
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})


@app.route('/api/gpu/predict', methods=['POST'])
def api_gpu_predict():
    """触发深度学习预测（随策略运行自动调用，也可手动触发）"""
    try:
        from gpu_predictor import predict_releases
        data = request.json if request.is_json else {}
        as_of = data.get('as_of_date', None)
        result = predict_releases([], as_of_date=as_of)
        return jsonify({
            'ok': True,
            'count': len(result),
            'predictions': {f"{k[0]}|{k[1]}": v for k, v in result.items()},
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/gpu/predict/status')
def api_gpu_predict_status():
    """预测进度（当前为同步，返回完成状态）"""
    return jsonify({'status': 'idle', 'message': '就绪'})


@app.route('/api/gpu/monte-carlo', methods=['POST'])
def api_gpu_monte_carlo():
    """触发蒙特卡洛模拟"""
    try:
        from gpu_monte_carlo import MonteCarloOptimizer
        data = request.json if request.is_json else {}

        # 从策略结果获取候选产品
        product_keys = []
        if data.get('product_keys'):
            product_keys = [tuple(k.split('|')) for k in data['product_keys']]
        else:
            opps = strategy_bridge.get_opportunities()
            for o in opps:
                if '★' in o.get('操作建议', ''):
                    product_keys.append((o['银行'], o['产品代码']))
            if not product_keys:
                lib = strategy_bridge._get('product_lib', [])
                for p in lib[:20]:
                    product_keys.append((p['银行'], p['产品代码']))

        optimizer = MonteCarloOptimizer()
        result = optimizer.simulate(product_keys)
        return jsonify({'ok': True, **result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/gpu/monte-carlo/results')
def api_gpu_monte_carlo_results():
    """MC 结果（同步，直接返回）"""
    return jsonify({'ok': True, 'message': '请使用 POST /api/gpu/monte-carlo 触发模拟'})


@app.route('/api/backtest/param-sweep', methods=['POST'])
def api_backtest_param_sweep():
    """触发参数寻优"""
    data = request.json if request.is_json else {}
    param_grid = data.get('param_grid', None)
    ok, msg = backtest_bridge.run_param_sweep(param_grid)
    return jsonify({'ok': ok, 'message': msg})


@app.route('/api/backtest/param-sweep/status')
def api_backtest_param_sweep_status():
    """SSE — 寻优进度"""
    def generate():
        prev = ''
        while True:
            st = backtest_bridge.get_sweep_status()
            payload = json.dumps(st, ensure_ascii=False)
            if payload != prev:
                yield f"data: {payload}\n\n"
                prev = payload
            if st['status'] in ('done', 'error', 'idle'):
                yield f"data: {payload}\n\n"
                break
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/backtest/param-sweep/results')
def api_backtest_param_sweep_results():
    """寻优结果"""
    return jsonify(backtest_bridge.get_sweep_results())


@app.route('/api/gpu/correlation')
def api_gpu_correlation():
    """关联分析结果（缓存）"""
    return jsonify({'ok': True, 'message': '请使用 POST /api/gpu/correlation/run 触发分析'})


@app.route('/api/gpu/correlation/run', methods=['POST'])
def api_gpu_correlation_run():
    """触发关联分析"""
    try:
        from gpu_correlation import CorrelationAnalyzer
        data = request.json if request.is_json else {}
        min_overlap = data.get('min_overlap', 30)
        max_products = data.get('max_products', 200)

        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(min_overlap=min_overlap, max_products=max_products)
        return jsonify({'ok': True, **result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


# ── 启动 ──

if __name__ == '__main__':
    print("=" * 60)
    print("  千将资本-银行理财智能决策平台 V12 Quantitative Pipeline")
    print(f"  Python {sys.version}")
    print(f"  Flask  {__import__('importlib').metadata.version('flask')}")
    print("  访问  http://localhost:5000")
    print("=" * 60)

    # 启动前检查关键目录/文件
    _tpl = os.path.join(BASE_DIR, 'templates', 'index-v2.html')
    _js  = os.path.join(BASE_DIR, 'static', 'js', 'dashboard-v2.js')
    for _f in (_tpl, _js):
        if not os.path.exists(_f):
            logger.warning('缺少文件: %s', _f)

    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True, use_reloader=False)
