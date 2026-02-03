/* ═══════════════════════════════════════════════════════════
   AI-FINANCE Dashboard — Frontend Logic + ECharts
   ═══════════════════════════════════════════════════════════ */

// ── Globals ──
const chartInstances = {};
let currentTab = 'overview';

// ── Utils ──
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }
function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }
function setHTML(id, v) { const el = document.getElementById(id); if (el) el.innerHTML = v; }

function fmt(n, d) {
    if (n === null || n === undefined || n === '' || isNaN(n)) return '--';
    return Number(n).toFixed(d === undefined ? 2 : d);
}

function fmtPct(n) {
    if (n === null || n === undefined || n === '' || isNaN(n)) return '--';
    return Number(n).toFixed(2) + '%';
}

function retClass(v) {
    if (v === null || v === undefined || v === '') return '';
    v = Number(v);
    if (v > 0) return 'td-pos';
    if (v < 0) return 'td-neg';
    return '';
}

function numCell(v, d) {
    const cls = retClass(v);
    return `<td class="td-num ${cls}">${fmt(v, d)}</td>`;
}

function tagCell(text, type) {
    const colors = {
        buy: 'background:rgba(16,185,129,0.15);color:#10b981',
        watch: 'background:rgba(245,158,11,0.15);color:#f59e0b',
        sell: 'background:rgba(239,68,68,0.15);color:#ef4444',
        hold: 'background:rgba(59,130,246,0.15);color:#3b82f6',
        info: 'background:rgba(139,92,246,0.15);color:#8b5cf6',
    };
    return `<span class="td-tag" style="${colors[type] || colors.info}">${text}</span>`;
}

function adviceTag(advice) {
    if (!advice) return '';
    if (advice.includes('★★★')) return tagCell(advice, 'buy');
    if (advice.includes('★★')) return tagCell(advice, 'buy');
    if (advice.includes('★')) return tagCell(advice, 'buy');
    if (advice.includes('可买入')) return tagCell(advice, 'buy');
    if (advice.includes('☆')) return tagCell(advice, 'watch');
    if (advice.includes('卖出') || advice.includes('赎回')) return tagCell(advice, 'sell');
    if (advice.includes('持有') || advice.includes('智持')) return tagCell(advice, 'hold');
    return tagCell(advice, 'info');
}

// ── Clock ──
function updateClock() {
    const d = new Date();
    setText('clock', d.toLocaleString('zh-CN', { hour12: false }));
}
setInterval(updateClock, 1000);
updateClock();

// ── Tab Navigation ──
$$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.tab-btn').forEach(b => b.classList.remove('active'));
        $$('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.dataset.tab;
        const panel = document.getElementById('tab-' + tab);
        if (panel) panel.classList.add('active');
        currentTab = tab;
        resizeCharts();
    });
});

// ── ECharts Helpers ──
function getChart(id) {
    if (!chartInstances[id]) {
        const el = document.getElementById(id);
        if (!el) return null;
        chartInstances[id] = echarts.init(el, 'dark');
    }
    return chartInstances[id];
}

function resizeCharts() {
    Object.values(chartInstances).forEach(c => c && c.resize());
}
window.addEventListener('resize', resizeCharts);

const CHART_THEME = {
    backgroundColor: 'transparent',
    textStyle: { fontFamily: 'Inter, sans-serif', color: '#8b95a8' },
    grid: { top: 40, right: 20, bottom: 30, left: 50, containLabel: true },
};

// ── API Fetchers ──

async function fetchJSON(url) {
    const res = await fetch(url);
    return res.json();
}

// ── Load Overview ──

async function loadOverview() {
    const data = await fetchJSON('/api/strategy/summary');
    if (!data || !data.product_lib_count) return;

    setText('kpiLibCount', data.product_lib_count);
    setText('kpiLibBadge', 'LIB');
    setText('kpiWatchCount', data.watch_pool_count);
    setText('kpiOppCount', data.opportunity_count);
    setText('kpiBuyCount', data.buy_signal_count);
    setText('kpiBuySigBadge', data.buy_signal_count + ' BUY');
    setText('kpiAvgRet', fmtPct(data.avg_return));
    setText('kpiHoldCount', data.holding_count);
    setText('kpiHoldAmt', fmt(data.total_holding_amount / 10000, 0) + '万');

    // Bank distribution chart
    const bankDist = data.bank_distribution || {};
    const banks = Object.keys(bankDist);
    const bankVals = Object.values(bankDist);
    const chartBank = getChart('chartBankDist');
    if (chartBank && banks.length > 0) {
        chartBank.setOption({
            ...CHART_THEME,
            tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
            series: [{
                type: 'pie',
                radius: ['40%', '70%'],
                label: { color: '#8b95a8', fontSize: 11 },
                data: banks.map((b, i) => ({ name: b, value: bankVals[i] })),
                emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.5)' } },
            }],
        });
    }

    // Signal strength distribution
    const opps = await fetchJSON('/api/strategy/opportunities');
    if (opps && opps.length > 0) {
        const buckets = { '>5%': 0, '4-5%': 0, '3.5-4%': 0, '3-3.5%': 0, '<3%': 0 };
        opps.forEach(o => {
            const r = o['最新收益率%'] || 0;
            if (r > 5) buckets['>5%']++;
            else if (r > 4) buckets['4-5%']++;
            else if (r > 3.5) buckets['3.5-4%']++;
            else if (r > 3) buckets['3-3.5%']++;
            else buckets['<3%']++;
        });
        const chartSig = getChart('chartSignalDist');
        if (chartSig) {
            chartSig.setOption({
                ...CHART_THEME,
                tooltip: { trigger: 'axis' },
                xAxis: { type: 'category', data: Object.keys(buckets), axisLabel: { color: '#8b95a8' } },
                yAxis: { type: 'value', axisLabel: { color: '#8b95a8' } },
                series: [{
                    type: 'bar',
                    data: Object.values(buckets),
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: '#3b82f6' },
                            { offset: 1, color: '#1e40af' },
                        ]),
                        borderRadius: [4, 4, 0, 0],
                    },
                    barWidth: '50%',
                }],
            });
        }
    }
}

// ── Load Recommendations ──

async function loadRecommendations() {
    const data = await fetchJSON('/api/strategy/top20');
    setText('recCount', data.length || 0);
    if (!data || data.length === 0) return;

    let html = '<table><thead><tr>';
    html += '<th>#</th><th>Bank</th><th>Code</th><th>Name</th><th>Liquidity</th>';
    html += '<th>Success%</th><th>Signals</th><th>Return%</th><th>Score</th>';
    html += '<th>Forecast</th><th>Advice</th>';
    html += '</tr></thead><tbody>';

    data.forEach(r => {
        const fc = r['前瞻加成'] === '是'
            ? tagCell(`${r['预测释放日']} (${fmtPct(r['预测置信度']*100)})`, 'info')
            : '';
        html += '<tr>';
        html += `<td class="td-num">${r['排名']}</td>`;
        html += `<td>${r['银行']}</td>`;
        html += `<td style="font-family:var(--font-mono)">${r['产品代码']}</td>`;
        html += `<td>${(r['产品名称'] || '').substring(0, 22)}</td>`;
        html += `<td>${r['流动性']}</td>`;
        html += numCell(r['历史成功率%'], 1);
        html += `<td class="td-num">${r['历史信号次数']}</td>`;
        html += numCell(r['最新收益率%'], 2);
        html += numCell(r['综合得分'], 1);
        html += `<td>${fc}</td>`;
        html += `<td>${adviceTag(r['操作建议'])}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('recTableWrap', html);
}

// ── Load Opportunities ──

async function loadOpportunities() {
    const data = await fetchJSON('/api/strategy/opportunities');
    setText('oppCount', data.length || 0);
    if (!data || data.length === 0) return;

    // Sort: buy signals first
    data.sort((a, b) => {
        const pa = (a['操作建议'] || '').includes('★') ? 0 : 1;
        const pb = (b['操作建议'] || '').includes('★') ? 0 : 1;
        if (pa !== pb) return pa - pb;
        return (b['最新收益率%'] || 0) - (a['最新收益率%'] || 0);
    });

    let html = '<table><thead><tr>';
    html += '<th>Bank</th><th>Code</th><th>Name</th><th>Source</th>';
    html += '<th>Success%</th><th>Signal Date</th><th>Days Ago</th>';
    html += '<th>Signal Ret%</th><th>Latest Ret%</th><th>Hold Days</th><th>Advice</th>';
    html += '</tr></thead><tbody>';

    data.slice(0, 100).forEach(o => {
        html += '<tr>';
        html += `<td>${o['银行']}</td>`;
        html += `<td style="font-family:var(--font-mono)">${o['产品代码']}</td>`;
        html += `<td>${(o['产品名称'] || '').substring(0, 20)}</td>`;
        html += `<td>${o['来源'] === '观察池' ? tagCell('Watch', 'watch') : tagCell('Lib', 'hold')}</td>`;
        html += numCell(o['历史成功率%'], 1);
        html += `<td class="td-num">${o['信号日期']}</td>`;
        html += `<td class="td-num">${o['信号距今天数']}</td>`;
        html += numCell(o['信号收益率%'], 2);
        html += numCell(o['最新收益率%'], 2);
        html += `<td class="td-num">${o['预期持有天数']}</td>`;
        html += `<td>${adviceTag(o['操作建议'])}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('oppTableWrap', html);
}

// ── Load Portfolio ──

async function loadPortfolio() {
    const data = await fetchJSON('/api/strategy/portfolio');
    if (!data || data.length === 0) return;

    let html = '<table><thead><tr>';
    html += '<th>Bank</th><th>Code</th><th>Name</th><th>Status</th>';
    html += '<th>Net Position</th><th>Days Held</th><th>Latest Ret%</th>';
    html += '<th>Avg Ret%</th><th>In Lib</th><th>Advice</th>';
    html += '</tr></thead><tbody>';

    data.forEach(p => {
        const statusTag = p['持仓状态'] === '持有中' ? tagCell('Active', 'buy') : tagCell('Closed', 'info');
        html += '<tr>';
        html += `<td>${p['银行']}</td>`;
        html += `<td style="font-family:var(--font-mono)">${p['产品代码']}</td>`;
        html += `<td>${(p['产品名称'] || '').substring(0, 20)}</td>`;
        html += `<td>${statusTag}</td>`;
        html += `<td class="td-num">${fmt(p['净持仓金额'], 0)}</td>`;
        html += `<td class="td-num">${p['持仓天数'] || '--'}</td>`;
        html += numCell(p['最新年化收益%'], 2);
        html += numCell(p['买入以来平均收益%'], 2);
        html += `<td>${p['是否在高成功率库'] === '是' ? tagCell('Yes', 'buy') : tagCell('No', 'info')}</td>`;
        html += `<td>${adviceTag(p['持仓建议'])}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('portTableWrap', html);
}

// ── Load Patterns ──

async function loadPatterns() {
    const data = await fetchJSON('/api/strategy/patterns');
    setText('patCount', data.length || 0);
    if (!data || data.length === 0) return;

    // Confidence chart
    const top20 = data.slice(0, 20);
    const chartConf = getChart('chartPatternConf');
    if (chartConf) {
        chartConf.setOption({
            ...CHART_THEME,
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: top20.map(p => p.code.substring(0, 10)),
                axisLabel: { rotate: 45, fontSize: 10, color: '#8b95a8' },
            },
            yAxis: { type: 'value', max: 1, axisLabel: { color: '#8b95a8' } },
            series: [{
                type: 'bar',
                data: top20.map(p => p.confidence),
                itemStyle: {
                    color: function(params) {
                        const v = params.value;
                        if (v >= 0.7) return '#10b981';
                        if (v >= 0.4) return '#f59e0b';
                        return '#ef4444';
                    },
                    borderRadius: [3, 3, 0, 0],
                },
                barWidth: '60%',
            }],
        });
    }

    // Period distribution
    const periods = data.filter(p => p.has_period).map(p => p.period_days);
    if (periods.length > 0) {
        const bins = { '5-15d': 0, '15-30d': 0, '30-60d': 0, '60-90d': 0, '90+d': 0 };
        periods.forEach(d => {
            if (d < 15) bins['5-15d']++;
            else if (d < 30) bins['15-30d']++;
            else if (d < 60) bins['30-60d']++;
            else if (d < 90) bins['60-90d']++;
            else bins['90+d']++;
        });
        const chartPer = getChart('chartPatternPeriod');
        if (chartPer) {
            chartPer.setOption({
                ...CHART_THEME,
                tooltip: { trigger: 'item' },
                series: [{
                    type: 'pie',
                    radius: ['35%', '65%'],
                    label: { color: '#8b95a8', fontSize: 11 },
                    data: Object.entries(bins).filter(([,v]) => v > 0).map(([k, v]) => ({ name: k, value: v })),
                }],
            });
        }
    }

    // Table
    let html = '<table><thead><tr>';
    html += '<th>Bank</th><th>Code</th><th>Period</th><th>CV</th>';
    html += '<th>Top Phase</th><th>Top Weekday</th><th>Events</th>';
    html += '<th>Confidence</th><th>Last Release</th><th>Prediction</th>';
    html += '</tr></thead><tbody>';

    data.forEach(p => {
        const confColor = p.confidence >= 0.7 ? 'td-pos' : (p.confidence >= 0.4 ? 'td-warn' : 'td-neg');
        const pred = p.prediction
            ? tagCell(`${p.prediction.predicted_date} (${fmt(p.prediction.confidence, 2)})`, 'info')
            : '';
        html += '<tr>';
        html += `<td>${p.bank}</td>`;
        html += `<td style="font-family:var(--font-mono)">${p.code}</td>`;
        html += `<td class="td-num">${p.has_period ? fmt(p.period_days, 0) + 'd' : '--'}</td>`;
        html += `<td class="td-num">${fmt(p.period_cv, 3)}</td>`;
        html += `<td>${p.top_phase || '--'}</td>`;
        html += `<td>${p.top_weekday || '--'}</td>`;
        html += `<td class="td-num">${p.n_events}</td>`;
        html += `<td class="td-num ${confColor}">${fmt(p.confidence, 3)}</td>`;
        html += `<td class="td-num">${p.last_release}</td>`;
        html += `<td>${pred}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('patTableWrap', html);
}

// ── Run Strategy ──

function runStrategy(forceRefresh) {
    const btn = $('#btnRunStrategy');
    const btn2 = $('#btnRefreshStrategy');
    btn.disabled = true;
    btn2.disabled = true;
    $('#strategyProgressWrap').style.display = '';
    setText('strategyProgressText', 'Starting...');
    updateGlobalStatus('running');

    fetch('/api/strategy/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force_refresh: forceRefresh }),
    }).then(() => {
        const es = new EventSource('/api/strategy/status');
        es.onmessage = function(e) {
            const d = JSON.parse(e.data);
            $('#strategyProgressBar').style.width = d.progress + '%';
            setText('strategyProgressText', d.message || '');
            if (d.status === 'done') {
                es.close();
                btn.disabled = false;
                btn2.disabled = false;
                updateGlobalStatus('done');
                refreshAll();
            } else if (d.status === 'error') {
                es.close();
                btn.disabled = false;
                btn2.disabled = false;
                updateGlobalStatus('error');
                setText('strategyProgressText', 'Error: ' + (d.error || 'unknown'));
            }
        };
        es.onerror = function() {
            es.close();
            btn.disabled = false;
            btn2.disabled = false;
            updateGlobalStatus('error');
        };
    });
}

// ── Run Backtest ──

function runBacktest() {
    const btn = $('#btnRunBacktest');
    btn.disabled = true;
    $('#btProgressWrap').style.display = '';
    setText('btProgressText', 'Starting...');

    fetch('/api/backtest/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
    }).then(() => {
        const es = new EventSource('/api/backtest/status');
        es.onmessage = function(e) {
            const d = JSON.parse(e.data);
            $('#btProgressBar').style.width = d.progress + '%';
            setText('btProgressText', d.message || '');
            if (d.status === 'done') {
                es.close();
                btn.disabled = false;
                loadBacktestResults();
            } else if (d.status === 'error') {
                es.close();
                btn.disabled = false;
                setText('btProgressText', 'Error: ' + (d.error || 'unknown'));
            }
        };
        es.onerror = function() {
            es.close();
            btn.disabled = false;
        };
    });
}

// ── Load Backtest Results ──

async function loadBacktestResults() {
    const results = await fetchJSON('/api/backtest/results');
    if (!results || !results.total_trades) return;

    // Show sections
    $('#btKpiRow').style.display = '';
    $('#btChartRow').style.display = '';
    $('#btDetailRow').style.display = '';
    $('#btBankRow').style.display = '';

    // KPIs
    const retColor = results.ann_return >= 0 ? 'td-pos' : 'td-neg';
    setText('btAnnRet', fmtPct(results.ann_return));
    document.getElementById('btAnnRet').className = 'kpi-value ' + retColor;
    setText('btTotalRet', fmtPct(results.total_return));
    setText('btSharpe', fmt(results.sharpe, 2));
    setText('btPF', fmt(results.profit_factor, 2));
    setText('btMaxDD', fmtPct(results.max_drawdown));
    setText('btDDDate', results.max_dd_date || '--');
    setText('btWinRate', fmtPct(results.win_rate));
    setText('btTrades', results.total_trades);
    setText('btWins', results.wins);
    setText('btLosses', results.losses);

    // NAV Chart
    const nav = await fetchJSON('/api/backtest/nav');
    if (nav && nav.length > 0) {
        const chartNav = getChart('chartNav');
        if (chartNav) {
            chartNav.setOption({
                ...CHART_THEME,
                tooltip: {
                    trigger: 'axis',
                    formatter: function(p) {
                        const d = p[0];
                        return `${d.axisValue}<br/>NAV: ${d.value[1]}万<br/>Cash: ${d.value[2]}万`;
                    },
                },
                xAxis: {
                    type: 'category',
                    data: nav.map(n => n.date),
                    axisLabel: { color: '#8b95a8', fontSize: 10 },
                },
                yAxis: {
                    type: 'value',
                    name: 'NAV (万)',
                    nameTextStyle: { color: '#8b95a8' },
                    axisLabel: { color: '#8b95a8' },
                    scale: true,
                },
                dataZoom: [{ type: 'inside', start: 0, end: 100 }],
                series: [
                    {
                        name: 'NAV',
                        type: 'line',
                        data: nav.map(n => [n.date, n.value, n.cash]),
                        showSymbol: false,
                        lineStyle: { width: 2, color: '#3b82f6' },
                        areaStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgba(59,130,246,0.3)' },
                                { offset: 1, color: 'rgba(59,130,246,0.02)' },
                            ]),
                        },
                    },
                ],
            });
        }
    }

    // Monthly returns
    const monthly = await fetchJSON('/api/backtest/trades');
    const monthlyRets = [];
    // Aggregate from trades is complex; use nav for monthly calc
    const monthlyData = {};
    if (nav && nav.length > 0) {
        nav.forEach(n => {
            const m = n.date.substring(0, 7);
            monthlyData[m] = n.value;
        });
        const months = Object.keys(monthlyData).sort();
        let prev = nav[0].value;
        months.forEach(m => {
            const cur = monthlyData[m];
            monthlyRets.push({ month: m, return_pct: ((cur / prev) - 1) * 100 });
            prev = cur;
        });
    }
    if (monthlyRets.length > 0) {
        const chartMon = getChart('chartMonthly');
        if (chartMon) {
            chartMon.setOption({
                ...CHART_THEME,
                tooltip: { trigger: 'axis', formatter: '{b}: {c}%' },
                xAxis: {
                    type: 'category',
                    data: monthlyRets.map(m => m.month),
                    axisLabel: { color: '#8b95a8', fontSize: 10 },
                },
                yAxis: { type: 'value', axisLabel: { color: '#8b95a8' } },
                series: [{
                    type: 'bar',
                    data: monthlyRets.map(m => ({
                        value: Number(m.return_pct.toFixed(4)),
                        itemStyle: { color: m.return_pct >= 0 ? '#10b981' : '#ef4444' },
                    })),
                    barWidth: '50%',
                }],
            });
        }
    }

    // Trade log
    const trades = await fetchJSON('/api/backtest/trades');
    setText('btTradeCount', trades.length || 0);
    if (trades && trades.length > 0) {
        let html = '<table><thead><tr>';
        html += '<th>Date</th><th>Action</th><th>Bank</th><th>Code</th>';
        html += '<th>NAV</th><th>Amount(万)</th><th>PnL(万)</th><th>Hold</th><th>Reason</th>';
        html += '</tr></thead><tbody>';
        trades.slice(-200).forEach(t => {
            const pnlCls = t.pnl > 0 ? 'td-pos' : (t.pnl < 0 ? 'td-neg' : '');
            html += '<tr>';
            html += `<td class="td-num">${t.date}</td>`;
            html += `<td>${t.action}</td>`;
            html += `<td>${t.bank}</td>`;
            html += `<td style="font-family:var(--font-mono)">${t.code}</td>`;
            html += `<td class="td-num">${fmt(t.nav, 6)}</td>`;
            html += `<td class="td-num">${fmt(t.amount, 2)}</td>`;
            html += `<td class="td-num ${pnlCls}">${fmt(t.pnl, 2)}</td>`;
            html += `<td class="td-num">${t.hold_days || ''}</td>`;
            html += `<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${(t.reason || '').substring(0, 40)}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table>';
        setHTML('btTradeTableWrap', html);
    }

    // Bank performance chart
    const bankStats = results.bank_stats || {};
    const bankNames = Object.keys(bankStats);
    if (bankNames.length > 0) {
        const chartBP = getChart('chartBankPerf');
        if (chartBP) {
            chartBP.setOption({
                ...CHART_THEME,
                tooltip: { trigger: 'axis' },
                legend: { data: ['PnL(万)', 'Win Rate(%)'], textStyle: { color: '#8b95a8' } },
                xAxis: {
                    type: 'category',
                    data: bankNames,
                    axisLabel: { color: '#8b95a8', fontSize: 10, rotate: 30 },
                },
                yAxis: [
                    { type: 'value', name: 'PnL(万)', axisLabel: { color: '#8b95a8' } },
                    { type: 'value', name: 'Win%', max: 100, axisLabel: { color: '#8b95a8' } },
                ],
                series: [
                    {
                        name: 'PnL(万)',
                        type: 'bar',
                        data: bankNames.map(b => ({
                            value: bankStats[b].pnl,
                            itemStyle: { color: bankStats[b].pnl >= 0 ? '#10b981' : '#ef4444' },
                        })),
                        barWidth: '40%',
                    },
                    {
                        name: 'Win Rate(%)',
                        type: 'line',
                        yAxisIndex: 1,
                        data: bankNames.map(b => bankStats[b].win_rate),
                        lineStyle: { color: '#f59e0b', width: 2 },
                        itemStyle: { color: '#f59e0b' },
                    },
                ],
            });
        }
    }
}

// ── Global Status ──

function updateGlobalStatus(status) {
    const dot = $('#globalStatus');
    dot.className = 'status-dot status-' + status;
}

// ── Refresh All ──

async function refreshAll() {
    await loadOverview();
    await loadRecommendations();
    await loadOpportunities();
    await loadPortfolio();
    await loadPatterns();
}

// ── Init ──

document.addEventListener('DOMContentLoaded', async () => {
    // Try loading existing data
    try {
        await refreshAll();
        // Check backtest
        const btResults = await fetchJSON('/api/backtest/results');
        if (btResults && btResults.total_trades) {
            await loadBacktestResults();
        }
    } catch (e) {
        console.log('Initial load — no data yet');
    }
});
