/* ═══════════════════════════════════════════════════════════
   千将资本 Dashboard V2 — 全新前端逻辑
   Apple + McKinsey 投行级交互
   ═══════════════════════════════════════════════════════════ */

// ── Globals ──
const chartInstances = {};
let currentTab = 'overview';

// 缓存的表格原始数据（用于筛选/排序）
const tableData = {
    rec: [],    // 推荐
    sig: [],    // 信号
    port: [],   // 持仓
    pat: [],    // 规律
    trade: [],  // 交易记录
    btTrade: [],// 回测交易
};

// OCR 缓存
let ocrTrades = [];

// ══════════════════════════════════════════
// 工具函数
// ══════════════════════════════════════════

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }
function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }
function setHTML(id, v) { const el = document.getElementById(id); if (el) el.innerHTML = v; }
function show(id) { const el = document.getElementById(id); if (el) el.style.display = ''; }
function hide(id) { const el = document.getElementById(id); if (el) el.style.display = 'none'; }

function fmt(n, d) {
    if (n === null || n === undefined || n === '' || isNaN(n)) return '--';
    return Number(n).toFixed(d === undefined ? 2 : d);
}

function fmtPct(n) {
    if (n === null || n === undefined || n === '' || isNaN(n)) return '--';
    return Number(n).toFixed(2) + '%';
}

function fmtMoney(n) {
    if (n === null || n === undefined || isNaN(n)) return '--';
    n = Number(n);
    if (Math.abs(n) >= 1e8) return (n / 1e8).toFixed(2) + '亿';
    if (Math.abs(n) >= 1e4) return (n / 1e4).toFixed(2) + '万';
    return n.toFixed(2);
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
    const cls = {
        buy: 'tag-buy', sell: 'tag-sell', hold: 'tag-hold',
        watch: 'tag-watch', info: 'tag-info',
    };
    return `<span class="td-tag ${cls[type] || cls.info}">${text}</span>`;
}

function adviceTag(advice) {
    if (!advice) return '';
    if (advice.startsWith('买入')) return tagCell(advice, 'buy');
    if (advice.includes('★★★')) return tagCell(advice, 'buy');
    if (advice.includes('★★')) return tagCell(advice, 'buy');
    if (advice.includes('★')) return tagCell(advice, 'buy');
    if (advice.includes('可买入') || advice.includes('加仓')) return tagCell(advice, 'buy');
    if (advice.startsWith('卖出')) return tagCell(advice, 'sell');
    if (advice.includes('卖出') || advice.includes('赎回')) return tagCell(advice, 'sell');
    if (advice.startsWith('持有')) return tagCell(advice, 'hold');
    if (advice.includes('持有') || advice.includes('智持') || advice.includes('继续')) return tagCell(advice, 'hold');
    if (advice.startsWith('观望')) return tagCell(advice, 'watch');
    if (advice.includes('☆') || advice.includes('观察')) return tagCell(advice, 'watch');
    return tagCell(advice, 'info');
}

function statusTag(status) {
    if (status === '持有中') return tagCell('持有中', 'hold');
    if (status === '已清仓') return tagCell('已清仓', 'info');
    return tagCell(status || '--', 'info');
}

function tradeTypeTag(type) {
    if (type === '买入') return tagCell('买入', 'buy');
    if (type === '卖出') return tagCell('卖出', 'sell');
    return tagCell(type || '--', 'info');
}

// ══════════════════════════════════════════
// Toast 通知
// ══════════════════════════════════════════

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ══════════════════════════════════════════
// 数字动画
// ══════════════════════════════════════════

function animateNumber(elementId, target, decimals = 0, suffix = '') {
    const el = document.getElementById(elementId);
    if (!el) return;

    const start = 0;
    const duration = 600;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = start + (target - start) * eased;

        el.textContent = current.toFixed(decimals) + suffix;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ══════════════════════════════════════════
// 确认对话框
// ══════════════════════════════════════════

let modalCallback = null;

function openModal(title, message, callback) {
    setText('modalTitle', title);
    setText('modalMessage', message);
    modalCallback = callback;
    document.getElementById('modalOverlay').classList.add('open');
}

function closeModal() {
    document.getElementById('modalOverlay').classList.remove('open');
    modalCallback = null;
}

function confirmModal() {
    if (modalCallback) modalCallback();
    closeModal();
}

// ══════════════════════════════════════════
// 时钟
// ══════════════════════════════════════════

function updateClock() {
    setText('clock', new Date().toLocaleString('zh-CN', { hour12: false }));
}
setInterval(updateClock, 1000);
updateClock();

// ══════════════════════════════════════════
// 标签导航
// ══════════════════════════════════════════

$$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.tab-btn').forEach(b => b.classList.remove('active'));
        $$('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.dataset.tab;
        const panel = document.getElementById('tab-' + tab);
        if (panel) panel.classList.add('active');
        currentTab = tab;
        setTimeout(resizeCharts, 50);

        // 切换到数据管理时加载统计
        if (tab === 'datamanage') loadDbStats();
        // 切换到持仓时加载
        if (tab === 'portfolio') { loadTradeHistory(); loadHoldingReturns(); show('mcKpiRow'); }
        // 切换到释放规律时加载 AI 预测
        if (tab === 'patterns') loadAiPredictions();
    });
});

// ══════════════════════════════════════════
// ECharts 浅色 Apple 主题
// ══════════════════════════════════════════

function getChart(id) {
    if (!chartInstances[id]) {
        const el = document.getElementById(id);
        if (!el) return null;
        chartInstances[id] = echarts.init(el);
    }
    return chartInstances[id];
}

function resizeCharts() {
    Object.values(chartInstances).forEach(c => { if (c) try { c.resize(); } catch(e) {} });
}
window.addEventListener('resize', resizeCharts);

const APPLE_COLORS = ['#2196f3', '#26a69a', '#ff9800', '#ef5350', '#ab47bc', '#26c6da', '#7e57c2', '#ec407a'];

const CHART_BASE = {
    backgroundColor: 'transparent',
    textStyle: { fontFamily: "'Microsoft YaHei', Consolas, sans-serif", color: '#b2b5be', fontSize: 13 },
    grid: { top: 40, right: 24, bottom: 36, left: 56, containLabel: true },
};

// V10.1: Eye-comfort axis defaults
const DARK_AXIS_LABEL = { color: '#787b86', fontSize: 12 };
const DARK_AXIS_LINE = { lineStyle: { color: '#2a2e39' } };
const DARK_SPLIT_LINE = { lineStyle: { color: '#22262f' } };

// ══════════════════════════════════════════
// API 请求
// ══════════════════════════════════════════

async function fetchJSON(url) {
    try {
        const res = await fetch(url);
        return res.json();
    } catch (e) {
        console.error('Fetch error:', url, e);
        return null;
    }
}

async function postJSON(url, data) {
    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return res.json();
    } catch (e) {
        console.error('Post error:', url, e);
        return null;
    }
}

// ══════════════════════════════════════════
// 全局状态
// ══════════════════════════════════════════

function updateGlobalStatus(status) {
    const dot = document.getElementById('globalStatus');
    if (dot) dot.className = 'status-dot status-' + status;
}

// ══════════════════════════════════════════
// 通用筛选/排序/导出系统
// ══════════════════════════════════════════

function filterTable(prefix) {
    // 获取该表格的筛选控件值
    const searchEl = document.getElementById(prefix + 'Search');
    const searchText = searchEl ? searchEl.value.toLowerCase() : '';

    let data = tableData[prefix] || [];
    let filtered = data;

    // 全文搜索
    if (searchText) {
        filtered = filtered.filter(row => {
            return Object.values(row).some(v =>
                String(v || '').toLowerCase().includes(searchText)
            );
        });
    }

    // 特定筛选器
    if (prefix === 'rec') {
        const bank = document.getElementById('recBankFilter')?.value;
        const liq = document.getElementById('recLiqFilter')?.value;
        if (bank) filtered = filtered.filter(r => r['银行'] === bank);
        if (liq) filtered = filtered.filter(r => r['流动性'] === liq);
    } else if (prefix === 'sig') {
        const bank = document.getElementById('sigBankFilter')?.value;
        const source = document.getElementById('sigSourceFilter')?.value;
        if (bank) filtered = filtered.filter(r => r['银行'] === bank);
        if (source) filtered = filtered.filter(r => r['来源'] === source);
    } else if (prefix === 'port') {
        const status = document.getElementById('portStatusFilter')?.value;
        if (status) filtered = filtered.filter(r => r.status === status);
    } else if (prefix === 'pat') {
        const bank = document.getElementById('patBankFilter')?.value;
        const conf = document.getElementById('patConfFilter')?.value;
        const period = document.getElementById('patPeriodFilter')?.value;
        if (bank) filtered = filtered.filter(r => r.bank === bank);
        if (conf === 'high') filtered = filtered.filter(r => r.confidence >= 0.7);
        else if (conf === 'medium') filtered = filtered.filter(r => r.confidence >= 0.4 && r.confidence < 0.7);
        else if (conf === 'low') filtered = filtered.filter(r => r.confidence < 0.4);
        if (period === 'yes') filtered = filtered.filter(r => r.has_period);
        else if (period === 'no') filtered = filtered.filter(r => !r.has_period);
    } else if (prefix === 'trade') {
        const type = document.getElementById('tradeTypeFilter')?.value;
        if (type) filtered = filtered.filter(r => r['交易'] === type);
    }

    // 更新筛选计数
    const countEl = document.getElementById(prefix + 'FilterCount');
    if (countEl) {
        countEl.textContent = `筛选 ${filtered.length} / 共 ${data.length} 条`;
    }

    // 重新渲染表格
    renderFilteredTable(prefix, filtered);
}

function sortTable(prefix) {
    const sortEl = document.getElementById(prefix + 'SortBy');
    if (!sortEl) return;
    const sortVal = sortEl.value;
    let data = tableData[prefix] || [];

    if (prefix === 'rec') {
        if (sortVal === 'score_desc') data.sort((a, b) => (b['综合得分'] || 0) - (a['综合得分'] || 0));
        else if (sortVal === 'return_desc') data.sort((a, b) => (b['最新收益率%'] || 0) - (a['最新收益率%'] || 0));
        else if (sortVal === 'success_desc') data.sort((a, b) => (b['历史成功率%'] || 0) - (a['历史成功率%'] || 0));
        else if (sortVal === 'sharpe_desc') data.sort((a, b) => (b['夏普比率'] || 0) - (a['夏普比率'] || 0));
    } else if (prefix === 'sig') {
        if (sortVal === 'return_desc') data.sort((a, b) => (b['最新收益率%'] || 0) - (a['最新收益率%'] || 0));
        else if (sortVal === 'annret_desc') data.sort((a, b) => (b['年化收益率%'] || 0) - (a['年化收益率%'] || 0));
        else if (sortVal === 'success_desc') data.sort((a, b) => (b['历史成功率%'] || 0) - (a['历史成功率%'] || 0));
    }

    tableData[prefix] = data;
    filterTable(prefix); // 重新应用筛选
}

function renderFilteredTable(prefix, data) {
    if (prefix === 'rec') renderRecTable(data);
    else if (prefix === 'sig') renderSigTable(data);
    else if (prefix === 'port') renderPortTable(data);
    else if (prefix === 'pat') renderPatTable(data);
    else if (prefix === 'trade') renderTradeTable(data);
    else if (prefix === 'btTrade') renderBtTradeTable(data);
}

function exportTable(prefix) {
    const data = tableData[prefix] || [];
    if (data.length === 0) {
        showToast('没有数据可导出', 'warning');
        return;
    }
    // CSV 导出
    const headers = Object.keys(data[0]);
    let csv = '\uFEFF'; // BOM for Excel
    csv += headers.join(',') + '\n';
    data.forEach(row => {
        csv += headers.map(h => {
            let v = row[h];
            if (v === null || v === undefined) v = '';
            v = String(v).replace(/"/g, '""');
            return `"${v}"`;
        }).join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `千将资本_${prefix}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('导出成功', 'success');
}

// 动态填充筛选器选项
function populateFilterOptions(prefix, field, selectId) {
    const data = tableData[prefix] || [];
    const el = document.getElementById(selectId);
    if (!el) return;

    const values = [...new Set(data.map(r => r[field]).filter(v => v))];
    values.sort();

    // 保留第一个默认选项
    const defaultOpt = el.options[0];
    el.innerHTML = '';
    el.appendChild(defaultOpt);

    values.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        el.appendChild(opt);
    });
}

// ══════════════════════════════════════════
// 加载总览
// ══════════════════════════════════════════

// V10.1: Macro Dashboard
function updateMacroDashboard(listB) {
    if (!listB || listB.length === 0) return;
    // Avg Yield
    const yields = listB.map(r => r['最新收益率%'] || r['预期年化收益%'] || 0).filter(v => v !== 0);
    const avgYield = yields.length > 0 ? (yields.reduce((a, b) => a + b, 0) / yields.length) : 0;
    const avgEl = document.getElementById('macroAvgYield');
    if (avgEl) {
        avgEl.textContent = avgYield.toFixed(2) + '%';
        avgEl.style.color = avgYield >= 3 ? '#26a69a' : '#ef5350';
    }
    // Pool Depth (SOPM)
    const sopmScores = listB.map(r => r['sopm_score'] || 0).filter(v => v > 0);
    const avgSopm = sopmScores.length > 0 ? (sopmScores.reduce((a, b) => a + b, 0) / sopmScores.length) : 0;
    const depthEl = document.getElementById('macroPoolDepth');
    if (depthEl) {
        depthEl.textContent = avgSopm >= 90 ? '充裕' : avgSopm >= 60 ? '适中' : '紧张';
        depthEl.style.color = avgSopm >= 90 ? '#26a69a' : avgSopm >= 60 ? '#ff9800' : '#ef5350';
    }
    // Product count
    const countEl = document.getElementById('macroProductCount');
    if (countEl) { countEl.textContent = listB.length; countEl.style.color = '#e0e3eb'; }
    // Momentum
    const velocities = listB.map(r => r['yield_velocity'] || r['收益加速度'] || 0);
    const avgVel = velocities.length > 0 ? (velocities.reduce((a, b) => a + b, 0) / velocities.length) : 0;
    const momEl = document.getElementById('macroMomentum');
    if (momEl) {
        momEl.textContent = avgVel > 0.1 ? '加速' : avgVel < -0.1 ? '减速' : '平稳';
        momEl.style.color = avgVel > 0.1 ? '#26a69a' : avgVel < -0.1 ? '#ef5350' : '#787b86';
    }
    // Signal quality
    const sigEl = document.getElementById('macroSigQuality');
    if (sigEl) {
        const highConf = listB.filter(r => (r['综合得分'] || 0) >= 40).length;
        const ratio = listB.length > 0 ? (highConf / listB.length * 100) : 0;
        sigEl.textContent = ratio.toFixed(0) + '% 优质';
        sigEl.style.color = ratio >= 50 ? '#26a69a' : '#ff9800';
    }
}

// V9: Market Ticker
function updateTicker(summaryData, oppsData) {
    const track = document.getElementById('tickerTrack');
    if (!track) return;

    const items = [];
    if (summaryData) {
        const avgRet = (summaryData.avg_return || 0).toFixed(2);
        items.push(`<span class="ticker-item"><span class="ticker-label">市场均值</span> <span class="ticker-val">${avgRet}%</span></span>`);
        items.push(`<span class="ticker-item"><span class="ticker-label">产品库</span> <span class="ticker-val">${summaryData.product_lib_count || 0}</span></span>`);
        items.push(`<span class="ticker-item"><span class="ticker-label">信号数</span> <span class="ticker-val">${summaryData.opportunity_count || 0}</span></span>`);
        items.push(`<span class="ticker-item"><span class="ticker-label">买入</span> <span class="ticker-val">${summaryData.buy_signal_count || 0}</span></span>`);
    }
    if (oppsData && oppsData.length > 0) {
        // Top gainer
        const sorted = [...oppsData].sort((a, b) => (b['最新收益率%'] || 0) - (a['最新收益率%'] || 0));
        const top = sorted[0];
        if (top) {
            items.push(`<span class="ticker-item"><span class="ticker-label">最高</span> <span class="ticker-val">${(top['产品名称'] || '').substring(0,12)} ${(top['最新收益率%'] || 0).toFixed(2)}%</span></span>`);
        }
        // Bottom
        const bot = sorted[sorted.length - 1];
        if (bot && sorted.length > 1) {
            items.push(`<span class="ticker-item"><span class="ticker-label">最低</span> <span class="ticker-neg">${(bot['产品名称'] || '').substring(0,12)} ${(bot['最新收益率%'] || 0).toFixed(2)}%</span></span>`);
        }
    }
    items.push(`<span class="ticker-item"><span class="ticker-label">时间</span> <span class="ticker-val">${new Date().toLocaleTimeString('zh-CN', {hour12:false})}</span></span>`);

    // Duplicate for seamless scroll
    const content = items.join('<span class="ticker-sep">|</span>');
    track.innerHTML = content + '<span class="ticker-sep">|</span>' + content;
}

async function loadOverview() {
    const data = await fetchJSON('/api/strategy/summary');
    if (!data) return;

    // KPI 动画
    if (data.product_lib_count !== undefined) {
        animateNumber('kpiLibCount', data.product_lib_count, 0);
        setText('kpiLibBadge', '库');
        setText('kpiWatchCount', data.watch_pool_count || 0);
        animateNumber('kpiOppCount', data.opportunity_count || 0, 0);
        setText('kpiBuyCount', data.buy_signal_count || 0);
        setText('kpiBuySigBadge', (data.buy_signal_count || 0) + ' 买入');

        const avgRet = data.avg_return || 0;
        animateNumber('kpiAvgRet', avgRet, 2, '%');

        animateNumber('kpiHoldCount', data.holding_count || 0, 0);
        setText('kpiHoldAmt', fmtMoney(data.total_holding_amount || 0));
    }

    // 数据时效信息
    const freshnessEl = document.getElementById('dataFreshness');
    if (freshnessEl && (data.last_run || data.nav_data_date)) {
        freshnessEl.style.display = 'flex';
        if (data.nav_data_date) {
            setText('navDataDate', data.nav_data_date);
        }
        if (data.last_run) {
            // 将 ISO 时间转换为易读格式
            const d = new Date(data.last_run);
            const timeStr = d.getFullYear() + '-' +
                String(d.getMonth() + 1).padStart(2, '0') + '-' +
                String(d.getDate()).padStart(2, '0') + ' ' +
                String(d.getHours()).padStart(2, '0') + ':' +
                String(d.getMinutes()).padStart(2, '0');
            setText('strategyRunTime', timeStr);
        }
        // 预测来源
        const predSource = data.prediction_source || 'chi_square';
        const predText = document.getElementById('predSourceText');
        if (predText) {
            predText.textContent = predSource === 'deep_learning' ? 'AI预测' : '统计分析';
            predText.style.color = predSource === 'deep_learning' ? '#26a69a' : '#787b86';
        }
    }

    // 银行分布环形图
    const bankDist = data.bank_distribution || {};
    const banks = Object.keys(bankDist);
    const bankVals = Object.values(bankDist);
    const chartBank = getChart('chartBankDist');
    if (chartBank && banks.length > 0) {
        chartBank.setOption({
            ...CHART_BASE,
            color: APPLE_COLORS,
            tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
            series: [{
                type: 'pie',
                radius: ['42%', '72%'],
                center: ['50%', '55%'],
                label: { color: '#b2b5be', fontSize: 13 },
                data: banks.map((b, i) => ({ name: b, value: bankVals[i] })),
                itemStyle: { borderRadius: 4, borderColor: '#131722', borderWidth: 2 },
                emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.15)' } },
            }],
        });
    }

    // 信号强度柱状图 + ticker
    const opps = await fetchJSON('/api/strategy/opportunities');
    updateTicker(data, opps);
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
            const colors = ['#2196f3', '#7e57c2', '#26c6da', '#26a69a', '#787b86'];
            chartSig.setOption({
                ...CHART_BASE,
                tooltip: { trigger: 'axis' },
                xAxis: { type: 'category', data: Object.keys(buckets), axisLabel: { color: '#787b86' }, axisLine: { lineStyle: { color: '#2a2e39' } } },
                yAxis: { type: 'value', axisLabel: { color: '#787b86' }, splitLine: { lineStyle: { color: '#22262f' } } },
                series: [{
                    type: 'bar',
                    data: Object.values(buckets).map((v, i) => ({ value: v, itemStyle: { color: colors[i] } })),
                    barWidth: '50%',
                    itemStyle: { borderRadius: [4, 4, 0, 0] },
                }],
            });
        }
    }
}

// ══════════════════════════════════════════
// VIP Sniper: 加载推荐 (双通道)
// ══════════════════════════════════════════

// 缓存 VIP 通道数据
let vipChannelData = null;
let listBFiltered = [];

async function loadRecommendations() {
    // 并行请求: 原始 top20 + VIP 通道
    const [data, channels] = await Promise.all([
        fetchJSON('/api/strategy/top20'),
        fetchJSON('/api/strategy/vip-channels'),
    ]);

    // 原始推荐表 (折叠区域)
    if (data) {
        // 只展示 Top 20 in the collapsed section
        const top20 = data.slice(0, 20);
        setText('recCount', top20.length || 0);
        tableData.rec = top20;
        populateFilterOptions('rec', '银行', 'recBankFilter');
        populateFilterOptions('rec', '流动性', 'recLiqFilter');
        renderRecTable(top20);
    }

    // VIP 通道
    if (channels && channels.list_a) {
        vipChannelData = channels;
        renderListATable(channels.list_a);
        renderListBTable(channels.list_b || []);
        listBFiltered = channels.list_b || [];
        setText('listACount', channels.list_a.length || 0);
        setText('listBCount', (channels.list_b || []).length || 0);

        // V10.1: Macro Dashboard — Market Vitals
        updateMacroDashboard(channels.list_b || []);

        // 填充 List B 银行筛选
        const banks = [...new Set((channels.list_b || []).map(r => r['银行']))];
        const sel = document.getElementById('listBBankFilter');
        if (sel) {
            sel.innerHTML = '<option value="">全部银行</option>';
            banks.forEach(b => { sel.innerHTML += `<option value="${b}">${b}</option>`; });
        }

        // 加载售罄列表
        loadSoldOutList();
    }
}

function renderListATable(data) {
    if (!data || data.length === 0) {
        setHTML('listATableWrap', '<div class="empty-state"><p>暂无私行预约产品</p></div>');
        return;
    }
    let html = '<table><thead><tr>';
    html += '<th>代码</th><th>名称</th><th>银行</th><th>类型</th>';
    html += '<th>收益率 / 趋势</th><th>综合得分</th><th>密度30</th><th>操作</th>';
    html += '</tr></thead><tbody>';

    data.forEach(r => {
        const typeTag = r['常热产品']
            ? '<span class="td-tag tag-hot">常热</span>'
            : '<span class="td-tag tag-vip">VIP</span>';
        const yieldVal = r['最新收益率%'];
        const sparkline = renderSparklineSVG(r['history_yields'] || [], 56, 18);
        const yieldColor = yieldVal >= 3 ? '#26a69a' : yieldVal >= 0 ? '#e0e3eb' : '#ef5350';
        html += '<tr>';
        html += `<td class="vip-code">${r['产品代码'] || ''}</td>`;
        html += `<td>${(r['产品名称'] || '').substring(0, 24)}</td>`;
        html += `<td>${r['银行'] || ''}</td>`;
        html += `<td>${typeTag}</td>`;
        html += `<td><span class="sparkline-cell"><span style="color:${yieldColor};font-weight:600">${yieldVal != null ? yieldVal.toFixed(2) + '%' : '--'}</span>${sparkline}</span></td>`;
        html += numCell(r['综合得分'], 1);
        html += `<td class="td-num">${r['高收益密度30'] ? (r['高收益密度30'] * 100).toFixed(0) + '%' : '--'}</td>`;
        html += `<td><button class="btn-copy-rm" onclick="copyRmScript('${esc(r['银行'])}','${esc(r['产品名称'])}','${esc(r['产品代码'])}')">复制话术</button></td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('listATableWrap', html);
}

// V9: SVG Sparkline generator
function renderSparklineSVG(yields, width, height) {
    if (!yields || yields.length < 2) return '';
    const w = width || 64;
    const h = height || 20;
    const n = yields.length;
    const min = Math.min(...yields);
    const max = Math.max(...yields);
    const range = max - min || 1;
    const points = yields.map((v, i) => {
        const x = (i / (n - 1)) * w;
        const y = h - ((v - min) / range) * (h - 2) - 1;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const trend = yields[yields.length - 1] >= yields[0];
    const color = trend ? '#26a69a' : '#ef5350';
    return `<svg class="sparkline-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
        `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>` +
        `</svg>`;
}

// V9.2: SOPM Segmented Liquidity Bar (5 blocks)
function renderSopmBar(score) {
    if (score === undefined || score === null) return '<span style="color:#556677">--</span>';
    const s = Number(score);
    // 5-block segmented bar
    const blocks = 5;
    const filled = Math.max(0, Math.min(blocks, Math.round((s / 150) * blocks)));
    let color;
    if (s >= 100) color = '#26a69a';
    else if (s >= 70) color = '#ff9800';
    else color = '#ef5350';
    let bar = '<span class="sopm-bar-wrap"><span style="display:inline-flex;gap:1px">';
    for (let i = 0; i < blocks; i++) {
        const bg = i < filled ? color : 'rgba(255,255,255,0.06)';
        const glow = i < filled ? `box-shadow:0 0 3px ${color}66;` : '';
        bar += `<span style="width:8px;height:10px;border-radius:1px;background:${bg};${glow}"></span>`;
    }
    bar += `</span><span class="sopm-val" style="color:${color}">${s}</span></span>`;
    return bar;
}

// V9.2: Freshness Battery Icon
function renderFreshnessBattery(val, windowDays) {
    // val: 0.0 (fresh) to 1.0 (stale)
    const v = Number(val) || 0;
    const remaining = Math.max(0, 1 - v);
    const pct = (remaining * 100).toFixed(0);
    let color;
    if (v <= 0.3) color = '#26a69a';
    else if (v <= 0.6) color = '#ff9800';
    else color = '#ef5350';
    const dayLabel = windowDays ? `D${Math.round(v * windowDays)}` : '';
    return `<span style="display:inline-flex;align-items:center;gap:4px">` +
        `<span style="display:inline-block;width:20px;height:10px;border:1px solid ${color};border-radius:2px;position:relative;overflow:hidden">` +
        `<span style="position:absolute;left:0;top:0;height:100%;width:${pct}%;background:${color};box-shadow:0 0 3px ${color}66"></span>` +
        `</span>` +
        `<span style="font-size:12px;color:${color};font-weight:700">${dayLabel}</span></span>`;
}

// V9.2: Factor Radar tooltip (hover on product name)
function renderFactorRadar(r) {
    const antiHook = r['anti_hook_score'] || r['anti_hook_score'] === 0 ? r['anti_hook_score'] : Math.round((r['quality_penalty'] || 1) * 100);
    const hypeRatio = r['炒作比'] || 0;
    const density = r['高收益密度30'] || 0;
    const pulseW = r['平均脉冲宽度'] || 0;
    const sopmKey = r['is_high_liq'] ? '天天发' : r['is_low_liq'] ? '私银' : '周期';
    const yieldScore = r['收益强度分'] || r['yield_rank_score'] || 0;
    const stableScore = r['稳定性分'] || 0;
    const freshScore = r['时效性分'] || 0;
    const sopmScore = r['sopm_score'] || 0;
    const name = (r['产品名称'] || '').substring(0, 16);
    return `<span class="alpha-radar-wrap">` +
        `<span style="color:#e0e3eb">${name}</span>` +
        `<span class="alpha-radar-tip">` +
        `<div class="tip-row"><span class="tip-label">反钩分</span><span class="tip-val">${antiHook}</span></div>` +
        `<div class="tip-row"><span class="tip-label">炒作比</span><span class="tip-val">${fmt(hypeRatio, 1)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">密度30</span><span class="tip-val">${fmt(density, 1)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">脉冲宽</span><span class="tip-val">${fmt(pulseW, 1)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">流动性</span><span class="tip-val">${sopmKey}</span></div>` +
        `<div style="border-top:1px solid #2a2e39;margin:3px 0;padding-top:3px">` +
        `<div class="tip-row"><span class="tip-label">收益分</span><span class="tip-val">${fmt(yieldScore, 0)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">稳定分</span><span class="tip-val">${fmt(stableScore, 0)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">时效分</span><span class="tip-val">${fmt(freshScore, 0)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">流动分</span><span class="tip-val">${fmt(sopmScore, 0)}</span></div>` +
        `</div></span></span>`;
}

// V9.2: Alpha score display (kept for other uses)
function renderAlphaRadar(r) {
    const yieldScore = r['收益强度分'] || r['yield_rank_score'] || 0;
    const stableScore = r['稳定性分'] || 0;
    const freshScore = r['时效性分'] || 0;
    const sopmScore = r['sopm_score'] || 0;
    return `<span class="alpha-radar-wrap">` +
        `<span class="td-num">${fmt(r['综合得分'], 1)}</span>` +
        `<span class="alpha-radar-tip">` +
        `<div class="tip-row"><span class="tip-label">收益</span><span class="tip-val">${fmt(yieldScore, 0)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">稳定</span><span class="tip-val">${fmt(stableScore, 0)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">时效</span><span class="tip-val">${fmt(freshScore, 0)}</span></div>` +
        `<div class="tip-row"><span class="tip-label">流动</span><span class="tip-val">${fmt(sopmScore, 0)}</span></div>` +
        `</span></span>`;
}

function renderListBTable(data) {
    if (!data || data.length === 0) {
        setHTML('listBTableWrap', '<div class="empty-state"><p>暂无推荐交易产品</p></div>');
        return;
    }
    let html = '<table><thead><tr>';
    html += '<th>#</th><th>产品</th>';
    html += '<th>收益率 / 趋势</th><th>流动性</th>';
    html += '<th>风险</th><th>新鲜度</th><th>评分</th><th>操作</th>';
    html += '</tr></thead><tbody>';

    data.forEach((r, i) => {
        const rowId = `listb-row-${i}`;
        const sparkline = renderSparklineSVG(r['history_yields'], 60, 20);
        const yieldPct = fmt(r['最新收益率%'], 2);
        const yieldColor = (r['最新收益率%'] || 0) >= 0 ? '#26a69a' : '#ef5350';

        const velocity = r['yield_velocity'] || r['收益加速度'] || 0;
        const velArrow = velocity > 0 ? '<span style="color:#26a69a">&#9650;</span>' : velocity < 0 ? '<span style="color:#ef5350">&#9660;</span>' : '<span style="color:#5d606b">&#9644;</span>';
        const maxDD = r['max_drawdown'] || 0;
        const ddColor = maxDD < -0.1 ? '#ef5350' : '#787b86';

        const freshVal = r['freshness_val'] || r['新鲜度进度'] || 0;
        const windowDays = r['预测窗口天数'] || 0;
        const freshBattery = renderFreshnessBattery(freshVal, windowDays);

        html += `<tr id="${rowId}">`;
        html += `<td class="td-num" style="color:#5d606b">${i + 1}</td>`;
        html += `<td>${renderFactorRadar(r)}<br><span style="color:#2196f3;font-size:12px">${r['产品代码'] || ''}</span></td>`;
        html += `<td><span style="color:${yieldColor};font-weight:600;font-size:13px">${yieldPct}%</span> ${sparkline}</td>`;
        html += `<td>${renderSopmBar(r['sopm_score'])}</td>`;
        html += `<td style="font-size:13px">${velArrow}<span style="color:#b2b5be;margin:0 3px">${fmt(velocity, 2)}</span><span style="color:${ddColor};font-size:12px">DD${fmt(maxDD, 2)}%</span></td>`;
        html += `<td>${freshBattery}</td>`;
        html += `<td>${renderAlphaRadar(r)}</td>`;
        html += `<td style="white-space:nowrap">`;
        html += `<button class="btn-sold-out" onclick="flashSoldOut('${rowId}','${esc(r['银行'])}','${esc(r['产品代码'])}','${esc(r['产品名称'])}')">售罄</button> `;
        html += `<button class="btn-copy-rm" onclick="markAvailable('${rowId}')">可以购买</button>`;
        html += `</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('listBTableWrap', html);
}

// V10.1: Flash sold-out row then remove
function flashSoldOut(rowId, bank, code, name) {
    const row = document.getElementById(rowId);
    if (row) {
        row.classList.add('row-flash-out');
        setTimeout(() => {
            row.style.display = 'none';
            toggleSoldOut(bank, code, name);
        }, 600);
    } else {
        toggleSoldOut(bank, code, name);
    }
}

// V10.3: 标记可以购买 — 绿色边框高亮整行
function markAvailable(rowId) {
    const row = document.getElementById(rowId);
    if (!row) return;
    const btn = row.querySelector('.btn-copy-rm');
    if (row.classList.contains('row-checked')) {
        row.classList.remove('row-checked');
        if (btn) { btn.textContent = '可以购买'; }
    } else {
        row.classList.add('row-checked');
        if (btn) { btn.textContent = '已确认'; }
    }
}

// 转义 HTML 单引号用于 onclick
function esc(s) { return (s || '').replace(/'/g, "\\'").replace(/"/g, '&quot;'); }

// 复制话术到剪贴板
function copyRmScript(bank, name, code) {
    const text = `经理您好，请帮我查看 ${bank}|${name}（代码: ${code}）的额度。如有额度请帮我预约。`;
    navigator.clipboard.writeText(text).then(() => {
        showToast('话术已复制到剪贴板', 'success');
    }).catch(() => {
        // fallback
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast('话术已复制', 'success');
    });
}

// 标记售罄 (乐观UI)
async function toggleSoldOut(bank, code, name) {
    const res = await postJSON('/api/strategy/sold-out/mark', { bank, code, name });
    if (res && res.ok) {
        showToast(`已标记售罄: ${name || code}`, 'info');
        // 刷新 VIP 通道数据
        const channels = await fetchJSON('/api/strategy/vip-channels');
        if (channels && channels.list_b) {
            vipChannelData = channels;
            renderListBTable(channels.list_b);
            listBFiltered = channels.list_b;
            setText('listBCount', channels.list_b.length);
            loadSoldOutList();
        }
    } else {
        showToast('操作失败: ' + (res ? res.message : '网络错误'), 'error');
    }
}

// 取消售罄
async function restoreSoldOut(bank, code) {
    const res = await postJSON('/api/strategy/sold-out/unmark', { bank, code });
    if (res && res.ok) {
        showToast('已恢复产品', 'success');
        const channels = await fetchJSON('/api/strategy/vip-channels');
        if (channels && channels.list_b) {
            vipChannelData = channels;
            renderListBTable(channels.list_b);
            listBFiltered = channels.list_b;
            setText('listBCount', channels.list_b.length);
            loadSoldOutList();
        }
    }
}

// 售罄列表
async function loadSoldOutList() {
    const list = await fetchJSON('/api/strategy/sold-out');
    if (!list || list.length === 0) {
        hide('soldOutSection');
        return;
    }
    show('soldOutSection');
    setText('soldOutCount', list.length);

    let html = '<table><thead><tr><th>银行</th><th>代码</th><th>产品名称</th><th>收益率</th><th>剩余时间</th><th>操作</th></tr></thead><tbody>';
    list.forEach(item => {
        const yieldVal = item.yield ? parseFloat(item.yield).toFixed(2) + '%' : '--';
        const yieldColor = (item.yield && item.yield >= 3) ? '#26a69a' : '#ef5350';
        html += '<tr>';
        html += `<td>${item.bank}</td>`;
        html += `<td class="text-mono">${item.code}</td>`;
        html += `<td>${item.name || '--'}</td>`;
        html += `<td style="color:${yieldColor};font-weight:600">${yieldVal}</td>`;
        html += `<td class="text-sm text-muted">${item.remaining_hours}h</td>`;
        html += `<td><button class="btn-restore" onclick="restoreSoldOut('${esc(item.bank)}','${esc(item.code)}')">恢复</button></td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('soldOutTableWrap', html);
}

function toggleSoldOutList() {
    const wrap = document.getElementById('soldOutListWrap');
    const arrow = document.getElementById('soldOutArrow');
    if (wrap.style.display === 'none') {
        wrap.style.display = '';
        arrow.innerHTML = '&#x25BC;';
    } else {
        wrap.style.display = 'none';
        arrow.innerHTML = '&#x25B6;';
    }
}

function toggleUnclassified() {
    const body = document.getElementById('recCollapsibleBody');
    if (body) {
        body.classList.toggle('open');
    }
}

// List B 筛选
function filterListB() {
    if (!vipChannelData) return;
    let data = vipChannelData.list_b || [];
    const search = (document.getElementById('listBSearch')?.value || '').toLowerCase();
    const bank = document.getElementById('listBBankFilter')?.value || '';

    if (search) {
        data = data.filter(r =>
            (r['产品名称'] || '').toLowerCase().includes(search) ||
            (r['产品代码'] || '').toLowerCase().includes(search)
        );
    }
    if (bank) {
        data = data.filter(r => r['银行'] === bank);
    }

    listBFiltered = data;
    const countEl = document.getElementById('listBFilterCount');
    if (countEl) countEl.textContent = `${data.length} / ${(vipChannelData.list_b || []).length}`;
    renderListBTable(data);
}

function renderRecTable(data) {
    if (!data || data.length === 0) {
        setHTML('recTableWrap', '<div class="empty-state"><p>暂无推荐数据</p></div>');
        return;
    }

    let html = '<table><thead><tr>';
    html += '<th>排名</th><th>银行</th><th>代码</th><th>名称</th><th>流动性</th>';
    html += '<th class="sortable">成功率%</th><th>信号次数</th><th class="sortable">收益率%</th>';
    html += '<th>赎回费</th>';  // V13: 新增赎回费列
    html += '<th class="sortable">综合得分</th><th>新鲜度</th>';
    html += '<th>前瞻预测</th><th>操作建议</th>';
    html += '</tr></thead><tbody>';

    data.forEach(r => {
        const fc = r['前瞻加成'] === '是'
            ? tagCell(`${r['预测释放日'] || ''} (${fmtPct((r['预测置信度'] || 0) * 100)})`, 'info')
            : '';
        const ftag = r['新鲜度标签'] || '';
        let freshCell = '';
        if (ftag.includes('FRESH')) {
            freshCell = tagCell(ftag, 'buy');
        } else if (ftag.includes('FLASH')) {
            freshCell = tagCell(ftag, 'info');
        } else if (ftag.includes('STALE')) {
            freshCell = tagCell(ftag, 'sell');
        } else if (ftag) {
            freshCell = tagCell(ftag, 'watch');
        }

        // V13: 赎回费显示
        const feeRate14 = r['赎回费费率_14天'] || 0;
        const hasFee = r['有赎回费'];
        let feeCell = '';
        if (hasFee === true || feeRate14 > 0) {
            feeCell = `<span style="color:var(--accent-orange)">${(feeRate14 * 100).toFixed(2)}%</span>`;
        } else if (hasFee === false) {
            feeCell = '<span style="color:var(--accent-green);font-size:11px">无</span>';
        } else {
            feeCell = '<span style="color:var(--text-muted);font-size:11px">--</span>';
        }

        html += '<tr>';
        html += `<td class="td-num">${r['排名'] || ''}</td>`;
        html += `<td>${r['银行'] || ''}</td>`;
        html += `<td class="text-mono">${r['产品代码'] || ''}</td>`;
        html += `<td>${(r['产品名称'] || '').substring(0, 22)}</td>`;
        html += `<td>${r['流动性'] || ''}</td>`;
        html += numCell(r['历史成功率%'], 1);
        html += `<td class="td-num">${r['历史信号次数'] || ''}</td>`;
        html += numCell(r['最新收益率%'], 2);
        html += `<td class="td-num">${feeCell}</td>`;  // V13
        html += numCell(r['综合得分'], 1);
        html += `<td>${freshCell}</td>`;
        html += `<td>${fc}</td>`;
        html += `<td>${adviceTag(r['操作建议'])}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('recTableWrap', html);
}

// ══════════════════════════════════════════
// 加载信号
// ══════════════════════════════════════════

async function loadSignals() {
    const data = await fetchJSON('/api/strategy/opportunities');
    if (!data) return;
    setText('sigCount', data.length || 0);

    // 默认排序：★ 买入信号优先
    data.sort((a, b) => {
        const pa = (a['操作建议'] || '').includes('★') ? 0 : 1;
        const pb = (b['操作建议'] || '').includes('★') ? 0 : 1;
        if (pa !== pb) return pa - pb;
        return (b['最新收益率%'] || 0) - (a['最新收益率%'] || 0);
    });

    tableData.sig = data;
    populateFilterOptions('sig', '银行', 'sigBankFilter');
    renderSigTable(data);
}

function renderSigTable(data) {
    if (!data || data.length === 0) {
        setHTML('sigTableWrap', '<div class="empty-state"><p>暂无信号数据</p></div>');
        return;
    }

    let html = '<table><thead><tr>';
    html += '<th>银行</th><th>代码</th><th>名称</th><th>来源</th>';
    html += '<th>成功率%</th><th>信号日期</th><th>距今天数</th>';
    html += '<th>信号收益%</th><th>最新收益%</th><th>赎回费</th>';  // V13
    html += '<th>波动率%</th><th>持有天数</th><th>操作建议</th>';
    html += '</tr></thead><tbody>';

    data.slice(0, 200).forEach(o => {
        // V13: 赎回费显示
        const feeRate14 = o['赎回费费率_14天'] || 0;
        const hasFee = o['有赎回费'];
        let feeCell = '';
        if (hasFee === true || feeRate14 > 0) {
            feeCell = `<span style="color:var(--accent-orange)">${(feeRate14 * 100).toFixed(2)}%</span>`;
        } else if (hasFee === false) {
            feeCell = '<span style="color:var(--accent-green);font-size:11px">无</span>';
        } else {
            feeCell = '<span style="color:var(--text-muted);font-size:11px">--</span>';
        }

        html += '<tr>';
        html += `<td>${o['银行'] || ''}</td>`;
        html += `<td class="text-mono">${o['产品代码'] || ''}</td>`;
        html += `<td>${(o['产品名称'] || '').substring(0, 20)}</td>`;
        html += `<td>${o['来源'] === '观察池' ? tagCell('观察池', 'watch') : tagCell('产品库', 'hold')}</td>`;
        html += numCell(o['历史成功率%'], 1);
        html += `<td class="td-num">${o['信号日期'] || ''}</td>`;
        html += `<td class="td-num">${o['信号距今天数'] || ''}</td>`;
        html += numCell(o['信号收益率%'], 2);
        html += numCell(o['最新收益率%'], 2);
        html += `<td class="td-num">${feeCell}</td>`;  // V13
        html += numCell(o['波动率%'] || '', 2);
        html += `<td class="td-num">${o['预期持有天数'] || ''}</td>`;
        html += `<td>${adviceTag(o['操作建议'])}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('sigTableWrap', html);
}

// ══════════════════════════════════════════
// 加载持仓收益
// ══════════════════════════════════════════

let holdingReturnData = null;

async function loadHoldingReturns() {
    const data = await fetchJSON('/api/portfolio/holding-returns');
    if (!data) return;

    holdingReturnData = data;
    const holdings = data.holdings || [];
    const summary = data.summary || {};

    // KPI
    if (holdings.length > 0 || summary.total_cost > 0) {
        show('portKpiRow');
        animateNumber('portTotalCost', summary.total_cost || 0, 0);
        animateNumber('portTotalMV', summary.total_market_value || 0, 0);

        const pnlEl = document.getElementById('portTotalPnl');
        if (pnlEl) {
            const pnl = summary.total_pnl || 0;
            animateNumber('portTotalPnl', pnl, 2);
            pnlEl.parentElement.className = 'kpi-value ' + (pnl >= 0 ? 'td-pos' : 'td-neg');
        }
        const retEl = document.getElementById('portTotalRet');
        if (retEl) {
            const ret = summary.total_return_pct || 0;
            animateNumber('portTotalRet', ret, 2, '%');
            retEl.parentElement.className = 'kpi-value ' + (ret >= 0 ? 'td-pos' : 'td-neg');
        }

        // 累计年化收益率
        const annRetEl = document.getElementById('portTotalAnnRet');
        if (annRetEl) {
            const annRet = summary.total_ann_return_pct || 0;
            animateNumber('portTotalAnnRet', annRet, 2, '%');
            annRetEl.parentElement.className = 'kpi-value ' + (annRet >= 0 ? 'td-pos' : 'td-neg');
        }

        // 今日收益
        const todayPnlEl = document.getElementById('portTodayPnl');
        if (todayPnlEl) {
            const todayPnl = summary.today_pnl || 0;
            animateNumber('portTodayPnl', todayPnl, 2);
            todayPnlEl.parentElement.className = 'kpi-value ' + (todayPnl >= 0 ? 'td-pos' : 'td-neg');
        }

        setText('portHoldingCount', summary.holding_count || 0);
    }

    // 总览 KPI: 用持仓真实数据覆盖（与"我的持仓"一致）
    if (summary.holding_count !== undefined) {
        animateNumber('kpiHoldCount', summary.holding_count || 0, 0);
    }
    if (summary.total_market_value !== undefined) {
        setText('kpiHoldAmt', fmtMoney(summary.total_market_value || 0));
    }
    const annRet = summary.total_ann_return_pct || 0;
    const annRetSpan = document.getElementById('kpiHoldAnnRet');
    if (annRetSpan) {
        annRetSpan.textContent = annRet !== 0 ? annRet.toFixed(2) + '%' : '--';
        annRetSpan.className = 'text-mono ' + (annRet >= 0 ? 'td-pos' : 'td-neg');
    }
    const profit = summary.total_pnl || 0;
    const profitSpan = document.getElementById('kpiHoldProfit');
    if (profitSpan) {
        profitSpan.textContent = profit !== 0 ? fmtMoney(profit) : '--';
        profitSpan.className = 'text-mono ' + (profit >= 0 ? 'td-pos' : 'td-neg');
    }

    // 持仓明细
    tableData.port = holdings;
    renderPortTable(holdings);

    // 产品选择器
    const selector = document.getElementById('portChartProduct');
    if (selector) {
        selector.innerHTML = '<option value="all">全部持仓</option>';
        holdings.forEach(h => {
            if (h.status === '持有中') {
                const opt = document.createElement('option');
                opt.value = h.product_code;
                opt.textContent = `${h.bank} - ${h.product_code}`;
                selector.appendChild(opt);
            }
        });
    }

    // 绘制趋势图
    const series = data.daily_series || [];
    renderCumulativeChart(series);
    renderDailyChart(series);
}

function renderPortTable(data) {
    if (!data || data.length === 0) {
        setHTML('portTableWrap', '<div class="empty-state"><p>暂无持仓数据</p></div>');
        return;
    }

    let html = '<table><thead><tr>';
    html += '<th>银行</th><th>代码</th><th>名称</th><th>状态</th>';
    html += '<th>成本</th><th>当前市值</th><th>累计收益</th><th>收益率%</th>';
    html += '<th>持仓天数</th><th>年化收益%</th><th>最新日收益</th><th>在库</th><th>持仓建议</th>';
    html += '</tr></thead><tbody>';

    data.forEach(h => {
        const pnlCls = retClass(h.cumulative_pnl);
        const retCls = retClass(h.cumulative_return_pct);
        html += '<tr>';
        html += `<td>${h.bank || ''}</td>`;
        html += `<td class="text-mono">${h.product_code || ''}</td>`;
        const pname = h.product_name || '';
        const pnameStyle = pname.length > 20 ? ' style="font-size:13px;line-height:1.3"' : '';
        html += `<td${pnameStyle}>${pname}</td>`;
        html += `<td>${statusTag(h.status)}</td>`;
        html += `<td class="td-num">${fmtMoney(h.cost)}</td>`;
        html += `<td class="td-num">${fmtMoney(h.market_value)}</td>`;
        html += `<td class="td-num ${pnlCls}">${fmtMoney(h.cumulative_pnl)}</td>`;
        html += `<td class="td-num ${retCls}">${fmtPct(h.cumulative_return_pct)}</td>`;
        html += `<td class="td-num">${h.holding_days || '--'}</td>`;
        html += numCell(h.ann_return_pct, 2);
        html += `<td class="td-num ${retClass(h.daily_return)}">${fmtMoney(h.daily_return)}</td>`;
        html += `<td>${h.in_lib ? tagCell('是', 'buy') : tagCell('否', 'info')}</td>`;
        html += `<td>${adviceTag(h.advice)}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('portTableWrap', html);
}

function renderCumulativeChart(dailySeries) {
    const chart = getChart('chartCumulativeReturns');
    if (!chart || !dailySeries || dailySeries.length === 0) return;

    const dates = dailySeries.map(d => d.date);
    const cumPnl = dailySeries.map(d => d.cumulative_pnl || 0);
    const cumAnnRet = dailySeries.map(d => d.cumulative_ann_return_pct || 0);

    chart.setOption({
        ...CHART_BASE,
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            formatter: function (params) {
                let tip = params[0].axisValue + '<br/>';
                params.forEach(p => {
                    const color = p.color;
                    const suffix = p.seriesName.includes('%') ? '%' : '元';
                    tip += `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:5px"></span>`;
                    tip += `${p.seriesName}: ${Number(p.value).toFixed(2)}${suffix}<br/>`;
                });
                return tip;
            },
        },
        legend: {
            data: ['累计收益(元)', '累计年化收益率(%)'],
            textStyle: { color: '#b2b5be', fontSize: 12 },
            top: 5,
        },
        xAxis: {
            type: 'category',
            data: dates,
            axisLabel: { color: '#b2b5be', fontSize: 12 },
            axisLine: { lineStyle: { color: '#2a2e39' } },
        },
        yAxis: [
            {
                type: 'value',
                name: '累计收益(元)',
                nameTextStyle: { color: '#787b86', fontSize: 12 },
                axisLabel: { color: '#787b86' },
                splitLine: { lineStyle: { color: '#22262f' } },
            },
            {
                type: 'value',
                name: '年化收益率(%)',
                nameTextStyle: { color: '#787b86', fontSize: 12 },
                axisLabel: { color: '#b2b5be', formatter: '{value}%' },
                splitLine: { show: false },
            },
        ],
        dataZoom: [
            { type: 'slider', start: 0, end: 100, bottom: 0, height: 24, borderColor: '#2a2e39', handleStyle: { color: '#2196f3' }, backgroundColor: 'rgba(19,23,34,0.9)', dataBackground: { lineStyle: { color: '#5d606b' }, areaStyle: { color: 'rgba(33,150,243,0.06)' } }, textStyle: { color: '#b2b5be' } },
            { type: 'inside' },
        ],
        series: [
            {
                name: '累计收益(元)',
                type: 'line',
                yAxisIndex: 0,
                data: cumPnl.map(v => Number(v.toFixed(2))),
                showSymbol: false,
                lineStyle: { width: 2, color: '#2196f3' },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(33,150,243,0.12)' },
                        { offset: 1, color: 'rgba(33,150,243,0.01)' },
                    ]),
                },
            },
            {
                name: '累计年化收益率(%)',
                type: 'line',
                yAxisIndex: 1,
                data: cumAnnRet.map(v => Number(v.toFixed(2))),
                showSymbol: false,
                lineStyle: { width: 2, color: '#ab47bc' },
            },
        ],
    });
}

function renderDailyChart(dailySeries) {
    const chart = getChart('chartDailyReturns');
    if (!chart || !dailySeries || dailySeries.length === 0) return;

    const dates = dailySeries.map(d => d.date);
    const dailyPnl = dailySeries.map(d => d.daily_pnl || 0);
    const dailyAnnRet = dailySeries.map(d => d.daily_ann_return_pct || 0);

    chart.setOption({
        ...CHART_BASE,
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            formatter: function (params) {
                let tip = params[0].axisValue + '<br/>';
                params.forEach(p => {
                    const color = typeof p.color === 'string' ? p.color : '#666';
                    const suffix = p.seriesName.includes('%') ? '%' : '元';
                    tip += `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:5px"></span>`;
                    tip += `${p.seriesName}: ${Number(p.value).toFixed(2)}${suffix}<br/>`;
                });
                return tip;
            },
        },
        legend: {
            data: ['日收益(元)', '日年化收益率(%)'],
            textStyle: { color: '#b2b5be', fontSize: 12 },
            top: 5,
        },
        xAxis: {
            type: 'category',
            data: dates,
            axisLabel: { color: '#b2b5be', fontSize: 12 },
            axisLine: { lineStyle: { color: '#2a2e39' } },
        },
        yAxis: [
            {
                type: 'value',
                name: '日收益(元)',
                nameTextStyle: { color: '#787b86', fontSize: 12 },
                axisLabel: { color: '#787b86' },
                splitLine: { lineStyle: { color: '#22262f' } },
            },
            {
                type: 'value',
                name: '日年化收益率(%)',
                nameTextStyle: { color: '#787b86', fontSize: 12 },
                axisLabel: { color: '#b2b5be', formatter: '{value}%' },
                splitLine: { show: false },
            },
        ],
        dataZoom: [
            { type: 'slider', start: 0, end: 100, bottom: 0, height: 24, borderColor: '#2a2e39', handleStyle: { color: '#2196f3' }, backgroundColor: 'rgba(19,23,34,0.9)', dataBackground: { lineStyle: { color: '#5d606b' }, areaStyle: { color: 'rgba(33,150,243,0.06)' } }, textStyle: { color: '#b2b5be' } },
            { type: 'inside' },
        ],
        series: [
            {
                name: '日收益(元)',
                type: 'bar',
                yAxisIndex: 0,
                data: dailyPnl.map(v => ({
                    value: Number(v.toFixed(2)),
                    itemStyle: { color: v >= 0 ? '#26a69a' : '#ef5350' },
                })),
                barWidth: '60%',
            },
            {
                name: '日年化收益率(%)',
                type: 'line',
                yAxisIndex: 1,
                data: dailyAnnRet.map(v => Number(v.toFixed(2))),
                showSymbol: false,
                lineStyle: { width: 2, color: '#ff9800' },
            },
        ],
    });
}

function updatePortChart() {
    if (!holdingReturnData) return;
    const series = holdingReturnData.daily_series || [];
    renderCumulativeChart(series);
    renderDailyChart(series);
}

// ══════════════════════════════════════════
// 交易记录
// ══════════════════════════════════════════

async function loadTradeHistory() {
    const data = await fetchJSON('/api/portfolio/trades');
    if (!data) return;
    setText('tradeRecordCount', data.length || 0);
    tableData.trade = data;
    renderTradeTable(data);
}

function renderTradeTable(data) {
    if (!data || data.length === 0) {
        setHTML('tradeTableWrap', '<div class="empty-state"><p>暂无交易记录</p></div>');
        return;
    }

    let html = '<table><thead><tr>';
    html += '<th>银行</th><th>产品代码</th><th>产品名称</th><th>交易类型</th><th>金额</th><th>日期</th>';
    html += '</tr></thead><tbody>';

    // 倒序（最新在前）
    [...data].reverse().forEach(t => {
        html += '<tr>';
        html += `<td>${t['银行'] || ''}</td>`;
        html += `<td class="text-mono">${t['产品代码'] || ''}</td>`;
        html += `<td>${(t['产品名称'] || '').substring(0, 22)}</td>`;
        html += `<td>${tradeTypeTag(t['交易'])}</td>`;
        html += `<td class="td-num">${fmtMoney(t['金额'])}</td>`;
        html += `<td class="td-num">${t['日期'] || ''}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('tradeTableWrap', html);
}

// ══════════════════════════════════════════
// 手工录入交易
// ══════════════════════════════════════════

let suggestTimeout = null;

function onTradeCodeInput() {
    const code = document.getElementById('tradeCode')?.value || '';
    if (code.length < 2) {
        hide('tradeSuggestions');
        return;
    }

    clearTimeout(suggestTimeout);
    suggestTimeout = setTimeout(async () => {
        const data = await fetchJSON(`/api/portfolio/suggestions?q=${encodeURIComponent(code)}`);
        if (!data || data.length === 0) {
            hide('tradeSuggestions');
            return;
        }

        let html = '';
        data.forEach(s => {
            html += `<div style="padding:6px 10px;cursor:pointer;font-size:13px;border-bottom:1px solid var(--divider)"
                          onmouseover="this.style.background='var(--accent-blue-light)'"
                          onmouseout="this.style.background=''"
                          onclick="selectSuggestion('${s.bank}','${s.product_code}','${(s.product_name || '').replace(/'/g, "\\'")}')">
                        <span class="text-mono">${s.product_code}</span>
                        <span style="color:var(--text-muted);margin-left:8px">${s.bank}</span>
                        <br><span style="font-size:13px;color:var(--text-secondary)">${s.product_name || ''}</span>
                    </div>`;
        });
        setHTML('tradeSuggestions', html);
        show('tradeSuggestions');
    }, 300);
}

function selectSuggestion(bank, code, name) {
    document.getElementById('tradeBank').value = bank;
    document.getElementById('tradeCode').value = code;
    document.getElementById('tradeName').value = name;
    hide('tradeSuggestions');
}

// 点击外部关闭建议
document.addEventListener('click', e => {
    if (!e.target.closest('#tradeCode') && !e.target.closest('#tradeSuggestions')) {
        hide('tradeSuggestions');
    }
});

async function submitTrade() {
    const bank = document.getElementById('tradeBank')?.value;
    const code = document.getElementById('tradeCode')?.value;
    const name = document.getElementById('tradeName')?.value;
    const type = document.getElementById('tradeType')?.value;
    const amount = document.getElementById('tradeAmount')?.value;
    const date = document.getElementById('tradeDate')?.value;

    if (!bank || !code || !amount || !date) {
        showToast('请填写必要字段（银行、代码、金额、日期）', 'warning');
        return;
    }

    const result = await postJSON('/api/portfolio/add', {
        bank, product_code: code, product_name: name || '',
        trade_type: type, amount: parseFloat(amount), date,
    });

    if (result && result.ok) {
        showToast('交易已录入', 'success');
        // 清空表单
        document.getElementById('tradeCode').value = '';
        document.getElementById('tradeName').value = '';
        document.getElementById('tradeAmount').value = '';
        // 刷新
        loadTradeHistory();
        loadHoldingReturns();
    } else {
        showToast('录入失败: ' + (result?.message || ''), 'error');
    }
}

// ══════════════════════════════════════════
// OCR 图片上传
// ══════════════════════════════════════════

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) handleFileUpload(files);
}

async function handleFileUpload(files) {
    if (!files || files.length === 0) return;

    const file = files[0];
    if (!file.type.startsWith('image/')) {
        showToast('请上传图片文件', 'warning');
        return;
    }

    showToast('正在识别...', 'info');

    const formData = new FormData();
    formData.append('image', file);

    try {
        const res = await fetch('/api/portfolio/ocr', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.ok && data.trades && data.trades.length > 0) {
            ocrTrades = data.trades;
            showOcrResults(data.trades);
            showToast(data.message, 'success');
        } else {
            showToast(data.message || '识别失败', 'error');
        }
    } catch (e) {
        showToast('上传失败', 'error');
    }
}

function showOcrResults(trades) {
    show('ocrResultWrap');

    let html = '<table><thead><tr>';
    html += '<th>银行</th><th>代码</th><th>名称</th><th>类型</th><th>金额</th><th>日期</th>';
    html += '</tr></thead><tbody>';

    trades.forEach((t, i) => {
        html += '<tr>';
        html += `<td><input class="form-input" style="width:80px" value="${t.bank || ''}" data-ocr="${i}" data-field="bank"></td>`;
        html += `<td><input class="form-input" style="width:100px" value="${t.product_code || ''}" data-ocr="${i}" data-field="product_code"></td>`;
        html += `<td><input class="form-input" style="width:150px" value="${t.product_name || ''}" data-ocr="${i}" data-field="product_name"></td>`;
        html += `<td><select class="form-select" style="width:70px" data-ocr="${i}" data-field="trade_type">
                    <option value="买入" ${t.trade_type === '买入' ? 'selected' : ''}>买入</option>
                    <option value="卖出" ${t.trade_type === '卖出' ? 'selected' : ''}>卖出</option>
                 </select></td>`;
        html += `<td><input class="form-input" style="width:100px" type="number" value="${t.amount || ''}" data-ocr="${i}" data-field="amount"></td>`;
        html += `<td><input class="form-input" style="width:120px" type="date" value="${t.date || ''}" data-ocr="${i}" data-field="date"></td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('ocrResultTable', html);
}

async function submitOcrTrades() {
    // 从可编辑表格中收集数据
    const inputs = document.querySelectorAll('[data-ocr]');
    const trades = {};
    inputs.forEach(el => {
        const idx = el.dataset.ocr;
        const field = el.dataset.field;
        if (!trades[idx]) trades[idx] = {};
        trades[idx][field] = el.value;
    });

    const tradeList = Object.values(trades);
    if (tradeList.length === 0) return;

    const result = await postJSON('/api/portfolio/add-batch', tradeList);
    if (result && result.ok) {
        showToast(result.message, 'success');
        hide('ocrResultWrap');
        ocrTrades = [];
        loadTradeHistory();
        loadHoldingReturns();
    } else {
        showToast('批量录入失败: ' + (result?.message || ''), 'error');
    }
}

// ══════════════════════════════════════════
// 加载规律
// ══════════════════════════════════════════

async function loadPatterns() {
    const data = await fetchJSON('/api/strategy/patterns');
    if (!data) return;
    setText('patCount', data.length || 0);
    tableData.pat = data;

    populateFilterOptions('pat', 'bank', 'patBankFilter');

    // 置信度分布图
    const top20 = data.slice(0, 20);
    const chartConf = getChart('chartPatternConf');
    if (chartConf && top20.length > 0) {
        chartConf.setOption({
            ...CHART_BASE,
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: top20.map(p => (p.code || '').substring(0, 10)),
                axisLabel: { rotate: 45, fontSize: 12, color: '#b2b5be' },
                axisLine: { lineStyle: { color: '#2a2e39' } },
            },
            yAxis: {
                type: 'value', max: 1,
                axisLabel: { color: '#787b86' },
                splitLine: { lineStyle: { color: '#22262f' } },
            },
            series: [{
                type: 'bar',
                data: top20.map(p => ({
                    value: p.confidence,
                    itemStyle: {
                        color: p.confidence >= 0.7 ? '#26a69a' : (p.confidence >= 0.4 ? '#ff9800' : '#ef5350'),
                    },
                })),
                barWidth: '60%',
                itemStyle: { borderRadius: [3, 3, 0, 0] },
            }],
        });
    }

    // 周期分布
    const periods = data.filter(p => p.has_period).map(p => p.period_days);
    if (periods.length > 0) {
        const bins = { '5-15天': 0, '15-30天': 0, '30-60天': 0, '60-90天': 0, '90+天': 0 };
        periods.forEach(d => {
            if (d < 15) bins['5-15天']++;
            else if (d < 30) bins['15-30天']++;
            else if (d < 60) bins['30-60天']++;
            else if (d < 90) bins['60-90天']++;
            else bins['90+天']++;
        });
        const chartPer = getChart('chartPatternPeriod');
        if (chartPer) {
            chartPer.setOption({
                ...CHART_BASE,
                color: APPLE_COLORS,
                tooltip: { trigger: 'item' },
                series: [{
                    type: 'pie',
                    radius: ['35%', '65%'],
                    center: ['50%', '55%'],
                    label: { color: '#b2b5be', fontSize: 13 },
                    data: Object.entries(bins).filter(([, v]) => v > 0).map(([k, v]) => ({ name: k, value: v })),
                    itemStyle: { borderRadius: 4, borderColor: '#131722', borderWidth: 2 },
                }],
            });
        }
    }

    renderPatTable(data);
}

async function loadAiPredictions() {
    try {
        const data = await fetchJSON('/api/gpu/predict/status');
        // Try fetching cached predictions from summary
        const summary = await fetchJSON('/api/strategy/summary');
        if (summary && summary.prediction_source === 'deep_learning') {
            // Fetch the actual predictions
            const predResult = await postJSON('/api/gpu/predict', {});
            if (predResult && predResult.ok && predResult.predictions) {
                renderPredictionHeatmap(predResult.predictions);
            }
        }
    } catch (e) {
        // AI predictions not available, that's ok
    }
}

function renderPatTable(data) {
    if (!data || data.length === 0) {
        setHTML('patTableWrap', '<div class="empty-state"><p>暂无规律数据</p></div>');
        return;
    }

    let html = '<table><thead><tr>';
    html += '<th>银行</th><th>代码</th><th>周期(天)</th><th>CV</th>';
    html += '<th>释放阶段</th><th>释放星期</th><th>事件数</th>';
    html += '<th>置信度</th><th>上次释放</th><th>预测</th>';
    html += '</tr></thead><tbody>';

    data.forEach(p => {
        const confCls = p.confidence >= 0.7 ? 'td-pos' : (p.confidence >= 0.4 ? 'td-warn' : 'td-neg');
        const pred = p.prediction
            ? tagCell(`${p.prediction.predicted_date || ''} (${fmt(p.prediction.confidence, 2)})`, 'info')
            : '';
        html += '<tr>';
        html += `<td>${p.bank || ''}</td>`;
        html += `<td class="text-mono">${p.code || ''}</td>`;
        html += `<td class="td-num">${p.has_period ? fmt(p.period_days, 0) + '天' : '--'}</td>`;
        html += `<td class="td-num">${fmt(p.period_cv, 3)}</td>`;
        html += `<td>${p.top_phase || '--'}</td>`;
        html += `<td>${p.top_weekday || '--'}</td>`;
        html += `<td class="td-num">${p.n_events || ''}</td>`;
        html += `<td class="td-num ${confCls}">${fmt(p.confidence, 3)}</td>`;
        html += `<td class="td-num">${p.last_release || ''}</td>`;
        html += `<td>${pred}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('patTableWrap', html);
}

// ══════════════════════════════════════════
// 策略执行
// ══════════════════════════════════════════

function runStrategy(forceRefresh) {
    const btn = document.getElementById('btnRunStrategy');
    const btn2 = document.getElementById('btnRefreshStrategy');
    if (btn) btn.disabled = true;
    if (btn2) btn2.disabled = true;
    show('strategyProgressWrap');
    setText('strategyProgressText', '启动中...');
    updateGlobalStatus('running');

    fetch('/api/strategy/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force_refresh: forceRefresh }),
    }).then(() => {
        const es = new EventSource('/api/strategy/status');
        es.onmessage = function (e) {
            const d = JSON.parse(e.data);
            document.getElementById('strategyProgressBar').style.width = d.progress + '%';
            setText('strategyProgressText', d.message || '');
            if (d.status === 'done') {
                es.close();
                if (btn) btn.disabled = false;
                if (btn2) btn2.disabled = false;
                updateGlobalStatus('done');
                showToast('策略运行完成', 'success');
                refreshAll();
            } else if (d.status === 'error') {
                es.close();
                if (btn) btn.disabled = false;
                if (btn2) btn2.disabled = false;
                updateGlobalStatus('error');
                setText('strategyProgressText', '错误: ' + (d.error || ''));
                showToast('策略运行失败', 'error');
            }
        };
        es.onerror = function () {
            es.close();
            if (btn) btn.disabled = false;
            if (btn2) btn2.disabled = false;
            updateGlobalStatus('error');
        };
    });
}

// ══════════════════════════════════════════
// 回测
// ══════════════════════════════════════════

function runBacktest() {
    const btn = document.getElementById('btnRunBacktest');
    if (btn) btn.disabled = true;
    show('btProgressWrap');
    setText('btProgressText', '启动中...');

    fetch('/api/backtest/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
    }).then(() => {
        const es = new EventSource('/api/backtest/status');
        es.onmessage = function (e) {
            const d = JSON.parse(e.data);
            document.getElementById('btProgressBar').style.width = d.progress + '%';
            setText('btProgressText', d.message || '');
            if (d.status === 'done') {
                es.close();
                if (btn) btn.disabled = false;
                showToast('回测运行完成', 'success');
                loadBacktestResults();
            } else if (d.status === 'error') {
                es.close();
                if (btn) btn.disabled = false;
                setText('btProgressText', '错误: ' + (d.error || ''));
                showToast('回测运行失败', 'error');
            }
        };
        es.onerror = function () {
            es.close();
            if (btn) btn.disabled = false;
        };
    });
}

async function loadBacktestResults() {
    const results = await fetchJSON('/api/backtest/results');
    if (!results || !results.total_trades) return;

    show('btKpiRow');
    show('btNavRow');
    show('btDetailRow');
    show('btBankRow');

    // KPI
    const retCls = results.ann_return >= 0 ? 'td-pos' : 'td-neg';
    const annRetEl = document.getElementById('btAnnRet');
    if (annRetEl) {
        animateNumber('btAnnRet', results.ann_return || 0, 2, '%');
        annRetEl.className = 'kpi-value ' + retCls;
    }
    setText('btTotalRet', fmtPct(results.total_return));
    animateNumber('btSharpe', results.sharpe || 0, 2);
    setText('btPF', fmt(results.profit_factor, 2));
    setText('btMaxDD', fmtPct(results.max_drawdown));
    setText('btDDDate', results.max_dd_date || '--');
    animateNumber('btWinRate', results.win_rate || 0, 1, '%');
    setText('btTrades', results.total_trades);
    setText('btWins', results.wins);
    setText('btLosses', results.losses);

    // 净值曲线
    const nav = await fetchJSON('/api/backtest/nav');
    if (nav && nav.length > 0) {
        const chartNav = getChart('chartNav');
        if (chartNav) {
            chartNav.setOption({
                ...CHART_BASE,
                tooltip: {
                    trigger: 'axis',
                    formatter: function (p) {
                        const d = p[0];
                        return `${d.axisValue}<br/>净值: ${d.value[1]}万<br/>现金: ${d.value[2]}万`;
                    },
                },
                xAxis: {
                    type: 'category',
                    data: nav.map(n => n.date),
                    axisLabel: { color: '#b2b5be', fontSize: 12 },
                    axisLine: { lineStyle: { color: '#2a2e39' } },
                },
                yAxis: {
                    type: 'value',
                    name: '净值 (万)',
                    nameTextStyle: { color: '#787b86' },
                    axisLabel: { color: '#787b86' },
                    splitLine: { lineStyle: { color: '#22262f' } },
                    scale: true,
                },
                dataZoom: [
                    { type: 'slider', start: 0, end: 100, bottom: 0, height: 24, borderColor: '#2a2e39', handleStyle: { color: '#2196f3' }, backgroundColor: 'rgba(19,23,34,0.9)', textStyle: { color: '#b2b5be' } },
                    { type: 'inside' },
                ],
                series: [{
                    name: '净值',
                    type: 'line',
                    data: nav.map(n => [n.date, n.value, n.cash]),
                    showSymbol: false,
                    lineStyle: { width: 2, color: '#2196f3' },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(0,113,227,0.15)' },
                            { offset: 1, color: 'rgba(0,113,227,0.01)' },
                        ]),
                    },
                }],
            });
        }
    }

    // 月度收益
    if (nav && nav.length > 0) {
        const monthlyData = {};
        nav.forEach(n => { monthlyData[n.date.substring(0, 7)] = n.value; });
        const months = Object.keys(monthlyData).sort();
        const monthlyRets = [];
        let prev = nav[0].value;
        months.forEach(m => {
            const cur = monthlyData[m];
            monthlyRets.push({ month: m, return_pct: ((cur / prev) - 1) * 100 });
            prev = cur;
        });

        if (monthlyRets.length > 0) {
            const chartMon = getChart('chartMonthly');
            if (chartMon) {
                chartMon.setOption({
                    ...CHART_BASE,
                    tooltip: { trigger: 'axis', formatter: '{b}: {c}%' },
                    xAxis: {
                        type: 'category',
                        data: monthlyRets.map(m => m.month),
                        axisLabel: { color: '#b2b5be', fontSize: 12 },
                        axisLine: { lineStyle: { color: '#2a2e39' } },
                    },
                    yAxis: {
                        type: 'value',
                        axisLabel: { color: '#787b86' },
                        splitLine: { lineStyle: { color: '#22262f' } },
                    },
                    series: [{
                        type: 'bar',
                        data: monthlyRets.map(m => ({
                            value: Number(m.return_pct.toFixed(4)),
                            itemStyle: {
                                color: m.return_pct >= 0 ? '#26a69a' : '#ef5350',
                                borderRadius: [3, 3, 0, 0],
                            },
                        })),
                        barWidth: '50%',
                    }],
                });
            }
        }
    }

    // 交易日志
    const trades = await fetchJSON('/api/backtest/trades');
    setText('btTradeCount', trades?.length || 0);
    tableData.btTrade = trades || [];
    renderBtTradeTable(trades || []);

    // 分银行绩效
    const bankStats = results.bank_stats || {};
    const bankNames = Object.keys(bankStats);
    if (bankNames.length > 0) {
        const chartBP = getChart('chartBankPerf');
        if (chartBP) {
            chartBP.setOption({
                ...CHART_BASE,
                tooltip: { trigger: 'axis' },
                legend: { data: ['盈亏(万)', '胜率(%)'], textStyle: { color: '#b2b5be' } },
                xAxis: {
                    type: 'category',
                    data: bankNames,
                    axisLabel: { color: '#b2b5be', fontSize: 12, rotate: 30 },
                    axisLine: { lineStyle: { color: '#2a2e39' } },
                },
                yAxis: [
                    { type: 'value', name: '盈亏(万)', axisLabel: { color: '#787b86' }, splitLine: { lineStyle: { color: '#22262f' } } },
                    { type: 'value', name: '胜率%', max: 100, axisLabel: { color: '#787b86' }, splitLine: { show: false } },
                ],
                series: [
                    {
                        name: '盈亏(万)',
                        type: 'bar',
                        data: bankNames.map(b => ({
                            value: bankStats[b].pnl,
                            itemStyle: { color: bankStats[b].pnl >= 0 ? '#26a69a' : '#ef5350', borderRadius: [3, 3, 0, 0] },
                        })),
                        barWidth: '40%',
                    },
                    {
                        name: '胜率(%)',
                        type: 'line',
                        yAxisIndex: 1,
                        data: bankNames.map(b => bankStats[b].win_rate),
                        lineStyle: { color: '#ff9800', width: 2 },
                        itemStyle: { color: '#ff9800' },
                    },
                ],
            });
        }
    }
}

function renderBtTradeTable(data) {
    if (!data || data.length === 0) {
        setHTML('btTradeTableWrap', '<div class="empty-state"><p>暂无交易数据</p></div>');
        return;
    }

    let html = '<table><thead><tr>';
    html += '<th>日期</th><th>操作</th><th>银行</th><th>代码</th>';
    html += '<th>净值</th><th>金额(万)</th><th>盈亏(万)</th><th>持有</th><th>原因</th>';
    html += '</tr></thead><tbody>';

    data.slice(-200).forEach(t => {
        const pnlCls = t.pnl > 0 ? 'td-pos' : (t.pnl < 0 ? 'td-neg' : '');
        html += '<tr>';
        html += `<td class="td-num">${t.date || ''}</td>`;
        html += `<td>${t.action || ''}</td>`;
        html += `<td>${t.bank || ''}</td>`;
        html += `<td class="text-mono">${t.code || ''}</td>`;
        html += `<td class="td-num">${fmt(t.nav, 6)}</td>`;
        html += `<td class="td-num">${fmt(t.amount, 2)}</td>`;
        html += `<td class="td-num ${pnlCls}">${fmt(t.pnl, 2)}</td>`;
        html += `<td class="td-num">${t.hold_days || ''}</td>`;
        html += `<td style="max-width:180px;overflow:hidden;text-overflow:ellipsis">${(t.reason || '').substring(0, 40)}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('btTradeTableWrap', html);
}

// ══════════════════════════════════════════
// 数据管理 — 净值数据库
// ══════════════════════════════════════════

async function loadDbStats(isRetry, forceRefresh) {
    const ts = Date.now();  // cache-bust
    const url = forceRefresh ? '/api/crawl/stats?refresh=1&_=' + ts : '/api/crawl/stats?_=' + ts;
    if (forceRefresh) {
        setHTML('dbStatsTableWrap', '<div class="empty-state"><p>正在刷新...</p></div>');
    }
    const resp = await fetchJSON(url);

    // V13: 新结构 {banks: [...], fee_db: {...}}
    const data = resp?.banks || resp || [];  // 兼容旧结构
    const feeDb = resp?.fee_db || {total: 0, with_fee: 0, coverage: '0%'};

    if (!data || data.length === 0) {
        setHTML('dbStatsTableWrap', '<div class="empty-state"><p>未找到数据库</p></div>');
        return;
    }

    // 如果后台还在加载中，3秒后重试
    const isLoading = data.length === 1 && (data[0].bank || '').includes('加载中');
    if (isLoading) {
        setHTML('dbStatsTableWrap', '<div class="empty-state"><p>数据库统计加载中...</p></div>');
        if (!isRetry) {
            setTimeout(() => loadDbStats(true), 3000);
        } else {
            setTimeout(() => loadDbStats(true), 5000);
        }
        return;
    }

    // V13: 费率数据库统计
    let feeHtml = '<div style="margin-bottom:12px;padding:8px 12px;background:var(--card-bg);border-radius:6px;font-size:13px;">';
    feeHtml += '<span style="color:var(--text-muted)">赎回费数据库:</span> ';
    feeHtml += `<span style="font-weight:600;color:var(--accent-blue)">${feeDb.with_fee}</span>`;
    feeHtml += `<span style="color:var(--text-muted)"> / ${feeDb.total} 产品已采集费率</span>`;
    if (feeDb.with_fee === 0) {
        feeHtml += ' <span style="color:var(--accent-red);font-size:11px">(⚠️ 暂无费率数据)</span>';
    } else if (feeDb.with_fee < 100) {
        feeHtml += ` <span style="color:var(--accent-orange);font-size:11px">(覆盖率 ${feeDb.coverage})</span>`;
    } else {
        feeHtml += ` <span style="color:var(--accent-green);font-size:11px">(覆盖率 ${feeDb.coverage})</span>`;
    }
    feeHtml += '</div>';

    let html = feeHtml;
    html += '<table><thead><tr>';
    html += '<th>银行</th><th class="td-num">产品数</th><th class="td-num">日期数</th><th class="td-num">最早日期</th><th class="td-num">最新日期</th>';
    html += '</tr></thead><tbody>';

    data.forEach(s => {
        html += '<tr>';
        html += `<td>${s.bank || ''}</td>`;
        html += `<td class="td-num">${(s.products || 0).toLocaleString()}</td>`;
        html += `<td class="td-num">${(s.dates || 0).toLocaleString()}</td>`;
        html += `<td class="td-num">${s.earliest_date || '--'}</td>`;
        html += `<td class="td-num">${s.latest_date || '--'}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('dbStatsTableWrap', html);
}

function getSelectedBanks() {
    return [...document.querySelectorAll('#crawlBankList input[type=checkbox]:checked')]
        .map(cb => cb.dataset.key);
}

async function loadBankList() {
    const wrap = document.getElementById('crawlBankList');
    if (!wrap) return;
    try {
        const banks = await fetchJSON('/api/crawl/banks');
        if (!banks || banks.length === 0) {
            wrap.innerHTML = '<span style="color:var(--text-muted);font-size:12px">暂无银行配置</span>';
            return;
        }
        const mainBanks = banks.filter(b => b.group === 'main');
        const extraBanks = banks.filter(b => b.group !== 'main');
        let html = '';
        function bankItem(b, checked) {
            const chk = checked ? ' checked' : '';
            return `<label class="checkbox-label bank-select-item">
                <input type="checkbox" data-key="${b.key}"${chk}>
                <span>${b.name}</span>
                <select class="bank-mode-select" data-bank="${b.key}">
                    <option value="incr">增量</option>
                    <option value="full">全量</option>
                </select>
            </label>`;
        }
        mainBanks.forEach(b => { html += bankItem(b, true); });
        if (mainBanks.length && extraBanks.length) {
            html += '<span style="color:var(--text-muted);font-size:11px;margin:0 6px">|</span>';
        }
        extraBanks.forEach(b => { html += bankItem(b, false); });
        wrap.innerHTML = html;
    } catch (e) {
        wrap.innerHTML = '<span style="color:var(--text-muted);font-size:12px">加载失败</span>';
    }
}

function toggleAllBanks() {
    const boxes = document.querySelectorAll('#crawlBankList input[type=checkbox]');
    const allChecked = [...boxes].every(cb => cb.checked);
    boxes.forEach(cb => cb.checked = !allChecked);
}

function confirmFullCrawl() {
    openModal(
        '确认全量抓取',
        '全量抓取将忽略各银行的增量/全量选择，全部以全量模式抓取。确定继续？',
        () => runCrawl(true)
    );
}

function _showCrawlRunning() {
    const btnIncr = document.getElementById('btnCrawlIncr');
    const btnFull = document.getElementById('btnCrawlFull');
    const btnStop = document.getElementById('btnCrawlStop');
    if (btnIncr) btnIncr.disabled = true;
    if (btnFull) btnFull.disabled = true;
    if (btnStop) btnStop.style.display = '';
    show('crawlProgressWrap');
}

function _showCrawlIdle() {
    const btnIncr = document.getElementById('btnCrawlIncr');
    const btnFull = document.getElementById('btnCrawlFull');
    const btnStop = document.getElementById('btnCrawlStop');
    if (btnIncr) btnIncr.disabled = false;
    if (btnFull) btnFull.disabled = false;
    if (btnStop) btnStop.style.display = 'none';
    // Keep per-bank progress visible briefly after completion (don't clear immediately)
}

function _updateCrawlProgress(d) {
    const pct = d.progress || 0;
    document.getElementById('crawlProgressBar').style.width = pct + '%';
    const pctEl = document.getElementById('crawlProgressPct');
    if (pctEl) pctEl.textContent = pct > 0 ? pct + '%' : '';

    // Overall text
    const msg = d.message || '';
    const pctPrefix = pct > 0 ? pct + '% — ' : '';
    setText('crawlProgressText', pctPrefix + (msg.includes(' | ') ? '抓取中...' : msg));

    // Per-bank progress bars
    const bp = d.bank_progress;
    const container = document.getElementById('crawlBankProgressList');
    if (!container) return;
    if (!bp || Object.keys(bp).length === 0) {
        container.innerHTML = '';
        return;
    }
    // Build / update rows
    const keys = Object.keys(bp);
    keys.forEach(key => {
        const info = bp[key];
        let row = container.querySelector('[data-bank-key="' + key + '"]');
        if (!row) {
            row = document.createElement('div');
            row.className = 'bank-progress-item';
            row.setAttribute('data-bank-key', key);
            row.innerHTML =
                '<span class="bank-progress-name"></span>' +
                '<div class="bank-progress-bar-wrap"><div class="progress-bar"></div></div>' +
                '<span class="bank-progress-text"></span>' +
                '<button class="bank-stop-btn" title="停止此银行" onclick="stopBank(\'' + key + '\')">&#x25A0;</button>';
            container.appendChild(row);
        }
        const nameEl = row.querySelector('.bank-progress-name');
        const barEl = row.querySelector('.progress-bar');
        const textEl = row.querySelector('.bank-progress-text');
        const stopBtn = row.querySelector('.bank-stop-btn');

        nameEl.textContent = info.name || key;

        const total = info.total || 0;
        const current = info.current || 0;
        const status = info.status || 'pending';
        const widthPct = total > 0 ? Math.min(Math.round(current / total * 100), 100) : (status === 'done' ? 100 : 0);

        barEl.style.width = widthPct + '%';
        barEl.className = 'progress-bar bp-' + status;

        if (status === 'done') {
            textEl.textContent = total > 0 ? current + '/' + total : '完成';
        } else if (status === 'error') {
            textEl.textContent = '失败';
        } else if (status === 'stopped') {
            textEl.textContent = '已停止';
        } else if (status === 'pending') {
            textEl.textContent = '等待中';
        } else {
            textEl.textContent = total > 0 ? current + '/' + total : '初始化...';
        }

        // Show stop button only for running/pending banks
        if (stopBtn) {
            stopBtn.style.display = (status === 'running' || status === 'pending') ? '' : 'none';
        }
    });
    // Remove stale rows
    container.querySelectorAll('.bank-progress-item').forEach(row => {
        if (!bp[row.getAttribute('data-bank-key')]) row.remove();
    });
}

function _subscribeCrawlSSE(crawlStart) {
    const es = new EventSource('/api/crawl/status');
    let elapsedTimer = null;
    if (crawlStart) {
        elapsedTimer = setInterval(() => {
            const sec = Math.floor((Date.now() - crawlStart) / 1000);
            const m = Math.floor(sec / 60), s = sec % 60;
            setText('crawlElapsed', `已用时 ${m}:${String(s).padStart(2, '0')}`);
        }, 1000);
    }

    function done() {
        if (elapsedTimer) clearInterval(elapsedTimer);
        _showCrawlIdle();
    }

    es.onmessage = function (e) {
        const d = JSON.parse(e.data);
        _updateCrawlProgress(d);

        if (d.status === 'done') {
            es.close(); done();
            showToast('抓取完成', 'success');
            loadDbStats(false, true);  // force refresh stats after crawl
            loadCrawlResults();
        } else if (d.status === 'stopped') {
            es.close(); done();
            showToast('抓取已停止', 'warning');
            loadDbStats(false, true);  // force refresh stats after stop
            loadCrawlResults();
        } else if (d.status === 'error') {
            es.close(); done();
            setText('crawlProgressText', '错误: ' + (d.error || ''));
            showToast('抓取失败', 'error');
        }
    };
    es.onerror = function () {
        es.close(); done();
    };
}

function getBankModes() {
    /** 读取每个被选中银行的增量/全量模式 */
    const modes = {};
    document.querySelectorAll('#crawlBankList input[type=checkbox]:checked').forEach(cb => {
        const key = cb.dataset.key;
        const sel = document.querySelector(`.bank-mode-select[data-bank="${key}"]`);
        modes[key] = sel && sel.value === 'full';  // true=全量, false=增量
    });
    return modes;
}

function runCrawl(forceFullAll) {
    const banks = getSelectedBanks();
    if (banks.length === 0) {
        showToast('请至少选择一家银行', 'warning');
        return;
    }

    _showCrawlRunning();
    hide('crawlResultWrap');
    setText('crawlProgressText', '启动中...');
    setText('crawlElapsed', '');
    const bpList = document.getElementById('crawlBankProgressList');
    if (bpList) bpList.innerHTML = '';

    const crawlStart = Date.now();

    // forceFullAll=true: 全量按钮覆盖所有银行为全量
    // forceFullAll=false/undefined: 按各银行各自的模式选择
    const bank_modes = forceFullAll ? null : getBankModes();
    const full = !!forceFullAll;

    postJSON('/api/crawl/run', { banks, full, bank_modes }).then(() => {
        _subscribeCrawlSSE(crawlStart);
    });
}

function stopCrawl() {
    const btnStop = document.getElementById('btnCrawlStop');
    if (btnStop) btnStop.disabled = true;
    postJSON('/api/crawl/stop', {}).then(res => {
        if (btnStop) btnStop.disabled = false;
    });
}

function stopBank(bankKey) {
    postJSON('/api/crawl/stop', { bank: bankKey });
}

// 页面加载时检查是否有正在运行的抓取，恢复进度显示
function checkCrawlOnLoad() {
    fetch('/api/crawl/status', { headers: { 'Accept': 'text/event-stream' } })
        .catch(() => {});
    // 用一次性 fetch 检查状态
    fetch('/api/crawl/status').then(r => {
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        reader.read().then(function process({ done: rdone, value }) {
            if (rdone) return;
            const text = decoder.decode(value, { stream: true });
            const lines = text.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const d = JSON.parse(line.slice(6));
                        if (d.status === 'running') {
                            _showCrawlRunning();
                            _updateCrawlProgress(d);
                            // 计算已用时
                            if (d.started_at) {
                                const startMs = new Date(d.started_at).getTime();
                                const sec = Math.floor((Date.now() - startMs) / 1000);
                                const m = Math.floor(sec / 60), s = sec % 60;
                                setText('crawlElapsed', `已用时 ${m}:${String(s).padStart(2, '0')}`);
                            }
                            reader.cancel();
                            _subscribeCrawlSSE(d.started_at ? new Date(d.started_at).getTime() : Date.now());
                            return;
                        }
                    } catch (e) {}
                }
            }
            reader.cancel();
        });
    }).catch(() => {});
}

async function loadCrawlResults() {
    const data = await fetchJSON('/api/crawl/results');
    if (!data || data.length === 0) return;

    show('crawlResultWrap');
    let html = '<table><thead><tr>';
    html += '<th>银行</th><th>状态</th><th>产品数</th><th>新日期数</th><th>更新单元格</th><th>错误</th>';
    html += '</tr></thead><tbody>';

    data.forEach(r => {
        html += '<tr>';
        html += `<td>${r.bank || ''}</td>`;
        html += `<td>${r.success ? tagCell('成功', 'buy') : tagCell('失败', 'sell')}</td>`;
        html += `<td class="td-num">${r.products || '--'}</td>`;
        html += `<td class="td-num">${r.new_dates || '--'}</td>`;
        html += `<td class="td-num">${r.updated_cells || '--'}</td>`;
        html += `<td>${r.error || ''}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('crawlResultTable', html);
}

// ══════════════════════════════════════════
// GPU 状态
// ══════════════════════════════════════════

async function loadGpuInfo() {
    const data = await fetchJSON('/api/gpu/info');
    if (!data) return;

    const dot = document.getElementById('gpuStatusDot');
    const text = document.getElementById('gpuStatusText');
    if (data.available) {
        if (dot) dot.className = 'status-dot status-done';
        if (text) text.textContent = data.device_name || 'GPU加速';
    } else if (data.torch_installed) {
        if (dot) dot.className = 'status-dot status-idle';
        if (text) text.textContent = 'CPU (PyTorch)';
    } else {
        if (dot) dot.className = 'status-dot status-error';
        if (text) text.textContent = '仅CPU';
    }
}

// ══════════════════════════════════════════
// 蒙特卡洛
// ══════════════════════════════════════════

async function runMonteCarlo() {
    const btn = document.getElementById('btnRunMC');
    if (btn) btn.disabled = true;
    setText('mcSimCount', '模拟中...');
    showToast('蒙特卡洛模拟运行中...', 'info');

    const result = await postJSON('/api/gpu/monte-carlo', {});
    if (btn) btn.disabled = false;

    if (!result || !result.ok) {
        showToast('蒙特卡洛模拟失败: ' + (result?.error || ''), 'error');
        return;
    }

    show('mcKpiRow');
    show('mcChartsRow');

    setText('mcVar95', fmtPct(result.var_95));
    setText('mcCvar95', fmtPct(result.cvar_95));
    setText('mcExpRet', fmtPct(result.expected_return));
    setText('mcSimCount', `${(result.n_simulations || 0).toLocaleString()} 条路径`);

    // 收益分布直方图
    renderReturnDistribution(result.return_distribution);

    // 最优仓位表
    renderMcWeights(result.optimal_weights, result.position_sizing);

    showToast('蒙特卡洛模拟完成', 'success');
}

function renderReturnDistribution(dist) {
    const chart = getChart('chartMcDist');
    if (!chart || !dist || !dist.counts) return;

    const labels = [];
    for (let i = 0; i < dist.bin_edges.length - 1; i++) {
        labels.push(((dist.bin_edges[i] + dist.bin_edges[i + 1]) / 2).toFixed(2) + '%');
    }

    chart.setOption({
        ...CHART_BASE,
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: labels,
            axisLabel: { color: '#b2b5be', fontSize: 9, rotate: 45 },
            axisLine: { lineStyle: { color: '#2a2e39' } },
        },
        yAxis: {
            type: 'value',
            name: '频次',
            axisLabel: { color: '#787b86' },
            splitLine: { lineStyle: { color: '#22262f' } },
        },
        series: [{
            type: 'bar',
            data: dist.counts.map((v, i) => ({
                value: v,
                itemStyle: {
                    color: dist.bin_edges[i] >= 0 ? '#26a69a' : '#ef5350',
                },
            })),
            barWidth: '90%',
        }],
    });
}

function renderMcWeights(weights, sizing) {
    if (!weights || Object.keys(weights).length === 0) {
        setHTML('mcWeightsTable', '<div class="empty-state"><p>无仓位建议</p></div>');
        return;
    }

    let html = '<table><thead><tr><th>产品</th><th>权重</th><th>建议金额</th></tr></thead><tbody>';
    const entries = Object.entries(weights).sort((a, b) => b[1] - a[1]);
    entries.forEach(([key, w]) => {
        const amount = sizing ? sizing[key] : 0;
        html += '<tr>';
        html += `<td class="text-mono">${key}</td>`;
        html += `<td class="td-num">${(w * 100).toFixed(1)}%</td>`;
        html += `<td class="td-num">${fmtMoney(amount)}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('mcWeightsTable', html);
}

// ══════════════════════════════════════════
// 参数寻优
// ══════════════════════════════════════════

function runParamSweep() {
    const btn = document.getElementById('btnRunSweep');
    if (btn) btn.disabled = true;
    show('btProgressWrap');
    setText('btProgressText', '启动参数寻优...');

    postJSON('/api/backtest/param-sweep', {}).then(() => {
        const es = new EventSource('/api/backtest/param-sweep/status');
        es.onmessage = function (e) {
            const d = JSON.parse(e.data);
            document.getElementById('btProgressBar').style.width = d.progress + '%';
            setText('btProgressText', d.message || '');
            if (d.status === 'done') {
                es.close();
                if (btn) btn.disabled = false;
                showToast('参数寻优完成', 'success');
                loadSweepResults();
            } else if (d.status === 'error') {
                es.close();
                if (btn) btn.disabled = false;
                setText('btProgressText', '错误: ' + (d.error || ''));
                showToast('参数寻优失败', 'error');
            }
        };
        es.onerror = function () {
            es.close();
            if (btn) btn.disabled = false;
        };
    });
}

async function loadSweepResults() {
    const data = await fetchJSON('/api/backtest/param-sweep/results');
    if (!data || !data.best_params) return;

    show('sweepResultsSection');

    // KPI
    setText('swBestSharpe', fmt(data.best_sharpe, 2));
    setText('swBestReturn', fmtPct(data.best_return));
    setText('swBestWinRate', fmtPct(data.best_win_rate));
    const stab = data.robustness ? data.robustness.stability_score : 0;
    setText('swStability', fmt(stab, 3));

    // 最优参数表
    renderBestParams(data.best_params);

    // 热力图
    if (data.heatmaps) {
        renderParamHeatmap('chartSweepSharpe', data.heatmaps, 'sharpe', '夏普比率');
        renderParamHeatmap('chartSweepReturn', data.heatmaps, 'return', '年化收益%');
    }

    // Top10 鲁棒性
    if (data.robustness && data.robustness.top10_params) {
        renderRobustnessTable(data.robustness.top10_params);
    }
}

function renderBestParams(params) {
    if (!params) return;
    const nameMap = {
        buy_threshold: '买入阈值%',
        sell_threshold: '卖出阈值%',
        max_hold_days: '最大持有天数',
        hold_eval_days: '评估天数',
        success_criterion: '成功标准%',
        min_success_rate: '最低成功率%',
    };
    let html = '<table><thead><tr><th>参数</th><th>最优值</th></tr></thead><tbody>';
    Object.entries(params).forEach(([k, v]) => {
        html += `<tr><td>${nameMap[k] || k}</td><td class="td-num">${v}</td></tr>`;
    });
    html += '</tbody></table>';
    setHTML('swBestParamsTable', html);
}

function renderParamHeatmap(chartId, heatmaps, field, title) {
    const chart = getChart(chartId);
    if (!chart) return;

    const buyThresholds = heatmaps.buy_thresholds || [];
    const sellThresholds = heatmaps.sell_thresholds || [];
    const matrix = heatmaps[field] || [];

    const data = [];
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < (matrix[i] || []).length; j++) {
            const v = matrix[i][j] || 0;
            data.push([j, i, v]);
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }

    chart.setOption({
        ...CHART_BASE,
        tooltip: {
            position: 'top',
            formatter: function (p) {
                return `买入: ${buyThresholds[p.value[1]]}%<br/>卖出: ${sellThresholds[p.value[0]]}%<br/>${title}: ${p.value[2]}`;
            },
        },
        xAxis: {
            type: 'category',
            data: sellThresholds.map(v => v + '%'),
            name: '卖出阈值',
            nameTextStyle: { color: '#787b86' },
            axisLabel: { color: '#b2b5be', fontSize: 12 },
            splitArea: { show: true },
        },
        yAxis: {
            type: 'category',
            data: buyThresholds.map(v => v + '%'),
            name: '买入阈值',
            nameTextStyle: { color: '#787b86' },
            axisLabel: { color: '#b2b5be', fontSize: 12 },
            splitArea: { show: true },
        },
        visualMap: {
            min: minVal,
            max: maxVal,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: 0,
            inRange: {
                color: ['#ef5350', '#ff9800', '#ffd54f', '#26a69a', '#2196f3'],
            },
            textStyle: { color: '#787b86' },
        },
        series: [{
            type: 'heatmap',
            data: data,
            label: {
                show: data.length <= 81,
                fontSize: 9,
                color: '#b2b5be',
            },
            emphasis: {
                itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' },
            },
        }],
    });
}

function renderRobustnessTable(top10) {
    if (!top10 || top10.length === 0) return;

    const nameMap = {
        buy_threshold: '买入%',
        sell_threshold: '卖出%',
        max_hold_days: '最大持有',
        hold_eval_days: '评估天数',
        success_criterion: '成功标准',
        min_success_rate: '最低成功率',
        sharpe: '夏普',
        ann_return: '年化%',
        max_drawdown: '最大回撤%',
        win_rate: '胜率%',
    };

    const cols = Object.keys(top10[0]);
    let html = '<table><thead><tr><th>#</th>';
    cols.forEach(c => { html += `<th>${nameMap[c] || c}</th>`; });
    html += '</tr></thead><tbody>';

    top10.forEach((row, idx) => {
        html += `<tr${idx === 0 ? ' style="background:rgba(52,199,89,0.08)"' : ''}>`;
        html += `<td class="td-num">${idx + 1}</td>`;
        cols.forEach(c => {
            const v = row[c];
            html += `<td class="td-num">${typeof v === 'number' ? fmt(v, 2) : (v || '--')}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('swTop10Table', html);
}

// ══════════════════════════════════════════
// 关联分析
// ══════════════════════════════════════════

async function runCorrelation() {
    const btn = document.getElementById('btnRunCorrelation');
    if (btn) btn.disabled = true;
    showToast('关联分析运行中...', 'info');

    const result = await postJSON('/api/gpu/correlation/run', {});
    if (btn) btn.disabled = false;

    if (!result || !result.ok) {
        showToast('关联分析失败: ' + (result?.error || ''), 'error');
        return;
    }

    show('corrResultSection');

    setText('corrDivScore', fmt(result.diversification_score, 3));
    setText('corrNClusters', result.n_clusters || 0);
    setText('corrSysRisk', fmtPct(result.systematic_risk_pct));
    setText('corrNProducts', result.n_products || 0);

    // 相关性矩阵热力图
    renderCorrelationMatrix(result.correlation_matrix, result.product_labels);

    // 聚类分布图
    renderClusterChart(result.clusters);

    // 配置建议
    renderCorrRecommendations(result.recommendations);

    showToast('关联分析完成', 'success');
}

function renderCorrelationMatrix(matrix, labels) {
    const chart = getChart('chartCorrMatrix');
    if (!chart || !matrix || matrix.length === 0) return;

    const n = matrix.length;
    const data = [];
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            data.push([j, i, matrix[i][j]]);
        }
    }

    // 截取标签（太长时只取前8字符）
    const shortLabels = labels.map(l => l.length > 10 ? l.substring(0, 10) : l);

    chart.setOption({
        ...CHART_BASE,
        grid: { top: 40, right: 80, bottom: 80, left: 80 },
        tooltip: {
            position: 'top',
            formatter: function (p) {
                return `${shortLabels[p.value[1]]} vs ${shortLabels[p.value[0]]}<br/>相关: ${p.value[2].toFixed(3)}`;
            },
        },
        xAxis: {
            type: 'category',
            data: shortLabels,
            axisLabel: { rotate: 90, fontSize: 8, color: '#b2b5be' },
            splitArea: { show: true },
        },
        yAxis: {
            type: 'category',
            data: shortLabels,
            axisLabel: { fontSize: 8, color: '#b2b5be' },
            splitArea: { show: true },
        },
        visualMap: {
            min: -1,
            max: 1,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: 0,
            inRange: { color: ['#2196f3', '#1e222d', '#ef5350'] },
            textStyle: { color: '#787b86' },
        },
        series: [{
            type: 'heatmap',
            data: data,
            label: { show: n <= 20, fontSize: 7 },
        }],
    });
}

function renderClusterChart(clusters) {
    const chart = getChart('chartCluster');
    if (!chart || !clusters || clusters.length === 0) return;

    chart.setOption({
        ...CHART_BASE,
        color: APPLE_COLORS,
        tooltip: {
            trigger: 'item',
            formatter: function (p) {
                return `群组 ${p.name}<br/>产品数: ${p.value}<br/>平均相关: ${(p.data.avgCorr || 0).toFixed(3)}`;
            },
        },
        series: [{
            type: 'pie',
            radius: ['30%', '65%'],
            center: ['50%', '50%'],
            label: {
                color: '#b2b5be',
                fontSize: 13,
                formatter: '{b}\n{c}个',
            },
            data: clusters.map(c => ({
                name: `群组${c.id}`,
                value: c.size,
                avgCorr: c.avg_corr,
            })),
            itemStyle: { borderRadius: 4, borderColor: '#131722', borderWidth: 2 },
        }],
    });
}

function renderCorrRecommendations(recs) {
    if (!recs || recs.length === 0) {
        setHTML('corrRecommTable', '<div class="empty-state"><p>暂无建议</p></div>');
        return;
    }

    let html = '<table><thead><tr><th>建议</th><th>涉及产品</th><th>原因</th></tr></thead><tbody>';
    recs.forEach(r => {
        const actionTag = r.action === '增配' ? tagCell('增配', 'buy')
            : r.action === '精选' ? tagCell('精选', 'watch')
            : tagCell(r.action || '', 'sell');
        html += '<tr>';
        html += `<td>${actionTag}</td>`;
        html += `<td class="text-mono">${(r.products || []).join(', ')}</td>`;
        html += `<td>${r.reason || ''}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    setHTML('corrRecommTable', html);
}

// ══════════════════════════════════════════
// AI 预测热力图
// ══════════════════════════════════════════

function renderPredictionHeatmap(predictions) {
    const chart = getChart('chartPredHeatmap');
    if (!chart || !predictions) return;

    const entries = Object.entries(predictions);
    if (entries.length === 0) return;

    show('aiPredHeatmapCard');

    // 取 Top 20 产品
    entries.sort((a, b) => (b[1].score || 0) - (a[1].score || 0));
    const top = entries.slice(0, 20);
    const products = top.map(([k]) => k.length > 12 ? k.substring(0, 12) : k);
    const days = ['D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6', 'D+7', 'D+8', 'D+9', 'D+10'];

    const data = [];
    top.forEach(([, pred], pi) => {
        const probs = pred.release_probs || [];
        probs.forEach((p, di) => {
            data.push([di, pi, Number((p * 100).toFixed(1))]);
        });
    });

    chart.setOption({
        ...CHART_BASE,
        grid: { top: 20, right: 80, bottom: 60, left: 120 },
        tooltip: {
            formatter: function (p) {
                return `${products[p.value[1]]} ${days[p.value[0]]}<br/>释放概率: ${p.value[2]}%`;
            },
        },
        xAxis: {
            type: 'category',
            data: days,
            axisLabel: { color: '#787b86' },
        },
        yAxis: {
            type: 'category',
            data: products,
            axisLabel: { color: '#b2b5be', fontSize: 12 },
        },
        visualMap: {
            min: 0,
            max: 100,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: 0,
            inRange: { color: ['#1e222d', '#ff9800', '#ef5350', '#ef5350'] },
            textStyle: { color: '#787b86' },
        },
        series: [{
            type: 'heatmap',
            data: data,
            label: { show: data.length <= 200, fontSize: 9 },
        }],
    });
}

// ══════════════════════════════════════════
// 刷新全部
// ══════════════════════════════════════════

async function refreshAll() {
    await Promise.all([
        loadOverview(),
        loadRecommendations(),
        loadSignals(),
        loadPatterns(),
        loadGpuInfo(),
        loadHoldingReturns(),
    ]);
}

// ══════════════════════════════════════════
// 初始化
// ══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    // 设置默认日期
    const today = new Date().toISOString().slice(0, 10);
    const dateInput = document.getElementById('tradeDate');
    if (dateInput) dateInput.value = today;

    // 动态加载银行列表
    loadBankList();
    const toggleBtn = document.getElementById('toggleAllBanks');
    if (toggleBtn) toggleBtn.addEventListener('click', toggleAllBanks);

    // 检查是否有正在运行的抓取（页面刷新后恢复）
    checkCrawlOnLoad();

    try {
        await refreshAll();
        // 尝试加载回测
        const btResults = await fetchJSON('/api/backtest/results');
        if (btResults && btResults.total_trades) {
            await loadBacktestResults();
        }
    } catch (e) {
        console.log('初始加载 — 暂无数据');
    }
});
