// Token Launch Radar dashboard: renders window.RADAR, no network.
// Row layout: [slug,name,catIdx,chainIdxs,tvl,chg,age,implied,lo,hi,verdictCode,ev1k,ev10k,breakeven,drivers,watch,hours]
const R = window.RADAR;
const VERDICTS = ['worth it', 'thin', 'skip'];
const catOf = (r) => R.cats[r[2]];
const chainsOf = (r) => r[3].map((i) => R.chains[i]).join(', ');
const verdictOf = (r) => VERDICTS[r[10]];
const $ = (id) => document.getElementById(id);
const fmt$ = (n) => n >= 1e9 ? '$' + (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? '$' + (n / 1e6).toFixed(1) + 'M' : '$' + Math.round(n).toLocaleString('en-US');
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const money = (n) => n >= 100 ? '$' + Math.round(n).toLocaleString('en-US') : '$' + n.toFixed(2);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const FEE = { Ethereum: 9, Solana: 0.05, Base: 0.15, Arbitrum: 0.3, Binance: 0.4, Polygon: 0.05, Optimism: 0.2, Avalanche: 0.3, 'Hyperliquid L1': 0.1, Monad: 0.05, Sei: 0.05, Berachain: 0.1, Sui: 0.05, Aptos: 0.05 };
const feeOf = (c) => FEE[c] ?? 0.5;

function farmDays(tvl) { return Math.max(45, Math.min(120, Math.round(30 + Math.log10(Math.max(1e6, tvl)) * 12))); }
function farmSteps(row) {
  const [slug, name] = row;
  const cat = catOf(row);
  const chain = chainsOf(row).split(',')[0] || 'Ethereum';
  const tvl = row[4];
  const fee = money(feeOf(chain));
  const days = farmDays(tvl);
  const tpls = R.templates[cat] || R.templates.__fallback;
  return tpls.map((t) => t.replaceAll('{NAME}', name).replaceAll('{CHAIN}', chain).replaceAll('{FEE}', fee).replaceAll('{CAP}', '$1,000').replaceAll('{DAYS}', String(days)).replaceAll('{CAT}', (cat || 'protocol').toLowerCase()));
}

// ---- header evidence ----
$('asof').textContent = R.asOf;
$('counts').textContent = R.nTraded + ' traded protocols fitted · ' + R.rows.length + ' pre-token rows · watchlist ' + R.watchlist.length;
const V = R.validation;
$('scorebox').innerHTML = '<b>Time-split held-out Spearman ' + V.timeSplit.spearman.toFixed(2) + '</b><span>5-fold CV RMSE ' + V.kfold.rmseMean.toFixed(2) + ' log pts (spread ' + V.kfold.rmseSpread.toFixed(2) + ') · walk-forward ' + V.walkForward.windows + ' windows, RMSE ' + V.walkForward.rmseByWindow.map((x) => x.toFixed(2)).join(', ') + '</span><span>Top implied value ' + fmt$(R.rows[0][7]) + ' · ' + esc(R.rows[0][1]) + '</span>';
$('validline').textContent = 'Time split (fit older ' + V.timeSplit.train + ', test newer ' + V.timeSplit.test + '): RMSE ' + V.timeSplit.rmse.toFixed(2) + ', Spearman ' + V.timeSplit.spearman.toFixed(2) + '. 5-fold CV Spearman ' + V.kfold.spearmanMean.toFixed(2) + '. Walk-forward Spearman by window: ' + V.walkForward.spearmanByWindow.map((x) => x.toFixed(2)).join(', ') + '.';

// ---- scatter ----
(function scatter() {
  const svg = $('scatter');
  const xs = R.scatter.map((p) => p[0]).concat(R.curve.map((p) => p[0]));
  const ys = R.scatter.map((p) => p[1]).concat(R.curve.map((p) => p[1]));
  const x0 = Math.min(...xs) - 0.2, x1 = Math.max(...xs) + 0.2, y0 = Math.min(...ys) - 0.2, y1 = Math.max(...ys) + 0.2;
  const X = (v) => 46 + (v - x0) / (x1 - x0) * 580;
  const Y = (v) => 272 - (v - y0) / (y1 - y0) * 244;
  let s = '';
  const tick$ = (v) => { const n = Math.pow(10, v); return n >= 1e9 ? '$' + (n / 1e9).toFixed(0) + 'B' : n >= 1e6 ? '$' + (n / 1e6).toFixed(0) + 'M' : '$' + Math.round(n).toLocaleString('en-US'); };
  for (let g = 0; g <= 4; g++) {
    const gx = x0 + (x1 - x0) * g / 4;
    s += '<line x1="' + X(gx) + '" y1="272" x2="' + X(gx) + '" y2="28" stroke="#263757" stroke-width="1"/><text x="' + X(gx) + '" y="292" fill="#9fb0c9" font-size="11" text-anchor="middle">' + tick$(gx) + '</text>';
  }
  for (let g = 0; g <= 4; g++) {
    const gy = y0 + (y1 - y0) * g / 4;
    s += '<text x="40" y="' + (Y(gy) + 4) + '" fill="#9fb0c9" font-size="11" text-anchor="end">' + tick$(gy) + '</text>';
  }
  s += '<polyline fill="none" stroke="#ffffff" stroke-width="2" points="' + R.curve.map((p) => X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join(' ') + '"/>';
  for (const p of R.scatter) s += '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="3.2" fill="#2dd4bf" opacity="0.75"/>';
  for (const r of R.rows) s += '<circle cx="' + X(Math.log10(Math.max(1, r[4]))).toFixed(1) + '" cy="' + Y(Math.log10(Math.max(1, r[7]))).toFixed(1) + '" r="2.4" fill="#f2b33d" opacity="0.85"/>';
  svg.innerHTML = s;
})();

// ---- term table ----
$('terms').querySelector('tbody').innerHTML = R.terms.map((t) =>
  '<tr><td>' + esc(t[0]) + '</td><td>' + (t[1] === 0 ? '—' : t[1].toFixed(2)) + '</td><td>' + t[2].toFixed(2) + ' vs ' + t[3].toFixed(2) + '</td><td>' + (t[4] ? 'kept' : 'dropped') + '</td></tr>').join('');

// ---- top farms ----
const farms = [...R.rows].filter((r) => r[11] > 0).sort((a, b) => (b[11] / b[16]) - (a[11] / a[16])).slice(0, 5);
$('topfarms').innerHTML = farms.map((r) => '<li><b>' + esc(r[1]) + '</b> — net ' + money(r[11]) + ' at $1k in ~' + r[16] + 'h (' + money(r[11] / r[16]) + '/h), implied ' + fmt$(r[7]) + '</li>').join('') || '<li>No farm clears costs at $1,000.</li>';

// ---- method ----
$('method').textContent = R.method;
$('method2').textContent = R.method2;

// ---- category filter ----
const cats = [...R.cats].sort();
$('cat').innerHTML = '<option value="all">All categories</option>' + cats.map((c) => '<option>' + esc(c) + '</option>').join('');

// ---- board ----
function visibleRows() {
  const q = $('q').value.trim().toLowerCase();
  const st = $('status').value, cat = $('cat').value, sort = $('sort').value;
  let rows = R.rows.filter((r) => {
    if (st === 'watch' && !r[15]) return false;
    if (st === 'worth' && verdictOf(r) !== 'worth it') return false;
    if (st === 'thin' && verdictOf(r) !== 'thin') return false;
    if (st === 'skip' && verdictOf(r) !== 'skip') return false;
    if (cat !== 'all' && catOf(r) !== cat) return false;
    if (q && !(r[1].toLowerCase().includes(q) || r[0].includes(q))) return false;
    return true;
  });
  const key = { rank: (r) => r[7], tvl: (r) => r[4], growth: (r) => r[5] ?? -999, ev: (r) => r[11] }[sort];
  rows = [...rows].sort((a, b) => key(b) - key(a));
  return rows;
}
function renderBoard() {
  const rows = visibleRows();
  $('rowcount').textContent = rows.length + ' of ' + R.rows.length + ' rows';
  $('board').querySelector('tbody').innerHTML = rows.map((r, i) => {
    const rank = R.rows.indexOf(r) + 1;
    return '<tr data-slug="' + esc(r[0]) + '"><td>' + rank + '</td><td><b>' + esc(r[1]) + '</b>' + (r[15] ? ' <span class="watch">● watchlist</span>' : '') + '<br><span style="color:#9fb0c9;font-size:12px">' + esc(r[0]) + '</span></td><td>' + esc(catOf(r)) + '</td><td>' + esc(chainsOf(r)) + '</td><td>' + fmt$(r[4]) + '</td><td>' + (r[5] === null ? '—' : (r[5] >= 0 ? '+' : '') + r[5].toFixed(1) + '%') + '</td><td>' + r[6].toFixed(1) + 'y</td><td class="val">' + fmt$(r[7]) + '</td><td>#' + r[8] + '–' + r[9] + '</td><td><span class="pill ' + (verdictOf(r) === 'worth it' ? 'worth' : verdictOf(r) === 'thin' ? 'thin' : 'skip') + '">' + verdictOf(r) + '</span><br><span style="color:#9fb0c9;font-size:12px">EV $1k ' + money(r[11]) + '</span></td><td style="font-size:12.5px;color:#c6d2e4">' + esc(r[14]) + '</td></tr>';
  }).join('');
  $('board').querySelectorAll('tbody tr').forEach((tr) => tr.addEventListener('click', () => { location.hash = 'p=' + tr.dataset.slug; }));
}

// ---- detail ----
function renderDetail() {
  const m = location.hash.match(/^#p=(.+)$/);
  const box = $('detail');
  if (!m) { box.hidden = true; box.innerHTML = ''; return; }
  const row = R.rows.find((r) => r[0] === m[1]);
  if (!row) { box.hidden = true; return; }
  const steps = farmSteps(row);
  const rank = R.rows.indexOf(row) + 1;
  const sens = [5, 10, 20].map((d) => {
    const v = row[7] * (1 - d / 100);
    const farmers = Math.max(500, Math.min(60000, Math.round(300 + row[4] / 200000)));
    const share = (s) => Math.pow(s, 0.7) / (farmers * Math.pow(2500, 0.7));
    const fees = Math.round(feeOf(chainsOf(row).split(',')[0] || 'Ethereum') * (steps.length + 2) * 100) / 100;
    const net = (s) => Math.round((0.06 * v * share(s) - (fees + 0.02 * s)) * 100) / 100;
    return '<tr><td>−' + d + '%</td><td>' + money(net(1000)) + '</td><td>' + money(net(10000)) + '</td></tr>';
  }).join('');
  box.hidden = false;
  box.innerHTML = '<h2>' + esc(row[1]) + ' <span style="color:#9fb0c9;font-size:14px">' + esc(row[0]) + '</span></h2>'
    + '<p>Rank <b>#' + rank + '</b> of ' + R.rows.length + ' · rank interval #' + row[8] + '–' + row[9] + ' from 50 bootstrap refits · ' + (row[15] ? 'on the watchlist · ' : '') + esc(catOf(row)) + ' · ' + esc(chainsOf(row)) + '</p>'
    + '<p class="big">' + fmt$(row[7]) + ' <span style="font-size:14px;color:#9fb0c9">implied value</span></p>'
    + '<p>TVL ' + fmt$(row[4]) + ' · 7-day growth ' + (row[5] === null ? 'not reported' : (row[5] >= 0 ? '+' : '') + row[5].toFixed(1) + '%') + ' · ' + row[6].toFixed(1) + ' years tracked with no token recorded.</p>'
    + '<h3>Why it ranks here</h3><p>' + esc(row[14]) + '</p>'
    + '<h3>Farm plan — verdict: ' + verdictOf(row) + ' (break-even implied value ' + fmt$(row[13]) + '; net EV $1k ' + money(row[11]) + ', $10k ' + money(row[12]) + ')</h3><ol>' + steps.map((s) => '<li>' + esc(s) + '</li>').join('') + '</ol>'
    + '<h3>Sensitivity — if the token lists below implied value</h3><table><thead><tr><th>List discount</th><th>Net EV at $1k</th><th>Net EV at $10k</th></tr></thead><tbody>' + sens + '</tbody></table>'
    + '<p class="hint"><a href="#">← Back to the board</a></p>';
  box.scrollIntoView();
}

function applyHash() {
  const h = location.hash;
  if (h === '#watch') { $('status').value = 'watch'; }
  else if (h === '#worth') { $('status').value = 'worth'; }
  else if (h.startsWith('#cat=')) { $('cat').value = decodeURIComponent(h.slice(5)); }
  renderBoard();
  renderDetail();
  if (h === '#watch' || h === '#worth' || h.startsWith('#cat=')) {
    document.getElementById('boardwrap').scrollIntoView();
  }
}
['q', 'status', 'cat', 'sort'].forEach((id) => $(id).addEventListener('input', renderBoard));
window.addEventListener('hashchange', applyHash);
applyHash();
