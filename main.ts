// Token Launch Radar (radar-5): reads the snapshot, fits the value model on
// traded protocols, ranks pre-token protocols by fitted implied value, and
// writes the board, the dashboard and the dossier.
//
// Usage: MAKEFUN_OUTPUT_DIR=out node main.ts --input input/snapshot.json
// Defaults: input /input/snapshot.json, output /output (or $MAKEFUN_OUTPUT_DIR).

import { readFileSync, writeFileSync, mkdirSync, cpSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDossier } from './vendor/makefun-board.ts';
import {
  fitModel, predict, rankIntervals, farmPlan, driverNote, ageYears,
  STEP_TABLE, type Snapshot, type Proto,
} from './lib/model.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const BOARD_ROWS = 200;

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1]! : fallback;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function main(): void {
  const input = arg('--input', '/input/snapshot.json');
  const outDir = arg('--output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');
  const snap = JSON.parse(readFileSync(input, 'utf8')) as Snapshot;
  const model = fitModel(snap);
  const watch = new Set(snap.watchlist);

  // Rank every protocol that records no token by fitted implied value.
  const pre = snap.protocols.filter((p) => p.mcap_usd === null);
  const scored = pre
    .map((p) => ({ p, value: predict(model, p, snap.as_of) }))
    .sort((a, b) => b.value - a.value);
  const intervals = rankIntervals(model, snap, scored.map((s) => s.p), 50);

  // Take the top rows, forcing the whole watchlist in.
  let picked = scored.slice(0, BOARD_ROWS);
  const pickedSlugs = new Set(picked.map((s) => s.p.slug));
  for (const s of scored) {
    if (picked.length >= BOARD_ROWS && watch.has(s.p.slug) && !pickedSlugs.has(s.p.slug)) {
      // Drop the lowest-ranked non-watchlist row to make room.
      for (let i = picked.length - 1; i >= 0; i--) {
        if (!watch.has(picked[i]!.p.slug)) { picked.splice(i, 1); break; }
      }
      picked.push(s);
      pickedSlugs.add(s.p.slug);
    }
  }
  picked = picked.sort((a, b) => b.value - a.value);

  const cats = [...new Set(picked.map((s) => s.p.category ?? 'Uncategorised'))].sort();
  const chainSet = [...new Set(picked.flatMap((s) => s.p.chains))].sort();
  const shortDrivers = (d: string): string => {
    if (d.length <= 42) return d;
    const cut = d.slice(0, 42);
    const sp = cut.lastIndexOf(' ');
    return (sp > 20 ? cut.slice(0, sp) : cut) + '…';
  };
  const rows = picked.map((s, i) => {
    const plan = farmPlan(s.p, s.value);
    const [lo, hi] = intervals.get(s.p.slug) ?? [i + 1, i + 1];
    const hours = round2(plan.steps.length * 0.4 + 1.5);
    return {
      slug: s.p.slug,
      name: s.p.name,
      rank: i + 1,
      implied_value_usd: Math.max(1, Math.round(s.value)),
      farm_steps: plan.steps,
      // Compact row for the dashboard: category and chains are indexes into
      // the cats/chains tables, verdict is a code, drivers are shortened.
      _row: [
        s.p.slug, s.p.name, cats.indexOf(s.p.category ?? 'Uncategorised'),
        s.p.chains.map((c) => chainSet.indexOf(c)),
        s.p.tvl_usd, s.p.change_7d === null ? null : round2(s.p.change_7d),
        Math.round(ageYears(s.p, snap.as_of, model.medianAge) * 10) / 10,
        Math.max(1, Math.round(s.value)), lo, hi,
        plan.verdict === 'worth it' ? 0 : plan.verdict === 'thin' ? 1 : 2,
        plan.netEv1k, plan.netEv10k, Math.max(1, Math.round(plan.breakEvenUsd)),
        shortDrivers(driverNote(model, s.p, snap.as_of)), watch.has(s.p.slug), hours,
      ] as unknown[],
    };
  });

  // Self-check the board against the gate before writing anything: ranks
  // exactly 1..N, implied values non-increasing with rank, full watchlist
  // coverage, and every farm plan 3-12 steps of 20-600 characters.
  const checkRows = rows.map((r) => ({
    slug: r.slug, name: r.name, rank: r.rank,
    implied_value_usd: r.implied_value_usd, farm_steps: r.farm_steps,
  }));
  const seen = new Set<string>();
  checkRows.forEach((r, i) => {
    if (r.rank !== i + 1) throw new Error(`board: rank ${r.rank} at position ${i}`);
    if (seen.has(r.slug)) throw new Error(`board: duplicate slug ${r.slug}`);
    seen.add(r.slug);
    if (i > 0 && r.implied_value_usd > checkRows[i - 1]!.implied_value_usd)
      throw new Error(`board: implied value rises at rank ${r.rank}`);
    if (!Number.isInteger(r.implied_value_usd) || r.implied_value_usd <= 0)
      throw new Error(`board: bad implied value for ${r.slug}`);
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12)
      throw new Error(`board: ${r.slug} has ${r.farm_steps.length} farm steps`);
    for (const s of r.farm_steps) {
      const t = s.trim();
      if (t.length < 20 || t.length > 600)
        throw new Error(`board: ${r.slug} farm step of ${t.length} chars`);
    }
  });
  for (const s of snap.watchlist) {
    if (!seen.has(s)) throw new Error(`board: watchlist slug ${s} missing`);
  }

  // Board file: only the gate's fields, in rank order.
  const board = {
    as_of: snap.as_of,
    model: 'log market cap on log TVL, growth vs median, log age, category and chain shifts; fitted on traded protocols',
    rows: rows.map((r) => ({
      slug: r.slug, name: r.name, rank: r.rank,
      implied_value_usd: r.implied_value_usd, farm_steps: r.farm_steps,
    })),
  };

  // Dashboard data: compact arrays, plus the fitted curve and the traded
  // scatter so the first screen shows the evidence.
  const traded = snap.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0);
  const scatter = traded.map((p) => [
    Math.round(Math.log10(Math.max(1, p.tvl_usd)) * 10) / 10,
    Math.round(Math.log10(p.mcap_usd!) * 10) / 10,
  ]);
  const curve: number[][] = [];
  const loTvl = Math.min(...traded.map((p) => Math.log10(Math.max(1, p.tvl_usd))));
  const hiTvl = Math.max(...traded.map((p) => Math.log10(Math.max(1, p.tvl_usd))));
  for (let k = 0; k <= 16; k++) {
    const lt = loTvl + ((hiTvl - loTvl) * k) / 16;
    const synth: Proto = {
      slug: 'curve', name: 'curve', category: null, chains: [],
      tvl_usd: Math.round(10 ** lt), change_7d: null, mcap_usd: null,
      listed_at: null, gecko_id: null, symbol: null,
    };
    curve.push([Math.round(lt * 10) / 10, Math.round(Math.log10(Math.max(1, predict(model, synth, snap.as_of))) * 10) / 10]);
  }
  const v = model.validation;
  const keptTerms = [
    'log TVL',
    ...(model.curveKept ? ['log-TVL squared'] : []),
    ...(model.growthKept ? ['7-day growth vs median'] : []),
    ...(model.ageKept ? ['log age'] : []),
    ...(model.catKept ? ['category shifts'] : []),
    ...(model.chainKept ? ['chain shifts'] : []),
  ].join(', ');
  const droppedTerms = [
    ...(!model.curveKept ? ['log-TVL squared'] : []),
    ...(!model.growthKept ? ['7-day growth'] : []),
    ...(!model.ageKept ? ['log age'] : []),
    ...(!model.catKept ? ['category shifts'] : []),
    ...(!model.chainKept ? ['chain shifts'] : []),
  ].join(', ');
  const method =
    `Fitted on ${v.nTraded} protocols whose token already trades: log market cap on ${keptTerms}. ` +
    `Dropped on held-out evidence: ${droppedTerms || 'none'}. ` +
    `Growth is measured against the traded median (${model.medianGrowth.toFixed(1)}%, clamped at +/-30 points). ` +
    `Every coefficient comes from this fit; no protocol name is used anywhere. ` +
    `Implied values use a smearing correction (residual sd ${v.residualSd.toFixed(2)} log pts). ` +
    `Rank intervals come from 50 bootstrap refits. Farm plans assume a launch distributes 6% of implied value to farmers, ` +
    `with the farmer count growing with TVL (crowding) and points scaling as stake^0.7 against a $2,500 median farmer; costs are network fees plus 2% of stake.`;
  const method2 =
    `Held-out evidence: time split Spearman ${v.timeSplit.spearman.toFixed(2)} (RMSE ${v.timeSplit.rmse.toFixed(2)}), ` +
    `5-fold CV RMSE ${v.kfold.rmseMean.toFixed(2)} (spread ${v.kfold.rmseSpread.toFixed(2)}), Spearman ${v.kfold.spearmanMean.toFixed(2)}, ` +
    `walk-forward RMSE by window ${v.walkForward.rmseByWindow.map((x) => x.toFixed(2)).join(', ')}. ` +
    `Each value term was also fitted alone and tested held-out; see the term table. Terms that did not earn their place were dropped and are marked so.`;

  const data = {
    asOf: snap.as_of,
    nTraded: v.nTraded,
    watchlist: snap.watchlist,
    rows: rows.map((r) => r._row),
    cats, chains: chainSet,
    scatter,
    curve,
    terms: model.terms.map((t) => [t.term, round2(t.coef), round2(t.heldOutRmse), round2(t.baselineRmse), t.kept]),
    validation: v,
    method,
    method2,
    templates: STEP_TABLE,
  };
  const dataJs = 'window.RADAR = ' + JSON.stringify(data) + ';\n';

  // Write outputs.
  mkdirSync(join(outDir, 'dashboard'), { recursive: true });
  mkdirSync(join(outDir, 'screenshots'), { recursive: true });
  writeFileSync(join(outDir, 'board.json'), JSON.stringify(board, null, 1) + '\n');
  writeFileSync(join(outDir, 'dashboard', 'data.js'), dataJs);
  for (const f of ['index.html', 'app.css', 'app.js']) {
    cpSync(join(HERE, 'dashboard', f), join(outDir, 'dashboard', f));
  }
  const shots = join(HERE, 'screenshots');
  if (existsSync(shots)) cpSync(shots, join(outDir, 'screenshots'), { recursive: true });
  const dashTotal = ['index.html', 'app.css', 'app.js', 'data.js']
    .reduce((s, f) => s + readFileSync(join(outDir, 'dashboard', f)).length, 0);
  if (dashTotal > 65536) throw new Error(`dashboard source ${dashTotal} bytes exceeds the 65536 cap`);
  writeFileSync(join(outDir, 'dossier.json'), buildDossier(outDir, { text_max_bytes: 65536 }));

  // Report.
  const dashBytes = ['index.html', 'app.css', 'app.js', 'data.js']
    .reduce((s, f) => s + Buffer.byteLength(readFileSync(join(outDir, 'dashboard', f), 'utf8'), 'utf8'), 0);
  console.log(`rows=${rows.length} watchlist_covered=${rows.filter((r) => watch.has(r.slug)).length}/40`);
  console.log(`time_split spearman=${v.timeSplit.spearman.toFixed(3)} rmse=${v.timeSplit.rmse.toFixed(3)} (train ${v.timeSplit.train}, test ${v.timeSplit.test})`);
  console.log(`kfold rmse_mean=${v.kfold.rmseMean.toFixed(3)} spread=${v.kfold.rmseSpread.toFixed(3)} spearman=${v.kfold.spearmanMean.toFixed(3)}`);
  console.log(`walk_forward windows=${v.walkForward.windows} rmse=${v.walkForward.rmseByWindow.map((x) => x.toFixed(2)).join(',')} spearman=${v.walkForward.spearmanByWindow.map((x) => x.toFixed(2)).join(',')}`);
  console.log(`terms: ${model.terms.map((t) => `${t.term} coef=${t.coef.toFixed(2)} heldout=${t.heldOutRmse.toFixed(2)} kept=${t.kept}`).join(' | ')}`);
  console.log(`dashboard_source_bytes=${dashBytes} (cap 65536)`);
  console.log(`top5=${rows.slice(0, 5).map((r) => `${r.slug}:$${r.implied_value_usd.toLocaleString('en-US')}`).join(', ')}`);
}

main();
