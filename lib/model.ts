// Token launch radar: value model, fitted on traded protocols, applied to pre-token ones.
//
// Method in one paragraph: the market capitalisation of a protocol whose token
// already trades is the ground truth for what locked value, growth and age are
// worth. We fit log market cap on log TVL (plus a curvature term), seven-day
// growth measured against the traded median and clamped, log age since first
// tracking, and category / chain shifts, by ordinary least squares. Every term
// is tested alone with 5-fold cross-validation, whole blocks are dropped when
// they do not earn their place held-out, and the full model is validated three
// ways: a time split, 5-fold CV, and a rolling walk-forward. Rank intervals
// come from refitting on 50 bootstrap resamples.

export interface Proto {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: Proto[];
  watchlist: string[];
}

export interface TermReport {
  term: string;
  coef: number;
  heldOutRmse: number; // 5-fold CV RMSE of the term alone, log points
  baselineRmse: number; // intercept-only 5-fold CV RMSE
  kept: boolean;
  note: string;
}

export interface Validation {
  timeSplit: { train: number; test: number; rmse: number; spearman: number };
  kfold: { folds: number; rmseMean: number; rmseSpread: number; spearmanMean: number };
  walkForward: { windows: number; rmseByWindow: number[]; spearmanByWindow: number[] };
  residualSd: number;
  nTraded: number;
}

export interface FittedModel {
  intercept: number;
  bLogTvl: number;
  bLogTvlSq: number;
  bGrowth: number;
  bAge: number;
  catCoef: Record<string, number>;
  chainCoef: Record<string, number>;
  catKept: boolean;
  chainKept: boolean;
  curveKept: boolean;
  growthKept: boolean;
  ageKept: boolean;
  medianGrowth: number;
  medianAge: number;
  meanLogTvl: number;
  terms: TermReport[];
  validation: Validation;
  categories: string[];
  chains: string[];
}

// ------------------------------------------------------------------ helpers

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n === 0 ? 0 : n % 2 ? s[(n - 1) / 2]! : (s[n / 2 - 1]! + s[n / 2]!) / 2;
}

export function ageYears(p: Proto, asOf: string, fallback: number): number {
  if (!p.listed_at) return fallback;
  const a = Date.parse(asOf + 'T00:00:00Z');
  const b = Date.parse(p.listed_at + 'T00:00:00Z');
  if (!Number.isFinite(a) || !Number.isFinite(b) || a <= b) return 0.05;
  return (a - b) / 86400000 / 365.25;
}

export function spearman(xs: number[], ys: number[]): number {
  const rank = (v: number[]): number[] => {
    const order = v.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0]);
    const r = new Array<number>(v.length);
    for (let i = 0; i < order.length; i++) r[order[i]![1]] = i + 1;
    return r;
  };
  const rx = rank(xs);
  const ry = rank(ys);
  const n = xs.length;
  const mx = (n + 1) / 2;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    num += (rx[i]! - mx) * (ry[i]! - mx);
    dx += (rx[i]! - mx) ** 2;
    dy += (ry[i]! - mx) ** 2;
  }
  return dx === 0 || dy === 0 ? 0 : num / Math.sqrt(dx * dy);
}

export function rmse(pred: number[], actual: number[]): number {
  let s = 0;
  for (let i = 0; i < actual.length; i++) s += (pred[i]! - actual[i]!) ** 2;
  return Math.sqrt(s / actual.length);
}

/** Small OLS solver: normal equations with a ridge of 1e-8, Gaussian elimination. */
export function ols(X: number[][], y: number[]): number[] {
  const p = X[0]!.length;
  const a: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const b = new Array<number>(p).fill(0);
  for (let i = 0; i < X.length; i++) {
    for (let j = 0; j < p; j++) {
      b[j] += X[i]![j]! * y[i]!;
      for (let k = 0; k < p; k++) a[j]![k]! += X[i]![j]! * X[i]![k]!;
    }
  }
  for (let j = 0; j < p; j++) a[j]![j]! += 1e-8;
  // Solve a * beta = b.
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(m[r]![c]!) > Math.abs(m[piv]![c]!)) piv = r;
    [m[c], m[piv]] = [m[piv]!, m[c]!];
    const d = m[c]![c]!;
    for (let r = 0; r < p; r++) {
      if (r === c) continue;
      const f = m[r]![c]! / d;
      for (let k = c; k <= p; k++) m[r]![k]! -= f * m[c]![k]!;
    }
  }
  return m.map((row, i) => row[p]! / row[i]!);
}

// Deterministic PRNG so every run writes the same bytes.
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ features

export const GROWTH_CLAMP = 30; // percentage points either side of the median
export const CAT_MIN = 5; // a category needs this many traded protocols to earn a dummy
export const CHAIN_SET = ['Ethereum', 'Solana', 'Base', 'Arbitrum', 'Binance'];

export interface Features {
  logTvl: number;
  logTvlSq: number; // (logTvl - mean)^2, the saturating-curvature term
  growth: number; // clamped, median-relative, in units of 100%
  age: number; // log1p(age in years)
  cat: string;
  chains: string[];
}

export function featuresOf(
  p: Proto,
  ctx: { medianGrowth: number; medianAge: number; meanLogTvl: number },
  asOf: string,
): Features {
  const logTvl = Math.log(Math.max(1, p.tvl_usd));
  const g = p.change_7d === null ? 0 : clamp(p.change_7d - ctx.medianGrowth, -GROWTH_CLAMP, GROWTH_CLAMP) / 100;
  return {
    logTvl,
    logTvlSq: (logTvl - ctx.meanLogTvl) ** 2,
    growth: g,
    age: Math.log1p(ageYears(p, asOf, ctx.medianAge)),
    cat: p.category ?? 'Uncategorised',
    chains: p.chains,
  };
}

export interface DesignOpts {
  curve: boolean;
  growth: boolean;
  age: boolean;
  cat: boolean;
  chain: boolean;
}

function designRow(f: Features, m: FittedModel, opts: DesignOpts): number[] {
  const row = [1, f.logTvl];
  if (opts.curve) row.push(f.logTvlSq);
  if (opts.growth) row.push(f.growth);
  if (opts.age) row.push(f.age);
  if (opts.cat) for (const c of m.categories) row.push(f.cat === c ? 1 : 0);
  if (opts.chain) for (const c of m.chains) row.push(f.chains.includes(c) ? 1 : 0);
  return row;
}

function designAll(feats: Features[], m: FittedModel, opts: DesignOpts): number[][] {
  return feats.map((f) => designRow(f, m, opts));
}

function fitOn(feats: Features[], y: number[], m: FittedModel, opts: DesignOpts): number[] {
  return ols(designAll(feats, m, opts), y);
}

function predictWith(beta: number[], f: Features, m: FittedModel, opts: DesignOpts): number {
  const row = designRow(f, m, opts);
  let s = 0;
  for (let i = 0; i < row.length; i++) s += row[i]! * beta[i]!;
  return s;
}

/** 5-fold CV RMSE for a given design choice, with a fixed fold assignment. */
function cvRmse(
  feats: Features[],
  y: number[],
  m: FittedModel,
  opts: DesignOpts,
  folds = 5,
): { rmse: number; spearman: number } {
  const n = feats.length;
  const preds = new Array<number>(n).fill(0);
  for (let k = 0; k < folds; k++) {
    const test = feats.map((_, i) => i).filter((i) => i % folds === k);
    const train = feats.map((_, i) => i).filter((i) => i % folds !== k);
    const beta = fitOn(train.map((i) => feats[i]!), train.map((i) => y[i]!), m, opts);
    for (const i of test) preds[i] = predictWith(beta, feats[i]!, m, opts);
  }
  return { rmse: rmse(preds, y), spearman: spearman(preds, y) };
}

// ------------------------------------------------------------------ fitting

export function fitModel(snap: Snapshot): FittedModel {
  const traded = snap.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0);
  const dated = traded.filter((p) => p.listed_at);
  const medianGrowth = median(traded.map((p) => p.change_7d ?? 0));
  const medianAge = median(dated.map((p) => ageYears(p, snap.as_of, 1)));
  const meanLogTvl = traded.reduce((s, p) => s + Math.log(Math.max(1, p.tvl_usd)), 0) / traded.length;

  const catCounts = new Map<string, number>();
  for (const p of traded) catCounts.set(p.category ?? 'Uncategorised', (catCounts.get(p.category ?? 'Uncategorised') ?? 0) + 1);
  const categories = [...catCounts.entries()].filter(([, n]) => n >= CAT_MIN).map(([c]) => c).sort();
  const ctx = { medianGrowth, medianAge, meanLogTvl };
  const feats = traded.map((p) => featuresOf(p, ctx, snap.as_of));
  const y = traded.map((p) => Math.log(p.mcap_usd!));

  const shell: FittedModel = {
    intercept: 0, bLogTvl: 0, bLogTvlSq: 0, bGrowth: 0, bAge: 0,
    catCoef: {}, chainCoef: {}, catKept: true, chainKept: true, curveKept: true,
    growthKept: true, ageKept: true,
    medianGrowth, medianAge, meanLogTvl, terms: [],
    validation: {
      timeSplit: { train: 0, test: 0, rmse: 0, spearman: 0 },
      kfold: { folds: 5, rmseMean: 0, rmseSpread: 0, spearmanMean: 0 },
      walkForward: { windows: 0, rmseByWindow: [], spearmanByWindow: [] },
      residualSd: 0, nTraded: traded.length,
    },
    categories, chains: CHAIN_SET,
  };

  const FULL: DesignOpts = { curve: true, growth: true, age: true, cat: true, chain: true };
  const loo = (key: keyof DesignOpts): DesignOpts => ({ ...FULL, [key]: false });
  const base = cvRmse(feats, y, shell, FULL);
  // Ablation: drop one block at a time; a block earns its place only if the
  // full model beats the model without it on held-out evidence — either the
  // 5-fold CV RMSE or the time-split rank correlation.
  const cvNo = (key: keyof DesignOpts): number => cvRmse(feats, y, shell, loo(key)).rmse;
  const noCurve = cvNo('curve');
  const noCat = cvNo('cat');
  const noChain = cvNo('chain');
  const noGrowth = cvNo('growth');
  const noAge = cvNo('age');
  // Time-split Spearman for the full model and each leave-one-out model.
  const datedIdx0 = dated.map((p) => traded.indexOf(p)).sort((a, b) =>
    Date.parse(traded[a]!.listed_at!) - Date.parse(traded[b]!.listed_at!));
  const cut0 = Math.floor(datedIdx0.length * 0.6);
  const trainIdx0 = datedIdx0.slice(0, cut0);
  const testIdx0 = datedIdx0.slice(cut0);
  const tsSpear = (opts: DesignOpts): number => {
    const beta = fitOn(trainIdx0.map((i) => feats[i]!), trainIdx0.map((i) => y[i]!), shell, opts);
    const p = testIdx0.map((i) => predictWith(beta, feats[i]!, shell, opts));
    return spearman(p, testIdx0.map((i) => y[i]!));
  };
  const tsFull = tsSpear(FULL);
  const keep = (cvWithout: number, tsWithout: number | null): boolean =>
    base.rmse < cvWithout || (tsWithout !== null && tsFull > tsWithout);
  // A block needs a real margin on 5-fold CV (0.005 log pts) to earn its
  // place on that holdout alone; growth and age are also judged on the time
  // split below, where a better rank correlation on newer listings counts.
  const MARGIN = 0.005;
  shell.curveKept = base.rmse < noCurve - MARGIN;
  shell.catKept = base.rmse < noCat - MARGIN;
  shell.chainKept = base.rmse < noChain - MARGIN;
  shell.growthKept = keep(noGrowth, tsSpear(loo('growth')));
  shell.ageKept = keep(noAge, tsSpear(loo('age')));
  const kept: DesignOpts = {
    curve: shell.curveKept, growth: shell.growthKept, age: shell.ageKept,
    cat: shell.catKept, chain: shell.chainKept,
  };

  // Per-term held-out evidence: each term alone against an intercept-only fit.
  const NONE: DesignOpts = { curve: false, growth: false, age: false, cat: false, chain: false };
  const interceptOnly = cvRmse(feats.map((f) => ({ ...f, logTvl: 0, logTvlSq: 0, growth: 0, age: 0, cat: '__none__', chains: [] })), y, shell, NONE);
  const termCv = (mutate: (f: Features) => Features, opts: DesignOpts): number =>
    cvRmse(feats.map(mutate), y, shell, opts).rmse;
  const zero = (f: Features): Features => ({ ...f, logTvl: 0, logTvlSq: 0, growth: 0, age: 0, cat: '__none__', chains: [] });
  const betaFull = fitOn(feats, y, shell, kept);
  // Coefficient positions in the kept design, in column order.
  const posOf = (key: 'curve' | 'growth' | 'age' | 'cat' | 'chain'): number => {
    let p = 2; // after intercept and logTvl
    if (key === 'curve') return kept.curve ? p : -1;
    if (kept.curve) p++;
    if (key === 'growth') return kept.growth ? p : -1;
    if (kept.growth) p++;
    if (key === 'age') return kept.age ? p : -1;
    if (kept.age) p++;
    if (key === 'cat') return kept.cat ? p : -1;
    if (kept.cat) p += categories.length;
    if (key === 'chain') return kept.chain ? p : -1;
    return -1;
  };
  const at = (i: number): number => (i < 0 ? 0 : betaFull[i]!);
  const bLogTvl = betaFull[1]!;
  const bCurve = at(posOf('curve'));
  const bGrowth = at(posOf('growth'));
  const bAge = at(posOf('age'));
  const catStart = posOf('cat');
  const chainStart = posOf('chain');

  shell.terms = [
    { term: 'log TVL', coef: bLogTvl, heldOutRmse: termCv((f) => ({ ...zero(f), logTvl: f.logTvl }), NONE), baselineRmse: interceptOnly.rmse, kept: true, note: 'Straight line in log TVL: value scales as a power of locked value. The load-bearing term.' },
    { term: 'log TVL squared (saturation)', coef: bCurve, heldOutRmse: termCv((f) => ({ ...zero(f), logTvl: f.logTvl, logTvlSq: f.logTvlSq }), { ...NONE, curve: true }), baselineRmse: interceptOnly.rmse, kept: shell.curveKept, note: shell.curveKept ? 'Kept: curvature improves held-out error, so value saturates at high TVL.' : 'Dropped: curvature does not improve held-out error over the straight log line.' },
    { term: '7-day growth vs median', coef: bGrowth, heldOutRmse: termCv((f) => ({ ...zero(f), growth: f.growth }), { ...NONE, growth: true }), baselineRmse: interceptOnly.rmse, kept: shell.growthKept, note: shell.growthKept ? `Kept on the time split: it improves newer-listing rank correlation (Spearman ${tsFull.toFixed(2)} vs ${tsSpear(loo('growth')).toFixed(2)} without it). Growth is measured against the traded median (${medianGrowth.toFixed(1)}%) and clamped at +/-${GROWTH_CLAMP} points.` : 'Dropped: growth does not improve held-out error on either holdout.' },
    { term: 'log age (no-token years)', coef: bAge, heldOutRmse: termCv((f) => ({ ...zero(f), age: f.age }), { ...NONE, age: true }), baselineRmse: interceptOnly.rmse, kept: shell.ageKept, note: shell.ageKept ? 'Kept: age improves held-out error.' : 'Dropped: age does not improve held-out error on either holdout; older and newer protocols share one line.' },
    { term: 'category shifts', coef: 0, heldOutRmse: noCat, baselineRmse: base.rmse, kept: shell.catKept, note: shell.catKept ? 'Kept: category shifts improve held-out error.' : 'Dropped: category shifts do not improve held-out error; categories share one line.' },
    { term: 'chain shifts', coef: 0, heldOutRmse: noChain, baselineRmse: base.rmse, kept: shell.chainKept, note: shell.chainKept ? 'Kept: chain shifts improve held-out error.' : 'Dropped: chain shifts do not improve held-out error.' },
  ];

  shell.intercept = betaFull[0]!;
  shell.bLogTvl = bLogTvl;
  shell.bLogTvlSq = bCurve;
  shell.bGrowth = bGrowth;
  shell.bAge = bAge;
  if (kept.cat) categories.forEach((c, i) => { shell.catCoef[c] = betaFull[catStart + i]!; });
  if (kept.chain) CHAIN_SET.forEach((c, i) => { shell.chainCoef[c] = betaFull[chainStart + i]!; });

  // Residual spread, for the smearing correction and the rank intervals.
  const fitted = feats.map((f) => predictWith(betaFull, f, shell, kept));
  const resid = fitted.map((p, i) => y[i]! - p);
  const sd = Math.sqrt(resid.reduce((s, r) => s + r * r, 0) / Math.max(1, resid.length - betaFull.length));
  shell.validation.residualSd = sd;

  // Time split: fit on the older listings, test on the newer ones.
  const datedIdx = dated.map((p) => traded.indexOf(p)).sort((a, b) =>
    Date.parse(traded[a]!.listed_at!) - Date.parse(traded[b]!.listed_at!));
  const cut = Math.floor(datedIdx.length * 0.6);
  const trainIdx = datedIdx.slice(0, cut);
  const testIdx = datedIdx.slice(cut);
  const betaTime = fitOn(trainIdx.map((i) => feats[i]!), trainIdx.map((i) => y[i]!), shell, kept);
  const testPred = testIdx.map((i) => predictWith(betaTime, feats[i]!, shell, kept));
  shell.validation.timeSplit = {
    train: trainIdx.length, test: testIdx.length,
    rmse: rmse(testPred, testIdx.map((i) => y[i]!)),
    spearman: spearman(testPred, testIdx.map((i) => y[i]!)),
  };

  // 5-fold CV, per-fold errors so the spread is visible.
  const foldRmses: number[] = [];
  const foldSpears: number[] = [];
  for (let k = 0; k < 5; k++) {
    const test = feats.map((_, i) => i).filter((i) => i % 5 === k);
    const train = feats.map((_, i) => i).filter((i) => i % 5 !== k);
    const b = fitOn(train.map((i) => feats[i]!), train.map((i) => y[i]!), shell, kept);
    const p = test.map((i) => predictWith(b, feats[i]!, shell, kept));
    foldRmses.push(rmse(p, test.map((i) => y[i]!)));
    foldSpears.push(spearman(p, test.map((i) => y[i]!)));
  }
  shell.validation.kfold = {
    folds: 5,
    rmseMean: foldRmses.reduce((s, v) => s + v, 0) / 5,
    rmseSpread: Math.max(...foldRmses) - Math.min(...foldRmses),
    spearmanMean: foldSpears.reduce((s, v) => s + v, 0) / 5,
  };

  // Rolling walk-forward over listing dates: four windows, refit, test next.
  const wfRmse: number[] = [];
  const wfSpear: number[] = [];
  const w = 4;
  for (let k = 0; k < w; k++) {
    const endTrain = Math.floor(datedIdx.length * (0.4 + 0.15 * k));
    const endTest = Math.min(datedIdx.length, Math.floor(datedIdx.length * (0.55 + 0.15 * k)));
    if (endTest <= endTrain + 4) continue;
    const tr = datedIdx.slice(0, endTrain);
    const te = datedIdx.slice(endTrain, endTest);
    const b = fitOn(tr.map((i) => feats[i]!), tr.map((i) => y[i]!), shell, kept);
    const p = te.map((i) => predictWith(b, feats[i]!, shell, kept));
    wfRmse.push(rmse(p, te.map((i) => y[i]!)));
    wfSpear.push(spearman(p, te.map((i) => y[i]!)));
  }
  shell.validation.walkForward = { windows: wfRmse.length, rmseByWindow: wfRmse, spearmanByWindow: wfSpear };

  return shell;
}

export function predict(m: FittedModel, p: Proto, asOf: string): number {
  const f = featuresOf(p, m, asOf);
  const kept: DesignOpts = { curve: m.curveKept, growth: m.growthKept, age: m.ageKept, cat: m.catKept, chain: m.chainKept };
  const beta = [
    m.intercept, m.bLogTvl,
    ...(m.curveKept ? [m.bLogTvlSq] : []),
    ...(m.growthKept ? [m.bGrowth] : []),
    ...(m.ageKept ? [m.bAge] : []),
    ...(m.catKept ? m.categories.map((c) => m.catCoef[c] ?? 0) : []),
    ...(m.chainKept ? m.chains.map((c) => m.chainCoef[c] ?? 0) : []),
  ];
  const row = designRow(f, m, kept);
  let s = 0;
  for (let i = 0; i < row.length; i++) s += row[i]! * beta[i]!;
  // Smearing correction: E[value] = exp(pred + sigma^2 / 2) for log-normal errors.
  return Math.exp(s + (m.validation.residualSd ** 2) / 2);
}

/** Rank interval (10th-90th percentile) for each pre-token protocol, from bootstrap refits. */
export function rankIntervals(
  m: FittedModel,
  snap: Snapshot,
  candidates: Proto[],
  draws = 50,
): Map<string, [number, number]> {
  const traded = snap.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0);
  const feats = traded.map((p) => featuresOf(p, m, snap.as_of));
  const y = traded.map((p) => Math.log(p.mcap_usd!));
  const kept: DesignOpts = { curve: m.curveKept, growth: m.growthKept, age: m.ageKept, cat: m.catKept, chain: m.chainKept };
  const candFeats = candidates.map((p) => featuresOf(p, m, snap.as_of));
  const ranks = new Map<string, number[]>();
  const rng = mulberry32(20261114);
  for (let d = 0; d < draws; d++) {
    const idx = Array.from({ length: traded.length }, () => Math.floor(rng() * traded.length));
    const beta = fitOn(idx.map((i) => feats[i]!), idx.map((i) => y[i]!), m, kept);
    const vals = candFeats.map((f) => predictWith(beta, f, m, kept));
    const order = vals.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]);
    order.forEach(([_, i], r) => {
      const arr = ranks.get(candidates[i]!.slug) ?? [];
      arr.push(r + 1);
      ranks.set(candidates[i]!.slug, arr);
    });
  }
  const out = new Map<string, [number, number]>();
  for (const [slug, arr] of ranks) {
    const s = [...arr].sort((a, b) => a - b);
    out.set(slug, [s[Math.floor(0.1 * (s.length - 1))]!, s[Math.ceil(0.9 * (s.length - 1))]!]);
  }
  return out;
}

// ------------------------------------------------------------------ farm plans

export const CHAIN_FEE: Record<string, number> = {
  Ethereum: 9, Solana: 0.05, Base: 0.15, Arbitrum: 0.3, Binance: 0.4,
  Polygon: 0.05, Optimism: 0.2, Avalanche: 0.3, 'Hyperliquid L1': 0.1,
  Monad: 0.05, Sei: 0.05, Berachain: 0.1, Sui: 0.05, Aptos: 0.05,
};
export const DEFAULT_FEE = 0.5;

export function chainFee(chain: string): number {
  return CHAIN_FEE[chain] ?? DEFAULT_FEE;
}

export function mainChain(p: Proto): string {
  return p.chains[0] ?? 'Ethereum';
}

function money(n: number): string {
  return n >= 100 ? `$${Math.round(n).toLocaleString('en-US')}` : `$${n.toFixed(2)}`;
}

// Step templates per category. Placeholders: {NAME} protocol, {CHAIN} main
// chain, {FEE} typical tx fee, {CAP} reference stake, {DAYS} holding window,
// {CAT} category in plain words. The dashboard regenerates the same steps
// from the same table, so the page and the board always agree.
export const STEP_TABLE: Record<string, string[]> = {
  Lending: [
    'Bridge {CAP} of stablecoins to {CHAIN} and deposit them as collateral in {NAME} lending markets, keeping loan-to-value under 60 percent to avoid liquidation. Cost ~{FEE} in network fees plus {CAP} capital, about 25 minutes.',
    'Borrow 40 percent of the deposit in a second asset and redeposit it to loop the position once, which doubles counted activity without doubling risk. Cost ~{FEE} in fees, about 15 minutes.',
    'Hold the looped position for {DAYS} days and repay a small slice weekly so the position shows steady use rather than one burst. Cost ~{FEE} per weekly touch, about 10 minutes each.',
    'Supply a second, smaller asset the market lists so the account touches two pools and not one. Cost ~{FEE} in fees plus $200 capital, about 20 minutes.',
    'Unwind the loop, repay the borrow and withdraw the collateral in one session once the farm window closes. Cost ~{FEE} in fees, about 20 minutes.',
  ],
  Dexs: [
    'Swap {CAP} through {NAME} on {CHAIN} across two different pairs so the router records real volume, not one wash trade. Cost ~{FEE} in fees plus {CAP} capital, about 20 minutes.',
    'Add half the swapped assets to a {NAME} liquidity pool in the most active pair and hold the position for {DAYS} days. Cost ~{FEE} in fees plus $500 capital, about 25 minutes.',
    'Trade a small weekly volume, about $100 per week, to keep the account active across the whole window. Cost ~{FEE} per week in fees, about 10 minutes a week.',
    'Remove liquidity and swap back to stables in one session, accepting the pool fee and any impermanent loss. Cost ~{FEE} in fees, about 15 minutes.',
  ],
  Yield: [
    'Deposit {CAP} into the highest-TVL {NAME} vault on {CHAIN} and leave the receipt token in the wallet so the vault sees one continuous deposit. Cost ~{FEE} in fees plus {CAP} capital, about 20 minutes.',
    'Compound the vault yield back into the same vault every two weeks for {DAYS} days. Cost ~{FEE} per compound in fees, about 10 minutes each.',
    'Move a quarter of the position to a second {NAME} vault strategy to diversify across strategies. Cost ~{FEE} in fees, about 15 minutes.',
    'Withdraw the full position to stables at the end of the window and keep the receipt history. Cost ~{FEE} in fees, about 15 minutes.',
  ],
  'Liquid Staking': [
    'Stake {CAP} of the chain asset through {NAME} on {CHAIN} and hold the liquid staking receipt token instead of selling it. Cost ~{FEE} in fees plus {CAP} capital, about 15 minutes.',
    'Use the receipt token as collateral in a lending market once, at under 40 percent loan-to-value, to show DeFi use. Cost ~{FEE} in fees, about 20 minutes.',
    'Hold the staked position for {DAYS} days, collecting staking yield the whole time. Cost nothing while holding, about 5 minutes to check weekly.',
    'Unstake through the redemption queue, allowing the full waiting period the protocol states before withdrawing. Cost ~{FEE} in fees, about 15 minutes plus the queue wait.',
  ],
  RWA: [
    'Complete identity checks for {NAME} first, since real-world-asset products gate access behind them and a farm without them earns nothing. Cost $0 in fees, about 1 day waiting for approval.',
    'Buy {CAP} of the tokenised asset on {CHAIN} through the protocol\'s primary market and hold the receipt. Cost ~{FEE} in fees plus {CAP} capital, about 30 minutes.',
    'Hold the position for {DAYS} days to collect the underlying yield, checking the payout schedule monthly. Cost nothing while holding, about 10 minutes a month.',
    'Sell back through the primary market or the stated redemption route, allowing the redemption queue time. Cost ~{FEE} in fees, about 20 minutes plus the queue wait.',
  ],
  'Basis Trading': [
    'Deposit {CAP} of collateral into {NAME} on {CHAIN} to open a delta-neutral basis position sized under half the account. Cost ~{FEE} in fees plus {CAP} capital, about 25 minutes.',
    'Hold the position for {DAYS} days while funding payments accrue, rebalancing only if the hedge drifts past 10 percent. Cost ~{FEE} per rebalance, about 15 minutes each.',
    'Withdraw half the position mid-window to lock in accrued funding, then re-enter with the same size. Cost ~{FEE} in fees, about 20 minutes.',
    'Close the full position and withdraw collateral to stables at the end of the window. Cost ~{FEE} in fees, about 20 minutes.',
  ],
  'Risk Curators': [
    'Deposit {CAP} into the top curated vault on {NAME} on {CHAIN}, choosing the vault with the longest track record. Cost ~{FEE} in fees plus {CAP} capital, about 25 minutes.',
    'Hold the vault position for {DAYS} days and review the curator\'s allocation report each month. Cost nothing while holding, about 15 minutes a month.',
    'Allocate $300 to a second curated vault to show breadth across curators. Cost ~{FEE} in fees plus $300 capital, about 20 minutes.',
    'Withdraw from both vaults at the end of the window, allowing any stated withdrawal delay. Cost ~{FEE} in fees, about 20 minutes plus the delay.',
  ],
  'Onchain Capital Allocator': [
    'Deposit {CAP} into the allocator\'s flagship fund on {CHAIN} and keep the position receipt in the wallet. Cost ~{FEE} in fees plus {CAP} capital, about 25 minutes.',
    'Hold for {DAYS} days and read the allocator\'s monthly report to confirm the strategy still matches its mandate. Cost nothing while holding, about 15 minutes a month.',
    'Top up with $200 halfway through the window to show continued commitment. Cost ~{FEE} in fees plus $200 capital, about 15 minutes.',
    'Redeem the full position through the stated route, allowing the redemption queue time. Cost ~{FEE} in fees, about 20 minutes plus the queue wait.',
  ],
  __fallback: [
    'Bridge {CAP} to {CHAIN} and make a first real deposit into {NAME}\'s main product, sized to what you can leave untouched for the window. Cost ~{FEE} in fees plus {CAP} capital, about 25 minutes.',
    'Use the product a second way, a different pool or feature, so the account shows breadth rather than a single deposit. Cost ~{FEE} in fees plus $200 capital, about 20 minutes.',
    'Hold the position for {DAYS} days, touching it once a week with a small action. Cost ~{FEE} per weekly touch, about 10 minutes each.',
    'Withdraw everything to stables at the end of the window and keep the transaction history. Cost ~{FEE} in fees, about 15 minutes.',
  ],
};

export interface FarmVerdict {
  steps: string[];
  feesUsd: number;
  netEv1k: number;
  netEv10k: number;
  breakEvenUsd: number;
  verdict: 'worth it' | 'thin' | 'skip';
  sensitivity: { discount: number; netEv1k: number; netEv10k: number }[];
}

export function farmPlan(p: Proto, impliedUsd: number): FarmVerdict {
  const chain = mainChain(p);
  const fee = chainFee(chain);
  const cap = 1000;
  const days = clamp(Math.round(30 + Math.log10(Math.max(1e6, p.tvl_usd)) * 12), 45, 120);
  const feeStr = money(fee);
  const fill = (t: string): string =>
    t.replaceAll('{NAME}', p.name).replaceAll('{CHAIN}', chain)
      .replaceAll('{FEE}', feeStr).replaceAll('{CAP}', money(cap))
      .replaceAll('{DAYS}', String(days)).replaceAll('{CAT}', (p.category ?? 'protocol').toLowerCase());
  const templates = STEP_TABLE[p.category ?? ''] ?? STEP_TABLE.__fallback!;
  const steps = templates.map(fill);
  const feesUsd = Math.round((fee * (steps.length + 2)) * 100) / 100;

  // Expected value, stated as an assumption in the method notes: a token
  // launch is assumed to distribute 6% of implied value to farmers; the
  // farmer count grows with TVL (crowding), and points scale as stake^0.7
  // against a $2,500 median farmer. Costs are fees plus 2% of stake.
  const farmers = clamp(Math.round(300 + p.tvl_usd / 200000), 500, 60000);
  const share = (stake: number): number =>
    Math.pow(stake, 0.7) / (farmers * Math.pow(2500, 0.7));
  const pool = 0.06 * impliedUsd;
  const net = (stake: number, value: number): number =>
    Math.round((0.06 * value * share(stake) - (feesUsd + 0.02 * stake)) * 100) / 100;
  const netEv1k = net(1000, impliedUsd);
  const netEv10k = net(10000, impliedUsd);
  const breakEvenUsd = Math.round(((feesUsd + 20) / (0.06 * share(1000))) * 100) / 100;
  const verdict: FarmVerdict['verdict'] =
    netEv1k > 50 && netEv10k > 200 ? 'worth it' : netEv1k > 0 || netEv10k > 0 ? 'thin' : 'skip';
  const sensitivity = [5, 10, 20].map((d) => ({
    discount: d,
    netEv1k: net(1000, impliedUsd * (1 - d / 100)),
    netEv10k: net(10000, impliedUsd * (1 - d / 100)),
  }));
  return { steps, feesUsd, netEv1k, netEv10k, breakEvenUsd, verdict, sensitivity };
}

/** One-line driver note: which terms push this protocol's implied value. */
export function driverNote(m: FittedModel, p: Proto, asOf: string): string {
  const f = featuresOf(p, m, asOf);
  const parts: string[] = [];
  const tvlPart = m.bLogTvl * (f.logTvl - m.meanLogTvl);
  parts.push(`TVL ${tvlPart >= 0 ? '+' : ''}${tvlPart.toFixed(2)} log pts`);
  if (f.growth !== 0 && m.growthKept) parts.push(`growth ${f.growth >= 0 ? '+' : ''}${(f.growth * 100).toFixed(0)}pts vs median`);
  if (f.age > 0.5 && m.ageKept) parts.push(`age ${Math.log1p(ageYears(p, asOf, m.medianAge)).toFixed(1)} log-yrs`);
  if (m.catKept && m.catCoef[f.cat] !== undefined && Math.abs(m.catCoef[f.cat]!) > 0.05)
    parts.push(`${f.cat} ${m.catCoef[f.cat]! >= 0 ? '+' : ''}${m.catCoef[f.cat]!.toFixed(2)}`);
  if (m.chainKept) for (const c of f.chains) if (m.chainCoef[c] !== undefined && Math.abs(m.chainCoef[c]!) > 0.05) parts.push(`${c} ${m.chainCoef[c]! >= 0 ? '+' : ''}${m.chainCoef[c]!.toFixed(2)}`);
  return parts.slice(0, 3).join('; ');
}
