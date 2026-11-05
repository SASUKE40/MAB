'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const S = M.SETTINGS.find((s) => s.id === 'bai');
const ALGO_IDS = S ? S.algorithms.map((a) => a.id) : [];
const ENV_IDS = S ? S.environments.map((e) => e.id) : [];

function run(cfg) {
  const r = S.createRunner(Object.assign({ env: 'bern-easy', K: 8, T: 1000, runs: 10, seed: 1, extra: { delta: 0.05 } }, cfg));
  r.step(r.config.runs);
  return r.results();
}
const byId = (res) => Object.fromEntries(res.map((x) => [x.id, x]));

test('bai setting is registered with a complete description', () => {
  assert.ok(S, 'setting bai registered');
  for (const k of ['id', 'name', 'short', 'blurb', 'families', 'environments', 'algorithms', 'defaultEnv', 'defaultLineup', 'controls', 'xLabel', 'metrics', 'final', 'tiles', 'columns', 'createRunner']) {
    assert.ok(S[k] != null, `missing ${k}`);
  }
  const fam = new Set(S.families.map((f) => f.id));
  const ids = new Set();
  for (const a of S.algorithms) {
    assert.ok(!ids.has(a.id), `duplicate ${a.id}`); ids.add(a.id);
    assert.ok(fam.has(a.family), `${a.id}: unknown family ${a.family}`);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.equal(typeof a[k], 'string', `${a.id}.${k}`);
    assert.ok(!a.formula.includes('\n'), `${a.id} formula is one line`);
    assert.ok(Array.isArray(a.params));
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${a.id}.${p.key} default in range`);
  }
  for (const e of S.environments) for (const k of ['id', 'name', 'description']) assert.equal(typeof e[k], 'string');
  assert.ok(ENV_IDS.includes(S.defaultEnv));
  assert.ok(S.defaultLineup.length <= 8 && S.defaultLineup.every((id) => ids.has(id)));
  for (const p of S.presets) {
    assert.ok(p.algos.every((id) => ids.has(id)), p.label);
    if (p.env) assert.ok(ENV_IDS.includes(p.env), p.label);
  }
  const want = ['uniform', 'se', 'me', 'lucb', 'lucbpp', 'ugape', 'lilucb', 'gifa', 'tas-c', 'tas-d', 'fws', 'ttts', 't3c', 'ttucb', 'sr', 'sh', 'sar', 'xy-static', 'xy-adaptive', 'rage', 'peace', 'lingape'];
  for (const id of want) assert.ok(ids.has(id), `algorithm ${id} present`);
  const c = S.controls;
  assert.ok(c.T.options.includes(c.T.default) && c.runs.options.includes(c.runs.default));
  for (const x of c.extra) assert.ok(x.default >= x.min && x.default <= x.max);
});

test('runner API: incremental steps and results shape', () => {
  const r = S.createRunner({ env: 'bern-easy', K: 6, T: 800, runs: 4, seed: 3, extra: { delta: 0.1 }, algorithms: S.defaultLineup.map((id) => ({ id })) });
  assert.equal(r.done, 0);
  assert.equal(r.finished, false);
  r.step(1);
  assert.equal(r.done, 1);
  const partial = r.results();
  assert.equal(partial[0].runs, 1);
  r.step(10);
  assert.equal(r.done, 4);
  assert.equal(r.finished, true);
  const res = r.results();
  assert.equal(res.length, S.defaultLineup.length);
  const keys = new Set([S.final.key, S.final.se, S.final.median, ...S.tiles.map((t) => t.key), ...S.columns.map((c) => c.key), 'msPerRun']);
  for (const x of res) {
    assert.ok(x.id && x.name && x.family && x.params);
    assert.equal(x.runs, 4);
    const L = x.x.length;
    assert.ok(L > 5 && x.x[L - 1] === 800);
    for (let i = 1; i < L; i++) assert.ok(x.x[i] > x.x[i - 1], 'x increasing');
    for (const m of S.metrics) {
      assert.equal(x.curves[m.key].length, L, m.key);
      for (const v of x.curves[m.key]) assert.ok(v >= 0 && v <= 1, `${x.id} ${m.key} in [0,1]`);
    }
    assert.equal(x.curvesSE.error.length, L);
    for (const k of keys) assert.ok(Number.isFinite(x.summary[k]), `${x.id} summary.${k}`);
    assert.equal(x.finals.length, 4);
    for (let i = 1; i < x.finals.length; i++) assert.ok(x.finals[i] >= x.finals[i - 1], 'finals sorted');
    assert.equal(x.pullShare.length, 6);
    assert.ok(Math.abs(x.pullShare.reduce((a, b) => a + b, 0) - 1) < 1e-9);
    // stopped-by-t is non-decreasing and ends at 1 − cap-hit rate
    for (let i = 1; i < L; i++) assert.ok(x.curves.stopped[i] >= x.curves.stopped[i - 1]);
    assert.ok(Math.abs(x.curves.stopped[L - 1] - (1 - x.summary.capHit)) < 1e-9);
    assert.ok(x.summary.samples >= 1 && x.summary.samples <= 800);
  }
});

test('missing params are filled from defaults; overrides are kept', () => {
  const res = run({ runs: 2, T: 300, algorithms: [{ id: 'ttts' }, { id: 'me', params: { eps: 0.5 } }] });
  assert.equal(res[0].params.beta, 0.5);
  assert.equal(res[1].params.eps, 0.5);
});

test('deterministic, and independent of the rest of the lineup', () => {
  const cfg = { env: 'gauss-arith', K: 6, T: 600, runs: 6, seed: 42, algorithms: [{ id: 'ttts' }, { id: 'tas-d' }, { id: 'sh' }] };
  const a = run(cfg), b = run(cfg);
  for (let i = 0; i < a.length; i++) {
    assert.deepEqual(Array.from(a[i].curves.error), Array.from(b[i].curves.error));
    assert.deepEqual(a[i].finals, b[i].finals);
    assert.equal(a[i].summary.errorStop, b[i].summary.errorStop);
  }
  const solo = run(Object.assign({}, cfg, { algorithms: [{ id: 'sh' }] }))[0];
  assert.deepEqual(Array.from(solo.curves.error), Array.from(byId(a).sh.curves.error));
  assert.deepEqual(solo.finals, byId(a).sh.finals);
  const other = run(Object.assign({}, cfg, { seed: 43 }));
  assert.notDeepEqual(other[0].finals, a[0].finals);
});

test('every algorithm runs on every environment (K = 2, 3, 7)', () => {
  for (const env of ENV_IDS) {
    for (const K of [2, 3, 7]) {
      const res = run({ env, K, T: 400, runs: 2, seed: 5, algorithms: ALGO_IDS.map((id) => ({ id })) });
      assert.equal(res.length, ALGO_IDS.length);
      for (const x of res) {
        for (const v of x.curves.error) assert.ok(v >= 0 && v <= 1, `${x.id} on ${env} K=${K}`);
        assert.ok(x.summary.samples > 0 && x.summary.samples <= 400, `${x.id} on ${env}`);
      }
    }
  }
});

test('plausibility: Track-and-Stop and TTTS stop sooner than uniform sampling (easy instance)', () => {
  const res = byId(run({ env: 'bern-easy', K: 8, T: 5000, runs: 60, seed: 11, algorithms: ['uniform', 'tas-d', 'tas-c', 'ttts', 't3c', 'ttucb', 'fws'].map((id) => ({ id })) }));
  const u = res.uniform.summary.samples;
  for (const id of ['tas-d', 'tas-c', 'ttts', 't3c', 'ttucb', 'fws']) {
    assert.ok(res[id].summary.samples < 0.8 * u, `${id} ${res[id].summary.samples} vs uniform ${u}`);
    assert.equal(res[id].summary.capHit, 0, `${id} never hits the cap`);
  }
});

test('plausibility: stopping rules are roughly δ-correct', () => {
  const ids = ['uniform', 'se', 'lucbpp', 'lilucb', 'gifa', 'tas-d', 'ttts', 'ttucb'];
  const res = run({ env: 'gauss-arith', K: 6, T: 5000, runs: 80, seed: 2, extra: { delta: 0.1 }, algorithms: ids.map((id) => ({ id })) });
  for (const x of res) assert.ok(x.summary.errorStop <= 0.1 + 0.08, `${x.id} wrong-at-stop ${x.summary.errorStop}`);
});

test('plausibility: SH and SR error falls with the budget', () => {
  const res = byId(run({ env: 'bern-close', K: 8, T: 5000, runs: 150, seed: 9, algorithms: ['sh', 'sr', 'sar', 'uniform'].map((id) => ({ id })) }));
  for (const id of ['sh', 'sr']) {
    const e = res[id].curves.error, L = e.length;
    const third = (a, b) => { let s = 0; for (let i = a; i < b; i++) s += e[i]; return s / (b - a); };
    const early = third(0, Math.floor(L / 3)), mid = third(Math.floor(L / 3), Math.floor(2 * L / 3)), late = third(Math.floor(2 * L / 3), L);
    assert.ok(early > mid && mid > late, `${id}: ${early.toFixed(3)} > ${mid.toFixed(3)} > ${late.toFixed(3)}`);
    assert.ok(e[L - 1] < 0.15, `${id} error at T ${e[L - 1]}`);
  }
  // with m = 1 SAR reduces to SR
  assert.deepEqual(Array.from(res.sar.curves.error), Array.from(res.sr.curves.error));
});

test('plausibility: linear methods exploit the shared structure', () => {
  const res = byId(run({ env: 'linear', K: 8, T: 5000, runs: 40, seed: 4, algorithms: ['uniform', 'tas-d', 'xy-static', 'xy-adaptive', 'lingape'].map((id) => ({ id })) }));
  for (const id of ['xy-static', 'xy-adaptive', 'lingape']) {
    assert.ok(res[id].summary.samples < 0.75 * res['tas-d'].summary.samples, `${id} ${res[id].summary.samples} vs tas-d ${res['tas-d'].summary.samples}`);
    assert.ok(res[id].summary.errorAvg < res.uniform.summary.errorAvg, `${id} error`);
  }
});

test('plausibility: top-m algorithms identify the top set', () => {
  const res = byId(run({ env: 'topm', K: 8, T: 5000, runs: 40, seed: 6, algorithms: ['uniform', 'gifa', 'lucb', 'lucbpp', 'sar', 'ttts'].map((id) => ({ id })) }));
  assert.ok(res.gifa.summary.samples < res.uniform.summary.samples, 'gifa beats uniform');
  assert.ok(res.lucbpp.summary.samples < res.lucb.summary.samples, 'LUCB++ improves on LUCB1');
  assert.ok(res.sar.summary.errorFinal <= 0.05);
  assert.ok(res.gifa.summary.samples < res.ttts.summary.samples, 'native top-m beats best-arm sampling');
});

test('optimal weights solve the lower-bound equations', () => {
  const { optimalWeights } = S._internal;
  const K = 6, w = new Float64Array(K), xs = new Float64Array(K), xl = new Float64Array(K);
  const mu = Float64Array.from([0.3, 0.7, 0.5, 0.6, 0.2, 0.45]);
  // Gaussian: closed form, Σ_{a≠1} (w_a/w_1)² = 1
  optimalWeights(mu, K, false, 1, w, xs, xl);
  let s = 0, q = 0;
  for (let a = 0; a < K; a++) { s += w[a]; if (a !== 1) q += (w[a] / w[1]) ** 2; }
  assert.ok(Math.abs(s - 1) < 1e-9 && Math.abs(q - 1) < 1e-6, `sum ${s}, F ${q}`);
  // Bernoulli: at w*, the transport costs w₁kl(μ₁,m) + wₐkl(μₐ,m) are equal across rivals
  optimalWeights(mu, K, true, 0.5, w, xs, xl);
  const costs = [];
  for (let a = 0; a < K; a++) {
    if (a === 1) continue;
    const m = (w[1] * mu[1] + w[a] * mu[a]) / (w[1] + w[a]);
    costs.push(w[1] * M.klBern(mu[1], m) + w[a] * M.klBern(mu[a], m));
  }
  const lo = Math.min(...costs), hi = Math.max(...costs);
  assert.ok((hi - lo) / hi < 0.01, `costs ${costs.map((c) => c.toFixed(5)).join(' ')}`);
  assert.ok(w[1] > w[3] && w[3] > w[2] && w[2] > w[5] && w[5] > w[0] && w[0] > w[4], 'smaller gap, more weight');
});

test('performance: one step of the default lineup at default controls is well under 100 ms', () => {
  const r = S.createRunner({
    env: S.defaultEnv, K: S.controls.K.default, T: S.controls.T.default, runs: 50, seed: 1,
    extra: { delta: S.controls.extra[0].default }, algorithms: S.defaultLineup.map((id) => ({ id })),
  });
  r.step(1); // warm-up (JIT)
  const times = [];
  for (let i = 0; i < 5; i++) { const t0 = performance.now(); r.step(1); times.push(performance.now() - t0); }
  times.sort((a, b) => a - b);
  assert.ok(times[2] < 100, `median step ${times[2].toFixed(1)} ms`);
});

test('comparison table (default env and controls)', (t) => {
  const res = run({ env: 'bern-easy', K: 8, T: 5000, runs: 100, seed: 1, algorithms: ALGO_IDS.map((id) => ({ id })) });
  const lines = ['algorithm      samples  median  cap%  wrongStop%  wrong@T%  wrongAvg%  ms/run'];
  for (const x of res) {
    const s = x.summary;
    lines.push(`${x.id.padEnd(12)} ${s.samples.toFixed(0).padStart(9)} ${s.samplesMedian.toFixed(0).padStart(7)} ${(100 * s.capHit).toFixed(0).padStart(5)} ${(100 * s.errorStop).toFixed(1).padStart(11)} ${(100 * s.errorFinal).toFixed(1).padStart(9)} ${(100 * s.errorAvg).toFixed(1).padStart(10)} ${s.msPerRun.toFixed(2).padStart(7)}`);
  }
  t.diagnostic('\n' + lines.join('\n'));
  if (process.env.BAI_TABLE) console.log(lines.join('\n'));
  const r = byId(res);
  // sanity: adaptive methods beat uniform; Median Elimination is the most sample-hungry
  assert.ok(r.gifa.summary.samples < r.uniform.summary.samples);
  assert.ok(r.lucbpp.summary.samples < r.lucb.summary.samples, 'LUCB++ improves on LUCB1');
  assert.equal(r.me.summary.capHit, 1);
  for (const x of res) assert.ok(x.summary.errorStop <= 0.05 + 0.05, `${x.id} δ-correct`);
});
