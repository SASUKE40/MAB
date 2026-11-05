'use strict';
// Tests for the knapsack, multi-player and restless settings.
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const IDS = ['knapsack', 'multiplayer', 'restless'];
const get = (id) => {
  const s = M.SETTINGS.find((x) => x.id === id);
  assert.ok(s, `setting ${id} is registered`);
  return s;
};
const extraDefaults = (S) => Object.fromEntries((S.controls.extra || []).map((x) => [x.key, x.default]));
function run(S, cfg) {
  const r = S.createRunner(Object.assign({ seed: 11, extra: extraDefaults(S) }, cfg));
  r.step(cfg.runs);
  return r.results();
}
const byId = (rows) => Object.fromEntries(rows.map((r) => [r.id, r]));
function table(title, rows, keys) {
  console.log(`\n${title}`);
  console.table(rows.map((r) => Object.fromEntries([['id', r.id], ...keys.map((k) => [k, +r.summary[k].toFixed(3)])])));
}

for (const id of IDS) {
  test(`${id}: API shape`, () => {
    const S = get(id);
    for (const k of ['id', 'name', 'short', 'blurb']) assert.equal(typeof S[k], 'string', k);
    assert.ok(S.families.length && S.environments.length >= 3 && S.algorithms.length >= 6);
    const fam = new Set(S.families.map((f) => f.id));
    const algoIds = new Set();
    for (const a of S.algorithms) {
      assert.ok(!algoIds.has(a.id), `duplicate ${a.id}`);
      algoIds.add(a.id);
      assert.ok(fam.has(a.family), `${a.id} family`);
      for (const k of ['name', 'formula', 'regret', 'idea']) assert.equal(typeof a[k], 'string', `${a.id}.${k}`);
      assert.equal(typeof a.create, 'function');
      for (const p of a.params) {
        assert.ok(p.default >= p.min && p.default <= p.max, `${a.id}.${p.key} default in range`);
        assert.equal(typeof p.label, 'string');
      }
    }
    const envIds = new Set(S.environments.map((e) => e.id));
    assert.equal(envIds.size, S.environments.length);
    for (const e of S.environments) assert.ok(e.name && e.description && Array.isArray(e.tags), e.id);
    assert.ok(envIds.has(S.defaultEnv));
    assert.ok(S.defaultLineup.length <= 8 && S.defaultLineup.every((x) => algoIds.has(x)));
    for (const p of S.presets) {
      assert.ok(p.algos.length <= 8 && p.algos.every((x) => algoIds.has(x)), p.label);
      if (p.env) assert.ok(envIds.has(p.env), p.label);
    }
    const c = S.controls;
    assert.ok(c.K.default >= c.K.min && c.K.default <= c.K.max);
    assert.ok(c.T.options.includes(c.T.default) && c.runs.options.includes(c.runs.default));
    for (const x of c.extra || []) assert.ok(x.default >= x.min && x.default <= x.max, x.key);

    const [r] = run(S, { env: S.defaultEnv, K: 4, T: 60, runs: 2, algorithms: [{ id: S.defaultLineup[0] }] });
    assert.equal(r.runs, 2);
    for (const m of S.metrics) {
      assert.ok(r.curves[m.key] instanceof Float64Array && r.curves[m.key].length === 60, `curve ${m.key}`);
      if (m.band) assert.ok(r.curvesSE[m.key].length === 60);
    }
    const keys = [S.final.key, S.final.se, S.final.median, ...S.tiles.map((t) => t.key), ...S.columns.map((t) => t.key), 'msPerRun'];
    for (const k of keys) assert.ok(Number.isFinite(r.summary[k]), `summary.${k}`);
    assert.equal(r.finals.length, 2);
  });

  test(`${id}: deterministic and independent of the lineup`, () => {
    const S = get(id);
    const all = S.algorithms.map((a) => ({ id: a.id }));
    for (const env of S.environments) {
      const base = { env: env.id, K: 5, T: 150, runs: 3 };
      const a = byId(run(S, { ...base, algorithms: all }));
      const b = byId(run(S, { ...base, algorithms: all.slice().reverse() }));
      const one = run(S, { ...base, algorithms: [all[all.length - 1]] })[0];
      for (const k of Object.keys(a)) {
        assert.deepEqual(Array.from(a[k].curves[S.metrics[0].key]), Array.from(b[k].curves[S.metrics[0].key]), `${env.id}/${k}`);
      }
      assert.equal(one.summary[S.final.key], a[one.id].summary[S.final.key], `${env.id}/${one.id} alone`);
    }
  });

  test(`${id}: every algorithm runs on every environment`, () => {
    const S = get(id);
    for (const env of S.environments) {
      for (const K of [3, 6]) {
        const rows = run(S, { env: env.id, K, T: 120, runs: 2, algorithms: S.algorithms.map((a) => ({ id: a.id })) });
        for (const r of rows) {
          for (const m of S.metrics) assert.ok(r.curves[m.key].every(Number.isFinite), `${env.id}/${r.id}/${m.key}`);
          for (const [k, v] of Object.entries(r.summary)) assert.ok(Number.isFinite(v), `${env.id}/${r.id}/${k}`);
        }
      }
      // extreme values of the extra controls (budget β, number of players / pulls)
      for (const x of S.controls.extra || []) {
        for (const v of [x.min, x.max]) {
          const rows = run(S, { env: env.id, K: 6, T: 80, runs: 1, extra: { ...extraDefaults(S), [x.key]: v }, algorithms: S.algorithms.map((a) => ({ id: a.id })) });
          for (const r of rows) assert.ok(r.curves[S.metrics[0].key].every(Number.isFinite), `${env.id}/${r.id}/${x.key}=${v}`);
        }
      }
    }
  });

  test(`${id}: one step of 8 algorithms at default controls is fast`, () => {
    const S = get(id);
    const lineup = S.algorithms.slice(0, 8).map((a) => ({ id: a.id }));
    let worst = 0;
    for (const env of S.environments) {
      const r = S.createRunner({ env: env.id, K: S.controls.K.default, T: S.controls.T.default, runs: 5, seed: 1, extra: extraDefaults(S), algorithms: lineup });
      r.step(1);                         // warm-up (JIT)
      const t0 = performance.now();
      r.step(1);
      worst = Math.max(worst, performance.now() - t0);
    }
    console.log(`${id}: slowest step(1) with ${lineup.length} algorithms: ${worst.toFixed(1)} ms`);
    assert.ok(worst < 200, `${worst} ms`);   // budget is ~100 ms; slack for loaded CI machines
  });
}

// ------------------------------------------------------------- internals ----
test('knapsack LP solver matches a hand-solved LP', () => {
  const lp = get('knapsack')._internal.makeLP(3, 2);
  // max 3x + 2y + z  s.t.  x + y + z ≤ 4,  x + 3y ≤ 6  → x = 4 (value 12)
  const out = new Float64Array(3);
  const v = lp(Float64Array.of(3, 2, 1), Float64Array.of(1, 1, 1, 1, 3, 0), Float64Array.of(4, 6), 3, 2, out);
  assert.ok(Math.abs(v - 12) < 1e-9 && Math.abs(out[0] - 4) < 1e-9);
  // max x + y  s.t. 2x + y ≤ 4, x + 2y ≤ 4 → x = y = 4/3
  const o2 = new Float64Array(2);
  const v2 = get('knapsack')._internal.makeLP(2, 2)(Float64Array.of(1, 1), Float64Array.of(2, 1, 1, 2), Float64Array.of(4, 4), 2, 2, o2);
  assert.ok(Math.abs(v2 - 8 / 3) < 1e-9 && Math.abs(o2[0] - 4 / 3) < 1e-9);
});

test('multiplayer Hungarian assignment matches brute force', () => {
  const { assignMax } = get('multiplayer')._internal;
  const rng = M.Rng(3);
  for (let it = 0; it < 50; it++) {
    const n = 1 + rng.int(3), m = n + rng.int(3);
    const w = Float64Array.from({ length: n * m }, () => rng.uniform());
    let best = -Infinity;
    const rec = (i, used, acc) => {
      if (i === n) { best = Math.max(best, acc); return; }
      for (let j = 0; j < m; j++) if (!(used & (1 << j))) rec(i + 1, used | (1 << j), acc + w[i * m + j]);
    };
    rec(0, 0, 0);
    assert.ok(Math.abs(assignMax(w, n, m) - best) < 1e-9);
  }
});

test('restless closed-form Whittle index agrees with bisection on the optimal advantage', () => {
  const { whittle, optGap, advantage } = get('restless')._internal;
  const rng = M.Rng(4);
  const P = new Float64Array(4);
  for (let it = 0; it < 300; it++) {
    for (let k = 0; k < 4; k++) P[k] = rng.uniform();
    for (const s of [0, 1]) {
      let lo = -25, hi = 25;
      for (let i = 0; i < 60; i++) {
        const mid = (lo + hi) / 2;
        if (advantage(P, 0, s, optGap(P, 0, mid, 0.95), mid, 0.95) > 0) lo = mid; else hi = mid;
      }
      assert.ok(Math.abs(whittle(P, 0, s, 0.95) - lo) < 1e-6, `P=${Array.from(P)} s=${s}`);
    }
  }
});

// ---------------------------------------------------------- plausibility ----
test('knapsack: UCB-BwK and PD-BwK beat cost-blind UCB1 on the tight-budget instance', () => {
  const S = get('knapsack');
  const rows = run(S, { env: 'one', K: 5, T: 1000, runs: 40, extra: { beta: 0.25 }, algorithms: S.algorithms.map((a) => ({ id: a.id })) });
  table('knapsack / one resource, β = 0.25, T = 1000, 40 runs', rows, ['regret', 'regretSE', 'ratio', 'left', 'stopT']);
  const r = byId(rows);
  assert.ok(r['ucb-bwk'].summary.regret < 0.6 * r.ucb1.summary.regret);
  assert.ok(r['pd-bwk'].summary.regret < 0.8 * r.ucb1.summary.regret);
  assert.ok(r.ucb1.summary.stopT < 0.6 * 1000, 'UCB1 runs out of budget early');
  for (const x of rows) assert.ok(x.summary.ratio < 1.05, `${x.id} cannot beat the LP by much`);
});

test('multiplayer: MCTopM and SIC-MMAB avoid accidental collisions; selfish UCB1 does not', () => {
  const S = get('multiplayer');
  const rows = run(S, { env: 'bernoulli', K: 8, T: 3000, runs: 20, extra: { M: 3 }, algorithms: S.algorithms.map((a) => ({ id: a.id })) });
  table('multiplayer / homogeneous Bernoulli, K = 8, M = 3, T = 3000, 20 runs', rows, ['regret', 'collisions', 'accidental', 'signalling', 'meanReward']);
  const r = byId(rows);
  const selfish = r['selfish-ucb'].summary.collisions;
  assert.ok(r.mctopm.summary.collisions < 0.5 * selfish, 'MCTopM collides far less');
  assert.ok(r.randtopm.summary.collisions < 0.5 * selfish, 'RandTopM collides far less');
  assert.ok(r['sic-mmab'].summary.accidental < 0.1 * selfish, 'SIC-MMAB only collides on purpose');
  assert.ok(r.dpe.summary.collisions < 0.5 * selfish, 'DPE collides far less');
  assert.ok(r.mctopm.summary.regret < r['selfish-ucb'].summary.regret);
  assert.ok(r.dpe.summary.regret < r['selfish-ucb'].summary.regret);
});

test('multiplayer: federated methods beat isolated learners when there are no collisions', () => {
  const S = get('multiplayer');
  const fb = byId(run(S, { env: 'fed-bernoulli', K: 8, T: 2000, runs: 20, extra: { M: 4 }, algorithms: [{ id: 'fed-ucb' }, { id: 'selfish-ucb' }] }));
  assert.ok(fb['fed-ucb'].summary.regret < fb['selfish-ucb'].summary.regret);
  assert.equal(fb['fed-ucb'].summary.collisions, 0);
  const fl = byId(run(S, { env: 'fed-linear', K: 8, T: 1000, runs: 10, extra: { M: 3 }, algorithms: [{ id: 'fed-linucb' }, { id: 'fed-ucb' }] }));
  assert.ok(fl['fed-linucb'].summary.regret < 0.2 * fl['fed-ucb'].summary.regret);
});

test('restless: UCWhittle approaches the oracle Whittle policy and beats random', () => {
  const S = get('restless');
  const T = 1000;
  const rows = run(S, { env: 'adherence', K: 10, T, runs: 30, extra: { M: 3 }, algorithms: S.algorithms.map((a) => ({ id: a.id })) });
  table('restless / adherence, N = 10, M = 3, T = 1000, 30 runs', rows, ['regret', 'regretSE', 'meanReward', 'goodShare']);
  const r = byId(rows);
  assert.ok(Math.abs(r['whittle-oracle'].summary.regret) < 1e-9, 'oracle is the benchmark');
  const uc = r.ucwhittle.curves.regret, rnd = r.random.summary.regret;
  assert.ok(r.ucwhittle.summary.regret < 0.3 * rnd);
  const early = uc[199] / 200, late = (uc[T - 1] - uc[T - 201]) / 200;
  assert.ok(late < 0.5 * early, `per-round regret shrinks: early ${early.toFixed(3)}, late ${late.toFixed(3)}`);
  assert.ok(r.ucwhittle.summary.meanReward > 0.97 * r['whittle-oracle'].summary.meanReward);
  assert.ok(r['ts-whittle'].summary.regret < 0.5 * rnd);
});
