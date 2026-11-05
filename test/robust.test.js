'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const MINE = [
  'robust-ucb-trunc', 'robust-ucb-mom', 'robust-ucb-catoni', 'ape2',
  'cvar-ucb', 'mv-ucb', 'marab', 'conservative-ucb', 'dp-ucb', 'dp-ts',
  'ucb-n', 'ucb-maxn', 'ts-n', 'exp3-set', 'exp3-dom', 'elp',
];
const regret = (res, id) => res.find((r) => r.id === id).finalRegret;

test('robust.js registers its algorithms, families and environments', () => {
  for (const id of MINE) {
    const a = M.ALGO_BY_ID[id];
    assert.ok(a, `missing ${id}`);
    assert.ok(M.FAMILIES.some((f) => f.id === a.family), `${id}: unknown family ${a.family}`);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(typeof a[k] === 'string' && a[k].length, `${id}.${k}`);
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${id}.${p.key} default out of range`);
  }
  for (const id of ['heavy-tailed', 'risky', 'side-observations']) assert.ok(M.ENV_BY_ID[id], `missing env ${id}`);
  const ctxIdx = M.FAMILIES.findIndex((f) => f.id === 'contextual');
  for (const f of ['robust', 'risk', 'safe', 'private', 'graphfb']) {
    const i = M.FAMILIES.findIndex((x) => x.id === f);
    assert.ok(i >= 0 && i < ctxIdx, `family ${f} should precede contextual`);
  }
});

test('every robust.js algorithm returns valid arms on every environment', () => {
  const K = 7, T = 300;
  for (const env of M.ENVIRONMENTS) {
    const D = env.d || 1;
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
    const contexts = new Float64Array(T * K * D).fill(D > 1 ? 0 : 1);
    const info = env.fill(K, T, M.Rng(3), means, rewards, contexts) || {};
    const graph = info.graph || null;
    for (const id of MINE) {
      const algo = M.ALGO_BY_ID[id];
      const agent = algo.create(K, { T, env, params: M.defaultParams(id), rng: M.Rng(1), d: D, info });
      const pulls = new Int32Array(K), X = new Float64Array(K * D);
      for (let t = 0; t < T; t++) {
        X.set(contexts.subarray(t * K * D, (t + 1) * K * D));
        const a = agent.select(t + 1, X);
        assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${id} on ${env.id} returned ${a}`);
        const r = env.rested ? rewards[pulls[a]++ * K + a] : rewards[t * K + a];
        assert.ok(Number.isFinite(r));
        agent.update(a, r, t + 1);
        if (graph && agent.observe) {
          for (let b = 0; b < K; b++) if (b !== a && graph[a * K + b]) agent.observe(b, rewards[t * K + b], t + 1);
        }
      }
    }
  }
});

test('graph-aware algorithms work without ctx.info (no side observations)', () => {
  const env = M.ENV_BY_ID.bernoulli;
  for (const id of MINE) {
    const agent = M.ALGO_BY_ID[id].create(5, { T: 200, env, params: M.defaultParams(id), rng: M.Rng(4) });
    const rng = M.Rng(5);
    for (let t = 1; t <= 200; t++) {
      const a = agent.select(t);
      assert.ok(Number.isInteger(a) && a >= 0 && a < 5, `${id} returned ${a}`);
      agent.update(a, rng.uniform() < 0.3 + 0.1 * a ? 1 : 0, t);
    }
  }
});

test('the side-observations environment returns a symmetric graph with self-loops', () => {
  const K = 12, T = 5, env = M.ENV_BY_ID['side-observations'];
  const info = env.fill(K, T, M.Rng(9), new Float64Array(T * K), new Float64Array(T * K), new Float64Array(T * K));
  assert.ok(info && info.graph instanceof Uint8Array && info.graph.length === K * K);
  let edges = 0;
  for (let a = 0; a < K; a++) {
    assert.equal(info.graph[a * K + a], 1);
    for (let b = 0; b < K; b++) { assert.equal(info.graph[a * K + b], info.graph[b * K + a]); if (a !== b) edges += info.graph[a * K + b]; }
  }
  assert.ok(edges > 0);
});

test('robust UCBs beat UCB1 on heavy-tailed rewards', () => {
  const res = M.simulate({
    env: 'heavy-tailed', K: 10, T: 1000, runs: 200, seed: 5,
    algorithms: ['ucb1', 'robust-ucb-trunc', 'robust-ucb-mom', 'robust-ucb-catoni', 'ape2'].map((id) => ({ id })),
  });
  const ucb = regret(res, 'ucb1');
  for (const id of ['robust-ucb-trunc', 'robust-ucb-mom', 'robust-ucb-catoni']) {
    assert.ok(regret(res, id) < ucb, `${id}: ${regret(res, id)} vs ucb1 ${ucb}`);
  }
});

test('side observations help: UCB-N / Thompson-N beat UCB1 / Thompson, EXP3-SET beats EXP3', () => {
  const res = M.simulate({
    env: 'side-observations', K: 10, T: 1000, runs: 100, seed: 6,
    algorithms: ['ucb1', 'thompson', 'exp3', 'ucb-n', 'ucb-maxn', 'ts-n', 'exp3-set', 'exp3-dom', 'elp'].map((id) => ({ id })),
  });
  assert.ok(regret(res, 'ucb-n') < regret(res, 'ucb1'), 'ucb-n vs ucb1');
  assert.ok(regret(res, 'ucb-maxn') < regret(res, 'ucb1'), 'ucb-maxn vs ucb1');
  assert.ok(regret(res, 'ts-n') < regret(res, 'thompson'), 'ts-n vs thompson');
  for (const id of ['exp3-set', 'exp3-dom', 'elp']) assert.ok(regret(res, id) < regret(res, 'exp3'), `${id} vs exp3`);
});

test('differentially private methods beat random on Bernoulli arms', () => {
  const res = M.simulate({
    env: 'bernoulli', K: 10, T: 1000, runs: 100, seed: 7,
    algorithms: ['random', 'dp-ucb', 'dp-ts'].map((id) => ({ id })),
  });
  const rnd = regret(res, 'random');
  assert.ok(regret(res, 'dp-ucb') < 0.85 * rnd, `dp-ucb ${regret(res, 'dp-ucb')} vs random ${rnd}`);
  assert.ok(regret(res, 'dp-ts') < 0.5 * rnd, `dp-ts ${regret(res, 'dp-ts')} vs random ${rnd}`);
});

test('risk-aware methods avoid the risky best-mean arm', () => {
  const res = M.simulate({
    env: 'risky', K: 10, T: 1000, runs: 100, seed: 8,
    algorithms: ['ucb1', 'cvar-ucb', 'mv-ucb', 'marab'].map((id) => ({ id })),
  });
  const share = (id) => res.find((r) => r.id === id).summary.bestShare;
  for (const id of ['cvar-ucb', 'mv-ucb', 'marab']) assert.ok(share(id) < share('ucb1') / 2, `${id} best-mean share ${share(id)} vs ucb1 ${share('ucb1')}`);
});

test('Conservative UCB keeps cumulative reward near (1 − α) × baseline', () => {
  const K = 10, T = 1000, alpha = 0.1, env = M.ENV_BY_ID.bernoulli;
  let violations = 0, steps = 0;
  for (let run = 0; run < 30; run++) {
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
    env.fill(K, T, M.Rng(M.hash32(run, 'env')), means, rewards, new Float64Array(T * K));
    const agent = M.ALGO_BY_ID['conservative-ucb'].create(K, { T, env, params: { alpha, c: 1.414 }, rng: M.Rng(run), d: 1, info: {} });
    let got = 0;
    for (let t = 0; t < T; t++) {
      const a = agent.select(t + 1);
      got += means[t * K + a];
      agent.update(a, rewards[t * K + a], t + 1);
      // pseudo-reward vs the constraint, with slack for the plug-in baseline estimate
      if (t >= 50) { steps++; if (got < (1 - alpha) * (t + 1) * means[0] - 3 * Math.sqrt(t + 1)) violations++; }
    }
  }
  assert.ok(violations / steps < 0.02, `violation rate ${violations / steps}`);
});

test('robust.js algorithms are fast enough (K = 10, T = 1000)', () => {
  for (const env of ['heavy-tailed', 'side-observations']) {
    const res = M.simulate({ env, K: 10, T: 1000, runs: 10, seed: 1, algorithms: MINE.map((id) => ({ id })) });
    for (const r of res) assert.ok(r.msPerRun < 25, `${r.id} on ${env}: ${r.msPerRun.toFixed(2)} ms/run`);
  }
});
