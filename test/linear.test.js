'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const LINEAR_IDS = ['oful', 'lin-ts', 'linrel', 'suplinrel', 'suplinucb', 'hybrid-linucb', 'g-optimal', 'spectral-ucb', 'clucb',
  'glm-ucb', 'ucb-glm', 'logistic-ucb1', 'logistic-ucb2', 'ofuglb', 'lasso-bandit', 'dr-lasso', 'estc', 'estr', 'dlinucb', 'sw-linucb'];
const LINEAR_ENVS = ['linear-shared', 'linear-fixed', 'logistic', 'sparse', 'low-rank', 'graph-smooth', 'linear-drift'];

const final = (res, id) => res.find((r) => r.id === id).finalRegret;
const sim = (env, ids, T = 1000, runs = 20, seed = 5, K = 10) => M.simulate({ env, K, T, runs, seed, algorithms: ids.map((id) => ({ id })) });

test('linear add-on registers its algorithms, environments and families', () => {
  for (const id of LINEAR_IDS) assert.ok(M.ALGO_BY_ID[id], `missing algorithm ${id}`);
  for (const id of LINEAR_ENVS) {
    const e = M.ENV_BY_ID[id];
    assert.ok(e, `missing environment ${id}`);
    assert.ok(e.contextual && e.d > 1 && e.tags.includes('context'), `${id} should be contextual with d > 1`);
  }
  assert.equal(M.ENV_BY_ID['linear-drift'].stationary, false);
  assert.ok(M.ENV_BY_ID['linear-drift'].tags.includes('changes'));
  for (const f of ['linear', 'glm', 'sparse', 'nslinear']) assert.ok(M.FAMILIES.some((x) => x.id === f), `missing family ${f}`);
  for (const id of LINEAR_IDS) {
    const a = M.ALGO_BY_ID[id];
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(a[k], `${id} is missing ${k}`);
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${id}.${p.key} default out of range`);
  }
});

test('environments fill finite means, rewards and contexts', () => {
  const K = 7, T = 50;
  for (const id of LINEAR_ENVS) {
    const env = M.ENV_BY_ID[id], d = env.d;
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K), ctx = new Float64Array(T * K * d);
    const info = env.fill(K, T, M.Rng(3), means, rewards, ctx) || {};
    assert.ok(!info.graph, `${id} must not return info.graph (reserved for side observations)`);
    for (const arr of [means, rewards, ctx]) for (const v of arr) assert.ok(Number.isFinite(v), `${id} produced ${v}`);
    if (env.type === 'bernoulli') for (const r of rewards) assert.ok(r === 0 || r === 1);
  }
  // graph-smooth exposes the Laplacian spectrum; low-rank documents its layout
  const g = M.ENV_BY_ID['graph-smooth'], K2 = 12;
  const info = g.fill(K2, 1, M.Rng(1), new Float64Array(K2), new Float64Array(K2), new Float64Array(K2 * g.d));
  assert.equal(info.spectral.eigenvalues.length, K2);
  assert.ok(Math.abs(info.spectral.eigenvalues[0]) < 1e-9, 'smallest Laplacian eigenvalue is 0');
  const lr = M.ENV_BY_ID['low-rank'];
  const li = lr.fill(3, 1, M.Rng(1), new Float64Array(3), new Float64Array(3), new Float64Array(3 * lr.d));
  assert.equal(li.bilinear.d1 * li.bilinear.d2, lr.d);
});

test('every linear algorithm returns valid arms on every environment', () => {
  const K = 7, T = 200;
  for (const env of M.ENVIRONMENTS) {
    const d = env.d || 1;
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K), ctx = new Float64Array(T * K * d).fill(d > 1 ? 0 : 1);
    const info = env.fill(K, T, M.Rng(11), means, rewards, ctx) || {};
    for (const id of LINEAR_IDS) {
      const agent = M.ALGO_BY_ID[id].create(K, { T, env, params: M.defaultParams(id), rng: M.Rng(1), d, info });
      for (let t = 0; t < T; t++) {
        const a = agent.select(t + 1, ctx.subarray(t * K * d, (t + 1) * K * d));
        assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${id} on ${env.id} returned ${a} at t=${t + 1}`);
        const r = env.rested ? rewards[a] : rewards[t * K + a];
        assert.ok(Number.isFinite(r));
        agent.update(a, r, t + 1);
      }
    }
  }
});

test('linear algorithms also run with no context and no d (core test harness style)', () => {
  const K = 5, T = 120, env = M.ENV_BY_ID.bernoulli;
  for (const id of LINEAR_IDS) {
    const agent = M.ALGO_BY_ID[id].create(K, { T, env, params: M.defaultParams(id), rng: M.Rng(2) });
    const rng = M.Rng(3);
    for (let t = 1; t <= T; t++) {
      const a = agent.select(t);
      assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${id} returned ${a}`);
      agent.update(a, rng.uniform() < 0.5 ? 1 : 0, t);
    }
  }
});

test('the runner produces finite regret for every linear algorithm on the new environments', () => {
  for (const env of LINEAR_ENVS) {
    const res = sim(env, LINEAR_IDS, 150, 2, 1, 6);
    for (const r of res) assert.ok(Number.isFinite(r.finalRegret), `${r.id} on ${env}: ${r.finalRegret}`);
  }
});

test('OFUL and LinTS clearly beat context-free Thompson sampling and UCB1 on a shared-θ linear bandit', () => {
  const res = sim('linear-shared', ['oful', 'lin-ts', 'thompson', 'ucb1']);
  for (const lin of ['oful', 'lin-ts']) for (const base of ['thompson', 'ucb1']) {
    assert.ok(final(res, lin) < final(res, base) / 4, `${lin} ${final(res, lin)} vs ${base} ${final(res, base)}`);
  }
});

test('GLM methods beat UCB1 on the logistic bandit', () => {
  const ids = ['glm-ucb', 'ucb-glm', 'logistic-ucb1', 'logistic-ucb2', 'ofuglb'];
  const res = sim('logistic', [...ids, 'ucb1']);
  const ucb1 = final(res, 'ucb1');
  for (const id of ids) assert.ok(final(res, id) < ucb1 / 3, `${id} ${final(res, id)} vs UCB1 ${ucb1}`);
});

test('Lasso-type methods exploit sparsity', () => {
  const res = sim('sparse', ['lasso-bandit', 'dr-lasso', 'estc', 'oful', 'ucb1']);
  const oful = final(res, 'oful'), ucb1 = final(res, 'ucb1');
  assert.ok(final(res, 'lasso-bandit') < oful, `Lasso Bandit ${final(res, 'lasso-bandit')} vs OFUL ${oful}`);
  assert.ok(final(res, 'estc') < oful, `ESTC ${final(res, 'estc')} vs OFUL ${oful}`);
  assert.ok(final(res, 'dr-lasso') < ucb1 / 2, `DR-Lasso ${final(res, 'dr-lasso')} vs UCB1 ${ucb1}`);
});

test('dLinUCB and SW-LinUCB track a switching θ better than OFUL', () => {
  const res = sim('linear-drift', ['dlinucb', 'sw-linucb', 'oful'], 3000, 10);
  const oful = final(res, 'oful');
  for (const id of ['dlinucb', 'sw-linucb']) assert.ok(final(res, id) < oful / 1.5, `${id} ${final(res, id)} vs OFUL ${oful}`);
});

test('SpectralUCB beats UCB1 on graph-smooth rewards; G-optimal elimination learns a fixed arm set', () => {
  const g = sim('graph-smooth', ['spectral-ucb', 'ucb1'], 1000, 20, 5, 30);
  assert.ok(final(g, 'spectral-ucb') < final(g, 'ucb1'), `SpectralUCB ${final(g, 'spectral-ucb')} vs UCB1 ${final(g, 'ucb1')}`);
  const f = sim('linear-fixed', ['g-optimal', 'random'], 2000, 20);
  assert.ok(final(f, 'g-optimal') < final(f, 'random') / 3, `G-optimal ${final(f, 'g-optimal')} vs random ${final(f, 'random')}`);
});

test('Frank–Wolfe G-optimal design meets the Kiefer–Wolfowitz bound max g ≈ d', () => {
  const K = 12, d = 4, X = new Float64Array(K * d), rng = M.Rng(4);
  for (let i = 0; i < X.length; i++) X[i] = rng.normal();
  const sp = M.linear.space(K, { d });
  sp.init(X);
  const arms = Array.from({ length: K }, (_, i) => i);
  const { pi, deff } = M.linear.gOptimalDesign(arms, X, sp);
  const s = pi.reduce((x, y) => x + y, 0);
  assert.ok(Math.abs(s - 1) < 1e-9);
  assert.ok(Math.abs(deff - d) < 0.05, `Σπg = ${deff}`);
});

test('Lasso coordinate descent recovers a sparse vector', () => {
  const d = 20, n = 200, rng = M.Rng(9), beta = new Float64Array(d), G = new Float64Array(d * d), c = new Float64Array(d);
  const truth = new Float64Array(d); truth[3] = 1; truth[11] = -0.7;
  const x = new Float64Array(d);
  for (let s = 0; s < n; s++) {
    for (let j = 0; j < d; j++) x[j] = rng.normal();
    let y = 0.1 * rng.normal();
    for (let j = 0; j < d; j++) y += truth[j] * x[j];
    for (let i = 0; i < d; i++) { c[i] += y * x[i]; for (let j = 0; j < d; j++) G[i * d + j] += x[i] * x[j]; }
  }
  M.linear.lasso(G, c, n, d, 0.02, beta);
  for (let j = 0; j < d; j++) assert.ok(Math.abs(beta[j] - truth[j]) < 0.08, `β[${j}] = ${beta[j]}`);
});

test('keeps 1,000-step K = 10 runs fast on the shared linear bandit', () => {
  const res = sim('linear-shared', LINEAR_IDS, 1000, 3);
  for (const r of res) assert.ok(r.msPerRun < 150, `${r.id} took ${r.msPerRun.toFixed(1)} ms per run`);
});
