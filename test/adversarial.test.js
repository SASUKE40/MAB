'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');
const ADV = require('../src/algos/adversarial.js');

const MINE = ADV.algorithms.map((a) => a.id);
const ENVS = ADV.environments.map((e) => e.id);

function drive(algo, env, K, T, seed) {
  const D = env.d || 1;
  const means = new Float64Array(T * K), rewards = new Float64Array(T * K), ctx = new Float64Array(T * K * D).fill(D > 1 ? 0 : 1);
  const info = env.fill(K, T, M.Rng(seed), means, rewards, ctx) || {};
  const agent = algo.create(K, { T, env, params: M.defaultParams(algo.id), rng: M.Rng(seed + 1), d: D, info });
  const X = new Float64Array(K * D), pulls = new Int32Array(K);
  for (let t = 0; t < T; t++) {
    X.set(ctx.subarray(t * K * D, (t + 1) * K * D));
    const a = agent.select(t + 1, X);
    assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${algo.id} on ${env.id} returned ${a} at t=${t + 1}`);
    const row = env.rested ? pulls[a]++ : t;
    agent.update(a, rewards[row * K + a], t + 1);
  }
  return agent;
}

test('adversarial add-on registers its algorithms, environments and family', () => {
  for (const id of MINE) assert.ok(M.ALGO_BY_ID[id], id);
  for (const id of ENVS) assert.ok(M.ENV_BY_ID[id], id);
  assert.ok(M.FAMILIES.some((f) => f.id === 'ftrl'));
  for (const a of ADV.algorithms) {
    assert.ok(M.FAMILIES.some((f) => f.id === a.family), `${a.id} family`);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(a[k], `${a.id} is missing ${k}`);
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${a.id}.${p.key} default out of range`);
  }
  for (const e of ADV.environments) {
    assert.ok(e.tags.includes('adversarial'));
    for (const k of ['name', 'short', 'description']) assert.ok(e[k], `${e.id} is missing ${k}`);
  }
});

test('new algorithms return valid arms on every environment', () => {
  for (const env of M.ENVIRONMENTS) {
    for (const id of MINE) drive(M.ALGO_BY_ID[id], env, 7, 300, 11);
  }
});

test('every algorithm returns valid arms on the new environments', () => {
  for (const id of ENVS) {
    for (const algo of M.ALGORITHMS) drive(algo, M.ENV_BY_ID[id], 7, 300, 5);
  }
});

test('new environments follow the fill() contract', () => {
  for (const id of ENVS) {
    const env = M.ENV_BY_ID[id];
    for (const [K, T] of [[2, 7], [7, 300], [10, 2000]]) {
      const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
      env.fill(K, T, M.Rng(3), means, rewards, new Float64Array(T * K).fill(1));
      for (let i = 0; i < T * K; i++) {
        assert.ok(means[i] >= 0 && means[i] <= 1, `${id} mean ${means[i]}`);
        assert.ok(rewards[i] === 0 || rewards[i] === 1, `${id} reward ${rewards[i]}`);
      }
    }
  }
  // stochastically constrained: the best arm and every gap stay fixed while the level moves
  const K = 6, T = 1000, means = new Float64Array(T * K), rewards = new Float64Array(T * K);
  M.ENV_BY_ID['adv-constrained'].fill(K, T, M.Rng(4), means, rewards);
  const gap0 = Array.from({ length: K }, (_, a) => means[a] - means[0]);
  const levels = new Set();
  for (let t = 0; t < T; t++) {
    levels.add(means[t * K].toFixed(6));
    for (let a = 0; a < K; a++) assert.ok(Math.abs(means[t * K + a] - means[t * K] - gap0[a]) < 1e-12);
  }
  assert.ok(levels.size === 2, 'the common level should switch between two values');
});

test('normalization helpers solve their equations', () => {
  const { logBarrierNormalize, tsallisNormalize, expPerturbProb } = ADV.helpers;
  const rng = M.Rng(9);
  for (let rep = 0; rep < 50; rep++) {
    const K = 2 + rng.int(20);
    const d = new Float64Array(K), c = new Float64Array(K), p = new Float64Array(K);
    for (let i = 0; i < K; i++) { d[i] = 50 * rng.uniform() - 10; c[i] = 0.01 + rng.uniform(); }
    const lam = logBarrierNormalize(d, c, p);
    let s = 0;
    for (let i = 0; i < K; i++) { assert.ok(d[i] + c[i] * lam > 0); s += 1 / (d[i] + c[i] * lam); }
    assert.ok(Math.abs(s - 1) < 1e-9, `log-barrier sum ${s}`);
    const w = d.map((x) => x / 10);
    tsallisNormalize(w, p, 0.8);
    assert.ok(Math.abs(p.reduce((x, y) => x + y, 0) - 1) < 1e-12);
    for (let i = 1; i < K; i++) if (w[i] > w[0]) assert.ok(p[i] <= p[0] + 1e-15);
  }
  // exact FPL probabilities sum to one and match Monte Carlo
  const K = 5, L = Float64Array.from([3, 0, 1.5, 10, 0.2]), eta = 0.7, sv = new Float64Array(K);
  const p = Array.from({ length: K }, (_, a) => expPerturbProb(L, eta, a, sv));
  assert.ok(Math.abs(p.reduce((x, y) => x + y, 0) - 1) < 1e-10);
  const cnt = new Float64Array(K), N = 200000, r = M.Rng(1);
  for (let n = 0; n < N; n++) {
    let b = 0, bv = Infinity;
    for (let i = 0; i < K; i++) { const v = eta * L[i] + Math.log(1 - r.uniform()); if (v < bv) { bv = v; b = i; } }
    cnt[b]++;
  }
  for (let a = 0; a < K; a++) assert.ok(Math.abs(cnt[a] / N - p[a]) < 0.005, `arm ${a}: ${cnt[a] / N} vs ${p[a]}`);
});

test('on the switching trap, adversarial methods beat UCB1 and greedy', () => {
  const strong = ['ftrl-logbarrier', 'fpl', 'ftpl-frechet', 'ftpl-gumbel', 'omd-tsallis', 'broad-omd', 'sao', 'tsallis-inf'];
  const res = M.simulate({ env: 'adv-switch', K: 10, T: 2000, runs: 40, seed: 21, algorithms: ['ucb1', 'greedy', 'tsallis-inf', ...MINE].map((id) => ({ id })) });
  const by = Object.fromEntries(res.map((r) => [r.id, r.finalRegret]));
  for (const id of MINE) assert.ok(by[id] < by.greedy, `${id}: ${by[id]} vs greedy ${by.greedy}`);
  for (const id of strong) assert.ok(by[id] < 0.9 * by.ucb1, `${id}: ${by[id]} vs UCB1 ${by.ucb1}`);
});

test('on the stochastically constrained adversary, FTRL methods beat UCB1', () => {
  const ids = ['ucb1', 'ftrl-logbarrier', 'broad-omd', 'omd-tsallis', 'fpl'];
  const res = M.simulate({ env: 'adv-constrained', K: 10, T: 2000, runs: 40, seed: 5, algorithms: ids.map((id) => ({ id })) });
  const ucb = res.find((r) => r.id === 'ucb1').finalRegret;
  for (const r of res) if (r.id !== 'ucb1') assert.ok(r.finalRegret < ucb, `${r.id}: ${r.finalRegret} vs UCB1 ${ucb}`);
});

test('on Bernoulli arms every new algorithm beats random', () => {
  const res = M.simulate({ env: 'bernoulli', K: 10, T: 2000, runs: 30, seed: 3, algorithms: ['random', ...MINE].map((id) => ({ id })) });
  const rnd = res.find((r) => r.id === 'random').finalRegret;
  for (const r of res) if (r.id !== 'random') assert.ok(r.finalRegret < 0.8 * rnd, `${r.id}: ${r.finalRegret} vs random ${rnd}`);
  // the best-of-both-worlds members are close to the stochastic optimum here
  for (const id of ['omd-tsallis', 'ftpl-frechet', 'ftrl-logbarrier']) {
    assert.ok(res.find((r) => r.id === id).finalRegret < rnd / 4, id);
  }
});

test('SAO stays stochastic on i.i.d. arms and falls back on adversarial ones', () => {
  const K = 10, T = 2000;
  let stochSwitches = 0, advSwitches = 0;
  for (let run = 0; run < 20; run++) {
    stochSwitches += drive(M.ALGO_BY_ID.sao, M.ENV_BY_ID.bernoulli, K, T, 100 + run).switchedAt ? 1 : 0;
    advSwitches += drive(M.ALGO_BY_ID.sao, M.ENV_BY_ID['adv-constrained'], K, T, 200 + run).switchedAt ? 1 : 0;
  }
  assert.equal(stochSwitches, 0);
  assert.ok(advSwitches >= 15, `SAO switched in only ${advSwitches}/20 adversarial runs`);
});

test('new algorithms are fast enough for the dashboard', () => {
  const res = M.simulate({ env: 'bernoulli', K: 10, T: 1000, runs: 10, seed: 1, algorithms: MINE.map((id) => ({ id })) });
  for (const r of res) assert.ok(r.msPerRun < 40, `${r.id}: ${r.msPerRun.toFixed(1)} ms per 1000-step run`);
});
