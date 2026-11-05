'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const IDS = ['efirst', 'vdbe', 'ebmc', 'gumbel-max', 'ao-ucb', 'ocucb', 'cp-ucb', 'dmed', 'besa', 'obs', 'kg-star', 'poker'];

test('classical algorithms are registered with complete metadata', () => {
  const fams = new Set(M.FAMILIES.map((f) => f.id));
  for (const id of IDS) {
    const a = M.ALGO_BY_ID[id];
    assert.ok(a, `${id} is not registered`);
    assert.ok(fams.has(a.family), `${id} has unknown family ${a.family}`);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(a[k], `${id} is missing ${k}`);
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${id}.${p.key} default out of range`);
  }
});

test('classical algorithms return valid arms on every environment', () => {
  const K = 7, T = 300;
  for (const env of M.ENVIRONMENTS) {
    const d = env.d || 1;
    for (const id of IDS) {
      const rng = M.Rng(2);
      const agent = M.ALGO_BY_ID[id].create(K, { T, env, params: M.defaultParams(id), rng: M.Rng(1), d, info: {} });
      const X = new Float64Array(K * d);
      for (let t = 1; t <= T; t++) {
        for (let i = 0; i < X.length; i++) X[i] = d > 1 ? rng.normal() : 1;
        const a = agent.select(t, X);
        assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${id} on ${env.id} returned ${a} at t=${t}`);
        const r = env.type === 'bernoulli' ? (rng.uniform() < 0.3 + 0.05 * a ? 1 : 0) : 0.2 * a + rng.normal();
        agent.update(a, r, t);
      }
    }
  }
});

test('classical algorithms run through the simulator on every environment', () => {
  for (const env of M.ENVIRONMENTS) {
    const res = M.simulate({ env: env.id, K: 7, T: 300, runs: 2, seed: 1, algorithms: IDS.map((id) => ({ id })) });
    for (const r of res) assert.ok(Number.isFinite(r.finalRegret), `${r.id} on ${env.id}: ${r.finalRegret}`);
  }
});

test('classical algorithms beat Random clearly on a Bernoulli bandit', () => {
  const res = M.simulate({
    env: 'bernoulli', K: 10, T: 1000, runs: 30, seed: 3,
    algorithms: ['random', ...IDS].map((id) => ({ id })),
  });
  const random = res.find((r) => r.id === 'random').finalRegret;
  // Gumbel-max is fixed-temperature softmax; it only has to beat Random by a smaller margin
  for (const r of res) {
    if (r.id === 'random') continue;
    const factor = r.id === 'gumbel-max' ? 2 : 4;
    assert.ok(r.finalRegret < random / factor, `${r.id}: ${r.finalRegret.toFixed(1)} vs random ${random.toFixed(1)}`);
  }
});

test('Gumbel-max samples the Boltzmann distribution', () => {
  const env = M.ENV_BY_ID.gaussian, K = 3, tau = 0.5;
  const agent = M.ALGO_BY_ID['gumbel-max'].create(K, { T: 10, env, params: { tau }, rng: M.Rng(4), d: 1, info: {} });
  const q = [0, 0.5, 1];
  for (let a = 0; a < K; a++) agent.update(a, q[a], a + 1);
  const cnt = [0, 0, 0], N = 40000;
  for (let i = 0; i < N; i++) cnt[agent.select(4)]++;
  const w = q.map((x) => Math.exp(x / (tau * env.scale))), z = w.reduce((x, y) => x + y, 0);
  for (let a = 0; a < K; a++) assert.ok(Math.abs(cnt[a] / N - w[a] / z) < 0.01, `arm ${a}: ${cnt[a] / N} vs ${w[a] / z}`);
});

test('optimism and divergence additions beat UCB1 on Bernoulli arms', () => {
  const ids = ['ucb1', 'ocucb', 'cp-ucb', 'dmed', 'besa', 'obs'];
  const res = M.simulate({ env: 'bernoulli', K: 10, T: 2000, runs: 40, seed: 11, algorithms: ids.map((id) => ({ id })) });
  const ucb1 = res.find((r) => r.id === 'ucb1').finalRegret;
  for (const r of res) if (r.id !== 'ucb1') assert.ok(r.finalRegret < ucb1, `${r.id}: ${r.finalRegret} vs UCB1 ${ucb1}`);
});
