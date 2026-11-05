'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');
const addon = require('../src/algos/policy-neural.js');

const IDS = addon.algorithms.map((a) => a.id);
const NEURAL_KERNEL = ['neuralucb', 'neuralts', 'neural-linucb', 'ee-net', 'enn-ts', 'ensemble-sampling', 'kernelucb'];

/** Drive an agent through one environment's real contexts and rewards, checking every arm. */
function drive(algo, env, K, T, seed, params) {
  const d = env.d || 1;
  const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
  const contexts = new Float64Array(T * K * d).fill(d > 1 ? 0 : 1);
  const info = env.fill(K, T, M.Rng(seed), means, rewards, contexts) || {};
  const agent = algo.create(K, { T, env, params: Object.assign(M.defaultParams(algo.id), params), rng: M.Rng(seed + 1), d, info });
  const X = new Float64Array(K * d), pulls = new Int32Array(K);
  for (let t = 0; t < T; t++) {
    X.set(contexts.subarray(t * K * d, (t + 1) * K * d));
    const a = agent.select(t + 1, X);
    assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${algo.id} on ${env.id} returned ${a} at t=${t + 1}`);
    // rested environments index rewards by the arm's own pull count
    const row = env.rested ? Math.min(pulls[a]++, T - 1) : t;
    agent.update(a, rewards[row * K + a], t + 1);
  }
}

test('policy/kernel/neural add-on registers its families, environments and algorithms', () => {
  for (const f of ['policy', 'kernel', 'neural']) assert.ok(M.FAMILIES.some((x) => x.id === f), f);
  for (const e of ['nonlinear', 'nonlinear-binary']) {
    const env = M.ENV_BY_ID[e];
    assert.ok(env && env.contextual && env.d === 5 && env.tags.includes('context'), e);
  }
  assert.equal(IDS.length, 22);
  for (const id of IDS) {
    const a = M.ALGO_BY_ID[id];
    assert.ok(a, id);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(a[k], `${id} is missing ${k}`);
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${id}.${p.key} default out of range`);
  }
});

test('non-linear environments fill means, rewards and a shared context', () => {
  const K = 4, T = 50, d = 5;
  for (const id of ['nonlinear', 'nonlinear-binary']) {
    const env = M.ENV_BY_ID[id];
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K), ctx = new Float64Array(T * K * d);
    env.fill(K, T, M.Rng(3), means, rewards, ctx);
    for (let t = 0; t < T; t++) {
      for (let a = 1; a < K; a++) for (let j = 0; j < d; j++) assert.equal(ctx[(t * K + a) * d + j], ctx[t * K * d + j]);
      for (let a = 0; a < K; a++) {
        const m = means[t * K + a], r = rewards[t * K + a];
        assert.ok(Number.isFinite(m) && Number.isFinite(r));
        if (env.type === 'bernoulli') assert.ok(m > 0 && m < 1 && (r === 0 || r === 1));
      }
    }
  }
});

test('every policy/kernel/neural algorithm returns valid arms on every environment', () => {
  for (const env of M.ENVIRONMENTS) {
    for (const id of IDS) drive(M.ALGO_BY_ID[id], env, 5, 150, 11);
  }
});

test('they also run without contexts (select(t), no d) and at K = 2 and K = 50', () => {
  const env = M.ENV_BY_ID.bernoulli;
  for (const id of IDS) {
    const algo = M.ALGO_BY_ID[id];
    const agent = algo.create(7, { T: 100, env, params: M.defaultParams(id), rng: M.Rng(4) });
    const rng = M.Rng(5);
    for (let t = 1; t <= 100; t++) {
      const a = agent.select(t);
      assert.ok(Number.isInteger(a) && a >= 0 && a < 7, `${id} returned ${a}`);
      agent.update(a, rng.uniform() < 0.5 ? 1 : 0, t);
    }
    drive(algo, M.ENV_BY_ID.contextual, 2, 80, 6);
    drive(algo, M.ENV_BY_ID.nonlinear, 50, 120, 7);
  }
});

test('KernelUCB keeps working once its sliding dictionary is full', () => {
  const [r] = M.simulate({ env: 'nonlinear', K: 3, T: 400, runs: 2, seed: 1, algorithms: [{ id: 'kernelucb', params: { cap: 10 } }] });
  assert.ok(Number.isFinite(r.finalRegret));
  const [s] = M.simulate({ env: 'gaussian', K: 3, T: 400, runs: 2, seed: 1, algorithms: [{ id: 'supkernelucb', params: { cap: 10 } }] });
  assert.ok(Number.isFinite(s.finalRegret));
});

test('results are reproducible for a seed', () => {
  for (const id of ['enn-ts', 'fgts', 'ilovetoconbandits', 'tree-bootstrap']) {
    const cfg = { env: 'nonlinear', K: 4, T: 200, runs: 2, seed: 5, algorithms: [{ id }] };
    assert.equal(M.simulate(cfg)[0].finalRegret, M.simulate(cfg)[0].finalRegret, id);
  }
});

test('neural and kernel methods beat LinUCB and Thompson sampling on the non-linear environment', () => {
  const ids = ['linucb', 'thompson', ...NEURAL_KERNEL];
  const res = M.simulate({ env: 'nonlinear', K: 10, T: 2000, runs: 6, seed: 7, algorithms: ids.map((id) => ({ id })) });
  const by = Object.fromEntries(res.map((r) => [r.id, r]));
  const bar = Math.min(by.linucb.finalRegret, by.thompson.finalRegret);
  for (const id of NEURAL_KERNEL) assert.ok(by[id].finalRegret < bar, `${id} ${by[id].finalRegret.toFixed(1)} vs best baseline ${bar.toFixed(1)}`);
  // the best of them should be clearly better, not just marginally
  assert.ok(Math.min(...NEURAL_KERNEL.map((id) => by[id].finalRegret)) < 0.75 * by.linucb.finalRegret);
});

test('SquareCB and EXP4 beat random on the linear contextual environment', () => {
  const res = M.simulate({ env: 'contextual', K: 10, T: 1000, runs: 20, seed: 2, algorithms: ['random', 'squarecb', 'exp4'].map((id) => ({ id })) });
  const by = Object.fromEntries(res.map((r) => [r.id, r.finalRegret]));
  assert.ok(by.squarecb < by.random / 2, `SquareCB ${by.squarecb} vs random ${by.random}`);
  assert.ok(by.exp4 < 0.8 * by.random, `EXP4 ${by.exp4} vs random ${by.random}`);
});

test('every policy-class method learns something on a Bernoulli bandit', () => {
  const ids = ['random', 'exp4', 'exp4p', 'epoch-greedy', 'ilovetoconbandits', 'rucb', 'online-cover', 'bootstrap-ts',
    'squarecb', 'fastcb', 'smoothcb', 'fgts', 'falcon', 'corral'];
  const res = M.simulate({ env: 'bernoulli', K: 10, T: 1000, runs: 10, seed: 3, algorithms: ids.map((id) => ({ id })) });
  const random = res.find((r) => r.id === 'random').finalRegret;
  for (const r of res) if (r.id !== 'random') assert.ok(r.finalRegret < 0.6 * random, `${r.id}: ${r.finalRegret} vs random ${random}`);
});
