'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/mab.js');

test('RNG is deterministic and in range', () => {
  const a = M.Rng(123), b = M.Rng(123);
  for (let i = 0; i < 1000; i++) {
    const x = a.uniform();
    assert.equal(x, b.uniform());
    assert.ok(x >= 0 && x < 1);
  }
});

test('Beta sampler has the right mean', () => {
  const r = M.Rng(5);
  let s = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) s += r.beta(3, 7);
  assert.ok(Math.abs(s / n - 0.3) < 0.01, `mean ${s / n}`);
});

test('normInv inverts the normal CDF at known points', () => {
  assert.ok(Math.abs(M.normInv(0.5)) < 1e-9);
  assert.ok(Math.abs(M.normInv(0.975) - 1.959964) < 1e-4);
  assert.ok(Math.abs(M.normInv(0.01) + 2.326348) < 1e-4);
});

test('KL-UCB index is above the mean and satisfies the KL bound', () => {
  const q = M.klUcbBern(0.4, 20, Math.log(100));
  assert.ok(q > 0.4 && q < 1);
  assert.ok(Math.abs(20 * M.klBern(0.4, q) - Math.log(100)) < 1e-3);
});

test('every algorithm returns valid arms on every environment', () => {
  for (const env of M.ENVIRONMENTS) {
    for (const algo of M.ALGORITHMS) {
      const K = 7, T = 300;
      const agent = algo.create(K, { T, env, params: M.defaultParams(algo.id), rng: M.Rng(1) });
      const rng = M.Rng(2);
      for (let t = 1; t <= T; t++) {
        const a = agent.select(t);
        assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${algo.id} on ${env.id} returned ${a}`);
        agent.update(a, env.type === 'bernoulli' ? (rng.uniform() < 0.5 ? 1 : 0) : rng.normal(), t);
      }
    }
  }
});

test('runner is reproducible and independent of the lineup', () => {
  const base = { env: 'bernoulli', K: 5, T: 200, runs: 10, seed: 9 };
  const a = M.simulate({ ...base, algorithms: [{ id: 'thompson' }] });
  const b = M.simulate({ ...base, algorithms: [{ id: 'ucb1' }, { id: 'thompson' }] });
  assert.equal(a[0].finalRegret, b[1].finalRegret);
});

test('stationary regret is non-decreasing and pull shares sum to 1', () => {
  const [r] = M.simulate({ env: 'gaussian', K: 10, T: 500, runs: 20, seed: 1, algorithms: [{ id: 'egreedy' }] });
  for (let t = 1; t < 500; t++) assert.ok(r.regret[t] >= r.regret[t - 1] - 1e-12);
  const s = r.pullShare.reduce((x, y) => x + y, 0);
  assert.ok(Math.abs(s - 1) < 1e-9);
});

test('learning algorithms beat random on a Bernoulli bandit', () => {
  const res = M.simulate({
    env: 'bernoulli', K: 10, T: 1000, runs: 50, seed: 3,
    algorithms: ['random', 'ucb1', 'klucb', 'thompson', 'egreedy'].map((id) => ({ id })),
  });
  const random = res.find((r) => r.id === 'random').finalRegret;
  for (const r of res) if (r.id !== 'random') assert.ok(r.finalRegret < random / 2, `${r.id}: ${r.finalRegret} vs ${random}`);
  const ts = res.find((r) => r.id === 'thompson');
  assert.ok(ts.optimalTail > 0.8, `TS best-arm rate ${ts.optimalTail}`);
});

test('Explore-then-commit explores round-robin then commits', () => {
  const env = M.ENV_BY_ID.bernoulli;
  const agent = M.ALGO_BY_ID.etc.create(3, { T: 100, env, params: { m: 2 }, rng: M.Rng(1) });
  const picks = [];
  for (let t = 1; t <= 10; t++) {
    const a = agent.select(t);
    picks.push(a);
    agent.update(a, a === 1 ? 1 : 0, t);
  }
  assert.deepEqual(picks.slice(0, 6), [0, 1, 2, 0, 1, 2]);
  assert.deepEqual(picks.slice(6), [1, 1, 1, 1]);
});

test('Gittins boundary function is non-negative and increasing', () => {
  let prev = -Infinity;
  for (const s of [0.01, 0.1, 0.2, 0.5, 1, 3, 5, 10, 15, 50, 1000]) {
    const v = M.gittinsPsi(s);
    assert.ok(v >= 0 && v >= prev - 0.05, `psi(${s}) = ${v}`);
    prev = v;
  }
});

test('normCdf matches known values', () => {
  assert.ok(Math.abs(M.normCdf(0) - 0.5) < 1e-7);
  assert.ok(Math.abs(M.normCdf(1.959964) - 0.975) < 1e-6);
  assert.ok(Math.abs(M.normCdf(-1) - 0.158655) < 1e-6);
});

test('every algorithm has complete metadata and a known family', () => {
  const fams = new Set(M.FAMILIES.map((f) => f.id));
  const ids = new Set();
  for (const a of M.ALGORITHMS) {
    assert.ok(!ids.has(a.id), `duplicate id ${a.id}`); ids.add(a.id);
    assert.ok(fams.has(a.family), `${a.id} has unknown family ${a.family}`);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(a[k], `${a.id} is missing ${k}`);
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${a.id}.${p.key} default out of range`);
  }
});

test('asymptotically optimal and Bayesian methods beat UCB1 on Bernoulli arms', () => {
  const ids = ['ucb1', 'klucb-plus', 'klucb-pp', 'imed', 'gittins', 'tsallis-inf'];
  const res = M.simulate({ env: 'bernoulli', K: 10, T: 2000, runs: 40, seed: 11, algorithms: ids.map((id) => ({ id })) });
  const ucb1 = res.find((r) => r.id === 'ucb1').finalRegret;
  for (const r of res) if (r.id !== 'ucb1') assert.ok(r.finalRegret < ucb1, `${r.id}: ${r.finalRegret} vs UCB1 ${ucb1}`);
});

test('sequential halving commits to a single arm after its budget', () => {
  const env = M.ENV_BY_ID.bernoulli;
  const agent = M.ALGO_BY_ID['seq-halving'].create(4, { T: 1000, env, params: { budget: 0.2 }, rng: M.Rng(3) });
  const picks = [];
  for (let t = 1; t <= 1000; t++) { const a = agent.select(t); picks.push(a); agent.update(a, a === 2 ? 1 : 0, t); }
  assert.ok(picks.slice(300).every((a) => a === 2));
});

test('change-detecting and forgetting methods beat stationary UCB1 on long drifting runs', () => {
  const res = M.simulate({ env: 'drift', K: 10, T: 5000, runs: 20, seed: 4, algorithms: ['ucb1', 'sw-ucb', 'dts'].map((id) => ({ id })) });
  const ucb1 = res.find((r) => r.id === 'ucb1').finalRegret;
  for (const r of res) if (r.id !== 'ucb1') assert.ok(r.finalRegret < ucb1, `${r.id}: ${r.finalRegret} vs UCB1 ${ucb1}`);
});

test('LinUCB uses the context and beats context-free methods on the contextual environment', () => {
  const res = M.simulate({ env: 'contextual', K: 10, T: 1000, runs: 20, seed: 2, algorithms: ['linucb', 'thompson', 'klucb'].map((id) => ({ id })) });
  const lin = res.find((r) => r.id === 'linucb').finalRegret;
  for (const r of res) if (r.id !== 'linucb') assert.ok(lin < r.finalRegret / 2, `LinUCB ${lin} vs ${r.id} ${r.finalRegret}`);
});
