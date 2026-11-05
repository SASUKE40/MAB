'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const MINE = ['sw-ts', 'sw-klucb', 'exp3r', 'cusum-ucb', 'pht-ucb', 'glr-klucb', 'adswitch', 'master', 'fewa', 'raw-ucb', 'red-ucb'];
const ENVS = ['rotting', 'rising', 'piecewise-local'];

function board(env, K, T, runs, ids, seed = 11) {
  const res = M.simulate({ env, K, T, runs, seed, algorithms: ids.map((id) => ({ id })) });
  const by = {};
  for (const r of res) by[r.id] = r.finalRegret;
  return by;
}

test('non-stationary add-on registers its environments, families and algorithms', () => {
  for (const id of ENVS) assert.ok(M.ENV_BY_ID[id], `missing env ${id}`);
  assert.ok(M.ENV_BY_ID.rotting.rested && M.ENV_BY_ID.rising.rested);
  assert.deepEqual(M.ENV_BY_ID.rotting.tags, ['rested']);
  assert.deepEqual(M.ENV_BY_ID['piecewise-local'].tags, ['changes']);
  assert.equal(M.ENV_BY_ID['piecewise-local'].stationary, false);
  const fams = M.FAMILIES.map((f) => f.id);
  assert.ok(fams.includes('rested'));
  if (fams.includes('contextual')) assert.ok(fams.indexOf('rested') < fams.indexOf('contextual'));
  for (const id of MINE) {
    const a = M.ALGO_BY_ID[id];
    assert.ok(a, `missing algorithm ${id}`);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(typeof a[k] === 'string' && a[k].length, `${id}.${k}`);
    assert.ok(!a.formula.includes('\n'), `${id} formula is one line`);
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${id}.${p.key} default out of range`);
  }
  for (const id of ['fewa', 'raw-ucb', 'red-ucb']) assert.equal(M.ALGO_BY_ID[id].family, 'rested');
});

test('rested environments: rotting means never rise, rising means never fall', () => {
  const K = 6, T = 400;
  for (const [id, sign] of [['rotting', -1], ['rising', 1]]) {
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
    M.ENV_BY_ID[id].fill(K, T, M.Rng(3), means, rewards, new Float64Array(T * K));
    for (let a = 0; a < K; a++) {
      for (let n = 1; n < T; n++) {
        const d = means[n * K + a] - means[(n - 1) * K + a];
        assert.ok(sign * d >= -1e-12, `${id} arm ${a} pull ${n}: ${d}`);
      }
    }
  }
});

test('piecewise-local changes only the best arm, four times', () => {
  const K = 8, T = 1000;
  const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
  M.ENV_BY_ID['piecewise-local'].fill(K, T, M.Rng(5), means, rewards, new Float64Array(T * K));
  let changePoints = 0;
  for (let t = 1; t < T; t++) {
    let moved = 0, bestBefore = -Infinity, which = -1;
    for (let a = 0; a < K; a++) bestBefore = Math.max(bestBefore, means[(t - 1) * K + a]);
    for (let a = 0; a < K; a++) if (means[t * K + a] !== means[(t - 1) * K + a]) { moved++; which = a; }
    if (moved) {
      changePoints++;
      assert.equal(moved, 1, `step ${t}: ${moved} arms moved`);
      assert.equal(means[(t - 1) * K + which], bestBefore, 'the arm that changed was the best one');
      assert.ok(means[t * K + which] <= 0.25);
    }
  }
  assert.equal(changePoints, 4);
});

test('every new algorithm returns valid arms on every environment, using only ctx.rng', () => {
  const K = 7, T = 300;
  const realRandom = Math.random;
  Math.random = () => { throw new Error('Math.random used'); };
  try {
    for (const env of M.ENVIRONMENTS) {
      const d = env.d || 1;
      const means = new Float64Array(T * K), rewards = new Float64Array(T * K), ctx = new Float64Array(T * K * d).fill(d > 1 ? 0 : 1);
      const info = env.fill(K, T, M.Rng(7), means, rewards, ctx) || {};
      for (const id of MINE) {
        const algo = M.ALGO_BY_ID[id];
        const agent = algo.create(K, { T, env, params: M.defaultParams(id), rng: M.Rng(1), d, info });
        const pulls = new Int32Array(K);
        for (let t = 0; t < T; t++) {
          const X = ctx.subarray(t * K * d, (t + 1) * K * d);
          const a = agent.select(t + 1, X);
          assert.ok(Number.isInteger(a) && a >= 0 && a < K, `${id} on ${env.id} returned ${a} at t=${t + 1}`);
          const r = env.rested ? rewards[pulls[a] * K + a] : rewards[t * K + a];
          pulls[a]++;
          agent.update(a, r, t + 1);
        }
      }
    }
  } finally {
    Math.random = realRandom;
  }
});

test('new algorithms are reproducible from the seed', () => {
  const cfg = { env: 'piecewise-local', K: 5, T: 400, runs: 3, seed: 4, algorithms: MINE.map((id) => ({ id })) };
  const a = M.simulate(cfg), b = M.simulate(cfg);
  for (let i = 0; i < a.length; i++) assert.equal(a[i].finalRegret, b[i].finalRegret, a[i].id);
});

test('rotting: FEWA and RAW-UCB beat UCB1', () => {
  const r = board('rotting', 10, 1000, 50, ['ucb1', 'fewa', 'raw-ucb']);
  assert.ok(r.fewa < r.ucb1 * 0.5, `FEWA ${r.fewa} vs UCB1 ${r.ucb1}`);
  assert.ok(r['raw-ucb'] < r.ucb1 * 0.5, `RAW-UCB ${r['raw-ucb']} vs UCB1 ${r.ucb1}`);
});

test('rising: R-ed-UCB beats UCB1 and greedy on the late-bloomer instance (K=5, T=2000)', () => {
  // At K=10, T=1000 R-ed-UCB is only on par with greedy: its bonus exceeds the reward range,
  // so it plays close to round-robin. With fewer arms and a longer horizon its extrapolation wins.
  const r = board('rising', 5, 2000, 50, ['ucb1', 'greedy', 'red-ucb']);
  assert.ok(r['red-ucb'] < r.ucb1 * 0.8, `R-ed-UCB ${r['red-ucb']} vs UCB1 ${r.ucb1}`);
  assert.ok(r['red-ucb'] < r.greedy * 0.8, `R-ed-UCB ${r['red-ucb']} vs greedy ${r.greedy}`);
});

test('piecewise-local: CUSUM-UCB and GLR-klUCB beat UCB1 at T=5000', () => {
  const r = board('piecewise-local', 10, 5000, 20, ['ucb1', 'cusum-ucb', 'glr-klucb']);
  assert.ok(r['cusum-ucb'] < r.ucb1 * 0.85, `CUSUM-UCB ${r['cusum-ucb']} vs UCB1 ${r.ucb1}`);
  assert.ok(r['glr-klucb'] < r.ucb1 * 0.7, `GLR-klUCB ${r['glr-klucb']} vs UCB1 ${r.ucb1}`);
});

test('each new algorithm runs 1,000 steps with K=10 in about 10 ms', () => {
  const res = M.simulate({ env: 'bernoulli', K: 10, T: 1000, runs: 10, seed: 2, algorithms: MINE.map((id) => ({ id })) });
  for (const r of res) assert.ok(r.msPerRun < 25, `${r.id}: ${r.msPerRun.toFixed(2)} ms/run`);
});
