'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const S = M.SETTINGS.find((s) => s.id === 'dueling');
const ids = (xs) => xs.map((x) => x.id);
const defaults = (a) => Object.fromEntries(a.params.map((p) => [p.key, p.default]));
const byId = (res) => Object.fromEntries(res.map((r) => [r.id, r]));

function run(env, algos, { K = 8, T = 2000, runs = 20, seed = 11, extra } = {}) {
  const r = S.createRunner({ env, K, T, runs, seed, extra, algorithms: algos.map((id) => (typeof id === 'string' ? { id } : id)) });
  r.step(runs);
  return byId(r.results());
}

test('dueling setting is registered with a complete description', () => {
  assert.ok(S, 'setting "dueling" not found in M.SETTINGS');
  assert.equal(S.name, 'Dueling bandits');
  for (const k of ['families', 'environments', 'algorithms', 'metrics', 'columns', 'tiles', 'presets']) assert.ok(Array.isArray(S[k]) && S[k].length, k);
  assert.equal(typeof S.createRunner, 'function');
  const fam = new Set(ids(S.families));
  const algoIds = ids(S.algorithms), envIds = ids(S.environments);
  assert.equal(new Set(algoIds).size, algoIds.length, 'algorithm ids unique');
  assert.equal(new Set(envIds).size, envIds.length, 'environment ids unique');
  for (const want of ['uniform', 'if', 'btm', 'savage', 'rucb', 'rcs', 'rmed1', 'rmed2', 'rmed2fh', 'dts', 'sparring-ucb',
    'sparring-exp3', 'vdb', 'doubler', 'multisbm', 'ccb', 'cw-rmed', 'borda-se', 'dbgd']) assert.ok(algoIds.includes(want), want);
  for (const want of ['bt', 'cyclic', 'copeland', 'borda', 'continuous']) assert.ok(envIds.includes(want), want);
  for (const a of S.algorithms) {
    assert.ok(fam.has(a.family), `${a.id}: family ${a.family}`);
    for (const k of ['name', 'formula', 'regret', 'idea']) assert.ok(typeof a[k] === 'string' && a[k].length, `${a.id}.${k}`);
    assert.ok(!a.formula.includes('\n'), `${a.id}: formula is one line`);
    for (const p of a.params) {
      assert.ok(p.min <= p.default && p.default <= p.max, `${a.id}.${p.key} default within range`);
      assert.ok(p.label && p.step > 0, `${a.id}.${p.key} label/step`);
    }
  }
  for (const e of S.environments) assert.ok(e.name && e.description && Array.isArray(e.tags), e.id);
  assert.ok(envIds.includes(S.defaultEnv));
  assert.ok(S.defaultLineup.length <= 8 && S.defaultLineup.every((id) => algoIds.includes(id)));
  for (const p of S.presets) {
    assert.ok(p.label && p.algos.every((id) => algoIds.includes(id)), p.label);
    if (p.env) assert.ok(envIds.includes(p.env), p.label);
  }
  const c = S.controls;
  assert.ok(c.K.min <= c.K.default && c.K.default <= c.K.max);
  assert.ok(c.T.options.includes(c.T.default) && c.runs.options.includes(c.runs.default));
  for (const x of c.extra) assert.ok(x.min <= x.default && x.default <= x.max, x.key);
});

test('runner API and results shape', () => {
  const T = 300;
  const r = S.createRunner({ env: 'bt', K: 6, T, runs: 3, seed: 1, extra: { strength: 1 }, algorithms: [{ id: 'rucb' }, { id: 'dts', params: { alpha: 0.6 } }] });
  assert.equal(r.done, 0);
  assert.equal(r.finished, false);
  r.step(1);
  assert.equal(r.done, 1);
  r.step(10);
  assert.equal(r.done, 3);
  assert.equal(r.finished, true);
  const res = r.results();
  assert.equal(res.length, 2);
  const keys = [S.final.key, S.final.se, S.final.median, ...S.tiles.map((x) => x.key), ...S.columns.map((x) => x.key)];
  for (const x of res) {
    assert.equal(x.runs, 3);
    assert.ok(x.name && x.family && x.params);
    for (const m of S.metrics) {
      assert.ok(x.curves[m.key] instanceof Float64Array, m.key);
      assert.equal(x.curves[m.key].length, T);
      if (m.band) assert.equal(x.curvesSE[m.key].length, T);
    }
    for (const k of keys) assert.ok(Number.isFinite(x.summary[k]), `summary.${k}`);
    assert.ok(Number.isFinite(x.summary.msPerRun));
    assert.equal(x.finals.length, 3);
    assert.ok(x.finals.every((v, i) => i === 0 || x.finals[i - 1] <= v), 'finals sorted');
    assert.ok(Math.abs(x.summary.finalRegret - x.curves.regret[T - 1]) < 1e-9);
  }
  assert.equal(byId(res).dts.params.alpha, 0.6);
  assert.throws(() => S.createRunner({ env: 'nope', K: 5, T: 10, runs: 1, seed: 1, algorithms: [] }));
  assert.throws(() => S.createRunner({ env: 'bt', K: 5, T: 10, runs: 1, seed: 1, algorithms: [{ id: 'nope' }] }));
});

test('results are deterministic and independent of the lineup', () => {
  const opts = { K: 6, T: 400, runs: 4, seed: 99 };
  const a = run('cyclic', ['rucb', 'dts', 'vdb'], opts);
  const b = run('cyclic', ['rucb', 'dts', 'vdb'], opts);
  const c = run('cyclic', ['dts'], opts);
  const d = run('cyclic', ['dts'], { ...opts, seed: 100 });
  for (const id of ['rucb', 'dts', 'vdb']) assert.deepEqual(Array.from(a[id].curves.regret), Array.from(b[id].curves.regret), id);
  assert.deepEqual(Array.from(a.dts.curves.regret), Array.from(c.dts.curves.regret));
  assert.deepEqual(a.dts.finals, c.dts.finals);
  assert.notDeepEqual(a.dts.finals, d.dts.finals);
});

test('common random numbers: equal choices see equal outcomes', () => {
  // Two copies of the same policy (same id, hence same seed) in one run read the same outcome
  // table, so they make identical choices and score identically.
  const r = S.createRunner({ env: 'cyclic', K: 6, T: 300, runs: 3, seed: 5, algorithms: [{ id: 'dts' }, { id: 'uniform' }, { id: 'dts' }] });
  r.step(3);
  const [x, , y] = r.results();
  assert.deepEqual(x.finals, y.finals);
  assert.deepEqual(Array.from(x.curves.winner), Array.from(y.curves.winner));
});

test('every algorithm returns valid duels on every environment', () => {
  for (const env of S.environments) {
    for (const algo of S.algorithms) {
      for (const K of [4, 9]) {
        const T = 250;
        const info = env.continuous ? { space: { d: env.d } } : {};
        const agent = algo.create(K, { T, params: defaults(algo), rng: M.Rng(3), env: { id: env.id }, info });
        const rng = M.Rng(4);
        for (let t = 1; t <= T; t++) {
          const [a, b] = agent.select(t);
          for (const x of [a, b]) {
            if (typeof x === 'number') assert.ok(Number.isInteger(x) && x >= 0 && x < K, `${algo.id}/${env.id}: bad arm ${x} at t=${t}`);
            else {
              assert.ok(env.continuous && x.length === env.d, `${algo.id}/${env.id}: points only on the continuous env`);
              assert.ok(Array.from(x).every(Number.isFinite) && Math.hypot(...x) <= 1 + 1e-9, `${algo.id}: point outside the unit ball`);
            }
          }
          agent.update(a, b, rng.uniform() < 0.5 ? 1 : 0, t);
        }
      }
    }
  }
});

test('every algorithm runs through the runner on every environment', () => {
  const all = ids(S.algorithms);
  for (const env of S.environments) {
    for (const K of [4, 12]) {
      const res = run(env.id, all, { K, T: 300, runs: 2, seed: 3, extra: { strength: K === 4 ? 0.3 : 3 } });
      for (const id of all) {
        const x = res[id];
        const reg = x.curves.regret, weak = x.curves.weak, win = x.curves.winner;
        for (let t = 0; t < reg.length; t++) {
          assert.ok(Number.isFinite(reg[t]) && Number.isFinite(weak[t]), `${id}/${env.id}: finite`);
          assert.ok(t === 0 || reg[t] >= reg[t - 1] - 1e-12, `${id}/${env.id}: regret nondecreasing`);
          assert.ok(weak[t] <= reg[t] + 1e-9, `${id}/${env.id}: weak ≤ strong`);
          assert.ok(win[t] >= 0 && win[t] <= 1, `${id}/${env.id}: winner rate`);
        }
        assert.ok(reg[reg.length - 1] <= 300, `${id}/${env.id}: regret per duel ≤ 1`);
      }
    }
  }
});

test('regret is zero for duelling the winner against itself', () => {
  // DTS, RMED and CCB all end up playing (w, w); a perfect oracle would have zero regret.
  const res = run('bt', ['rmed2'], { K: 5, T: 3000, runs: 5, seed: 2, extra: { strength: 3 } });
  const reg = res.rmed2.curves.regret, T = reg.length;
  assert.ok(reg[T - 1] - reg[T - 501] < 0.05 * reg[T - 1], 'regret flattens once the winner is found');
  assert.ok(res.rmed2.summary.winnerTail > 0.8);
});

// ------------------------------------------------------------- plausibility ----
const table = {};
function show(label, res) {
  table[label] = res;
  const rows = Object.values(res).sort((x, y) => x.summary.finalRegret - y.summary.finalRegret);
  console.log(`\n${label}`);
  console.log('  algorithm        strong (±95%)     weak   (w,w) last 10%   ms/run');
  for (const x of rows) {
    const s = x.summary;
    console.log(`  ${x.id.padEnd(14)} ${s.finalRegret.toFixed(1).padStart(7)} ±${(1.96 * s.finalSE).toFixed(1).padEnd(6)} ${s.finalWeak.toFixed(1).padStart(7)} ${(100 * s.winnerTail).toFixed(0).padStart(8)}%  ${s.msPerRun.toFixed(2).padStart(9)}`);
  }
}
const reg = (res, id) => res[id].summary.finalRegret;

test('Condorcet (Bradley–Terry): RUCB, RMED and DTS beat uniform, IF and BTM', () => {
  const res = run('bt', ['uniform', 'if', 'btm', 'savage', 'rucb', 'rmed1', 'rmed2', 'dts', 'sparring-ucb']);
  show('bt  K=8 T=2000 runs=20', res);
  for (const good of ['rucb', 'rmed1', 'rmed2', 'dts']) {
    for (const bad of ['uniform', 'if', 'btm']) assert.ok(reg(res, good) < reg(res, bad), `${good} (${reg(res, good)}) < ${bad} (${reg(res, bad)})`);
  }
  assert.ok(reg(res, 'dts') < 0.3 * reg(res, 'uniform'));
  assert.ok(res.dts.summary.winnerTail > 0.8);
});

test('Condorcet with cycles: confidence methods still find the winner', () => {
  const res = run('cyclic', ['uniform', 'if', 'btm', 'rucb', 'rmed1', 'dts']);
  show('cyclic  K=8 T=2000 runs=20', res);
  for (const good of ['rucb', 'rmed1', 'dts']) assert.ok(reg(res, good) < 0.7 * reg(res, 'uniform'), good);
  assert.ok(reg(res, 'rmed1') < reg(res, 'if') && reg(res, 'dts') < reg(res, 'if'));
});

test('Copeland env: CCB and CW-RMED find the Copeland winner', () => {
  const res = run('copeland', ['uniform', 'rucb', 'savage', 'ccb', 'cw-rmed', 'dts']);
  show('copeland  K=8 T=2000 runs=20', res);
  for (const good of ['ccb', 'cw-rmed']) {
    assert.ok(reg(res, good) < 0.65 * reg(res, 'uniform'), `${good} vs uniform`);
    assert.ok(reg(res, good) < reg(res, 'rucb'), `${good} vs RUCB`);
    assert.ok(res[good].summary.winnerTail > 0.6, `${good} duels the Copeland winner`);
  }
  // RUCB assumes a Condorcet winner and never settles on (w, w) here.
  assert.ok(res.rucb.summary.winnerTail < 0.2);
});

test('Borda env: the Borda method beats Condorcet methods', () => {
  const res = run('borda', ['uniform', 'rucb', 'dts', 'ccb', 'vdb', 'borda-se']);
  show('borda  K=8 T=2000 runs=20', res);
  for (const other of ['uniform', 'rucb', 'dts', 'ccb']) assert.ok(reg(res, 'borda-se') < reg(res, other), other);
  assert.ok(res['borda-se'].summary.winnerTail > 0.8);
});

test('Continuous env: DBGD climbs the utility', () => {
  const res = run('continuous', ['uniform', 'dbgd', 'rucb', 'dts']);
  show('continuous  K=8 T=2000 runs=20', res);
  assert.ok(reg(res, 'dbgd') < 0.5 * reg(res, 'uniform'));
  const c = res.dbgd.curves.regret, T = c.length;
  assert.ok(c[T - 1] - c[T / 2 - 1] < c[T / 2 - 1], 'DBGD regret grows sublinearly');
});

test('performance: one step with the default lineup at default controls', () => {
  const r = S.createRunner({
    env: S.defaultEnv, K: S.controls.K.default, T: S.controls.T.default, runs: 5, seed: 1,
    extra: { strength: 1 }, algorithms: S.defaultLineup.map((id) => ({ id })),
  });
  r.step(1); // warm-up (JIT)
  const t0 = M.util.now();
  r.step(3);
  const ms = (M.util.now() - t0) / 3;
  console.log(`\nstep(1), ${S.defaultLineup.length} algorithms, K=${S.controls.K.default}, T=${S.controls.T.default}: ${ms.toFixed(1)} ms`);
  assert.ok(ms < 250, `${ms.toFixed(1)} ms per step`); // budget is ~100 ms; slack for loaded CI machines
});
