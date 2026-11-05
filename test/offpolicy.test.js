'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const S = M.SETTINGS.find((s) => s.id === 'offpolicy');
const ids = (kind) => S.algorithms.filter((a) => a.kind === kind).map((a) => a.id);
const byId = (res) => Object.fromEntries(res.map((r) => [r.id, r]));
function run(env, algorithms, runs, extra = {}, T = 1000, seed = 7) {
  const r = S.createRunner({ env, K: 5, T, runs, seed, extra, algorithms: algorithms.map((a) => (typeof a === 'string' ? { id: a } : a)) });
  r.step(runs);
  return r.results();
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

test('offpolicy setting is registered with a complete description', () => {
  assert.ok(S, 'setting registered');
  assert.equal(S.name, 'Off-policy & offline');
  const fam = new Set(S.families.map((f) => f.id));
  const algoIds = new Set(S.algorithms.map((a) => a.id));
  assert.equal(algoIds.size, S.algorithms.length, 'unique algorithm ids');
  for (const a of S.algorithms) {
    for (const k of ['id', 'name', 'family', 'formula', 'regret', 'idea']) assert.equal(typeof a[k], 'string', `${a.id}.${k}`);
    assert.ok(fam.has(a.family), `${a.id} family`);
    assert.ok(a.kind === 'eval' || a.kind === 'learn', `${a.id} kind`);
    assert.ok(Array.isArray(a.params));
    for (const p of a.params) assert.ok(p.default >= p.min && p.default <= p.max, `${a.id}.${p.key} default in range`);
  }
  for (const id of ['ips', 'snips', 'dm', 'dr', 'mrdr', 'switch-dr', 'cab']) assert.ok(algoIds.has(id), id);
  for (const id of ['poem', 'svp', 'ips-greedy', 'dm-greedy', 'pevi', 'linlcb', 'neurallcb']) assert.ok(algoIds.has(id), id);
  assert.ok(S.environments.some((e) => e.id === S.defaultEnv));
  assert.ok(S.defaultLineup.length <= 8 && S.defaultLineup.every((id) => algoIds.has(id)));
  for (const p of S.presets) {
    assert.ok(p.algos.length <= 8 && p.algos.every((id) => algoIds.has(id)), p.label);
    if (p.env) assert.ok(S.environments.some((e) => e.id === p.env), p.label);
  }
  // presets never mix estimators and learners
  for (const p of S.presets) assert.equal(new Set(p.algos.map((id) => S.algorithms.find((a) => a.id === id).kind)).size, 1, p.label);
});

test('runner API shape: incremental, curves on the n grid, summary keys', () => {
  const r = S.createRunner({ env: 'linear', K: 4, T: 500, runs: 3, seed: 1, extra: {}, algorithms: [{ id: 'dr' }, { id: 'linlcb' }] });
  assert.equal(r.done, 0);
  assert.equal(r.finished, false);
  r.step(1);
  assert.equal(r.done, 1);
  r.step(5);
  assert.equal(r.done, 3);
  assert.equal(r.finished, true);
  const res = r.results();
  assert.equal(res.length, 2);
  const keys = new Set([S.final.key, S.final.se, S.final.median, ...S.tiles.map((t) => t.key), ...S.columns.map((c) => c.key)]);
  for (const x of res) {
    assert.equal(x.runs, 3);
    assert.ok(x.x instanceof Float64Array);
    const L = x.x.length;
    assert.equal(x.x[L - 1], 500);
    for (let i = 1; i < L; i++) assert.ok(x.x[i] > x.x[i - 1]);
    for (const m of S.metrics) {
      assert.equal(x.curves[m.key].length, L, m.key);
      for (const v of x.curves[m.key]) assert.ok(Number.isFinite(v), `${x.id} ${m.key}`);
    }
    assert.equal(x.curvesSE.loss.length, L);
    for (const k of keys) assert.ok(Number.isFinite(x.summary[k]), `${x.id} summary.${k}`);
    assert.ok(Number.isFinite(x.summary.msPerRun));
    assert.equal(x.finals.length, 3);
  }
  // small-sample environment caps n
  const [s] = run('small', ['ips'], 1, {}, 1000);
  assert.equal(s.x[s.x.length - 1], 150);
});

test('deterministic and independent of the lineup', () => {
  const a = run('lowoverlap', ['dr', 'neurallcb'], 4, {}, 500, 3);
  const b = run('lowoverlap', ['ips', 'neurallcb', 'poem', 'dr'], 4, {}, 500, 3);
  const A = byId(a), B = byId(b);
  for (const id of ['dr', 'neurallcb']) {
    assert.deepEqual(Array.from(A[id].curves.loss), Array.from(B[id].curves.loss), id);
    assert.deepEqual(Array.from(A[id].curves.bias), Array.from(B[id].curves.bias), id);
  }
});

test('every method runs on every environment with finite results', () => {
  const all = S.algorithms.map((a) => a.id);
  for (const env of S.environments) {
    for (const extra of [{}, { d: 2, P: 10 }]) {
      const res = run(env.id, all, 2, extra, 200, 5);
      for (const x of res) {
        for (const m of S.metrics) for (const v of x.curves[m.key]) assert.ok(Number.isFinite(v), `${x.id} on ${env.id} ${m.key}`);
        if (x.kind === 'learn') for (const v of x.curves.loss) assert.ok(v >= -1e-9, `${x.id} on ${env.id}: suboptimality ≥ 0`);
      }
    }
  }
  // K = 2 and larger K also work
  for (const K of [2, 12]) {
    const r = S.createRunner({ env: 'misspecified', K, T: 200, runs: 1, seed: 2, extra: {}, algorithms: all.map((id) => ({ id })) });
    r.step(1);
    for (const x of r.results()) assert.ok(Number.isFinite(x.summary.loss), `${x.id} K=${K}`);
  }
});

test('instances: propensities and policies are distributions, truth is consistent', () => {
  const inst = S.makeInstance('lowoverlap', 5, 5, 50, 300, 9);
  const { data, mu } = inst;
  for (let p = 0; p < data.P; p++) {
    let sb = 0, se = 0;
    for (let a = 0; a < data.K; a++) { sb += data.pb[p * 5 + a]; se += data.pe[p * 5 + a]; assert.ok(data.pb[p * 5 + a] >= 0.01 - 1e-12); }
    assert.ok(Math.abs(sb - 1) < 1e-9 && Math.abs(se - 1) < 1e-9);
  }
  assert.ok(inst.Vstar >= inst.Ve && inst.Vstar >= inst.Vb);
  for (let i = 0; i < 300; i++) assert.equal(data.prop[i], data.pb[data.cx[i] * 5 + data.act[i]]);
  assert.ok(!('mu' in data) && mu.length === 250, 'truth is not handed to the methods');
});

test('IPS is unbiased but has high variance under low overlap', () => {
  const R = 300;
  const res = byId(run('lowoverlap', ['ips', 'snips', 'dr', 'switch-dr'], R, {}, 500, 21));
  const ips = res.ips;
  const L = ips.x.length;
  // bias within 3 standard errors at every n
  for (let l = 0; l < L; l++) {
    const se = ips.curves.sd[l] / Math.sqrt(R);
    assert.ok(Math.abs(ips.curves.bias[l]) < 3 * se + 1e-3, `IPS bias ${ips.curves.bias[l]} vs SE ${se} at n=${ips.x[l]}`);
  }
  // and the largest spread
  for (const id of ['snips', 'dr', 'switch-dr']) assert.ok(ips.summary.sd > 1.5 * res[id].summary.sd, `IPS sd ${ips.summary.sd} vs ${id} ${res[id].summary.sd}`);
  // low overlap inflates IPS variance compared with good overlap
  const good = byId(run('linear', ['ips'], R, {}, 500, 21)).ips;
  assert.ok(ips.summary.sd > 1.5 * good.summary.sd, `low overlap ${ips.summary.sd} vs good ${good.summary.sd}`);
});

test('DR beats IPS with a good (well-specified) reward model; DM is biased when misspecified', () => {
  const lin = byId(run('linear', ['ips', 'dm', 'dr'], 150, {}, 1000, 4));
  for (let l = 0; l < lin.dr.x.length; l++) assert.ok(lin.dr.curves.loss[l] < lin.ips.curves.loss[l], `n=${lin.dr.x[l]}`);
  assert.ok(lin.dr.summary.loss < 0.7 * lin.ips.summary.loss);
  // RMSE shrinks with n
  assert.ok(lin.dr.curves.loss[lin.dr.x.length - 1] < 0.5 * lin.dr.curves.loss[0]);
  const mis = byId(run('misspecified', ['dm', 'dr'], 150, {}, 1000, 4));
  assert.ok(Math.abs(mis.dm.summary.bias) > 3 * mis.dm.summary.sd / Math.sqrt(150), `DM bias ${mis.dm.summary.bias}`);
  assert.ok(mis.dr.summary.loss < mis.dm.summary.loss);
});

test('pessimistic learners beat their greedy counterparts under low coverage', () => {
  const R = 60;
  const specs = [
    { id: 'dm-greedy' }, { id: 'linlcb' },
    { id: 'pevi', params: { beta: 0 } }, { id: 'pevi' },
    { id: 'neurallcb', params: { beta: 0 } }, { id: 'neurallcb' },
  ];
  // the runner seeds by algorithm id, so run greedy (β = 0) variants separately to keep ids apart
  const res = run('lowoverlap', specs.filter((s) => !s.params), R, {}, 1000, 8);
  const greedy = run('lowoverlap', specs.filter((s) => s.params), R, {}, 1000, 8);
  const P = byId(res), G = byId(greedy);
  const area = (x) => mean(Array.from(x.curves.loss));
  assert.ok(area(P.linlcb) < area(P['dm-greedy']), `LinLCB ${area(P.linlcb)} vs DM-greedy ${area(P['dm-greedy'])}`);
  assert.ok(area(P.pevi) < area(G.pevi), `PEVI ${area(P.pevi)} vs tabular greedy ${area(G.pevi)}`);
  assert.ok(area(P.neurallcb) < area(G.neurallcb), `NeuralLCB ${area(P.neurallcb)} vs neural greedy ${area(G.neurallcb)}`);
  // greedy methods overestimate their own value; pessimistic ones do not
  assert.ok(P['dm-greedy'].summary.bias > 0 && P.linlcb.summary.bias < 0);
  assert.ok(G.pevi.summary.bias > 0 && P.pevi.summary.bias < 0);
  // and every learner improves on the IPS-greedy baseline here
  const ipsg = byId(run('lowoverlap', ['ips-greedy'], R, {}, 1000, 8))['ips-greedy'];
  for (const id of ['linlcb', 'pevi', 'neurallcb']) assert.ok(P[id].summary.loss < ipsg.summary.loss, id);
});

test('learners improve with more data in the well-specified case', () => {
  const res = run('linear', ids('learn'), 20, {}, 1000, 12);
  for (const x of res) {
    const c = x.curves.loss;
    assert.ok(c[c.length - 1] < c[0], `${x.id}: ${c[0]} → ${c[c.length - 1]}`);
  }
});

test('one step at default controls with 8 methods stays fast', () => {
  const c = S.controls;
  for (const algos of [ids('eval'), ids('learn')]) {
    const r = S.createRunner({ env: S.defaultEnv, K: c.K.default, T: c.T.default, runs: c.runs.default, seed: 1, extra: {}, algorithms: algos.slice(0, 8).map((id) => ({ id })) });
    r.step(2);   // warm up the JIT
    const t = [];
    for (let i = 0; i < 5; i++) { const t0 = performance.now(); r.step(1); t.push(performance.now() - t0); }
    t.sort((a, b) => a - b);
    assert.ok(t[2] < 150, `median step(1) ${t[2].toFixed(1)} ms`);
  }
});
