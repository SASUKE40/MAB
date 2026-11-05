'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const S = M.SETTINGS.find((s) => s.id === 'continuum');
const ALGO_IDS = S ? S.algorithms.map((a) => a.id) : [];
const ENV_IDS = S ? S.environments.map((e) => e.id) : [];

function run(config) {
  const r = S.createRunner(Object.assign({ seed: 11, extra: { sigma: 0.1 } }, config));
  r.step(config.runs);
  const out = {};
  for (const x of r.results()) out[x.id] = x;
  return out;
}
const ids = (...list) => list.map((id) => ({ id }));

test('continuum setting is registered with a complete description', () => {
  assert.ok(S, 'setting continuum not found');
  assert.equal(S.name, 'Continuum-armed & GP');
  for (const k of ['short', 'blurb', 'xLabel']) assert.equal(typeof S[k], 'string', k);
  assert.equal(typeof S.createRunner, 'function');
  const fam = new Set(S.families.map((f) => f.id));
  for (const f of S.families) for (const k of ['id', 'name', 'blurb']) assert.equal(typeof f[k], 'string');
  for (const e of S.environments) {
    for (const k of ['id', 'name', 'description']) assert.equal(typeof e[k], 'string', `${e.id}.${k}`);
    assert.ok(Array.isArray(e.tags));
  }
  assert.equal(new Set(ALGO_IDS).size, ALGO_IDS.length, 'algorithm ids are unique');
  for (const a of S.algorithms) {
    for (const k of ['id', 'name', 'formula', 'regret', 'idea']) assert.ok(typeof a[k] === 'string' && a[k].length, `${a.id}.${k}`);
    assert.ok(fam.has(a.family), `${a.id} family ${a.family}`);
    assert.equal(typeof a.create, 'function');
    for (const p of a.params) {
      assert.ok(p.min <= p.default && p.default <= p.max, `${a.id}.${p.key} default in range`);
      assert.equal(typeof p.label, 'string');
    }
  }
  assert.ok(ENV_IDS.includes(S.defaultEnv));
  assert.ok(S.defaultLineup.length <= 8 && S.defaultLineup.every((id) => ALGO_IDS.includes(id)));
  for (const p of S.presets) {
    assert.ok(p.algos.every((id) => ALGO_IDS.includes(id)), p.label);
    if (p.env) assert.ok(ENV_IDS.includes(p.env), p.label);
  }
  assert.ok(S.controls.T.options.includes(S.controls.T.default));
  assert.ok(S.controls.runs.options.includes(S.controls.runs.default));
  for (const x of S.controls.extra) assert.ok(x.min <= x.default && x.default <= x.max);
  assert.ok(!S.controls.K, 'no K control: the action is continuous');
  // the requested algorithm and environment coverage
  for (const id of ['uniform', 'ucb-grid', 'zooming', 'contextual-zooming', 'hoo', 'hct', 'vhct', 'poo', 'uct', 'doo', 'soo', 'stosoo',
    'fkm', 'bgd', 'scr', 'kernel-bco', 'gp-ucb', 'igp-ucb', 'gp-ts', 'bkb', 'safeopt', 'stageopt']) assert.ok(ALGO_IDS.includes(id), id);
  assert.ok(S.environments.some((e) => e.d === 2), 'has a 2-D environment');
  for (const id of ['smooth', 'multimodal', 'needle', 'branin', 'convex', 'safe', 'contextual']) assert.ok(ENV_IDS.includes(id), id);
});

test('runner results have the documented shape', () => {
  const T = 120, runs = 3;
  const r = S.createRunner({ env: S.defaultEnv, T, runs, seed: 1, extra: { sigma: 0.1 }, algorithms: ids('uniform', 'hoo') });
  assert.equal(r.done, 0);
  assert.equal(r.finished, false);
  r.step(1);
  assert.equal(r.done, 1);
  r.step(10);
  assert.equal(r.done, runs);
  assert.equal(r.finished, true);
  const res = r.results();
  assert.equal(res.length, 2);
  const keys = new Set([S.final.key, S.final.se, S.final.median, ...S.tiles.map((t) => t.key), ...S.columns.map((c) => c.key), 'msPerRun']);
  for (const x of res) {
    assert.equal(x.runs, runs);
    assert.equal(typeof x.name, 'string');
    assert.equal(typeof x.family, 'string');
    const L = x.x.length;
    assert.ok(L > 10 && L <= T);
    assert.equal(x.x[L - 1], T);
    for (const m of S.metrics) {
      assert.equal(x.curves[m.key].length, L, m.key);
      if (m.band) assert.equal(x.curvesSE[m.key].length, L);
      assert.ok(x.curves[m.key].every(Number.isFinite));
    }
    for (const k of keys) assert.ok(Number.isFinite(x.summary[k]), `summary.${k}`);
    assert.equal(x.finals.length, runs);
    for (let i = 1; i < runs; i++) assert.ok(x.finals[i] >= x.finals[i - 1]);
    assert.ok(Math.abs(x.summary.finalRegret - x.curves.regret[L - 1]) < 1e-12);
    for (let k = 1; k < L; k++) assert.ok(x.curves.regret[k] >= x.curves.regret[k - 1] - 1e-9, 'cumulative regret is non-decreasing');
  }
});

test('results are deterministic and independent of the lineup', () => {
  const base = { env: 'multimodal', T: 150, runs: 3, seed: 5, extra: { sigma: 0.1 } };
  const a = S.createRunner({ ...base, algorithms: ids('gp-ucb', 'stosoo') });
  const b = S.createRunner({ ...base, algorithms: ids('uniform', 'stosoo', 'zooming', 'gp-ucb') });
  a.step(3); b.step(3);
  const ra = a.results(), rb = b.results();
  assert.deepEqual(Array.from(ra[0].curves.regret), Array.from(rb[3].curves.regret));
  assert.deepEqual(Array.from(ra[1].curves.simple), Array.from(rb[1].curves.simple));
  const c = S.createRunner({ ...base, extra: { sigma: 0.3 }, algorithms: ids('gp-ucb') });
  c.step(3);
  assert.notDeepEqual(Array.from(c.results()[0].curves.regret), Array.from(ra[0].curves.regret), 'noise level is used');
});

test('every algorithm runs on every environment and plays points in the domain', () => {
  for (const envId of ENV_IDS) {
    const env = S.ENV_BY_ID[envId];
    const d = env.d, T = 60;
    for (const algo of S.algorithms) {
      const rng = M.Rng(3), inst = env.make(M.Rng(4));
      const agent = algo.create({
        T, d, sigma: 0.1, threshold: inst.h, safeSeed: inst.seed, contextual: !!env.contextual,
        params: S.defaultParams(algo.id), rng,
      });
      for (let t = 1; t <= T; t++) {
        const c = env.contextual ? rng.uniform() : 0.5;
        const x = agent.select(t, c);
        assert.equal(x.length, d, `${algo.id} on ${envId}`);
        for (let j = 0; j < d; j++) assert.ok(x[j] >= -1e-9 && x[j] <= 1 + 1e-9, `${algo.id} on ${envId} played ${x[j]}`);
        const y = inst.f(x, c) + 0.1 * rng.normal();
        agent.update(Float64Array.from(x), y, t, c);
        if (agent.recommend && t % 20 === 0) {
          const r = agent.recommend(c);
          assert.equal(r.length, d);
          for (let j = 0; j < d; j++) assert.ok(Number.isFinite(r[j]) && r[j] >= -1e-9 && r[j] <= 1 + 1e-9, `${algo.id} recommend`);
        }
      }
    }
    // and through the runner
    const res = run({ env: envId, T: 40, runs: 1, algorithms: ALGO_IDS.map((id) => ({ id })) });
    for (const id of ALGO_IDS) {
      const s = res[id].summary;
      assert.ok(Number.isFinite(s.finalRegret) && s.finalRegret >= -1e-9, `${id} on ${envId}`);
      assert.ok(s.simpleRegret >= -1e-9);
    }
  }
});

test('GP-UCB, HOO and Zooming beat uniform random on smooth environments', () => {
  for (const env of ['smooth', 'multimodal']) {
    const res = run({ env, T: 300, runs: 6, algorithms: ids('uniform', 'gp-ucb', 'hoo', 'zooming') });
    const u = res.uniform.summary.finalRegret;
    for (const id of ['gp-ucb', 'hoo', 'zooming']) {
      assert.ok(res[id].summary.finalRegret < 0.5 * u, `${id} on ${env}: ${res[id].summary.finalRegret.toFixed(1)} vs uniform ${u.toFixed(1)}`);
    }
  }
});

test('FKM and BGD make progress on the convex environment', () => {
  const res = run({ env: 'convex', T: 500, runs: 10, algorithms: ids('uniform', 'fkm', 'bgd') });
  for (const id of ['fkm', 'bgd']) {
    const sr = res[id].curves.simple, start = sr[0], end = sr[sr.length - 1];
    assert.ok(end < 0.6 * start, `${id}: simple regret ${start.toFixed(3)} -> ${end.toFixed(3)}`);
    assert.ok(res[id].summary.finalRegret < res.uniform.summary.finalRegret, `${id} beats uniform`);
  }
  assert.ok(res.bgd.summary.simpleRegret < 0.02);
});

test('SafeOpt and StageOpt keep unsafe evaluations near zero', () => {
  const res = run({ env: 'safe', T: 300, runs: 6, algorithms: ids('uniform', 'safeopt', 'stageopt') });
  const u = res.uniform.summary.unsafe;
  assert.ok(u > 50, `uniform is often unsafe (${u})`);
  for (const id of ['safeopt', 'stageopt']) {
    const k = res[id].summary.unsafe;
    assert.ok(k <= 5 && k < 0.05 * u, `${id}: ${k} unsafe evaluations per run`);
    assert.ok(res[id].summary.simpleRegret < 0.05, `${id} still finds a good safe point`);
  }
  const plain = run({ env: 'smooth', T: 60, runs: 1, algorithms: ids('uniform') });
  assert.equal(plain.uniform.summary.unsafe, 0, 'no unsafe evaluations without a threshold');
});

test('Contextual Zooming exploits the context', () => {
  const res = run({ env: 'contextual', T: 500, runs: 5, algorithms: ids('zooming', 'contextual-zooming', 'gp-ucb') });
  const cz = res['contextual-zooming'].summary;
  assert.ok(cz.finalRegret < 0.8 * res.zooming.summary.finalRegret);
  assert.ok(cz.finalRegret < 0.8 * res['gp-ucb'].summary.finalRegret);
  assert.ok(cz.simpleRegret < 0.5 * res.zooming.summary.simpleRegret);
});

test('default lineup is fast enough for the dashboard, and a sanity table', () => {
  const T = S.controls.T.default;
  const algos = S.defaultLineup.map((id) => ({ id }));
  const r = S.createRunner({ env: S.defaultEnv, T, runs: 10, seed: 2, extra: { sigma: 0.1 }, algorithms: algos });
  r.step(1); // warm up (JIT, cached kernel matrices)
  const t0 = M.util.now();
  r.step(1);
  const ms = M.util.now() - t0;
  r.step(8);
  const rows = r.results().map((x) => ({
    id: x.id, regret: +x.summary.finalRegret.toFixed(1), ci: +(1.96 * x.summary.finalSE).toFixed(1),
    simple: +x.summary.simpleRegret.toFixed(3), msPerRun: +x.summary.msPerRun.toFixed(1),
  }));
  console.log(`\n${S.defaultEnv}, T = ${T}, 10 runs; one step(1) of the default lineup took ${ms.toFixed(0)} ms`);
  console.table(rows);
  assert.ok(ms < 600, `step(1) took ${ms.toFixed(0)} ms`); // generous: target is ~150 ms on a typical machine
  const u = rows.find((x) => x.id === 'uniform').regret;
  // local (zeroth-order) methods may lose to uniform on the multimodal default; the rest must not
  const local = new Set(S.algorithms.filter((a) => a.family === 'zeroth-order').map((a) => a.id));
  for (const x of rows) if (x.id !== 'uniform' && !local.has(x.id)) assert.ok(x.regret < u, `${x.id} beats uniform`);
});
