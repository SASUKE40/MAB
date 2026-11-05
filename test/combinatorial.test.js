'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../src/index.js');

const S = M.SETTINGS.find((s) => s.id === 'combinatorial');
const I = S && S.internals;
const ALGO_IDS = S ? S.algorithms.map((a) => a.id) : [];
const ENV_IDS = S ? S.environments.map((e) => e.id) : [];
const STATIONARY = ['topm', 'matroid', 'cascade', 'dcm', 'pbm', 'cascade-lin'];

function run(env, algos, { L = 16, m = 4, T = 2000, runs = 20, seed = 11 } = {}) {
  const r = S.createRunner({ env, K: L, T, runs, seed, extra: { m }, algorithms: algos.map((id) => (typeof id === 'string' ? { id } : id)) });
  r.step(runs);
  const out = {};
  for (const x of r.results()) out[x.id] = x;
  return out;
}

test('setting is registered with the documented shape', () => {
  assert.ok(S, 'combinatorial setting registered');
  assert.equal(S.name, 'Combinatorial & ranking');
  for (const k of ['short', 'blurb', 'xLabel']) assert.equal(typeof S[k], 'string', k);
  assert.ok(Array.isArray(S.families) && S.families.length >= 5);
  assert.equal(S.environments.length, 7);
  for (const e of S.environments) {
    for (const k of ['id', 'name', 'description']) assert.equal(typeof e[k], 'string', `${e.id}.${k}`);
    assert.ok(Array.isArray(e.tags));
    assert.ok(e.description.length > 60, `${e.id} has a real description`);
  }
  const fam = new Set(S.families.map((f) => f.id));
  const ids = new Set();
  for (const a of S.algorithms) {
    assert.ok(!ids.has(a.id), 'unique id ' + a.id); ids.add(a.id);
    for (const k of ['id', 'name', 'family', 'formula', 'regret', 'idea']) assert.equal(typeof a[k], 'string', `${a.id}.${k}`);
    assert.ok(fam.has(a.family), `${a.id} family ${a.family}`);
    assert.ok(!a.formula.includes('\n'), `${a.id} formula is one line`);
    assert.equal(typeof a.create, 'function');
    for (const p of a.params) {
      assert.ok(p.default >= p.min && p.default <= p.max, `${a.id}.${p.key} default in range`);
      assert.equal(typeof p.label, 'string');
    }
  }
  for (const id of ['random', 'greedy', 'cucb', 'combucb1', 'cts', 'escb', 'exp3m', 'comband', 'osmd', 'omm',
    'cascadeucb1', 'cascadeklucb', 'cascadets', 'cascadelinucb', 'cascadelints', 'dcmucb', 'dcmklucb',
    'pbmucb', 'pbmpie', 'rba', 'toprank']) assert.ok(ids.has(id), 'has ' + id);
  assert.ok(ENV_IDS.includes(S.defaultEnv));
  assert.ok(S.defaultLineup.length <= 8 && S.defaultLineup.every((id) => ids.has(id)));
  for (const p of S.presets) {
    assert.ok(p.algos.every((id) => ids.has(id)), p.label);
    if (p.env) assert.ok(ENV_IDS.includes(p.env), p.label);
  }
  const c = S.controls;
  assert.ok(c.K.default >= c.K.min && c.K.default <= c.K.max);
  assert.ok(c.T.options.includes(c.T.default) && c.runs.options.includes(c.runs.default));
  const mCtl = c.extra.find((x) => x.key === 'm');
  assert.ok(mCtl && mCtl.default >= mCtl.min && mCtl.default <= mCtl.max);
  assert.ok(S.metrics.length >= 2 && S.metrics[0].key === 'regret');
  assert.equal(S.final.key, 'finalRegret');
});

test('runner results carry every curve and summary key the setting declares', () => {
  const T = 300;
  const r = S.createRunner({ env: 'topm', K: 10, T, runs: 3, seed: 1, extra: { m: 3 }, algorithms: [{ id: 'cucb' }, { id: 'random', params: {} }] });
  assert.equal(r.done, 0); assert.equal(r.finished, false);
  r.step(1); assert.equal(r.done, 1);
  r.step(10); assert.equal(r.done, 3); assert.equal(r.finished, true);
  const res = r.results();
  assert.equal(res.length, 2);
  const keys = new Set([S.final.key, S.final.se, S.final.median, ...S.tiles.map((t) => t.key), ...S.columns.map((c) => c.key)]);
  for (const x of res) {
    assert.equal(x.runs, 3);
    for (const k of ['id', 'name', 'family']) assert.equal(typeof x[k], 'string');
    assert.equal(typeof x.params, 'object');
    for (const m of S.metrics) {
      assert.ok(x.curves[m.key] instanceof Float64Array && x.curves[m.key].length === T, m.key);
      assert.ok(x.curves[m.key].every(Number.isFinite), m.key + ' finite');
      if (m.band) assert.equal(x.curvesSE[m.key].length, T);
    }
    for (const k of keys) assert.ok(Number.isFinite(x.summary[k]), 'summary.' + k);
    assert.ok(Number.isFinite(x.summary.msPerRun));
    assert.equal(x.finals.length, 3);
    for (let i = 1; i < x.finals.length; i++) assert.ok(x.finals[i] >= x.finals[i - 1]);
    assert.equal(x.summary.finalRegret, x.curves.regret[T - 1]);
  }
  assert.equal(r.config.m, 3);
});

test('runs are deterministic and independent of the lineup', () => {
  for (const env of ['topm', 'cascade', 'adversarial']) {
    const base = { env, K: 12, T: 400, runs: 4, seed: 99, extra: { m: 3 } };
    const a = S.createRunner({ ...base, algorithms: [{ id: 'cts' }, { id: 'exp3m' }] }); a.step(4);
    const b = S.createRunner({ ...base, algorithms: [{ id: 'exp3m' }, { id: 'toprank' }, { id: 'cts' }] }); b.step(2); b.step(2);
    const ra = a.results(), rb = b.results();
    assert.deepEqual(Array.from(ra[0].curves.regret), Array.from(rb[2].curves.regret), env + ' cts');
    assert.deepEqual(Array.from(ra[1].curves.regret), Array.from(rb[0].curves.regret), env + ' exp3m');
  }
});

test('the best fixed list is optimal (brute force over ordered lists)', () => {
  const L = 6, m = 3, T = 5;
  const lists = [];
  const rec = (pre) => {
    if (pre.length === m) { lists.push(Int32Array.from(pre)); return; }
    for (let i = 0; i < L; i++) if (!pre.includes(i)) rec([...pre, i]);
  };
  rec([]);
  for (const env of S.environments) {
    if (env.id === 'adversarial') continue;
    for (let s = 0; s < 5; s++) {
      const inst = env.make(L, m, T, M.Rng(1000 + s));
      let best = -Infinity;
      for (const l of lists) {
        if (inst.space.type === 'partition') {
          const b = new Set(Array.from(l, (i) => inst.space.grp[i]));
          if (b.size !== m) continue;
        }
        best = Math.max(best, inst.expected(l, 0));
      }
      assert.ok(Math.abs(best - inst.best[0]) < 1e-12, `${env.id}: oracle ${inst.best[0]} vs brute force ${best}`);
    }
  }
});

test('every algorithm returns valid, feasible lists on every environment', () => {
  const combos = [[16, 4], [5, 4], [4, 1], [12, 8], [7, 2]];
  for (const env of S.environments) {
    for (const [L, m] of combos) {
      const T = 120;
      const inst = env.make(L, m, T, M.Rng(M.hash32(L, m, env.id)));
      for (const id of ALGO_IDS) {
        const agent = I.createAgent(id, env, inst, L, m, T, M.Rng(M.hash32(5, id)), {});
        const fb = I.makeFeedback(inst, m);
        const cur = new Int32Array(m);
        for (let t = 0; t < T; t++) {
          const a = agent.select(t + 1);
          assert.ok(a && a.length === m, `${id} on ${env.id} (L=${L}, m=${m}) list length`);
          const seen = new Set();
          for (let k = 0; k < m; k++) {
            const i = a[k];
            assert.ok(Number.isInteger(i) && i >= 0 && i < L, `${id} on ${env.id}: item ${i}`);
            assert.ok(!seen.has(i), `${id} on ${env.id} (L=${L}, m=${m}) duplicate ${i} at t=${t}`);
            seen.add(i); cur[k] = i;
          }
          if (inst.space.type === 'partition') {
            const blocks = new Set(Array.from(cur, (i) => inst.space.grp[i]));
            assert.equal(blocks.size, m, `${id} on ${env.id}: one item per block`);
          }
          const e = inst.expected(cur, t);
          assert.ok(Number.isFinite(e));
          inst.feedback(cur, t, fb);
          agent.update(cur, fb, t + 1);
        }
      }
    }
  }
});

test('feedback follows each click model', () => {
  const L = 10, m = 4, T = 400, list = Int32Array.from([0, 1, 2, 3]);
  const E = S.internals.ENV_BY_ID;
  const cas = E.cascade.make(L, m, T, M.Rng(1)), fb = I.makeFeedback(cas, m);
  const dcm = E.dcm.make(L, m, T, M.Rng(2)), fd = I.makeFeedback(dcm, m);
  const full = E.adversarial.make(L, m, T, M.Rng(3)), ff = I.makeFeedback(full, m);
  const semi = E.topm.make(L, m, T, M.Rng(4)), fs = I.makeFeedback(semi, m);
  let multi = 0;
  for (let t = 0; t < T; t++) {
    cas.feedback(list, t, fb);
    const clicks = Array.from(fb.clicks).reduce((x, y) => x + y, 0);
    assert.ok(clicks <= 1);
    assert.equal(fb.reward, clicks);
    const c = fb.clicks.indexOf(1);
    for (let k = 0; k < m; k++) {
      if (c >= 0 && k > c) assert.ok(Number.isNaN(fb.obs[k])); else assert.equal(fb.obs[k], k === c ? 1 : 0);
    }
    dcm.feedback(list, t, fd);
    const last = fd.clicks.lastIndexOf(1);
    if (Array.from(fd.clicks).filter((x) => x).length > 1) multi++;
    for (let k = 0; k < m; k++) if (last >= 0 && k > last) assert.ok(Number.isNaN(fd.obs[k]));
    full.feedback(list, t, ff);
    assert.ok(ff.full && Array.from(ff.obs).every(Number.isNaN));
    assert.ok(Math.abs(ff.proxy - ff.reward / m) < 1e-12);
    semi.feedback(list, t, fs);
    assert.equal(fs.clicks, null);
    assert.equal(fs.reward, Array.from(fs.obs).reduce((x, y) => x + y, 0));
  }
  assert.ok(multi > 0, 'DCM produces multiple clicks sometimes');
});

test('dependent rounding has the requested marginals', () => {
  const p = Float64Array.from([0.9, 0.5, 0.3, 0.8, 0.2, 0.3]);  // sums to 3
  const idx = Int32Array.from([0, 1, 2, 3, 4, 5]), work = new Float64Array(6), out = new Int32Array(3);
  const cnt = new Float64Array(6), rng = M.Rng(4), N = 20000;
  for (let s = 0; s < N; s++) {
    const n = I.depRound(p, idx, 3, rng, work, out, 0);
    assert.equal(n, 3);
    assert.equal(new Set(out).size, 3);
    for (const i of out) cnt[i]++;
  }
  for (let i = 0; i < 6; i++) assert.ok(Math.abs(cnt[i] / N - p[i]) < 0.015, `item ${i}: ${cnt[i] / N} vs ${p[i]}`);
});

test('regret is non-decreasing on stationary environments', () => {
  for (const env of STATIONARY) {
    const res = run(env, ALGO_IDS, { L: 8, m: 3, T: 300, runs: 3 });
    for (const id of ALGO_IDS) {
      const r = res[id].curves.regret;
      for (let t = 1; t < r.length; t++) assert.ok(r[t] >= r[t - 1] - 1e-9, `${id} on ${env} at ${t}`);
    }
  }
});

test('plausibility: learners beat baselines where they should (table printed)', () => {
  const table = {};
  const get = (env, ids, opt) => { const r = run(env, ids, opt); for (const id of ids) (table[id] = table[id] || {})[env] = r[id].summary.finalRegret; return r; };
  const topm = get('topm', ['random', 'greedy', 'cucb', 'combucb1', 'cts', 'escb', 'omm', 'exp3m', 'osmd']);
  for (const id of ['cucb', 'combucb1', 'cts', 'escb', 'omm']) {
    assert.ok(topm[id].summary.finalRegret < topm.random.summary.finalRegret / 3, `${id} beats random on top-m`);
  }
  assert.ok(topm.exp3m.summary.finalRegret < topm.random.summary.finalRegret, 'EXP3.M beats random on top-m');
  assert.ok(topm.cts.summary.finalRegret < topm.cucb.summary.finalRegret, 'CTS beats CUCB on top-m');

  const mat = get('matroid', ['random', 'omm', 'cts', 'osmd']);
  assert.ok(mat.omm.summary.finalRegret < mat.random.summary.finalRegret / 3, 'OMM learns the matroid');
  assert.ok(mat.osmd.summary.finalRegret < mat.random.summary.finalRegret / 2, 'OSMD learns the matroid');

  const cas = get('cascade', ['random', 'cascadeucb1', 'cascadeklucb', 'cascadets', 'rba', 'toprank']);
  for (const id of ['cascadeucb1', 'cascadeklucb', 'cascadets']) {
    assert.ok(cas[id].summary.finalRegret < cas.rba.summary.finalRegret, `${id} beats RBA on cascade`);
  }
  assert.ok(cas.toprank.summary.finalRegret < cas.random.summary.finalRegret, 'TopRank beats random on cascade');

  const dcm = get('dcm', ['random', 'dcmucb', 'dcmklucb', 'rba']);
  assert.ok(dcm.dcmklucb.summary.finalRegret < dcm.rba.summary.finalRegret / 2, 'dcmKL-UCB beats RBA on DCM');

  const pbm = get('pbm', ['random', 'pbmucb', 'pbmpie', 'cascadeucb1', 'rba', 'toprank']);
  for (const id of ['pbmucb', 'pbmpie']) {
    assert.ok(pbm[id].summary.finalRegret < pbm.random.summary.finalRegret / 4, `${id} beats random on PBM`);
    assert.ok(pbm[id].summary.finalRegret < pbm.cascadeucb1.summary.finalRegret, `${id} beats CascadeUCB1 on PBM`);
    assert.ok(pbm[id].summary.finalRegret < pbm.rba.summary.finalRegret / 2, `${id} beats RBA on PBM`);
  }

  const lin = get('cascade-lin', ['random', 'cascadelinucb', 'cascadelints', 'cascadeucb1']);
  for (const id of ['cascadelinucb', 'cascadelints']) {
    assert.ok(lin[id].summary.finalRegret < lin.cascadeucb1.summary.finalRegret, `${id} uses the features`);
  }

  const adv = get('adversarial', ['random', 'comband', 'exp3m', 'osmd']);
  assert.ok(adv.comband.summary.finalRegret < adv.random.summary.finalRegret * 0.7, 'ComBand learns from total reward');
  for (const id of ['exp3m', 'osmd']) assert.ok(adv[id].summary.finalRegret < adv.random.summary.finalRegret, `${id} beats random`);

  const envs = ['topm', 'matroid', 'cascade', 'dcm', 'pbm', 'cascade-lin', 'adversarial'];
  const pad = (s, n) => String(s).padStart(n);
  const lines = [pad('final regret (L=16, m=4, T=2000, 20 runs)', 44), pad('algo', 14) + envs.map((e) => pad(e, 12)).join('')];
  for (const id of Object.keys(table)) lines.push(pad(id, 14) + envs.map((e) => pad(table[id][e] != null ? table[id][e].toFixed(1) : '·', 12)).join(''));
  console.log(lines.join('\n'));
});

test('performance: one step(1) of the default lineup at default controls', () => {
  const c = S.controls;
  const r = S.createRunner({
    env: S.defaultEnv, K: c.K.default, T: c.T.default, runs: 4, seed: 1,
    extra: { m: c.extra[0].default }, algorithms: S.defaultLineup.map((id) => ({ id })),
  });
  r.step(1); // warm up the JIT
  const t0 = performance.now();
  r.step(3);
  const ms = (performance.now() - t0) / 3;
  console.log(`default lineup, ${S.defaultEnv}: ${ms.toFixed(1)} ms per step(1)`);
  assert.ok(ms < 400, `step(1) took ${ms} ms`);
});
