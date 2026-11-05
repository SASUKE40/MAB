/*
 * Setting: restless bandits.
 *
 * N arms, each a two-state Markov chain (0 = bad, 1 = good) with its own transition
 * probabilities under the active (pulled) and passive action: P[a][s] = Pr(next = good | s, a).
 * Every round the learner sees all states, pulls exactly M arms, and every arm transitions
 * (the "restless" part: passive arms move too). Reward = number of arms in the good state, all N
 * of them, as in the healthcare-adherence model of UCWhittle (Wang et al. 2023); we score each
 * round by its conditional expectation Σᵢ P[aᵢ][sᵢ] (the expected number of good arms next round).
 *
 * Benchmark: the Whittle-index policy computed from the TRUE parameters (discount γ = 0.95),
 * run on the same random transitions. Regret(t) = its expected reward minus the learner's,
 * summed to t. Whittle is a heuristic, not the optimal policy, so regret can be negative.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;
  const GAMMA = 0.95;
  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

  // ----------------------------------------------------------- Whittle index ----
  // Arm parameters in a Float64Array at offset o: [P0_0, P0_1, P1_0, P1_1] (passive bad, passive
  // good, active bad, active good). For a fixed policy π with p0 = P[π(0)][0], p1 = P[π(1)][1] and
  // subsidy λ for passivity, V(1) − V(0) = (c₁ − c₀) / (1 + γ(p0 − p1)), where cₛ = s + λ·[π(s) = 0].
  // In state s, Q(s,1) − Q(s,0) = γ·(P1_s − P0_s)·(V(1) − V(0)) − λ, which is affine in λ for a
  // fixed policy, so the indifference subsidy has a closed form for each choice b of the action in
  // the other state; we keep the b that is itself optimal at that subsidy (policy-iteration check).
  function advantage(P, o, s, delta, lam, g) {
    const d = s ? P[o + 3] - P[o + 1] : P[o + 2] - P[o];
    return g * d * delta - lam;
  }
  /** Optimal V(1) − V(0) at subsidy λ by enumerating the four stationary policies. */
  function optGap(P, o, lam, g) {
    let best = -Infinity, gap = 0;
    for (let pi0 = 0; pi0 < 2; pi0++) for (let pi1 = 0; pi1 < 2; pi1++) {
      const p0 = pi0 ? P[o + 2] : P[o], p1 = pi1 ? P[o + 3] : P[o + 1];
      const c0 = pi0 ? 0 : lam, c1 = 1 + (pi1 ? 0 : lam);
      const det = (1 - g) * (1 + g * (p0 - p1));
      const v0 = (c0 * (1 - g * p1) + g * p0 * c1) / det;
      const v1 = ((1 - g * (1 - p0)) * c1 + g * (1 - p1) * c0) / det;
      if (v0 + v1 > best) { best = v0 + v1; gap = v1 - v0; }
    }
    return gap;
  }
  function whittle(P, o, s, g = GAMMA) {
    const other = 1 - s;
    for (let b = 1; b >= 0; b--) {
      // policy: active in s, action b in the other state
      const pi0 = s === 0 ? 1 : b, pi1 = s === 1 ? 1 : b;
      // Δ(λ) = (A + Bλ)/den  ⇒  γ·d_s·Δ(λ) − λ = 0
      const p0 = pi0 ? P[o + 2] : P[o], p1 = pi1 ? P[o + 3] : P[o + 1];
      const den = 1 + g * (p0 - p1);
      const B = (pi0 ? 0 : -1) + (pi1 ? 0 : 1);       // c₁ − c₀ = 1 + Bλ
      const ds = s ? P[o + 3] - P[o + 1] : P[o + 2] - P[o];
      const k = (g * ds) / den;                       // equation: k(1 + Bλ) = λ
      const denom = 1 - k * B;
      if (Math.abs(denom) < 1e-12) continue;
      const lam = k / denom;
      const delta = (1 + B * lam) / den;
      const adv = advantage(P, o, other, delta, lam, g);
      if ((b === 1 && adv >= -1e-10) || (b === 0 && adv <= 1e-10)) return lam;
    }
    // Fallback (non-indexable corner cases): bisection on the optimal advantage.
    let lo = -1 / (1 - g), hi = 1 / (1 - g);
    for (let i = 0; i < 50; i++) {
      const mid = (lo + hi) / 2;
      if (advantage(P, o, s, optGap(P, o, mid, g), mid, g) > 0) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /** Mark the M largest values in act (exact count, random tie-break). */
  function pickTop(vals, N, M, rng, act, order) {
    for (let i = 0; i < N; i++) order[i] = i;
    for (let i = N - 1; i > 0; i--) { const j = rng.int(i + 1); const x = order[i]; order[i] = order[j]; order[j] = x; }
    order.sort((x, y) => vals[y] - vals[x]);
    act.fill(0);
    for (let j = 0; j < M; j++) act[order[j]] = 1;
  }

  // ------------------------------------------------------------ families ----
  const families = [
    { id: 'oracle', name: 'Oracle reference', blurb: 'Knows the true transition matrices. Not a learner: it defines the benchmark.' },
    { id: 'index-learning', name: 'Learning Whittle indices', blurb: 'Estimate the transitions from observed state changes and plug them into the Whittle index.' },
    { id: 'baseline', name: 'Baselines', blurb: 'Myopic, round-robin and random scheduling.' },
  ];

  // --------------------------------------------------------- environments ----
  // generate(N, rng, P) fills P[4i .. 4i+3] = [P0_0, P0_1, P1_0, P1_1] for each arm i.
  const environments = [
    {
      id: 'adherence', name: 'Adherence (indexable)', short: 'Adherence',
      description: 'Healthcare-adherence style arms: "good" = adherent. Adherence is sticky (Pr(stay good) 0.6–0.95 when left alone), recovery is rare without help (0.02–0.3), and a call raises both chances (more so from the bad state). Acting always helps and the good state is always better, the structure under which the arms are indexable.',
      tags: ['indexable', 'healthcare'],
      generate(N, rng, P) {
        for (let i = 0; i < N; i++) {
          const o = 4 * i;
          P[o + 1] = 0.6 + 0.35 * rng.uniform();
          P[o] = 0.02 + 0.28 * rng.uniform();
          P[o + 2] = clamp01(P[o] + 0.1 + 0.4 * rng.uniform());
          P[o + 3] = clamp01(P[o + 1] + 0.2 * rng.uniform() * (1 - P[o + 1]) / 0.4);
        }
      },
    },
    {
      id: 'hetero', name: 'Heterogeneous arm types', short: 'Heterogeneous',
      description: 'Each arm is one of four types: responds mainly in the bad state, responds mainly in the good state (prevents relapse), barely responds, or recovers on its own. Effects range from 0 to 0.6, so knowing which arm is which matters more than in the adherence instance.',
      tags: ['indexable', 'heterogeneous'],
      generate(N, rng, P) {
        for (let i = 0; i < N; i++) {
          const o = 4 * i, type = rng.int(4);
          let p00, p01, e0, e1;
          if (type === 0) { p00 = 0.05 + 0.2 * rng.uniform(); p01 = 0.5 + 0.4 * rng.uniform(); e0 = 0.3 + 0.3 * rng.uniform(); e1 = 0.05 * rng.uniform(); }
          else if (type === 1) { p00 = 0.05 + 0.2 * rng.uniform(); p01 = 0.3 + 0.3 * rng.uniform(); e0 = 0.05 * rng.uniform(); e1 = 0.3 + 0.3 * rng.uniform(); }
          else if (type === 2) { p00 = 0.1 + 0.3 * rng.uniform(); p01 = 0.4 + 0.4 * rng.uniform(); e0 = 0.03 * rng.uniform(); e1 = 0.03 * rng.uniform(); }
          else { p00 = 0.5 + 0.3 * rng.uniform(); p01 = 0.6 + 0.3 * rng.uniform(); e0 = 0.1 * rng.uniform(); e1 = 0.1 * rng.uniform(); }
          P[o] = p00; P[o + 1] = p01; P[o + 2] = clamp01(p00 + e0); P[o + 3] = clamp01(p01 + e1);
        }
      },
    },
    {
      id: 'fatigue', name: 'Intervention fatigue (hard)', short: 'Hard',
      description: 'Hard, non-monotone instance: acting helps a lot in the bad state but hurts in the good state (fatigue: Pr(stay good) drops when pulled), and a third of the arms are near-copies with tiny differences. These arms break the usual sufficient conditions for indexability (acting always helps); a numerical check still finds a single Whittle threshold per state, so the hardness is in learning sign-flipping effects and near-ties, and Whittle remains only a heuristic for the coupled M-of-N problem — regret can go negative.',
      tags: ['non-monotone', 'hard'],
      generate(N, rng, P) {
        const base = [0.15 + 0.1 * rng.uniform(), 0.8 + 0.1 * rng.uniform()];
        for (let i = 0; i < N; i++) {
          const o = 4 * i;
          if (i % 3 === 2) {
            P[o] = base[0] + 0.02 * rng.uniform(); P[o + 1] = base[1] + 0.02 * rng.uniform();
            P[o + 2] = clamp01(P[o] + 0.35); P[o + 3] = clamp01(P[o + 1] - 0.15);
          } else {
            P[o] = 0.05 + 0.3 * rng.uniform(); P[o + 1] = 0.6 + 0.35 * rng.uniform();
            P[o + 2] = clamp01(P[o] + 0.15 + 0.5 * rng.uniform());
            P[o + 3] = clamp01(P[o + 1] - 0.05 - 0.3 * rng.uniform());
          }
        }
      },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of environments) ENV_BY_ID[e.id] = e;

  // ---------------------------------------------------------- algorithms ----
  // create(N, M, ctx) → { select(t, s, act), update(t, s, act, next) }; ctx.P is the true model
  // (only the oracle reads it). Counts: n[4i + 2a + s] transitions observed from (s, a), g[...] of
  // which ended good.
  function counts(N) { return { n: new Float64Array(4 * N), g: new Float64Array(4 * N) }; }
  function record(c, N, s, act, next) {
    for (let i = 0; i < N; i++) {
      const k = 4 * i + 2 * act[i] + s[i];
      c.n[k] += 1; c.g[k] += next[i];
    }
  }
  // map count slot (a, s) → parameter slot [P0_0, P0_1, P1_0, P1_1]: same layout (2a + s)

  function indexPolicy(N, M, fill) {
    const Q = new Float64Array(4 * N), idx = new Float64Array(N), order = new Int32Array(N);
    return {
      select(t, s, act, rng) {
        fill(Q, t, rng);
        for (let i = 0; i < N; i++) idx[i] = whittle(Q, 4 * i, s[i]);
        pickTop(idx, N, M, rng, act, order);
      },
    };
  }

  const algorithms = [
    {
      id: 'whittle-oracle', name: 'Whittle index (oracle)', family: 'oracle',
      params: [{ key: 'gamma', label: 'Discount γ', default: 0.95, min: 0.5, max: 0.995, step: 0.005 }],
      formula: 'pull the M arms with the largest W(sᵢ) = inf{λ : passive optimal in sᵢ with subsidy λ}, true P',
      regret: '0 by definition (it is the benchmark at γ = 0.95)',
      idea: 'Whittle (1988): give each arm a subsidy for staying passive; the index of a state is the subsidy that makes acting and resting equally good. With the true transitions this is the reference policy every learner is compared to. Oracle: it is told the model.',
      create(N, M, { params, P }) {
        const idx = new Float64Array(N), order = new Int32Array(N);
        const W = new Float64Array(2 * N);
        for (let i = 0; i < N; i++) { W[2 * i] = whittle(P, 4 * i, 0, params.gamma); W[2 * i + 1] = whittle(P, 4 * i, 1, params.gamma); }
        return {
          select(t, s, act, rng) {
            for (let i = 0; i < N; i++) idx[i] = W[2 * i + s[i]];
            pickTop(idx, N, M, rng, act, order);
          },
          update() {},
        };
      },
    },
    {
      id: 'ucwhittle', name: 'UCWhittle (simplified)', family: 'index-learning',
      params: [{ key: 'c', label: 'Confidence scale c', default: 0.5, min: 0, max: 3, step: 0.05 }],
      formula: 'P̃¹ₛ = P̂¹ₛ + c√(ln t / n¹ₛ),  P̃⁰ₛ = P̂⁰ₛ − c√(ln t / n⁰ₛ);  pull top-M W(sᵢ; P̃)',
      regret: 'Õ(√T) (episodic, Wang et al. 2023)',
      idea: 'Wang, Xiong, Liu, Jia & Tambe (2023): optimism for restless bandits — compute Whittle indices from the most favourable transitions in the confidence region. Simplified: for two-state arms the optimistic corner is taken directly (active probabilities at their upper bound, passive at their lower bound), recomputed every round instead of per episode.',
      create(N, M, { params }) {
        const c = counts(N);
        const pol = indexPolicy(N, M, (Q, t) => {
          const lt = Math.log(t + 1);
          for (let k = 0; k < 4 * N; k++) {
            const n = c.n[k], mean = n ? c.g[k] / n : 0.5, w = n ? params.c * Math.sqrt(lt / n) : 1;
            Q[k] = clamp01((k & 2) ? mean + w : mean - w);
          }
        });
        return { select: pol.select, update(t, s, act, next) { record(c, N, s, act, next); } };
      },
    },
    {
      id: 'ts-whittle', name: 'Thompson-sampling Whittle', family: 'index-learning',
      params: [{ key: 'H', label: 'Resample every H rounds', default: 1, min: 1, max: 50, step: 1 }],
      formula: 'P̃ₖ ~ Beta(1 + goodₖ, 1 + badₖ) for every (arm, action, state);  pull top-M W(sᵢ; P̃)',
      regret: 'Õ(√T) Bayesian (posterior-sampling RL)',
      idea: 'Posterior sampling for restless bandits (in the spirit of Jung & Tewari 2019 and Akbarzadeh & Mahajan 2022): draw each transition probability from its Beta posterior, compute the Whittle indices of that sampled model and act on them. Resampled every H rounds.',
      create(N, M, { params }) {
        const c = counts(N), H = Math.max(1, Math.round(params.H));
        const pol = indexPolicy(N, M, (Q, t, rng) => {
          if ((t - 1) % H !== 0) return;
          for (let k = 0; k < 4 * N; k++) Q[k] = rng.beta(1 + c.g[k], 1 + c.n[k] - c.g[k]);
        });
        return { select: pol.select, update(t, s, act, next) { record(c, N, s, act, next); } };
      },
    },
    {
      id: 'ce-whittle', name: 'Plug-in Whittle (greedy)', family: 'index-learning',
      params: [],
      formula: 'P̂ = (good + 1)/(n + 2);  pull top-M W(sᵢ; P̂)',
      regret: 'no guarantee (certainty equivalence)',
      idea: 'Certainty equivalence: plug the posterior-mean estimates into the Whittle index with no optimism. Passive transitions are learned for free, but an arm whose active effect is underestimated early may never be pulled again.',
      create(N, M) {
        const c = counts(N);
        const pol = indexPolicy(N, M, (Q) => {
          for (let k = 0; k < 4 * N; k++) Q[k] = (c.g[k] + 1) / (c.n[k] + 2);
        });
        return { select: pol.select, update(t, s, act, next) { record(c, N, s, act, next); } };
      },
    },
    {
      id: 'myopic', name: 'Myopic (greedy one-step)', family: 'baseline',
      params: [],
      formula: 'pull top-M of P̂¹(sᵢ) − P̂⁰(sᵢ)',
      regret: 'none (ignores the future)',
      idea: 'Pull the arms whose estimated chance of being good next round rises the most when pulled. Ignores that a pull now also changes where the arm will be later, which matters when states are sticky.',
      create(N, M) {
        const c = counts(N), idx = new Float64Array(N), order = new Int32Array(N);
        return {
          select(t, s, act, rng) {
            for (let i = 0; i < N; i++) {
              const k1 = 4 * i + 2 + s[i], k0 = 4 * i + s[i];
              idx[i] = (c.g[k1] + 1) / (c.n[k1] + 2) - (c.g[k0] + 1) / (c.n[k0] + 2);
            }
            pickTop(idx, N, M, rng, act, order);
          },
          update(t, s, act, next) { record(c, N, s, act, next); },
        };
      },
    },
    {
      id: 'round-robin', name: 'Round-robin', family: 'baseline',
      params: [],
      formula: 'pull arms (M(t−1) + j) mod N, j = 0..M−1',
      regret: 'Θ(T)',
      idea: 'Cycle through the arms in a fixed order, ignoring their states. A common real-world scheduling default.',
      create(N, M) {
        return {
          select(t, s, act) {
            act.fill(0);
            for (let j = 0; j < M; j++) act[(M * (t - 1) + j) % N] = 1;
          },
          update() {},
        };
      },
    },
    {
      id: 'random', name: 'Random', family: 'baseline',
      params: [],
      formula: 'pull a uniformly random set of M arms',
      regret: 'Θ(T)',
      idea: 'Ignores states and history.',
      create(N, M) {
        const idx = new Float64Array(N), order = new Int32Array(N);
        return {
          select(t, s, act, rng) { for (let i = 0; i < N; i++) idx[i] = rng.uniform(); pickTop(idx, N, M, rng, act, order); },
          update() {},
        };
      },
    },
  ];
  const ALGO_BY_ID = {};
  for (const a of algorithms) ALGO_BY_ID[a.id] = a;
  function defaults(def) {
    const o = {};
    for (const p of def.params) o[p.key] = p.default;
    return o;
  }

  // -------------------------------------------------------------- runner ----
  function createRunner(config) {
    const env = ENV_BY_ID[config.env];
    if (!env) throw new Error('Unknown environment: ' + config.env);
    const N = config.K | 0, T = config.T | 0, runs = config.runs | 0, seed = config.seed >>> 0;
    if (N < 2) throw new Error('Need at least 2 arms');
    const ex = config.extra || {};
    const M = Math.min(N - 1, Math.max(1, Math.round(ex.M != null ? +ex.M : 3)));
    const P = new Float64Array(4 * N), U = new Float64Array(T * N), s0 = new Uint8Array(N), bench = new Float64Array(T);
    const s = new Uint8Array(N), next = new Uint8Array(N), act = new Uint8Array(N);
    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaults(def), spec.params || {}),
        reg: new Float64Array(T), reg2: new Float64Array(T), rw: new Float64Array(T),
        finals: [], totRew: 0, timeMs: 0,
      };
    });
    let done = 0;

    // Simulate a policy on this run's shared randomness; out[t] = expected reward of round t.
    function play(agent, rng, out) {
      s.set(s0);
      for (let t = 0; t < T; t++) {
        agent.select(t + 1, s, act, rng);
        let k = 0;
        for (let i = 0; i < N; i++) { act[i] = act[i] ? 1 : 0; k += act[i]; }
        for (let i = 0; k > M && i < N; i++) if (act[i]) { act[i] = 0; k--; }
        for (let i = 0; k < M && i < N; i++) if (!act[i]) { act[i] = 1; k++; }
        let r = 0;
        for (let i = 0; i < N; i++) {
          const p = P[4 * i + 2 * act[i] + s[i]];
          r += p;
          next[i] = U[t * N + i] < p ? 1 : 0;
        }
        out[t] = r;
        agent.update(t + 1, s, act, next);
        s.set(next);
      }
    }

    const tmp = new Float64Array(T);
    function runOne(run) {
      const envRng = Rng(hash32(seed, run, 'env'));
      env.generate(N, envRng, P);
      for (let i = 0; i < N; i++) s0[i] = envRng.uniform() < 0.5 ? 1 : 0;
      for (let i = 0; i < T * N; i++) U[i] = envRng.uniform();
      const benchDef = ALGO_BY_ID['whittle-oracle'];
      play(benchDef.create(N, M, { T, P, params: { gamma: GAMMA } }), Rng(hash32(seed, run, 'benchmark')), bench);
      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const agent = slot.def.create(N, M, { T, P, params: slot.params, rng, env });
        const t0 = util.now();
        play(agent, rng, tmp);
        slot.timeMs += util.now() - t0;
        let cum = 0, tot = 0;
        for (let t = 0; t < T; t++) {
          cum += bench[t] - tmp[t]; tot += tmp[t];
          slot.reg[t] += cum; slot.reg2[t] += cum * cum; slot.rw[t] += tmp[t];
        }
        slot.finals.push(cum);
        slot.totRew += tot / T;
      }
    }

    return {
      config: { env: env.id, K: N, T, runs, seed, extra: { M } },
      get done() { return done; },
      get finished() { return done >= runs; },
      step(n = 1) {
        const stop = Math.min(runs, done + n);
        while (done < stop) runOne(done++);
        return done;
      },
      results() {
        const R = Math.max(1, done);
        return slots.map((sl) => {
          const regret = new Float64Array(T), regretSE = new Float64Array(T), reward = new Float64Array(T);
          for (let t = 0; t < T; t++) {
            const m = sl.reg[t] / R;
            regret[t] = m;
            regretSE[t] = R > 1 ? Math.sqrt(Math.max(0, sl.reg2[t] / R - m * m) / (R - 1)) : 0;
            reward[t] = sl.rw[t] / R;
          }
          const sorted = sl.finals.slice().sort((a, b) => a - b);
          const mean = sl.totRew / R;
          const summary = {
            regret: regret[T - 1], regretSE: regretSE[T - 1],
            regretMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            meanReward: mean, goodShare: mean / N, msPerRun: done ? sl.timeMs / done : 0,
          };
          return {
            id: sl.def.id, name: sl.def.name, family: sl.def.family, params: sl.params, runs: done,
            curves: { regret, reward }, curvesSE: { regret: regretSE },
            summary, finals: sorted,
          };
        });
      },
    };
  }

  return MAB.registerSetting({
    id: 'restless',
    name: 'Restless bandits',
    short: 'Restless',
    blurb: 'Arms are two-state Markov chains that keep moving whether pulled or not; pull M of N each round to keep as many as possible in the good state.',
    families, environments, algorithms,
    defaultEnv: 'adherence',
    defaultLineup: ['whittle-oracle', 'ucwhittle', 'ts-whittle', 'ce-whittle', 'myopic', 'round-robin', 'random'],
    presets: [
      { label: 'Learning the Whittle index', algos: ['whittle-oracle', 'ucwhittle', 'ts-whittle', 'ce-whittle'], env: 'adherence' },
      { label: 'Index vs myopic', algos: ['whittle-oracle', 'ucwhittle', 'myopic', 'round-robin'], env: 'hetero' },
      { label: 'Hard instance', algos: ['whittle-oracle', 'ucwhittle', 'ts-whittle', 'myopic', 'random'], env: 'fatigue' },
    ],
    controls: {
      K: { label: 'Arms (N)', min: 3, max: 30, default: 10 },
      T: { label: 'Horizon (T rounds)', options: [100, 250, 500, 1000, 2000], default: 500 },
      runs: { label: 'Runs averaged', options: [20, 50, 100, 200], default: 50 },
      extra: [{ key: 'M', label: 'Pulls per round (M)', min: 1, max: 10, step: 1, default: 3 }],
    },
    xLabel: 'round t',
    metrics: [
      { key: 'regret', label: 'Regret vs oracle Whittle', better: 'lower', fmt: 'num', band: true, note: 'Expected reward of the Whittle-index policy with the true transitions (γ = 0.95), run on the same random transitions, minus the learner\'s. Whittle is a strong heuristic, not the optimum, so this can dip below 0.' },
      { key: 'reward', label: 'Good arms per round', better: 'higher', fmt: 'fixed2', smooth: true, note: 'Expected number of arms in the good state after each round.' },
    ],
    final: { key: 'regret', se: 'regretSE', median: 'regretMedian', label: 'Regret vs Whittle oracle', better: 'lower' },
    tiles: [
      { key: 'meanReward', label: 'Most arms kept good', better: 'higher', fmt: 'fixed2', sub: 'good arms per round' },
    ],
    columns: [
      { key: 'regret', label: 'Regret', fmt: 'num', better: 'lower' },
      { key: 'regretSE', label: '95% CI', fmt: 'ci' },
      { key: 'regretMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'meanReward', label: 'Good arms / round', fmt: 'fixed2', better: 'higher' },
      { key: 'goodShare', label: 'Share good', fmt: 'pct', better: 'higher' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: false,
    createRunner,
    _internal: { whittle, optGap, advantage },
  });
});
