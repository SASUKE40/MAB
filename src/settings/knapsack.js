/*
 * Setting: bandits with knapsacks (BwK).
 *
 * Each pull of arm a yields a reward in {0, 1} and consumes one unit of resource j with
 * probability c[a][j] (d resources, each with budget B = β·T). The run stops as soon as any
 * resource has less than one unit left, or at the horizon T. Algorithms may also skip a round
 * (select returns -1: no reward, no consumption), which plays the role of the "null arm".
 *
 * Benchmark: OPT_LP = T · max { Σₐ pₐ μₐ : Σₐ pₐ cₐⱼ ≤ β ∀j, Σₐ pₐ ≤ 1, p ≥ 0 }, the LP relaxation
 * on the true means; for the contextual instance the LP ranges over context-dependent
 * distributions p(a | x), and for the shifting instance over phase-dependent ones. It
 * upper-bounds the expected reward of any policy. Regret(t) = OPT_LP · t / T − expected reward collected by t.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;

  // ------------------------------------------------------------ small LP ----
  /**
   * Dense-tableau simplex for  max c·x  s.t.  A x ≤ b,  x ≥ 0  with b ≥ 0 (so x = 0 is feasible).
   * A is row-major m × n. Writes the solution into `out` (length ≥ n) and returns the optimum.
   * Workspace is reused between calls; sizes up to (nMax, mMax).
   */
  function makeLP(nMax, mMax) {
    const W = nMax + mMax + 1;
    const tab = new Float64Array((mMax + 1) * W);
    const basis = new Int32Array(mMax);
    return function solve(c, A, b, n, m, out) {
      const w = n + m + 1, rhs = w - 1;
      tab.fill(0, 0, (m + 1) * w);
      for (let i = 0; i < m; i++) {
        const row = i * w;
        for (let j = 0; j < n; j++) tab[row + j] = A[i * n + j];
        tab[row + n + i] = 1;
        tab[row + rhs] = Math.max(0, b[i]);
        basis[i] = n + i;
      }
      const z = m * w;
      for (let j = 0; j < n; j++) tab[z + j] = -c[j];
      for (let it = 0; it < 50 * (n + m); it++) {
        let e = -1, best = -1e-11;
        for (let j = 0; j < n + m; j++) if (tab[z + j] < best) { best = tab[z + j]; e = j; }
        if (e < 0) break;
        let l = -1, ratio = Infinity;
        for (let i = 0; i < m; i++) {
          const a = tab[i * w + e];
          if (a > 1e-11) {
            const r = tab[i * w + rhs] / a;
            if (r < ratio - 1e-13 || (r <= ratio + 1e-13 && l >= 0 && basis[i] < basis[l])) { ratio = r; l = i; }
          }
        }
        if (l < 0) break; // unbounded (cannot happen with Σx ≤ 1 rows)
        const pr = l * w, pv = tab[pr + e];
        for (let j = 0; j < w; j++) tab[pr + j] /= pv;
        for (let i = 0; i <= m; i++) {
          if (i === l) continue;
          const f = tab[i * w + e];
          if (f === 0) continue;
          const ri = i * w;
          for (let j = 0; j < w; j++) tab[ri + j] -= f * tab[pr + j];
        }
        basis[l] = e;
      }
      for (let j = 0; j < n; j++) out[j] = 0;
      for (let i = 0; i < m; i++) if (basis[i] < n) out[basis[i]] = Math.max(0, tab[i * w + rhs]);
      return tab[z + rhs];
    };
  }

  const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

  // ------------------------------------------------------------ families ----
  const families = [
    { id: 'baseline', name: 'Cost-unaware & greedy baselines', blurb: 'Ignore the budget, or chase reward per unit of cost without planning.' },
    { id: 'primal-dual', name: 'Primal–dual', blurb: 'Price each resource with a multiplicative-weights dual and pick the arm with the best reward net of prices.' },
    { id: 'optimistic-lp', name: 'Optimistic LP', blurb: 'Solve the BwK linear program on optimistic estimates every round and play its solution.' },
    { id: 'contextual', name: 'Contextual', blurb: 'Read a context each round; rewards and costs depend on it linearly.' },
  ];

  // --------------------------------------------------------- environments ----
  // Every environment writes, for t = 0..T-1: mr[t*K+a] (mean reward), mc[(t*K+a)*d+j] (mean
  // cost), and the context x[t*dx + i]. It returns { d, dx, opt } where opt is the per-round LP
  // value (OPT_LP / T). Realized Bernoulli outcomes are drawn by the runner from shared uniforms.
  function stationaryLP(K, d, mu, cost, beta) {
    const lp = makeLP(K, d + 1);
    const A = new Float64Array((d + 1) * K), b = new Float64Array(d + 1), out = new Float64Array(K);
    for (let j = 0; j < d; j++) { for (let a = 0; a < K; a++) A[j * K + a] = cost[a * d + j]; b[j] = beta; }
    for (let a = 0; a < K; a++) A[d * K + a] = 1;
    b[d] = 1;
    return lp(mu, A, b, K, d + 1, out);
  }
  /**
   * One-resource LP over (type, arm) pairs, types occurring with frequencies q:
   * max Σᵢ qᵢ Σₐ R[i,a] p(a|i)  s.t.  Σᵢ qᵢ Σₐ C[i,a] p(a|i) ≤ β,  Σₐ p(a|i) ≤ 1 for each i.
   */
  function typedLP(K, nX, q, R, C, beta) {
    const n = nX * K, m = 1 + nX;
    const A = new Float64Array(m * n), b = new Float64Array(m), cvec = new Float64Array(n), out = new Float64Array(n);
    for (let i = 0; i < nX; i++) for (let a = 0; a < K; a++) {
      const v = i * K + a;
      cvec[v] = R[v] * q[i]; A[v] = C[v] * q[i]; A[(1 + i) * n + v] = 1;
    }
    b[0] = beta;
    for (let i = 0; i < nX; i++) b[1 + i] = 1;
    return makeLP(n, m)(cvec, A, b, n, m, out);
  }
  function writeStationary(K, T, d, mu, cost, mr, mc, x) {
    for (let t = 0; t < T; t++) {
      x[t] = 1;
      for (let a = 0; a < K; a++) {
        mr[t * K + a] = mu[a];
        for (let j = 0; j < d; j++) mc[(t * K + a) * d + j] = cost[a * d + j];
      }
    }
  }
  function oneResourceInstance(K, rng) {
    const mu = new Float64Array(K), cost = new Float64Array(K);
    let top = 0;
    for (let a = 0; a < K; a++) {
      mu[a] = 0.1 + 0.8 * rng.uniform();
      cost[a] = clamp(0.03 + mu[a] * mu[a] + 0.3 * (rng.uniform() - 0.5), 0.03, 1);
      if (mu[a] > mu[top]) top = a;
    }
    cost[top] = Math.max(cost[top], 0.9);   // the most rewarding arm is also the most expensive
    return { mu, cost };
  }

  const environments = [
    {
      id: 'one', name: 'One resource, tight budget', short: 'One resource',
      description: 'Bernoulli rewards and Bernoulli consumption of a single resource. Cost grows with reward and the most rewarding arm is the most expensive, so a cost-blind learner burns the budget early. Budget B = β·T.',
      tags: ['stochastic', '1 resource'],
      generate(K, T, beta, rng, mr, mc, x) {
        const { mu, cost } = oneResourceInstance(K, rng);
        writeStationary(K, T, 1, mu, cost, mr, mc, x);
        return { d: 1, dx: 1, opt: stationaryLP(K, 1, mu, cost, beta) };
      },
    },
    {
      id: 'two', name: 'Two resources', short: 'Two resources',
      description: 'Each pull consumes two resources with independent Bernoulli draws; arms trade one resource against the other, so the LP optimum mixes two arms. Both budgets are β·T.',
      tags: ['stochastic', '2 resources'],
      generate(K, T, beta, rng, mr, mc, x) {
        const d = 2, mu = new Float64Array(K), cost = new Float64Array(K * d);
        for (let a = 0; a < K; a++) {
          mu[a] = 0.1 + 0.8 * rng.uniform();
          const s = rng.uniform();                 // how the arm splits its consumption
          const tot = clamp(0.15 + 1.1 * mu[a] + 0.3 * (rng.uniform() - 0.5), 0.1, 1.6);
          cost[a * d] = clamp(tot * s, 0.02, 1);
          cost[a * d + 1] = clamp(tot * (1 - s), 0.02, 1);
        }
        writeStationary(K, T, d, mu, cost, mr, mc, x);
        return { d, dx: 1, opt: stationaryLP(K, d, mu, cost, beta) };
      },
    },
    {
      id: 'shift', name: 'Shifting (adversarial-ish)', short: 'Shifting',
      description: 'One resource. Halfway through, the reward means are reversed by rank (the best arm becomes the worst) while costs stay put. Benchmark: the LP over (phase, arm), i.e. the best policy that knows the switch and may split the budget between the two halves; it upper-bounds every policy. Methods that average all history adapt slowly; LagrangeBwK\'s adversarial (EXP3) primal is built for changing outcomes.',
      tags: ['changes', 'adversarial', '1 resource'],
      generate(K, T, beta, rng, mr, mc, x) {
        const { mu, cost } = oneResourceInstance(K, rng);
        const order = Array.from({ length: K }, (_, i) => i).sort((p, q) => mu[q] - mu[p]);
        const mu2 = new Float64Array(K);
        for (let r = 0; r < K; r++) mu2[order[r]] = mu[order[K - 1 - r]];
        const half = Math.floor(T / 2);
        for (let t = 0; t < T; t++) {
          x[t] = 1;
          const m = t < half ? mu : mu2;
          for (let a = 0; a < K; a++) { mr[t * K + a] = m[a]; mc[t * K + a] = cost[a]; }
        }
        const R = new Float64Array(2 * K), C = new Float64Array(2 * K);
        R.set(mu, 0); R.set(mu2, K); C.set(cost, 0); C.set(cost, K);
        return { d: 1, dx: 1, opt: typedLP(K, 2, Float64Array.of(half / T, (T - half) / T), R, C, beta) };
      },
    },
    {
      id: 'ctx', name: 'Contextual, linear rewards & costs', short: 'Contextual',
      description: 'Each round shows one of 8 context types x = (1, z₁, z₂), drawn uniformly. Mean reward θₐ·x and mean cost wₐ·x are linear in x; one resource. The LP benchmark ranges over context-dependent policies p(a | x), so it pays to save budget for good contexts.',
      tags: ['context', '1 resource'],
      contextual: true,
      generate(K, T, beta, rng, mr, mc, x) {
        const dx = 3, nX = 8, d = 1;
        const types = new Float64Array(nX * dx);
        for (let i = 0; i < nX; i++) { types[i * dx] = 1; types[i * dx + 1] = rng.uniform(); types[i * dx + 2] = rng.uniform(); }
        const th = new Float64Array(K * dx), w = new Float64Array(K * dx);
        for (let a = 0; a < K; a++) {
          th[a * dx] = 0.05 + 0.3 * rng.uniform(); th[a * dx + 1] = 0.3 * rng.uniform(); th[a * dx + 2] = 0.3 * rng.uniform();
          w[a * dx] = 0.05 + 0.25 * rng.uniform(); w[a * dx + 1] = 0.35 * rng.uniform(); w[a * dx + 2] = 0.35 * rng.uniform();
        }
        const R = new Float64Array(nX * K), C = new Float64Array(nX * K);
        for (let i = 0; i < nX; i++) for (let a = 0; a < K; a++) {
          let r = 0, c = 0;
          for (let k = 0; k < dx; k++) { r += th[a * dx + k] * types[i * dx + k]; c += w[a * dx + k] * types[i * dx + k]; }
          R[i * K + a] = clamp(r, 0, 1); C[i * K + a] = clamp(c, 0, 1);
        }
        for (let t = 0; t < T; t++) {
          const i = rng.int(nX);
          for (let k = 0; k < dx; k++) x[t * dx + k] = types[i * dx + k];
          for (let a = 0; a < K; a++) { mr[t * K + a] = R[i * K + a]; mc[t * K + a] = C[i * K + a]; }
        }
        const q = new Float64Array(nX).fill(1 / nX);
        return { d, dx, opt: typedLP(K, nX, q, R, C, beta) };
      },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of environments) ENV_BY_ID[e.id] = e;

  // ------------------------------------------------------------ helpers ----
  function armStats(K, d) {
    return { n: new Float64Array(K), r: new Float64Array(K), c: new Float64Array(K * d) };
  }
  function recordArm(s, d, a, r, cv) {
    s.n[a] += 1; s.r[a] += r;
    for (let j = 0; j < d; j++) s.c[a * d + j] += cv[j];
  }
  /** Confidence radius of Badanidiyuru et al.: √(C·ν/n) + C/n, with C = cr·ln(t+1). */
  function rad(nu, n, C) { return Math.sqrt((C * Math.max(nu, 0)) / n) + C / n; }

  /** Hedge over d resources plus a "time" dummy resource (index d) whose gain is always β. */
  function makeDual(d, beta, eta) {
    const logw = new Float64Array(d + 1), lam = new Float64Array(d + 1);
    function refresh() {
      let m = -Infinity, s = 0;
      for (let j = 0; j <= d; j++) if (logw[j] > m) m = logw[j];
      for (let j = 0; j <= d; j++) { lam[j] = Math.exp(logw[j] - m); s += lam[j]; }
      for (let j = 0; j <= d; j++) lam[j] /= s;
    }
    refresh();
    return {
      lam,
      update(cv) {           // cv = observed consumption (null for a skipped round)
        for (let j = 0; j < d; j++) logw[j] += eta * (cv ? cv[j] : 0);
        logw[d] += eta * beta;
        refresh();
      },
    };
  }

  // ---------------------------------------------------------- algorithms ----
  const crParam = { key: 'cr', label: 'Confidence scale C', default: 0.5, min: 0, max: 3, step: 0.05 };
  const algorithms = [
    {
      id: 'ucb1', name: 'UCB1 (ignores costs)', family: 'baseline',
      params: [{ key: 'c', label: 'Exploration c', default: 1.414, min: 0, max: 3, step: 0.01 }],
      formula: 'aₜ = argmaxₐ μ̂ₐ + c·√(ln t / nₐ)',
      regret: 'Ω(T) in BwK: spends the budget on the best-reward arm',
      idea: 'Plain UCB1 on rewards. It never looks at consumption, so when the best-reward arm is expensive it exhausts the budget early and then collects nothing.',
      create(K, { params }) {
        const n = new Float64Array(K), s = new Float64Array(K), idx = new Float64Array(K);
        return {
          select(t, x, rem, rng) {
            const u = util.untried(n);
            if (u >= 0) return u;
            const lt = Math.log(t);
            for (let a = 0; a < K; a++) idx[a] = s[a] / n[a] + params.c * Math.sqrt(lt / n[a]);
            return util.argmax(idx, rng);
          },
          update(a, r) { if (a >= 0) { n[a] += 1; s[a] += r; } },
        };
      },
    },
    {
      id: 'ucb-ratio', name: 'Reward-per-cost UCB', family: 'baseline',
      params: [{ key: 'c', label: 'Exploration c', default: 1, min: 0, max: 3, step: 0.01 }],
      formula: 'aₜ = argmaxₐ UCB(μₐ) / maxⱼ LCB(cₐⱼ)',
      regret: 'none in general (greedy on a ratio; ignores the time constraint)',
      idea: 'Greedy bang-per-buck: optimistic reward divided by optimistic (low) consumption of the scarcest resource. Good when budget binds hard, but it also starves the horizon when budget is loose.',
      create(K, { params, d }) {
        const s = armStats(K, d), idx = new Float64Array(K);
        return {
          select(t, x, rem, rng) {
            const u = util.untried(s.n);
            if (u >= 0) return u;
            const lt = Math.log(t);
            for (let a = 0; a < K; a++) {
              const b = params.c * Math.sqrt(lt / s.n[a]);
              let cmax = 0;
              for (let j = 0; j < d; j++) cmax = Math.max(cmax, s.c[a * d + j] / s.n[a] - b);
              idx[a] = Math.min(1, s.r[a] / s.n[a] + b) / Math.max(cmax, 0.01);
            }
            return util.argmax(idx, rng);
          },
          update(a, r, cv) { if (a >= 0) recordArm(s, d, a, r, cv); },
        };
      },
    },
    {
      id: 'pd-bwk', name: 'PD-BwK', family: 'primal-dual',
      params: [crParam],
      formula: 'aₜ = argmaxₐ UCBₜ(μₐ) / (vₜ · LCBₜ(cₐ));  vⱼ ← vⱼ (1+ε)^{LCB(cₐⱼ)}',
      regret: 'Õ(√(K·OPT) + OPT·√(K/B))',
      idea: 'Badanidiyuru, Kleinberg & Slivkins (2013/2018). Multiplicative weights v price the resources (plus time, consumed at rate β); each round pulls the arm with the best optimistic reward per unit of priced optimistic cost. Resources consumed faster than their pace become expensive.',
      create(K, { params, d, beta, B }) {
        const s = armStats(K, d), idx = new Float64Array(K);
        const D = d + 1, v = new Float64Array(D).fill(1), L = new Float64Array(K * D);
        const eps = Math.sqrt(Math.log(D) / B);
        let last = -1;
        return {
          select(t, x, rem, rng) {
            const u = util.untried(s.n);
            if (u >= 0) { for (let j = 0; j < d; j++) L[u * D + j] = 0; L[u * D + d] = beta; return (last = u); }
            const C = params.cr * Math.log(t + 1);
            let vs = 0;
            for (let j = 0; j < D; j++) vs += v[j];
            for (let a = 0; a < K; a++) {
              const na = s.n[a], mr = s.r[a] / na;
              let den = v[d] * beta;
              for (let j = 0; j < d; j++) {
                const mc = s.c[a * d + j] / na;
                const l = Math.max(0, mc - rad(mc, na, C));
                L[a * D + j] = l;
                den += v[j] * l;
              }
              L[a * D + d] = beta;
              idx[a] = Math.min(1, mr + rad(mr, na, C)) / Math.max(den / vs, 1e-9);
            }
            return (last = util.argmax(idx, rng));
          },
          update(a, r, cv) {
            if (a < 0) return;
            recordArm(s, d, a, r, cv);
            let m = 0;
            for (let j = 0; j < D; j++) { v[j] *= Math.pow(1 + eps, L[last * D + j]); if (v[j] > m) m = v[j]; }
            for (let j = 0; j < D; j++) v[j] /= m;   // rescale only (argmax is invariant)
          },
        };
      },
    },
    {
      id: 'ucb-bwk', name: 'UCB-BwK', family: 'optimistic-lp',
      params: [crParam],
      formula: 'pₜ = argmax { Σ pₐ UCB(μₐ) : Σ pₐ LCB(cₐⱼ) ≤ Bⱼ,rem / (T−t+1), Σ pₐ ≤ 1 };  aₜ ~ pₜ',
      regret: 'Õ(OPT·√(K/B) + √(K·OPT))',
      idea: 'Agrawal & Devanur (2014). Every round solve the BwK linear program with optimistic rewards and optimistic (low) costs, and sample an arm from its solution (or skip). Re-solving with the remaining budget per remaining round is a common practical variant of the fixed β·T constraint.',
      create(K, { params, d, T }) {
        const s = armStats(K, d), lp = makeLP(K, d + 1);
        const A = new Float64Array((d + 1) * K), b = new Float64Array(d + 1), cv = new Float64Array(K), p = new Float64Array(K);
        for (let a = 0; a < K; a++) A[d * K + a] = 1;
        b[d] = 1;
        return {
          select(t, x, rem, rng) {
            const u = util.untried(s.n);
            if (u >= 0) return u;
            const C = params.cr * Math.log(t + 1);
            for (let a = 0; a < K; a++) {
              const na = s.n[a], mr = s.r[a] / na;
              cv[a] = Math.min(1, mr + rad(mr, na, C));
              for (let j = 0; j < d; j++) {
                const mc = s.c[a * d + j] / na;
                A[j * K + a] = Math.max(0, mc - rad(mc, na, C));
              }
            }
            for (let j = 0; j < d; j++) b[j] = rem[j] / (T - t + 1);
            lp(cv, A, b, K, d + 1, p);
            let uu = rng.uniform();
            for (let a = 0; a < K; a++) { uu -= p[a]; if (uu < 0) return a; }
            return -1;
          },
          update(a, r, c) { if (a >= 0) recordArm(s, d, a, r, c); },
        };
      },
    },
    {
      id: 'lagrange-bwk', name: 'LagrangeBwK', family: 'primal-dual',
      params: [
        { key: 'primal', label: 'Primal: 0 = EXP3, 1 = UCB', default: 0, min: 0, max: 1, step: 1 },
        { key: 'eta', label: 'Dual rate scale', default: 1, min: 0.1, max: 10, step: 0.1 },
      ],
      formula: 'ℒ(a, λ) = r(a) + 1 − (T/B)·Σⱼ λⱼ cⱼ(a);  primal (EXP3/UCB) maximizes ℒ, dual (Hedge over resources + time) minimizes it',
      regret: 'O(√(KT log T)) stochastic (with time-dummy); O(log T)-competitive adversarial',
      idea: 'Immorlica, Sankararaman, Schapire & Slivkins (2019). A repeated Lagrangian game: Hedge prices the resources (a dummy "time" resource is consumed at rate β), and a bandit learner, EXP3 by default (adversarial) or UCB, plays arms against those prices. The null arm (skip) is one of the primal actions.',
      create(K, { params, d, beta, T }) {
        const etaD = params.eta * Math.sqrt(2 * Math.log(d + 1) / T);
        const dual = makeDual(d, beta, etaD), lam = dual.lam;
        const s = armStats(K, d), idx = new Float64Array(K);
        const K1 = K + 1, logw = new Float64Array(K1), prob = new Float64Array(K1);
        const gam = util.exp3Gamma(K1, T), eta = gam / K1;
        const lo = 1 - 1 / beta, span = 1 + 1 / beta;
        let pa = 1;
        return {
          select(t, x, rem, rng) {
            if (params.primal >= 0.5) {
              const u = util.untried(s.n);
              if (u >= 0) return u;
              const lt = Math.log(t);
              for (let a = 0; a < K; a++) {
                const na = s.n[a], b = Math.sqrt(2 * lt / na);
                let pen = 0;
                for (let j = 0; j < d; j++) pen += lam[j] * Math.max(0, s.c[a * d + j] / na - b);
                idx[a] = Math.min(1, s.r[a] / na + b) - pen / beta - lam[d];
              }
              const a = util.argmax(idx, rng);
              return idx[a] >= -lam[d] ? a : -1;
            }
            let m = -Infinity;
            for (let a = 0; a < K1; a++) if (logw[a] > m) m = logw[a];
            let sw = 0;
            for (let a = 0; a < K1; a++) { prob[a] = Math.exp(logw[a] - m); sw += prob[a]; }
            for (let a = 0; a < K1; a++) prob[a] = (1 - gam) * prob[a] / sw + gam / K1;
            const a = util.sampleIndex(prob, rng);
            pa = prob[a];
            return a === K ? -1 : a;
          },
          update(a, r, cv) {
            if (a >= 0) recordArm(s, d, a, r, cv);
            if (params.primal < 0.5) {
              let L = (a >= 0 ? r : 0) + 1 - lam[d];
              if (a >= 0) for (let j = 0; j < d; j++) L -= lam[j] * cv[j] / beta;
              const arm = a < 0 ? K : a;
              logw[arm] += eta * ((L - lo) / span) / pa;
            }
            dual.update(a >= 0 ? cv : null);
          },
        };
      },
    },
    {
      id: 'cbwk', name: 'CBwK (linear, simplified)', family: 'contextual',
      params: [
        { key: 'alpha', label: 'Confidence width α', default: 0.5, min: 0, max: 3, step: 0.05 },
        { key: 'eta', label: 'Dual rate scale', default: 1, min: 0.1, max: 10, step: 0.1 },
      ],
      formula: 'aₜ = argmaxₐ [θ̂ₐ·x + α‖x‖_{Aₐ⁻¹}] − (T/B)·Σⱼ λⱼ [ŵₐⱼ·x − α‖x‖_{Aₐ⁻¹}]⁺  (skip if < 0)',
      regret: 'Õ((OPT/B + 1)·d√T) (linear CBwK)',
      idea: 'A simplified linear version of contextual BwK (Agrawal, Devanur & Li 2016; linear form as in Agrawal & Devanur 2016): per-arm ridge regressions give optimistic rewards and costs, Hedge over resources (plus time) sets the prices, and the arm with the best priced score is played, or the round is skipped to save budget for better contexts.',
      create(K, { params, d, dx, beta, T }) {
        const dual = makeDual(d, beta, params.eta * Math.sqrt(2 * Math.log(d + 1) / T)), lam = dual.lam;
        const Ainv = new Float64Array(K * dx * dx), br = new Float64Array(K * dx), bc = new Float64Array(K * dx * d);
        for (let a = 0; a < K; a++) for (let i = 0; i < dx; i++) Ainv[a * dx * dx + i * dx + i] = 1;
        const v = new Float64Array(dx), idx = new Float64Array(K);
        let xs = null;
        return {
          select(t, x, rem, rng) {
            xs = x;
            for (let a = 0; a < K; a++) {
              const o = a * dx * dx;
              let q = 0;
              for (let i = 0; i < dx; i++) { let z = 0; for (let k = 0; k < dx; k++) z += Ainv[o + i * dx + k] * x[k]; v[i] = z; q += z * x[i]; }
              const w = params.alpha * Math.sqrt(Math.max(q, 0));
              let r = 0;
              for (let i = 0; i < dx; i++) r += v[i] * br[a * dx + i];
              let pen = 0;
              for (let j = 0; j < d; j++) {
                let c = 0;
                for (let i = 0; i < dx; i++) c += v[i] * bc[(a * d + j) * dx + i];
                pen += lam[j] * Math.max(0, c - w);
              }
              idx[a] = Math.min(1, r + w) - pen / beta;
            }
            const a = util.argmax(idx, rng);
            return idx[a] >= 0 ? a : -1;
          },
          update(a, r, cv) {
            if (a >= 0) {
              const o = a * dx * dx;
              let q = 0;
              for (let i = 0; i < dx; i++) { let z = 0; for (let k = 0; k < dx; k++) z += Ainv[o + i * dx + k] * xs[k]; v[i] = z; q += z * xs[i]; }
              for (let i = 0; i < dx; i++) for (let k = 0; k < dx; k++) Ainv[o + i * dx + k] -= (v[i] * v[k]) / (1 + q);
              for (let i = 0; i < dx; i++) {
                br[a * dx + i] += r * xs[i];
                for (let j = 0; j < d; j++) bc[(a * d + j) * dx + i] += cv[j] * xs[i];
              }
            }
            dual.update(a >= 0 ? cv : null);
          },
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
    const K = config.K | 0, T = config.T | 0, runs = config.runs | 0, seed = config.seed >>> 0;
    if (K < 2) throw new Error('K must be at least 2');
    const ex = config.extra || {};
    const beta = clamp(+(ex.beta != null ? ex.beta : 0.25), 0.01, 1);
    const B = beta * T;
    const DMAX = 2, DXMAX = 3;
    const mr = new Float64Array(T * K), mc = new Float64Array(T * K * DMAX), xs = new Float64Array(T * DXMAX);
    const ur = new Float64Array(T * K), uc = new Float64Array(T * K * DMAX);
    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaults(def), spec.params || {}),
        reg: new Float64Array(T), reg2: new Float64Array(T), rew: new Float64Array(T), bud: new Float64Array(T),
        finals: [], totRew: 0, totRatio: 0, totLeft: 0, totStop: 0, timeMs: 0,
      };
    });
    let done = 0;

    function runOne(run) {
      const envRng = Rng(hash32(seed, run, 'env'));
      const inst = env.generate(K, T, beta, envRng, mr, mc, xs);
      const { d, dx, opt } = inst;
      for (let i = 0; i < T * K; i++) ur[i] = envRng.uniform();
      for (let i = 0; i < T * K * d; i++) uc[i] = envRng.uniform();
      const x = new Float64Array(dx), rem = new Float64Array(d), cv = new Float64Array(d);
      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const agent = slot.def.create(K, { T, d, dx, B, beta, params: slot.params, rng, env });
        rem.fill(B);
        let got = 0, stopped = false, stopT = T;
        const t0 = util.now();
        for (let t = 0; t < T; t++) {
          if (!stopped) {
            for (let j = 0; j < d; j++) if (rem[j] < 1) stopped = true;
            if (stopped) stopT = t;
          }
          if (!stopped) {
            for (let k = 0; k < dx; k++) x[k] = xs[t * dx + k];
            let a = agent.select(t + 1, x, rem, rng);
            if (!(a >= 0 && a < K)) a = -1;
            if (a >= 0) {
              const i = t * K + a;
              const r = ur[i] < mr[i] ? 1 : 0;
              for (let j = 0; j < d; j++) { cv[j] = uc[i * d + j] < mc[i * d + j] ? 1 : 0; rem[j] -= cv[j]; }
              got += mr[i];
              agent.update(a, r, cv, t + 1);
            } else {
              agent.update(-1, 0, null, t + 1);
            }
          }
          const reg = opt * (t + 1) - got;
          slot.reg[t] += reg; slot.reg2[t] += reg * reg; slot.rew[t] += got;
          let left = 1;
          for (let j = 0; j < d; j++) left = Math.min(left, rem[j] / B);
          slot.bud[t] += left;
        }
        slot.timeMs += util.now() - t0;
        const final = opt * T - got;
        slot.finals.push(final);
        slot.totRew += got;
        slot.totRatio += opt > 0 ? got / (opt * T) : 1;
        let left = 1;
        for (let j = 0; j < d; j++) left = Math.min(left, rem[j] / B);
        slot.totLeft += left;
        slot.totStop += stopT;
      }
    }

    return {
      config: { env: env.id, K, T, runs, seed, extra: { beta } },
      get done() { return done; },
      get finished() { return done >= runs; },
      step(n = 1) {
        const stop = Math.min(runs, done + n);
        while (done < stop) runOne(done++);
        return done;
      },
      results() {
        const R = Math.max(1, done);
        return slots.map((s) => {
          const regret = new Float64Array(T), regretSE = new Float64Array(T), reward = new Float64Array(T), budget = new Float64Array(T);
          for (let t = 0; t < T; t++) {
            const m = s.reg[t] / R;
            regret[t] = m;
            regretSE[t] = R > 1 ? Math.sqrt(Math.max(0, s.reg2[t] / R - m * m) / (R - 1)) : 0;
            reward[t] = s.rew[t] / R;
            budget[t] = done ? s.bud[t] / R : 1;
          }
          const sorted = s.finals.slice().sort((a, b) => a - b);
          const summary = {
            regret: regret[T - 1], regretSE: regretSE[T - 1],
            regretMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            reward: s.totRew / R, ratio: s.totRatio / R, left: s.totLeft / R,
            stopT: done ? s.totStop / R : T, msPerRun: done ? s.timeMs / done : 0,
          };
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, params: s.params, runs: done,
            curves: { regret, reward, budget }, curvesSE: { regret: regretSE },
            summary, finals: sorted,
          };
        });
      },
    };
  }

  return MAB.registerSetting({
    id: 'knapsack',
    name: 'Bandits with knapsacks',
    short: 'Knapsacks',
    blurb: 'Every pull earns a reward and spends limited resources; the run ends when any budget is exhausted. Compared against the LP-relaxation optimum.',
    families, environments, algorithms,
    defaultEnv: 'one',
    defaultLineup: ['ucb-bwk', 'pd-bwk', 'lagrange-bwk', 'cbwk', 'ucb-ratio', 'ucb1'],
    presets: [
      { label: 'Cost-aware vs cost-blind', algos: ['ucb-bwk', 'pd-bwk', 'ucb-ratio', 'ucb1'], env: 'one' },
      { label: 'Two resources', algos: ['ucb-bwk', 'pd-bwk', 'lagrange-bwk', 'ucb-ratio'], env: 'two' },
      { label: 'Shifting: Lagrangian game', algos: ['lagrange-bwk', 'ucb-bwk', 'pd-bwk', 'ucb1'], env: 'shift' },
      { label: 'Contextual budgets', algos: ['cbwk', 'ucb-bwk', 'lagrange-bwk', 'ucb1'], env: 'ctx' },
    ],
    controls: {
      K: { label: 'Arms (K)', min: 2, max: 20, default: 5 },
      T: { label: 'Horizon (T rounds)', options: [250, 500, 1000, 2000, 5000], default: 1000 },
      runs: { label: 'Runs averaged', options: [20, 50, 100, 200], default: 50 },
      extra: [{ key: 'beta', label: 'Budget per round β (B = β·T)', min: 0.05, max: 1, step: 0.05, default: 0.25 }],
    },
    xLabel: 'round t',
    metrics: [
      { key: 'regret', label: 'Regret vs LP optimum', better: 'lower', fmt: 'num', band: true, note: 'OPT_LP·t/T minus the expected reward collected so far. OPT_LP solves the BwK linear program on the true means; after the budget runs out, regret grows at the LP rate.' },
      { key: 'reward', label: 'Cumulative reward', better: 'higher', fmt: 'num', note: 'Expected reward collected so far (sum of the pulled arms\' means).' },
      { key: 'budget', label: 'Budget left', better: 'higher', fmt: 'pct', zeroToOne: true, note: 'Remaining share of the scarcest resource. Ending with budget left is wasted reward; running dry early is too.' },
    ],
    final: { key: 'regret', se: 'regretSE', median: 'regretMedian', label: 'Regret vs OPT_LP', better: 'lower' },
    tiles: [
      { key: 'ratio', label: 'Closest to the LP optimum', better: 'higher', fmt: 'pct', sub: 'of OPT_LP collected' },
      { key: 'stopT', label: 'Lasts longest', better: 'higher', fmt: 'int', sub: 'rounds before a budget ran out' },
    ],
    columns: [
      { key: 'regret', label: 'Regret', fmt: 'num', better: 'lower' },
      { key: 'regretSE', label: '95% CI', fmt: 'ci' },
      { key: 'regretMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'reward', label: 'Reward', fmt: 'num', better: 'higher' },
      { key: 'ratio', label: '% of OPT_LP', fmt: 'pct', better: 'higher' },
      { key: 'left', label: 'Budget left', fmt: 'pct' },
      { key: 'stopT', label: 'Stop round', fmt: 'int' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: false,
    createRunner,
    _internal: { makeLP },
  });
});
