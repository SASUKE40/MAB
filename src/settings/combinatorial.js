/*
 * combinatorial.js — Combinatorial semi-bandits, click models and online ranking.
 *
 * Each round the learner picks an ordered list of m distinct items out of L ground items
 * (or, on the matroid environment, one item from each of m blocks) and receives the feedback
 * of the environment: semi-bandit (every chosen item's weight), cascading (clicks up to the
 * first click), DCM or PBM click vectors, or only the total reward (full bandit).
 *
 * Interface shared by every algorithm in this setting:
 *   create(L, ctx)   ctx = { T, L, m, env, info, space, params, rng, oracle }
 *     select(t)            -> list of m item indices (Int32Array or array), t = 1..T
 *     update(list, fb, t)  fb = { obs, clicks, reward, full, proxy }
 *   fb.obs[k]    item-level observation of the item at position k, in [0, 1]; NaN = unobserved
 *                (cascade env: items after the first click; DCM env: items after the last click)
 *   fb.clicks[k] raw click vector on click-model environments, otherwise null
 *   fb.reward    the scalar reward the learner sees (DCM: 1 if any click, satisfaction is hidden)
 *   fb.full      true on full-bandit environments: obs is all NaN; fb.proxy = reward / m is the
 *                per-item credit that item-level learners use instead (stated in their docs).
 *   ctx.oracle(scores) returns the best feasible action for a linear objective with these item
 *                scores, sorted by decreasing score (top-m, or best item per block on the matroid).
 *   ctx.space = { type: 'topm' | 'partition', groups: [Int32Array], caps: [k], grp: Int32Array(L) }
 *   ctx.info  = { kind, rMax, clicks, full, kappa?, features?, d? } (read-only, shared)
 *
 * Fairness: every run pre-generates the item outcome table (T x L) and the position table
 * (T x m: examination or satisfaction coins) from hash32(seed, run, 'env'); all algorithms read
 * the same tables (common random numbers). Each algorithm's RNG is hash32(seed, run, id).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;
  const { logPlus, now } = util;

  // ------------------------------------------------------------ helpers ----
  function makeSpace(type, L, m, rng) {
    const grp = new Int32Array(L);
    if (type === 'topm') {
      return { type, groups: [Int32Array.from({ length: L }, (_, i) => i)], caps: [m], grp };
    }
    // partition matroid: shuffle items, deal them round-robin into m blocks
    const perm = Int32Array.from({ length: L }, (_, i) => i);
    for (let i = L - 1; i > 0; i--) { const j = rng.int(i + 1); const x = perm[i]; perm[i] = perm[j]; perm[j] = x; }
    const lists = Array.from({ length: m }, () => []);
    for (let r = 0; r < L; r++) { const b = r % m; lists[b].push(perm[r]); grp[perm[r]] = b; }
    for (const l of lists) l.sort((a, b) => a - b);
    return { type, groups: lists.map((l) => Int32Array.from(l)), caps: lists.map(() => 1), grp };
  }

  /** Linear maximisation oracle over the action space (random tie-break), output sorted by score. */
  function makeOracle(space, L, m, rng) {
    const out = new Int32Array(m);
    if (space.type === 'partition') {
      const grp = space.grp, bi = new Int32Array(m), bv = new Float64Array(m), ties = new Int32Array(m);
      return function oracle(sc) {
        bi.fill(-1); ties.fill(0);
        for (let i = 0; i < L; i++) {
          const b = grp[i], v = sc[i];
          if (bi[b] < 0 || v > bv[b]) { bi[b] = i; bv[b] = v; ties[b] = 1; }
          else if (v === bv[b]) { ties[b]++; if (rng.uniform() * ties[b] < 1) bi[b] = i; }
        }
        for (let k = 0; k < m; k++) {
          const x = bi[k], v = sc[x];
          let j = k - 1;
          while (j >= 0 && sc[out[j]] < v) { out[j + 1] = out[j]; j--; }
          out[j + 1] = x;
        }
        return out;
      };
    }
    const used = new Uint8Array(L);
    return function oracle(sc) {
      used.fill(0);
      for (let k = 0; k < m; k++) {
        let idx = -1, best = -Infinity, ties = 0;
        for (let i = 0; i < L; i++) {
          if (used[i]) continue;
          const v = sc[i];
          if (idx < 0 || v > best) { idx = i; best = v; ties = 1; }
          else if (v === best) { ties++; if (rng.uniform() * ties < 1) idx = i; }
        }
        out[k] = idx; used[idx] = 1;
      }
      return out;
    };
  }

  /** Item-level observations as given by the environment (full bandit: reward / m per item). */
  function semiView(fb, m, out) {
    if (fb.full) out.fill(fb.proxy);
    else for (let k = 0; k < m; k++) out[k] = fb.obs[k];
    return out;
  }
  /** Cascade reading of a click vector: items above the first click were examined and skipped. */
  function cascadeView(fb, m, out) {
    if (!fb.clicks) return semiView(fb, m, out);
    let c = -1;
    for (let k = 0; k < m; k++) if (fb.clicks[k] > 0) { c = k; break; }
    for (let k = 0; k < m; k++) out[k] = c < 0 || k < c ? 0 : k === c ? 1 : NaN;
    return out;
  }
  /** DCM reading of a click vector: everything up to the last click was examined. */
  function dcmView(fb, m, out) {
    if (!fb.clicks) return semiView(fb, m, out);
    let c = -1;
    for (let k = 0; k < m; k++) if (fb.clicks[k] > 0) c = k;
    for (let k = 0; k < m; k++) out[k] = c < 0 ? 0 : k <= c ? (fb.clicks[k] > 0 ? 1 : 0) : NaN;
    return out;
  }
  /** Per-position click (or weight) signal: raw clicks if any, else the item-level view (NaN -> 0). */
  function clickSignal(fb, m, out) {
    if (fb.clicks) { for (let k = 0; k < m; k++) out[k] = fb.clicks[k]; return out; }
    semiView(fb, m, out);
    for (let k = 0; k < m; k++) if (out[k] !== out[k]) out[k] = 0;
    return out;
  }

  /**
   * Dependent rounding (Gandhi et al. 2006): sample a k-subset of idx whose inclusion
   * probabilities are exactly p[idx[j]] (which must sum to k). Writes chosen items into `out`
   * from position `at`; returns the next free position.
   */
  function depRound(p, idx, k, rng, work, out, at) {
    const n = idx.length, eps = 1e-12;
    for (let j = 0; j < n; j++) work[j] = Math.min(1, Math.max(0, p[idx[j]]));
    let a = -1;
    for (let b = 0; b < n; b++) {
      if (work[b] <= eps || work[b] >= 1 - eps) continue;
      if (a < 0) { a = b; continue; }
      const al = Math.min(1 - work[a], work[b]), be = Math.min(work[a], 1 - work[b]);
      if (rng.uniform() * (al + be) < be) { work[a] += al; work[b] -= al; } else { work[a] -= be; work[b] += be; }
      if (work[a] <= eps || work[a] >= 1 - eps) a = work[b] > eps && work[b] < 1 - eps ? b : -1;
    }
    // collect, then repair rounding drift so exactly k items are returned
    let cnt = 0;
    for (let j = 0; j < n; j++) if (work[j] > 0.5) { work[j] = 2; cnt++; }
    while (cnt < k) {
      let bj = -1;
      for (let j = 0; j < n; j++) if (work[j] !== 2 && (bj < 0 || p[idx[j]] > p[idx[bj]])) bj = j;
      work[bj] = 2; cnt++;
    }
    while (cnt > k) {
      let bj = -1;
      for (let j = 0; j < n; j++) if (work[j] === 2 && (bj < 0 || p[idx[j]] < p[idx[bj]])) bj = j;
      work[bj] = 0; cnt--;
    }
    for (let j = 0; j < n; j++) if (work[j] === 2) out[at++] = idx[j];
    return at;
  }

  /** Sample an action with marginals x (per group sums to the group capacity), sorted by x. */
  function sampleMarginals(space, x, rng, work, out) {
    let at = 0;
    for (let g = 0; g < space.groups.length; g++) at = depRound(x, space.groups[g], space.caps[g], rng, work, out, at);
    for (let k = 1; k < at; k++) {
      const v = out[k]; let j = k - 1;
      while (j >= 0 && x[out[j]] < x[v]) { out[j + 1] = out[j]; j--; }
      out[j + 1] = v;
    }
    return out;
  }

  function sortByScore(list, sc) {
    for (let k = 1; k < list.length; k++) {
      const v = list[k]; let j = k - 1;
      while (j >= 0 && sc[list[j]] < sc[v]) { list[j + 1] = list[j]; j--; }
      list[j + 1] = v;
    }
    return list;
  }

  function lnChoose(n, k) {
    let s = 0;
    for (let i = 0; i < k; i++) s += Math.log((n - i) / (i + 1));
    return s;
  }

  /** In-place Cholesky solve of the SPD system A x = b (A is overwritten). */
  function cholSolve(A, b, x, n) {
    for (let j = 0; j < n; j++) {
      let s = A[j * n + j];
      for (let k = 0; k < j; k++) s -= A[j * n + k] * A[j * n + k];
      const d = Math.sqrt(Math.max(s, 1e-300));
      A[j * n + j] = d;
      for (let i = j + 1; i < n; i++) {
        let t = A[i * n + j];
        for (let k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
        A[i * n + j] = t / d;
      }
    }
    for (let i = 0; i < n; i++) {
      let t = b[i];
      for (let k = 0; k < i; k++) t -= A[i * n + k] * x[k];
      x[i] = t / A[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let t = x[i];
      for (let k = i + 1; k < n; k++) t -= A[k * n + i] * x[k];
      x[i] = t / A[i * n + i];
    }
    return x;
  }

  /**
   * Bernoulli KL-UCB: largest q with n·kl(p, q) ≤ bound. Newton's method in u = −ln(1 − q), where
   * g(u) = kl(p, q(u)) is convex and increasing for q > p (and nearly linear near q = 1). Started
   * above the root at the Pinsker bound p + √(bound / 2n), the iterates decrease monotonically.
   */
  function klUcb(p, n, bound) {
    if (n <= 0) return 1;
    if (bound <= 0) return p;
    p = Math.min(Math.max(p, 0), 1);
    const d = bound / n;
    if (p >= 1) return 1;
    const lp = p > 0 ? Math.log(p) : 0, l1p = Math.log(1 - p);
    let q = Math.min(1 - 1e-15, p + Math.sqrt(d / 2)), u = -Math.log(1 - q);
    for (let it = 0; it < 8; it++) {
      const f = (p > 0 ? p * (lp - Math.log(q)) : 0) + (1 - p) * (l1p + u) - d;
      if (f <= 1e-9) break;
      u -= (f * q) / (q - p);
      q = -Math.expm1(-u);
      if (q <= p) return p;
    }
    return q;
  }

  /** Bernoulli trick for fractional observations (Agrawal & Goyal 2012). */
  function binarize(v, rng) { return v <= 0 ? 0 : v >= 1 ? 1 : rng.uniform() < v ? 1 : 0; }

  // ------------------------------------------------------- environments ----
  function bitTable(T, L, probAt, rng) {
    const Y = new Uint8Array(T * L);
    for (let t = 0; t < T; t++) {
      const p = probAt(t);
      for (let i = 0; i < L; i++) Y[t * L + i] = rng.uniform() < p[i] ? 1 : 0;
    }
    return Y;
  }
  function uniformVec(L, lo, hi, rng) {
    const w = new Float64Array(L);
    for (let i = 0; i < L; i++) w[i] = lo + (hi - lo) * rng.uniform();
    return w;
  }
  /** Best list for weights w: top-m (or best per block), sorted by decreasing weight. */
  function bestList(space, w, L, m) {
    const o = makeOracle(space, L, m, Rng(1));
    return Int32Array.from(o(w));
  }
  function constBest(T, v) { return new Float64Array(T).fill(v); }

  /** Shared pieces of the stationary environments. */
  function instance(space, info, w, T, L, m, expected, feedback) {
    const best = bestList(space, w, L, m);
    return { space, info, w, bestList: best, best: constBest(T, expected(best, 0)), expected, feedback };
  }

  function linearEnv(type) {
    return function make(L, m, T, rng) {
      const space = makeSpace(type, L, m, rng);
      const w = uniformVec(L, 0.1, 0.8, rng);
      const Y = bitTable(T, L, () => w, rng);
      const info = { kind: 'semi', rMax: m, clicks: false, full: false };
      return instance(space, info, w, T, L, m,
        (list) => { let s = 0; for (let k = 0; k < m; k++) s += w[list[k]]; return s; },
        (list, t, fb) => {
          let r = 0;
          for (let k = 0; k < m; k++) { const y = Y[t * L + list[k]]; fb.obs[k] = y; r += y; }
          fb.reward = r;
        });
    };
  }

  function cascadeMake(w, L, m, T, rng, extraInfo) {
    const space = makeSpace('topm', L, m, rng);
    const Y = bitTable(T, L, () => w, rng);
    const info = Object.assign({ kind: 'cascade', rMax: 1, clicks: true, full: false }, extraInfo || {});
    return instance(space, info, w, T, L, m,
      (list) => { let q = 1; for (let k = 0; k < m; k++) q *= 1 - w[list[k]]; return 1 - q; },
      (list, t, fb) => {
        let c = -1;
        for (let k = 0; k < m; k++) if (Y[t * L + list[k]]) { c = k; break; }
        for (let k = 0; k < m; k++) {
          fb.clicks[k] = k === c ? 1 : 0;
          fb.obs[k] = c < 0 || k < c ? 0 : k === c ? 1 : NaN;
        }
        fb.reward = c >= 0 ? 1 : 0;
      });
  }

  const DCM_V = (k) => 0.9 * Math.pow(0.8, k);
  const PBM_KAPPA = (k) => 1 / (1 + 0.5 * k);

  const ENVIRONMENTS = [
    {
      id: 'topm', name: 'Top-m semi-bandit', short: 'Top-m',
      description: 'L items with Bernoulli weights wᵢ ~ U(0.1, 0.8). Choose any m items; reward is the sum of their realised weights and every chosen item\'s weight is observed (semi-bandit). Best action: the m heaviest items.',
      tags: ['semi-bandit', 'linear'],
      make: linearEnv('topm'),
    },
    {
      id: 'matroid', name: 'Partition matroid', short: 'Matroid',
      description: 'Items are dealt at random into m blocks; an action is a basis of the partition matroid (exactly one item per block). Linear Bernoulli rewards with semi-bandit feedback, wᵢ ~ U(0.1, 0.8). Best action: the heaviest item of each block. Top-m algorithms use the matroid oracle; adversarial ones factorise per block.',
      tags: ['semi-bandit', 'matroid'],
      make: linearEnv('partition'),
    },
    {
      id: 'cascade', name: 'Cascading clicks', short: 'Cascade',
      description: 'Cascade model: the user scans the list top-down and clicks the first attractive item (attraction wᵢ ~ U(0.05, 0.35)), then leaves. Reward 1 if any click, so E[r] = 1 − Π(1 − wₐₖ). Items below the click are unobserved.',
      tags: ['click model', 'cascade'],
      make(L, m, T, rng) { return cascadeMake(uniformVec(L, 0.05, 0.35, rng), L, m, T, rng); },
    },
    {
      id: 'dcm', name: 'Dependent click model', short: 'DCM',
      description: 'DCM (Guo et al. 2009): the user clicks every attractive item (wᵢ ~ U(0.05, 0.4)) while scanning; after a click at position k she is satisfied and stops with probability vₖ = 0.9·0.8ᵏ. Reward = satisfaction (hidden; learners see clicks and "any click"), E[r] = 1 − Π(1 − vₖ wₐₖ).',
      tags: ['click model', 'multiple clicks'],
      make(L, m, T, rng) {
        const space = makeSpace('topm', L, m, rng);
        const w = uniformVec(L, 0.05, 0.4, rng);
        const v = Float64Array.from({ length: m }, (_, k) => DCM_V(k));
        const Y = bitTable(T, L, () => w, rng), S = bitTable(T, m, () => v, rng);
        const info = { kind: 'dcm', rMax: 1, clicks: true, full: false };
        return instance(space, info, w, T, L, m,
          (list) => { let q = 1; for (let k = 0; k < m; k++) q *= 1 - v[k] * w[list[k]]; return 1 - q; },
          (list, t, fb) => {
            let stop = false, last = -1;
            for (let k = 0; k < m; k++) {
              const c = !stop && Y[t * L + list[k]] ? 1 : 0;
              fb.clicks[k] = c;
              if (c) { last = k; if (S[t * m + k]) stop = true; }
            }
            for (let k = 0; k < m; k++) fb.obs[k] = last < 0 ? 0 : k <= last ? fb.clicks[k] : NaN;
            fb.reward = last >= 0 ? 1 : 0;
          });
      },
    },
    {
      id: 'pbm', name: 'Position-based model', short: 'PBM',
      description: 'PBM: position k is examined with probability κₖ = 1/(1 + k/2) (known to PBM learners), item i is attractive with probability wᵢ ~ U(0.05, 0.7), and a click needs both. Reward = number of clicks, E[r] = Σ κₖ wₐₖ; the best list puts the most attractive items on top.',
      tags: ['click model', 'position bias'],
      make(L, m, T, rng) {
        const space = makeSpace('topm', L, m, rng);
        const w = uniformVec(L, 0.05, 0.7, rng);
        const kappa = Float64Array.from({ length: m }, (_, k) => PBM_KAPPA(k));
        const Y = bitTable(T, L, () => w, rng), E = bitTable(T, m, () => kappa, rng);
        const info = { kind: 'pbm', rMax: m, clicks: true, full: false, kappa };
        return instance(space, info, w, T, L, m,
          (list) => { let s = 0; for (let k = 0; k < m; k++) s += kappa[k] * w[list[k]]; return s; },
          (list, t, fb) => {
            let r = 0;
            for (let k = 0; k < m; k++) {
              const c = Y[t * L + list[k]] & E[t * m + k];
              fb.clicks[k] = c; fb.obs[k] = c; r += c;
            }
            fb.reward = r;
          });
      },
    },
    {
      id: 'cascade-lin', name: 'Cascade with item features', short: 'Linear cascade',
      description: 'Cascade clicks where attraction is linear in known item features: xᵢ on the 5-dimensional simplex, wᵢ = θᵀxᵢ with θⱼ ~ U(0, 0.5). CascadeLinUCB/LinTS read the features; the rest treat items as unrelated.',
      tags: ['click model', 'cascade', 'features'],
      make(L, m, T, rng) {
        const d = 5, X = new Float64Array(L * d), theta = uniformVec(d, 0, 0.5, rng), w = new Float64Array(L);
        for (let i = 0; i < L; i++) {
          let s = 0;
          for (let j = 0; j < d; j++) { const e = -Math.log(1 - rng.uniform()); X[i * d + j] = e; s += e; }
          for (let j = 0; j < d; j++) { X[i * d + j] /= s; w[i] += theta[j] * X[i * d + j]; }
        }
        return cascadeMake(w, L, m, T, rng, { kind: 'cascade-lin', features: X, d });
      },
    },
    {
      id: 'adversarial', name: 'Adversarial full-bandit', short: 'Adversarial',
      description: 'Full-bandit linear rewards: only the total Σ yₜᵢ of the chosen m items is observed (no per-item feedback). Means switch at T/3: a decoy set is best early (0.85 → 0.15) while the target set starts low (0.35 → 0.75); others ~ U(0.1, 0.5). Regret is against the best fixed set in hindsight and can dip below zero early.',
      tags: ['adversarial', 'full bandit', 'changes'],
      make(L, m, T, rng) {
        const space = makeSpace('topm', L, m, rng);
        const base = uniformVec(L, 0.1, 0.5, rng);
        const perm = Int32Array.from({ length: L }, (_, i) => i);
        for (let i = L - 1; i > 0; i--) { const j = rng.int(i + 1); const x = perm[i]; perm[i] = perm[j]; perm[j] = x; }
        const h = Math.min(m, Math.floor(L / 2));
        const mu1 = Float64Array.from(base), mu2 = Float64Array.from(base);
        for (let r = 0; r < h; r++) { mu1[perm[r]] = 0.85; mu2[perm[r]] = 0.15; mu1[perm[h + r]] = 0.35; mu2[perm[h + r]] = 0.75; }
        const T1 = Math.floor(T / 3), avg = new Float64Array(L);
        for (let i = 0; i < L; i++) avg[i] = (T1 * mu1[i] + (T - T1) * mu2[i]) / T;
        const Y = bitTable(T, L, (t) => (t < T1 ? mu1 : mu2), rng);
        const expected = (list, t) => { const mu = t < T1 ? mu1 : mu2; let s = 0; for (let k = 0; k < m; k++) s += mu[list[k]]; return s; };
        const bl = bestList(space, avg, L, m), best = new Float64Array(T);
        for (let t = 0; t < T; t++) best[t] = expected(bl, t);
        const info = { kind: 'full', rMax: m, clicks: false, full: true };
        return {
          space, info, w: avg, bestList: bl, best, expected,
          feedback(list, t, fb) {
            let r = 0;
            for (let k = 0; k < m; k++) { r += Y[t * L + list[k]]; fb.obs[k] = NaN; }
            fb.reward = r; fb.proxy = r / m;
          },
        };
      },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of ENVIRONMENTS) ENV_BY_ID[e.id] = e;

  // ------------------------------------------------------- algorithms ----
  const FAMILIES = [
    { id: 'baselines', name: 'Baselines', blurb: 'Reference points: no learning, or learning without exploration.' },
    { id: 'semi-bandit', name: 'Stochastic semi-bandit', blurb: 'Per-item statistics plus a combinatorial oracle; optimism or posterior sampling.' },
    { id: 'adversarial', name: 'Adversarial', blurb: 'Exponential weights or mirror descent over the action polytope; no stochastic assumption.' },
    { id: 'cascading', name: 'Cascading', blurb: 'Learn attraction probabilities from cascade feedback: skipped items above the click count as failures.' },
    { id: 'click-models', name: 'Click models', blurb: 'Use the structure of a specific click model (DCM, PBM) to read clicks correctly.' },
    { id: 'ranking', name: 'Ranking', blurb: 'Model-light online learning to rank: per-slot bandits or pairwise comparisons.' },
  ];

  /** Optimistic/greedy index learner over per-item statistics. */
  function indexLearner(L, ctx, view, index, prep) {
    const { m, oracle } = ctx;
    const n = new Float64Array(L), s = new Float64Array(L), sc = new Float64Array(L), v = new Float64Array(m);
    return {
      select(t) {
        const b = prep ? prep(t) : t;
        for (let i = 0; i < L; i++) sc[i] = n[i] === 0 ? Infinity : index(s[i] / n[i], n[i], b);
        return oracle(sc);
      },
      update(list, fb) {
        view(fb, m, v);
        for (let k = 0; k < m; k++) if (v[k] === v[k]) { n[list[k]] += 1; s[list[k]] += v[k]; }
      },
    };
  }
  const klBound = (t) => Math.log(t) + 3 * logPlus(Math.log(t));

  /** Ridge-regression attraction model on item features (falls back to one-hot = per-item). */
  function linModel(L, info, lambda) {
    if (info.features) {
      const d = info.d, X = info.features;
      const Ainv = new Float64Array(d * d), b = new Float64Array(d), th = new Float64Array(d), Ax = new Float64Array(d);
      for (let i = 0; i < d; i++) Ainv[i * d + i] = 1 / lambda;
      return {
        d, X, Ainv, th,
        add(i, r) {
          const o = i * d; let den = 1;
          for (let p = 0; p < d; p++) { let v = 0; for (let q = 0; q < d; q++) v += Ainv[p * d + q] * X[o + q]; Ax[p] = v; den += X[o + p] * v; }
          for (let p = 0; p < d; p++) for (let q = 0; q < d; q++) Ainv[p * d + q] -= (Ax[p] * Ax[q]) / den;
          for (let p = 0; p < d; p++) b[p] += r * X[o + p];
        },
        refresh() { for (let p = 0; p < d; p++) { let v = 0; for (let q = 0; q < d; q++) v += Ainv[p * d + q] * b[q]; th[p] = v; } },
        mean(i, theta) { let v = 0; for (let p = 0; p < d; p++) v += (theta || th)[p] * X[i * d + p]; return v; },
        width(i) {
          const o = i * d; let v = 0;
          for (let p = 0; p < d; p++) { let u = 0; for (let q = 0; q < d; q++) u += Ainv[p * d + q] * X[o + q]; v += X[o + p] * u; }
          return Math.sqrt(Math.max(0, v));
        },
      };
    }
    const n = new Float64Array(L), s = new Float64Array(L);
    return {
      d: 0,
      add(i, r) { n[i] += 1; s[i] += r; },
      refresh() {},
      mean(i) { return s[i] / (lambda + n[i]); },
      width(i) { return 1 / Math.sqrt(lambda + n[i]); },
    };
  }

  /** Cholesky factor (lower) of a small SPD matrix, for multivariate normal sampling. */
  function cholLower(A, d, out) {
    out.fill(0);
    for (let j = 0; j < d; j++) {
      let s = A[j * d + j];
      for (let k = 0; k < j; k++) s -= out[j * d + k] * out[j * d + k];
      const dj = Math.sqrt(Math.max(s, 1e-12));
      out[j * d + j] = dj;
      for (let i = j + 1; i < d; i++) {
        let t = A[i * d + j];
        for (let k = 0; k < j; k++) t -= out[i * d + k] * out[j * d + k];
        out[i * d + j] = t / dj;
      }
    }
    return out;
  }

  /** EXP3.M marginals (Uchiya et al. 2010) for one group: capped exponential weights + γ mixing. */
  function exp3mProbs(lw, idx, k, gamma, p, capped, wbuf, order) {
    const n = idx.length;
    if (k >= n) { for (let j = 0; j < n; j++) { p[idx[j]] = 1; capped[idx[j]] = 1; } return; }
    let mx = -Infinity;
    for (let j = 0; j < n; j++) if (lw[idx[j]] > mx) mx = lw[idx[j]];
    let sum = 0;
    for (let j = 0; j < n; j++) { wbuf[j] = Math.exp(lw[idx[j]] - mx); sum += wbuf[j]; capped[idx[j]] = 0; order[j] = j; }
    const thr = (1 / k - gamma / n) / (1 - gamma);
    let alpha = Infinity, ncap = 0;
    if (gamma < 1 && k > 1) {
      // sort descending (small n: insertion sort of indices)
      for (let a = 1; a < n; a++) { const v = order[a]; let b = a - 1; while (b >= 0 && wbuf[order[b]] < wbuf[v]) { order[b + 1] = order[b]; b--; } order[b + 1] = v; }
      if (wbuf[order[0]] >= thr * sum) {
        let rest = sum;
        for (let c = 1; c < k; c++) {
          rest -= wbuf[order[c - 1]];
          const a = (thr * rest) / (1 - c * thr);
          if (wbuf[order[c]] <= a) { alpha = a; ncap = c; break; }
        }
      }
    }
    let s2 = 0;
    for (let j = 0; j < n; j++) { const v = Math.min(wbuf[j], alpha); s2 += v; }
    for (let c = 0; c < ncap; c++) capped[idx[order[c]]] = 1;
    for (let j = 0; j < n; j++) {
      const v = Math.min(wbuf[j], alpha);
      p[idx[j]] = Math.min(1, k * ((1 - gamma) * v / s2 + gamma / n));
    }
  }

  /** KL projection of positive weights y onto {x ≤ 1, Σx = k} over one group: x = min(1, c·y). */
  function capSimplexProject(ly, idx, k, x, wbuf, order) {
    const n = idx.length;
    let mx = -Infinity;
    for (let j = 0; j < n; j++) if (ly[idx[j]] > mx) mx = ly[idx[j]];
    for (let j = 0; j < n; j++) { wbuf[j] = Math.exp(ly[idx[j]] - mx); order[j] = j; }
    for (let a = 1; a < n; a++) { const v = order[a]; let b = a - 1; while (b >= 0 && wbuf[order[b]] < wbuf[v]) { order[b + 1] = order[b]; b--; } order[b + 1] = v; }
    let tail = 0;
    for (let j = 0; j < n; j++) tail += wbuf[j];
    let c = k / tail, ncap = 0;
    while (ncap < k && c * wbuf[order[ncap]] > 1) { tail -= wbuf[order[ncap]]; ncap++; c = (k - ncap) / tail; }
    for (let j = 0; j < n; j++) x[idx[j]] = Math.max(1e-12, Math.min(1, c * wbuf[j]));
    for (let q = 0; q < ncap; q++) x[idx[order[q]]] = 1;
  }

  const ALGORITHMS = [
    // ---------------------------------------------------------- baselines
    {
      id: 'random', name: 'Random', family: 'baselines', params: [],
      formula: 'Aₜ ~ Uniform(feasible actions)',
      regret: 'Θ(T)',
      idea: 'Ignores feedback and plays a uniformly random feasible list. Its regret is the price of pure exploration.',
      create(L, { rng, oracle }) {
        const sc = new Float64Array(L);
        return { select() { for (let i = 0; i < L; i++) sc[i] = rng.uniform(); return oracle(sc); }, update() {} };
      },
    },
    {
      id: 'greedy', name: 'Greedy', family: 'baselines',
      params: [{ key: 'eps', label: 'ε (random list)', default: 0, min: 0, max: 1, step: 0.01 }],
      formula: 'Aₜ = Oracle(ŵ)   (unseen items first; random list w.p. ε)',
      regret: 'Θ(T) in the worst case',
      idea: 'Tries every item once, then always plays the oracle on empirical means. Locks onto lists that looked good early.',
      create(L, ctx) {
        const { rng, params } = ctx, base = indexLearner(L, ctx, semiView, (mu) => mu), sc = new Float64Array(L);
        return {
          select(t) {
            if (params.eps > 0 && rng.uniform() < params.eps) { for (let i = 0; i < L; i++) sc[i] = rng.uniform(); return ctx.oracle(sc); }
            return base.select(t);
          },
          update: base.update,
        };
      },
    },
    // -------------------------------------------------------- semi-bandit
    {
      id: 'cucb', name: 'CUCB', family: 'semi-bandit',
      params: [{ key: 'c', label: 'c', default: 1.5, min: 0, max: 5, step: 0.05 }],
      formula: 'Aₜ = Oracle(μ̄),   μ̄ᵢ = min(1, ŵᵢ + √(c·ln t / nᵢ))',
      regret: 'O(Σᵢ m log T / Δᵢ)  (bounded-smoothness rewards)',
      idea: 'Chen, Wang & Yuan (2013): clipped UCB per item fed to an exact oracle; works for any monotone reward with bounded smoothness. Uses the item observations the environment gives.',
      create(L, ctx) { const c = ctx.params.c; return indexLearner(L, ctx, semiView, (mu, n, b) => Math.min(1, mu + Math.sqrt(b / n)), (t) => c * Math.log(t)); },
    },
    {
      id: 'combucb1', name: 'CombUCB1', family: 'semi-bandit',
      params: [{ key: 'c', label: 'c', default: 1.5, min: 0, max: 5, step: 0.05 }],
      formula: 'Aₜ = argmax_A Σᵢ∈A (ŵᵢ + √(c·log t / nᵢ))',
      regret: 'O(L m log T / Δ),  gap-free O(√(L m T log T))',
      idea: 'Kveton et al. (2015): unclipped UCB on each item\'s weight and a linear maximisation oracle; tight analysis for linear rewards.',
      create(L, ctx) { const c = ctx.params.c; return indexLearner(L, ctx, semiView, (mu, n, b) => mu + Math.sqrt(b / n), (t) => c * Math.log(t)); },
    },
    {
      id: 'cts', name: 'CTS', family: 'semi-bandit', params: [],
      formula: 'θᵢ ~ Beta(1 + Sᵢ, 1 + Fᵢ),   Aₜ = Oracle(θ)',
      regret: 'O(Σᵢ m log T / Δᵢ)',
      idea: 'Wang & Chen (2018) combinatorial Thompson sampling: one posterior draw per item, then the oracle. Fractional observations are binarised.',
      create(L, ctx) {
        const { rng, m, oracle } = ctx, a = new Float64Array(L).fill(1), b = new Float64Array(L).fill(1);
        const sc = new Float64Array(L), v = new Float64Array(m);
        return {
          select() { for (let i = 0; i < L; i++) sc[i] = rng.beta(a[i], b[i]); return oracle(sc); },
          update(list, fb) {
            semiView(fb, m, v);
            for (let k = 0; k < m; k++) if (v[k] === v[k]) { if (binarize(v[k], rng)) a[list[k]]++; else b[list[k]]++; }
          },
        };
      },
    },
    {
      id: 'escb', name: 'ESCB', family: 'semi-bandit',
      params: [{ key: 'c', label: 'c (f(t) = c·ln t)', default: 1, min: 0, max: 5, step: 0.05 }],
      formula: 'Aₜ ≈ argmax_A Σᵢ∈A ŵᵢ + √(f(t)/2 · Σᵢ∈A 1/nᵢ)',
      regret: 'O(√m · L log T / Δ)',
      idea: 'Combes et al. (2015) ESCB-2: one confidence ellipsoid for the whole set instead of per-item bonuses. Exact maximisation is costly, so the index is maximised greedily plus 1-swap local search (approximate; simplified f(t) = c·ln t).',
      create(L, ctx) {
        const { m, oracle, space, params, rng } = ctx, grp = space.grp, caps = space.caps;
        const n = new Float64Array(L), s = new Float64Array(L), sc = new Float64Array(L), v = new Float64Array(m);
        const inA = new Uint8Array(L), left = new Int32Array(caps.length), list = new Int32Array(m);
        return {
          select(t) {
            const f = params.c * Math.log(t);
            let untried = false;
            for (let i = 0; i < L; i++) {
              if (n[i] === 0) { untried = true; sc[i] = Infinity; } else sc[i] = s[i] / n[i] + Math.sqrt(f / (2 * n[i]));
            }
            if (untried) return oracle(sc);
            inA.fill(0);
            for (let g = 0; g < caps.length; g++) left[g] = caps[g];
            let sm = 0, si = 0;
            for (let k = 0; k < m; k++) {
              let bi = -1, bv = -Infinity, ties = 0;
              for (let i = 0; i < L; i++) {
                if (inA[i] || left[grp[i]] === 0) continue;
                const val = sm + s[i] / n[i] + Math.sqrt((f / 2) * (si + 1 / n[i]));
                if (val > bv) { bv = val; bi = i; ties = 1; } else if (val === bv && rng.uniform() * ++ties < 1) bi = i;
              }
              inA[bi] = 1; left[grp[bi]]--; list[k] = bi; sm += s[bi] / n[bi]; si += 1 / n[bi];
            }
            for (let pass = 0; pass < 2; pass++) {
              let improved = false, cur = sm + Math.sqrt((f / 2) * si);
              for (let k = 0; k < m; k++) {
                const a = list[k];
                for (let j = 0; j < L; j++) {
                  if (inA[j] || grp[j] !== grp[a]) continue;
                  const nm = sm - s[a] / n[a] + s[j] / n[j], ni = si - 1 / n[a] + 1 / n[j];
                  const val = nm + Math.sqrt((f / 2) * ni);
                  if (val > cur + 1e-12) { inA[a] = 0; inA[j] = 1; list[k] = j; sm = nm; si = ni; cur = val; improved = true; break; }
                }
              }
              if (!improved) break;
            }
            return sortByScore(list, sc);
          },
          update(list, fb) {
            semiView(fb, m, v);
            for (let k = 0; k < m; k++) if (v[k] === v[k]) { n[list[k]] += 1; s[list[k]] += v[k]; }
          },
        };
      },
    },
    {
      id: 'omm', name: 'OMM', family: 'semi-bandit',
      params: [{ key: 'c', label: 'c', default: 2, min: 0, max: 5, step: 0.05 }],
      formula: 'Aₜ = Greedy_matroid(ŵᵢ + √(c·log t / nᵢ))',
      regret: 'O(L log T / Δ)  on any matroid',
      idea: 'Kveton et al. (2014) optimistic matroid maximisation: the greedy algorithm on UCBs is optimal on matroids. On the uniform matroid (top-m) it is CombUCB1 with a larger constant.',
      create(L, ctx) { const c = ctx.params.c; return indexLearner(L, ctx, semiView, (mu, n, b) => mu + Math.sqrt(b / n), (t) => c * Math.log(t)); },
    },
    // -------------------------------------------------------- adversarial
    {
      id: 'exp3m', name: 'EXP3.M', family: 'adversarial',
      params: [{ key: 'gamma', label: 'γ (0 = tuned to T)', default: 0, min: 0, max: 1, step: 0.01 }],
      formula: 'pᵢ = m((1−γ)w̃ᵢ/Σw̃ + γ/L),   wᵢ ← wᵢ·exp(mγ xᵢ / (L pᵢ))',
      regret: 'O(√(m L T log(L/m)))',
      idea: 'Uchiya, Nakamura & Kudo (2010): multiple-play EXP3 with capped weights and dependent rounding. Runs per block on the matroid; on full-bandit feedback each item gets credit reward/m.',
      create(L, ctx) {
        const { rng, space, params, T, m } = ctx;
        const lw = new Float64Array(L), p = new Float64Array(L), capped = new Uint8Array(L);
        const wbuf = new Float64Array(L), order = new Int32Array(L), work = new Float64Array(L), out = new Int32Array(m), v = new Float64Array(m);
        const gam = space.groups.map((g, j) => {
          const n = g.length, k = space.caps[j];
          if (params.gamma > 0) return Math.min(0.999, params.gamma);
          return k >= n ? 1 : Math.min(0.999, Math.sqrt((n * Math.log(n / k)) / ((Math.E - 1) * k * T)));
        });
        return {
          select() {
            for (let g = 0; g < space.groups.length; g++) exp3mProbs(lw, space.groups[g], space.caps[g], gam[g], p, capped, wbuf, order);
            return sampleMarginals(space, p, rng, work, out);
          },
          update(list, fb) {
            semiView(fb, m, v);
            for (let k = 0; k < m; k++) {
              const i = list[k];
              if (v[k] !== v[k] || capped[i]) continue;
              const g = space.type === 'topm' ? 0 : space.grp[i], n = space.groups[g].length;
              lw[i] += (space.caps[g] * gam[g] * v[k]) / (n * p[i]);
            }
          },
        };
      },
    },
    {
      id: 'comband', name: 'ComBand (CExp3)', family: 'adversarial',
      params: [
        { key: 'eta', label: 'η scale', default: 1, min: 0.05, max: 10, step: 0.05 },
        { key: 'gamma', label: 'γ (uniform exploration)', default: 0.05, min: 0.001, max: 0.5, step: 0.001 },
      ],
      formula: 'q(A) ∝ (1−γ)·exp(−η Σᵢ∈A L̂ᵢ) + γ·unif,   ℓ̂ = ℓₜ·P⁺v_A,  P = E_q[v vᵀ]',
      regret: 'O(√(m T L log |𝒜|))  with bandit (total-reward) feedback',
      idea: 'Cesa-Bianchi & Lugosi (2012): exponential weights over all m-sets, using only the total loss and the covariance P of the sampling law. Exact sampling and exact P via elementary symmetric polynomials (per block on the matroid).',
      create(L, ctx) {
        const { rng, space, params, T, m, info } = ctx, G = space.groups.length;
        let lnN = 0;
        for (let g = 0; g < G; g++) lnN += lnChoose(space.groups[g].length, space.caps[g]);
        const eta = params.eta * Math.sqrt(Math.max(lnN, 0.1) / (T * L)), gamma = params.gamma;
        const Lh = new Float64Array(L), z = new Float64Array(L), pe = new Float64Array(L), pu = new Float64Array(L);
        const PIe = new Float64Array(L * L), P = new Float64Array(L * L), rhs = new Float64Array(L), sol = new Float64Array(L);
        const maxN = Math.max(...space.groups.map((g) => g.length)), maxK = Math.max(...space.caps) + 1;
        const pre = new Float64Array((maxN + 1) * maxK), suf = space.groups.map((g, j) => new Float64Array((g.length + 1) * (space.caps[j] + 1)));
        const R = new Float64Array(maxK), out = new Int32Array(m), inA = new Uint8Array(L);
        for (let g = 0; g < G; g++) {
          const idx = space.groups[g], n = idx.length, k = space.caps[g];
          for (const i of idx) pu[i] = k / n;
        }
        function esp(g) {
          const idx = space.groups[g], n = idx.length, k = space.caps[g], K1 = k + 1, S = suf[g];
          pre.fill(0, 0, (n + 1) * K1); pre[0] = 1;
          for (let j = 0; j < n; j++) {
            const zj = z[idx[j]];
            for (let a = 0; a <= k; a++) pre[(j + 1) * K1 + a] = pre[j * K1 + a] + (a > 0 ? zj * pre[j * K1 + a - 1] : 0);
          }
          S.fill(0); S[n * K1] = 1;
          for (let j = n - 1; j >= 0; j--) {
            const zj = z[idx[j]];
            for (let a = 0; a <= k; a++) S[j * K1 + a] = S[(j + 1) * K1 + a] + (a > 0 ? zj * S[(j + 1) * K1 + a - 1] : 0);
          }
          const E = S[k];
          for (let j = 0; j < n; j++) {
            let e = 0;
            for (let a = 0; a <= k - 1; a++) e += pre[j * K1 + a] * S[(j + 1) * K1 + (k - 1 - a)];
            pe[idx[j]] = (z[idx[j]] * e) / E;
          }
          if (k < 2) return;
          for (let i = 0; i < n; i++) {
            for (let a = 0; a <= k - 2; a++) R[a] = pre[i * K1 + a];
            for (let j = i + 1; j < n; j++) {
              let e = 0;
              for (let a = 0; a <= k - 2; a++) e += R[a] * S[(j + 1) * K1 + (k - 2 - a)];
              const val = (z[idx[i]] * z[idx[j]] * e) / E;
              PIe[idx[i] * L + idx[j]] = val; PIe[idx[j] * L + idx[i]] = val;
              const zj = z[idx[j]];
              for (let a = k - 2; a > 0; a--) R[a] += zj * R[a - 1];
            }
          }
        }
        return {
          select() {
            let mn = Infinity;
            for (let i = 0; i < L; i++) if (Lh[i] < mn) mn = Lh[i];
            for (let i = 0; i < L; i++) z[i] = Math.max(1e-35, Math.exp(-eta * (Lh[i] - mn)));
            PIe.fill(0);
            for (let g = 0; g < G; g++) esp(g);
            const unif = rng.uniform() < gamma;
            let at = 0;
            for (let g = 0; g < G; g++) {
              const idx = space.groups[g], n = idx.length, k = space.caps[g], K1 = k + 1, S = suf[g];
              let need = k;
              for (let j = 0; j < n && need > 0; j++) {
                const p = unif ? need / (n - j) : (z[idx[j]] * S[(j + 1) * K1 + need - 1]) / S[j * K1 + need];
                if (n - j === need || rng.uniform() < p) { out[at++] = idx[j]; need--; }
              }
            }
            return sortByScore(out, pe);
          },
          update(list, fb) {
            // P = (1−γ) E_exp[vvᵀ] + γ E_unif[vvᵀ]; groups are independent under both laws
            const grp = space.grp;
            for (let i = 0; i < L; i++) {
              const gi = grp[i], ki = space.caps[gi], ni = space.groups[gi].length;
              for (let j = 0; j < L; j++) {
                let ve, vu;
                if (i === j) { ve = pe[i]; vu = pu[i]; }
                else if (grp[j] === gi) { ve = PIe[i * L + j]; vu = (ki * (ki - 1)) / (ni * (ni - 1)); }
                else { ve = pe[i] * pe[j]; vu = pu[i] * pu[j]; }
                P[i * L + j] = (1 - gamma) * ve + gamma * vu + (i === j ? 1e-9 : 0);
              }
            }
            inA.fill(0);
            for (let k = 0; k < m; k++) inA[list[k]] = 1;
            for (let i = 0; i < L; i++) rhs[i] = inA[i];
            cholSolve(P, rhs, sol, L);
            const loss = m * (1 - Math.min(1, Math.max(0, fb.reward / info.rMax)));
            for (let i = 0; i < L; i++) Lh[i] += loss * sol[i];
          },
        };
      },
    },
    {
      id: 'osmd', name: 'OSMD (negentropy)', family: 'adversarial',
      params: [{ key: 'eta', label: 'η scale', default: 1, min: 0.05, max: 10, step: 0.05 }],
      formula: 'xₜ₊₁ = argmin_{x∈𝒫} η⟨ℓ̂ₜ, x⟩ + KL(x‖xₜ),   ℓ̂ᵢ = ℓᵢ·1{i∈Aₜ}/xᵢ',
      regret: 'O(√(m L T log(L/m)))  semi-bandit',
      idea: 'Audibert, Bubeck & Lugosi (2014): mirror descent with the negative-entropy regulariser on the m-set polytope (capped simplex; per-block simplex on the matroid), sampling a set with the current marginals by dependent rounding.',
      create(L, ctx) {
        const { rng, space, params, T, m } = ctx;
        const x = new Float64Array(L), ly = new Float64Array(L), wbuf = new Float64Array(L), order = new Int32Array(L);
        const work = new Float64Array(L), out = new Int32Array(m), v = new Float64Array(m);
        const etas = space.groups.map((g, j) => {
          const n = g.length, k = space.caps[j];
          return params.eta * Math.sqrt((2 * k * (1 + Math.log(n / k))) / (n * T));
        });
        space.groups.forEach((g, j) => { for (const i of g) x[i] = space.caps[j] / g.length; });
        return {
          select() { return sampleMarginals(space, x, rng, work, out); },
          update(list, fb) {
            semiView(fb, m, v);
            for (let i = 0; i < L; i++) ly[i] = Math.log(x[i]);
            for (let k = 0; k < m; k++) {
              const i = list[k];
              if (v[k] !== v[k]) continue;
              const g = space.type === 'topm' ? 0 : space.grp[i];
              ly[i] -= (etas[g] * (1 - v[k])) / x[i];
            }
            for (let g = 0; g < space.groups.length; g++) capSimplexProject(ly, space.groups[g], space.caps[g], x, wbuf, order);
          },
        };
      },
    },
    // ---------------------------------------------------------- cascading
    {
      id: 'cascadeucb1', name: 'CascadeUCB1', family: 'cascading',
      params: [{ key: 'c', label: 'c', default: 1.5, min: 0, max: 5, step: 0.05 }],
      formula: 'Aₜ = top-m of ŵᵢ + √(c·log t / nᵢ),   ŵ from items up to the first click',
      regret: 'O((L − m) log T / Δ)',
      idea: 'Kveton et al. (2015): UCB on attraction probabilities; every item above the first click counts as examined-and-skipped. Reads any click vector as a cascade; uses item observations directly when there are no clicks.',
      create(L, ctx) { const c = ctx.params.c; return indexLearner(L, ctx, cascadeView, (mu, n, b) => mu + Math.sqrt(b / n), (t) => c * Math.log(t)); },
    },
    {
      id: 'cascadeklucb', name: 'CascadeKL-UCB', family: 'cascading', params: [],
      formula: 'Uᵢ = max{q : nᵢ·kl(ŵᵢ, q) ≤ log t + 3 log log t}',
      regret: 'O((L − m) log T / Δ),  tighter constants',
      idea: 'Kveton et al. (2015): CascadeUCB1 with Bernoulli KL-UCB indices, which are much tighter for small attraction probabilities.',
      create(L, ctx) { return indexLearner(L, ctx, cascadeView, (mu, n, b) => klUcb(mu, n, b), klBound); },
    },
    {
      id: 'cascadets', name: 'CascadeTS', family: 'cascading', params: [],
      formula: 'θᵢ = ŵᵢ + Z·max(√(ν̂ᵢ log(t+1)/(nᵢ+1)), log(t+1)/(nᵢ+1)),   Z ~ N(0, 1)',
      regret: 'Õ(√(L T))',
      idea: 'Cheung et al. (2019) TS-Cascade: one shared Gaussian draw per round scales each item\'s empirical-variance width, which keeps the samples correlated and cheap.',
      create(L, ctx) {
        const { rng, m, oracle } = ctx, n = new Float64Array(L), s = new Float64Array(L), sc = new Float64Array(L), v = new Float64Array(m);
        return {
          select(t) {
            const Z = rng.normal(), lg = Math.log(t + 1);
            for (let i = 0; i < L; i++) {
              const mu = n[i] ? s[i] / n[i] : 0, nu = mu * (1 - mu);
              sc[i] = mu + Z * Math.max(Math.sqrt((nu * lg) / (n[i] + 1)), lg / (n[i] + 1));
            }
            return oracle(sc);
          },
          update(list, fb) {
            cascadeView(fb, m, v);
            for (let k = 0; k < m; k++) if (v[k] === v[k]) { n[list[k]] += 1; s[list[k]] += v[k]; }
          },
        };
      },
    },
    {
      id: 'cascadelinucb', name: 'CascadeLinUCB', family: 'cascading',
      params: [
        { key: 'alpha', label: 'α (bonus)', default: 0.5, min: 0, max: 5, step: 0.05 },
        { key: 'lambda', label: 'λ (ridge)', default: 1, min: 0.01, max: 10, step: 0.01 },
      ],
      formula: 'Uᵢ = min(1, θ̂ᵀxᵢ + α‖xᵢ‖_{M⁻¹}),   M = λI + Σ x xᵀ over examined items',
      regret: 'Õ(m d √T)',
      idea: 'Zong et al. (2016): attraction is linear in item features, so one ridge regression generalises across items. Without features it falls back to one-hot features (per-item ridge UCB).',
      create(L, ctx) {
        const { m, oracle, params, info } = ctx, mod = linModel(L, info, params.lambda), sc = new Float64Array(L), v = new Float64Array(m);
        return {
          select() {
            mod.refresh();
            for (let i = 0; i < L; i++) sc[i] = Math.min(1, mod.mean(i) + params.alpha * mod.width(i));
            return oracle(sc);
          },
          update(list, fb) {
            cascadeView(fb, m, v);
            for (let k = 0; k < m; k++) if (v[k] === v[k]) mod.add(list[k], v[k]);
          },
        };
      },
    },
    {
      id: 'cascadelints', name: 'CascadeLinTS', family: 'cascading',
      params: [
        { key: 'v', label: 'v (posterior scale)', default: 0.5, min: 0.01, max: 5, step: 0.01 },
        { key: 'lambda', label: 'λ (ridge)', default: 1, min: 0.01, max: 10, step: 0.01 },
      ],
      formula: 'θ̃ ~ N(θ̂, v²M⁻¹),   Aₜ = top-m of θ̃ᵀxᵢ',
      regret: 'Õ(m d^{3/2} √T)  (Bayesian-style)',
      idea: 'Zong et al. (2016): Thompson sampling on the linear attraction model. Without features each item gets an independent Gaussian sample (one-hot fallback).',
      create(L, ctx) {
        const { m, oracle, params, info, rng } = ctx, mod = linModel(L, info, params.lambda), sc = new Float64Array(L), v = new Float64Array(m);
        const d = mod.d, Lc = new Float64Array(d * d), zz = new Float64Array(d), tt = new Float64Array(d);
        return {
          select() {
            mod.refresh();
            if (d) {
              cholLower(mod.Ainv, d, Lc);
              for (let p = 0; p < d; p++) zz[p] = rng.normal();
              for (let p = 0; p < d; p++) { let u = 0; for (let q = 0; q <= p; q++) u += Lc[p * d + q] * zz[q]; tt[p] = mod.th[p] + params.v * u; }
              for (let i = 0; i < L; i++) sc[i] = mod.mean(i, tt);
            } else {
              for (let i = 0; i < L; i++) sc[i] = mod.mean(i) + params.v * mod.width(i) * rng.normal();
            }
            return oracle(sc);
          },
          update(list, fb) {
            cascadeView(fb, m, v);
            for (let k = 0; k < m; k++) if (v[k] === v[k]) mod.add(list[k], v[k]);
          },
        };
      },
    },
    // ------------------------------------------------------- click models
    {
      id: 'dcmucb', name: 'DCM-UCB', family: 'click-models',
      params: [{ key: 'c', label: 'c', default: 1.5, min: 0, max: 5, step: 0.05 }],
      formula: 'Uᵢ = ŵᵢ + √(c·log t / nᵢ),   ŵ from items up to the last click',
      regret: 'O((L − m) log T / Δ)  (DCM)',
      idea: 'Katariya et al. (2016) with a UCB1 bonus: under DCM the user may continue after a click, so every item up to the last click was examined; items below it are ignored.',
      create(L, ctx) { const c = ctx.params.c; return indexLearner(L, ctx, dcmView, (mu, n, b) => mu + Math.sqrt(b / n), (t) => c * Math.log(t)); },
    },
    {
      id: 'dcmklucb', name: 'dcmKL-UCB', family: 'click-models', params: [],
      formula: 'Uᵢ = max{q : nᵢ·kl(ŵᵢ, q) ≤ log t + 3 log log t},   DCM click reading',
      regret: 'O((L − m) log T / Δ)  (DCM)',
      idea: 'Katariya et al. (2016): KL-UCB indices on attraction with the DCM reading of clicks; the list is the top-m by index in decreasing order, matching decreasing termination probabilities.',
      create(L, ctx) { return indexLearner(L, ctx, dcmView, (mu, n, b) => klUcb(mu, n, b), klBound); },
    },
    {
      id: 'pbmucb', name: 'PBM-UCB', family: 'click-models',
      params: [{ key: 'delta', label: 'δ', default: 1, min: 0.1, max: 5, step: 0.05 }],
      formula: 'Uᵢ = Sᵢ/Ñᵢ + √(Nᵢ/Ñᵢ)·√(δ log t / (2Ñᵢ)),   Ñᵢ = Σ κ_pos',
      regret: 'O((L − m) log T / Δ)  with known κ',
      idea: 'Lagrée, Vernade & Cappé (2016): clicks are debiased by the known examination probabilities κ (Ñ counts effective views). Environments without position bias are read with κ = 1.',
      create(L, ctx) {
        const { m, oracle, params, info } = ctx, kappa = info.kappa || new Float64Array(m).fill(1);
        const S = new Float64Array(L), N = new Float64Array(L), Nt = new Float64Array(L), sc = new Float64Array(L), v = new Float64Array(m);
        return {
          select(t) {
            const lg = params.delta * Math.log(t);
            for (let i = 0; i < L; i++) sc[i] = N[i] === 0 ? Infinity : S[i] / Nt[i] + Math.sqrt(N[i] / Nt[i]) * Math.sqrt(lg / (2 * Nt[i]));
            return oracle(sc);
          },
          update(list, fb) {
            if (fb.clicks) clickSignal(fb, m, v); else semiView(fb, m, v);
            for (let k = 0; k < m; k++) if (v[k] === v[k]) { const i = list[k]; S[i] += v[k]; N[i] += 1; Nt[i] += kappa[k]; }
          },
        };
      },
    },
    {
      id: 'pbmpie', name: 'PBM-PIE', family: 'click-models', params: [],
      formula: 'leaders = top-m by Sᵢ/Ñᵢ;  last slot ← argmax_{i∉leaders} Bᵢ if Bᵢ > ŵ_(m),  Ñᵢ kl(ŵᵢ, Bᵢ) = log t',
      regret: 'Asymptotically optimal O(Σ log T / (κₘ kl))  with known κ',
      idea: 'Lagrée et al. (2016) parsimonious item exploration: exploit the empirical leaders and explore only in the last slot. Simplified: the KL index uses the pooled effective count Ñᵢ.',
      create(L, ctx) {
        const { m, oracle, info, space } = ctx, kappa = info.kappa || new Float64Array(m).fill(1);
        const S = new Float64Array(L), N = new Float64Array(L), Nt = new Float64Array(L), sc = new Float64Array(L), v = new Float64Array(m);
        const inL = new Uint8Array(L);
        return {
          select(t) {
            for (let i = 0; i < L; i++) sc[i] = N[i] === 0 ? Infinity : S[i] / Nt[i];
            const list = oracle(sc);
            if (m === 0 || sc[list[m - 1]] === Infinity) return list;
            inL.fill(0);
            for (let k = 0; k < m; k++) inL[list[k]] = 1;
            const last = list[m - 1], lm = sc[last], lg = Math.log(t);
            let bi = -1, bv = lm;
            for (let i = 0; i < L; i++) {
              if (inL[i] || (space.type !== 'topm' && space.grp[i] !== space.grp[last])) continue;
              const B = klUcb(sc[i], Nt[i], lg);
              if (B > bv) { bv = B; bi = i; }
            }
            if (bi >= 0) list[m - 1] = bi;
            return list;
          },
          update(list, fb) {
            if (fb.clicks) clickSignal(fb, m, v); else semiView(fb, m, v);
            for (let k = 0; k < m; k++) if (v[k] === v[k]) { const i = list[k]; S[i] += v[k]; N[i] += 1; Nt[i] += kappa[k]; }
          },
        };
      },
    },
    // ------------------------------------------------------------ ranking
    {
      id: 'rba', name: 'Ranked bandits (RBA)', family: 'ranking', params: [],
      formula: 'slot k: UCB1 over items;  reward 1 iff first click is at slot k',
      regret: 'O(T^{…}) vs (1 − 1/e)-optimal list; linear vs the best list',
      idea: 'Radlinski, Kleinberg & Joachims (2008): one UCB1 bandit per slot learns a diverse ranking; a duplicate pick is replaced at random and earns 0. Without clicks each slot is rewarded with its item\'s observation.',
      create(L, ctx) {
        const { m, rng, space } = ctx;
        const N = new Float64Array(m * L), S = new Float64Array(m * L), Tk = new Float64Array(m);
        const chosen = new Int32Array(m), replaced = new Uint8Array(m), used = new Uint8Array(L), list = new Int32Array(m), v = new Float64Array(m);
        const allowed = (k) => (space.type === 'topm' ? space.groups[0] : space.groups[k]);
        return {
          select() {
            used.fill(0);
            for (let k = 0; k < m; k++) {
              const idx = allowed(k), lt = Math.log(Math.max(1, Tk[k]));
              let bi = -1, bv = -Infinity, ties = 0;
              for (const i of idx) {
                const n = N[k * L + i], val = n === 0 ? Infinity : S[k * L + i] / n + Math.sqrt((2 * lt) / n);
                if (val > bv) { bv = val; bi = i; ties = 1; } else if (val === bv && rng.uniform() * ++ties < 1) bi = i;
              }
              chosen[k] = bi; replaced[k] = 0;
              if (used[bi]) {
                replaced[k] = 1;
                let pick = -1, cnt = 0;
                for (const i of idx) if (!used[i] && rng.uniform() * ++cnt < 1) pick = i;
                bi = pick;
              }
              used[bi] = 1; list[k] = bi;
            }
            return list;
          },
          update(list, fb) {
            let first = -1;
            if (fb.clicks) { for (let k = 0; k < m; k++) if (fb.clicks[k] > 0) { first = k; break; } }
            else semiView(fb, m, v);
            for (let k = 0; k < m; k++) {
              let r = fb.clicks ? (k === first ? 1 : 0) : v[k] === v[k] ? v[k] : 0;
              if (replaced[k]) r = 0;
              N[k * L + chosen[k]] += 1; S[k * L + chosen[k]] += r; Tk[k] += 1;
            }
          },
        };
      },
    },
    {
      id: 'toprank', name: 'TopRank', family: 'ranking',
      params: [{ key: 'c', label: 'c (threshold)', default: 3.43, min: 0.5, max: 10, step: 0.01 }],
      formula: 'i ≻ j once Sᵢⱼ ≥ √(2Nᵢⱼ log(c·T·√Nᵢⱼ)),   Sᵢⱼ = Σ (Cᵢ − Cⱼ) within a block',
      regret: 'O(m³ L log T / Δ),  Õ(√(m³ L T))',
      idea: 'Lattimore et al. (2018): builds a partial order from pairwise click differences of items that share a randomly permuted block; needs only that more attractive items get more clicks in any position (covers cascade, DCM, PBM).',
      create(L, ctx) {
        const { m, rng, oracle, params, T } = ctx;
        const S = new Float64Array(L * L), N = new Float64Array(L * L), better = new Uint8Array(L * L);
        const level = new Int32Array(L), sc = new Float64Array(L), C = new Float64Array(L), v = new Float64Array(m);
        const shownLev = new Uint8Array(L + 1), members = new Int32Array(L);
        const logc = Math.log(params.c * T);
        function relevel() {
          level.fill(0);
          for (let pass = 0; pass < L; pass++) {
            let changed = false;
            for (let j = 0; j < L; j++) for (let i = 0; i < L; i++) {
              if (better[j * L + i] && level[j] < level[i] + 1 && level[i] + 1 < L) { level[j] = level[i] + 1; changed = true; }
            }
            if (!changed) break;
          }
        }
        return {
          select() {
            for (let i = 0; i < L; i++) sc[i] = -(level[i] + rng.uniform());
            return oracle(sc);
          },
          update(list, fb) {
            clickSignal(fb, m, v);
            C.fill(0); shownLev.fill(0);
            for (let k = 0; k < m; k++) { C[list[k]] = v[k]; shownLev[level[list[k]]] = 1; }
            let cnt = 0;
            for (let i = 0; i < L; i++) if (shownLev[level[i]]) members[cnt++] = i;
            let added = false;
            for (let a = 0; a < cnt; a++) {
              const i = members[a];
              for (let b = a + 1; b < cnt; b++) {
                const j = members[b];
                if (level[i] !== level[j]) continue;
                const u = C[i] - C[j];
                if (u === 0) continue;
                const au = Math.abs(u);
                S[i * L + j] += u; N[i * L + j] += au; S[j * L + i] -= u; N[j * L + i] += au;
                const n = N[i * L + j], thr = Math.sqrt(2 * n * Math.max(0, logc + 0.5 * Math.log(n)));
                if (S[i * L + j] >= thr && !better[j * L + i]) { better[j * L + i] = 1; added = true; }
                else if (S[j * L + i] >= thr && !better[i * L + j]) { better[i * L + j] = 1; added = true; }
              }
            }
            if (added) relevel();
          },
        };
      },
    },
  ];
  const ALGO_BY_ID = {};
  for (const a of ALGORITHMS) ALGO_BY_ID[a.id] = a;

  function defaultParams(id) {
    const out = {};
    for (const p of ALGO_BY_ID[id].params) out[p.key] = p.default;
    return out;
  }

  // ------------------------------------------------------------- runner ----
  const DEFAULT_L = 16, DEFAULT_M = 4;
  function listSize(L, extra) {
    const raw = extra && extra.m != null ? Math.round(extra.m) : DEFAULT_M;
    return Math.max(1, Math.min(L - 1, raw));
  }

  /** Build one agent on an environment instance, as the runner does (used by tests and tools). */
  function createAgent(algoId, env, inst, L, m, T, rng, params) {
    return ALGO_BY_ID[algoId].create(L, {
      T, L, m, env, info: inst.info, space: inst.space,
      params: Object.assign(defaultParams(algoId), params || {}), rng, oracle: makeOracle(inst.space, L, m, rng),
    });
  }
  function makeFeedback(inst, m) {
    return { obs: new Float64Array(m), clicks: inst.info.clicks ? new Float64Array(m) : null, reward: 0, full: !!inst.info.full, proxy: 0 };
  }

  function createRunner(config) {
    const env = ENV_BY_ID[config.env];
    if (!env) throw new Error('Unknown environment: ' + config.env);
    const L = Math.max(2, (config.K | 0) || DEFAULT_L), T = (config.T | 0) || 2000, runs = (config.runs | 0) || 1;
    const seed = config.seed >>> 0, m = listSize(L, config.extra);
    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaultParams(spec.id), spec.params || {}),
        regSum: new Float64Array(T), regSq: new Float64Array(T), rewSum: new Float64Array(T), optSum: new Float64Array(T),
        finals: [], timeMs: 0,
      };
    });
    const cur = new Int32Array(m);
    let done = 0;

    function runOne(run) {
      const inst = env.make(L, m, T, Rng(hash32(seed, run, 'env')));
      const best = inst.best;
      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const t0 = now();
        const agent = createAgent(slot.def.id, env, inst, L, m, T, rng, slot.params);
        const fb = makeFeedback(inst, m);
        const { regSum, regSq, rewSum, optSum } = slot;
        let cum = 0;
        for (let t = 0; t < T; t++) {
          const a = agent.select(t + 1);
          for (let k = 0; k < m; k++) cur[k] = a[k];
          const e = inst.expected(cur, t);
          cum += best[t] - e;
          inst.feedback(cur, t, fb);
          agent.update(cur, fb, t + 1);
          regSum[t] += cum; regSq[t] += cum * cum; rewSum[t] += e;
          if (e >= best[t] - 1e-9) optSum[t] += 1;
        }
        slot.timeMs += now() - t0;
        slot.finals.push(cum);
      }
    }

    return {
      config: { env: env.id, K: L, T, runs, seed, m },
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
          const regret = new Float64Array(T), regretSE = new Float64Array(T), reward = new Float64Array(T), optimal = new Float64Array(T);
          for (let t = 0; t < T; t++) {
            const mu = s.regSum[t] / R;
            regret[t] = mu;
            regretSE[t] = R > 1 ? Math.sqrt(Math.max(0, s.regSq[t] / R - mu * mu) / (R - 1)) : 0;
            reward[t] = s.rewSum[t] / R;
            optimal[t] = s.optSum[t] / R;
          }
          const sorted = s.finals.slice().sort((a, b) => a - b);
          const tail = Math.max(1, Math.floor(T / 10));
          let optTail = 0, rewAll = 0;
          for (let t = T - tail; t < T; t++) optTail += optimal[t] / tail;
          for (let t = 0; t < T; t++) rewAll += reward[t] / T;
          const summary = {
            finalRegret: regret[T - 1], finalSE: regretSE[T - 1],
            finalMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            meanReward: rewAll, optimalTail: optTail, msPerRun: done ? s.timeMs / done : 0,
          };
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, params: s.params, runs: done,
            curves: { regret, reward, optimal }, curvesSE: { regret: regretSE },
            summary, finals: sorted,
            finalRegret: summary.finalRegret, finalSE: summary.finalSE, finalMedian: summary.finalMedian,
            meanReward: rewAll, optimalTail: optTail, msPerRun: summary.msPerRun,
          };
        });
      },
    };
  }

  const setting = MAB.registerSetting({
    id: 'combinatorial',
    name: 'Combinatorial & ranking',
    short: 'Combinatorial',
    blurb: 'Choose a list of m items out of L each round and learn from semi-bandit, cascade, click-model or total-reward feedback.',
    families: FAMILIES,
    environments: ENVIRONMENTS,
    algorithms: ALGORITHMS,
    defaultEnv: 'topm',
    defaultLineup: ['random', 'greedy', 'cucb', 'cts', 'escb', 'exp3m', 'cascadeklucb', 'toprank'],
    presets: [
      { label: 'Semi-bandit classics', algos: ['random', 'greedy', 'cucb', 'combucb1', 'cts', 'escb', 'omm', 'osmd'], env: 'topm' },
      { label: 'Matroid', algos: ['random', 'omm', 'cucb', 'cts', 'escb', 'exp3m', 'osmd', 'comband'], env: 'matroid' },
      { label: 'Cascade', algos: ['random', 'cascadeucb1', 'cascadeklucb', 'cascadets', 'cts', 'rba', 'toprank', 'dcmklucb'], env: 'cascade' },
      { label: 'Click models', algos: ['cascadeklucb', 'dcmucb', 'dcmklucb', 'pbmucb', 'pbmpie', 'toprank', 'rba', 'cucb'], env: 'dcm' },
      { label: 'Position bias', algos: ['random', 'pbmucb', 'pbmpie', 'cucb', 'cascadeklucb', 'toprank', 'rba', 'cts'], env: 'pbm' },
      { label: 'Features', algos: ['random', 'cascadelinucb', 'cascadelints', 'cascadeucb1', 'cascadeklucb', 'cascadets'], env: 'cascade-lin' },
      { label: 'Adversarial', algos: ['random', 'exp3m', 'comband', 'osmd', 'cucb', 'cts', 'greedy'], env: 'adversarial' },
    ],
    controls: {
      K: { label: 'Items (L)', min: 4, max: 40, default: DEFAULT_L },
      T: { label: 'Horizon (T rounds)', options: [500, 1000, 2000, 5000], default: 2000 },
      runs: { label: 'Runs averaged', options: [10, 20, 50, 100, 200], default: 50 },
      extra: [{ key: 'm', label: 'List size (m)', min: 1, max: 8, step: 1, default: DEFAULT_M }],
    },
    xLabel: 'round t',
    metrics: [
      { key: 'regret', label: 'Cumulative regret', better: 'lower', fmt: 'num', band: true,
        note: 'Expected reward of the best fixed list under the environment\'s reward model minus that of the chosen lists. Bands: 95% CI across runs.' },
      { key: 'reward', label: 'Reward per round', better: 'higher', fmt: 'fixed3', smooth: true,
        note: 'Expected reward of the chosen list at each round (sum of weights, click probability or expected clicks).' },
      { key: 'optimal', label: 'Best-list rate', better: 'higher', fmt: 'pct', smooth: true, zeroToOne: true,
        note: 'Share of runs whose list was at least as good as the best fixed list at that round.' },
    ],
    final: { key: 'finalRegret', se: 'finalSE', median: 'finalMedian', label: 'Final regret', better: 'lower' },
    tiles: [
      { key: 'meanReward', label: 'Highest reward', better: 'higher', fmt: 'fixed3', sub: 'expected reward per round' },
      { key: 'optimalTail', label: 'Finds the best list', better: 'higher', fmt: 'pct', sub: 'of the last 10% of rounds' },
    ],
    columns: [
      { key: 'finalRegret', label: 'Regret', fmt: 'num', better: 'lower' },
      { key: 'finalSE', label: '95% CI', fmt: 'ci' },
      { key: 'finalMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'meanReward', label: 'Reward / round', fmt: 'fixed3', better: 'higher' },
      { key: 'optimalTail', label: 'Best list, last 10%', fmt: 'pct', better: 'higher' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: false,
    createRunner,
  });
  // Internals for tests and tooling.
  setting.internals = { ENV_BY_ID, ALGO_BY_ID, defaultParams, makeOracle, makeFeedback, createAgent, depRound, listSize };
  return setting;
});
