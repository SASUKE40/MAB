/*
 * mab.js — Multi-armed bandit algorithms, environments and a benchmark runner.
 *
 * One file, no dependencies. Works as a browser <script> (exposes `window.MAB`)
 * and as a CommonJS module in Node (`require('./src/mab.js')`).
 *
 * Design notes
 *  - Every run draws one bandit instance and pre-generates the full reward table
 *    (T x K). All algorithms in that run read from the same table, so if two
 *    algorithms pull the same arm at the same step they see the same reward
 *    (common random numbers). This removes most between-algorithm noise.
 *  - Regret is pseudo-regret: sum over t of (mu*_t - mu_{a_t}). For
 *    non-stationary environments mu*_t is the best mean at step t (dynamic regret).
 *  - Seeds are derived from (seed, run, algorithm id), so an algorithm's results
 *    do not depend on which other algorithms are selected.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MAB = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------- RNG ----
  function hash32(...parts) {
    let h = 0x9e3779b9;
    for (const p of parts) {
      let x = typeof p === 'string' ? strHash(p) : (p >>> 0);
      h ^= x + 0x9e3779b9 + (h << 6) + (h >>> 2);
      h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
      h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
      h ^= h >>> 16;
    }
    return h >>> 0;
  }
  function strHash(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return h >>> 0;
  }

  /** mulberry32 PRNG with normal, gamma and beta samplers. */
  function Rng(seed) {
    let a = seed >>> 0;
    let spare = null;
    const rng = {
      uniform() {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      },
      normal() {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let u, v, s;
        do {
          u = rng.uniform() * 2 - 1;
          v = rng.uniform() * 2 - 1;
          s = u * u + v * v;
        } while (s >= 1 || s === 0);
        const m = Math.sqrt(-2 * Math.log(s) / s);
        spare = v * m;
        return u * m;
      },
      gamma(k) {
        // Marsaglia & Tsang (2000)
        if (k < 1) return rng.gamma(k + 1) * Math.pow(rng.uniform() || 1e-300, 1 / k);
        const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
        for (;;) {
          let x, v;
          do { x = rng.normal(); v = 1 + c * x; } while (v <= 0);
          v = v * v * v;
          const u = rng.uniform();
          if (u < 1 - 0.0331 * x * x * x * x) return d * v;
          if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
      },
      beta(alpha, beta) {
        const x = rng.gamma(alpha), y = rng.gamma(beta);
        return x / (x + y);
      },
      int(n) { return Math.floor(rng.uniform() * n); },
    };
    return rng;
  }

  // ------------------------------------------------------------ helpers ----
  /** Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.2e-9 after refinement-free use). */
  function normInv(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425;
    let q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - pl) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  /** Bernoulli KL divergence kl(p, q). */
  function klBern(p, q) {
    const eps = 1e-12;
    p = Math.min(Math.max(p, eps), 1 - eps);
    q = Math.min(Math.max(q, eps), 1 - eps);
    return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
  }

  /** Largest q in [p, 1] with n * kl(p, q) <= bound (bisection). */
  function klUcbBern(p, n, bound) {
    let lo = p, hi = 1;
    const target = bound / n;
    for (let i = 0; i < 16; i++) {
      const mid = (lo + hi) / 2;
      if (klBern(p, mid) > target) hi = mid; else lo = mid;
    }
    return lo;
  }

  /** argmax with uniformly random tie-breaking. */
  function argmax(values, rng) {
    let best = -Infinity, idx = 0, ties = 0;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v > best) { best = v; idx = i; ties = 1; }
      else if (v === best) { ties++; if (rng.uniform() * ties < 1) idx = i; }
    }
    return idx;
  }

  /** First arm with zero pulls, or -1. */
  function untried(n) {
    for (let i = 0; i < n.length; i++) if (n[i] === 0) return i;
    return -1;
  }

  /** Sample an index from a probability vector. */
  function sampleIndex(p, rng) {
    let u = rng.uniform(), acc = 0;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (u < acc) return i; }
    return p.length - 1;
  }

  function softmaxInto(out, x, scale) {
    let m = -Infinity;
    for (let i = 0; i < x.length; i++) if (x[i] * scale > m) m = x[i] * scale;
    let s = 0;
    for (let i = 0; i < x.length; i++) { out[i] = Math.exp(x[i] * scale - m); s += out[i]; }
    for (let i = 0; i < x.length; i++) out[i] /= s;
    return out;
  }

  // ------------------------------------------------------- environments ----
  /*
   * Every environment fills `means` (T*K, row-major by step) and `rewards` (T*K).
   *  type     'bernoulli' (rewards in {0,1}) or 'gaussian' (unit variance)
   *  scale    sub-Gaussian width used to scale exploration bonuses
   *  normalize(r) maps a reward into [0, 1] for algorithms that need bounded rewards
   */
  const ENVIRONMENTS = [
    {
      id: 'bernoulli',
      name: 'Bernoulli, uniform means',
      short: 'Bernoulli',
      type: 'bernoulli', stationary: true,
      description: 'Each arm pays 1 with probability μ ~ U(0, 1), else 0. Gaps are usually large.',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K);
        for (let a = 0; a < K; a++) mu[a] = rng.uniform();
        for (let t = 0; t < T; t++) {
          for (let a = 0; a < K; a++) {
            means[t * K + a] = mu[a];
            rewards[t * K + a] = rng.uniform() < mu[a] ? 1 : 0;
          }
        }
      },
    },
    {
      id: 'bernoulli-close',
      name: 'Bernoulli, close arms',
      short: 'Close arms',
      type: 'bernoulli', stationary: true,
      description: 'One arm pays 0.55; the rest draw μ ~ U(0.40, 0.52). Small gaps make exploration expensive.',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K);
        const best = rng.int(K);
        for (let a = 0; a < K; a++) mu[a] = a === best ? 0.55 : 0.40 + 0.12 * rng.uniform();
        for (let t = 0; t < T; t++) {
          for (let a = 0; a < K; a++) {
            means[t * K + a] = mu[a];
            rewards[t * K + a] = rng.uniform() < mu[a] ? 1 : 0;
          }
        }
      },
    },
    {
      id: 'gaussian',
      name: 'Gaussian testbed',
      short: 'Gaussian',
      type: 'gaussian', stationary: true,
      description: 'The classic 10-armed testbed: μ ~ N(0, 1), rewards ~ N(μ, 1).',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K);
        for (let a = 0; a < K; a++) mu[a] = rng.normal();
        for (let t = 0; t < T; t++) {
          for (let a = 0; a < K; a++) {
            means[t * K + a] = mu[a];
            rewards[t * K + a] = mu[a] + rng.normal();
          }
        }
      },
    },
    {
      id: 'drift',
      name: 'Gaussian, drifting means',
      short: 'Drifting',
      type: 'gaussian', stationary: false,
      description: 'Starts like the Gaussian testbed, then every mean takes a random-walk step N(0, 0.03²) each round.',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K);
        for (let a = 0; a < K; a++) mu[a] = rng.normal();
        for (let t = 0; t < T; t++) {
          for (let a = 0; a < K; a++) {
            means[t * K + a] = mu[a];
            rewards[t * K + a] = mu[a] + rng.normal();
            mu[a] += 0.03 * rng.normal();
          }
        }
      },
    },
    {
      id: 'abrupt',
      name: 'Bernoulli, abrupt changes',
      short: 'Abrupt',
      type: 'bernoulli', stationary: false,
      description: 'Bernoulli arms whose means are all redrawn from U(0, 1) at four evenly spaced change points.',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K);
        const segment = Math.ceil(T / 5);
        for (let t = 0; t < T; t++) {
          if (t % segment === 0) for (let a = 0; a < K; a++) mu[a] = rng.uniform();
          for (let a = 0; a < K; a++) {
            means[t * K + a] = mu[a];
            rewards[t * K + a] = rng.uniform() < mu[a] ? 1 : 0;
          }
        }
      },
    },
  ];
  for (const e of ENVIRONMENTS) {
    if (e.type === 'bernoulli') {
      e.sigma = 0.5; e.scale = 1;
      e.normalize = (r) => r;
    } else {
      e.sigma = 1; e.scale = Math.SQRT2;
      e.normalize = (r) => Math.min(1, Math.max(0, (r + 4) / 8));
    }
  }
  const ENV_BY_ID = Object.fromEntries(ENVIRONMENTS.map((e) => [e.id, e]));

  // --------------------------------------------------------- algorithms ----
  /*
   * Each algorithm: create(K, ctx) -> { select(t), update(arm, reward, t) }
   *   t runs 1..T. ctx = { T, env, params, rng }.
   */
  const FAMILIES = [
    { id: 'baseline', name: 'Baselines', blurb: 'Reference points: no learning, or learning with no exploration.' },
    { id: 'undirected', name: 'Undirected exploration', blurb: 'Explore by injecting randomness that ignores how uncertain each arm is.' },
    { id: 'optimism', name: 'Optimism in the face of uncertainty', blurb: 'Act greedily on an optimistic estimate: mean plus a bonus that shrinks with pulls.' },
    { id: 'bayesian', name: 'Bayesian / posterior', blurb: 'Keep a posterior over each mean; sample from it or read off an upper quantile.' },
    { id: 'gradient', name: 'Policy gradient', blurb: 'Learn action preferences directly by stochastic gradient ascent on expected reward.' },
    { id: 'adversarial', name: 'Adversarial', blurb: 'No stochastic assumptions; guarantees hold even if rewards are chosen by an opponent.' },
    { id: 'nonstationary', name: 'Non-stationary', blurb: 'Forget old evidence so the agent can track arms whose means move.' },
  ];

  function stats(K) {
    return { n: new Float64Array(K), sum: new Float64Array(K), mean: new Float64Array(K) };
  }
  function record(s, a, r) {
    s.n[a] += 1; s.sum[a] += r; s.mean[a] = s.sum[a] / s.n[a];
  }

  const ALGORITHMS = [
    {
      id: 'random', name: 'Random', family: 'baseline',
      params: [],
      formula: 'aₜ ~ Uniform{1, …, K}',
      regret: 'Θ(T)',
      idea: 'Ignores all feedback. Its regret is the cost of pure exploration.',
      create(K, { rng }) {
        return { select: () => rng.int(K), update() {} };
      },
    },
    {
      id: 'greedy', name: 'Greedy', family: 'baseline',
      params: [],
      formula: 'aₜ = argmaxₐ Q̂ₐ   (Q̂ starts at 0)',
      regret: 'Θ(T)',
      idea: 'Always exploits the current sample mean. Locks onto whichever arm looked good early.',
      create(K, { rng }) {
        const s = stats(K);
        return { select: () => argmax(s.mean, rng), update: (a, r) => record(s, a, r) };
      },
    },
    {
      id: 'egreedy', name: 'ε-greedy', family: 'undirected',
      params: [{ key: 'epsilon', label: 'ε', default: 0.1, min: 0, max: 1, step: 0.01 }],
      formula: 'random arm w.p. ε, else argmaxₐ Q̂ₐ',
      regret: 'Θ(T)  (keeps exploring at rate ε)',
      idea: 'Simple and robust, but never stops paying for exploration.',
      create(K, { rng, params }) {
        const s = stats(K), eps = params.epsilon;
        return {
          select: () => (rng.uniform() < eps ? rng.int(K) : argmax(s.mean, rng)),
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'egreedy-decay', name: 'Decaying ε-greedy', family: 'undirected',
      params: [{ key: 'c', label: 'c', default: 2, min: 0.1, max: 50, step: 0.1 }],
      formula: 'εₜ = min(1, c·K / t)',
      regret: 'O(log T)  with c tuned to the gap',
      idea: 'Explores heavily early and anneals. Logarithmic only if c matches the unknown gap.',
      create(K, { rng, params }) {
        const s = stats(K), c = params.c;
        return {
          select: (t) => (rng.uniform() < Math.min(1, (c * K) / t) ? rng.int(K) : argmax(s.mean, rng)),
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'etc', name: 'Explore-then-commit', family: 'undirected',
      params: [{ key: 'm', label: 'm (pulls per arm)', default: 20, min: 1, max: 500, step: 1 }],
      formula: 'round-robin for m·K steps, then argmaxₐ Q̂ₐ forever',
      regret: 'O(T^⅔)  untuned; O(log T) if m knows the gap',
      idea: 'Separates exploration and exploitation into two phases. A bad commit is permanent.',
      create(K, { rng, params }) {
        const s = stats(K), m = Math.max(1, Math.round(params.m));
        let committed = -1;
        return {
          select(t) {
            if (t <= m * K) return (t - 1) % K;
            if (committed < 0) committed = argmax(s.mean, rng);
            return committed;
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'softmax', name: 'Softmax (Boltzmann)', family: 'undirected',
      params: [{ key: 'tau', label: 'τ (temperature)', default: 0.1, min: 0.005, max: 5, step: 0.005 }],
      formula: 'P(a) ∝ exp(Q̂ₐ / τ)',
      regret: 'Θ(T)  for fixed τ',
      idea: 'Explores in proportion to estimated value, so bad arms are tried less than with ε-greedy.',
      create(K, { rng, params, env }) {
        const s = stats(K), p = new Float64Array(K), inv = 1 / (params.tau * env.scale);
        return {
          select: () => sampleIndex(softmaxInto(p, s.mean, inv), rng),
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'optimistic', name: 'Optimistic initial values', family: 'optimism',
      params: [
        { key: 'q0', label: 'Q₀', default: 5, min: 0, max: 50, step: 0.5 },
        { key: 'alpha', label: 'α (step size)', default: 0.1, min: 0.01, max: 1, step: 0.01 },
      ],
      formula: 'Q̂ₐ ← Q̂ₐ + α(r − Q̂ₐ),  Q̂ starts at Q₀;  greedy',
      regret: 'Θ(T)  (exploration only at the start)',
      idea: 'Every arm disappoints until its inflated estimate decays, which forces early exploration.',
      create(K, { rng, params }) {
        const q = new Float64Array(K).fill(params.q0), alpha = params.alpha;
        return {
          select: () => argmax(q, rng),
          update(a, r) { q[a] += alpha * (r - q[a]); },
        };
      },
    },
    {
      id: 'ucb1', name: 'UCB1', family: 'optimism',
      params: [{ key: 'c', label: 'c', default: 1.414, min: 0, max: 5, step: 0.01 }],
      formula: 'argmaxₐ  Q̂ₐ + c·s·√(ln t / nₐ)',
      regret: 'O(Σ log T / Δₐ)',
      idea: 'Bonus grows slowly with time and shrinks with pulls: under-sampled arms get revisited.',
      create(K, { rng, params, env }) {
        const s = stats(K), idx = new Float64Array(K), c = params.c * env.scale;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            const lt = Math.log(t);
            for (let a = 0; a < K; a++) idx[a] = s.mean[a] + c * Math.sqrt(lt / s.n[a]);
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'ucb-tuned', name: 'UCB1-Tuned', family: 'optimism',
      params: [],
      formula: 'argmaxₐ  Q̂ₐ + √(ln t / nₐ · min(¼, Vₐ)),  Vₐ = σ̂ₐ² + √(2 ln t / nₐ)',
      regret: 'No proof; strong in practice',
      idea: 'Plugs the empirical variance into the bonus, so low-variance arms get a tighter interval.',
      create(K, { rng, env }) {
        const s = stats(K), sq = new Float64Array(K), idx = new Float64Array(K);
        const cap = env.type === 'bernoulli' ? 0.25 : Infinity;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            const lt = Math.log(t);
            for (let a = 0; a < K; a++) {
              const n = s.n[a], m = s.mean[a];
              const v = Math.max(0, sq[a] / n - m * m) + Math.sqrt((2 * lt) / n);
              idx[a] = m + Math.sqrt((lt / n) * Math.min(cap, v));
            }
            return argmax(idx, rng);
          },
          update(a, r) { record(s, a, r); sq[a] += r * r; },
        };
      },
    },
    {
      id: 'klucb', name: 'KL-UCB', family: 'optimism',
      params: [{ key: 'c', label: 'c (log-log term)', default: 0, min: 0, max: 3, step: 0.1 }],
      formula: 'max { q : nₐ·kl(Q̂ₐ, q) ≤ ln t + c ln ln t }',
      regret: 'Asymptotically optimal (matches Lai–Robbins)',
      idea: 'Replaces Hoeffding with the KL divergence of the reward family, giving the tightest optimistic index.',
      create(K, { rng, params, env }) {
        const s = stats(K), idx = new Float64Array(K), c = params.c;
        const bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            const lt = Math.log(t);
            const bound = lt + (t > 3 ? c * Math.log(lt) : 0);
            for (let a = 0; a < K; a++) {
              idx[a] = bern ? klUcbBern(s.mean[a], s.n[a], bound) : s.mean[a] + Math.sqrt((2 * s2 * bound) / s.n[a]);
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'moss', name: 'MOSS', family: 'optimism',
      params: [],
      formula: 'argmaxₐ  Q̂ₐ + s·√(max(0, ln(T / (K·nₐ))) / nₐ)',
      regret: 'O(√(KT))  minimax optimal; needs T',
      idea: 'Uses the known horizon to stop exploring an arm once it has had its fair share T/K.',
      create(K, { rng, env, T }) {
        const s = stats(K), idx = new Float64Array(K), sc = env.scale;
        return {
          select() {
            const u = untried(s.n); if (u >= 0) return u;
            for (let a = 0; a < K; a++) {
              const n = s.n[a];
              idx[a] = s.mean[a] + sc * Math.sqrt(Math.max(0, Math.log(T / (K * n))) / n);
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'bayes-ucb', name: 'Bayes-UCB', family: 'bayesian',
      params: [],
      formula: 'argmaxₐ  Quantile(1 − 1/t, posteriorₐ)',
      regret: 'Asymptotically optimal (Bernoulli)',
      idea: 'Optimism read off the posterior: pull the arm whose upper credible bound is highest.',
      create(K, { rng, env }) {
        const s = stats(K), idx = new Float64Array(K);
        const bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select(t) {
            const z = normInv(1 - 1 / (t + 1));
            for (let a = 0; a < K; a++) {
              const n = s.n[a];
              if (bern) {
                // Beta(1 + successes, 1 + failures), normal approximation to its quantile
                const al = 1 + s.sum[a], be = 1 + n - s.sum[a], ab = al + be;
                const m = al / ab, v = (al * be) / (ab * ab * (ab + 1));
                idx[a] = Math.min(1, m + z * Math.sqrt(v));
              } else {
                // N(0,1) prior, known unit noise -> N(sum/(n+1), σ²/(n+1))
                idx[a] = s.sum[a] / (n + 1) + z * Math.sqrt(s2 / (n + 1));
              }
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'thompson', name: 'Thompson sampling', family: 'bayesian',
      params: [],
      formula: 'θₐ ~ posteriorₐ,   aₜ = argmaxₐ θₐ',
      regret: 'Asymptotically optimal (Bernoulli, Gaussian)',
      idea: 'Probability matching: each arm is played with the posterior probability that it is best.',
      create(K, { rng, env }) {
        const s = stats(K), th = new Float64Array(K);
        const bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select() {
            for (let a = 0; a < K; a++) {
              const n = s.n[a];
              th[a] = bern
                ? rng.beta(1 + s.sum[a], 1 + n - s.sum[a])
                : s.sum[a] / (n + 1) + Math.sqrt(s2 / (n + 1)) * rng.normal();
            }
            return argmax(th, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'gradient', name: 'Gradient bandit', family: 'gradient',
      params: [{ key: 'alpha', label: 'α (step size)', default: 0.1, min: 0.005, max: 2, step: 0.005 }],
      formula: 'Hₐ ← Hₐ + α(r − r̄)(𝟙[a = aₜ] − πₐ),  π = softmax(H)',
      regret: 'No logarithmic guarantee',
      idea: 'REINFORCE with a running-average baseline. Learns relative preferences, not values.',
      create(K, { rng, params }) {
        const h = new Float64Array(K), p = new Float64Array(K), alpha = params.alpha;
        let base = 0, count = 0;
        return {
          select: () => sampleIndex(softmaxInto(p, h, 1), rng),
          update(a, r) {
            count++; base += (r - base) / count;
            const g = alpha * (r - base);
            for (let i = 0; i < K; i++) h[i] += g * ((i === a ? 1 : 0) - p[i]);
          },
        };
      },
    },
    {
      id: 'exp3', name: 'EXP3', family: 'adversarial',
      params: [{ key: 'gamma', label: 'γ (0 = tuned to T)', default: 0, min: 0, max: 1, step: 0.01 }],
      formula: 'πₐ = (1−γ)·wₐ/Σw + γ/K,   wₐ ← wₐ·exp(γ r̃ / (K πₐ))',
      regret: 'O(√(KT log K))  against any reward sequence',
      idea: 'Exponential weights on importance-weighted rewards. Pays for robustness with slow learning.',
      create(K, { rng, params, env, T }) {
        const lw = new Float64Array(K), p = new Float64Array(K);
        const gamma = params.gamma > 0 ? params.gamma : Math.min(1, Math.sqrt((K * Math.log(K)) / ((Math.E - 1) * T)));
        return {
          select() {
            softmaxInto(p, lw, 1);
            for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma / K;
            return sampleIndex(p, rng);
          },
          update(a, r) { lw[a] += (gamma * env.normalize(r)) / (K * p[a]); },
        };
      },
    },
    {
      id: 'sw-ucb', name: 'Sliding-window UCB', family: 'nonstationary',
      params: [
        { key: 'window', label: 'τ (window)', default: 500, min: 10, max: 5000, step: 10 },
        { key: 'c', label: 'c', default: 1, min: 0, max: 5, step: 0.01 },
      ],
      formula: 'UCB1 computed on the last τ plays only',
      regret: 'Õ(√(ΓT))  with Γ change points',
      idea: 'Evidence older than τ steps is dropped, so the index can follow a moving best arm.',
      create(K, { rng, params, env }) {
        const W = Math.max(1, Math.round(params.window)), c = params.c * env.scale;
        const n = new Float64Array(K), sum = new Float64Array(K), idx = new Float64Array(K);
        const histA = new Int32Array(W), histR = new Float64Array(W);
        let filled = 0, head = 0;
        return {
          select(t) {
            const u = untried(n); if (u >= 0) return u;
            const lt = Math.log(Math.min(t, W));
            for (let a = 0; a < K; a++) idx[a] = sum[a] / n[a] + c * Math.sqrt(lt / n[a]);
            return argmax(idx, rng);
          },
          update(a, r) {
            if (filled === W) { const oa = histA[head]; n[oa] -= 1; sum[oa] -= histR[head]; }
            else filled++;
            histA[head] = a; histR[head] = r; head = (head + 1) % W;
            n[a] += 1; sum[a] += r;
          },
        };
      },
    },
    {
      id: 'dts', name: 'Discounted Thompson', family: 'nonstationary',
      params: [{ key: 'gamma', label: 'γ (discount)', default: 0.99, min: 0.5, max: 1, step: 0.001 }],
      formula: 'all counts ← γ·counts each step;  then Thompson sampling',
      regret: 'No tight bound; strong empirically',
      idea: 'Geometric forgetting keeps the posterior wide enough to notice when an arm changes.',
      create(K, { rng, params, env }) {
        const n = new Float64Array(K), sum = new Float64Array(K), th = new Float64Array(K);
        const g = params.gamma, bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select() {
            for (let a = 0; a < K; a++) {
              th[a] = bern
                ? rng.beta(1 + sum[a], 1 + n[a] - sum[a])
                : sum[a] / (n[a] + 1) + Math.sqrt(s2 / (n[a] + 1)) * rng.normal();
            }
            return argmax(th, rng);
          },
          update(a, r) {
            for (let i = 0; i < K; i++) { n[i] *= g; sum[i] *= g; }
            n[a] += 1; sum[a] += r;
          },
        };
      },
    },
  ];
  const ALGO_BY_ID = Object.fromEntries(ALGORITHMS.map((a) => [a.id, a]));

  function defaultParams(algoId) {
    const out = {};
    for (const p of ALGO_BY_ID[algoId].params) out[p.key] = p.default;
    return out;
  }

  // ------------------------------------------------------------- runner ----
  /**
   * Create an incremental benchmark runner.
   * config = { env: id, K, T, runs, seed, algorithms: [{ id, params? }] }
   * runner.step(n) runs n more Monte-Carlo runs; runner.results() summarises so far.
   */
  function createRunner(config) {
    const env = ENV_BY_ID[config.env];
    if (!env) throw new Error('Unknown environment: ' + config.env);
    const K = config.K | 0, T = config.T | 0, runs = config.runs | 0, seed = config.seed >>> 0;
    if (K < 2) throw new Error('K must be at least 2');
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
    const best = new Float64Array(T), avgMean = new Float64Array(K), rank = new Int32Array(K);

    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def,
        params: Object.assign(defaultParams(spec.id), spec.params || {}),
        regSum: new Float64Array(T), regSq: new Float64Array(T),
        optSum: new Float64Array(T), rewSum: new Float64Array(T),
        pullsByRank: new Float64Array(K), finals: [], timeMs: 0,
      };
    });
    let done = 0;

    function runOne(run) {
      const envRng = Rng(hash32(seed, run, 'env'));
      env.fill(K, T, envRng, means, rewards);
      avgMean.fill(0);
      for (let t = 0; t < T; t++) {
        let b = -Infinity;
        for (let a = 0; a < K; a++) {
          const m = means[t * K + a];
          if (m > b) b = m;
          avgMean[a] += m / T;
        }
        best[t] = b;
      }
      // rank[arm] = 0 for the arm with the highest time-averaged mean
      const order = Array.from({ length: K }, (_, i) => i).sort((x, y) => avgMean[y] - avgMean[x]);
      order.forEach((arm, r) => { rank[arm] = r; });

      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const agent = slot.def.create(K, { T, env, params: slot.params, rng });
        const t0 = now();
        let cum = 0;
        const { regSum, regSq, optSum, rewSum, pullsByRank } = slot;
        for (let t = 0; t < T; t++) {
          const a = agent.select(t + 1);
          const r = rewards[t * K + a];
          agent.update(a, r, t + 1);
          const m = means[t * K + a];
          cum += best[t] - m;
          regSum[t] += cum;
          regSq[t] += cum * cum;
          if (m === best[t]) optSum[t] += 1;
          rewSum[t] += r;
          pullsByRank[rank[a]] += 1;
        }
        slot.timeMs += now() - t0;
        slot.finals.push(cum);
      }
    }

    return {
      config: { env: env.id, K, T, runs, seed },
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
          const regret = new Float64Array(T), regretSE = new Float64Array(T);
          const optimal = new Float64Array(T), reward = new Float64Array(T);
          for (let t = 0; t < T; t++) {
            const m = s.regSum[t] / R;
            regret[t] = m;
            const v = Math.max(0, s.regSq[t] / R - m * m);
            regretSE[t] = R > 1 ? Math.sqrt(v / (R - 1)) : 0;
            optimal[t] = s.optSum[t] / R;
            reward[t] = s.rewSum[t] / R;
          }
          const totalPulls = R * T;
          const sorted = s.finals.slice().sort((a, b) => a - b);
          const tail = Math.max(1, Math.floor(T / 10));
          let optTail = 0, rewAll = 0;
          for (let t = T - tail; t < T; t++) optTail += optimal[t] / tail;
          for (let t = 0; t < T; t++) rewAll += reward[t] / T;
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, params: s.params,
            runs: done, regret, regretSE, optimal, reward,
            pullShare: Array.from(s.pullsByRank, (x) => x / totalPulls),
            finalRegret: regret[T - 1], finalSE: regretSE[T - 1],
            finalMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            finals: sorted,
            optimalTail: optTail, meanReward: rewAll,
            msPerRun: done ? s.timeMs / done : 0,
          };
        });
      },
    };
  }

  function now() {
    return typeof performance !== 'undefined' ? performance.now() : Date.now();
  }

  /** Run a full benchmark synchronously. */
  function simulate(config) {
    const r = createRunner(config);
    r.step(config.runs);
    return r.results();
  }

  return {
    Rng, hash32, normInv, klBern, klUcbBern, argmax,
    ENVIRONMENTS, ENV_BY_ID, FAMILIES, ALGORITHMS, ALGO_BY_ID,
    defaultParams, createRunner, simulate,
  };
});
