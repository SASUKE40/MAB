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
    {
      id: 'contextual',
      name: 'Linear contextual',
      short: 'Contextual',
      type: 'gaussian', stationary: true, contextual: true, d: 5,
      description: 'Each round shows a context x = (1, z₁…z₄), z ~ N(0, 1). Arm a pays N(θₐ·x, 1) with a hidden θₐ ~ N(0, 0.5²I). Without the context, only the average arm can be learned.',
      fill(K, T, rng, means, rewards, ctx) {
        const d = 5, theta = new Float64Array(K * d), x = new Float64Array(d);
        for (let i = 0; i < K * d; i++) theta[i] = 0.5 * rng.normal();
        for (let t = 0; t < T; t++) {
          x[0] = 1;
          for (let j = 1; j < d; j++) x[j] = rng.normal();
          for (let a = 0; a < K; a++) {
            let m = 0;
            const row = (t * K + a) * d;
            for (let j = 0; j < d; j++) { m += theta[a * d + j] * x[j]; ctx[row + j] = x[j]; }
            means[t * K + a] = m;
            rewards[t * K + a] = m + rng.normal();
          }
        }
      },
    },
  ];
  for (const e of ENVIRONMENTS) prepareEnv(e);
  const ENV_BY_ID = {};
  for (const e of ENVIRONMENTS) ENV_BY_ID[e.id] = e;

  // --------------------------------------------------------- algorithms ----
  /*
   * Each algorithm: create(K, ctx) -> { select(t), update(arm, reward, t) }
   *   t runs 1..T. ctx = { T, env, params, rng }.
   */
  const FAMILIES = [
    { id: 'baseline', name: 'Baselines', blurb: 'Reference points: no learning, or learning with no exploration.' },
    { id: 'undirected', name: 'Undirected exploration', blurb: 'Explore by injecting randomness that ignores how uncertain each arm is.' },
    { id: 'optimism', name: 'Optimism in the face of uncertainty', blurb: 'Act greedily on an optimistic estimate: mean plus a bonus that shrinks with pulls.' },
    { id: 'elimination', name: 'Elimination', blurb: 'Sample the surviving arms evenly and discard any arm that is confidently worse than the leader.' },
    { id: 'divergence', name: 'Minimum empirical divergence', blurb: 'Pick the arm that is cheapest to believe could be best, measured by KL divergence.' },
    { id: 'bayesian', name: 'Bayesian / posterior', blurb: 'Keep a posterior over each mean; sample from it, read off a quantile, or value the information a pull buys.' },
    { id: 'perturbation', name: 'Randomized perturbation', blurb: 'Add calibrated random noise to the estimates, then act greedily. Thompson-like, with no posterior.' },
    { id: 'gradient', name: 'Policy gradient & learning automata', blurb: 'Adjust a probability distribution over arms directly instead of estimating values.' },
    { id: 'adversarial', name: 'Adversarial', blurb: 'No stochastic assumptions; guarantees hold even if rewards are chosen by an opponent.' },
    { id: 'nonstationary', name: 'Non-stationary', blurb: 'Forget old evidence so the agent can track arms whose means move.' },
    { id: 'contextual', name: 'Contextual', blurb: 'Read a context vector each round and learn how each arm\'s reward depends on it.' },
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

  // ------------------------------------------------- more algorithms ----
  function normPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
  /** Standard normal CDF via a Chebyshev fit to erfc (|err| < 1.2e-7). */
  function normCdf(x) {
    const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r / 2 : r / 2;
  }
  function logPlus(x) { return x > 1 ? Math.log(x) : 0; }
  /** KL upper confidence index for Bernoulli or unit-variance Gaussian arms. */
  function klIndex(bern, mean, n, bound, s2) {
    if (bound <= 0) return mean;
    return bern ? klUcbBern(mean, n, bound) : mean + Math.sqrt((2 * s2 * bound) / n);
  }
  /** Posterior mean and variance: Beta(1,1) prior for Bernoulli, N(0,1) prior with known noise for Gaussian. */
  function posterior(bern, s2, sum, n) {
    if (bern) {
      const al = 1 + sum, be = 1 + n - sum, ab = al + be;
      return [al / ab, (al * be) / (ab * ab * (ab + 1))];
    }
    return [sum / (n + 1), s2 / (n + 1)];
  }
  function samplePosterior(rng, bern, s2, sum, n) {
    return bern ? rng.beta(1 + sum, 1 + n - sum) : sum / (n + 1) + Math.sqrt(s2 / (n + 1)) * rng.normal();
  }
  /** Sum of m fair coin flips. */
  function binomHalf(rng, m) {
    if (m <= 40) { let s = 0; for (let i = 0; i < m; i++) if (rng.uniform() < 0.5) s++; return s; }
    return Math.min(m, Math.max(0, Math.round(m / 2 + 0.5 * Math.sqrt(m) * rng.normal())));
  }
  /** Brezzi & Lai (2002) approximation to the Gittins boundary function. */
  function gittinsPsi(s) {
    if (s <= 0.2) return Math.sqrt(s / 2);
    if (s <= 1) return 0.49 - 0.11 / Math.sqrt(s);
    if (s <= 5) return 0.63 - 0.26 / Math.sqrt(s);
    if (s <= 15) return 0.77 - 0.58 / Math.sqrt(s);
    return Math.sqrt(Math.max(0, 2 * Math.log(s) - Math.log(Math.log(s)) - Math.log(16 * Math.PI)));
  }
  function exp3Gamma(K, T) { return Math.min(1, Math.sqrt((K * Math.log(K)) / ((Math.E - 1) * T))); }

  ALGORITHMS.push(
    // ---- undirected
    {
      id: 'softmax-decay', name: 'Annealed softmax', family: 'undirected',
      params: [{ key: 'tau0', label: 'τ₀', default: 1, min: 0.01, max: 20, step: 0.01 }],
      formula: 'P(a) ∝ exp(Q̂ₐ / τₜ),   τₜ = τ₀ / ln(t + 1)',
      regret: 'Depends on the schedule; no general log T bound',
      idea: 'Starts close to uniform and cools toward greedy as evidence accumulates.',
      create(K, { rng, params, env }) {
        const s = stats(K), p = new Float64Array(K), c = 1 / (params.tau0 * env.scale);
        return {
          select: (t) => sampleIndex(softmaxInto(p, s.mean, c * Math.log(t + 1)), rng),
          update: (a, r) => record(s, a, r),
        };
      },
    },
    // ---- optimism
    {
      id: 'ucb2', name: 'UCB2', family: 'optimism',
      params: [{ key: 'alpha', label: 'α', default: 0.1, min: 0.001, max: 1, step: 0.001 }],
      formula: 'argmaxₐ Q̂ₐ + √((1+α) ln(e·t / τ(rₐ)) / 2τ(rₐ)),  played for an epoch,  τ(r) = ⌈(1+α)^r⌉',
      regret: 'O(Σ log T / Δₐ), leading constant near optimal for small α',
      idea: 'Commits to the chosen arm for geometrically growing epochs, so it rarely switches arms.',
      create(K, { rng, params, env }) {
        const s = stats(K), r = new Int32Array(K), idx = new Float64Array(K), al = params.alpha, sc = env.scale;
        const tau = (x) => Math.ceil(Math.pow(1 + al, x));
        let cur = -1, left = 0;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            if (left > 0) { left--; return cur; }
            for (let a = 0; a < K; a++) {
              const ta = tau(r[a]);
              idx[a] = s.mean[a] + sc * Math.sqrt(((1 + al) * Math.max(0, Math.log((Math.E * t) / ta))) / (2 * ta));
            }
            cur = argmax(idx, rng);
            left = Math.max(1, tau(r[cur] + 1) - tau(r[cur])) - 1;
            r[cur]++;
            return cur;
          },
          update: (a, x) => record(s, a, x),
        };
      },
    },
    {
      id: 'ucb-v', name: 'UCB-V', family: 'optimism',
      params: [],
      formula: 'argmaxₐ Q̂ₐ + √(2 V̂ₐ ζ ln t / nₐ) + 3bζ ln t / nₐ,   ζ = 1.2',
      regret: 'O(Σ (σₐ²/Δₐ + b) log T)',
      idea: 'Uses an empirical Bernstein bound, so arms with low observed variance get small bonuses.',
      create(K, { rng, env }) {
        const s = stats(K), sq = new Float64Array(K), idx = new Float64Array(K), z = 1.2, b = env.scale;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            const lt = Math.log(t);
            for (let a = 0; a < K; a++) {
              const n = s.n[a], m = s.mean[a], v = Math.max(0, sq[a] / n - m * m);
              idx[a] = m + Math.sqrt((2 * v * z * lt) / n) + (3 * b * z * lt) / n;
            }
            return argmax(idx, rng);
          },
          update(a, r) { record(s, a, r); sq[a] += r * r; },
        };
      },
    },
    {
      id: 'ucb-normal', name: 'UCB1-Normal', family: 'optimism',
      params: [],
      formula: 'play any arm with nₐ < ⌈8 ln t⌉;  else argmaxₐ Q̂ₐ + √(16 σ̂ₐ² ln(t−1) / nₐ)',
      regret: 'O(Σ σ² log T / Δₐ) for Gaussian rewards',
      idea: 'Auer et al.\'s variant for Gaussian rewards with unknown variance. Its forced sampling is expensive at short horizons.',
      create(K, { rng }) {
        const s = stats(K), sq = new Float64Array(K), idx = new Float64Array(K);
        return {
          select(t) {
            const need = Math.max(2, Math.ceil(8 * Math.log(t)));
            for (let a = 0; a < K; a++) if (s.n[a] < need) return a;
            const lt = Math.log(Math.max(2, t - 1));
            for (let a = 0; a < K; a++) {
              const n = s.n[a], m = s.mean[a], v = Math.max(0, (sq[a] - n * m * m) / (n - 1));
              idx[a] = m + Math.sqrt((16 * v * lt) / n);
            }
            return argmax(idx, rng);
          },
          update(a, r) { record(s, a, r); sq[a] += r * r; },
        };
      },
    },
    {
      id: 'improved-ucb', name: 'Improved UCB', family: 'elimination',
      params: [],
      formula: 'phase m: Δ̃ = 2⁻ᵐ; sample survivors to ⌈2 ln(TΔ̃²)/Δ̃²⌉; drop arms whose UCB < best LCB',
      regret: 'O(Σ log(TΔₐ²) / Δₐ); needs T',
      idea: 'Auer & Ortner\'s phased elimination: halve the gap it can resolve each phase and discard arms that are clearly worse.',
      create(K, { rng, T, env }) {
        const s = stats(K), active = new Uint8Array(K).fill(1), sc = env.scale;
        let D = 1, nActive = K, done = false, rr = 0, target = 0, choice = -1;
        const setTarget = () => { target = Math.ceil((2 * Math.log(Math.max(Math.E, T * D * D))) / (D * D)); };
        setTarget();
        function select() {
          while (!done) {
            for (let i = 0; i < K; i++) {
              const a = (rr + i) % K;
              if (active[a] && s.n[a] < target) { rr = (a + 1) % K; return a; }
            }
            const w = sc * Math.sqrt(Math.log(Math.max(Math.E, T * D * D)) / (2 * target));
            let best = -Infinity;
            for (let a = 0; a < K; a++) if (active[a] && s.mean[a] - w > best) best = s.mean[a] - w;
            for (let a = 0; a < K; a++) if (active[a] && nActive > 1 && s.mean[a] + w < best) { active[a] = 0; nActive--; }
            D /= 2;
            if (nActive === 1 || D < Math.sqrt(Math.E / T)) done = true; else setTarget();
          }
          if (choice < 0) {
            let bv = -Infinity;
            for (let a = 0; a < K; a++) if (active[a] && s.mean[a] > bv) { bv = s.mean[a]; choice = a; }
          }
          return choice;
        }
        return { select, update: (a, r) => record(s, a, r) };
      },
    },
    {
      id: 'klucb-plus', name: 'KL-UCB+', family: 'optimism',
      params: [],
      formula: 'max { q : nₐ·kl(Q̂ₐ, q) ≤ ln(t / nₐ) }',
      regret: 'Asymptotically optimal; usually below KL-UCB in practice',
      idea: 'Dividing t by the arm\'s own count shrinks the exploration rate for arms that are already well sampled.',
      create(K, { rng, env }) {
        const s = stats(K), idx = new Float64Array(K), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            for (let a = 0; a < K; a++) idx[a] = klIndex(bern, s.mean[a], s.n[a], logPlus(t / s.n[a]), s2);
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'klucb-pp', name: 'KL-UCB++', family: 'optimism',
      params: [],
      formula: 'max { q : nₐ·kl(Q̂ₐ, q) ≤ ln₊( T/(K nₐ) · (ln₊²(T/(K nₐ)) + 1) ) }',
      regret: 'Asymptotically optimal and minimax optimal; needs T',
      idea: 'Ménard & Garivier\'s fusion of KL-UCB with the MOSS exploration function: best of both regimes.',
      create(K, { rng, env, T }) {
        const s = stats(K), idx = new Float64Array(K), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select() {
            const u = untried(s.n); if (u >= 0) return u;
            for (let a = 0; a < K; a++) {
              const x = T / (K * s.n[a]), l = logPlus(x);
              idx[a] = klIndex(bern, s.mean[a], s.n[a], logPlus(x * (l * l + 1)), s2);
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'moss-anytime', name: 'MOSS-anytime', family: 'optimism',
      params: [{ key: 'alpha', label: 'α', default: 1.35, min: 0, max: 5, step: 0.05 }],
      formula: 'argmaxₐ Q̂ₐ + s·√( (1+α)/2 · ln₊(t / (K nₐ)) / nₐ )',
      regret: 'O(√(KT)) without knowing T',
      idea: 'Degenne & Perchet\'s version of MOSS that replaces the horizon with the current step.',
      create(K, { rng, params, env }) {
        const s = stats(K), idx = new Float64Array(K), c = (1 + params.alpha) / 2, sc = env.scale;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            for (let a = 0; a < K; a++) idx[a] = s.mean[a] + sc * Math.sqrt((c * logPlus(t / (K * s.n[a]))) / s.n[a]);
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    // ---- elimination
    {
      id: 'successive-elim', name: 'Successive elimination', family: 'elimination',
      params: [],
      formula: 'round-robin over survivors;  drop a if Q̂ₐ + w < max Q̂ − w,   w = s·√(2 ln T / r)',
      regret: 'O(Σ log T / Δₐ);  O(√(KT log T)) worst case',
      idea: 'Even-Dar et al.\'s action elimination: never pulls an arm again once it is confidently worse. Its confidence widths are conservative, so it needs long horizons.',
      create(K, { T, env }) {
        const s = stats(K), active = new Uint8Array(K).fill(1), sc = env.scale, LT = Math.log(T);
        let nActive = K, rr = 0;
        return {
          select() {
            let minN = Infinity, maxN = -1;
            for (let a = 0; a < K; a++) if (active[a]) { minN = Math.min(minN, s.n[a]); maxN = Math.max(maxN, s.n[a]); }
            if (minN === maxN && minN > 0 && nActive > 1) {
              const w = sc * Math.sqrt((2 * LT) / minN);
              let best = -Infinity;
              for (let a = 0; a < K; a++) if (active[a] && s.mean[a] > best) best = s.mean[a];
              for (let a = 0; a < K; a++) if (active[a] && nActive > 1 && s.mean[a] + w < best - w) { active[a] = 0; nActive--; }
              minN = Infinity;
              for (let a = 0; a < K; a++) if (active[a]) minN = Math.min(minN, s.n[a]);
            }
            for (let i = 0; i < K; i++) {
              const a = (rr + i) % K;
              if (active[a] && s.n[a] === minN) { rr = (a + 1) % K; return a; }
            }
            return rr;
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'seq-halving', name: 'Sequential halving + commit', family: 'elimination',
      params: [{ key: 'budget', label: 'budget (share of T)', default: 0.2, min: 0.01, max: 1, step: 0.01 }],
      formula: '⌈log₂K⌉ rounds, each splits the budget evenly over survivors and keeps the top half; then commit',
      regret: 'A best-arm identification method: cost ≈ budget, then a commit that may be wrong',
      idea: 'Karnin et al.\'s fixed-budget identification, used here as an explore-then-commit strategy.',
      create(K, { rng, T, params }) {
        const R = Math.max(1, Math.ceil(Math.log2(K))), B = Math.max(K, Math.floor(params.budget * T));
        const rsum = new Float64Array(K), rn = new Float64Array(K);
        let alive = Array.from({ length: K }, (_, i) => i), queue = [], qi = 0, committed = -1;
        const plan = () => {
          const per = Math.max(1, Math.floor(B / (alive.length * R)));
          queue = []; qi = 0; rsum.fill(0); rn.fill(0);
          for (let j = 0; j < per; j++) for (const a of alive) queue.push(a);
        };
        plan();
        return {
          select() {
            if (committed >= 0) return committed;
            if (qi >= queue.length) {
              const key = alive.map((a) => [a, rsum[a] / rn[a] + 1e-9 * rng.uniform()]).sort((x, y) => y[1] - x[1]);
              alive = key.slice(0, Math.ceil(alive.length / 2)).map((x) => x[0]);
              if (alive.length === 1) return (committed = alive[0]);
              plan();
            }
            return queue[qi++];
          },
          update(a, r) { rsum[a] += r; rn[a] += 1; },
        };
      },
    },
    // ---- minimum empirical divergence
    {
      id: 'imed', name: 'IMED', family: 'divergence',
      params: [],
      formula: 'argminₐ  nₐ·KL(Q̂ₐ, max Q̂) + ln nₐ',
      regret: 'Asymptotically optimal (Honda & Takemura)',
      idea: 'Deterministic like UCB but with no confidence level to tune: it balances how unlikely an arm is to be best against how rarely it has been tried.',
      create(K, { rng, env }) {
        const s = stats(K), idx = new Float64Array(K), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select() {
            const u = untried(s.n); if (u >= 0) return u;
            let best = -Infinity;
            for (let a = 0; a < K; a++) if (s.mean[a] > best) best = s.mean[a];
            for (let a = 0; a < K; a++) {
              const d = bern ? klBern(s.mean[a], best) : ((best - s.mean[a]) ** 2) / (2 * s2);
              idx[a] = -(s.n[a] * d + Math.log(s.n[a]));
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    // ---- bayesian
    {
      id: 'gittins', name: 'Gittins index (approx.)', family: 'bayesian',
      params: [{ key: 'gamma', label: 'γ (0 = 1 − 1/T)', default: 0, min: 0, max: 0.99999, step: 0.0001 }],
      formula: 'argmaxₐ  μₐ + σₐ·ψ(σₐ² / (σ² ln(1/γ)))   (Brezzi–Lai approximation)',
      regret: 'Exact index is Bayes-optimal for discounted reward; no frequentist bound',
      idea: 'The Gittins index solves the discounted Bayesian bandit exactly. This uses Brezzi & Lai\'s closed-form approximation.',
      create(K, { rng, params, env, T }) {
        const s = stats(K), idx = new Float64Array(K), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        const g = params.gamma > 0 ? params.gamma : 1 - 1 / T, c = -Math.log(g);
        return {
          select() {
            for (let a = 0; a < K; a++) {
              const [m, v] = posterior(bern, s2, s.sum[a], s.n[a]);
              const noise = bern ? Math.max(0.01, m * (1 - m)) : s2;
              idx[a] = m + Math.sqrt(v) * gittinsPsi(v / (noise * c));
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'kg', name: 'Knowledge gradient', family: 'bayesian',
      params: [],
      formula: 'argmaxₐ  μₐ + (T − t)·σ̃ₐ f(−|μₐ − maxᵦ≠ₐ μᵦ| / σ̃ₐ),   f(z) = zΦ(z) + φ(z)',
      regret: 'No regret bound; a one-step lookahead policy',
      idea: 'Ryzhov, Powell & Frazier\'s online KG: pay for a pull with its immediate reward plus the value of what it teaches, times the steps left.',
      create(K, { rng, env, T }) {
        const s = stats(K), idx = new Float64Array(K), mu = new Float64Array(K), va = new Float64Array(K);
        const bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select(t) {
            let b1 = -Infinity, b2 = -Infinity, i1 = -1;
            for (let a = 0; a < K; a++) {
              const [m, v] = posterior(bern, s2, s.sum[a], s.n[a]);
              mu[a] = m; va[a] = v;
              if (m > b1) { b2 = b1; b1 = m; i1 = a; } else if (m > b2) b2 = m;
            }
            for (let a = 0; a < K; a++) {
              const noise = bern ? Math.max(0.01, mu[a] * (1 - mu[a])) : s2;
              const vNext = 1 / (1 / va[a] + 1 / noise);
              const st = Math.sqrt(Math.max(1e-18, va[a] - vNext));
              const other = a === i1 ? b2 : b1;
              const z = -Math.abs(mu[a] - other) / st;
              idx[a] = mu[a] + (T - t) * st * (z * normCdf(z) + normPdf(z));
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'ids', name: 'Information-directed sampling', family: 'bayesian',
      params: [{ key: 'M', label: 'posterior samples', default: 50, min: 10, max: 1000, step: 10 }],
      formula: 'π = argmin over distributions of  (E_π Δ)² / E_π g,   Δ = regret, g = information gain about a*',
      regret: 'Bayesian regret O(√(KT log K))',
      idea: 'Russo & Van Roy: choose the action mix with the best ratio of squared regret to information. Estimated from posterior samples, so it is the slowest algorithm here.',
      create(K, { rng, env, params }) {
        const s = stats(K), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma, M = Math.max(10, Math.round(params.M));
        const th = new Float64Array(K), cnt = new Float64Array(K), cond = new Float64Array(K * K);
        const mu = new Float64Array(K), D = new Float64Array(K), G = new Float64Array(K);
        return {
          select() {
            cnt.fill(0); cond.fill(0); mu.fill(0);
            let rho = 0;
            for (let m = 0; m < M; m++) {
              let best = 0, bv = -Infinity;
              for (let a = 0; a < K; a++) {
                const x = samplePosterior(rng, bern, s2, s.sum[a], s.n[a]);
                th[a] = x; mu[a] += x;
                if (x > bv) { bv = x; best = a; }
              }
              cnt[best]++; rho += bv;
              const row = best * K;
              for (let a = 0; a < K; a++) cond[row + a] += th[a];
            }
            rho /= M;
            for (let a = 0; a < K; a++) mu[a] /= M;
            for (let a = 0; a < K; a++) {
              D[a] = Math.max(0, rho - mu[a]);
              let g = 0;
              for (let b = 0; b < K; b++) {
                if (!cnt[b]) continue;
                const d = cond[b * K + a] / cnt[b] - mu[a];
                g += (cnt[b] / M) * d * d;
              }
              G[a] = g;
            }
            const ratio = (d, g) => (d <= 1e-12 ? 0 : g <= 1e-15 ? Infinity : (d * d) / g);
            let i = 0, bestR = Infinity;
            for (let a = 0; a < K; a++) { const r = ratio(D[a], G[a]); if (r < bestR) { bestR = r; i = a; } }
            let j = i, q = 1;
            for (let b = 0; b < K; b++) {
              if (b === i) continue;
              for (let k = 1; k < 20; k++) {
                const w = k / 20, r = ratio(w * D[i] + (1 - w) * D[b], w * G[i] + (1 - w) * G[b]);
                if (r < bestR) { bestR = r; j = b; q = w; }
              }
            }
            return rng.uniform() < q ? i : j;
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    // ---- randomized perturbation
    {
      id: 'phe', name: 'Perturbed-history exploration', family: 'perturbation',
      params: [{ key: 'a', label: 'a (pseudo-reward ratio)', default: 1.1, min: 0.1, max: 10, step: 0.1 }],
      formula: 'argmaxₐ (Σ rewards + Σ ⌈a·nₐ⌉ pseudo-rewards) / (nₐ + ⌈a·nₐ⌉)',
      regret: 'O(Σ log T / Δₐ) for Bernoulli rewards with a > 1',
      idea: 'Kveton et al.: mix in fresh coin-flip pseudo-rewards every round. The noise does the exploring and no posterior is needed.',
      create(K, { rng, params, env }) {
        const s = stats(K), idx = new Float64Array(K), bern = env.type === 'bernoulli', sig = env.sigma;
        return {
          select() {
            const u = untried(s.n); if (u >= 0) return u;
            for (let a = 0; a < K; a++) {
              const n = s.n[a], m = Math.ceil(params.a * n);
              const pseudo = bern ? binomHalf(rng, m) : Math.sqrt(m) * sig * rng.normal();
              idx[a] = (s.sum[a] + pseudo) / (n + m);
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'bge', name: 'Boltzmann–Gumbel', family: 'perturbation',
      params: [{ key: 'C', label: 'C', default: 0.2, min: 0.01, max: 10, step: 0.01 }],
      formula: 'argmaxₐ  Q̂ₐ + C·s·Zₐ / √nₐ,   Zₐ ~ Gumbel(0, 1)',
      regret: 'O(Σ log² T / Δₐ)',
      idea: 'Cesa-Bianchi et al.\'s fix for softmax: each arm gets its own temperature, shrinking with its pull count.',
      create(K, { rng, params, env }) {
        const s = stats(K), idx = new Float64Array(K), c = params.C * env.scale;
        return {
          select() {
            const u = untried(s.n); if (u >= 0) return u;
            for (let a = 0; a < K; a++) {
              const z = -Math.log(-Math.log(rng.uniform() || 1e-300) || 1e-300);
              idx[a] = s.mean[a] + (c * z) / Math.sqrt(s.n[a]);
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    // ---- learning automata
    {
      id: 'pursuit', name: 'Pursuit', family: 'gradient',
      params: [{ key: 'beta', label: 'β (pursuit rate)', default: 0.01, min: 0.0005, max: 0.5, step: 0.0005 }],
      formula: 'π ← π + β(e_greedy − π);   sample aₜ ~ π',
      regret: 'ε-optimal for small β; no log T bound',
      idea: 'A learning automaton that moves its action probabilities a little toward the current greedy arm each step.',
      create(K, { rng, params }) {
        const s = stats(K), p = new Float64Array(K).fill(1 / K), b = params.beta;
        return {
          select: () => sampleIndex(p, rng),
          update(a, r) {
            record(s, a, r);
            const g = argmax(s.mean, rng);
            for (let i = 0; i < K; i++) p[i] += b * ((i === g ? 1 : 0) - p[i]);
          },
        };
      },
    },
    {
      id: 'lri', name: 'Linear reward-inaction', family: 'gradient',
      params: [{ key: 'lambda', label: 'λ (learning rate)', default: 0.01, min: 0.0005, max: 0.5, step: 0.0005 }],
      formula: 'on reward r̃ ∈ [0, 1]:  π ← π + λ r̃ (e_aₜ − π);   no change on failure',
      regret: 'ε-optimal for small λ; can lock onto a wrong arm',
      idea: 'The classic L_R-I automaton from learning-automata theory: reinforce what paid, ignore what did not.',
      create(K, { rng, params, env }) {
        const p = new Float64Array(K).fill(1 / K), lam = params.lambda;
        return {
          select: () => sampleIndex(p, rng),
          update(a, r) {
            const x = lam * env.normalize(r);
            for (let i = 0; i < K; i++) p[i] += x * ((i === a ? 1 : 0) - p[i]);
          },
        };
      },
    },
    {
      id: 'reinforcement-comparison', name: 'Reinforcement comparison', family: 'gradient',
      params: [
        { key: 'beta', label: 'β (preference step)', default: 0.1, min: 0.005, max: 2, step: 0.005 },
        { key: 'alpha', label: 'α (reference step)', default: 0.1, min: 0.005, max: 1, step: 0.005 },
      ],
      formula: 'h_aₜ ← h_aₜ + β(r − r̄),   r̄ ← r̄ + α(r − r̄),   π = softmax(h)',
      regret: 'No regret bound',
      idea: 'Sutton & Barto\'s precursor to the gradient bandit: compare each reward with a running reference level.',
      create(K, { rng, params }) {
        const h = new Float64Array(K), p = new Float64Array(K);
        let ref = 0;
        return {
          select: () => sampleIndex(softmaxInto(p, h, 1), rng),
          update(a, r) { h[a] += params.beta * (r - ref); ref += params.alpha * (r - ref); },
        };
      },
    },
    // ---- adversarial
    {
      id: 'exp3-ix', name: 'EXP3-IX', family: 'adversarial',
      params: [],
      formula: 'π ∝ exp(−η L̂),   L̂ₐ += ℓ / (πₐ + γ),   η = √(2 ln K / KT),  γ = η/2',
      regret: 'O(√(KT log K)) with high probability',
      idea: 'Neu\'s implicit exploration: a small bias in the loss estimate replaces forced uniform exploration and tames the variance.',
      create(K, { rng, env, T }) {
        const lw = new Float64Array(K), p = new Float64Array(K);
        const eta = Math.sqrt((2 * Math.log(K)) / (K * T)), gamma = eta / 2;
        return {
          select: () => sampleIndex(softmaxInto(p, lw, 1), rng),
          update(a, r) { lw[a] -= (eta * (1 - env.normalize(r))) / (p[a] + gamma); },
        };
      },
    },
    {
      id: 'exp3p', name: 'EXP3.P', family: 'adversarial',
      params: [],
      formula: 'wᵢ ← wᵢ·exp(γ/3K · (x̂ᵢ + α / (πᵢ√(KT)))),   π = (1−γ)w/Σw + γ/K',
      regret: 'O(√(KT log(K/δ))) with probability 1 − δ',
      idea: 'Auer et al. add an optimistic bonus to every arm\'s weight, turning EXP3\'s expected bound into a high-probability one.',
      create(K, { rng, env, T }) {
        const lw = new Float64Array(K), p = new Float64Array(K);
        const gamma = Math.min(0.6, 2 * Math.sqrt((0.6 * K * Math.log(K)) / T));
        const alpha = 2 * Math.sqrt(Math.log((K * T) / 0.1));
        const bonus = alpha / Math.sqrt(K * T);
        return {
          select() {
            softmaxInto(p, lw, 1);
            for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma / K;
            return sampleIndex(p, rng);
          },
          update(a, r) {
            const x = env.normalize(r);
            for (let i = 0; i < K; i++) lw[i] += (gamma / (3 * K)) * ((i === a ? x / p[i] : 0) + bonus / p[i]);
          },
        };
      },
    },
    {
      id: 'tsallis-inf', name: 'Tsallis-INF', family: 'adversarial',
      params: [],
      formula: 'πₐ = 4 / (ηₜ(L̂ₐ − x))²,  x normalizes π,  ηₜ = 2/√t',
      regret: 'Best of both worlds: O(√(KT)) adversarial, O(Σ log T / Δₐ) stochastic',
      idea: 'Zimmert & Seldin\'s FTRL with the ½-Tsallis entropy. One algorithm, optimal against both stochastic and adversarial rewards.',
      create(K, { rng, env }) {
        const L = new Float64Array(K), p = new Float64Array(K);
        return {
          select(t) {
            const eta = 2 / Math.sqrt(t);
            let minL = Infinity;
            for (let a = 0; a < K; a++) if (L[a] < minL) minL = L[a];
            let x = minL - 2 / eta;
            for (let it = 0; it < 50; it++) {
              let S = 0, S15 = 0;
              for (let a = 0; a < K; a++) { const w = 4 / ((eta * (L[a] - x)) ** 2); S += w; S15 += w ** 1.5; }
              if (Math.abs(S - 1) < 1e-12) break;
              x -= (S - 1) / (eta * S15);
            }
            let S = 0;
            for (let a = 0; a < K; a++) { p[a] = 4 / ((eta * (L[a] - x)) ** 2); S += p[a]; }
            for (let a = 0; a < K; a++) p[a] /= S;
            return sampleIndex(p, rng);
          },
          update(a, r) { L[a] += (1 - env.normalize(r)) / p[a]; },
        };
      },
    },
    // ---- non-stationary
    {
      id: 'egreedy-const', name: 'ε-greedy, constant step', family: 'nonstationary',
      params: [
        { key: 'epsilon', label: 'ε', default: 0.1, min: 0, max: 1, step: 0.01 },
        { key: 'alpha', label: 'α (step size)', default: 0.1, min: 0.01, max: 1, step: 0.01 },
      ],
      formula: 'Q̂ₐ ← Q̂ₐ + α(r − Q̂ₐ);  random arm w.p. ε, else argmax Q̂',
      regret: 'Θ(T); tracks drift',
      idea: 'Sutton & Barto\'s recipe for non-stationary problems: a constant step size weights recent rewards exponentially more.',
      create(K, { rng, params }) {
        const q = new Float64Array(K), eps = params.epsilon, al = params.alpha;
        return {
          select: () => (rng.uniform() < eps ? rng.int(K) : argmax(q, rng)),
          update(a, r) { q[a] += al * (r - q[a]); },
        };
      },
    },
    {
      id: 'd-ucb', name: 'Discounted UCB', family: 'nonstationary',
      params: [
        { key: 'gamma', label: 'γ (discount)', default: 0.99, min: 0.5, max: 1, step: 0.001 },
        { key: 'c', label: 'c', default: 1, min: 0, max: 5, step: 0.01 },
      ],
      formula: 'argmaxₐ  X̄ₐ(γ) + c·s·√(ln nₜ(γ) / Nₐ(γ)),  all statistics discounted by γ',
      regret: 'Õ(√(ΓT)) with Γ change points',
      idea: 'Garivier & Moulines: UCB on exponentially discounted counts and sums, so old evidence fades.',
      create(K, { rng, params, env }) {
        const n = new Float64Array(K), sum = new Float64Array(K), pulls = new Float64Array(K), idx = new Float64Array(K);
        const g = params.gamma, c = params.c * env.scale;
        return {
          select() {
            const u = untried(pulls); if (u >= 0) return u;
            let nt = 0;
            for (let a = 0; a < K; a++) nt += n[a];
            const l = Math.log(Math.max(1, nt));
            for (let a = 0; a < K; a++) idx[a] = sum[a] / n[a] + c * Math.sqrt(l / n[a]);
            return argmax(idx, rng);
          },
          update(a, r) {
            for (let i = 0; i < K; i++) { n[i] *= g; sum[i] *= g; }
            n[a] += 1; sum[a] += r; pulls[a] += 1;
          },
        };
      },
    },
    {
      id: 'm-ucb', name: 'Monitored UCB', family: 'nonstationary',
      params: [
        { key: 'w', label: 'w (detection window)', default: 100, min: 10, max: 2000, step: 10 },
        { key: 'gamma', label: 'γ (forced exploration)', default: 0.05, min: 0, max: 0.5, step: 0.01 },
      ],
      formula: 'UCB1 since the last reset;  reset all arms when |Σ second half − Σ first half| of an arm\'s last w rewards > b',
      regret: 'O(√(ΓKT log T)) with Γ change points',
      idea: 'Cao et al.: pair UCB with a change detector and a trickle of forced exploration; restart from scratch when an arm shifts.',
      create(K, { rng, params, env, T }) {
        const W = Math.max(2, 2 * Math.round(params.w / 2)), half = W / 2;
        const b = Math.sqrt((W / 2) * Math.log(2 * K * T * T)) * (env.type === 'bernoulli' ? 1 : 2 * env.sigma);
        const period = params.gamma > 0 ? Math.max(K, Math.floor(K / params.gamma)) : 0;
        const buf = new Float64Array(K * W), cnt = new Int32Array(K), head = new Int32Array(K);
        const idx = new Float64Array(K), c = Math.SQRT2 * env.scale;
        let s = stats(K), tau = 0;
        return {
          select(t) {
            if (period) { const ph = (t - tau - 1) % period; if (ph < K) return ph; }
            const u = untried(s.n); if (u >= 0) return u;
            const lt = Math.log(Math.max(2, t - tau));
            for (let a = 0; a < K; a++) idx[a] = s.mean[a] + c * Math.sqrt(lt / s.n[a]);
            return argmax(idx, rng);
          },
          update(a, r, t) {
            record(s, a, r);
            const base = a * W;
            buf[base + head[a]] = r; head[a] = (head[a] + 1) % W;
            if (cnt[a] < W) cnt[a]++;
            if (cnt[a] === W) {
              let s1 = 0, s2 = 0;
              for (let i = 0; i < W; i++) {
                const v = buf[base + ((head[a] + i) % W)];
                if (i < half) s1 += v; else s2 += v;
              }
              if (Math.abs(s2 - s1) > b) { s = stats(K); cnt.fill(0); head.fill(0); tau = t; }
            }
          },
        };
      },
    },
    {
      id: 'exp3s', name: 'EXP3.S', family: 'nonstationary',
      params: [],
      formula: 'wᵢ ← wᵢ·exp(γ x̂ᵢ / K) + (eα/K)·Σⱼ wⱼ,   α = 1/T',
      regret: 'O(√(S K T log(KT))) against the best sequence with S switches',
      idea: 'Auer et al.\'s fixed-share EXP3: every arm keeps a floor of weight, so it can win again after a switch.',
      create(K, { rng, env, T }) {
        const w = new Float64Array(K).fill(1 / K), p = new Float64Array(K);
        const alpha = 1 / T, gamma = Math.min(1, Math.sqrt((K * Math.log(K * T)) / T));
        return {
          select() {
            let W = 0;
            for (let a = 0; a < K; a++) W += w[a];
            for (let a = 0; a < K; a++) p[a] = ((1 - gamma) * w[a]) / W + gamma / K;
            return sampleIndex(p, rng);
          },
          update(a, r) {
            let W = 0;
            for (let i = 0; i < K; i++) W += w[i];
            const share = (Math.E * alpha * W) / K;
            w[a] *= Math.exp((gamma * env.normalize(r)) / p[a] / K);
            let S = 0;
            for (let i = 0; i < K; i++) { w[i] += share; S += w[i]; }
            for (let i = 0; i < K; i++) w[i] /= S;
          },
        };
      },
    },
    {
      id: 'rexp3', name: 'Rexp3', family: 'nonstationary',
      params: [{ key: 'batch', label: 'restart every (steps)', default: 250, min: 10, max: 10000, step: 10 }],
      formula: 'run EXP3 (γ tuned to the batch) and restart it every Δ steps',
      regret: 'O((K log K)^⅓ V^⅓ T^⅔) with variation budget V',
      idea: 'Besbes, Gur & Zeevi: periodic restarts are enough to bound regret when the total drift is bounded.',
      create(K, { rng, params, env }) {
        const Bn = Math.max(1, Math.round(params.batch)), lw = new Float64Array(K), p = new Float64Array(K);
        const gamma = exp3Gamma(K, Bn);
        return {
          select(t) {
            if ((t - 1) % Bn === 0) lw.fill(0);
            softmaxInto(p, lw, 1);
            for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma / K;
            return sampleIndex(p, rng);
          },
          update(a, r) { lw[a] += (gamma * env.normalize(r)) / (K * p[a]); },
        };
      },
    },
  );

  ALGORITHMS.push({
    id: 'linucb', name: 'LinUCB', family: 'contextual',
    params: [{ key: 'alpha', label: 'α (exploration)', default: 1, min: 0, max: 5, step: 0.05 }],
    formula: 'argmaxₐ  θ̂ₐ·xₜ + α√(xₜᵀ Aₐ⁻¹ xₜ),   Aₐ = I + Σ x xᵀ,  θ̂ₐ = Aₐ⁻¹ Σ r x',
    regret: 'Õ(d√(KT)) for linear rewards',
    idea: 'Li et al.\'s disjoint LinUCB: one ridge regression per arm on the context, plus an optimistic bonus from the regression\'s uncertainty. With no context it reduces to a ridge-regression UCB.',
    create(K, { rng, params, d = 1 }) {
      const Ainv = new Float64Array(K * d * d), bv = new Float64Array(K * d), idx = new Float64Array(K);
      const Ax = new Float64Array(d), x = new Float64Array(d), fallback = new Float64Array(K * d);
      for (let a = 0; a < K; a++) { fallback[a * d] = 1; for (let i = 0; i < d; i++) Ainv[a * d * d + i * d + i] = 1; }
      let last = fallback;
      return {
        select(t, Xt) {
          Xt = Xt && Xt.length === K * d ? Xt : fallback;
          last = Xt;
          for (let a = 0; a < K; a++) {
            const o = a * d * d, row = a * d;
            let mean = 0, q = 0;
            for (let i = 0; i < d; i++) {
              let th = 0, ax = 0;
              for (let j = 0; j < d; j++) { th += Ainv[o + i * d + j] * bv[a * d + j]; ax += Ainv[o + i * d + j] * Xt[row + j]; }
              mean += th * Xt[row + i]; q += Xt[row + i] * ax;
            }
            idx[a] = mean + params.alpha * Math.sqrt(Math.max(0, q));
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          const o = a * d * d;
          for (let j = 0; j < d; j++) x[j] = last[a * d + j];
          // Sherman–Morrison: A⁻¹ ← A⁻¹ − (A⁻¹x)(A⁻¹x)ᵀ / (1 + xᵀA⁻¹x)
          let den = 1;
          for (let i = 0; i < d; i++) {
            let v = 0;
            for (let j = 0; j < d; j++) v += Ainv[o + i * d + j] * x[j];
            Ax[i] = v; den += x[i] * v;
          }
          for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Ainv[o + i * d + j] -= (Ax[i] * Ax[j]) / den;
          for (let i = 0; i < d; i++) bv[a * d + i] += r * x[i];
        },
      };
    },
  });

  // Keep each family's members together, in family order, for listings.
  const famOrder = (id) => FAMILIES.findIndex((f) => f.id === id);
  ALGORITHMS.sort((a, b) => famOrder(a.family) - famOrder(b.family));
  const ALGO_BY_ID = {};
  for (const a of ALGORITHMS) ALGO_BY_ID[a.id] = a;

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
    // means/rewards are indexed [t*K + a]; for rested environments they are indexed by the
    // arm's own pull count instead: [n*K + a] is the mean/reward of arm a's (n+1)-th pull.
    const means = new Float64Array(T * K), rewards = new Float64Array(T * K);
    const D = env.d || 1, contexts = new Float64Array(T * K * D), X = new Float64Array(K * D);
    const best = new Float64Array(T), avgMean = new Float64Array(K), rank = new Int32Array(K);
    const rested = !!env.rested, oracleCum = new Float64Array(T), pulls = new Int32Array(K);

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

    function restedOracle() {
      // Best of: greedy on current means (optimal when means only fall), and each fixed arm
      // (optimal when means only rise). Store the better policy's cumulative expected reward.
      const cnt = new Int32Array(K), greedy = new Float64Array(T);
      let acc = 0;
      for (let t = 0; t < T; t++) {
        let b = 0, bv = -Infinity;
        for (let a = 0; a < K; a++) { const m = means[cnt[a] * K + a]; if (m > bv) { bv = m; b = a; } }
        acc += bv; cnt[b]++; greedy[t] = acc;
      }
      let bestArm = -1, bestTot = greedy[T - 1];
      for (let a = 0; a < K; a++) {
        let tot = 0;
        for (let n = 0; n < T; n++) tot += means[n * K + a];
        if (tot > bestTot) { bestTot = tot; bestArm = a; }
      }
      if (bestArm < 0) { oracleCum.set(greedy); return; }
      acc = 0;
      for (let n = 0; n < T; n++) { acc += means[n * K + bestArm]; oracleCum[n] = acc; }
    }

    function runOne(run) {
      const envRng = Rng(hash32(seed, run, 'env'));
      contexts.fill(D > 1 ? 0 : 1);
      const info = env.fill(K, T, envRng, means, rewards, contexts) || {};
      avgMean.fill(0);
      if (rested) {
        restedOracle();
        for (let a = 0; a < K; a++) avgMean[a] = means[a];
      } else {
        for (let t = 0; t < T; t++) {
          let b = -Infinity;
          for (let a = 0; a < K; a++) {
            const m = means[t * K + a];
            if (m > b) b = m;
            avgMean[a] += m / T;
          }
          best[t] = b;
        }
      }
      // rank[arm] = 0 for the arm with the highest time-averaged (rested: initial) mean
      const order = Array.from({ length: K }, (_, i) => i).sort((x, y) => avgMean[y] - avgMean[x]);
      order.forEach((arm, r) => { rank[arm] = r; });
      const graph = info.graph || null;

      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const agent = slot.def.create(K, { T, env, params: slot.params, rng, d: D, info });
        const observe = graph && typeof agent.observe === 'function';
        const t0 = now();
        let cum = 0, got = 0;
        pulls.fill(0);
        const { regSum, regSq, optSum, rewSum, pullsByRank } = slot;
        for (let t = 0; t < T; t++) {
          const base = t * K * D;
          for (let i = 0; i < K * D; i++) X[i] = contexts[base + i];
          const a = agent.select(t + 1, X);
          let r, m, isBest;
          if (rested) {
            const n = pulls[a];
            r = rewards[n * K + a]; m = means[n * K + a];
            let bv = -Infinity;
            for (let b = 0; b < K; b++) { const mb = means[pulls[b] * K + b]; if (mb > bv) bv = mb; }
            isBest = m === bv;
            pulls[a]++;
            got += m;
            cum = oracleCum[t] - got;
          } else {
            r = rewards[t * K + a]; m = means[t * K + a];
            cum += best[t] - m;
            isBest = m === best[t];
          }
          agent.update(a, r, t + 1);
          if (observe) {
            for (let b = 0; b < K; b++) if (b !== a && graph[a * K + b]) agent.observe(b, rewards[t * K + b], t + 1);
          }
          regSum[t] += cum;
          regSq[t] += cum * cum;
          if (isBest) optSum[t] += 1;
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
          const pullShare = Array.from(s.pullsByRank, (x) => x / totalPulls);
          const msPerRun = done ? s.timeMs / done : 0;
          const summary = {
            finalRegret: regret[T - 1], finalSE: regretSE[T - 1],
            finalMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            optimalTail: optTail, bestShare: pullShare[0], meanReward: rewAll, msPerRun,
          };
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, params: s.params,
            runs: done, regret, regretSE, optimal, reward, pullShare,
            finalRegret: summary.finalRegret, finalSE: summary.finalSE, finalMedian: summary.finalMedian,
            finals: sorted, optimalTail: optTail, meanReward: rewAll, msPerRun,
            curves: { regret, optimal, reward }, curvesSE: { regret: regretSE }, summary,
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

  // --------------------------------------------------------- registry ----
  /**
   * Add families, environments and algorithms to the K-armed / contextual arena.
   * Add-on files call MAB.register({ families, environments, algorithms }).
   * A family may carry `before: '<family id>'` to be listed ahead of that family.
   */
  function prepareEnv(e) {
    if (!e.d) e.d = 1;
    if (e.type === 'bernoulli') {
      if (e.sigma == null) e.sigma = 0.5;
      if (e.scale == null) e.scale = 1;
      if (!e.normalize) e.normalize = (r) => Math.min(1, Math.max(0, r));
    } else {
      if (e.sigma == null) e.sigma = 1;
      if (e.scale == null) e.scale = Math.SQRT2 * e.sigma;
      if (!e.normalize) e.normalize = (r) => Math.min(1, Math.max(0, (r + 4) / 8));
    }
    return e;
  }
  function register({ families = [], environments = [], algorithms = [] } = {}) {
    for (const f of families) {
      if (FAMILIES.some((x) => x.id === f.id)) continue;
      const at = f.before ? FAMILIES.findIndex((x) => x.id === f.before) : -1;
      if (at >= 0) FAMILIES.splice(at, 0, f); else FAMILIES.push(f);
    }
    for (const e of environments) {
      if (ENV_BY_ID[e.id]) throw new Error('Duplicate environment id: ' + e.id);
      ENVIRONMENTS.push(prepareEnv(e)); ENV_BY_ID[e.id] = e;
    }
    for (const a of algorithms) {
      if (ALGO_BY_ID[a.id]) throw new Error('Duplicate algorithm id: ' + a.id);
      if (!FAMILIES.some((f) => f.id === a.family)) throw new Error(`Unknown family ${a.family} for ${a.id}`);
      ALGORITHMS.push(a); ALGO_BY_ID[a.id] = a;
    }
    const order = (id) => FAMILIES.findIndex((f) => f.id === id);
    ALGORITHMS.sort((a, b) => order(a.family) - order(b.family));
  }

  /**
   * Settings are separate problem types (best-arm identification, dueling, ...), each with its
   * own environments, algorithms, measures and runner. See docs/ARCHITECTURE.md.
   */
  const SETTINGS = [];
  function registerSetting(s) {
    if (SETTINGS.some((x) => x.id === s.id)) throw new Error('Duplicate setting id: ' + s.id);
    SETTINGS.push(s);
    return s;
  }
  registerSetting({
    id: 'kab',
    name: 'K-armed & contextual',
    short: 'K-armed',
    blurb: 'Pick one of K arms each round and see only its reward. Includes adversarial, non-stationary, rested and contextual environments.',
    families: FAMILIES, environments: ENVIRONMENTS, algorithms: ALGORITHMS,
    defaultEnv: 'bernoulli',
    controls: {
      K: { label: 'Arms (K)', min: 2, max: 50, default: 10 },
      T: { label: 'Horizon (T steps)', options: [250, 500, 1000, 2000, 5000, 10000], default: 1000 },
      runs: { label: 'Runs averaged', options: [20, 50, 100, 200, 500, 1000], default: 200 },
    },
    xLabel: 'step t',
    metrics: [
      { key: 'regret', label: 'Cumulative regret', better: 'lower', fmt: 'num', band: true, note: 'Shaded bands: 95% confidence interval of the mean across runs. Logarithmic growth shows up as a straight line on the log time axis.' },
      { key: 'optimal', label: 'Best-arm rate', better: 'higher', fmt: 'pct', smooth: true, zeroToOne: true, note: 'Share of runs that pulled the best arm at each step' },
      { key: 'reward', label: 'Reward per step', better: 'higher', fmt: 'fixed3', smooth: true, note: 'Average reward received at each step' },
    ],
    final: { key: 'finalRegret', se: 'finalSE', median: 'finalMedian', label: 'Final regret', better: 'lower' },
    tiles: [
      { key: 'optimalTail', label: 'Picks the best arm most', better: 'higher', fmt: 'pct', sub: 'of the last 10% of steps' },
    ],
    columns: [
      { key: 'finalRegret', label: 'Regret', fmt: 'num', better: 'lower' },
      { key: 'finalSE', label: '95% CI', fmt: 'ci' },
      { key: 'finalMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'optimalTail', label: 'Best arm, last 10%', fmt: 'pct', better: 'higher' },
      { key: 'bestShare', label: 'Pulls on best arm', fmt: 'pct', better: 'higher' },
      { key: 'meanReward', label: 'Mean reward', fmt: 'fixed3', better: 'higher' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: true,
    createRunner,
  });

  const util = {
    stats, record, argmax, untried, sampleIndex, softmaxInto, klBern, klUcbBern, klIndex,
    posterior, samplePosterior, normCdf, normPdf, normInv, logPlus, binomHalf, gittinsPsi, exp3Gamma, now,
  };

  return {
    Rng, hash32, normInv, normCdf, klBern, klUcbBern, argmax, gittinsPsi, util,
    ENVIRONMENTS, ENV_BY_ID, FAMILIES, ALGORITHMS, ALGO_BY_ID,
    SETTINGS, register, registerSetting,
    defaultParams, createRunner, simulate,
  };
});
