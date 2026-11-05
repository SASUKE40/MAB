/*
 * robust.js — heavy-tailed, risk-aware, conservative, differentially private and
 * graph-feedback bandits for the K-armed arena, plus three environments that exercise them.
 *
 * Environments: 'heavy-tailed' (Student-t noise, ν = 2.2), 'risky' (the best-mean arm has a
 * long left tail) and 'side-observations' (Bernoulli arms + an Erdős–Rényi feedback graph).
 *
 * Simplifications are stated in each algorithm's `idea` and in the comments below. The robust
 * estimators (truncated mean, median of means, Catoni, APE's p-robust estimator) are recomputed
 * lazily — when an arm's sample count or the confidence level ln(1/δ) has grown by 5% — so a
 * 1,000-step, 10-arm run stays within a few milliseconds.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { stats, record, argmax, untried, sampleIndex, softmaxInto, samplePosterior } = MAB.util;

  // ------------------------------------------------------------ helpers ----
  /** Student-t with ν degrees of freedom: Z / √(χ²_ν / ν). */
  function studentT(rng, nu) {
    return rng.normal() / Math.sqrt((2 * rng.gamma(nu / 2)) / nu);
  }
  function laplace(rng, b) {
    const u = rng.uniform() - 0.5;
    return -b * Math.sign(u) * Math.log(Math.max(1e-300, 1 - 2 * Math.abs(u)));
  }
  function gumbel(rng) {
    return -Math.log(-Math.log(Math.max(1e-300, rng.uniform())) || 1e-300);
  }
  /** Confidence level ln(1/δ) with δ = t⁻², floored at 1. */
  function logDelta(t) { return Math.max(1, 2 * Math.log(t)); }

  /** Per-arm growable sample storage (optionally kept sorted). */
  function sampleStore(K) {
    const data = [];
    for (let a = 0; a < K; a++) data.push(new Float64Array(16));
    const n = new Int32Array(K);
    function grow(a) {
      if (n[a] < data[a].length) return;
      const d = new Float64Array(data[a].length * 2);
      d.set(data[a]); data[a] = d;
    }
    return {
      data, n,
      push(a, x) { grow(a); data[a][n[a]++] = x; },
      insertSorted(a, x) {
        grow(a);
        const d = data[a];
        let lo = 0, hi = n[a];
        while (lo < hi) { const mid = (lo + hi) >> 1; if (d[mid] <= x) lo = mid + 1; else hi = mid; }
        d.copyWithin(lo + 1, lo, n[a]);
        d[lo] = x; n[a]++;
      },
    };
  }

  /** Feedback graph from ctx.info, with self-loops; identity (no side observations) when absent. */
  function feedbackGraph(K, info) {
    const G = new Uint8Array(K * K);
    const src = info && info.graph;
    const ok = src && src.length === K * K;
    for (let a = 0; a < K; a++) {
      for (let b = 0; b < K; b++) G[a * K + b] = a === b ? 1 : ok && src[a * K + b] ? 1 : 0;
    }
    const inN = [], outN = [];
    for (let i = 0; i < K; i++) {
      const ins = [], outs = [];
      for (let j = 0; j < K; j++) { if (G[j * K + i]) ins.push(j); if (G[i * K + j]) outs.push(j); }
      inN.push(Int32Array.from(ins)); outN.push(Int32Array.from(outs));
    }
    return { G, inN, outN, has: !!ok };
  }
  /** Greedy independent set size of the symmetrised graph (approximates α(G)). */
  function independenceNumber(K, G) {
    const alive = new Uint8Array(K).fill(1);
    let size = 0, left = K;
    while (left > 0) {
      let best = -1, bd = Infinity;
      for (let i = 0; i < K; i++) {
        if (!alive[i]) continue;
        let d = 0;
        for (let j = 0; j < K; j++) if (j !== i && alive[j] && (G[i * K + j] || G[j * K + i])) d++;
        if (d < bd) { bd = d; best = i; }
      }
      size++;
      for (let j = 0; j < K; j++) {
        if (alive[j] && (j === best || G[best * K + j] || G[j * K + best])) { alive[j] = 0; left--; }
      }
    }
    return size;
  }
  /** Greedy dominating set: repeatedly take the arm that reveals the most unobserved arms. */
  function dominatingSet(K, G) {
    const covered = new Uint8Array(K), D = [];
    let left = K;
    while (left > 0) {
      let best = 0, bc = -1;
      for (let i = 0; i < K; i++) {
        let c = 0;
        for (let j = 0; j < K; j++) if (!covered[j] && G[i * K + j]) c++;
        if (c > bc) { bc = c; best = i; }
      }
      D.push(best);
      for (let j = 0; j < K; j++) if (!covered[j] && G[best * K + j]) { covered[j] = 1; left--; }
    }
    return D;
  }
  /**
   * Approximate solution of ELP's LP  max_s min_i Σ_{j→i} s_j  (s a distribution): the
   * zero-sum game "explorer picks j, adversary picks i" solved by multiplicative weights.
   * Returns { s, m } with m = min_i Σ_{j→i} s_j.
   */
  function exploreLP(K, G) {
    const iters = 400, beta = Math.sqrt((8 * Math.log(K + 1)) / iters);
    const cover = new Float64Array(K), y = new Float64Array(K), s = new Float64Array(K);
    for (let it = 0; it < iters; it++) {
      let mn = Infinity;
      for (let i = 0; i < K; i++) if (cover[i] < mn) mn = cover[i];
      let ys = 0;
      for (let i = 0; i < K; i++) { y[i] = Math.exp(-beta * (cover[i] - mn)); ys += y[i]; }
      let bj = 0, bv = -1;
      for (let j = 0; j < K; j++) {
        let v = 0;
        for (let i = 0; i < K; i++) if (G[j * K + i]) v += y[i];
        if (v > bv + 1e-12) { bv = v; bj = j; }
      }
      s[bj] += 1;
      for (let i = 0; i < K; i++) if (G[bj * K + i]) cover[i] += 1;
    }
    for (let j = 0; j < K; j++) s[j] /= iters;
    let m = Infinity;
    for (let i = 0; i < K; i++) {
      let q = 0;
      for (let j = 0; j < K; j++) if (G[j * K + i]) q += s[j];
      if (q < m) m = q;
    }
    return { s, m: Math.max(m, 1 / K) };
  }

  // -------------------------------------------------- robust estimators ----
  /** Catoni ψ(x) = sign(x)·ln(1 + |x| + x²/2) and its derivative. */
  function catoniPsi(x) { const a = Math.abs(x); return Math.sign(x) * Math.log(1 + a + 0.5 * a * a); }
  function catoniDpsi(x) { const a = Math.abs(x); return (1 + a) / (1 + a + 0.5 * a * a); }
  /** Root of Σ ψ(α(xᵢ − μ)) = 0: safeguarded Newton inside [min x, max x]. */
  function catoniMean(xs, n, alpha, start) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { if (xs[i] < lo) lo = xs[i]; if (xs[i] > hi) hi = xs[i]; }
    if (hi - lo < 1e-12) return lo;
    let mu = Math.min(hi, Math.max(lo, start));
    for (let it = 0; it < 40; it++) {
      let f = 0, df = 0;
      for (let i = 0; i < n; i++) { const z = alpha * (xs[i] - mu); f += catoniPsi(z); df += catoniDpsi(z); }
      if (f > 0) lo = mu; else hi = mu;               // f is decreasing in μ
      let next = mu + f / (alpha * df);
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      if (Math.abs(next - mu) < 1e-7 * (1 + Math.abs(mu))) return next;
      mu = next;
    }
    return mu;
  }
  /** Median of k block means (blocks cover all n samples, sizes differ by at most one). */
  function medianOfMeans(xs, n, k, scratch) {
    for (let j = 0; j < k; j++) {
      const a = Math.floor((j * n) / k), b = Math.floor(((j + 1) * n) / k);
      let s = 0;
      for (let i = a; i < b; i++) s += xs[i];
      scratch[j] = s / (b - a);
    }
    const v = scratch.subarray(0, k).sort();
    return k % 2 ? v[(k - 1) >> 1] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
  }
  /** Lee et al.'s influence function: ψ(x) = ±ln(1 ± x + b_p|x|^p). */
  function apePsi(x, p, bp) {
    const a = Math.abs(x), w = bp * Math.pow(a, p);
    return x >= 0 ? Math.log(1 + x + w) : -Math.log(1 - x + w);
  }
  function apeB(p) {
    const r = (2 - p) / (p - 1);
    return Math.pow(2 * Math.pow(r, 1 - 2 / p) + Math.pow(r, 2 - 2 / p), -p / 2);
  }
  /** Lazy-recompute rule shared by the cached estimators. */
  function stale(n, nC, L, LC) {
    return n !== nC && (n <= 30 || n >= 1.05 * nC) || L > 1.05 * LC;
  }

  /*
   * Robust-UCB (Bubeck, Cesa-Bianchi & Lugosi 2013). Index: estimator + c·u^{1/(1+ε)}·(L/n)^{ε/(1+ε)}
   * with L = ln(1/δ) = 2 ln t and u = v·σ^{1+ε} the assumed (1+ε)-th moment bound.
   * The paper's constants are c = 4 (truncated), (12)^{1/(1+ε)}16^{ε/(1+ε)} (median of means) and
   * 2 with v the variance (Catoni); here c is a tunable scale with a practical default.
   */
  function robustUcb(kind) {
    return function create(K, { rng, params, env }) {
      const eps = kind === 'catoni' ? 1 : params.eps, sig = env.sigma;
      const u = params.v * Math.pow(sig, 1 + eps), uRoot = Math.pow(params.v, 1 / (1 + eps)) * sig;
      const c = params.c, ex = eps / (1 + eps), inv = 1 / (1 + eps);
      const store = sampleStore(K), est = new Float64Array(K), idx = new Float64Array(K);
      const nC = new Int32Array(K), LC = new Float64Array(K), S = new Float64Array(K);
      const scratch = new Float64Array(512);
      let curL = 1;
      function recompute(a, L) {
        const xs = store.data[a], n = store.n[a], had = nC[a] > 0;
        nC[a] = n; LC[a] = L;
        if (kind === 'trunc') {
          let s = 0;
          for (let i = 0; i < n; i++) if (Math.abs(xs[i]) <= Math.pow((u * (i + 1)) / L, inv)) s += xs[i];
          S[a] = s; est[a] = s / n;
        } else if (kind === 'mom') {
          const k = Math.max(1, Math.min(Math.floor(n / 2), Math.floor(1 + 8 * L), scratch.length));
          est[a] = medianOfMeans(xs, n, k, scratch);
        } else {
          const alpha = Math.sqrt((2 * L) / (n * params.v * sig * sig));
          est[a] = catoniMean(xs, n, alpha, had ? est[a] : xs[0]);
        }
      }
      return {
        select(t) {
          const v0 = untried(store.n); if (v0 >= 0) return v0;
          const L = logDelta(t);
          curL = L;
          for (let a = 0; a < K; a++) {
            const n = store.n[a];
            if (stale(n, nC[a], L, LC[a])) recompute(a, L);
            idx[a] = est[a] + c * uRoot * Math.pow(L / n, ex);
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          store.push(a, r);
          const n = store.n[a];
          if (kind === 'trunc' && LC[a] > 0) {
            // incremental: the threshold of the new sample uses the frozen confidence level LC[a]
            if (Math.abs(r) <= Math.pow((u * n) / LC[a], inv)) S[a] += r;
            nC[a] = n; est[a] = S[a] / n;
          } else if (n === 1) {
            recompute(a, curL);
          }
        },
      };
    };
  }

  const robustParams = (cDefault, vDefault) => [
    { key: 'c', label: 'c (bonus scale)', default: cDefault, min: 0, max: 5, step: 0.05 },
    { key: 'v', label: 'v (moment bound, × σ^{1+ε})', default: vDefault, min: 0.1, max: 50, step: 0.1 },
    { key: 'eps', label: 'ε (moment order 1+ε)', default: 1, min: 0.1, max: 1, step: 0.05 },
  ];

  // ---------------------------------------------------------------------------------------------
  MAB.register({
    families: [
      { id: 'robust', name: 'Heavy-tailed & robust', before: 'contextual',
        blurb: 'Swap the sample mean for an estimator that outliers cannot drag around, so confidence bounds stay valid when rewards have heavy tails.' },
      { id: 'risk', name: 'Risk-aware', before: 'contextual',
        blurb: 'Optimise a risk measure (CVaR, mean–variance) instead of the mean. Their mean-reward regret looks worse by design.' },
      { id: 'safe', name: 'Conservative / safe', before: 'contextual',
        blurb: 'Explore only while the cumulative reward provably stays above a fraction of what a baseline arm would have earned.' },
      { id: 'private', name: 'Differentially private', before: 'contextual',
        blurb: 'Add calibrated Laplace noise so the sequence of actions reveals almost nothing about any single reward.' },
      { id: 'graphfb', name: 'Graph feedback', before: 'contextual',
        blurb: 'Pulling an arm also reveals its neighbours\' rewards in a feedback graph; use every observation, not only the pulled arm\'s.' },
    ],

    environments: [
      {
        id: 'heavy-tailed',
        name: 'Heavy-tailed (Student-t, ν = 2.2)',
        short: 'Heavy tails',
        type: 'gaussian', stationary: true, tags: ['heavy tails'],
        sigma: 1,
        normalize: (r) => Math.min(1, Math.max(0, (r + 5) / 10)),
        description: 'Means μ ~ N(0, 1); rewards are μ plus Student-t noise with ν = 2.2 (scale 1, variance ≈ 11, third moment infinite). Rare huge outliers wreck sample means.',
        fill(K, T, rng, means, rewards) {
          const mu = new Float64Array(K);
          for (let a = 0; a < K; a++) mu[a] = rng.normal();
          for (let t = 0; t < T; t++) {
            for (let a = 0; a < K; a++) {
              means[t * K + a] = mu[a];
              rewards[t * K + a] = mu[a] + studentT(rng, 2.2);
            }
          }
        },
      },
      {
        id: 'risky',
        name: 'Risky best arm (skewed losses)',
        short: 'Risky',
        type: 'gaussian', stationary: true, tags: ['risk'],
        sigma: 1,
        normalize: (r) => Math.min(1, Math.max(0, (r + 6) / 10)),
        description: 'Means μ ~ U(0, 1); rewards are μ + σₐ(1 − E), E ~ Exp(1), a long left tail. Other arms have σₐ ~ U(0.2, 1), but the best-mean arm has σ = 2.5, so its CVaR and mean–variance score are poor.',
        fill(K, T, rng, means, rewards) {
          const mu = new Float64Array(K), sd = new Float64Array(K);
          let best = 0;
          for (let a = 0; a < K; a++) { mu[a] = rng.uniform(); sd[a] = 0.2 + 0.8 * rng.uniform(); if (mu[a] > mu[best]) best = a; }
          sd[best] = 2.5;
          for (let t = 0; t < T; t++) {
            for (let a = 0; a < K; a++) {
              means[t * K + a] = mu[a];
              rewards[t * K + a] = mu[a] + sd[a] * (1 + Math.log(Math.max(1e-300, rng.uniform())));
            }
          }
        },
      },
      {
        id: 'side-observations',
        name: 'Bernoulli with side observations',
        short: 'Side obs.',
        type: 'bernoulli', stationary: true, tags: ['side observations'],
        description: 'Bernoulli arms, μ ~ U(0, 1), plus a random undirected feedback graph (Erdős–Rényi, p = 0.25, self-loops): pulling an arm also reveals its neighbours\' rewards to algorithms that listen.',
        fill(K, T, rng, means, rewards) {
          const mu = new Float64Array(K), graph = new Uint8Array(K * K);
          for (let a = 0; a < K; a++) mu[a] = rng.uniform();
          for (let a = 0; a < K; a++) {
            graph[a * K + a] = 1;
            for (let b = a + 1; b < K; b++) if (rng.uniform() < 0.25) { graph[a * K + b] = 1; graph[b * K + a] = 1; }
          }
          for (let t = 0; t < T; t++) {
            for (let a = 0; a < K; a++) {
              means[t * K + a] = mu[a];
              rewards[t * K + a] = rng.uniform() < mu[a] ? 1 : 0;
            }
          }
          return { graph };
        },
      },
    ],

    algorithms: [
      // ------------------------------------------------------------ heavy tails & robust ----
      {
        id: 'robust-ucb-trunc', name: 'Robust-UCB (truncated mean)', family: 'robust',
        params: robustParams(0.2, 30),
        formula: 'argmaxₐ  (1/n)Σ Xₛ𝟙{|Xₛ| ≤ (u s / ln t²)^{1/(1+ε)}} + c·u^{1/(1+ε)}(ln t² / nₐ)^{ε/(1+ε)}',
        regret: 'O(Σ (u/Δₐ)^{1/ε} log T)  with only a (1+ε)-th moment',
        idea: 'Bubeck, Cesa-Bianchi & Lugosi: drop samples whose size exceeds a threshold that grows with their index, so one outlier cannot move the estimate. v bounds the raw moment E|X|^{1+ε} (the default 30 is generous; too small a v truncates good arms\' early rewards). Paper constant c = 4; the default is smaller for practical runs.',
        create: robustUcb('trunc'),
      },
      {
        id: 'robust-ucb-mom', name: 'Robust-UCB (median of means)', family: 'robust',
        params: robustParams(1, 1),
        formula: 'argmaxₐ  median of k block means + c·v^{1/(1+ε)}(ln t² / nₐ)^{ε/(1+ε)},  k = min(1 + 8 ln t², n/2)',
        regret: 'O(Σ (v/Δₐ)^{1/ε} log T)  with a centred (1+ε)-th moment',
        idea: 'Split the samples into k blocks and take the median of the block means: an outlier spoils only one block. Recomputed when an arm\'s sample count grows 5%.',
        create: robustUcb('mom'),
      },
      {
        id: 'robust-ucb-catoni', name: 'Robust-UCB (Catoni)', family: 'robust',
        params: robustParams(1, 1).slice(0, 2),
        formula: 'argmaxₐ  μ̂ₐ + c·σ√(v ln t² / nₐ),   Σₛ ψ(α(Xₛ − μ̂ₐ)) = 0,  ψ(x) = sign x·ln(1 + |x| + x²/2)',
        regret: 'O(Σ v log T / Δₐ)  with finite variance',
        idea: 'Catoni\'s M-estimator: a log-damped influence function makes the mean behave as if the noise were Gaussian. Uses the simplified scale α = √(2 ln t² / (n v σ²)). Paper constant c = 2.',
        create: robustUcb('catoni'),
      },
      {
        id: 'ape2', name: 'APE² (perturbed, heavy tails)', family: 'robust',
        params: [
          { key: 'c', label: 'c (scale)', default: 0.5, min: 0.05, max: 5, step: 0.05 },
          { key: 'p', label: 'p (moment order)', default: 2, min: 1.1, max: 2, step: 0.05 },
        ],
        formula: 'argmaxₐ  Ŷₐ + c/nₐ^{1−1/p}·Gₐ,  G ~ Gumbel,  Ŷ = (c/n^{1−1/p}) Σ ψₚ(Xₛ / (c n^{1/p}))',
        regret: 'O(√(K^{p−1}) T^{1/p} log K)  minimax-optimal up to log K',
        idea: 'Lee, Yang, Hwang & Lim (2020): a p-robust estimator plus random Gumbel perturbations whose scale shrinks like n^{−(1−1/p)}. No moment bound needed in the bonus; c is scaled by σ.',
        create(K, { rng, params, env }) {
          const p = params.p, bp = apeB(p), c = params.c * env.sigma;
          const store = sampleStore(K), est = new Float64Array(K), idx = new Float64Array(K);
          const nC = new Int32Array(K), lam = new Float64Array(K), S = new Float64Array(K);
          return {
            select() {
              const v0 = untried(store.n); if (v0 >= 0) return v0;
              for (let a = 0; a < K; a++) {
                const n = store.n[a];
                idx[a] = est[a] + (c / Math.pow(n, 1 - 1 / p)) * gumbel(rng);
              }
              return argmax(idx, rng);
            },
            update(a, r) {
              store.push(a, r);
              const n = store.n[a];
              if (n <= 30 || n >= 1.05 * nC[a]) {
                const xs = store.data[a];
                lam[a] = c * Math.pow(n, 1 / p); nC[a] = n;
                let s = 0;
                for (let i = 0; i < n; i++) s += apePsi(xs[i] / lam[a], p, bp);
                S[a] = s;
              } else {
                S[a] += apePsi(r / lam[a], p, bp);   // frozen scale between recomputes
              }
              est[a] = (lam[a] * S[a]) / n;
            },
          };
        },
      },

      // ------------------------------------------------------------------- risk-aware ----
      {
        id: 'cvar-ucb', name: 'CVaR-UCB', family: 'risk',
        params: [
          { key: 'alpha', label: 'α (CVaR level)', default: 0.2, min: 0.01, max: 1, step: 0.01 },
          { key: 'c', label: 'c (DKW width scale)', default: 0.5, min: 0, max: 3, step: 0.05 },
        ],
        formula: 'argmaxₐ  CVaR_α(F̃ₐ),   F̃ₐ = F̂ₐ with mass εₐ = c√(ln(2t²)/2nₐ) moved from the bottom to the top',
        regret: 'O(log T) CVaR-regret (Tamkin et al. 2019)',
        idea: 'Tamkin, Keramati, Dann & Brunskill: optimism on the whole distribution via the DKW band, then read off the mean of the worst α of outcomes (c = 1 is the DKW band; the default halves it, else the band swamps the α-tail for the first ~200 pulls). Optimises CVaR of normalised rewards, not the mean, so its mean-reward regret looks worse by design.',
        create(K, { rng, params, env }) {
          const alpha = params.alpha, c = params.c, store = sampleStore(K), idx = new Float64Array(K);
          const lgC = new Float64Array(K);   // ln(2t²) at the last evaluation; 0 = arm changed since
          function cvar(a, lg) {
            const n = store.n[a], xs = store.data[a];
            const e = Math.min(1, c * Math.sqrt(lg / (2 * n)));
            const lo = e, hi = Math.min(1, e + alpha);
            let s = Math.max(0, e + alpha - 1);       // mass placed at the top of the support (1)
            const i0 = Math.floor(lo * n), i1 = Math.min(n - 1, Math.ceil(hi * n) - 1);
            for (let i = i0; i <= i1; i++) {
              const w = Math.min(hi, (i + 1) / n) - Math.max(lo, i / n);
              if (w > 0) s += w * xs[i];
            }
            lgC[a] = lg;
            return s / alpha;
          }
          return {
            select(t) {
              const v0 = untried(store.n); if (v0 >= 0) return v0;
              const lg = Math.log(2 * t * t);
              // re-evaluate an arm when it gained a sample or the confidence level grew 2%
              for (let a = 0; a < K; a++) if (lgC[a] === 0 || lg > 1.02 * lgC[a]) idx[a] = cvar(a, lg);
              return argmax(idx, rng);
            },
            update(a, r) { store.insertSorted(a, env.normalize(r)); lgC[a] = 0; },
          };
        },
      },
      {
        id: 'mv-ucb', name: 'MV-UCB (mean–variance)', family: 'risk',
        params: [
          { key: 'rho', label: 'ρ (risk tolerance)', default: 1, min: 0, max: 20, step: 0.1 },
          { key: 'c', label: 'c (bonus scale)', default: 1, min: 0, max: 5, step: 0.05 },
        ],
        formula: 'argmaxₐ  ρμ̂ₐ − σ̂ₐ² + c(ρs + √2σ²)√(2 ln t / nₐ)',
        regret: 'O(log T / Γ²) mean–variance regret (Sani et al. 2012, MV-LCB)',
        idea: 'Sani, Lazaric & Munos: minimise the mean–variance σ² − ρμ with a lower confidence bound (written here as maximising ρμ − σ² optimistically). Optimises a risk measure, so its mean-reward regret looks worse by design.',
        create(K, { rng, params, env }) {
          const s = stats(K), sq = new Float64Array(K), idx = new Float64Array(K);
          const rho = params.rho, w = params.c * (rho * env.scale + Math.SQRT2 * env.sigma * env.sigma);
          return {
            select(t) {
              const v0 = untried(s.n); if (v0 >= 0) return v0;
              const lt = 2 * Math.log(t);
              for (let a = 0; a < K; a++) {
                const n = s.n[a], m = s.mean[a], v = Math.max(0, sq[a] / n - m * m);
                idx[a] = rho * m - v + w * Math.sqrt(lt / n);
              }
              return argmax(idx, rng);
            },
            update(a, r) { record(s, a, r); sq[a] += r * r; },
          };
        },
      },
      {
        id: 'marab', name: 'MaRaB', family: 'risk',
        params: [
          { key: 'alpha', label: 'α (CVaR level)', default: 0.2, min: 0.01, max: 1, step: 0.01 },
          { key: 'C', label: 'C (caution)', default: 1, min: 0, max: 5, step: 0.05 },
        ],
        formula: 'argmaxₐ  ĈVaR_α(a) − C·s·√(ln⌈tα⌉ / ⌈nₐα⌉)',
        regret: 'No log bound; designed for safety',
        idea: 'Galichet, Sebag & Teytaud\'s Multi-Armed Risk-Aware Bandit: rank arms by the empirical mean of their worst α outcomes, minus a pessimistic margin, so it is cautious about arms it barely knows. Optimises CVaR, so its mean-reward regret looks worse by design.',
        create(K, { rng, params, env }) {
          const alpha = params.alpha, C = params.C * env.scale, store = sampleStore(K), idx = new Float64Array(K);
          const tail = new Float64Array(K), m = new Float64Array(K);   // empirical CVaR, ⌈nα⌉
          return {
            select(t) {
              const v0 = untried(store.n); if (v0 >= 0) return v0;
              const lt = Math.log(Math.max(1, Math.ceil(t * alpha)));
              for (let a = 0; a < K; a++) idx[a] = tail[a] - C * Math.sqrt(lt / m[a]);
              return argmax(idx, rng);
            },
            update(a, r) {
              store.insertSorted(a, r);
              const k = Math.max(1, Math.ceil(store.n[a] * alpha)), xs = store.data[a];
              let s = 0;
              for (let i = 0; i < k; i++) s += xs[i];
              tail[a] = s / k; m[a] = k;
            },
          };
        },
      },

      // ------------------------------------------------------------------ conservative ----
      {
        id: 'conservative-ucb', name: 'Conservative UCB', family: 'safe',
        params: [
          { key: 'alpha', label: 'α (allowed shortfall)', default: 0.1, min: 0.01, max: 1, step: 0.01 },
          { key: 'c', label: 'c', default: 1.414, min: 0, max: 5, step: 0.01 },
        ],
        formula: 'play UCB arm J if Σₐ≠₀ nₐ·LCBₐ + LCB_J + n₀μ̂₀ ≥ (1 − α)·t·μ̂₀, else baseline arm 0;  LCB/UCB = μ̂ ∓ c·s√(ln t / n)',
        regret: 'O(Σ log T / Δₐ + K log T / (αμ₀)) and never below (1−α)× baseline w.h.p.',
        idea: 'Wu, Shariff, Lattimore & Szepesvári (2016): explore only while a pessimistic estimate of the total reward stays above (1 − α) times what always playing the baseline (arm 0) would earn. Uses normalised rewards and the plug-in baseline mean μ̂₀ instead of a known μ₀.',
        create(K, { rng, params, env }) {
          // confidence widths in normalised units: env.scale × the slope of env.normalize
          const slope = Math.abs(env.normalize(0.75) - env.normalize(0.25)) / 0.5;
          const s = stats(K), idx = new Float64Array(K), alpha = params.alpha, c = params.c * env.scale * slope;
          return {
            select(t) {
              if (s.n[0] === 0) return 0;
              const u = untried(s.n);
              const lt = Math.log(t);
              let J = u;
              if (J < 0) {
                for (let a = 0; a < K; a++) idx[a] = s.mean[a] + c * Math.sqrt(lt / s.n[a]);
                J = argmax(idx, rng);
              }
              if (J === 0) return 0;
              const lcb = (a) => (s.n[a] ? Math.max(0, s.mean[a] - c * Math.sqrt(lt / s.n[a])) : 0);
              let lhs = s.n[0] * s.mean[0] + lcb(J);
              for (let a = 1; a < K; a++) lhs += s.n[a] * lcb(a);
              return lhs >= (1 - alpha) * t * s.mean[0] ? J : 0;
            },
            update(a, r) { record(s, a, env.normalize(r)); },
          };
        },
      },

      // ------------------------------------------------------------------------ private ----
      {
        id: 'dp-ucb', name: 'DP-UCB (tree mechanism)', family: 'private',
        params: [{ key: 'epsilon', label: 'ε (privacy)', default: 1, min: 0.05, max: 10, step: 0.05 }],
        formula: 'argmaxₐ  S̃ₐ/nₐ + √(2 ln t / nₐ) + b√(2mₐ ln t)/nₐ,  S̃ = tree-mechanism sum, b = ⌈log₂T + 1⌉/ε',
        regret: 'O(Σ log T / Δₐ + K log^{2.5} T / ε)  (ε-DP)',
        idea: 'Mishra & Thakurta (2015): each arm\'s running sum of normalised rewards is released through the binary-tree (counter) mechanism, so every reward is noised in at most log₂T + 1 Laplace p-sums; the index adds a width for that noise (mₐ = p-sums in use).',
        create(K, { rng, params, env, T }) {
          const levels = Math.ceil(Math.log2(T + 1)) + 1, b = levels / params.epsilon;
          const exact = new Float64Array(K * levels), noisy = new Float64Array(K * levels);
          const n = new Float64Array(K), idx = new Float64Array(K), sumN = new Float64Array(K), bits = new Float64Array(K);
          return {
            select(t) {
              const u = untried(n); if (u >= 0) return u;
              const lt = Math.log(t);
              for (let a = 0; a < K; a++) {
                idx[a] = sumN[a] / n[a] + Math.sqrt((2 * lt) / n[a]) + (b * Math.sqrt(2 * bits[a] * lt)) / n[a];
              }
              return argmax(idx, rng);
            },
            update(a, r) {
              const x = env.normalize(r), k = ++n[a], o = a * levels;
              let i = 0;
              while (((k >> i) & 1) === 0) i++;                    // lowest set bit of k
              let v = x;
              for (let j = 0; j < i; j++) { v += exact[o + j]; exact[o + j] = 0; noisy[o + j] = 0; }
              exact[o + i] = v; noisy[o + i] = v + laplace(rng, b);
              let s = 0, m = 0;
              for (let j = 0; j < levels; j++) if ((k >> j) & 1) { s += noisy[o + j]; m++; }
              sumN[a] = s; bits[a] = m;
            },
          };
        },
      },
      {
        id: 'dp-ts', name: 'DP-TS (epoch Laplace)', family: 'private',
        params: [{ key: 'epsilon', label: 'ε (privacy)', default: 1, min: 0.05, max: 10, step: 0.05 }],
        formula: 'θₐ ~ N(μ̃ₐ, 1/(4Lₐ) + 2/(εLₐ)²),  μ̃ₐ = (Σ epoch rewards + Lap(1/ε)) / Lₐ,  Lₐ doubles each epoch',
        regret: 'O(Σ log T / Δₐ + K log T / ε)  (ε-DP)',
        idea: 'Hu & Hegde (2022): each arm learns in doubling epochs and publishes only the latest epoch\'s mean plus one Laplace draw, so every reward is used once; Gaussian Thompson sampling runs on those private means.',
        create(K, { rng, params, env }) {
          const eps = params.epsilon, mu = new Float64Array(K), len = new Float64Array(K);
          const cnt = new Float64Array(K), sum = new Float64Array(K), th = new Float64Array(K);
          return {
            select() {
              for (let a = 0; a < K; a++) if (len[a] === 0) return a;
              for (let a = 0; a < K; a++) {
                const L = len[a], v = 1 / (4 * L) + 2 / (eps * L * eps * L);
                th[a] = mu[a] + Math.sqrt(v) * rng.normal();
              }
              return argmax(th, rng);
            },
            update(a, r) {
              sum[a] += env.normalize(r); cnt[a]++;
              const target = len[a] === 0 ? 1 : 2 * len[a];
              if (cnt[a] >= target) {
                mu[a] = (sum[a] + laplace(rng, 1 / eps)) / cnt[a];
                len[a] = cnt[a]; cnt[a] = 0; sum[a] = 0;
              }
            },
          };
        },
      },

      // ----------------------------------------------------------------- graph feedback ----
      {
        id: 'ucb-n', name: 'UCB-N', family: 'graphfb',
        params: [{ key: 'c', label: 'c', default: 1.414, min: 0, max: 5, step: 0.01 }],
        formula: 'argmaxₐ  Q̂ₐ + c·s·√(ln t / Oₐ),  Oₐ = pulls + side observations of a',
        regret: 'O(Σ_{cliques C} max_{a∈C} log T / Δₐ)',
        idea: 'Caron, Kveton, Lelarge & Bhagat (2012): plain UCB1 that also learns from every neighbour reward the graph reveals. Without a graph it is UCB1.',
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
            observe: (b, r) => record(s, b, r),
          };
        },
      },
      {
        id: 'ucb-maxn', name: 'UCB-MaxN', family: 'graphfb',
        params: [{ key: 'c', label: 'c', default: 1.414, min: 0, max: 5, step: 0.01 }],
        formula: 'i* = argmaxₐ UCBₐ;  play argmax_{j ∈ N(i*)} Q̂ⱼ',
        regret: 'Same order as UCB-N; better constants on dense graphs',
        idea: 'Caron et al.: find the most optimistic arm, then pull its empirically best neighbour, which still reveals the optimistic arm but costs less.',
        create(K, { rng, params, env, info }) {
          const s = stats(K), idx = new Float64Array(K), c = params.c * env.scale;
          const { outN } = feedbackGraph(K, info);
          return {
            select(t) {
              const u = untried(s.n); if (u >= 0) return u;
              const lt = Math.log(t);
              for (let a = 0; a < K; a++) idx[a] = s.mean[a] + c * Math.sqrt(lt / s.n[a]);
              const i = argmax(idx, rng), nb = outN[i];
              let best = i, bv = -Infinity;
              for (let k = 0; k < nb.length; k++) if (s.mean[nb[k]] > bv) { bv = s.mean[nb[k]]; best = nb[k]; }
              return best;
            },
            update: (a, r) => record(s, a, r),
            observe: (b, r) => record(s, b, r),
          };
        },
      },
      {
        id: 'ts-n', name: 'Thompson-N', family: 'graphfb',
        params: [],
        formula: 'θₐ ~ posteriorₐ(pulls + side observations),   aₜ = argmaxₐ θₐ',
        regret: 'Empirically ≤ UCB-N; no tight graph-dependent bound in the original paper',
        idea: 'Caron et al.\'s Thompson sampling with side observations: every revealed reward updates its arm\'s posterior. Without a graph it is Thompson sampling.',
        create(K, { rng, env }) {
          const s = stats(K), th = new Float64Array(K);
          const bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
          return {
            select() {
              for (let a = 0; a < K; a++) {
                const n = s.n[a];
                th[a] = samplePosterior(rng, bern, s2, s.sum[a], n);
              }
              return argmax(th, rng);
            },
            update: (a, r) => record(s, a, r),
            observe: (b, r) => record(s, b, r),
          };
        },
      },
      {
        id: 'exp3-set', name: 'EXP3-SET', family: 'graphfb',
        params: [],
        formula: 'π ∝ exp(−η L̂),   L̂ᵢ += ℓᵢ / qᵢ for every observed i,  qᵢ = Σ_{j→i} πⱼ,  η = √(2 ln K / α̂T)',
        regret: 'O(√(α(G) T log K))  α = independence number',
        idea: 'Alon, Cesa-Bianchi, Gentile & Mansour (2013): exponential weights where each observed loss is divided by the probability it was observed at all. α̂ is a greedy independent-set size. Without a graph it is EXP3.',
        create(K, { rng, env, T, info }) {
          const { G, inN } = feedbackGraph(K, info);
          const eta = Math.sqrt((2 * Math.log(K)) / (independenceNumber(K, G) * T));
          const lw = new Float64Array(K), p = new Float64Array(K);
          function learn(b, r) {
            const nb = inN[b];
            let q = 0;
            for (let k = 0; k < nb.length; k++) q += p[nb[k]];
            lw[b] -= (eta * (1 - env.normalize(r))) / Math.max(q, 1e-12);
          }
          return {
            select() { softmaxInto(p, lw, 1); return sampleIndex(p, rng); },
            update: learn,
            observe: learn,
          };
        },
      },
      {
        id: 'exp3-dom', name: 'EXP3-DOM', family: 'graphfb',
        params: [],
        formula: 'π = (1−γ)·softmax(−ηL̂) + γ·Uniform(D),  D = dominating set,  γ = √(|D| ln K / T),  η = γ/|D|',
        regret: 'O(√(|D| T log K))  (directed graphs)',
        idea: 'Alon et al. (2013): mix in uniform exploration over a dominating set, which observes every arm, so each loss estimate stays bounded. Single epoch with a greedy dominating set instead of the paper\'s doubling epochs.',
        create(K, { rng, env, T, info }) {
          const { G, inN } = feedbackGraph(K, info);
          const D = dominatingSet(K, G), nd = D.length;
          const gamma = Math.min(0.5, Math.sqrt((nd * Math.log(K)) / T)), eta = gamma / nd;
          const lw = new Float64Array(K), p = new Float64Array(K);
          function learn(b, r) {
            const nb = inN[b];
            let q = 0;
            for (let k = 0; k < nb.length; k++) q += p[nb[k]];
            lw[b] -= (eta * (1 - env.normalize(r))) / Math.max(q, 1e-12);
          }
          return {
            select() {
              softmaxInto(p, lw, 1);
              for (let a = 0; a < K; a++) p[a] *= 1 - gamma;
              for (let k = 0; k < nd; k++) p[D[k]] += gamma / nd;
              return sampleIndex(p, rng);
            },
            update: learn,
            observe: learn,
          };
        },
      },
      {
        id: 'elp', name: 'ELP', family: 'graphfb',
        params: [],
        formula: 'π = (1−γ)·w/Σw + γ·s,  s = argmax_s minᵢ Σ_{j→i} sⱼ (LP),  wᵢ ← wᵢ exp(η gᵢ/qᵢ),  η = γ·m*',
        regret: 'O(√(α(G) T log K))  (undirected graphs)',
        idea: 'Mannor & Shamir (2011): exponential weights whose exploration distribution comes from an LP that maximises the least-observed arm\'s observation probability m*. The LP is solved approximately by multiplicative weights (400 rounds) once per run.',
        create(K, { rng, env, T, info }) {
          const { G, inN } = feedbackGraph(K, info);
          const { s: ex, m } = exploreLP(K, G);
          const gamma = Math.min(0.5, Math.sqrt(Math.log(K) / (m * T))), eta = gamma * m;
          const lw = new Float64Array(K), p = new Float64Array(K);
          function learn(b, r) {
            const nb = inN[b];
            let q = 0;
            for (let k = 0; k < nb.length; k++) q += p[nb[k]];
            lw[b] += (eta * env.normalize(r)) / Math.max(q, 1e-12);
          }
          return {
            select() {
              softmaxInto(p, lw, 1);
              for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma * ex[a];
              return sampleIndex(p, rng);
            },
            update: learn,
            observe: learn,
          };
        },
      },
    ],
  });
});
