/*
 * classical.js — more classical stochastic K-armed algorithms for the arena.
 *
 * Undirected:   ε-first, VDBE, ε-BMC, Gumbel-max exploration
 * Optimism:     asymptotically optimal UCB, OCUCB, CP-UCB (Clopper–Pearson)
 * Divergence:   DMED
 * Perturbation: BESA (sub-sampling duels)
 * Bayesian:     Optimistic Bayesian sampling, KG(*), POKER
 *
 * All of them ignore the context X, so they run (context-free) on contextual environments.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { util } = MAB;
  const { stats, record, argmax, untried, klBern, posterior, samplePosterior, normCdf, normPdf, normInv } = util;

  // ------------------------------------------------------------ helpers ----
  /** ln Γ(x) for x > 0 (Lanczos, |rel err| < 2e-10). */
  function lnGamma(x) {
    const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5, ser = 1.000000000190015;
    tmp -= (x + 0.5) * Math.log(tmp);
    for (let j = 0; j < 6; j++) ser += g[j] / ++y;
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }
  /** Continued fraction for the incomplete beta function (modified Lentz). */
  function betacf(a, b, x) {
    const FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-10) break;
    }
    return h;
  }
  /** Regularized incomplete beta I_x(a, b); lnB = ln B(a, b). */
  function betaInc(x, a, b, lnB) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lnB);
    if (x < (a + 1) / (a + b + 2)) return (front * betacf(a, b, x)) / a;
    return 1 - (front * betacf(b, a, 1 - x)) / b;
  }
  /**
   * Clopper–Pearson upper bound: the largest p with P(Bin(n, p) ≤ x) ≥ δ, i.e. the (1 − δ)
   * quantile of Beta(x + 1, n − x). Safeguarded Newton inside [x/n, Chernoff bound]; the
   * Chernoff (KL-UCB) bound always lies above the exact one.
   */
  function cpUpper(x, n, delta) {
    if (x >= n) return 1;
    const a = x + 1, b = n - x, target = 1 - delta;
    const lnB = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
    let lo = x / n, hi = Math.min(1, util.klUcbBern(lo, n, Math.log(1 / delta)) + 1e-4);
    let p = hi;
    for (let i = 0; i < 30; i++) {
      const f = betaInc(p, a, b, lnB) - target;
      if (f < 0) lo = p; else hi = p;
      if (hi - lo < 1e-7) break;
      const dens = Math.exp((a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p) - lnB);
      let next = dens > 0 ? p - f / dens : NaN;
      if (!(next > lo && next < hi)) next = (lo + hi) / 2;
      if (Math.abs(next - p) < 1e-9) { p = next; break; }
      p = next;
    }
    return p;
  }
  const gumbel = (rng) => -Math.log(-Math.log(rng.uniform() || 1e-300) || 1e-300);
  /** KL divergence of the reward model: Bernoulli, or Gaussian with known variance s2. */
  const divergence = (bern, s2, p, q) => (bern ? klBern(p, q) : ((p - q) * (p - q)) / (2 * s2));

  // --------------------------------------------------------- algorithms ----
  const algorithms = [
    // ---- undirected
    {
      id: 'efirst', name: 'ε-first', family: 'undirected',
      params: [{ key: 'epsilon', label: 'ε (share of T)', default: 0.1, min: 0.001, max: 1, step: 0.001 }],
      formula: 'aₜ ~ Uniform{1, …, K} for t ≤ ε·T,  then argmaxₐ Q̂ₐ',
      regret: 'Pays ≈ εT·Δ̄ up front; O(T^⅔) with ε ∝ T^−⅓',
      idea: 'Tran-Thanh et al.\'s budgeted strategy: spend a fixed share of the horizon on uniformly random pulls, then exploit (still updating the means). Unlike ETC it samples at random instead of round-robin.',
      create(K, { rng, params, T }) {
        const s = stats(K), explore = Math.ceil(params.epsilon * T);
        return {
          select: (t) => (t <= explore ? rng.int(K) : argmax(s.mean, rng)),
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'vdbe', name: 'VDBE', family: 'undirected',
      params: [{ key: 'sigma', label: 'σ (inverse sensitivity)', default: 0.1, min: 0.001, max: 10, step: 0.001 }],
      formula: 'ε ← δ·(1 − e^{−|ΔQ|/σ})/(1 + e^{−|ΔQ|/σ}) + (1 − δ)ε,   δ = 1/K;  ε-greedy',
      regret: 'No regret bound; a heuristic schedule',
      idea: 'Tokic\'s value-difference based exploration: large changes in the estimates (surprise) push ε up, stable estimates let it fade. This is the single-state version with sample-average updates.',
      create(K, { rng, params, env }) {
        const s = stats(K), sig = params.sigma * env.scale, delta = 1 / K;
        let eps = 1;
        return {
          select: () => (rng.uniform() < eps ? rng.int(K) : argmax(s.mean, rng)),
          update(a, r) {
            const old = s.mean[a];
            record(s, a, r);
            const e = Math.exp(-Math.abs(s.mean[a] - old) / sig);
            eps = delta * ((1 - e) / (1 + e)) + (1 - delta) * eps;
          },
        };
      },
    },
    {
      id: 'ebmc', name: 'ε-BMC', family: 'undirected',
      params: [],
      formula: 'ε = E[ε],  Beta(α, β) ← moment match of (ε·N(r; mean Q̂, σ̂²) + (1−ε)·N(r; max Q̂, σ̂²))·Beta(α, β);  ε-greedy',
      regret: 'No regret bound; ε adapts from data',
      idea: 'Gimelfarb, Sanner & Lee treat ε as the weight of a uniform policy in a Bayesian combination with the greedy one, and update its Beta posterior by moment matching. Simplified bandit version: after each greedy pull, the reward is scored against the uniform model\'s mean Q̂ and the greedy model\'s max Q̂ (σ̂² = running reward variance).',
      create(K, { rng }) {
        const s = stats(K);
        let al = 1, be = 1, eps = 0.5, cnt = 0, rm = 0, m2 = 0, greedy = false;
        return {
          select() {
            greedy = rng.uniform() >= eps;
            return greedy ? argmax(s.mean, rng) : rng.int(K);
          },
          update(a, r) {
            if (!greedy) { record(s, a, r); return; }
            // the two models' expected rewards, under the estimates before this update
            let mx = -Infinity, avg = 0;
            for (let i = 0; i < K; i++) { avg += s.mean[i] / K; if (s.mean[i] > mx) mx = s.mean[i]; }
            // running variance of rewards (Welford)
            cnt++; const dlt = r - rm; rm += dlt / cnt; m2 += dlt * (r - rm);
            const v = Math.max(1e-4, cnt > 1 ? m2 / (cnt - 1) : 1);
            const lu = Math.exp(-((r - avg) ** 2) / (2 * v)), lg = Math.exp(-((r - mx) ** 2) / (2 * v));
            const n = al + be, z = (al * lu + be * lg) / n;
            if (z > 1e-300) {
              // moments of the mixture posterior ∝ (ε·Lu + (1 − ε)·Lg)·Beta(α, β)
              const e1 = (al * (al + 1) * lu + al * be * lg) / (n * (n + 1) * z);
              const e2 = (al * (al + 1) * ((al + 2) * lu + be * lg)) / (n * (n + 1) * (n + 2) * z);
              const vr = e2 - e1 * e1;
              if (vr > 1e-12 && e1 > 0 && e1 < 1) {
                const k = (e1 * (1 - e1)) / vr - 1;
                if (k > 0) { al = e1 * k; be = (1 - e1) * k; }
              }
            }
            eps = al / (al + be);
            record(s, a, r);
          },
        };
      },
    },
    {
      id: 'gumbel-max', name: 'Gumbel-max softmax', family: 'undirected',
      params: [{ key: 'tau', label: 'τ (temperature)', default: 0.1, min: 0.005, max: 5, step: 0.005 }],
      formula: 'argmaxₐ  Q̂ₐ / τ + Gₐ,   Gₐ ~ Gumbel(0, 1)',
      regret: 'Θ(T) for fixed τ (same as Softmax)',
      idea: 'The Gumbel-max trick samples the Boltzmann distribution P(a) ∝ exp(Q̂ₐ/τ) exactly, so this is Softmax in distribution and differs only in its random draws. It needs no normalising sum; contrast Boltzmann–Gumbel, whose noise shrinks per arm.',
      create(K, { rng, params, env }) {
        const s = stats(K), idx = new Float64Array(K), inv = 1 / (params.tau * env.scale);
        return {
          select() {
            for (let a = 0; a < K; a++) idx[a] = s.mean[a] * inv + gumbel(rng);
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    // ---- optimism
    {
      id: 'ao-ucb', name: 'Asymptotically optimal UCB', family: 'optimism',
      params: [],
      formula: 'argmaxₐ  Q̂ₐ + σ·√(2 ln f(t) / nₐ),   f(t) = 1 + t ln² t',
      regret: 'Asymptotically optimal for Gaussian arms: Σ 2σ² ln T / Δₐ',
      idea: 'The UCB of Lattimore & Szepesvári\'s book (ch. 8): a slightly larger confidence level than ln t buys the optimal constant. Uses the sub-Gaussian width σ (½ for Bernoulli).',
      create(K, { rng, env }) {
        const s = stats(K), idx = new Float64Array(K), s2 = 2 * env.sigma * env.sigma;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            const lt = Math.log(t), lf = Math.log(1 + t * lt * lt);
            for (let a = 0; a < K; a++) idx[a] = s.mean[a] + Math.sqrt((s2 * lf) / s.n[a]);
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'ocucb', name: 'OCUCB', family: 'optimism',
      params: [
        { key: 'alpha', label: 'α', default: 3, min: 1, max: 10, step: 0.1 },
        { key: 'psi', label: 'ψ', default: 2, min: 1, max: 20, step: 0.1 },
      ],
      formula: 'argmaxₐ  Q̂ₐ + σ·√(2α / nₐ · ln(ψT / t))',
      regret: 'O(√(KT)) and O(Σ log(T/Hₐ) / Δₐ); needs T (theory: α > 2, ψ ≥ 2)',
      idea: 'Lattimore\'s optimally confident UCB: the confidence level ln(ψT/t) shrinks as the horizon runs out, so late exploration is cheap. Order-optimal in both the problem-dependent and worst-case sense.',
      create(K, { rng, params, env, T }) {
        const s = stats(K), idx = new Float64Array(K);
        const c = 2 * params.alpha * env.sigma * env.sigma, PT = params.psi * T;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            const l = Math.log(Math.max(1, PT / t));
            for (let a = 0; a < K; a++) idx[a] = s.mean[a] + Math.sqrt((c * l) / s.n[a]);
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'cp-ucb', name: 'CP-UCB (Clopper–Pearson)', family: 'optimism',
      params: [],
      formula: 'argmaxₐ  sup{ p : P(Bin(nₐ, p) ≤ Sₐ) ≥ 1/t }   (Gaussian: Q̂ₐ + σ z₁₋₁⁄ₜ / √nₐ)',
      regret: 'Logarithmic in practice; its index never exceeds KL-UCB\'s',
      idea: 'Uses the exact binomial (Clopper–Pearson) upper confidence bound at level 1/t instead of a Hoeffding or Chernoff bound (for speed the level is refreshed on a 10% geometric grid of t). Gaussian arms have no binomial count, so it falls back to the exact Gaussian bound with known σ.',
      create(K, { rng, env }) {
        const s = stats(K), idx = new Float64Array(K), bern = env.type === 'bernoulli', sig = env.sigma;
        const dirty = new Uint8Array(K).fill(1);
        let delta = 1, next = 0;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            if (t >= next) { delta = 1 / t; next = Math.ceil(t * 1.1); dirty.fill(1); }
            if (bern) {
              // exact CP bounds are cached; only arms pulled since, or every arm after a level change, are redone
              for (let a = 0; a < K; a++) {
                if (!dirty[a]) continue;
                const n = s.n[a];
                idx[a] = cpUpper(Math.min(n, Math.max(0, Math.round(s.sum[a]))), n, delta);
                dirty[a] = 0;
              }
              return argmax(idx, rng);
            }
            const z = normInv(1 - 1 / t);
            for (let a = 0; a < K; a++) idx[a] = s.mean[a] + (sig * z) / Math.sqrt(s.n[a]);
            return argmax(idx, rng);
          },
          update(a, r) { record(s, a, r); dirty[a] = 1; },
        };
      },
    },
    // ---- minimum empirical divergence
    {
      id: 'dmed', name: 'DMED', family: 'divergence',
      params: [],
      formula: 'play every arm in the list once;  next list = { a : nₐ·KL(Q̂ₐ, max Q̂) ≤ ln t − ln nₐ }',
      regret: 'Asymptotically optimal (Honda & Takemura, bounded support)',
      idea: 'Deterministic minimum empirical divergence: an arm stays on the to-play list while the evidence that it is worse than the leader is weaker than ln(t / nₐ). Uses the Bernoulli KL, or the Gaussian KL with known σ.',
      create(K, { env }) {
        const s = stats(K), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        const list = new Int32Array(K);
        let len = 0, pos = 0;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            if (pos >= len) {
              let best = -Infinity;
              for (let a = 0; a < K; a++) if (s.mean[a] > best) best = s.mean[a];
              const lt = Math.log(t);
              len = 0; pos = 0;
              for (let a = 0; a < K; a++) {
                if (s.n[a] * divergence(bern, s2, s.mean[a], best) <= lt - Math.log(s.n[a])) list[len++] = a;
              }
              if (!len) for (let a = 0; a < K; a++) if (s.mean[a] === best) { list[len++] = a; break; }
            }
            return list[pos++];
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    // ---- randomized perturbation
    {
      id: 'besa', name: 'BESA', family: 'perturbation',
      params: [],
      formula: 'duel(a, b), nₐ ≤ n_b:  compare Q̂ₐ with the mean of nₐ of b\'s rewards drawn without replacement;  K arms: tournament',
      regret: 'O(log T) for two arms (Baransi et al.); no parameters',
      idea: 'Baransi, Maillard & Mannor\'s sub-sampling duels: the better-sampled arm must win on an equal footing, a random subsample the size of its rival\'s history. Needs no reward model and no confidence level. K arms play a knockout tournament in random order.',
      create(K, { rng, T }) {
        const s = stats(K), hist = [];
        for (let a = 0; a < K; a++) hist.push(new Float64Array(T));
        const cur = new Int32Array(K);
        // sum of m rewards of arm b drawn without replacement (in-place partial Fisher–Yates; order is irrelevant)
        function subMean(b, m) {
          const h = hist[b], n = s.n[b];
          let sum = 0;
          for (let i = 0; i < m; i++) {
            const j = i + rng.int(n - i), x = h[j];
            h[j] = h[i]; h[i] = x; sum += x;
          }
          return sum / m;
        }
        function duel(a, b) {
          const na = s.n[a], nb = s.n[b];
          let va, vb;
          if (na === nb) { va = s.mean[a]; vb = s.mean[b]; }
          else if (na < nb) { va = s.mean[a]; vb = subMean(b, na); }
          else { va = subMean(a, nb); vb = s.mean[b]; }
          if (va !== vb) return va > vb ? a : b;
          if (na !== nb) return na < nb ? a : b;
          return rng.uniform() < 0.5 ? a : b;
        }
        return {
          select() {
            const u = untried(s.n); if (u >= 0) return u;
            for (let i = 0; i < K; i++) cur[i] = i;
            for (let i = K - 1; i > 0; i--) { const j = rng.int(i + 1), x = cur[i]; cur[i] = cur[j]; cur[j] = x; }
            let m = K;
            while (m > 1) {
              let w = 0;
              for (let i = 0; i + 1 < m; i += 2) cur[w++] = duel(cur[i], cur[i + 1]);
              if (m & 1) cur[w++] = cur[m - 1];
              m = w;
            }
            return cur[0];
          },
          update(a, r) { hist[a][s.n[a]] = r; record(s, a, r); },
        };
      },
    },
    // ---- bayesian
    {
      id: 'obs', name: 'Optimistic Bayesian sampling', family: 'bayesian',
      params: [],
      formula: 'argmaxₐ  max(θₐ, E[μₐ]),   θₐ ~ posteriorₐ',
      regret: 'Proven consistent (May et al.); empirically at or below Thompson',
      idea: 'May, Korda, Lee & Leslie\'s variant of Thompson sampling: a posterior draw is never allowed below the posterior mean, so sampling noise can only make an arm look better, never worse.',
      create(K, { rng, env }) {
        const s = stats(K), th = new Float64Array(K), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select() {
            for (let a = 0; a < K; a++) {
              const x = samplePosterior(rng, bern, s2, s.sum[a], s.n[a]);
              const m = bern ? (1 + s.sum[a]) / (2 + s.n[a]) : s.sum[a] / (s.n[a] + 1);
              th[a] = x > m ? x : m;
            }
            return argmax(th, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'kg-star', name: 'KG(*)', family: 'bayesian',
      params: [],
      formula: 'argmaxₐ  μₐ + (T − t)·maxₘ νₐ(m)/m,   νₐ(m) = σ̃ₐ(m) f(−|μₐ − maxᵦ≠ₐ μᵦ| / σ̃ₐ(m))',
      regret: 'No regret bound; a look-ahead policy',
      idea: 'Ryzhov, Frazier & Powell\'s fix for the knowledge gradient: one pull may be worth almost nothing while several are worth a lot, so value an arm by the best average information per pull over m repeated pulls (m on a doubling grid).',
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
            const left = Math.max(1, T - t);
            for (let a = 0; a < K; a++) {
              const noise = bern ? Math.max(0.01, mu[a] * (1 - mu[a])) : s2;
              const v = va[a], gap = Math.abs(mu[a] - (a === i1 ? b2 : b1));
              // ν(m)/m rises then falls in m, so stop at the first decrease
              let best = 0;
              for (let m = 1; m <= left; m *= 2) {
                const st = Math.sqrt(Math.max(1e-18, (v * v * m) / (noise + v * m)));
                const z = -gap / st;
                if (z < -8) continue; // ν(m) ≈ 0: needs more pulls to matter
                const val = (st * (z * normCdf(z) + normPdf(z))) / m;
                if (val > best) best = val; else if (m > 1) break;
              }
              idx[a] = mu[a] + left * best;
            }
            return argmax(idx, rng);
          },
          update: (a, r) => record(s, a, r),
        };
      },
    },
    {
      id: 'poker', name: 'POKER', family: 'bayesian',
      params: [],
      formula: 'argmaxₐ  μₐ + δμ·(T − t)·P(μₐ ≥ μ̂* + δμ),   δμ = (μ̂* − μ̂₍⌈√K⌉₎) / √K',
      regret: 'No regret bound; a horizon-aware heuristic',
      idea: 'Vermorel & Mohri\'s "price of knowledge and estimated reward": the value of exploring an arm is the chance that it beats the leader by a typical gap δμ, times that gap, times the steps left. Simplified: every arm is pulled once first, and each arm\'s spread is shrunk toward the average spread so two lucky draws cannot freeze it out.',
      create(K, { rng, env, T }) {
        const s = stats(K), sq = new Float64Array(K), idx = new Float64Array(K), sorted = new Float64Array(K), NU = 2;
        return {
          select(t) {
            const u = untried(s.n); if (u >= 0) return u;
            let sdSum = 0, sdN = 0;
            for (let a = 0; a < K; a++) {
              sorted[a] = s.mean[a];
              const n = s.n[a];
              if (n > 1) {
                const sd = Math.sqrt(Math.max(0, (sq[a] - n * s.mean[a] * s.mean[a]) / (n - 1)));
                if (sd > 0) { sdSum += sd; sdN++; }
              }
            }
            const sdAvg = sdN ? sdSum / sdN : env.sigma;
            sorted.sort().reverse();
            // δμ: gap between the best mean and the ⌈√K⌉-th best, divided by √K
            const best = sorted[0], k = Math.min(K - 1, Math.max(1, Math.ceil(Math.sqrt(K)) - 1));
            const dmu = (best - sorted[k]) / Math.sqrt(K), h = T - t;
            for (let a = 0; a < K; a++) {
              const n = s.n[a], m = s.mean[a];
              // per-arm variance, shrunk toward the pooled one with NU pseudo degrees of freedom
              const sd = Math.sqrt((Math.max(0, sq[a] - n * m * m) + NU * sdAvg * sdAvg) / (n - 1 + NU));
              const p = 1 - normCdf((best + dmu - m) / (sd / Math.sqrt(n)));
              idx[a] = m + dmu * h * p;
            }
            return argmax(idx, rng);
          },
          update(a, r) { record(s, a, r); sq[a] += r * r; },
        };
      },
    },
  ];

  MAB.register({ algorithms });
});
