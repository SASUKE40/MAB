/*
 * adversarial.js — adversarial and best-of-both-worlds bandits for the arena.
 *
 * Environments (tag 'adversarial', rewards in {0, 1}):
 *   adv-switch        oblivious adversary: the leader switches, then a lean phase traps averaging learners
 *   adv-constrained   stochastically constrained adversary: fixed gaps, the common level jumps
 *   adv-corrupted     stochastic arms whose first C rounds of rewards are corrupted
 *
 * Adversarial:  EXP3.1, EXP3++, SAO (simplified)
 * FTRL & mirror descent (family 'ftrl'):
 *   FTRL with log-barrier, OMD with the ½-Tsallis mirror map, Poly INF (q = 2), BROAD-OMD,
 *   FPL (exponential perturbations, exact probabilities), FPL-GR, FTPL-Fréchet, FTPL-Gumbel
 *
 * Every algorithm works on losses ℓ = 1 − normalize(r) ∈ [0, 1] (INF works on gains) and ignores
 * the context X, so it runs on every environment.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { util } = MAB;
  const { sampleIndex, softmaxInto } = util;

  // ------------------------------------------------------- environments ----
  /** Random permutation of 0..K-1. */
  function shuffle(K, rng) {
    const p = Array.from({ length: K }, (_, i) => i);
    for (let i = K - 1; i > 0; i--) { const j = rng.int(i + 1); const x = p[i]; p[i] = p[j]; p[j] = x; }
    return p;
  }

  const environments = [
    {
      id: 'adv-switch',
      name: 'Adversarial, switching trap',
      short: 'Switching trap',
      type: 'bernoulli', stationary: false, tags: ['adversarial'],
      description: 'An oblivious adversary in three phases of doubling length (T/7, 2T/7, 4T/7). Arm B leads while every arm pays well; then arm A takes over while every arm pays poorly; then all pay well again and A stays 0.2 ahead. Learners that trust averages of old rewards cling to B, then starve A because its average is dragged down by the lean phase.',
      fill(K, T, rng, means, rewards) {
        const perm = shuffle(K, rng), A = perm[0], B = perm[1];
        const off = new Float64Array(K);
        for (let a = 0; a < K; a++) off[a] = 0.1 * rng.uniform();
        const e1 = Math.round(T / 7), e2 = Math.round((3 * T) / 7);
        for (let t = 0; t < T; t++) {
          const lead = t < e1 ? B : A, level = t >= e1 && t < e2 ? 0.1 : 0.75;
          for (let a = 0; a < K; a++) {
            const m = a === lead ? level + 0.2 : level - off[a];
            means[t * K + a] = m;
            rewards[t * K + a] = rng.uniform() < m ? 1 : 0;
          }
        }
      },
    },
    {
      id: 'adv-constrained',
      name: 'Stochastically constrained adversary',
      short: 'Constrained adv.',
      type: 'bernoulli', stationary: false, tags: ['adversarial'],
      description: 'Zimmert & Seldin\'s construction: the gaps are fixed (best arm always ahead by Δₐ ~ U(0.1, 0.2)) but all means jump together between a low level (0.25) and a high level (0.9) in phases that grow by 1.6×. Best-of-both-worlds methods keep logarithmic regret; stale averages fool UCB and Thompson sampling.',
      fill(K, T, rng, means, rewards) {
        const best = rng.int(K), gap = new Float64Array(K);
        for (let a = 0; a < K; a++) gap[a] = a === best ? 0 : 0.1 + 0.1 * rng.uniform();
        const P = 9, ratio = 1.6, unit = (T * (ratio - 1)) / (ratio ** P - 1);
        let phase = 0, len = unit, end = unit;
        for (let t = 0; t < T; t++) {
          while (t >= end) { phase++; len *= ratio; end += len; }
          const c = phase % 2 === 0 ? 0.25 : 0.9;
          for (let a = 0; a < K; a++) {
            const m = c - gap[a];
            means[t * K + a] = m;
            rewards[t * K + a] = rng.uniform() < m ? 1 : 0;
          }
        }
      },
    },
    {
      id: 'adv-corrupted',
      name: 'Stochastic with adversarial corruptions',
      short: 'Corrupted',
      type: 'bernoulli', stationary: true, tags: ['adversarial', 'corruption'],
      description: 'Bernoulli arms with μ ~ U(0, 1), but an adversary corrupts the first C = ⌈4√T⌉ rounds: the best arm pays 0 and the worst arm pays 1. Regret is still measured against the true means.',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K);
        let best = 0, worst = 0;
        for (let a = 0; a < K; a++) {
          mu[a] = rng.uniform();
          if (mu[a] > mu[best]) best = a;
          if (mu[a] < mu[worst]) worst = a;
        }
        const C = Math.min(T, Math.ceil(4 * Math.sqrt(T)));
        for (let t = 0; t < T; t++) {
          for (let a = 0; a < K; a++) {
            means[t * K + a] = mu[a];
            let r = rng.uniform() < mu[a] ? 1 : 0;
            if (t < C) { if (a === best) r = 0; else if (a === worst) r = 1; }
            rewards[t * K + a] = r;
          }
        }
      },
    },
  ];

  // ------------------------------------------------------------ helpers ----
  const families = [
    {
      id: 'ftrl', name: 'FTRL, mirror descent & perturbed leaders', before: 'nonstationary',
      blurb: 'Adversarial methods built from one recipe: regularize (or randomly perturb) the cumulative loss estimates and play the minimizer. The regularizer decides the trade-offs.',
    },
  ];

  /**
   * Solve Σᵢ 1/(dᵢ + cᵢλ) = 1 for λ with every dᵢ + cᵢλ > 0 (cᵢ > 0), then write
   * pᵢ = 1/(dᵢ + cᵢλ). This is the normalization of every log-barrier FTRL/OMD step.
   * The left side is convex and decreasing in λ, so Newton started from a point where it is ≥ 1
   * increases monotonically to the root.
   */
  function logBarrierNormalize(d, c, p) {
    const K = d.length;
    let lam = -Infinity;
    for (let i = 0; i < K; i++) { const v = (1 - d[i]) / c[i]; if (v > lam) lam = v; }
    for (let it = 0; it < 100; it++) {
      let f = -1, g = 0;
      for (let i = 0; i < K; i++) { const q = 1 / (d[i] + c[i] * lam); f += q; g += c[i] * q * q; }
      if (f < 1e-13) break;
      lam += f / g;
    }
    let S = 0;
    for (let i = 0; i < K; i++) { p[i] = 1 / (d[i] + c[i] * lam); S += p[i]; }
    for (let i = 0; i < K; i++) p[i] /= S;
    return lam;
  }

  /**
   * Solve Σᵢ 1/(wᵢ + λ)² = s for λ with wᵢ + λ > 0, then write pᵢ ∝ 1/(wᵢ + λ)² (summing to 1).
   * Normalization step of ½-Tsallis (Poly-INF, q = 2) maps; monotone Newton as above.
   */
  function tsallisNormalize(w, p, s) {
    const K = w.length;
    let mn = Infinity;
    for (let i = 0; i < K; i++) if (w[i] < mn) mn = w[i];
    let lam = 1 / Math.sqrt(s) - mn;
    for (let it = 0; it < 100; it++) {
      let f = -s, g = 0;
      for (let i = 0; i < K; i++) { const q = 1 / (w[i] + lam); const q2 = q * q; f += q2; g += 2 * q2 * q; }
      if (f < 1e-13 * s) break;
      lam += f / g;
    }
    let S = 0;
    for (let i = 0; i < K; i++) { const q = 1 / (w[i] + lam); p[i] = q * q; S += p[i]; }
    for (let i = 0; i < K; i++) p[i] /= S;
    return lam;
  }

  /** Gauss–Legendre nodes and weights on [0, 1]. */
  function gaussLegendre01(n) {
    const x = new Float64Array(n), w = new Float64Array(n);
    for (let i = 0; i < Math.ceil(n / 2); i++) {
      let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5)), pp = 1;
      for (let it = 0; it < 100; it++) {
        let p1 = 1, p2 = 0;
        for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j; }
        pp = (n * (z * p1 - p2)) / (z * z - 1);
        const z1 = z;
        z = z1 - p1 / pp;
        if (Math.abs(z - z1) < 1e-15) break;
      }
      x[i] = (1 - z) / 2; x[n - 1 - i] = (1 + z) / 2;
      w[i] = w[n - 1 - i] = 1 / ((1 - z * z) * pp * pp);
    }
    return { x, w };
  }
  const GL = gaussLegendre01(24); // exact for polynomials of degree ≤ 47, i.e. K ≤ 48

  /**
   * Probability that arm a wins argminᵢ ηLᵢ − Zᵢ with Zᵢ ~ Exp(1) i.i.d.:
   *   πₐ = sₐ ∫₀¹ Πⱼ≠ₐ (1 − sⱼv) dv,   sⱼ = exp(−η(Lⱼ − min L)) ∈ (0, 1].
   * The integrand is a polynomial of degree K − 1, so Gauss–Legendre quadrature is exact for K ≤ 48.
   */
  function expPerturbProb(L, eta, a, sv) {
    const K = L.length;
    let mn = Infinity;
    for (let i = 0; i < K; i++) if (L[i] < mn) mn = L[i];
    for (let i = 0; i < K; i++) sv[i] = Math.exp(-eta * (L[i] - mn));
    let I = 0;
    for (let k = 0; k < GL.x.length; k++) {
      const v = GL.x[k];
      let prod = 1;
      for (let j = 0; j < K; j++) if (j !== a) prod *= 1 - sv[j] * v;
      I += GL.w[k] * prod;
    }
    return Math.max(1e-12, sv[a] * I);
  }

  // Perturbation samplers (U ∈ [0, 1)).
  const expo = (rng) => -Math.log(1 - rng.uniform());
  const frechet2 = (rng) => 1 / Math.sqrt(-Math.log(rng.uniform()) || 1e-300);
  const gumbel = (rng) => -Math.log(-Math.log(rng.uniform()));

  /**
   * Follow the perturbed leader with geometric resampling (Neu & Bartók 2013).
   * Plays argminᵢ ηₜL̂ᵢ − Zᵢ; the loss estimate of the played arm is ℓ·M, where M counts fresh
   * perturbation draws until the same arm wins again (E[M] = 1/pₐ, capped at `cap`).
   */
  function ftplGR(K, rng, env, sample, eta, cap) {
    const L = new Float64Array(K), order = Int32Array.from({ length: K }, (_, i) => i);
    let curEta = 0;
    // One fresh perturbation draw: does arm a win? Rivals are tried in order of increasing L̂
    // (most likely winners first), so the draw usually stops after a few samples.
    function wins(a, e) {
      const va = e * L[a] - sample(rng);
      for (let k = 0; k < K; k++) {
        const j = order[k];
        if (j !== a && e * L[j] - sample(rng) < va) return false;
      }
      return true;
    }
    return {
      select(t) {
        curEta = eta(t);
        let b = 0, bv = Infinity;
        for (let i = 0; i < K; i++) { const v = curEta * L[i] - sample(rng); if (v < bv) { bv = v; b = i; } }
        return b;
      },
      update(a, r, t) {
        const loss = 1 - env.normalize(r);
        if (loss <= 0) return;
        const M = cap(t);
        let m = 1;
        while (m < M && !wins(a, curEta)) m++;
        L[a] += loss * m;
        // L̂ₐ only grew: move a towards the back of the order (insertion step)
        let k = order.indexOf(a);
        while (k + 1 < K && L[order[k + 1]] < L[a]) { order[k] = order[k + 1]; k++; }
        order[k] = a;
      },
    };
  }

  // --------------------------------------------------------- algorithms ----
  const algorithms = [
    // ---- adversarial
    {
      id: 'exp3-1', name: 'EXP3.1', family: 'adversarial',
      params: [],
      formula: 'EXP3 in epochs r = 0,1,…:  γᵣ = 2⁻ʳ,  restart when maxᵢ Ĝᵢ > gᵣ − K/γᵣ,  gᵣ = K ln K·4ʳ/(e−1)',
      regret: 'O(√(G* K ln K)) without knowing T or G*',
      idea: 'Auer et al.\'s anytime EXP3: guess the best arm\'s total reward, run EXP3 tuned to the guess, and quadruple the guess (restarting the weights) once the estimated rewards exceed it.',
      create(K, { rng, env }) {
        const lw = new Float64Array(K), p = new Float64Array(K), G = new Float64Array(K);
        const g0 = (K * Math.log(K)) / (Math.E - 1);
        let r = 0, gamma = 1, maxG = 0;
        return {
          select() {
            while (maxG > g0 * 4 ** r - K / gamma) { r++; gamma = Math.min(1, 2 ** -r); lw.fill(0); }
            softmaxInto(p, lw, 1);
            for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma / K;
            return sampleIndex(p, rng);
          },
          update(a, rew) {
            const xh = env.normalize(rew) / p[a];
            G[a] += xh; if (G[a] > maxG) maxG = G[a];
            lw[a] += (gamma * xh) / K;
          },
        };
      },
    },
    {
      id: 'exp3pp', name: 'EXP3++', family: 'adversarial',
      params: [{ key: 'c', label: 'c (gap exploration)', default: 18, min: 0, max: 50, step: 0.5 }],
      formula: 'πₐ = (1−Σε)·e^{−ηL̃ₐ}/Σe^{−ηL̃} + εₐ,  εₐ = min(1/2K, β, c ln²t/(tΔ̂ₐ²)),  η = β = ½√(ln K/tK)',
      regret: 'O(√(KT ln K)) adversarial;  O(Σ ln³T/Δₐ) stochastic',
      idea: 'Seldin & Slivkins: EXP3 whose per-arm exploration shrinks as the empirical gap Δ̂ₐ from the importance-weighted losses grows, so it gets logarithmic regret when the arms are stochastic.',
      create(K, { rng, env, params }) {
        const L = new Float64Array(K), q = new Float64Array(K), p = new Float64Array(K), eps = new Float64Array(K);
        const c = params.c;
        return {
          select(t) {
            const beta = 0.5 * Math.sqrt(Math.log(K) / (t * K));
            let mn = Infinity;
            for (let a = 0; a < K; a++) if (L[a] < mn) mn = L[a];
            const lt2 = Math.log(t) ** 2;
            let se = 0;
            for (let a = 0; a < K; a++) {
              const gap = Math.min(1, (L[a] - mn) / t);
              const xi = gap > 0 ? (c * lt2) / (t * gap * gap) : Infinity;
              eps[a] = Math.min(1 / (2 * K), beta, xi);
              se += eps[a];
            }
            softmaxInto(q, L, -beta);
            for (let a = 0; a < K; a++) p[a] = (1 - se) * q[a] + eps[a];
            return sampleIndex(p, rng);
          },
          update(a, r) { L[a] += (1 - env.normalize(r)) / p[a]; },
        };
      },
    },
    {
      id: 'sao', name: 'SAO (simplified)', family: 'adversarial',
      params: [],
      formula: 'successive elimination + probes of eliminated arms;  on a failed consistency test switch to EXP3 (ηₜ = √(2 ln K / Kt))',
      regret: 'O(polylog T · Σ 1/Δₐ) stochastic;  Õ(√(KT)) adversarial (original SAO)',
      idea: 'Bubeck & Slivkins\' stochastic-and-adversarial-optimal recipe, simplified: play successive elimination, keep probing eliminated arms at rate ~ ln T/(tΔ̂²), and fall back to EXP3 for good as soon as recent rewards (of one arm, or pooled over a round) disagree with their history or an eliminated arm overtakes the leader.',
      create(K, { rng, env, T }) {
        const n = new Float64Array(K), s = new Float64Array(K), active = new Uint8Array(K).fill(1);
        const pre = Array.from({ length: K }, () => [0]);   // per-arm prefix sums of its rewards (O(T) in total)
        const pn = new Float64Array(K), ps = new Float64Array(K), gap = new Float64Array(K), q = new Float64Array(K);
        const lw = new Float64Array(K), p = new Float64Array(K);
        const bElim = Math.log(K * T), bTest = Math.log(T) + 1, lnT = Math.log(Math.max(T, 3));
        const rad = (b, m) => Math.sqrt(b / (2 * m));
        let adversarial = false, tSwitch = 0, nActive = K, probing = false;
        let pool = [0];   // prefix sums of the round-robin rewards since the last elimination
        const shifted = (pre, N, m) => {
          const old = pre[N - m];
          return Math.abs((pre[N] - old) / m - old / (N - m)) > rad(bTest, m) + rad(bTest, N - m);
        };
        function leader() {
          let b = -1, bv = -Infinity;
          for (let a = 0; a < K; a++) if (active[a] && n[a] > 0 && s[a] / n[a] > bv) { bv = s[a] / n[a]; b = a; }
          return b;
        }
        return {
          /** Step at which the consistency tests failed and EXP3 took over (0 = still stochastic). */
          get switchedAt() { return adversarial ? tSwitch : 0; },
          get activeArms() { return nActive; },
          select(t) {
            if (adversarial) {
              softmaxInto(p, lw, Math.sqrt((2 * Math.log(K)) / (K * (t - tSwitch))));
              return sampleIndex(p, rng);
            }
            probing = false;
            if (nActive < K) {
              let u = rng.uniform();
              for (let a = 0; a < K; a++) {
                if (active[a]) continue;
                q[a] = Math.min(1 / (2 * K), (4 * lnT) / (t * gap[a] * gap[a]));
                if (u < q[a]) { probing = true; return a; }
                u -= q[a];
              }
            }
            let b = -1, bn = Infinity;
            for (let a = 0; a < K; a++) if (active[a] && n[a] < bn) { bn = n[a]; b = a; }
            return b;
          },
          update(a, r, t) {
            const x = env.normalize(r);
            if (adversarial) { lw[a] -= (1 - x) / p[a]; return; }
            let fail = false;
            if (probing) {
              pn[a]++; ps[a] += x;
              const l = leader();
              if (ps[a] / pn[a] - rad(bTest, pn[a]) > s[l] / n[l] + rad(bTest, n[l])) fail = true;
            } else {
              n[a]++; s[a] += x;
              pre[a].push(s[a]);
              // change tests: this arm's last m rewards against its earlier ones (m = 4, 8, 16, …),
              // and the same on the pooled round-robin stream (blocks of whole rounds), which
              // catches shifts that move every arm together
              for (let m = 4; 2 * m <= n[a] && !fail; m *= 2) fail = shifted(pre[a], n[a], m);
              pool.push(pool[pool.length - 1] + x);
              const P = pool.length - 1;
              for (let m = 4 * nActive; 2 * m <= P && !fail; m *= 2) fail = shifted(pool, P, m);
              // elimination
              const l = leader();
              if (!fail && nActive > 1 && l >= 0) {
                const lo = s[l] / n[l] - rad(bElim, n[l]);
                for (let j = 0; j < K; j++) {
                  if (!active[j] || j === l || n[j] === 0) continue;
                  if (s[j] / n[j] + rad(bElim, n[j]) < lo) {
                    active[j] = 0; nActive--; pool = [0];
                    gap[j] = Math.max(s[l] / n[l] - s[j] / n[j], 1e-3);
                  }
                }
              }
            }
            if (fail) { adversarial = true; tSwitch = t; }
          },
        };
      },
    },

    // ---- FTRL, mirror descent and perturbed leaders
    {
      id: 'ftrl-logbarrier', name: 'FTRL, log-barrier', family: 'ftrl',
      params: [],
      formula: 'π = argmin ⟨π, L̂⟩ + (1/ηₜ)Σ −ln πₐ  ⇒  πₐ = 1/(ηₜ(L̂ₐ − λ)),   ηₜ = min(½, √(K ln(t+1)/t))',
      regret: 'O(√(KT log T)) adversarial; small-loss and stochastic adaptivity',
      idea: 'Follow the regularized leader with the log-barrier instead of the negative entropy (which would give EXP3). Its polynomial tails keep exploring more and handle very small probabilities stably.',
      create(K, { rng, env }) {
        const L = new Float64Array(K), d = new Float64Array(K), c = new Float64Array(K).fill(1), p = new Float64Array(K);
        return {
          select(t) {
            const eta = Math.min(0.5, Math.sqrt((K * Math.log(t + 1)) / t));
            let mn = Infinity;
            for (let a = 0; a < K; a++) if (L[a] < mn) mn = L[a];
            for (let a = 0; a < K; a++) d[a] = eta * (L[a] - mn);
            logBarrierNormalize(d, c, p);
            return sampleIndex(p, rng);
          },
          update(a, r) { L[a] += (1 - env.normalize(r)) / p[a]; },
        };
      },
    },
    {
      id: 'omd-tsallis', name: 'OMD, Tsallis mirror map', family: 'ftrl',
      params: [{ key: 'c', label: 'c (step scale)', default: 2, min: 0.05, max: 5, step: 0.05 }],
      formula: '1/√πₐ ← 1/√πₐ + ηₜ ℓ̂ₐ + λ  (Bregman projection),   ηₜ = c/√(t),  ℓ̂ = ℓ/πₐ',
      regret: 'O(√(KT)) adversarial',
      idea: 'Online mirror descent with the ½-Tsallis entropy as mirror map: a gradient step in the dual 1/√π coordinates, then a projection back to the simplex. Unlike Tsallis-INF (FTRL) the shrinking step scales each new loss rather than the whole history, so early losses weigh more and it can be slow to recover after the leader changes.',
      create(K, { rng, env, params }) {
        const w = new Float64Array(K).fill(Math.sqrt(K)), p = new Float64Array(K).fill(1 / K);
        return {
          select: () => sampleIndex(p, rng),
          update(a, r, t) {
            const loss = 1 - env.normalize(r);
            if (loss <= 0) return;
            for (let i = 0; i < K; i++) w[i] = 1 / Math.sqrt(p[i]);
            w[a] += ((params.c / Math.sqrt(t)) * loss) / p[a];
            tsallisNormalize(w, p, 1);
            for (let i = 0; i < K; i++) if (p[i] < 1e-300) p[i] = 1e-300;
          },
        };
      },
    },
    {
      id: 'poly-inf', name: 'Poly INF (q = 2)', family: 'ftrl',
      params: [],
      formula: 'πₐ = (η / (C − G̃ₐ))² + γ/K,  C normalizes π,   η = √(5T),  γ = min(½, √(3K/T))',
      regret: 'O(√(KT)) minimax (no log K)',
      idea: 'Audibert & Bubeck\'s Implicitly Normalized Forecaster on importance-weighted gains with ψ(x) = (η/−x)² + γ/K. Equivalent to FTRL with the ½-Tsallis entropy, the first algorithm to remove the √log K factor.',
      create(K, { rng, env, T }) {
        const G = new Float64Array(K), w = new Float64Array(K), p = new Float64Array(K);
        const eta = Math.sqrt(5 * T), gamma = Math.min(0.5, Math.sqrt((3 * K) / T));
        return {
          select() {
            let mx = -Infinity;
            for (let a = 0; a < K; a++) if (G[a] > mx) mx = G[a];
            for (let a = 0; a < K; a++) w[a] = (mx - G[a]) / eta;   // (C − Gₐ)/η up to the shift λ
            tsallisNormalize(w, p, 1 - gamma);
            for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma / K;
            return sampleIndex(p, rng);
          },
          update(a, r) { G[a] += env.normalize(r) / p[a]; },
        };
      },
    },
    {
      id: 'broad-omd', name: 'BROAD-OMD', family: 'ftrl',
      params: [{ key: 'eta', label: 'η₀ (0 = auto)', default: 0, min: 0, max: 0.5, step: 0.005 }],
      formula: 'log-barrier OMD with optimistic mₜ;  ℓ̂ = m + (ℓ−m)1ₐ/πₐ;  ηₜ,ₐ ← κηₜ,ₐ when 1/πₐ > ρₐ,  κ = e^{1/ln T}',
      regret: 'Õ(√(K Σₜ (ℓₜ,ᵢₜ − mₜ,ᵢₜ)²)): small-loss, variance and path-length bounds',
      idea: 'Wei & Luo\'s optimistic log-barrier mirror descent: it predicts each loss by the arm\'s running mean, and raises an arm\'s own learning rate whenever its probability hits a new low. That gives data-dependent bounds (here: variance), much smaller than √(KT) on benign sequences. The default η₀ = min(¼, √(K ln T/T)) is far above the analysis\' 1/162, for practical speed.',
      create(K, { rng, env, params, T }) {
        const wp = new Float64Array(K).fill(1 / K), w = new Float64Array(K), eta = new Float64Array(K), rho = new Float64Array(K).fill(2 * K);
        const m = new Float64Array(K), cnt = new Float64Array(K), d = new Float64Array(K), c = new Float64Array(K), tmp = new Float64Array(K);
        const lnT = Math.log(Math.max(T, 3)), kappa = Math.exp(1 / lnT);
        eta.fill(params.eta > 0 ? params.eta : Math.min(1 / 4, Math.sqrt((K * lnT) / T)));
        return {
          select() {
            // wₜ = argmin ⟨w, mₜ⟩ + D(w, w′ₜ):  1/(ηᵢwᵢ) = 1/(ηᵢw′ᵢ) + mᵢ + λ
            for (let i = 0; i < K; i++) { d[i] = 1 / wp[i] + eta[i] * m[i]; c[i] = eta[i]; }
            logBarrierNormalize(d, c, w);
            return sampleIndex(w, rng);
          },
          update(a, r) {
            const loss = 1 - env.normalize(r);
            // w′ₜ₊₁ = argmin ⟨w, ℓ̂ + corr⟩ + D(w, w′ₜ),  corrₐ = 6ηₐwₐ(ℓ̂ₐ − mₐ)²
            for (let i = 0; i < K; i++) {
              let v = m[i];
              if (i === a) {
                const dv = (loss - m[a]) / w[a];
                v += dv + 6 * eta[a] * w[a] * dv * dv;
              }
              d[i] = 1 / wp[i] + eta[i] * v; c[i] = eta[i];
            }
            logBarrierNormalize(d, c, tmp);
            for (let i = 0; i < K; i++) {
              wp[i] = Math.max(tmp[i], 1e-300);
              if (1 / wp[i] > rho[i]) { rho[i] = 2 / wp[i]; eta[i] = Math.min(0.5, eta[i] * kappa); }
            }
            cnt[a]++; m[a] += (loss - m[a]) / cnt[a];
          },
        };
      },
    },
    {
      id: 'fpl', name: 'FPL (exponential)', family: 'ftrl',
      params: [],
      formula: 'play argminₐ ηₜL̂ₐ − Zₐ,  Z ~ Exp(1);  L̂ₐ += ℓ/πₐ,  πₐ = sₐ∫₀¹Πⱼ≠ₐ(1 − sⱼv)dv,  sⱼ = e^{−ηₜ(L̂ⱼ−min L̂)}',
      regret: 'O(√(KT log K))',
      idea: 'Kalai & Vempala\'s follow the perturbed leader with one-sided exponential noise. For exponential noise the probability of playing an arm is a one-dimensional polynomial integral, computed exactly by quadrature, so the loss estimate is unbiased.',
      create(K, { rng, env }) {
        const L = new Float64Array(K), sv = new Float64Array(K);
        let eta = 0;
        return {
          select(t) {
            eta = Math.sqrt((2 * Math.log(K)) / (K * t));
            let b = 0, bv = Infinity;
            for (let i = 0; i < K; i++) { const v = eta * L[i] - expo(rng); if (v < bv) { bv = v; b = i; } }
            return b;
          },
          update(a, r) {
            const loss = 1 - env.normalize(r);
            if (loss > 0) L[a] += loss / expPerturbProb(L, eta, a, sv);
          },
        };
      },
    },
    {
      id: 'fpl-gr', name: 'FPL-GR', family: 'ftrl',
      params: [],
      formula: 'argminₐ ηL̂ₐ − Zₐ,  Z ~ Exp(1);  L̂ₐ += ℓ·M,  M = redraws until a wins again (≤ ⌈√(KT)⌉),  η = √(ln K/KT)',
      regret: 'O(√(KT log K))',
      idea: 'Neu & Bartók\'s geometric resampling: instead of computing the probability of the played arm, redraw the perturbations until that arm wins again. The number of redraws is an unbiased estimate of 1/πₐ, which makes FPL usable for bandits (and combinatorial bandits) without ever computing π.',
      create(K, { rng, env, T }) {
        const eta = Math.sqrt(Math.log(K) / (K * T)), M = Math.ceil(Math.sqrt(K * T));
        return ftplGR(K, rng, env, expo, () => eta, () => M);
      },
    },
    {
      id: 'ftpl-frechet', name: 'FTPL, Fréchet', family: 'ftrl',
      params: [{ key: 'c', label: 'c (rate scale)', default: 2, min: 0.05, max: 5, step: 0.05 }],
      formula: 'argminₐ ηₜL̂ₐ − Zₐ,  Z ~ Fréchet(2),  ηₜ = c/√t;  ℓ̂ by geometric resampling',
      regret: 'Best of both worlds: O(√(KT)) adversarial, O(Σ log T/Δₐ) stochastic',
      idea: 'Honda, Ito & Tsuchiya (after Kim & Tewari): with heavy-tailed Fréchet perturbations of shape 2, FTPL mimics Tsallis-INF and is optimal in both regimes, while only ever needing an argmin.',
      create(K, { rng, env, params, T }) {
        const M = Math.ceil(K * Math.sqrt(T));
        return ftplGR(K, rng, env, frechet2, (t) => params.c / Math.sqrt(t), () => M);
      },
    },
    {
      id: 'ftpl-gumbel', name: 'FTPL, Gumbel', family: 'ftrl',
      params: [],
      formula: 'argminₐ ηₜL̂ₐ − Zₐ,  Z ~ Gumbel,  ηₜ = √(2 ln K/Kt);  ℓ̂ by geometric resampling',
      regret: 'O(√(KT log K))',
      idea: 'With Gumbel noise the perturbed leader is exactly an exponential-weights draw (the Gumbel-max trick), so this is anytime EXP3 without forced exploration, using geometric resampling instead of 1/π.',
      create(K, { rng, env, T }) {
        const M = Math.ceil(K * Math.sqrt(T));
        return ftplGR(K, rng, env, gumbel, (t) => Math.sqrt((2 * Math.log(K)) / (K * t)), () => M);
      },
    },
  ];

  MAB.register({ families, environments, algorithms });
  return { environments, algorithms, helpers: { logBarrierNormalize, tsallisNormalize, expPerturbProb } };
});
