/*
 * nonstationary.js — change-detecting, multi-scale and rested (rotting / rising) bandits.
 *
 * Environments
 *   rotting          rested: each arm's mean falls with its own pull count
 *   rising           rested: each arm's mean follows a concave learning curve in its pull count
 *   piecewise-local  piecewise-stationary Bernoulli where only the current best arm changes (breaks)
 *
 * Algorithms
 *   family 'nonstationary': SW-TS, SW-klUCB, EXP3.R, CUSUM-UCB, PHT-UCB, GLR-klUCB (local restarts),
 *                           AdSwitch (simplified), MASTER (simplified, UCB1 base)
 *   family 'rested':        FEWA, RAW-UCB, R-ed-UCB
 *
 * Change detectors (CUSUM, Page–Hinkley) work on z = r / (2σ), which puts Bernoulli rewards and
 * Gaussian rewards of any σ on the same variance scale (≤ ¼), so one threshold fits both.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { util } = MAB;
  const { stats, record, argmax, untried, sampleIndex, softmaxInto, klBern, klIndex, samplePosterior } = util;

  const clamp01 = (x) => Math.min(1, Math.max(0, x));

  // ------------------------------------------------------------ environments ----
  const ENVIRONMENTS = [
    {
      id: 'rotting',
      name: 'Rotting arms (rested)',
      short: 'Rotting',
      type: 'gaussian', sigma: 0.1, stationary: false, rested: true, tags: ['rested'],
      normalize: (r) => clamp01((r + 0.3) / 1.6),
      description: 'Arms wear out with use: arm a pays N(μₐ, 0.1²), μₐ ~ U(0.3, 1), until it has been pulled mₐ times, then its mean drops by dₐ ~ U(0.3, 0.8) for good. The wear-out points mₐ are log-uniform around T/(2K) pulls. Pulling whichever arm is best right now (the greedy oracle) is optimal; an average over all past pulls keeps overrating a worn-out arm.',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K), drop = new Float64Array(K), at = new Float64Array(K);
        for (let a = 0; a < K; a++) {
          mu[a] = 0.3 + 0.7 * rng.uniform();
          drop[a] = 0.3 + 0.5 * rng.uniform();
          at[a] = Math.max(1, Math.floor((T / (2 * K)) * Math.exp(2 * rng.uniform() - 1)));
        }
        for (let n = 0; n < T; n++) {
          for (let a = 0; a < K; a++) {
            const m = n < at[a] ? mu[a] : Math.max(0, mu[a] - drop[a]);
            means[n * K + a] = m;
            rewards[n * K + a] = m + 0.1 * rng.normal();
          }
        }
      },
    },
    {
      id: 'rising',
      name: 'Rising arms (rested)',
      short: 'Rising',
      type: 'gaussian', sigma: 0.1, stationary: false, rested: true, tags: ['rested'],
      normalize: (r) => clamp01((r + 0.3) / 1.6),
      description: 'Arms improve with use, like models that learn from training: the n-th pull of arm a has mean cₐ·(1 − e^(−(n+1)/τₐ)), rewards N(mean, 0.1²). One late bloomer has a high ceiling (c ~ U(0.85, 1)) but learns slowly (τ ~ U(T/40, T/8)); the others learn fast (τ log-uniform in [T/500, T/50]) to modest ceilings (c ~ U(0.3, 0.7)). Committing to the late bloomer is optimal, but early on it looks like the worst arm.',
      fill(K, T, rng, means, rewards) {
        const c = new Float64Array(K), tau = new Float64Array(K), bloomer = rng.int(K);
        for (let a = 0; a < K; a++) {
          if (a === bloomer) { c[a] = 0.85 + 0.15 * rng.uniform(); tau[a] = T * (1 / 40 + (1 / 8 - 1 / 40) * rng.uniform()); }
          else { c[a] = 0.3 + 0.4 * rng.uniform(); tau[a] = Math.max(1, (T / 500) * Math.exp(Math.log(10) * rng.uniform())); }
        }
        for (let n = 0; n < T; n++) {
          for (let a = 0; a < K; a++) {
            const m = c[a] * (1 - Math.exp(-(n + 1) / tau[a]));
            means[n * K + a] = m;
            rewards[n * K + a] = m + 0.1 * rng.normal();
          }
        }
      },
    },
    {
      id: 'piecewise-local',
      name: 'Bernoulli, local changes',
      short: 'Local changes',
      type: 'bernoulli', stationary: false, tags: ['changes'],
      description: 'Piecewise-stationary Bernoulli arms, μ ~ U(0.1, 0.9) at the start. At four evenly spaced change points only one arm moves: the current best arm breaks and drops to μ ~ U(0.05, 0.25). The other arms never change, so a per-arm change detector that restarts only the broken arm keeps everything else it has learned.',
      fill(K, T, rng, means, rewards) {
        const mu = new Float64Array(K);
        for (let a = 0; a < K; a++) mu[a] = 0.1 + 0.8 * rng.uniform();
        const segment = Math.ceil(T / 5);
        for (let t = 0; t < T; t++) {
          if (t > 0 && t % segment === 0) {
            let b = 0;
            for (let a = 1; a < K; a++) if (mu[a] > mu[b]) b = a;
            mu[b] = 0.05 + 0.2 * rng.uniform();
          }
          for (let a = 0; a < K; a++) {
            means[t * K + a] = mu[a];
            rewards[t * K + a] = rng.uniform() < mu[a] ? 1 : 0;
          }
        }
      },
    },
  ];

  // ---------------------------------------------------------------- helpers ----
  /** Ring-buffer of the last W plays shared by the sliding-window methods. */
  function slidingWindow(K, W) {
    const n = new Float64Array(K), sum = new Float64Array(K), histA = new Int32Array(W), histR = new Float64Array(W);
    let filled = 0, head = 0;
    return {
      n, sum,
      push(a, r) {
        if (filled === W) { const oa = histA[head]; n[oa] -= 1; sum[oa] -= histR[head]; if (n[oa] < 0.5) { n[oa] = 0; sum[oa] = 0; } }
        else filled++;
        histA[head] = a; histR[head] = r; head = (head + 1) % W;
        n[a] += 1; sum[a] += r;
      },
    };
  }

  /** Per-arm sequential change detector on z = r/(2σ): CUSUM (Liu et al.) or Page–Hinkley. */
  function makeDetector(kind, K, M, eps, h) {
    const k = new Int32Array(K), ref = new Float64Array(K), gp = new Float64Array(K), gm = new Float64Array(K);
    return {
      reset(a) { k[a] = 0; ref[a] = 0; gp[a] = 0; gm[a] = 0; },
      /** Feed one sample; returns true on an alarm. */
      push(a, z) {
        if (kind === 'cusum') {
          if (k[a] < M) { ref[a] += z; k[a]++; if (k[a] === M) ref[a] /= M; return false; }
        } else {
          k[a]++; ref[a] += (z - ref[a]) / k[a];
        }
        gp[a] = Math.max(0, gp[a] + z - ref[a] - eps);
        gm[a] = Math.max(0, gm[a] + ref[a] - z - eps);
        return gp[a] > h || gm[a] > h;
      },
    };
  }

  function detectorUcb(kind) {
    return function create(K, { rng, params, env, T }) {
      const ups = Math.max(1, params.changes), lTU = Math.log(Math.max(Math.E, T / ups));
      const h = lTU, alpha = Math.min(1, Math.sqrt((ups / T) * lTU));
      const det = makeDetector(kind, K, Math.max(1, Math.round(params.M || 1)), params.eps, h);
      const s = stats(K), idx = new Float64Array(K), c = Math.SQRT2 * env.scale, z2 = 1 / (2 * env.sigma);
      return {
        select() {
          if (rng.uniform() < alpha) return rng.int(K);
          const u = untried(s.n); if (u >= 0) return u;
          let nt = 0;
          for (let a = 0; a < K; a++) nt += s.n[a];
          const l = Math.log(nt);
          for (let a = 0; a < K; a++) idx[a] = s.mean[a] + c * Math.sqrt(l / s.n[a]);
          return argmax(idx, rng);
        },
        update(a, r) {
          record(s, a, r);
          if (det.push(a, r * z2)) { s.n[a] = 0; s.sum[a] = 0; s.mean[a] = 0; det.reset(a); }
        },
      };
    };
  }

  /** UCB1 on rewards in [0, 1]; f̃ is its optimistic estimate of the best mean (MASTER's base). */
  function ucbInstance(K) {
    return { n: new Float64Array(K), sum: new Float64Array(K), t: 0 };
  }

  // ------------------------------------------------------------- algorithms ----
  const NS = 'nonstationary', RESTED = 'rested';

  const ALGORITHMS = [
    {
      id: 'sw-ts', name: 'Sliding-window Thompson', family: NS,
      params: [{ key: 'window', label: 'τ (window)', default: 500, min: 10, max: 5000, step: 10 }],
      formula: 'θₐ ~ posterior of arm a from the last τ plays;  argmaxₐ θₐ',
      regret: 'Õ(√(ΓT)) with Γ change points (Trovò et al. 2020)',
      idea: 'Thompson sampling that only remembers the last τ plays, so a posterior that has collapsed on a stale arm reopens once its old evidence slides out.',
      create(K, { rng, params, env }) {
        const W = Math.max(1, Math.round(params.window)), win = slidingWindow(K, W), th = new Float64Array(K);
        const bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select() {
            for (let a = 0; a < K; a++) {
              const n = win.n[a], sm = bern ? Math.min(n, Math.max(0, win.sum[a])) : win.sum[a];
              th[a] = samplePosterior(rng, bern, s2, sm, n);
            }
            return argmax(th, rng);
          },
          update: (a, r) => win.push(a, r),
        };
      },
    },
    {
      id: 'sw-klucb', name: 'Sliding-window KL-UCB', family: NS,
      params: [{ key: 'window', label: 'τ (window)', default: 500, min: 10, max: 5000, step: 10 }],
      formula: 'max { q : Nₐ(τ)·kl(X̄ₐ(τ), q) ≤ ln min(t, τ) },  statistics over the last τ plays',
      regret: 'Õ(√(ΓT)) with Γ change points',
      idea: 'Garivier & Moulines\' sliding window with the KL-UCB index instead of Hoeffding: a tighter interval for Bernoulli arms, and forgetting so it can follow a moving best arm.',
      create(K, { rng, params, env }) {
        const W = Math.max(1, Math.round(params.window)), win = slidingWindow(K, W), idx = new Float64Array(K);
        const bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        return {
          select(t) {
            const u = untried(win.n); if (u >= 0) return u;
            const bound = Math.log(Math.min(t, W));
            for (let a = 0; a < K; a++) {
              const m = win.sum[a] / win.n[a];
              idx[a] = klIndex(bern, bern ? clamp01(m) : m, win.n[a], bound, s2);
            }
            return argmax(idx, rng);
          },
          update: (a, r) => win.push(a, r),
        };
      },
    },
    {
      id: 'exp3r', name: 'EXP3.R', family: NS,
      params: [
        { key: 'gamma', label: 'γ (exploration)', default: 0.2, min: 0.01, max: 1, step: 0.01 },
        { key: 'H', label: 'H (test interval)', default: 400, min: 20, max: 5000, step: 10 },
      ],
      formula: 'EXP3;  every H steps reset if ∃k: μ̂ₖ − εₖ > μ̂ₖₘₐₓ + εₖₘₐₓ  (μ̂ from uniform-exploration pulls only)',
      regret: 'O(Γ·√(K T log T)) with Γ changes (Allesiardo & Féraud 2015)',
      idea: 'EXP3 whose uniform-exploration pulls double as unbiased samples for a drift test: if some arm now confidently beats the arm EXP3 favours, the weights restart. Simplified: the confidence radius is Hoeffding on the actual sample counts, not the paper\'s fixed ε.',
      create(K, { rng, params, env, T }) {
        const gamma = params.gamma, H = Math.max(1, Math.round(params.H)), lsT = Math.log(Math.sqrt(Math.max(2, T)));
        const lw = new Float64Array(K), w = new Float64Array(K), p = new Float64Array(K);
        const cnt = new Float64Array(K), gs = new Float64Array(K), conf = (n) => Math.sqrt(lsT / (2 * n));
        let explored = false;
        function drift() {
          let kmax = 0;
          for (let a = 1; a < K; a++) if (lw[a] > lw[kmax]) kmax = a;
          if (!cnt[kmax]) return false;
          const lo = gs[kmax] / cnt[kmax] + conf(cnt[kmax]);
          for (let k = 0; k < K; k++) if (k !== kmax && cnt[k] && gs[k] / cnt[k] - conf(cnt[k]) > lo) return true;
          return false;
        }
        return {
          select(t) {
            if (t > 1 && (t - 1) % H === 0) {
              if (drift()) lw.fill(0);
              cnt.fill(0); gs.fill(0);
            }
            softmaxInto(w, lw, 1);
            for (let a = 0; a < K; a++) p[a] = (1 - gamma) * w[a] + gamma / K;
            explored = rng.uniform() < gamma;
            return explored ? rng.int(K) : sampleIndex(w, rng);
          },
          update(a, r) {
            const x = env.normalize(r);
            lw[a] += (gamma * x) / (K * p[a]);
            if (explored) { cnt[a]++; gs[a] += x; }
          },
        };
      },
    },
    {
      id: 'cusum-ucb', name: 'CUSUM-UCB', family: NS,
      params: [
        { key: 'changes', label: 'Υ (expected changes)', default: 5, min: 1, max: 50, step: 1 },
        { key: 'M', label: 'M (warm-up samples)', default: 50, min: 2, max: 200, step: 1 },
        { key: 'eps', label: 'ε (drift allowance)', default: 0.1, min: 0, max: 0.5, step: 0.01 },
      ],
      formula: 'UCB on samples since each arm\'s last alarm, uniform arm w.p. α;  alarm when g± = max(0, g± ± (z − u₀) − ε) > ln(T/Υ)',
      regret: 'O(√(Υ T log(T/Υ))) (Liu, Lee & Shroff 2018)',
      idea: 'Each arm runs a two-sided CUSUM test against the mean of its first M samples; an alarm resets only that arm. A trickle α = √(Υ ln(T/Υ)/T) of random pulls keeps watch on arms UCB has abandoned.',
      create: detectorUcb('cusum'),
    },
    {
      id: 'pht-ucb', name: 'PHT-UCB', family: NS,
      params: [
        { key: 'changes', label: 'Υ (expected changes)', default: 5, min: 1, max: 50, step: 1 },
        { key: 'eps', label: 'ε (drift allowance)', default: 0.1, min: 0, max: 0.5, step: 0.01 },
      ],
      formula: 'as CUSUM-UCB, but the Page–Hinkley test compares each sample with the running mean z̄ₖ',
      regret: 'O(√(Υ T log(T/Υ))) (Liu, Lee & Shroff 2018)',
      idea: 'The Page–Hinkley variant of CUSUM-UCB: no warm-up phase, the reference level is the running mean since the arm\'s last reset.',
      create: detectorUcb('pht'),
    },
    {
      id: 'glr-klucb', name: 'GLR-klUCB (local)', family: NS,
      params: [
        { key: 'changes', label: 'Υ (expected changes)', default: 5, min: 1, max: 50, step: 1 },
        { key: 'alpha0', label: 'α₀ (forced exploration)', default: 0.1, min: 0, max: 2, step: 0.01 },
      ],
      formula: 'kl-UCB since each arm\'s restart;  restart arm a if maxₛ [s·kl(μ̂₁:ₛ, μ̂) + (n−s)·kl(μ̂ₛ₊₁:ₙ, μ̂)] > ln(3n^{3/2}/δ)',
      regret: 'Õ(√(Υ K T log T)) (Besson, Kaufmann, Maillard & Seznec 2022)',
      idea: 'A Bernoulli (or Gaussian) generalized-likelihood-ratio test scans every split point of an arm\'s history since its last restart; on an alarm only that arm restarts (local restarts). α = α₀√(Υ ln T / T) forced exploration, δ = 1/√(ΥT). To stay fast the test runs every 5th sample and scans every 5th split.',
      create(K, { rng, params, env, T }) {
        const ups = Math.max(1, params.changes), bern = env.type === 'bernoulli', s2 = env.sigma * env.sigma;
        const delta = 1 / Math.sqrt(ups * T), alpha = Math.min(1, params.alpha0 * Math.sqrt((ups * Math.log(T)) / T));
        const per = alpha > 0 ? Math.max(K, Math.floor(K / alpha)) : 0, STEP = 5, L = T + 1;
        const pre = new Float64Array(K * L), n = new Int32Array(K), tau = new Int32Array(K), idx = new Float64Array(K);
        function glr(a) {
          const N = n[a], o = a * L, tot = pre[o + N], mu = tot / N, thr = Math.log((3 * N * Math.sqrt(N)) / delta);
          for (let s = STEP; s < N; s += STEP) {
            const m1 = pre[o + s] / s, m2 = (tot - pre[o + s]) / (N - s);
            const g = bern ? s * klBern(m1, mu) + (N - s) * klBern(m2, mu) : ((s * (N - s)) / N) * ((m1 - m2) * (m1 - m2)) / (2 * s2);
            if (g > thr) return true;
          }
          return false;
        }
        return {
          select(t) {
            if (per) { const ph = t % per; if (ph < K) return ph; }
            const u = untried(n); if (u >= 0) return u;
            for (let a = 0; a < K; a++) {
              const m = pre[a * L + n[a]] / n[a];
              idx[a] = klIndex(bern, bern ? clamp01(m) : m, n[a], Math.log(Math.max(1, t - tau[a])), s2);
            }
            return argmax(idx, rng);
          },
          update(a, r, t) {
            const o = a * L;
            pre[o + n[a] + 1] = pre[o + n[a]] + r; n[a]++;
            if (n[a] >= 2 * STEP && n[a] % STEP === 0 && glr(a)) { n[a] = 0; tau[a] = t; }
          },
        };
      },
    },
    {
      id: 'adswitch', name: 'AdSwitch (simplified)', family: NS,
      params: [{ key: 'rate', label: 'check rate', default: 0.1, min: 0, max: 2, step: 0.01 }],
      formula: 'round-robin over GOOD arms;  evict a when μ̂* − μ̂ₐ > r(n*) + r(nₐ);  w.p. ∝ 2⁻ⁱ re-sample evicted arms ~4ⁱ ln T times;  new episode on any detected change',
      regret: 'Õ(√(S K T)) with S switches, no knowledge of S (Auer, Gajane & Ortner 2019)',
      idea: 'Successive elimination that never fully trusts an eviction: evicted arms are re-checked at random at several resolutions, good arms are compared across windows, and any change restarts the episode. Simplified: window checks use dyadic suffixes of each arm\'s history, Hoeffding radii r(n) = σ√(2 ln T / n), only scales 2⁻ⁱ ≥ Δ̃ₐ/2 are checked, and the check probability is scaled by "check rate". After each restart it must re-eliminate arms from scratch, which is slow at T ≤ 10⁴.',
      create(K, { rng, params, env, T }) {
        const sig = env.sigma, lT = Math.log(Math.max(3, T)), L = T + 1;
        const rad = (m) => sig * Math.sqrt((2 * lT) / m);
        const pre = new Float64Array(K * L), N = new Int32Array(K), good = new Uint8Array(K);
        const gapE = new Float64Array(K), muE = new Float64Array(K), nE = new Int32Array(K);
        const obl = new Int32Array(K), oblStart = new Int32Array(K), cand = new Int32Array(K);
        let ep = 0, newEp = true;
        const mean = (a, from, to) => (pre[a * L + to] - pre[a * L + from]) / (to - from);
        function startEpisode() { ep++; N.fill(0); good.fill(1); obl.fill(0); newEp = false; }
        return {
          select() {
            if (newEp) startEpisode();
            const base = params.rate * Math.sqrt(ep / (K * T * lT));
            for (let a = 0; a < K; a++) {
              if (good[a] || obl[a] > 0 || base <= 0) continue;
              for (let i = 1, pw = 0.5; pw >= gapE[a] / 2 || i === 1; i++, pw /= 2) {
                if (rng.uniform() < pw * base) {
                  obl[a] = Math.ceil((8 * sig * sig * lT) / (pw * pw)); oblStart[a] = N[a]; break;
                }
                if (i > 20) break;
              }
            }
            let m = 0, best = Infinity;
            for (let a = 0; a < K; a++) {
              if (!good[a] && obl[a] === 0) continue;
              if (N[a] < best) { best = N[a]; m = 0; }
              if (N[a] === best) cand[m++] = a;
            }
            return cand[m > 1 ? rng.int(m) : 0];
          },
          update(a, r) {
            if (newEp) startEpisode();
            const o = a * L;
            pre[o + N[a] + 1] = pre[o + N[a]] + r; N[a]++;
            const n = N[a];
            if (obl[a] > 0 && --obl[a] === 0) {
              const k = n - oblStart[a];
              if (Math.abs(mean(a, oblStart[a], n) - muE[a]) > rad(k) + rad(nE[a])) newEp = true;
              return;
            }
            if (!good[a]) return;
            // change check: mean of the last w pulls vs the whole episode, w = 2, 4, 8, ...
            const all = mean(a, 0, n), ra = rad(n);
            for (let w = 2; w <= n / 2; w *= 2) {
              if (Math.abs(mean(a, n - w, n) - all) > rad(w) + ra) { newEp = true; return; }
            }
            // elimination among good arms
            let b = -1, bm = -Infinity;
            for (let i = 0; i < K; i++) if (good[i] && N[i] && mean(i, 0, N[i]) > bm) { bm = mean(i, 0, N[i]); b = i; }
            for (let i = 0; i < K; i++) {
              if (!good[i] || i === b || !N[i]) continue;
              const mi = mean(i, 0, N[i]), gap = bm - mi;
              if (gap > rad(N[i]) + rad(N[b])) { good[i] = 0; gapE[i] = gap; muE[i] = mi; nE[i] = N[i]; }
            }
          },
        };
      },
    },
    {
      id: 'master', name: 'MASTER (UCB1 base)', family: NS,
      params: [{ key: 'c', label: 'c (test scale)', default: 1, min: 0.1, max: 3, step: 0.05 }],
      formula: 'blocks of 2ⁿ steps; start UCB1 copies of length 2ᵐ w.p. √(2ᵐ⁻ⁿ); restart if a copy\'s mean reward > Uₜ + 9ρ(2ᵐ) or avg(f̃ − R) > 3ρ',
      regret: 'Õ(min{√(ΓT), Δ^⅓T^⅔ + √T}) without knowing Γ or Δ (Wei & Luo 2021)',
      idea: 'A black-box reduction: fresh UCB1 copies run at many time scales at once, the shortest active one acts, and two consistency tests on the copies\' optimistic estimates f̃ trigger a full restart. Simplified: ρ(L) = c·√(K ln T / L), without the paper\'s log factors. At T ≤ 10⁴ it trails plain UCB1, since most steps are played by short, fresh copies that are still exploring.',
      create(K, { rng, params, env, T }) {
        const lT = Math.log(Math.max(3, T)), rho = (L) => Math.min(1, params.c * Math.sqrt((K * lT) / L));
        const idx = new Float64Array(K);
        let order = 0, len = 0, off = 0, sched = [], si = 0, stack = [], U = Infinity, gapSum = 0, P = [0];
        let cur = null, curF = 1;
        function startBlock(n) {
          order = n; len = 1 << n; off = 0; si = 0; stack = []; U = Infinity; gapSum = 0; P = [0];
          sched = [{ o: 0, m: n }];
          for (let m = n - 1; m >= 0; m--) {
            const p = Math.sqrt(Math.pow(2, m - n));
            for (let o = 0; o < len; o += 1 << m) if (rng.uniform() < p) sched.push({ o, m });
          }
          sched.sort((x, y) => x.o - y.o || y.m - x.m);
        }
        startBlock(0);
        return {
          select() {
            if (off >= len) startBlock(order + 1);
            while (si < sched.length && sched[si].o === off) {
              const { o, m } = sched[si++];
              stack.push({ inst: ucbInstance(K), m, s: o, e: o + (1 << m) });
            }
            cur = stack[stack.length - 1].inst;
            const u = untried(cur.n);
            if (u >= 0) { curF = 1; return u; }
            const lt = Math.log(Math.max(1, cur.t));
            for (let a = 0; a < K; a++) idx[a] = cur.sum[a] / cur.n[a] + Math.sqrt((2 * lt) / cur.n[a]);
            const a = argmax(idx, rng);
            curF = Math.min(1, idx[a]);
            return a;
          },
          update(a, r) {
            const x = env.normalize(r);
            cur.n[a]++; cur.sum[a] += x; cur.t++;
            U = Math.min(U, curF); gapSum += curF - x; P.push(P[P.length - 1] + x);
            off++;
            let restart = gapSum / off >= 3 * rho(off);
            while (stack.length && stack[stack.length - 1].e === off) {
              const { m, s, e } = stack.pop();
              if ((P[e] - P[s]) / (e - s) >= U + 9 * rho(1 << m)) restart = true;
            }
            if (restart) startBlock(0);
          },
        };
      },
    },

    // ---- rested: rotting & rising
    {
      id: 'fewa', name: 'FEWA', family: RESTED,
      params: [{ key: 'alpha', label: 'α (δₜ = t^−α)', default: 0.1, min: 0.01, max: 5, step: 0.01 }],
      formula: 'for h = 1, 2, …: drop a if μ̂ₐʰ < maxᵦ μ̂ᵦʰ − 2σ√(2α ln t / h);  pull a surviving arm with Nₐ = h',
      regret: 'Õ(√(K T)) on rested rotting bandits, problem-free (Seznec et al. 2019)',
      idea: 'Filtering on Expanding Window Average: compare arms on their last h pulls only, widening h until a surviving arm has only h pulls, and pull it. Recent pulls reveal how far an arm has rotted; long windows only resolve small gaps. Windows above 16 follow a ×1.2 geometric grid (EFF-FEWA style) to stay fast; α defaults below the theory value of 4.',
      create(K, { rng, params, env, T }) {
        const sig = env.sigma, L = T + 1, pre = new Float64Array(K * L), N = new Int32Array(K);
        const act = new Int32Array(K), mh = new Float64Array(K), cand = new Int32Array(K);
        const nextH = (h) => (h < 16 ? h + 1 : Math.ceil(h * 1.2));
        return {
          select(t) {
            const u = untried(N); if (u >= 0) return u;
            const c0 = 2 * sig * Math.sqrt(2 * params.alpha * Math.log(Math.max(2, t)));
            let na = K;
            for (let a = 0; a < K; a++) act[a] = a;
            for (let h = 1; ; h = nextH(h)) {
              let mx = -Infinity;
              for (let i = 0; i < na; i++) {
                const a = act[i], o = a * L;
                mh[i] = (pre[o + N[a]] - pre[o + N[a] - h]) / h;
                if (mh[i] > mx) mx = mh[i];
              }
              const thr = mx - c0 / Math.sqrt(h), hn = nextH(h);
              let k = 0, nc = 0;
              for (let i = 0; i < na; i++) {
                if (mh[i] < thr) continue;
                const a = act[i];
                act[k++] = a;
                if (N[a] < hn) cand[nc++] = a; // cannot be compared on the next window
              }
              na = k;
              if (nc) return cand[nc > 1 ? rng.int(nc) : 0];
            }
          },
          update(a, r) { const o = a * L; pre[o + N[a] + 1] = pre[o + N[a]] + r; N[a]++; },
        };
      },
    },
    {
      id: 'raw-ucb', name: 'RAW-UCB', family: RESTED,
      params: [{ key: 'alpha', label: 'α (δₜ = t^−α)', default: 1, min: 0.1, max: 5, step: 0.1 }],
      formula: 'argmaxₐ  minₕ≤ₙₐ [ μ̂ₐʰ + σ√(2α ln t / h) ],  μ̂ₐʰ = mean of a\'s last h rewards',
      regret: 'Õ(√(K T)) on rotting bandits, rested or restless (Seznec et al. 2020)',
      idea: 'Rotting Adaptive Window UCB: each arm\'s index is the most pessimistic of its UCBs over all trailing windows, so a recent slump is never averaged away by a good past. Windows above 16 use a geometric grid (the paper\'s efficient variant).',
      create(K, { rng, params, env, T }) {
        const sig = env.sigma, L = T + 1, pre = new Float64Array(K * L), N = new Int32Array(K), idx = new Float64Array(K);
        return {
          select(t) {
            const u = untried(N); if (u >= 0) return u;
            const c0 = sig * Math.sqrt(2 * params.alpha * Math.log(Math.max(2, t)));
            for (let a = 0; a < K; a++) {
              const o = a * L, n = N[a], top = pre[o + n];
              let best = Infinity;
              for (let h = 1; ; h = h < 16 ? h + 1 : Math.ceil(h * 1.2)) {
                if (h > n) h = n;
                const v = (top - pre[o + n - h]) / h + c0 / Math.sqrt(h);
                if (v < best) best = v;
                if (h === n) break;
              }
              idx[a] = best;
            }
            return argmax(idx, rng);
          },
          update(a, r) { const o = a * L; pre[o + N[a] + 1] = pre[o + N[a]] + r; N[a]++; },
        };
      },
    },
    {
      id: 'red-ucb', name: 'R-ed-UCB', family: RESTED,
      params: [
        { key: 'eps', label: 'ε (window share)', default: 0.25, min: 0.05, max: 0.5, step: 0.01 },
        { key: 'alpha', label: 'α (δₜ = t^−α)', default: 0.1, min: 0.01, max: 5, step: 0.01 },
      ],
      formula: 'μ̂ₐ = 1/h Σₗ [Xₗ + (t − l)(Xₗ − Xₗ₋ₕ)/h] over a\'s last h = ⌊εNₐ⌋ pulls;  argmax μ̂ₐ + σ(t − Nₐ + h − 1)√(10α ln t / h³)',
      regret: 'Õ(T^{2/3}) for rested rising bandits with concave curves (Metelli et al. 2022)',
      idea: 'For rising (rested) arms: estimate each arm\'s recent slope from its last 2h pulls and extrapolate it to step t, plus a bonus for that extrapolation. Concavity makes the extrapolation optimistic, so a slow arm with a high ceiling is not given up on.',
      create(K, { rng, params, env, T }) {
        const sig = env.sigma, L = T + 1, eps = params.eps;
        const p1 = new Float64Array(K * L), p2 = new Float64Array(K * L), N = new Int32Array(K), idx = new Float64Array(K);
        const need = Math.max(2, Math.ceil(1 / eps));
        return {
          select(t) {
            let few = -1;
            for (let a = 0; a < K; a++) if (N[a] < need && (few < 0 || N[a] < N[few])) few = a;
            if (few >= 0) return few;
            const la = 10 * params.alpha * Math.log(Math.max(2, t));
            for (let a = 0; a < K; a++) {
              const o = a * L, n = N[a], h = Math.max(1, Math.floor(eps * n));
              const A = p1[o + n] - p1[o + n - h], B = p2[o + n] - p2[o + n - h];
              const A2 = p1[o + n - h] - p1[o + n - 2 * h], B2 = p2[o + n - h] - p2[o + n - 2 * h];
              // Σₗ (t − l)(Xₗ − Xₗ₋ₕ) over l = n−h+1..n, with Σ l·Xₗ₋ₕ = B2 + h·A2
              const slope = t * (A - A2) - (B - B2 - h * A2);
              idx[a] = A / h + slope / (h * h) + sig * (t - n + h - 1) * Math.sqrt(la / (h * h * h));
            }
            return argmax(idx, rng);
          },
          update(a, r) {
            const o = a * L, n = N[a] + 1;
            p1[o + n] = p1[o + n - 1] + r; p2[o + n] = p2[o + n - 1] + n * r; N[a] = n;
          },
        };
      },
    },
  ];

  MAB.register({
    families: [{
      id: RESTED, name: 'Rotting & rising', before: 'contextual',
      blurb: 'For rested arms whose means fall (rotting) or rise (learning curves) with their own pull count: compare arms on recent windows, or extrapolate the trend.',
    }],
    environments: ENVIRONMENTS,
    algorithms: ALGORITHMS,
  });
  return MAB;
});
