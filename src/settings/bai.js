/*
 * bai.js — Best-arm identification (pure exploration), both regimes in one pass.
 *
 * Every run draws one instance and pre-generates a reward stream per arm (the k-th pull of
 * arm a returns streams[a][k] for every algorithm: common random numbers). Each algorithm then
 * samples for exactly T pulls (T = budget = stopping cap) and we record two views:
 *
 *  - Fixed confidence: the first pull τ at which its stopping rule fires at level δ (its own
 *    rule when the paper defines one, otherwise the GLR rule with threshold
 *    β(t, δ) = ln((1 + ln t)/δ)), whether the answer given at τ was wrong, and whether it
 *    never stopped within the cap T (cap hit; τ is then counted as T).
 *  - Fixed budget: whether the recommendation after t pulls is wrong, on a log-spaced grid of
 *    t up to T. Anytime algorithms keep sampling past τ (their sampling rule does not need a
 *    stop). Fixed-budget algorithms (SR, SH, SAR) need to know the budget, so they are
 *    re-run from scratch at every grid budget on the same reward streams.
 *
 * Top-m instances (env.m > 1): the answer is a set of m arms and is wrong unless it equals the
 * true top-m set. Algorithms flagged `topm` handle m natively; the others keep their best-arm
 * sampling rule and are stopped by the top-m GLR rule and judged on their empirical top m.
 *
 * Linear instances: each arm has features xₐ ∈ ℝᵈ and mean xₐᵀθ. Linear algorithms use them;
 * on unstructured instances they treat arms as one-hot features (diagonal design).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;
  const { klBern, argmax, now } = util;
  const LN2 = Math.LN2;

  // ------------------------------------------------------------ helpers ----
  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

  /** Write the indices of the m largest vals (restricted to mask when given) into out; inTop marks them. */
  function topMInto(out, inTop, vals, m, mask) {
    const K = vals.length;
    inTop.fill(0);
    for (let k = 0; k < m; k++) {
      let bi = -1, bv = -Infinity;
      for (let a = 0; a < K; a++) {
        if (inTop[a] || (mask && !mask[a])) continue;
        if (bi < 0 || vals[a] > bv) { bv = vals[a]; bi = a; }
      }
      if (bi < 0) for (let a = 0; a < K; a++) if (!inTop[a] && (bi < 0 || vals[a] > bv)) { bv = vals[a]; bi = a; }
      out[k] = bi; inTop[bi] = 1;
    }
    return out;
  }

  /** GLR statistic for "i is better than j": 0 unless μ̂ᵢ > μ̂ⱼ. */
  function pairZ(S, kl, i, j) {
    const ni = S.n[i], nj = S.n[j];
    if (ni === 0 || nj === 0) return 0;
    const mi = S.mean[i], mj = S.mean[j];
    if (mi <= mj) return 0;
    const w = (ni * mi + nj * mj) / (ni + nj);
    return ni * kl(mi, w) + nj * kl(mj, w);
  }

  function empBest(S, K) {
    let b = -1, bv = -Infinity;
    for (let a = 0; a < K; a++) if (S.n[a] > 0 && (b < 0 || S.mean[a] > bv)) { bv = S.mean[a]; b = a; }
    return b < 0 ? 0 : b;
  }

  function firstUntried(S, K) {
    for (let a = 0; a < K; a++) if (S.n[a] === 0) return a;
    return -1;
  }

  function allPairs(arms) {
    const out = [];
    for (let i = 0; i < arms.length; i++) for (let j = i + 1; j < arms.length; j++) out.push(arms[i], arms[j]);
    return out; // flat [i0, j0, i1, j1, ...]
  }

  // --------------------------------------------------- small linear algebra ----
  function invertInto(A, d, out) {
    const M = Float64Array.from(A);
    out.fill(0);
    for (let i = 0; i < d; i++) out[i * d + i] = 1;
    for (let c = 0; c < d; c++) {
      let p = c, pv = Math.abs(M[c * d + c]);
      for (let r = c + 1; r < d; r++) if (Math.abs(M[r * d + c]) > pv) { pv = Math.abs(M[r * d + c]); p = r; }
      if (pv < 1e-300) { M[c * d + c] = 1e-12; p = c; }
      if (p !== c) for (let k = 0; k < d; k++) {
        let t = M[c * d + k]; M[c * d + k] = M[p * d + k]; M[p * d + k] = t;
        t = out[c * d + k]; out[c * d + k] = out[p * d + k]; out[p * d + k] = t;
      }
      const inv = 1 / M[c * d + c];
      for (let k = 0; k < d; k++) { M[c * d + k] *= inv; out[c * d + k] *= inv; }
      for (let r = 0; r < d; r++) {
        if (r === c) continue;
        const f = M[r * d + c];
        if (f === 0) continue;
        for (let k = 0; k < d; k++) { M[r * d + k] -= f * M[c * d + k]; out[r * d + k] -= f * out[c * d + k]; }
      }
    }
    return out;
  }

  /** Lower Cholesky factor of a symmetric positive-definite matrix (jittered if needed). */
  function cholesky(A, d) {
    const L = new Float64Array(d * d);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * d + j];
        for (let k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k];
        if (i === j) L[i * d + i] = Math.sqrt(Math.max(s, 1e-12));
        else L[i * d + j] = s / L[j * d + j];
      }
    }
    return L;
  }

  function quadForm(Ainv, d, y) {
    let s = 0;
    for (let i = 0; i < d; i++) {
      let r = 0;
      for (let j = 0; j < d; j++) r += Ainv[i * d + j] * y[j];
      s += y[i] * r;
    }
    return s;
  }

  /**
   * Least-squares model over arm features. On unstructured instances (lin.linear false) it is the
   * one-hot special case: θ̂ = empirical means, ‖eᵢ − eⱼ‖²_{A⁻¹} = 1/nᵢ + 1/nⱼ.
   */
  function makeLin(c) {
    const { K, S, lin } = c;
    if (!lin.linear) {
      return {
        diag: true,
        refresh() {},
        update() {},
        mean(a) { return S.n[a] ? S.mean[a] : 0; },
        ySq(i, j) { return 1 / Math.max(S.n[i], 1e-9) + 1 / Math.max(S.n[j], 1e-9); },
      };
    }
    const d = lin.d, X = lin.X, lam0 = 0.01;
    const Ainv = new Float64Array(d * d), b = new Float64Array(d), theta = new Float64Array(d);
    const means = new Float64Array(K), u = new Float64Array(d), y = new Float64Array(d);
    for (let i = 0; i < d; i++) Ainv[i * d + i] = 1 / lam0;
    let dirty = false;
    const model = {
      diag: false, d, X, Ainv,
      update(a, r) {
        let den = 1;
        for (let i = 0; i < d; i++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += Ainv[i * d + j] * X[a * d + j];
          u[i] = s; den += X[a * d + i] * s;
        }
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Ainv[i * d + j] -= u[i] * u[j] / den;
        for (let i = 0; i < d; i++) b[i] += r * X[a * d + i];
        dirty = true;
      },
      refresh() {
        if (!dirty) return;
        for (let i = 0; i < d; i++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += Ainv[i * d + j] * b[j];
          theta[i] = s;
        }
        for (let a = 0; a < K; a++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += X[a * d + j] * theta[j];
          means[a] = s;
        }
        dirty = false;
      },
      mean(a) { model.refresh(); return means[a]; },
      ySq(i, j) {
        for (let k = 0; k < d; k++) y[k] = X[i * d + k] - X[j * d + k];
        return quadForm(Ainv, d, y);
      },
    };
    return model;
  }

  /** Phase-local least squares (RAGE / Peace re-estimate θ from each phase's samples only). */
  function makePhaseAcc(c) {
    const { K, lin } = c;
    if (!lin.linear) {
      const n = new Float64Array(K), sum = new Float64Array(K);
      return {
        reset() { n.fill(0); sum.fill(0); },
        add(a, r) { n[a]++; sum[a] += r; },
        estimate(out) { for (let a = 0; a < K; a++) out[a] = n[a] ? sum[a] / n[a] : 0; },
      };
    }
    const d = lin.d, X = lin.X;
    const A = new Float64Array(d * d), b = new Float64Array(d), Ai = new Float64Array(d * d);
    return {
      reset() { A.fill(0); b.fill(0); },
      add(a, r) {
        for (let i = 0; i < d; i++) {
          b[i] += r * X[a * d + i];
          for (let j = 0; j < d; j++) A[i * d + j] += X[a * d + i] * X[a * d + j];
        }
      },
      estimate(out) {
        const M = Float64Array.from(A);
        for (let i = 0; i < d; i++) M[i * d + i] += 1e-6;
        invertInto(M, d, Ai);
        for (let a = 0; a < K; a++) {
          let s = 0;
          for (let i = 0; i < d; i++) {
            let th = 0;
            for (let j = 0; j < d; j++) th += Ai[i * d + j] * b[j];
            s += X[a * d + i] * th;
          }
          out[a] = s;
        }
      },
    };
  }

  /**
   * XY-optimal design: λ ∈ Δ_K minimising max over directions y = xᵢ − xⱼ of ‖y‖²_{A(λ)⁻¹},
   * A(λ) = Σ λₐ xₐxₐᵀ, by Frank–Wolfe. One-hot case: uniform over the arms involved.
   * Returns { lam, rho, Ainv } (rho = the attained max).
   */
  function designXY(c, pairs, iters = 120) {
    const { K, lin } = c;
    const lam = new Float64Array(K);
    if (!lin.linear) {
      const inv = new Uint8Array(K);
      for (const a of pairs) inv[a] = 1;
      let cnt = 0;
      for (let a = 0; a < K; a++) cnt += inv[a];
      for (let a = 0; a < K; a++) lam[a] = inv[a] / Math.max(1, cnt);
      return { lam, rho: 2 * cnt };
    }
    const d = lin.d, X = lin.X;
    const A = new Float64Array(d * d), Ai = new Float64Array(d * d), y = new Float64Array(d), u = new Float64Array(d);
    lam.fill(1 / K);
    let best = null, bestRho = Infinity;
    for (let it = 0; it < iters; it++) {
      A.fill(0);
      for (let a = 0; a < K; a++) {
        const l = lam[a];
        if (l === 0) continue;
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) A[i * d + j] += l * X[a * d + i] * X[a * d + j];
      }
      for (let i = 0; i < d; i++) A[i * d + i] += 1e-9;
      invertInto(A, d, Ai);
      let maxV = -1, pi = 0;
      for (let p = 0; p < pairs.length; p += 2) {
        const i = pairs[p], j = pairs[p + 1];
        for (let k = 0; k < d; k++) y[k] = X[i * d + k] - X[j * d + k];
        const v = quadForm(Ai, d, y);
        if (v > maxV) { maxV = v; pi = p; }
      }
      if (maxV < bestRho) { bestRho = maxV; best = Float64Array.from(lam); }
      const i0 = pairs[pi], j0 = pairs[pi + 1];
      for (let k = 0; k < d; k++) y[k] = X[i0 * d + k] - X[j0 * d + k];
      for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += Ai[i * d + j] * y[j]; u[i] = s; }
      let ba = 0, bv = -1;
      for (let a = 0; a < K; a++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += u[k] * X[a * d + k];
        if (s * s > bv) { bv = s * s; ba = a; }
      }
      const g = 2 / (it + 3);
      for (let a = 0; a < K; a++) lam[a] *= 1 - g;
      lam[ba] += g;
    }
    for (let a = 0; a < K; a++) if (best[a] < 1e-4) best[a] = 0;
    let s = 0;
    for (let a = 0; a < K; a++) s += best[a];
    for (let a = 0; a < K; a++) best[a] /= s;
    return { lam: best, rho: bestRho };
  }

  /** Monte-Carlo Gaussian width E max_{y} |yᵀ A(λ)^{-1/2} η| over the given directions. */
  function gaussWidth(c, lam, pairs, rng, M = 64) {
    const { K, lin } = c;
    const proj = new Float64Array(K);
    let tot = 0;
    if (!lin.linear) {
      for (let s = 0; s < M; s++) {
        for (let a = 0; a < K; a++) proj[a] = lam[a] > 0 ? rng.normal() / Math.sqrt(lam[a]) : 0;
        let mx = 0;
        for (let p = 0; p < pairs.length; p += 2) mx = Math.max(mx, Math.abs(proj[pairs[p]] - proj[pairs[p + 1]]));
        tot += mx;
      }
      return tot / M;
    }
    const d = lin.d, X = lin.X;
    const A = new Float64Array(d * d), Ai = new Float64Array(d * d), v = new Float64Array(d), eta = new Float64Array(d);
    for (let a = 0; a < K; a++) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) A[i * d + j] += lam[a] * X[a * d + i] * X[a * d + j];
    for (let i = 0; i < d; i++) A[i * d + i] += 1e-9;
    invertInto(A, d, Ai);
    const L = cholesky(Ai, d);
    for (let s = 0; s < M; s++) {
      for (let i = 0; i < d; i++) eta[i] = rng.normal();
      for (let i = 0; i < d; i++) { let t = 0; for (let j = 0; j <= i; j++) t += L[i * d + j] * eta[j]; v[i] = t; }
      for (let a = 0; a < K; a++) { let t = 0; for (let i = 0; i < d; i++) t += X[a * d + i] * v[i]; proj[a] = t; }
      let mx = 0;
      for (let p = 0; p < pairs.length; p += 2) mx = Math.max(mx, Math.abs(proj[pairs[p]] - proj[pairs[p + 1]]));
      tot += mx;
    }
    return tot / M;
  }

  /** Pull the support arm that is furthest behind its target share λ (tracking). */
  function trackPick(lam, cnt, step, K) {
    let b = -1, bv = Infinity;
    for (let a = 0; a < K; a++) {
      if (lam[a] <= 0) continue;
      const v = cnt[a] - step * lam[a];
      if (v < bv) { bv = v; b = a; }
    }
    return b < 0 ? 0 : b;
  }

  // ---------------------------------------- optimal weights (Track-and-Stop) ----
  /** Smallest x ≥ x0 with g(x) = kl(p1, m) + x·kl(pa, m) = y, m = (p1 + x pa)/(1 + x). Newton from the left. */
  function invG(p1, pa, y, x0) {
    let x = x0;
    for (let i = 0; i < 60; i++) {
      const m = (p1 + x * pa) / (1 + x);
      const kla = klBern(pa, m);
      const diff = y - klBern(p1, m) - x * kla;
      if (diff <= 1e-8 * y || kla <= 0) break;
      x += diff / kla;
    }
    return x;
  }

  /**
   * w*(μ) for best-arm identification (Garivier & Kaufmann 2016, Thm 5): solve F(y) = 1 by bisection,
   * Gaussian in closed form, Bernoulli with Newton for xₐ(y). Writes into w; uniform on ties.
   */
  function optimalWeights(mu, K, bern, sigma, w, xs, xlo) {
    let b = 0;
    for (let a = 1; a < K; a++) if (mu[a] > mu[b]) b = a;
    for (let a = 0; a < K; a++) if (a !== b && mu[a] >= mu[b]) { w.fill(1 / K); return false; }
    if (!bern) {
      let cmin = Infinity;
      for (let a = 0; a < K; a++) if (a !== b) { xs[a] = (mu[b] - mu[a]) ** 2 / (2 * sigma * sigma); cmin = Math.min(cmin, xs[a]); }
      let lo = 0, hi = cmin;
      for (let it = 0; it < 40; it++) {
        const y = (lo + hi) / 2;
        let F = 0;
        for (let a = 0; a < K; a++) if (a !== b) { const x = y / (xs[a] - y); F += x * x; }
        if (F > 1) hi = y; else lo = y;
      }
      const y = (lo + hi) / 2;
      let tot = 1;
      for (let a = 0; a < K; a++) if (a !== b) { w[a] = y / (xs[a] - y); tot += w[a]; }
      w[b] = 1;
      for (let a = 0; a < K; a++) w[a] /= tot;
      return true;
    }
    const e = 1e-6;
    const p1 = clamp(mu[b], e, 1 - e);
    let ymax = Infinity;
    for (let a = 0; a < K; a++) if (a !== b) ymax = Math.min(ymax, klBern(p1, clamp(mu[a], e, 1 - e)));
    let lo = 0, hi = ymax;
    xlo.fill(0);
    for (let it = 0; it < 17; it++) {
      const y = (lo + hi) / 2;
      let F = 0;
      for (let a = 0; a < K; a++) {
        if (a === b) continue;
        const pa = clamp(mu[a], e, 1 - e);
        const x = invG(p1, pa, y, xlo[a]);
        xs[a] = x;
        const m = (p1 + x * pa) / (1 + x);
        F += klBern(p1, m) / Math.max(klBern(pa, m), 1e-300);
      }
      if (F > 1) hi = y;
      else { lo = y; for (let a = 0; a < K; a++) xlo[a] = xs[a]; }
    }
    let tot = 1;
    for (let a = 0; a < K; a++) if (a !== b) { w[a] = Math.max(xlo[a], 1e-6); tot += w[a]; }
    w[b] = 1;
    for (let a = 0; a < K; a++) w[a] /= tot;
    return true;
  }

  // ------------------------------------------------------------ families ----
  const families = [
    { id: 'baseline', name: 'Baseline', blurb: 'Uniform round-robin sampling, stopped by the GLR rule. Every adaptive method should beat it.' },
    { id: 'elimination', name: 'Elimination', blurb: 'Sample the surviving arms evenly and discard arms that are confidently (or by median) worse.' },
    { id: 'ucb-based', name: 'Confidence bounds & gap index', blurb: 'Sample the arms whose confidence intervals still overlap the most; stop once they separate.' },
    { id: 'optimal-allocation', name: 'Optimal allocation', blurb: 'Plug the empirical means into the lower-bound allocation w*(μ) and track it. Asymptotically optimal.' },
    { id: 'top-two', name: 'Top two', blurb: 'Each round pick a leader and a challenger and pull the leader with probability β.' },
    { id: 'fixed-budget', name: 'Fixed budget', blurb: 'Split a known budget T into phases and reject (or accept) arms at phase ends. Re-run for every budget on the curve.' },
    { id: 'linear', name: 'Linear', blurb: 'Use arm features xₐ ∈ ℝᵈ with a shared θ (one-hot features on unstructured instances).' },
  ];

  // --------------------------------------------------------- environments ----
  const environments = [
    {
      id: 'bern-easy', name: 'Bernoulli, one clear best', short: 'Clear best', type: 'bernoulli',
      tags: ['bernoulli', 'easy'],
      description: 'The best arm pays 0.7; the others draw μ ~ U(0.2, 0.5), so every gap is at least 0.2.',
      draw(K, rng) {
        const mu = new Float64Array(K);
        mu[0] = 0.7;
        for (let a = 1; a < K; a++) mu[a] = 0.2 + 0.3 * rng.uniform();
        return { mu };
      },
    },
    {
      id: 'bern-close', name: 'Bernoulli, close gaps', short: 'Close gaps', type: 'bernoulli',
      tags: ['bernoulli', 'hard'],
      description: 'Best 0.5, runner-up 0.45, the rest μ ~ U(0.2, 0.42). The 0.05 gap needs thousands of pulls at small δ.',
      draw(K, rng) {
        const mu = new Float64Array(K);
        mu[0] = 0.5;
        if (K > 1) mu[1] = 0.45;
        for (let a = 2; a < K; a++) mu[a] = 0.2 + 0.22 * rng.uniform();
        return { mu };
      },
    },
    {
      id: 'gauss-arith', name: 'Gaussian, evenly spaced', short: 'Even gaps', type: 'gaussian', sigma: 1,
      tags: ['gaussian'],
      description: 'Means 1, 0.75, 0.5, … (gap 0.25 between neighbours), unit-variance Gaussian noise.',
      draw(K) {
        const mu = new Float64Array(K);
        for (let a = 0; a < K; a++) mu[a] = 1 - 0.25 * a;
        return { mu };
      },
    },
    {
      id: 'gauss-one', name: 'Gaussian, one good arm', short: 'One good arm', type: 'gaussian', sigma: 1,
      tags: ['gaussian'],
      description: 'One arm has mean 0.5, the other K − 1 have mean 0; unit-variance noise. The optimal allocation gives the best arm about √(K−1) times each rival\'s share.',
      draw(K) {
        const mu = new Float64Array(K);
        mu[0] = 0.5;
        return { mu };
      },
    },
    {
      id: 'linear', name: 'Linear (Soare et al.)', short: 'Linear', type: 'gaussian', sigma: 1, linear: true,
      tags: ['gaussian', 'linear'],
      description: 'Arms are vectors in ℝᵈ (d = min(4, K − 1), at least 2) with θ = e₁: x₁ = e₁, x₂ = (cos 0.6, sin 0.6, 0, …), then e₂…e_d and random unit vectors. The close pair x₁, x₂ is cheap to tell apart by pulling e₂, which only linear methods exploit. Noise N(0, 1).',
      d(K) { return Math.max(2, Math.min(4, K - 1)); },
      draw(K, rng) {
        const d = Math.max(2, Math.min(4, K - 1)), w = 0.6;
        const X = new Float64Array(K * d), mu = new Float64Array(K);
        const rows = [];
        const e = (i) => { const r = new Float64Array(d); r[i] = 1; return r; };
        rows.push(e(0));
        const x2 = new Float64Array(d); x2[0] = Math.cos(w); x2[1] = Math.sin(w); rows.push(x2);
        for (let i = 1; i < d; i++) rows.push(e(i));
        while (rows.length < K) {
          const v = new Float64Array(d);
          let nrm = 0;
          for (let i = 0; i < d; i++) { v[i] = rng.normal(); nrm += v[i] * v[i]; }
          nrm = Math.sqrt(nrm);
          for (let i = 0; i < d; i++) v[i] /= nrm;
          if (v[0] <= 0.5) rows.push(v);
        }
        for (let a = 0; a < K; a++) { X.set(rows[a], a * d); mu[a] = rows[a][0]; }
        return { mu, X, d };
      },
    },
    {
      id: 'topm', name: 'Bernoulli, top-3', short: 'Top-3', type: 'bernoulli',
      tags: ['bernoulli', 'top-m'],
      description: 'Identify the m = min(3, K − 1) best arms as a set. Top arms μ ~ U(0.6, 0.75), the rest μ ~ U(0.2, 0.5).',
      m(K) { return Math.max(1, Math.min(3, K - 1)); },
      draw(K, rng) {
        const m = Math.max(1, Math.min(3, K - 1));
        const mu = new Float64Array(K);
        for (let a = 0; a < K; a++) mu[a] = a < m ? 0.6 + 0.15 * rng.uniform() : 0.2 + 0.3 * rng.uniform();
        return { mu };
      },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of environments) ENV_BY_ID[e.id] = e;

  // ----------------------------------------------------------- algorithms ----
  /*
   * create(c) with c = { K, m, T (budget), delta, S, rng, params, bern, sigma, s, kl, beta, lin }
   *   S     shared per-run statistics { n, sum, mean } updated by the runner before update()
   *   s     sub-Gaussian scale (1/2 for Bernoulli, σ for Gaussian)
   *   kl    KL divergence of the reward family; beta(t) the GLR threshold ln((1 + ln t)/δ)
   *   lin   { linear, d, X } arm features (linear env only)
   * returns { select(t), update(a, r, t), recommend?() -> arms[m], stop?(t) -> bool }
   */

  // ---- baseline
  const uniform = {
    id: 'uniform', name: 'Uniform sampling', family: 'baseline', topm: true,
    params: [],
    formula: 'aₜ = t mod K; stop when min_{b≠â} Z_{â,b}(t) > β(t,δ)',
    regret: 'E[τ] = O(K Δ_min⁻² ln(1/δ))',
    idea: 'Round-robin over all arms with GLR stopping. The fair yardstick: every adaptive method should need fewer samples.',
    create({ K }) { return { select: (t) => (t - 1) % K, update() {} }; },
  };

  // ---- elimination
  const se = {
    id: 'se', name: 'Successive Elimination', family: 'elimination', topm: true,
    params: [],
    formula: 'drop a if #{b: LCB_b > UCB_a} ≥ m,  c_r = s√(2 ln(4Kr²/δ)/r)',
    regret: 'O(Σₐ Δₐ⁻² ln(K/(δΔₐ)))',
    idea: 'Even-Dar et al. (2006): sample all survivors once per round and eliminate an arm as soon as its upper bound falls below the leader\'s lower bound. Top-m: rejection only (simplified).',
    create({ K, m, S, s, delta }) {
      let active = Array.from({ length: K }, (_, a) => a);
      let queue = active.slice().reverse();
      const top = new Int32Array(m), inTop = new Uint8Array(K), mask = new Uint8Array(K), vals = new Float64Array(K);
      function eliminate() {
        const w = new Float64Array(K);
        for (const a of active) { const n = S.n[a]; w[a] = s * Math.sqrt(2 * Math.log(4 * K * n * n / delta) / n); }
        active = active.filter((a) => {
          let beat = 0;
          for (const b of active) if (b !== a && S.mean[b] - w[b] > S.mean[a] + w[a]) beat++;
          return beat < m;
        });
      }
      return {
        select() { return queue.pop(); },
        update() {
          if (queue.length) return;
          if (active.length > m) eliminate();
          queue = active.slice().reverse();
        },
        stop() { return active.length <= m; },
        recommend() {
          mask.fill(0);
          for (const a of active) { mask[a] = 1; }
          for (let a = 0; a < K; a++) vals[a] = S.mean[a];
          return topMInto(top, inTop, vals, m, mask);
        },
      };
    },
  };

  const me = {
    id: 'me', name: 'Median Elimination', family: 'elimination', topm: true,
    params: [{ key: 'eps', label: 'ε (PAC slack)', default: 0.3, min: 0.02, max: 1, step: 0.01 }],
    formula: 'round l: pull each survivor (2s)²·(4/ε_l²)·ln(3/δ_l) times, keep the top half; ε_{l+1} = ¾ε_l, δ_{l+1} = δ_l/2',
    regret: 'O((K/ε²) ln(1/δ)) samples; returns an ε-good arm w.p. 1 − δ',
    idea: 'Even-Dar et al. (2006): fixed-length rounds that halve the arm set by the median of fresh round means. Worst-case optimal but very sample-hungry; its target is only an ε-good arm, while errors here count against the exact best arm.',
    create({ K, m, S, s, delta, params }) {
      let active = Array.from({ length: K }, (_, a) => a);
      let el = params.eps / 4, dl = delta / 2, ptr = 0, inRound = 0;
      const rs = new Float64Array(K), rn = new Float64Array(K);
      const top = new Int32Array(m), inTop = new Uint8Array(K), mask = new Uint8Array(K), vals = new Float64Array(K);
      const quota = () => Math.max(1, Math.ceil(4 * s * s * (4 / (el * el)) * Math.log(3 / dl)));
      let q = quota();
      return {
        select() { const a = active[ptr]; ptr = (ptr + 1) % active.length; return a; },
        update(a, r) {
          if (active.length <= m) return;
          rs[a] += r; rn[a]++; inRound++;
          if (inRound < q * active.length) return;
          const sorted = active.slice().sort((x, y) => rs[y] / rn[y] - rs[x] / rn[x]);
          active = sorted.slice(0, Math.max(m, Math.ceil(active.length / 2)));
          el *= 0.75; dl /= 2; q = quota(); inRound = 0; ptr = 0;
          rs.fill(0); rn.fill(0);
        },
        stop() { return active.length <= m; },
        recommend() {
          mask.fill(0);
          for (const a of active) mask[a] = 1;
          for (let a = 0; a < K; a++) vals[a] = S.mean[a];
          return topMInto(top, inTop, vals, m, mask);
        },
      };
    },
  };

  // ---- LUCB family
  function lucbCreate(kind) {
    return function create({ K, m, S, s, delta }) {
      const top = new Int32Array(m), inTop = new Uint8Array(K);
      const queue = [];
      let h = 0, l = 0, gap = Infinity;
      // LUCB1: β(u, t) = s√(2 ln(5Kt⁴/(4δ))/u). LUCB++: anytime LIL-type width with δ/(2(K−m)) for the
      // top set and δ/(2m) for the rest.
      const lil = (u, d) => s * Math.sqrt(2 * (Math.log(1 / d) + 2 * Math.log(1 + Math.log(u)) + 1) / u);
      let lt = 0;
      const wTop = kind === 'lucb' ? (u) => s * Math.sqrt(2 * lt / u) : (u) => lil(u, delta / (2 * (K - m)));
      const wBot = kind === 'lucb' ? wTop : (u) => lil(u, delta / (2 * m));
      function compute(t) {
        lt = Math.log(5 * K / (4 * delta)) + 4 * Math.log(t);
        topMInto(top, inTop, S.mean, m);
        let hv = Infinity, lv = -Infinity;
        for (let a = 0; a < K; a++) {
          if (inTop[a]) { const v = S.mean[a] - wTop(S.n[a], t); if (v < hv) { hv = v; h = a; } }
          else { const v = S.mean[a] + wBot(S.n[a], t); if (v > lv) { lv = v; l = a; } }
        }
        gap = lv - hv;
      }
      return {
        select(t) {
          if (t <= K) return t - 1;
          if (!queue.length) { compute(t); queue.push(h, l); }
          return queue.shift();
        },
        update() {},
        stop(t) { if (t < K) return false; compute(t); return gap < 0; },
        recommend() { return topMInto(top, inTop, S.mean, m); },
      };
    };
  }

  const lucb = {
    id: 'lucb', name: 'LUCB1', family: 'ucb-based', topm: true,
    params: [],
    formula: 'J = top-m(μ̂); pull h = argmin_{J} LCB and l = argmax_{Jᶜ} UCB; stop when LCB_h > UCB_l',
    regret: 'O(H₁ ln(H₁/δ)),  H₁ = Σₐ Δₐ⁻²',
    idea: 'Kalyanakrishnan et al. (2012): each round pulls the weakest-looking member of the empirical top set and the strongest outsider, and stops once their confidence intervals separate.',
    create: lucbCreate('lucb'),
  };
  const lucbpp = {
    id: 'lucbpp', name: 'LUCB++', family: 'ucb-based', topm: true,
    params: [],
    formula: 'as LUCB, LCB with δ/(2(K−m)), UCB with δ/(2m), anytime LIL widths',
    regret: 'O(Σₐ Δₐ⁻² ln(ln(Δₐ⁻²)/δ)), optimal dependence on m and K',
    idea: 'Simchowitz, Jamieson & Recht (2017): LUCB with asymmetric confidence levels for the top and bottom sets and law-of-the-iterated-logarithm widths, removing LUCB\'s ln K and ln t overhead. Width constants simplified.',
    create: lucbCreate('lucbpp'),
  };

  const ugape = {
    id: 'ugape', name: 'UGapE', family: 'ucb-based', topm: true,
    params: [],
    formula: 'B_k = max⁽ᵐ⁾_{i≠k} U_i − L_k; J = m smallest B; pull the wider of u = argmax_{Jᶜ} U, l = argmin_J L',
    regret: 'O(H_ε ln(H_ε/δ))',
    idea: 'Gabillon, Ghavamzadeh & Lazaric (2012), fixed-confidence version with ε = 0: a gap index bounds how much each arm could lose to the others; stop when every chosen arm\'s index is negative.',
    create({ K, m, S, s, delta }) {
      const U = new Float64Array(K), Lo = new Float64Array(K), W = new Float64Array(K), B = new Float64Array(K), negB = new Float64Array(K);
      const mu1 = Math.min(m + 1, K);
      const topU = new Int32Array(mu1), inTopU = new Uint8Array(K), J = new Int32Array(m), inJ = new Uint8Array(K), inFirstM = new Uint8Array(K);
      let pick = 0, maxB = Infinity;
      function compute(t) {
        const lt = Math.log(4 * K * t * t * t / delta);
        for (let a = 0; a < K; a++) {
          W[a] = s * Math.sqrt(2 * lt / S.n[a]);
          U[a] = S.mean[a] + W[a]; Lo[a] = S.mean[a] - W[a];
        }
        topMInto(topU, inTopU, U, mu1);
        inFirstM.fill(0);
        for (let k = 0; k < m; k++) inFirstM[topU[k]] = 1;
        for (let k = 0; k < K; k++) {
          const ref = inFirstM[k] ? U[topU[mu1 - 1]] : U[topU[m - 1]];
          B[k] = ref - Lo[k];
          negB[k] = -B[k];
        }
        topMInto(J, inJ, negB, m);
        let u = -1, uv = -Infinity, l = -1, lv = Infinity;
        maxB = -Infinity;
        for (let a = 0; a < K; a++) {
          if (inJ[a]) { if (Lo[a] < lv) { lv = Lo[a]; l = a; } if (B[a] > maxB) maxB = B[a]; }
          else if (U[a] > uv) { uv = U[a]; u = a; }
        }
        pick = u < 0 ? l : W[u] >= W[l] ? u : l;
      }
      return {
        select(t) { if (t <= K) return t - 1; compute(t); return pick; },
        update() {},
        stop(t) { if (t < K) return false; compute(t); return maxB < 0; },
        recommend() { return J; },
      };
    },
  };

  const lilucb = {
    id: 'lilucb', name: "lil'UCB", family: 'ucb-based',
    params: [
      { key: 'beta', label: 'β (bonus)', default: 0.5, min: 0.01, max: 3, step: 0.01 },
      { key: 'eps', label: 'ε', default: 0.01, min: 0.001, max: 1, step: 0.001 },
      { key: 'lambda', label: 'λ (stopping ratio)', default: 2.5, min: 1, max: 50, step: 0.5 },
    ],
    formula: 'aₜ = argmax μ̂ₐ + (1+β)(1+√ε)·σ√(2(1+ε) ln(ln((1+ε)nₐ)/δ′)/nₐ); stop when nᵢ ≥ 1 + λ Σ_{j≠i} nⱼ',
    regret: 'O(Σₐ Δₐ⁻² ln(ln(Δₐ⁻²)/δ)) — optimal up to constants (LIL)',
    idea: 'Jamieson et al. (2014): a UCB rule with law-of-the-iterated-logarithm bonuses; it stops when one arm has been pulled λ times more than all others combined and recommends the most-pulled arm. Heuristic constants from the paper\'s experiments: δ′ = δ/5 and λ ≈ 1 + 10/K (theory asks for λ ≈ ((2+β)/β)²).',
    create({ K, S, s, delta, params, rng }) {
      const { beta, eps, lambda } = params;
      const dp = delta / 5, idx = new Float64Array(K);
      const c = (1 + beta) * (1 + Math.sqrt(eps)) * s * Math.sqrt(2 * (1 + eps));
      return {
        select(t) {
          if (t <= K) return t - 1;
          for (let a = 0; a < K; a++) {
            const n = S.n[a];
            idx[a] = S.mean[a] + c * Math.sqrt(Math.max(0, Math.log(Math.log(Math.E + (1 + eps) * n) / dp)) / n);
          }
          return argmax(idx, rng);
        },
        update() {},
        stop(t) {
          if (t <= K) return false;
          for (let a = 0; a < K; a++) if (S.n[a] >= 1 + lambda * (t - S.n[a])) return true;
          return false;
        },
        recommend() {
          let b = 0;
          for (let a = 1; a < K; a++) if (S.n[a] > S.n[b] || (S.n[a] === S.n[b] && S.mean[a] > S.mean[b])) b = a;
          return [b];
        },
      };
    },
  };

  const gifa = {
    id: 'gifa', name: 'GIFA (LUCB-type gap index)', family: 'ucb-based', topm: true,
    params: [],
    formula: 'B_{i,j} = μ̂ⱼ − μ̂ᵢ + s√(2(1/nᵢ+1/nⱼ)β(t,δ)); (c, b) = argmax_{i∈J, j∉J} B; pull the less-sampled; stop when max B ≤ 0',
    regret: 'δ-correct; O(H₁ ln(H₁/δ))-type, like LUCB / m-LinGapE',
    idea: 'One instance of the gap-index focused algorithms of Reda, Kaufmann & Delahaye-Duriez (2021): a paired (not per-arm) gap index between the empirical top m and the rest, LUCB-style pair selection and the "largest variance" rule. Unstructured instance.',
    create({ K, m, S, s, beta }) {
      const top = new Int32Array(m), inTop = new Uint8Array(K);
      let pick = 0, maxB = Infinity;
      function compute(t) {
        topMInto(top, inTop, S.mean, m);
        const bt = beta(t);
        maxB = -Infinity;
        let bi = 0, bj = 0;
        for (let k = 0; k < m; k++) {
          const i = top[k];
          for (let j = 0; j < K; j++) {
            if (inTop[j]) continue;
            const v = S.mean[j] - S.mean[i] + s * Math.sqrt(2 * (1 / S.n[i] + 1 / S.n[j]) * bt);
            if (v > maxB) { maxB = v; bi = i; bj = j; }
          }
        }
        pick = S.n[bi] <= S.n[bj] ? bi : bj;
      }
      return {
        select(t) { if (t <= K) return t - 1; compute(t); return pick; },
        update() {},
        stop(t) { if (t < K) return false; compute(t); return maxB <= 0; },
        recommend() { return topMInto(top, inTop, S.mean, m); },
      };
    },
  };

  // ---- optimal allocation
  function tasCreate(tracking) {
    return function create({ K, S, bern, sigma }) {
      const w = new Float64Array(K).fill(1 / K), xs = new Float64Array(K), xlo = new Float64Array(K), cum = new Float64Array(K);
      let next = 0;
      return {
        select(t) {
          if (t <= K) { for (let a = 0; a < K; a++) cum[a] += 1 / K; return t - 1; }
          if (t >= next) { optimalWeights(S.mean, K, bern, sigma, w, xs, xlo); next = t + Math.max(1, Math.floor(t / 10)); }
          if (tracking === 'D') {
            const thr = Math.sqrt(t) - K / 2;
            let u = -1;
            for (let a = 0; a < K; a++) if (S.n[a] < thr && (u < 0 || S.n[a] < S.n[u])) u = a;
            if (u >= 0) return u;
            let b = 0, bv = -Infinity;
            for (let a = 0; a < K; a++) { const v = t * w[a] - S.n[a]; if (v > bv) { bv = v; b = a; } }
            return b;
          }
          const e = 0.5 / Math.sqrt(K * K + t);
          let b = 0, bv = -Infinity;
          for (let a = 0; a < K; a++) {
            cum[a] += (1 - K * e) * w[a] + e;
            const v = cum[a] - S.n[a];
            if (v > bv) { bv = v; b = a; }
          }
          return b;
        },
        update() {},
      };
    };
  }

  const tasC = {
    id: 'tas-c', name: 'Track-and-Stop (C-tracking)', family: 'optimal-allocation',
    params: [],
    formula: 'aₜ = argmaxₐ Σ_{s≤t} w^{ε_s}ₐ(μ̂(s)) − nₐ,  ε_s = ½(K²+s)^{-½}; GLR stop',
    regret: 'lim_{δ→0} E[τ]/ln(1/δ) = T*(μ) (asymptotically optimal)',
    idea: 'Garivier & Kaufmann (2016): compute the optimal proportions w*(μ̂) by bisection on F(y) = 1 and track their running sum, mixed with a vanishing uniform share. Weights refreshed every ~10% of pulls (lazy).',
    create: tasCreate('C'),
  };
  const tasD = {
    id: 'tas-d', name: 'Track-and-Stop (D-tracking)', family: 'optimal-allocation',
    params: [],
    formula: 'aₜ = argmin nₐ if minₐ nₐ < √t − K/2, else argmaxₐ t·w*ₐ(μ̂) − nₐ; GLR stop',
    regret: 'lim_{δ→0} E[τ]/ln(1/δ) = T*(μ) (asymptotically optimal)',
    idea: 'Garivier & Kaufmann (2016): forced exploration keeps every arm above √t pulls; otherwise pull the arm furthest below its optimal share. Weights refreshed every ~10% of pulls (lazy).',
    create: tasCreate('D'),
  };

  const fws = {
    id: 'fws', name: 'Frank–Wolfe Sampling', family: 'optimal-allocation',
    params: [],
    formula: 'z = argmax_{z∈Δ} min_{b∈B_r} ⟨∇f_b(w), z − w⟩,  w = n/t, f_b = transport cost; track Σz; GLR stop',
    regret: 'lim_{δ→0} E[τ]/ln(1/δ) = T*(μ) (asymptotically optimal)',
    idea: 'Wang, Tzeng & Proutiere (2021): a Frank–Wolfe step on the non-smooth lower-bound objective using its r-subdifferential (all nearly-active rival arms). Simplified: the r-subdifferential is a relative band r_t = f_min·t^{-1/2}, the small max–min game is solved by water-filling plus golden-section search, and the Frank–Wolfe directions are tracked instead of sampled.',
    create({ K, S, kl }) {
      const f = new Float64Array(K), k1 = new Float64Array(K), k2 = new Float64Array(K), z = new Float64Array(K), cum = new Float64Array(K);
      const B = [], cB = new Float64Array(K), ord = new Int32Array(K);
      // Game value for z_h = p: the largest v with Σ_{b∈B} max(0, (v − c_b)/k2_b) = 1 − p, c_b = p·k1_b − f_b
      // (water-filling over the sorted breakpoints c_b).
      function level(p) {
        const nb = B.length;
        for (let i = 0; i < nb; i++) {
          const b = B[i], c = p * k1[b] - f[b];
          cB[b] = c;
          let j = i;
          while (j > 0 && cB[ord[j - 1]] > c) { ord[j] = ord[j - 1]; j--; }
          ord[j] = b;
        }
        let num = 1 - p, den = 0, v = 0;
        for (let i = 0; i < nb; i++) {
          const b = ord[i];
          num += cB[b] / k2[b]; den += 1 / k2[b];
          v = num / den;
          if (i + 1 === nb || v <= cB[ord[i + 1]]) break;
        }
        return v;
      }
      const G = (Math.sqrt(5) - 1) / 2;
      return {
        select(t) {
          if (t <= K) return t - 1;
          const st = Math.sqrt(t);
          let u = -1;
          for (let a = 0; a < K; a++) if (S.n[a] < st && (u < 0 || S.n[a] < S.n[u])) u = a;
          if (u >= 0) return u;
          const h = empBest(S, K);
          let fmin = Infinity;
          for (let b = 0; b < K; b++) {
            if (b === h) continue;
            const wh = S.n[h] / t, wb = S.n[b] / t;
            const mm = (wh * S.mean[h] + wb * S.mean[b]) / (wh + wb);
            k1[b] = kl(S.mean[h], mm); k2[b] = kl(S.mean[b], mm);
            f[b] = wh * k1[b] + wb * k2[b];
            if (f[b] < fmin) fmin = f[b];
          }
          // r-subdifferential: rival arms whose cost is within a vanishing relative band of the minimum
          const r = fmin * Math.pow(t, -0.5);
          B.length = 0;
          for (let b = 0; b < K; b++) if (b !== h && f[b] <= fmin + r && k2[b] > 0) B.push(b);
          if (!B.length) {
            let b2 = h === 0 ? 1 : 0;
            for (let b = 0; b < K; b++) if (b !== h && f[b] < f[b2]) b2 = b;
            return S.n[h] <= S.n[b2] ? h : b2;
          }
          let p;
          if (B.length === 1) p = k1[B[0]] >= k2[B[0]] ? 1 : 0;
          else {
            // V(p) is concave: golden-section search
            let lo = 0, hi = 1, x1 = hi - G * (hi - lo), x2 = lo + G * (hi - lo), v1 = level(x1), v2 = level(x2);
            for (let it = 0; it < 14; it++) {
              if (v1 < v2) { lo = x1; x1 = x2; v1 = v2; x2 = lo + G * (hi - lo); v2 = level(x2); }
              else { hi = x2; x2 = x1; v2 = v1; x1 = hi - G * (hi - lo); v1 = level(x1); }
            }
            p = (lo + hi) / 2;
          }
          z.fill(0);
          z[h] = p;
          let tot = p;
          if (B.length === 1) { z[B[0]] = 1 - p; tot = 1; }
          else {
            const v = level(p);
            for (const b of B) { z[b] = Math.max(0, (v + f[b] - p * k1[b]) / k2[b]); tot += z[b]; }
          }
          let best = h, bv = -Infinity;
          for (let a = 0; a < K; a++) {
            cum[a] += tot > 0 ? z[a] / tot : (a === h ? 1 : 0);
            const val = cum[a] - S.n[a];
            if (z[a] > 0 && val > bv) { bv = val; best = a; }
          }
          return best;
        },
        update() {},
      };
    },
  };

  // ---- top two
  function topTwoCreate(kind) {
    return function create({ K, S, bern, sigma, s, kl, rng, params }) {
      const beta = params.beta;
      const theta = new Float64Array(K), W = new Float64Array(K), ucb = new Float64Array(K);
      const pm = new Float64Array(K), ps = new Float64Array(K);
      const leadCnt = new Float64Array(K), leadPull = new Float64Array(K);
      function posterior() {
        for (let a = 0; a < K; a++) {
          const n = S.n[a];
          if (bern) { const m = (S.sum[a] + 1) / (n + 2); pm[a] = m; ps[a] = Math.sqrt(m * (1 - m) / (n + 3)); }
          else { pm[a] = n ? S.mean[a] : 0; ps[a] = n ? sigma / Math.sqrt(n) : 10 * sigma; }
        }
      }
      function sample() {
        let b = 0, bv = -Infinity;
        for (let a = 0; a < K; a++) { const v = pm[a] + ps[a] * rng.normal(); if (v > bv) { bv = v; b = a; } }
        return b;
      }
      function logPhi(x) {
        if (x > -5) return Math.log(Math.max(util.normCdf(x), 1e-300));
        return -0.5 * x * x - Math.log(-x) - 0.9189385332046727;
      }
      /**
       * TTTS challenger once rejection keeps failing (concentrated posterior): draw j ≠ leader with
       * probability ∝ P(θⱼ > θ_leader) under the Gaussian posteriors (pairwise approximation).
       */
      function pairChallenger(l) {
        let mx = -Infinity;
        for (let a = 0; a < K; a++) {
          if (a === l) { W[a] = -Infinity; continue; }
          W[a] = logPhi((pm[a] - pm[l]) / Math.sqrt(ps[a] * ps[a] + ps[l] * ps[l]));
          if (W[a] > mx) mx = W[a];
        }
        let tot = 0;
        for (let a = 0; a < K; a++) { W[a] = a === l ? 0 : Math.exp(W[a] - mx); tot += W[a]; }
        let u = rng.uniform() * tot;
        for (let a = 0; a < K; a++) { if (a === l) continue; u -= W[a]; if (u <= 0) return a; }
        return l === 0 ? 1 : 0;
      }
      function tc(l) {
        // transportation-cost challenger: argmin_{b≠l} Z_{l,b} (0 when μ̂_b ≥ μ̂_l), random ties
        for (let b = 0; b < K; b++) W[b] = b === l ? -Infinity : -pairZ(S, kl, l, b);
        return argmax(W, rng);
      }
      return {
        select(t) {
          if (t <= K) return t - 1;
          if (kind === 'ttucb') {
            const lt = Math.log(t);
            for (let a = 0; a < K; a++) ucb[a] = S.mean[a] + s * Math.sqrt(2 * lt / S.n[a]);
            const l = argmax(ucb, rng);
            leadCnt[l]++;
            if (leadPull[l] <= beta * leadCnt[l]) { leadPull[l]++; return l; }
            return tc(l);
          }
          posterior();
          const l = sample();
          if (rng.uniform() < beta) return l;
          if (kind === 't3c') return tc(l);
          for (let tries = 0; tries < 6; tries++) {
            const c = sample();
            if (c !== l) return c;
          }
          return pairChallenger(l);
        },
        update() {},
      };
    };
  }
  const betaParam = [{ key: 'beta', label: 'β (leader share)', default: 0.5, min: 0.05, max: 0.95, step: 0.05 }];
  const ttts = {
    id: 'ttts', name: 'TTTS', family: 'top-two',
    params: betaParam,
    formula: 'leader = argmax θ, θ ~ Π; w.p. 1−β resample until argmax θ′ ≠ leader; GLR stop',
    regret: 'E[τ]/ln(1/δ) → T*_β(μ) (β-optimal) with GLR stopping',
    idea: 'Russo (2016): Thompson sampling picks a leader; with probability 1 − β the posterior is resampled until another arm wins, and that challenger is pulled. Simplified for speed: Beta posteriors approximated by moment-matched Gaussians, and after 6 failed resamples (a concentrated posterior) the challenger is drawn with probability ∝ P(θⱼ > θ_leader), a pairwise approximation of the resampling outcome.',
    create: topTwoCreate('ttts'),
  };
  const t3c = {
    id: 't3c', name: 'T3C', family: 'top-two',
    params: betaParam,
    formula: 'leader = argmax θ, θ ~ Π; challenger = argmin_{b≠leader} W(leader, b); GLR stop',
    regret: 'E[τ]/ln(1/δ) → T*_β(μ) (β-optimal), cheaper than TTTS',
    idea: 'Shang et al. (2020): replaces TTTS\'s costly resampling with the arm that is cheapest to confuse with the leader, measured by the GLR transportation cost.',
    create: topTwoCreate('t3c'),
  };
  const ttucb = {
    id: 'ttucb', name: 'TTUCB', family: 'top-two',
    params: betaParam,
    formula: 'leader = argmax μ̂ₐ + s√(2 ln t/nₐ); challenger = argmin W(leader, b); tracking per leader; GLR stop',
    regret: 'asymptotically β-optimal with a non-asymptotic E[τ] bound',
    idea: 'Jourdan & Degenne (2022): a deterministic top-two method with a UCB leader and transportation-cost challenger; the leader/challenger split is tracked per leader rather than randomised.',
    create: topTwoCreate('ttucb'),
  };

  // ---- fixed budget (re-run for every budget)
  function logBar(K) { let s = 0.5; for (let i = 2; i <= K; i++) s += 1 / i; return s; }
  function srSchedule(K, B) {
    const n = new Float64Array(K);
    const lb = logBar(K);
    for (let k = 1; k < K; k++) n[k] = Math.ceil((B - K) / (lb * (K + 1 - k)));
    return n;
  }

  const sr = {
    id: 'sr', name: 'Successive Rejects', family: 'fixed-budget', budgeted: true, topm: true,
    params: [],
    formula: 'phase k: pull survivors to n_k = ⌈(T−K)/(loḡK·(K+1−k))⌉, reject argmin μ̂',
    regret: 'P(err) ≤ K(K−1)/2 · exp(−(T−K)/(loḡK · H₂))',
    idea: 'Audibert & Bubeck (2010): K − 1 phases of growing length; the empirically worst survivor is rejected at the end of each phase. Parameter-free. Top-m: rejects until m remain.',
    create({ K, m, T, S }) {
      const nk = srSchedule(K, T);
      let active = Array.from({ length: K }, (_, a) => a), k = 1, ptr = 0;
      const top = new Int32Array(m), inTop = new Uint8Array(K), mask = new Uint8Array(K);
      return {
        select() {
          for (;;) {
            if (active.length <= m || k >= K) { ptr = (ptr + 1) % active.length; return active[ptr]; }
            let u = -1;
            for (const a of active) if (S.n[a] < nk[k] && (u < 0 || S.n[a] < S.n[u])) u = a;
            if (u >= 0) return u;
            let w = active[0];
            for (const a of active) if (S.mean[a] < S.mean[w]) w = a;
            active = active.filter((a) => a !== w);
            k++;
          }
        },
        update() {},
        recommend() {
          mask.fill(0);
          for (const a of active) mask[a] = 1;
          return topMInto(top, inTop, S.mean, m, mask);
        },
      };
    },
  };

  const sh = {
    id: 'sh', name: 'Sequential Halving', family: 'fixed-budget', budgeted: true, topm: true,
    params: [],
    formula: 'round r of ⌈log₂ K⌉: pull each survivor ⌊T/(|S_r|⌈log₂K⌉)⌋ times, keep the top half by round means',
    regret: 'P(err) ≤ 3 log₂K · exp(−T/(8 H₂ log₂K))',
    idea: 'Karnin, Koren & Somekh (2013): split the budget evenly over ⌈log₂ K⌉ rounds and halve the arm set after each, judging arms only by that round\'s fresh samples. Top-m: never keeps fewer than m.',
    create({ K, m, T, S }) {
      const R = Math.max(1, Math.ceil(Math.log2(K / m)));
      let active = Array.from({ length: K }, (_, a) => a), ptr = 0, done = active.length <= m;
      const rs = new Float64Array(K), rn = new Float64Array(K);
      let per = Math.max(1, Math.floor(T / (active.length * R))), inRound = 0;
      const top = new Int32Array(m), inTop = new Uint8Array(K), mask = new Uint8Array(K);
      return {
        select() { const a = active[ptr]; ptr = (ptr + 1) % active.length; return a; },
        update(a, r) {
          if (done) return;
          rs[a] += r; rn[a]++; inRound++;
          if (inRound < per * active.length) return;
          const sorted = active.slice().sort((x, y) => rs[y] / rn[y] - rs[x] / rn[x]);
          active = sorted.slice(0, Math.max(m, Math.ceil(active.length / 2)));
          rs.fill(0); rn.fill(0); inRound = 0; ptr = 0;
          if (active.length <= m) done = true;
          per = Math.max(1, Math.floor(T / (active.length * R)));
        },
        recommend() {
          mask.fill(0);
          for (const a of active) mask[a] = 1;
          return topMInto(top, inTop, S.mean, m, mask);
        },
      };
    },
  };

  const sar = {
    id: 'sar', name: 'SAR (Successive Accepts and Rejects)', family: 'fixed-budget', budgeted: true, topm: true,
    params: [],
    formula: 'SR phases; at phase end accept the top arm if μ̂₍₁₎ − μ̂₍ₘ′₊₁₎ > μ̂₍ₘ′₎ − μ̂₍|A|₎, else reject the bottom one',
    regret: 'P(err) ≤ 2K² · exp(−(T−K)/(8 loḡK · H⟨m⟩))',
    idea: 'Bubeck, Wang & Viswanathan (2013): for top-m, each phase either accepts the arm most clearly in the top set or rejects the one most clearly out, whichever empirical gap is larger. Equals SR when m = 1.',
    create({ K, m, T, S }) {
      const nk = srSchedule(K, T);
      let active = Array.from({ length: K }, (_, a) => a), k = 1, ptr = 0;
      const accepted = [];
      const out = new Int32Array(m), inTop = new Uint8Array(K), mask = new Uint8Array(K), tmp = new Int32Array(m);
      const done = () => accepted.length >= m || accepted.length + active.length <= m || k >= K;
      return {
        select() {
          for (;;) {
            if (done()) { if (!active.length) return accepted[0]; ptr = (ptr + 1) % active.length; return active[ptr]; }
            let u = -1;
            for (const a of active) if (S.n[a] < nk[k] && (u < 0 || S.n[a] < S.n[u])) u = a;
            if (u >= 0) return u;
            // ties: the bottom of the order is the lowest-index arm among the worst (as in SR)
            const srt = active.slice().sort((x, y) => S.mean[y] - S.mean[x] || y - x);
            const mp = m - accepted.length, L = srt.length;
            const gTop = S.mean[srt[0]] - S.mean[srt[mp]];
            const gBot = S.mean[srt[mp - 1]] - S.mean[srt[L - 1]];
            if (gTop > gBot) { accepted.push(srt[0]); active = active.filter((a) => a !== srt[0]); }
            else active = active.filter((a) => a !== srt[L - 1]);
            k++;
          }
        },
        update() {},
        recommend() {
          const rest = m - accepted.length;
          for (let i = 0; i < accepted.length && i < m; i++) out[i] = accepted[i];
          if (rest > 0) {
            mask.fill(0);
            for (const a of active) mask[a] = 1;
            for (const a of accepted) mask[a] = 0;
            const t2 = rest === m ? tmp : new Int32Array(rest);
            topMInto(t2, inTop, S.mean, rest, mask);
            for (let i = 0; i < rest; i++) out[accepted.length + i] = t2[i];
          }
          return out;
        },
      };
    },
  };

  // ---- linear
  /** Linear GLR stop on structured instances: min_b ((x_â − x_b)ᵀθ̂)² / (2σ²‖x_â − x_b‖²_{A⁻¹}) > β. */
  function linGLR(model, K, t, beta, s) {
    let i = 0, iv = -Infinity;
    for (let a = 0; a < K; a++) { const v = model.mean(a); if (v > iv) { iv = v; i = a; } }
    const thr = beta(t);
    for (let b = 0; b < K; b++) {
      if (b === i) continue;
      const g = iv - model.mean(b);
      if (g <= 0 || g * g / (2 * s * s * model.ySq(i, b)) <= thr) return false;
    }
    return true;
  }
  function linBest(model, K, mask) {
    let i = -1, iv = -Infinity;
    for (let a = 0; a < K; a++) { if (mask && !mask[a]) continue; const v = model.mean(a); if (i < 0 || v > iv) { iv = v; i = a; } }
    return [i];
  }
  /** Soare et al.-style confidence: s‖y‖_{A⁻¹}√(2 ln(6K²t²/(π²δ))). */
  const linConf = (K, t, delta) => Math.sqrt(2 * Math.log(6 * K * K * t * t / (Math.PI * Math.PI * delta)));

  const xyStatic = {
    id: 'xy-static', name: 'XY-Static', family: 'linear',
    params: [],
    formula: 'λ = argmin_λ max_{y∈X−X} ‖y‖²_{A(λ)⁻¹} (Frank–Wolfe), track λ; linear GLR stop',
    regret: 'O(d Δ_min⁻² ln(1/δ)) up to log factors; static, not instance-adaptive',
    idea: 'Soare, Lazaric & Munos (2014): a fixed design that shrinks the uncertainty in every direction xᵢ − xⱼ equally, computed once and tracked. On unstructured instances it is uniform sampling.',
    create(c) {
      const { K, S, s, lin, beta } = c;
      const model = makeLin(c);
      const { lam } = designXY(c, allPairs(Array.from({ length: K }, (_, a) => a)));
      const agent = {
        select(t) { return trackPick(lam, S.n, t, K); },
        update(a, r) { model.update(a, r); },
      };
      if (lin.linear) {
        agent.stop = (t) => linGLR(model, K, t, beta, s);
        agent.recommend = () => linBest(model, K);
      }
      return agent;
    },
  };

  const xyAdaptive = {
    id: 'xy-adaptive', name: 'XY-Adaptive', family: 'linear',
    params: [],
    formula: 'phase j: XY-design on Ŷ_j = {x − x′ : x, x′ ∈ X̂_j}; drop x if (x′ − x)ᵀθ̂ > s‖x′ − x‖_{A⁻¹}c_t',
    regret: 'O(M* ln(1/δ)) up to log factors, M* ≈ the oracle allocation\'s complexity',
    idea: 'Soare, Lazaric & Munos (2014): re-plan the XY design on the directions between the arms still in play. Simplified: a new phase starts whenever an arm is eliminated (instead of the α-ratio rule) and all samples are kept.',
    create(c) {
      const { K, S, s, delta } = c;
      const model = makeLin(c);
      let active = Array.from({ length: K }, (_, a) => a), lam, done = false, nextCheck = K;
      const cnt = new Float64Array(K), mask = new Uint8Array(K);
      let phaseT = 0;
      function plan() { lam = designXY(c, allPairs(active)).lam; cnt.fill(0); phaseT = 0; }
      plan();
      function check(t) {
        const ct = s * linConf(K, t, delta);
        const keep = active.filter((z) => {
          for (const z2 of active) if (z2 !== z && model.mean(z2) - model.mean(z) > ct * Math.sqrt(model.ySq(z2, z))) return false;
          return true;
        });
        if (keep.length && keep.length < active.length) {
          active = keep;
          if (active.length === 1) done = true; else plan();
        }
      }
      return {
        select() {
          if (done) return active[0];
          phaseT++;
          return trackPick(lam, cnt, phaseT, K);
        },
        update(a, r, t) {
          model.update(a, r); cnt[a]++;
          if (!done && t >= nextCheck) { nextCheck = t + Math.max(1, Math.floor(t / 50)); check(t); }
        },
        stop() { return done; },
        recommend() {
          mask.fill(0);
          for (const a of active) mask[a] = 1;
          return linBest(model, K, mask);
        },
      };
    },
  };

  function eliminationPhasesCreate(kind) {
    return function create(c) {
      const { K, s, delta, rng } = c;
      const model = makeLin(c), acc = makePhaseAcc(c);
      let active = Array.from({ length: K }, (_, a) => a), k = 0, eps = 1, lam, nk = 0, phaseT = 0, done = false;
      const cnt = new Float64Array(K), est = new Float64Array(K), mask = new Uint8Array(K);
      function start() {
        k++; eps = Math.pow(2, -k);
        const pairs = allPairs(active);
        const des = designXY(c, pairs);
        lam = des.lam;
        let supp = 0;
        for (let a = 0; a < K; a++) if (lam[a] > 0) supp++;
        const dk = delta / (2 * k * k);
        if (kind === 'rage') {
          nk = Math.ceil(8 * s * s * des.rho * Math.log(2 * (pairs.length / 2) / dk) / (eps * eps));
        } else {
          const g = gaussWidth(c, lam, pairs, rng);
          nk = Math.ceil(4 * s * s * (g + Math.sqrt(2 * des.rho * Math.log(2 / dk))) ** 2 / (eps * eps));
        }
        nk = Math.max(nk, supp);
        cnt.fill(0); acc.reset(); phaseT = 0;
      }
      start();
      function end() {
        acc.estimate(est);
        let mx = -Infinity;
        for (const a of active) mx = Math.max(mx, est[a]);
        active = active.filter((a) => mx - est[a] <= eps / 2);
        if (active.length <= 1) done = true; else start();
      }
      return {
        select() {
          if (done) return active[0];
          return trackPick(lam, cnt, phaseT + 1, K);
        },
        update(a, r) {
          model.update(a, r);
          if (done) return;
          cnt[a]++; acc.add(a, r); phaseT++;
          if (phaseT >= nk) end();
        },
        stop() { return done; },
        recommend() {
          mask.fill(0);
          for (const a of active) mask[a] = 1;
          return linBest(model, K, mask);
        },
      };
    };
  }

  const rage = {
    id: 'rage', name: 'RAGE', family: 'linear',
    params: [],
    formula: 'phase k: λ_k = argmin max_{y∈Y(Ẑ_k)} ‖y‖²_{A(λ)⁻¹}, n_k = ⌈8σ²ρ_k ln(2|Y_k|/δ_k)/ε_k²⌉, drop z if max_{z′}(z′−z)ᵀθ̂_k > ε_k/2',
    regret: 'O(ρ* ln(1/Δ_min)(ln(1/δ) + ln|Z| + ln ln Δ_min⁻¹))',
    idea: 'Fiez, Jain, Jamieson & Ratliff (2019): phased elimination with ε_k = 2⁻ᵏ; each phase samples an XY-optimal design for the surviving arms\' differences, estimates θ from that phase only, and drops arms worse by ε_k/2.',
    create: eliminationPhasesCreate('rage'),
  };
  const peace = {
    id: 'peace', name: 'Peace (simplified)', family: 'linear',
    params: [],
    formula: 'as RAGE, n_k = ⌈4σ²(γ(Y_k) + √(2ρ_k ln(2/δ_k)))²/ε_k²⌉, γ = E max_y |yᵀA(λ)^{-½}η|',
    regret: 'O((ρ* ln(1/δ) + γ*) ln(1/Δ_min)), Gaussian-width instance-optimal up to logs',
    idea: 'Katz-Samuels, Jain, Karnin & Jamieson (2020) replace the union bound over directions by the Gaussian width of the surviving set. Simplified: keeps RAGE\'s XY design and uses the Monte-Carlo Gaussian width only for each phase\'s length, so with few arms it is looser than RAGE; it pays off when many arms share directions.',
    create: eliminationPhasesCreate('peace'),
  };

  const lingape = {
    id: 'lingape', name: 'LinGapE', family: 'linear',
    params: [],
    formula: 'î = argmax x̂ᵀθ̂, ĵ = argmax (xⱼ − x_î)ᵀθ̂ + C_t‖xⱼ − x_î‖_{A⁻¹}; pull argminₐ ‖x_î − x_ĵ‖²_{(A+xₐxₐᵀ)⁻¹}; stop when B ≤ 0',
    regret: 'O(H_lin ln(H_lin/δ)), H_lin ≤ d·Σ max(Δᵢ, ε)⁻²-type',
    idea: 'Xu, Honda & Sugiyama (2018): the UGapE gap index on linear arms; the greedy rule pulls the arm that most shrinks the uncertainty of the direction between the leader and its most ambiguous rival. C_t = s√(2 ln(6K²t²/(π²δ))).',
    create(c) {
      const { K, S, s, delta, lin } = c;
      const model = makeLin(c);
      let i = 0, j = 1, B = Infinity;
      const yv = new Float64Array(lin.linear ? lin.d : 1), u = new Float64Array(lin.linear ? lin.d : 1);
      function compute(t) {
        const ct = s * linConf(K, t, delta);
        let iv = -Infinity;
        for (let a = 0; a < K; a++) { const v = model.mean(a); if (v > iv) { iv = v; i = a; } }
        B = -Infinity;
        for (let a = 0; a < K; a++) {
          if (a === i) continue;
          const v = model.mean(a) - iv + ct * Math.sqrt(model.ySq(a, i));
          if (v > B) { B = v; j = a; }
        }
      }
      return {
        select(t) {
          if (t <= K) return t - 1;
          compute(t);
          if (model.diag) return S.n[i] <= S.n[j] ? i : j;
          const d = model.d, X = model.X, Ai = model.Ainv;
          for (let k = 0; k < d; k++) yv[k] = X[i * d + k] - X[j * d + k];
          for (let r = 0; r < d; r++) { let sm = 0; for (let q = 0; q < d; q++) sm += Ai[r * d + q] * yv[q]; u[r] = sm; }
          let best = 0, bv = -1;
          for (let a = 0; a < K; a++) {
            let xu = 0, xAx = 0;
            for (let r = 0; r < d; r++) {
              xu += X[a * d + r] * u[r];
              let sm = 0;
              for (let q = 0; q < d; q++) sm += Ai[r * d + q] * X[a * d + q];
              xAx += X[a * d + r] * sm;
            }
            const gain = xu * xu / (1 + xAx);
            if (gain > bv) { bv = gain; best = a; }
          }
          return best;
        },
        update(a, r) { model.update(a, r); },
        stop(t) { if (t < K) return false; compute(t); return B <= 0; },
        recommend() { return linBest(model, K); },
      };
    },
  };

  const algorithms = [
    uniform, se, me, lucb, lucbpp, ugape, lilucb, gifa, tasC, tasD, fws, ttts, t3c, ttucb,
    sr, sh, sar, xyStatic, xyAdaptive, rage, peace, lingape,
  ];
  const ALGO_BY_ID = {};
  for (const a of algorithms) ALGO_BY_ID[a.id] = a;

  function defaultParams(def) {
    const p = {};
    for (const q of def.params) p[q.key] = q.default;
    return p;
  }

  // -------------------------------------------------------------- runner ----
  function makeGrid(K, T, L = 24) {
    const start = Math.min(T, Math.max(2, K));
    const xs = [];
    for (let i = 0; i < L; i++) {
      const v = i === L - 1 ? T : Math.round(start * Math.pow(T / start, i / (L - 1)));
      if (!xs.length || v > xs[xs.length - 1]) xs.push(v);
    }
    return xs;
  }

  function createRunner(config) {
    const env = ENV_BY_ID[config.env] || environments[0];
    const K = clamp(Math.round(config.K || 8), 2, 50);
    const T = Math.max(K, Math.round(config.T || 2000));
    const runs = Math.max(1, Math.round(config.runs || 100));
    const seed = (config.seed == null ? 1 : config.seed) >>> 0;
    const extra = config.extra || {};
    const delta = clamp(extra.delta == null ? 0.05 : +extra.delta, 1e-9, 0.999);
    const m = env.m ? env.m(K) : 1;
    const bern = env.type === 'bernoulli';
    const sigma = bern ? 0.5 : (env.sigma || 1);
    const s = sigma;
    const kl = bern ? klBern : (p, q) => (p - q) * (p - q) / (2 * sigma * sigma);
    const beta = (t) => Math.log((1 + Math.log(t)) / delta);
    const grid = makeGrid(K, T), L = grid.length;
    const gridAt = new Int32Array(T + 1).fill(-1);
    grid.forEach((x, i) => { gridAt[x] = i; });

    const slots = (config.algorithms || []).map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaultParams(def), spec.params || {}),
        err: new Float64Array(L), taus: [], stops: [], capped: 0, stopWrong: 0,
        pullsByRank: new Float64Array(K), pullsTotal: 0, timeMs: 0,
      };
    });

    const streams = new Float64Array(K * T);
    const S = { n: new Int32Array(K), sum: new Float64Array(K), mean: new Float64Array(K) };
    const isTop = new Uint8Array(K), rank = new Int32Array(K);
    const recVals = new Float64Array(K), recOut = new Int32Array(m), recIn = new Uint8Array(K);
    let lin = { linear: false, d: K, X: null };
    let done = 0;

    function drawRun(run) {
      const rng = Rng(hash32(seed, run, 'env'));
      const raw = env.draw(K, rng);
      const perm = Array.from({ length: K }, (_, a) => a);
      for (let i = K - 1; i > 0; i--) { const j = rng.int(i + 1); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
      const mu = new Float64Array(K);
      for (let i = 0; i < K; i++) mu[perm[i]] = raw.mu[i];
      if (env.linear) {
        const d = raw.d, X = new Float64Array(K * d);
        for (let i = 0; i < K; i++) for (let j = 0; j < d; j++) X[perm[i] * d + j] = raw.X[i * d + j];
        lin = { linear: true, d, X };
      } else lin = { linear: false, d: K, X: null };
      const order = Array.from({ length: K }, (_, a) => a).sort((x, y) => mu[y] - mu[x] || x - y);
      order.forEach((a, r) => { rank[a] = r; isTop[a] = r < m ? 1 : 0; });
      // one independent stream per arm, so the k-th pull of arm a does not depend on T or K
      const armSeed = rng.int(4294967296);
      for (let a = 0; a < K; a++) {
        const base = a * T, p = mu[a], ar = Rng(hash32(armSeed, a));
        if (bern) for (let k = 0; k < T; k++) streams[base + k] = ar.uniform() < p ? 1 : 0;
        else for (let k = 0; k < T; k++) streams[base + k] = p + sigma * ar.normal();
      }
      return mu;
    }

    function correct(arms) {
      for (let k = 0; k < m; k++) if (!isTop[arms[k]]) return false;
      if (m > 1) { const seen = new Set(); for (let k = 0; k < m; k++) seen.add(arms[k]); if (seen.size !== m) return false; }
      return true;
    }
    function empRec() {
      for (let a = 0; a < K; a++) recVals[a] = S.n[a] ? S.mean[a] : -Infinity;
      return topMInto(recOut, recIn, recVals, m);
    }
    function glrStop(t) {
      const thr = beta(t);
      empRec();
      for (let k = 0; k < m; k++) {
        const i = recOut[k];
        for (let j = 0; j < K; j++) if (!recIn[j] && pairZ(S, kl, i, j) <= thr) return false;
      }
      return true;
    }

    /** Play one algorithm for `budget` pulls; mode 'full' records stopping and grid errors. */
    function play(slot, run, budget, mode) {
      S.n.fill(0); S.sum.fill(0); S.mean.fill(0);
      const def = slot.def;
      const rng = Rng(hash32(seed, run, def.id));
      const agent = def.create({ K, m, T: budget, delta, S, rng, params: slot.params, bern, sigma, s, kl, beta, lin });
      const native = m === 1 || def.topm;
      const ownStop = native && typeof agent.stop === 'function';
      const ownRec = native && typeof agent.recommend === 'function';
      const rec = () => (ownRec ? agent.recommend() : empRec());
      const full = mode === 'full', gridRec = full && !def.budgeted;
      let tau = 0;
      for (let t = 1; t <= budget; t++) {
        const a = agent.select(t);
        if (!(a >= 0 && a < K) || a !== (a | 0)) throw new Error(`${def.id} selected invalid arm ${a}`);
        const r = streams[a * T + S.n[a]];
        S.n[a]++; S.sum[a] += r; S.mean[a] = S.sum[a] / S.n[a];
        agent.update(a, r, t);
        if (full && !tau) {
          slot.pullsByRank[rank[a]]++;
          if (ownStop ? agent.stop(t) : glrStop(t)) {
            tau = t;
            if (!correct(ownStop ? rec() : recOut)) slot.stopWrong++;
          }
        }
        if (gridRec && gridAt[t] >= 0 && !correct(rec())) slot.err[gridAt[t]]++;
      }
      if (full) {
        if (tau) { slot.taus.push(tau); slot.stops.push(tau); } else { slot.taus.push(budget); slot.capped++; }
        slot.pullsTotal += tau || budget;
        if (def.budgeted && !correct(rec())) slot.err[L - 1]++;
        return true;
      }
      return correct(rec());
    }

    function runOne(run) {
      drawRun(run);
      for (const slot of slots) {
        const t0 = now();
        play(slot, run, T, 'full');
        if (slot.def.budgeted) for (let i = 0; i < L - 1; i++) if (!play(slot, run, grid[i], 'budget')) slot.err[i]++;
        slot.timeMs += now() - t0;
      }
    }

    return {
      config: { env: env.id, K, T, runs, seed, delta, m },
      get done() { return done; },
      get finished() { return done >= runs; },
      step(n = 1) {
        const stop = Math.min(runs, done + n);
        while (done < stop) runOne(done++);
        return done;
      },
      results() {
        const R = Math.max(1, done);
        const x = Float64Array.from(grid);
        return slots.map((sl) => {
          const error = new Float64Array(L), errorSE = new Float64Array(L), stopped = new Float64Array(L);
          for (let i = 0; i < L; i++) {
            const p = sl.err[i] / R;
            error[i] = p;
            errorSE[i] = R > 1 ? Math.sqrt(p * (1 - p) / (R - 1)) : 0;
          }
          // stopped-by-t curve: runs still going at the cap never count as stopped
          const sorted = sl.taus.slice().sort((a, b) => a - b);
          const stops = sl.stops.slice().sort((a, b) => a - b);
          for (let i = 0, j = 0; i < L; i++) {
            while (j < stops.length && stops[j] <= grid[i]) j++;
            stopped[i] = j / R;
          }
          let sum = 0, sq = 0;
          for (const v of sorted) { sum += v; sq += v * v; }
          const n = sorted.length || 1;
          const mean = sum / n;
          const sd = n > 1 ? Math.sqrt(Math.max(0, (sq - n * mean * mean) / (n - 1))) : 0;
          let avg = 0;
          for (let i = 0; i < L; i++) avg += error[i] / L;
          const summary = {
            samples: sorted.length ? mean : 0,
            samplesSE: sorted.length ? sd / Math.sqrt(n) : 0,
            samplesMedian: sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : 0,
            capHit: done ? sl.capped / done : 0,
            errorStop: done ? sl.stopWrong / done : 0,
            errorFinal: error[L - 1],
            errorAvg: avg,
            msPerRun: done ? sl.timeMs / done : 0,
          };
          const pullShare = Array.from(sl.pullsByRank, (v) => (sl.pullsTotal ? v / sl.pullsTotal : 0));
          return {
            id: sl.def.id, name: sl.def.name, family: sl.def.family, params: sl.params, runs: done,
            x, curves: { error, stopped }, curvesSE: { error: errorSE },
            summary, finals: sorted, pullShare,
          };
        });
      },
    };
  }

  return MAB.registerSetting({
    id: 'bai',
    name: 'Best-arm identification',
    short: 'Best arm',
    blurb: 'Pure exploration: sample arms to name the best one (or the top m) — as fast as possible at confidence 1 − δ, or as reliably as possible within a budget of T pulls.',
    families, environments, algorithms,
    defaultEnv: 'bern-easy',
    defaultLineup: ['uniform', 'se', 'lucb', 'ugape', 'tas-d', 'ttts', 'ttucb', 'sh'],
    presets: [
      { label: 'Fixed confidence', algos: ['uniform', 'se', 'lucb', 'lucbpp', 'ugape', 'lilucb', 'tas-c', 'tas-d'] },
      { label: 'Asymptotically optimal', algos: ['tas-c', 'tas-d', 'fws', 'ttts', 't3c', 'ttucb', 'uniform'] },
      { label: 'Fixed budget', algos: ['sr', 'sh', 'uniform', 'ttts', 'ugape', 'tas-d', 'me'] },
      { label: 'Top-m', algos: ['sar', 'sr', 'sh', 'lucb', 'lucbpp', 'ugape', 'gifa', 'uniform'], env: 'topm' },
      { label: 'Linear', algos: ['xy-static', 'xy-adaptive', 'rage', 'peace', 'lingape', 'tas-d', 'lucb', 'uniform'], env: 'linear' },
    ],
    controls: {
      K: { label: 'Arms (K)', min: 2, max: 30, default: 8 },
      T: { label: 'Budget T = stopping cap (pulls)', options: [500, 1000, 2000, 5000, 10000, 20000], default: 5000 },
      runs: { label: 'Runs averaged', options: [20, 50, 100, 200, 500], default: 100 },
      extra: [{ key: 'delta', label: 'Confidence δ', min: 0.001, max: 0.5, step: 0.001, default: 0.05 }],
    },
    xLabel: 'pulls t (budget)',
    metrics: [
      {
        key: 'error', label: 'P(wrong answer) after t pulls', better: 'lower', fmt: 'pct', band: true, zeroToOne: true,
        note: 'Fixed-budget view on a log grid of budgets: share of runs whose recommendation after t pulls is not the best arm (top-m: not exactly the top-m set). Anytime algorithms keep sampling past their stopping time; SR, SH and SAR are re-run from scratch with budget t on the same reward streams. Bands: 95% CI.',
      },
      {
        key: 'stopped', label: 'Stopped by t (fixed confidence)', better: 'higher', fmt: 'pct', zeroToOne: true,
        note: 'Share of runs whose stopping rule had fired at level δ by t pulls — the distribution of samples-to-stop. Algorithms without their own rule (Track-and-Stop, FWS, top-two, uniform, SR/SH/SAR, XY-Static on unstructured arms) use the GLR rule with β(t, δ) = ln((1 + ln t)/δ). Runs still going at the cap T never count.',
      },
    ],
    final: { key: 'samples', se: 'samplesSE', median: 'samplesMedian', label: 'Samples to stop (cap counts as T)', better: 'lower' },
    tiles: [
      { key: 'samples', label: 'Stops soonest', better: 'lower', fmt: 'num', sub: 'mean pulls to reach confidence 1 − δ' },
      { key: 'errorFinal', label: 'Most accurate at budget T', better: 'lower', fmt: 'pct', sub: 'wrong answers after T pulls' },
      { key: 'errorAvg', label: 'Most accurate across budgets', better: 'lower', fmt: 'pct', sub: 'error averaged over the log grid' },
    ],
    columns: [
      { key: 'samples', label: 'Samples to stop', fmt: 'num', better: 'lower' },
      { key: 'samplesSE', label: '95% CI', fmt: 'ci' },
      { key: 'samplesMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'capHit', label: 'Hit cap T', fmt: 'pct', better: 'lower' },
      { key: 'errorStop', label: 'Wrong at stop (≤ δ?)', fmt: 'pct', better: 'lower' },
      { key: 'errorFinal', label: 'Wrong at T', fmt: 'pct', better: 'lower' },
      { key: 'errorAvg', label: 'Wrong, avg over t', fmt: 'pct', better: 'lower' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: true,
    createRunner,
    // exposed for tests
    _internal: { optimalWeights, designXY, makeGrid, ENV_BY_ID, ALGO_BY_ID },
  });
});
