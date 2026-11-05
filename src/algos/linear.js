/*
 * linear.js — linear, generalized-linear, sparse, low-rank and non-stationary linear bandits,
 * plus environments built for them.
 *
 * Feature convention (docs/ARCHITECTURE.md): select(t, X) receives X = Float64Array(K*d) whose
 * row a holds arm a's features. Most algorithms here assume ONE shared parameter θ, so the
 * expected reward of arm a is f(θ·xₐ). That only identifies arms whose feature rows differ.
 * When every row is identical (context-free environments pass d = 1 and all ones; the core
 * 'contextual' environment shows the same context to every arm) we switch to the standard
 * disjoint embedding φ(x, a) = eₐ ⊗ x: one independent d-dimensional block per arm. Each
 * algorithm keeps one statistics block per "block" and arm a reads block `of[a]`, so the same
 * code is a shared-θ linear bandit on feature-rich environments and a per-arm model elsewhere.
 * The mode is decided from the first context seen and kept for the run.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { util } = MAB;
  const { argmax, normInv } = util;

  // ======================================================== linear algebra ====
  /** Gauss–Jordan inverse with partial pivoting. Returns ln|det A| (NaN if singular). */
  function invert(A, n, out, work) {
    const M = work && work.length >= n * n ? work : new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) M[i] = A[i];
    out.fill(0, 0, n * n);
    for (let i = 0; i < n; i++) out[i * n + i] = 1;
    let logdet = 0;
    for (let c = 0; c < n; c++) {
      let p = c, best = Math.abs(M[c * n + c]);
      for (let r = c + 1; r < n; r++) { const v = Math.abs(M[r * n + c]); if (v > best) { best = v; p = r; } }
      if (!(best > 1e-300)) return NaN;
      if (p !== c) {
        for (let j = 0; j < n; j++) {
          let tmp = M[p * n + j]; M[p * n + j] = M[c * n + j]; M[c * n + j] = tmp;
          tmp = out[p * n + j]; out[p * n + j] = out[c * n + j]; out[c * n + j] = tmp;
        }
      }
      const piv = M[c * n + c];
      logdet += Math.log(Math.abs(piv));
      const inv = 1 / piv;
      for (let j = 0; j < n; j++) { M[c * n + j] *= inv; out[c * n + j] *= inv; }
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r * n + c];
        if (f === 0) continue;
        for (let j = 0; j < n; j++) { M[r * n + j] -= f * M[c * n + j]; out[r * n + j] -= f * out[c * n + j]; }
      }
    }
    return logdet;
  }

  /** Lower Cholesky factor L (LLᵀ = A). Returns false if A is not positive definite. */
  function cholesky(A, n, L) {
    L.fill(0, 0, n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) { if (!(s > 0)) return false; L[i * n + i] = Math.sqrt(s); }
        else L[i * n + j] = s / L[j * n + j];
      }
    }
    return true;
  }

  /** Inverse of a symmetric positive-definite matrix via Cholesky (≈ n³/2 flops); falls back to Gauss–Jordan. */
  function invertSPD(A, n, out, L, Li) {
    if (!cholesky(A, n, L)) return invert(A, n, out);
    Li.fill(0, 0, n * n);
    for (let i = 0; i < n; i++) {
      Li[i * n + i] = 1 / L[i * n + i];
      for (let j = 0; j < i; j++) {
        let s = 0;
        for (let k = j; k < i; k++) s += L[i * n + k] * Li[k * n + j];
        Li[i * n + j] = -s / L[i * n + i];
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let k = i; k < n; k++) s += Li[k * n + i] * Li[k * n + j];
        out[i * n + j] = out[j * n + i] = s;
      }
    }
    return 0;
  }

  /**
   * Cyclic Jacobi eigen-decomposition of a symmetric n×n matrix. Fills vals (descending) and
   * vecs with vecs[i*n + k] = i-th component of the k-th eigenvector.
   */
  function eigSym(A, n, vals, vecs) {
    const a = new Float64Array(A.subarray ? A.subarray(0, n * n) : A), V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    let scale = 0;
    for (let i = 0; i < n * n; i++) scale += a[i] * a[i];
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
      if (off <= 1e-24 * scale || off === 0) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = a[p * n + q];
          if (Math.abs(apq) < 1e-300) continue;
          const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {
            if (k === p || k === q) continue;
            const akp = a[k * n + p], akq = a[k * n + q];
            a[k * n + p] = a[p * n + k] = c * akp - s * akq;
            a[k * n + q] = a[q * n + k] = s * akp + c * akq;
          }
          a[p * n + p] -= t * apq;
          a[q * n + q] += t * apq;
          a[p * n + q] = a[q * n + p] = 0;
          for (let k = 0; k < n; k++) {
            const vkp = V[k * n + p], vkq = V[k * n + q];
            V[k * n + p] = c * vkp - s * vkq;
            V[k * n + q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[y * n + y] - a[x * n + x]);
    for (let k = 0; k < n; k++) {
      const src = order[k];
      vals[k] = a[src * n + src];
      for (let i = 0; i < n; i++) vecs[i * n + k] = V[i * n + src];
    }
  }

  /** Sherman–Morrison: Minv ← (M + w·xxᵀ)⁻¹ given Minv = M⁻¹. Returns 1 + w·xᵀM⁻¹x (0 if rejected). */
  function shermanMorrison(Minv, d, X, o, w, u) {
    let q = 0;
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += Minv[i * d + j] * X[o + j];
      u[i] = s; q += X[o + i] * s;
    }
    const den = 1 + w * q;
    if (!(den > 1e-9)) return 0;
    const f = w / den;
    for (let i = 0; i < d; i++) {
      const fi = f * u[i];
      if (fi === 0) continue;
      for (let j = 0; j < d; j++) Minv[i * d + j] -= fi * u[j];
    }
    return den;
  }

  /** xᵀ M x for the row of X starting at o. */
  function quadForm(M, d, X, o) {
    let q = 0;
    for (let i = 0; i < d; i++) {
      const xi = X[o + i];
      if (xi === 0) continue;
      let s = 0;
      for (let j = 0; j < d; j++) s += M[i * d + j] * X[o + j];
      q += xi * s;
    }
    return q > 0 ? q : 0;
  }

  function dotRow(v, d, X, o) {
    let s = 0;
    for (let j = 0; j < d; j++) s += v[j] * X[o + j];
    return s;
  }

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }

  function softThreshold(z, l) { return z > l ? z - l : z < -l ? z + l : 0; }

  /**
   * Lasso by cyclic coordinate descent on sufficient statistics:
   * minimise (1/2n)·βᵀGβ − (1/n)·cᵀβ + lam·‖β‖₁ with G = Σxxᵀ, c = Σrx. Warm-started from beta.
   * One sweep costs O(d²) whatever n is, so refits are cheap.
   */
  function lasso(G, c, n, d, lam, beta, sweeps = 100, tol = 1e-6) {
    if (n <= 0) { beta.fill(0); return beta; }
    for (let s = 0; s < sweeps; s++) {
      let change = 0;
      for (let j = 0; j < d; j++) {
        const gjj = G[j * d + j];
        if (!(gjj > 0)) { beta[j] = 0; continue; }
        let rho = c[j];
        for (let k = 0; k < d; k++) if (k !== j) rho -= G[j * d + k] * beta[k];
        const nb = softThreshold(rho / n, lam) / (gjj / n);
        const ch = Math.abs(nb - beta[j]);
        if (ch > change) change = ch;
        beta[j] = nb;
      }
      if (change < tol) break;
    }
    return beta;
  }

  /** Largest eigenvalue of a symmetric PSD matrix by power iteration. */
  function lambdaMax(G, d, iters = 50) {
    let v = new Float64Array(d).fill(1 / Math.sqrt(d)), w = new Float64Array(d), lam = 0;
    for (let it = 0; it < iters; it++) {
      let nrm = 0;
      for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += G[i * d + j] * v[j]; w[i] = s; nrm += s * s; }
      nrm = Math.sqrt(nrm);
      if (!(nrm > 0)) return 0;
      lam = nrm;
      for (let i = 0; i < d; i++) v[i] = w[i] / nrm;
    }
    return lam;
  }

  // ====================================================== shared models =====
  /**
   * Ridge regression / recursive least squares with prior precision diag(Λ) (a number λ or a
   * Float64Array). Keeps V⁻¹, b = Σ w r x, θ̂ = V⁻¹b and ln(det V / det Λ).
   */
  function Ridge(d, prior) {
    const Vinv = new Float64Array(d * d), b = new Float64Array(d), theta = new Float64Array(d), u = new Float64Array(d);
    const m = {
      d, Vinv, b, theta, logdet: 0, n: 0,
      reset() {
        Vinv.fill(0); b.fill(0); theta.fill(0); m.logdet = 0; m.n = 0;
        for (let i = 0; i < d; i++) Vinv[i * d + i] = 1 / (typeof prior === 'number' ? prior : prior[i]);
      },
      mean(X, o) { return dotRow(theta, d, X, o); },
      quad(X, o) { return quadForm(Vinv, d, X, o); },
      /** V += w·xxᵀ, b += w·r·x (w = −1 removes an old observation). */
      add(X, o, r, w = 1) {
        const den = shermanMorrison(Vinv, d, X, o, w, u);
        if (!den) return;
        for (let i = 0; i < d; i++) b[i] += w * r * X[o + i];
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += Vinv[i * d + j] * b[j]; theta[i] = s; }
        m.logdet += Math.log(den); m.n += w;
      },
      /** Load explicit statistics: V (d×d), b, and ln(det V / det Λ). */
      setFrom(V, bv, logdetRatio, n) {
        const ld = invert(V, d, Vinv);
        if (Number.isNaN(ld)) return;
        for (let i = 0; i < d; i++) b[i] = bv[i];
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += Vinv[i * d + j] * b[j]; theta[i] = s; }
        m.logdet = logdetRatio; m.n = n;
      },
    };
    m.reset();
    return m;
  }

  /** Decide between shared-θ and disjoint (one block per arm) feature maps. */
  function space(K, ctx) {
    const d = ctx && ctx.d > 0 ? ctx.d | 0 : 1;
    const fb = new Float64Array(K * d);
    for (let a = 0; a < K; a++) fb[a * d] = 1;
    const sp = {
      K, d, B: 1, shared: true, ready: false, of: new Int32Array(K),
      view(X) { return X && X.length === K * d ? X : fb; },
      init(X) {
        if (sp.ready) return;
        let same = true;
        for (let a = 1; a < K && same; a++) for (let j = 0; j < d; j++) if (X[a * d + j] !== X[j]) { same = false; break; }
        sp.shared = !same;
        sp.B = sp.shared ? 1 : K;
        for (let a = 0; a < K; a++) sp.of[a] = sp.shared ? 0 : a;
        sp.ready = true;
      },
      blocks(make) { const out = []; for (let b = 0; b < sp.B; b++) out.push(make(b)); return out; },
    };
    return sp;
  }

  const sigmaOf = (ctx) => (ctx && ctx.env && ctx.env.sigma > 0 ? ctx.env.sigma : 1);
  const horizon = (ctx) => (ctx && ctx.T > 0 ? ctx.T : 1000);
  const isLogistic = (ctx) => !!(ctx && ctx.env && ctx.env.type === 'bernoulli');

  /** OFUL radius βₜ = R√(2 ln(1/δ) + ln(det V/det Λ)) + √λ·S. */
  function ofulBeta(m, R, lambda, delta, S) {
    return R * Math.sqrt(2 * Math.log(1 / delta) + Math.max(0, m.logdet)) + Math.sqrt(lambda) * S;
  }

  /** Growable store of (x, r) rows. */
  function History(d) {
    let cap = 64;
    const h = { n: 0, X: new Float64Array(cap * d), y: new Float64Array(cap) };
    h.push = (X, o, r) => {
      if (h.n === cap) {
        cap *= 2;
        const nx = new Float64Array(cap * d); nx.set(h.X); h.X = nx;
        const ny = new Float64Array(cap); ny.set(h.y); h.y = ny;
      }
      for (let j = 0; j < d; j++) h.X[h.n * d + j] = X[o + j];
      h.y[h.n++] = r;
    };
    return h;
  }

  /**
   * Generalized linear model block: logistic link for Bernoulli environments, identity otherwise.
   * Keeps the design V = λI + Σxxᵀ, the ridge-regularised MLE θ̂ (Newton/IRLS, refit lazily
   * whenever the sample count grows by 10%) and H⁻¹ = (λI + Σ μ̇(xᵀθ̂)xxᵀ)⁻¹, which is updated
   * by Sherman–Morrison between refits and recomputed exactly at each refit.
   */
  function GlmBlock(d, lam, logistic) {
    const V = Ridge(d, lam);
    if (!logistic) {
      return {
        d, V, theta: V.theta, Hinv: V.Vinv, logistic: false,
        get n() { return V.n; },
        mu: (z) => z, dmu: () => 1,
        push(X, o, r) { V.add(X, o, r); },
      };
    }
    const theta = new Float64Array(d), Hinv = new Float64Array(d * d), H = new Float64Array(d * d);
    const g = new Float64Array(d), u = new Float64Array(d), work = new Float64Array(d * d), hist = History(d);
    for (let i = 0; i < d; i++) Hinv[i * d + i] = 1 / lam;
    let nextFit = 1;
    function fit() {
      const n = hist.n, Xh = hist.X, y = hist.y;
      for (let it = 0; it < 3; it++) {
        H.fill(0);
        for (let i = 0; i < d; i++) { H[i * d + i] = lam; g[i] = lam * theta[i]; }
        for (let s = 0; s < n; s++) {
          const o = s * d;
          const p = sigmoid(dotRow(theta, d, Xh, o)), w = p * (1 - p), e = p - y[s];
          for (let i = 0; i < d; i++) {
            const xi = Xh[o + i];
            g[i] += e * xi;
            const wx = w * xi;
            for (let j = i; j < d; j++) H[i * d + j] += wx * Xh[o + j];
          }
        }
        for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) H[i * d + j] = H[j * d + i];
        if (Number.isNaN(invert(H, d, Hinv, work))) break;
        let nrm = 0;
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += Hinv[i * d + j] * g[j]; u[i] = s; nrm += s * s; }
        nrm = Math.sqrt(nrm);
        const damp = nrm > 3 ? 3 / nrm : 1;   // damped Newton step for stability early on
        for (let i = 0; i < d; i++) theta[i] -= damp * u[i];
        if (nrm < 1e-6) break;
      }
    }
    return {
      d, V, theta, Hinv, logistic: true,
      get n() { return hist.n; },
      mu: sigmoid,
      dmu(z) { const p = sigmoid(z); return p * (1 - p); },
      push(X, o, r) {
        hist.push(X, o, r);
        V.add(X, o, r);
        const p = sigmoid(dotRow(theta, d, X, o));
        shermanMorrison(Hinv, d, X, o, p * (1 - p), u);
        if (hist.n >= nextFit) { fit(); nextFit = hist.n + Math.max(1, Math.floor(hist.n / 10)); }
      },
    };
  }

  /** Statistics for Lasso-type estimators: G = Σxxᵀ, c = Σrx, n, and a warm-started β̂. */
  function GramBlock(d) {
    const s = { d, G: new Float64Array(d * d), c: new Float64Array(d), n: 0, beta: new Float64Array(d) };
    s.add = (X, o, r) => {
      for (let i = 0; i < d; i++) {
        const xi = X[o + i];
        if (xi === 0) continue;
        s.c[i] += r * xi;
        for (let j = 0; j < d; j++) s.G[i * d + j] += xi * X[o + j];
      }
      s.n++;
    };
    return s;
  }

  /** Approximate G-optimal (= D-optimal, Kiefer–Wolfowitz) design on `arms` by Frank–Wolfe. */
  function gOptimalDesign(arms, X, sp, iters = 200) {
    const { d, K, of, B } = sp, pi = new Float64Array(K), g = new Float64Array(K);
    const Vb = new Float64Array(B * d * d), Vinv = new Float64Array(B * d * d), tmp = new Float64Array(d * d), work = new Float64Array(d * d);
    for (const a of arms) pi[a] = 1 / arms.length;
    let deff = d;
    for (let it = 0; it < iters; it++) {
      Vb.fill(0);
      for (const a of arms) {
        const o = of[a] * d * d, p = pi[a];
        if (p === 0) continue;
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Vb[o + i * d + j] += p * X[a * d + i] * X[a * d + j];
      }
      for (let b = 0; b < B; b++) {
        const o = b * d * d;
        let tr = 0;
        for (let i = 0; i < d; i++) tr += Vb[o + i * d + i];
        for (let i = 0; i < d * d; i++) tmp[i] = Vb[o + i];
        for (let i = 0; i < d; i++) tmp[i * d + i] += 1e-7 * (tr / d) + 1e-12;
        const out = Vinv.subarray(o, o + d * d);
        invert(tmp, d, out, work);
      }
      let gmax = -Infinity, best = arms[0];
      deff = 0;
      for (const a of arms) {
        g[a] = quadForm(Vinv.subarray(of[a] * d * d, (of[a] + 1) * d * d), d, X, a * d);
        deff += pi[a] * g[a];
        if (g[a] > gmax) { gmax = g[a]; best = a; }
      }
      if (gmax <= 1.02 * deff || gmax <= 1) break;
      const gamma = Math.min(1, Math.max(0, (gmax / deff - 1) / (gmax - 1)));
      if (!(gamma > 0)) break;
      for (const a of arms) pi[a] *= 1 - gamma;
      pi[best] += gamma;
    }
    return { pi, deff: Math.max(1, deff) };
  }

  // =========================================================== families =====
  const families = [
    { id: 'linear', name: 'Linear bandits', blurb: 'One shared parameter θ: an arm\'s expected reward is θ·x for its feature vector x, so every pull teaches something about every arm.' },
    { id: 'glm', name: 'Generalized linear', blurb: 'Rewards follow a link function of θ·x (logistic for clicks); fit θ by maximum likelihood and stay optimistic about it.' },
    { id: 'sparse', name: 'Sparse & low-rank', blurb: 'High-dimensional θ with hidden structure (few non-zero coordinates, or a low-rank matrix): exploit it to learn from far fewer samples.' },
    { id: 'nslinear', name: 'Non-stationary linear', blurb: 'Linear bandits whose θ moves: forget old data with a sliding window, or keep a pool of models and drop the ones that stop fitting.' },
  ];

  // ======================================================= environments =====
  function unitInto(rng, v, o, d, scale = 1) {
    let n = 0;
    for (let j = 0; j < d; j++) { const z = rng.normal(); v[o + j] = z; n += z * z; }
    n = Math.sqrt(n) || 1;
    for (let j = 0; j < d; j++) v[o + j] *= scale / n;
    return v;
  }
  /** Fresh N(0, I) features per arm per round, reward θₜ·x + N(0, σ²). */
  function fillFreshLinear(K, T, rng, means, rewards, ctx, d, thetaAt, sigma) {
    for (let t = 0; t < T; t++) {
      const theta = thetaAt(t);
      for (let a = 0; a < K; a++) {
        const row = (t * K + a) * d;
        let m = 0;
        for (let j = 0; j < d; j++) { const x = rng.normal(); ctx[row + j] = x; m += theta[j] * x; }
        means[t * K + a] = m;
        rewards[t * K + a] = m + sigma * rng.normal();
      }
    }
  }

  const environments = [
    {
      id: 'linear-shared', name: 'Linear bandit, shared θ', short: 'Linear',
      description: 'Every round each arm gets fresh features xₜ,ₐ ~ N(0, I₈). Arm a pays θ·xₜ,ₐ + N(0, 1) with one hidden θ on the unit sphere, shared by all arms. Ignoring the features, every arm is equally good on average.',
      type: 'gaussian', stationary: true, contextual: true, d: 8, tags: ['context', 'linear'],
      fill(K, T, rng, means, rewards, ctx) {
        const d = 8, theta = unitInto(rng, new Float64Array(d), 0, d);
        fillFreshLinear(K, T, rng, means, rewards, ctx, d, () => theta, 1);
      },
    },
    {
      id: 'linear-fixed', name: 'Linear bandit, fixed arm set', short: 'Fixed arms',
      description: 'K fixed arms xₐ drawn uniformly on the unit sphere in R⁵ (the same features every round). Arm a pays θ·xₐ + N(0, 0.5²) with a hidden unit θ. A finite-armed linear bandit, the home ground of G-optimal design and elimination.',
      type: 'gaussian', sigma: 0.5, stationary: true, contextual: true, d: 5, tags: ['context', 'linear', 'fixed arms'],
      fill(K, T, rng, means, rewards, ctx) {
        const d = 5, theta = unitInto(rng, new Float64Array(d), 0, d), arms = new Float64Array(K * d), mu = new Float64Array(K);
        for (let a = 0; a < K; a++) { unitInto(rng, arms, a * d, d); mu[a] = dotRow(theta, d, arms, a * d); }
        for (let t = 0; t < T; t++) {
          ctx.set(arms, t * K * d);
          for (let a = 0; a < K; a++) { means[t * K + a] = mu[a]; rewards[t * K + a] = mu[a] + 0.5 * rng.normal(); }
        }
      },
    },
    {
      id: 'logistic', name: 'Logistic (GLM) bandit', short: 'Logistic',
      description: 'Fresh features xₜ,ₐ ~ N(0, I₆) every round; arm a pays 1 with probability σ(θ·xₜ,ₐ), σ the logistic function, with a hidden θ of norm 2. A click-through model where rewards saturate near 0 and 1.',
      type: 'bernoulli', stationary: true, contextual: true, d: 6, tags: ['context', 'glm'],
      fill(K, T, rng, means, rewards, ctx) {
        const d = 6, theta = unitInto(rng, new Float64Array(d), 0, d, 2);
        for (let t = 0; t < T; t++) {
          for (let a = 0; a < K; a++) {
            const row = (t * K + a) * d;
            let z = 0;
            for (let j = 0; j < d; j++) { const x = rng.normal(); ctx[row + j] = x; z += theta[j] * x; }
            const p = sigmoid(z);
            means[t * K + a] = p;
            rewards[t * K + a] = rng.uniform() < p ? 1 : 0;
          }
        }
      },
    },
    {
      id: 'sparse', name: 'Sparse high-dimensional linear', short: 'Sparse',
      description: 'Fresh features in R⁴⁰ every round, xₜ,ₐ = √½·uₜ + √½·eₜ,ₐ with u, e ~ N(0, I) (a shared per-round part makes arms correlated); reward θ·xₜ,ₐ + N(0, 1). Only s = 4 of the 40 coordinates of θ are non-zero (‖θ‖ = 1). Methods that assume sparsity (Lasso) need far fewer samples than d-dimensional ridge regression.',
      type: 'gaussian', stationary: true, contextual: true, d: 40, tags: ['context', 'sparse'],
      fill(K, T, rng, means, rewards, ctx) {
        const d = 40, s = 4, theta = new Float64Array(d);
        let placed = 0, nrm = 0;
        while (placed < s) {
          const j = rng.int(d);
          if (theta[j] !== 0) continue;
          theta[j] = (rng.uniform() < 0.5 ? -1 : 1) * (0.5 + 0.5 * rng.uniform());
          nrm += theta[j] * theta[j]; placed++;
        }
        nrm = Math.sqrt(nrm);
        for (let j = 0; j < d; j++) theta[j] /= nrm;
        // xₜ,ₐ = √ρ·uₜ + √(1−ρ)·eₜ,ₐ: a shared per-round component makes arms' features correlated
        const rho = 0.5, u = new Float64Array(d), su = Math.sqrt(rho), se = Math.sqrt(1 - rho);
        for (let t = 0; t < T; t++) {
          for (let j = 0; j < d; j++) u[j] = su * rng.normal();
          for (let a = 0; a < K; a++) {
            const row = (t * K + a) * d;
            let m = 0;
            for (let j = 0; j < d; j++) { const x = u[j] + se * rng.normal(); ctx[row + j] = x; m += theta[j] * x; }
            means[t * K + a] = m;
            rewards[t * K + a] = m + rng.normal();
          }
        }
      },
    },
    {
      id: 'low-rank', name: 'Bilinear, low-rank', short: 'Low-rank',
      description: 'Each round a user context xₜ ~ N(0, I₅); arm a is an item with fixed features zₐ on the unit sphere in R⁵. Reward xₜᵀΘzₐ + N(0, 0.5²) with a hidden rank-2 Θ (singular values 1.5 and 1). Layout: row a of X is vec(xₜzₐᵀ), X[a·25 + 5i + j] = xₜ[i]·zₐ[j], so the problem is linear in 25 dimensions but only 2·(5+5)−4 = 16 directions matter.',
      type: 'gaussian', sigma: 0.5, stationary: true, contextual: true, d: 25, tags: ['context', 'low-rank'],
      fill(K, T, rng, means, rewards, ctx) {
        const d1 = 5, d2 = 5, d = d1 * d2, r = 2, sv = [1.5, 1];
        const U = new Float64Array(d1 * r), V = new Float64Array(d2 * r), Th = new Float64Array(d1 * d2);
        // orthonormal columns by Gram–Schmidt
        for (const [M, n] of [[U, d1], [V, d2]]) {
          for (let k = 0; k < r; k++) {
            for (;;) {
              let nrm = 0;
              for (let i = 0; i < n; i++) M[i * r + k] = rng.normal();
              for (let l = 0; l < k; l++) {
                let dp = 0;
                for (let i = 0; i < n; i++) dp += M[i * r + k] * M[i * r + l];
                for (let i = 0; i < n; i++) M[i * r + k] -= dp * M[i * r + l];
              }
              for (let i = 0; i < n; i++) nrm += M[i * r + k] * M[i * r + k];
              nrm = Math.sqrt(nrm);
              if (nrm > 1e-6) { for (let i = 0; i < n; i++) M[i * r + k] /= nrm; break; }
            }
          }
        }
        for (let i = 0; i < d1; i++) for (let j = 0; j < d2; j++) for (let k = 0; k < r; k++) Th[i * d2 + j] += sv[k] * U[i * r + k] * V[j * r + k];
        const Z = new Float64Array(K * d2), x = new Float64Array(d1), Tz = new Float64Array(K * d1);
        for (let a = 0; a < K; a++) {
          unitInto(rng, Z, a * d2, d2);
          for (let i = 0; i < d1; i++) { let s = 0; for (let j = 0; j < d2; j++) s += Th[i * d2 + j] * Z[a * d2 + j]; Tz[a * d1 + i] = s; }
        }
        for (let t = 0; t < T; t++) {
          for (let i = 0; i < d1; i++) x[i] = rng.normal();
          for (let a = 0; a < K; a++) {
            const row = (t * K + a) * d;
            let m = 0;
            for (let i = 0; i < d1; i++) {
              m += x[i] * Tz[a * d1 + i];
              for (let j = 0; j < d2; j++) ctx[row + i * d2 + j] = x[i] * Z[a * d2 + j];
            }
            means[t * K + a] = m;
            rewards[t * K + a] = m + 0.5 * rng.normal();
          }
        }
        return { bilinear: { d1, d2, layout: 'row-major vec(x zᵀ)' } };
      },
    },
    {
      id: 'graph-smooth', name: 'Smooth on a graph', short: 'Graph-smooth',
      description: 'Arms are the nodes of a random graph (a ring plus random chords). Means are smooth on the graph: μ = Σₖ αₖuₖ over the 8 smoothest Laplacian eigenvectors uₖ with αₖ ~ N(0, 1)/(0.2 + λₖ), then centred and scaled to standard deviation 1; rewards add N(0, 1). Row a of X holds node a\'s coordinates in the 8 smoothest eigenvectors (zero-padded when K < 8); info.spectral carries the eigenvalues and eigenvectors.',
      type: 'gaussian', stationary: true, contextual: true, d: 8, tags: ['context', 'graph'],
      fill(K, T, rng, means, rewards, ctx) {
        const d = 8, W = new Uint8Array(K * K), L = new Float64Array(K * K);
        const link = (a, b) => { if (a !== b) { W[a * K + b] = 1; W[b * K + a] = 1; } };
        for (let a = 0; a < K; a++) link(a, (a + 1) % K);
        for (let a = 0; a < K; a++) for (let b = a + 2; b < K; b++) if (rng.uniform() < 1.5 / K) link(a, b);
        for (let a = 0; a < K; a++) {
          let deg = 0;
          for (let b = 0; b < K; b++) if (W[a * K + b]) { L[a * K + b] = -1; deg++; }
          L[a * K + a] = deg;
        }
        const valsDesc = new Float64Array(K), vecsDesc = new Float64Array(K * K);
        eigSym(L, K, valsDesc, vecsDesc);
        // ascending order: smoothest first
        const vals = new Float64Array(K), vecs = new Float64Array(K * K);
        for (let k = 0; k < K; k++) {
          vals[k] = Math.max(0, valsDesc[K - 1 - k]);
          for (let i = 0; i < K; i++) vecs[i * K + k] = vecsDesc[i * K + K - 1 - k];
        }
        const m = Math.min(d, K), mu = new Float64Array(K);
        for (let k = 0; k < m; k++) {
          const alpha = rng.normal() / (0.2 + vals[k]);
          for (let a = 0; a < K; a++) mu[a] += alpha * vecs[a * K + k];
        }
        let mean = 0, sd = 0;
        for (let a = 0; a < K; a++) mean += mu[a] / K;
        for (let a = 0; a < K; a++) sd += (mu[a] - mean) * (mu[a] - mean) / K;
        sd = Math.sqrt(sd) || 1;
        for (let a = 0; a < K; a++) mu[a] = (mu[a] - mean) / sd;
        const row = new Float64Array(K * d);
        for (let a = 0; a < K; a++) for (let k = 0; k < m; k++) row[a * d + k] = vecs[a * K + k];
        for (let t = 0; t < T; t++) {
          ctx.set(row, t * K * d);
          for (let a = 0; a < K; a++) { means[t * K + a] = mu[a]; rewards[t * K + a] = mu[a] + rng.normal(); }
        }
        return { spectral: { d, m, eigenvalues: vals, eigenvectors: vecs, adjacency: W } };
      },
    },
    {
      id: 'linear-drift', name: 'Linear bandit, switching θ', short: 'Linear, switching',
      description: 'Like the shared-θ linear bandit (fresh features xₜ,ₐ ~ N(0, I₈), reward θ·xₜ,ₐ + N(0, 1)), but the hidden unit θ is redrawn at four evenly spaced change points.',
      type: 'gaussian', stationary: false, contextual: true, d: 8, tags: ['context', 'linear', 'changes'],
      fill(K, T, rng, means, rewards, ctx) {
        const d = 8, theta = new Float64Array(d), segment = Math.ceil(T / 5);
        fillFreshLinear(K, T, rng, means, rewards, ctx, d, (t) => {
          if (t % segment === 0) unitInto(rng, theta, 0, d);
          return theta;
        }, 1);
      },
    },
  ];

  // ========================================================= algorithms =====
  const P = {
    lambda: { key: 'lambda', label: 'λ (ridge)', default: 1, min: 0.01, max: 10, step: 0.01 },
    delta: { key: 'delta', label: 'δ (confidence)', default: 0.05, min: 0.001, max: 0.5, step: 0.001 },
    S: { key: 'S', label: 'S (bound on ‖θ‖)', default: 1, min: 0, max: 10, step: 0.1 },
  };

  /** Shared skeleton: lazily set up blocks on the first context, score arms, argmax. */
  function indexPolicy(K, ctx, makeBlock, score, learn) {
    const sp = space(K, ctx), idx = new Float64Array(K), rng = ctx.rng;
    let M = null, X = null;
    return {
      sp,
      select(t, Xt) {
        X = sp.view(Xt);
        if (!M) { sp.init(X); M = sp.blocks(makeBlock); }
        score(t, X, M, idx);
        return argmax(idx, rng);
      },
      update(a, r, t) { if (M) learn(M[sp.of[a]], X, a * sp.d, r, t, a); },
    };
  }

  // -------------------------------------------------------------- OFUL -----
  const oful = {
    id: 'oful', name: 'OFUL', family: 'linear',
    params: [P.lambda, P.delta, P.S],
    formula: 'argmaxₐ θ̂·xₐ + βₜ‖xₐ‖_{V⁻¹},  βₜ = σ√(2 ln(1/δ) + ln(det Vₜ/λᵈ)) + √λ·S',
    regret: 'Õ(d√T)',
    idea: 'Abbasi-Yadkori, Pál & Szepesvári: one ridge regression for a shared θ and a self-normalised confidence ellipsoid around it; play the arm with the most optimistic reward over the ellipsoid.',
    create(K, ctx) {
      const { lambda, delta, S } = ctx.params, R = sigmaOf(ctx), d = space(K, ctx).d;
      return indexPolicy(K, ctx, () => Ridge(d, lambda), (t, X, M, idx) => {
        const beta = M.map((m) => ofulBeta(m, R, lambda, delta, S));
        for (let a = 0; a < K; a++) {
          const b = M.length === 1 ? 0 : a, m = M[b];
          idx[a] = m.mean(X, a * d) + beta[b] * Math.sqrt(m.quad(X, a * d));
        }
      }, (m, X, o, r) => m.add(X, o, r));
    },
  };

  // ------------------------------------------------------------- LinTS -----
  const lints = {
    id: 'lin-ts', name: 'LinTS', family: 'linear',
    params: [{ key: 'v', label: 'v (posterior scale × σ)', default: 1, min: 0.05, max: 5, step: 0.05 }, P.lambda],
    formula: 'θ̃ ~ N(θ̂, v²σ²V⁻¹);  argmaxₐ θ̃·xₐ',
    regret: 'Õ(d^{3/2}√T)  (Õ(d√T) Bayesian)',
    idea: 'Agrawal & Goyal\'s linear Thompson sampling: draw one θ from a Gaussian around the ridge estimate, shaped by the design matrix, and act greedily on the draw.',
    create(K, ctx) {
      const { v, lambda } = ctx.params, R = sigmaOf(ctx), rng = ctx.rng, d = space(K, ctx).d, scale = v * R;
      const L = new Float64Array(d * d), z = new Float64Array(d), jit = new Float64Array(d * d);
      const sample = new Float64Array(d);
      return indexPolicy(K, ctx, () => Ridge(d, lambda), (t, X, M, idx) => {
        for (let b = 0; b < M.length; b++) {
          const m = M[b];
          if (!cholesky(m.Vinv, d, L)) {
            jit.set(m.Vinv);
            for (let i = 0; i < d; i++) jit[i * d + i] += 1e-9 + 1e-6 * Math.abs(jit[i * d + i]);
            if (!cholesky(jit, d, L)) { L.fill(0); for (let i = 0; i < d; i++) L[i * d + i] = Math.sqrt(Math.max(0, m.Vinv[i * d + i])); }
          }
          for (let i = 0; i < d; i++) z[i] = rng.normal();
          for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j <= i; j++) s += L[i * d + j] * z[j]; sample[i] = m.theta[i] + scale * s; }
          if (M.length === 1) { for (let a = 0; a < K; a++) idx[a] = dotRow(sample, d, X, a * d); }
          else idx[b] = dotRow(sample, d, X, b * d);
        }
      }, (m, X, o, r) => m.add(X, o, r));
    },
  };

  // ------------------------------------------------------ LinRel family -----
  /** LinRel block (Auer 2002): eigen-decompose ZᵀZ, regress on directions with eigenvalue ≥ 1. */
  function LinRelBlock(d) {
    const G = new Float64Array(d * d), c = new Float64Array(d), vals = new Float64Array(d), vecs = new Float64Array(d * d);
    const proj = new Float64Array(d);
    let n = 0, eigN = -1, projN = -1;
    for (let k = 0; k < d; k++) vecs[k * d + k] = 1;
    function refresh() {
      if (eigN < 0 || n - eigN >= Math.max(1, Math.round((n * d) / 160))) { eigSym(G, d, vals, vecs); eigN = n; projN = -1; }
      if (projN !== n) {
        for (let k = 0; k < d; k++) {
          if (vals[k] < 1) { proj[k] = 0; continue; }
          let s = 0;
          for (let i = 0; i < d; i++) s += vecs[i * d + k] * c[i];
          proj[k] = s / vals[k];
        }
        projN = n;
      }
    }
    return {
      /** out = [mean, ‖a‖, ‖v‖] for the row at o. */
      est(X, o, out) {
        refresh();
        let m = 0, a2 = 0, v2 = 0;
        for (let k = 0; k < d; k++) {
          let ck = 0;
          for (let i = 0; i < d; i++) ck += vecs[i * d + k] * X[o + i];
          if (vals[k] >= 1) { m += ck * proj[k]; a2 += (ck * ck) / vals[k]; } else v2 += ck * ck;
        }
        out[0] = m; out[1] = Math.sqrt(a2); out[2] = Math.sqrt(v2);
        return out;
      },
      add(X, o, r) {
        for (let i = 0; i < d; i++) { c[i] += r * X[o + i]; for (let j = 0; j < d; j++) G[i * d + j] += X[o + i] * X[o + j]; }
        n++;
      },
    };
  }
  /** Ridge block exposed with the same est() interface (BaseLinUCB of SupLinUCB). */
  function RidgeEstBlock(d) {
    const m = Ridge(d, 1);
    return {
      est(X, o, out) { out[0] = m.mean(X, o); out[1] = Math.sqrt(m.quad(X, o)); out[2] = 0; return out; },
      add(X, o, r) { m.add(X, o, r); },
    };
  }

  const linrel = {
    id: 'linrel', name: 'LinRel', family: 'linear',
    params: [P.delta, { key: 'S', label: 'S (weight on ‖ṽ‖)', default: 1, min: 0, max: 10, step: 0.1 }],
    formula: 'argmaxₐ θ̂_rel·xₐ + σ√(2 ln(2TK/δ))·‖aₓ‖ + S‖ṽₓ‖  (eigen-truncated ZᵀZ)',
    regret: 'Õ(√(Td))  (via SupLinRel)',
    idea: 'Auer\'s associative reinforcement learning: eigen-decompose the past design ZᵀZ, regress only along directions with eigenvalue ≥ 1, and add an optimism term for both the regression error and the unexplored remainder ṽ.',
    create(K, ctx) {
      const { delta, S } = ctx.params, R = sigmaOf(ctx), T = horizon(ctx), d = space(K, ctx).d;
      const alpha = R * Math.sqrt(2 * Math.log((2 * T * K) / delta)), out = new Float64Array(3);
      return indexPolicy(K, ctx, () => LinRelBlock(d), (t, X, M, idx) => {
        for (let a = 0; a < K; a++) {
          M[M.length === 1 ? 0 : a].est(X, a * d, out);
          idx[a] = out[0] + alpha * out[1] + S * out[2];
        }
      }, (m, X, o, r) => m.add(X, o, r));
    },
  };

  /** Master algorithm of SupLinRel / SupLinUCB: layered elimination over ⌊ln T⌋ stages. */
  function supLin(K, ctx, makeBlock, Sv) {
    const sp = space(K, ctx), d = sp.d, rng = ctx.rng, R = sigmaOf(ctx), T = horizon(ctx);
    const alpha = R * Math.sqrt(2 * Math.log((2 * T * K) / ctx.params.delta));
    const stagesN = Math.max(1, Math.floor(Math.log(T))), floor = 1 / Math.sqrt(T);
    const m = new Float64Array(K), w = new Float64Array(K), sc = new Float64Array(K), out = new Float64Array(3);
    let stages = null, X = null, chosenStage = -1;
    const pick = (act, val) => {
      sc.fill(-Infinity);
      for (const a of act) sc[a] = val(a);
      return argmax(sc, rng);
    };
    return {
      select(t, Xt) {
        X = sp.view(Xt);
        if (!stages) { sp.init(X); stages = Array.from({ length: stagesN }, () => sp.blocks(() => makeBlock(d))); }
        let act = Array.from({ length: K }, (_, i) => i);
        for (let s = 0; ; s++) {
          if (s >= stagesN) { chosenStage = -1; return pick(act, (a) => m[a] + w[a]); }
          const thr = Math.pow(2, -(s + 1));
          let allFloor = true, allThr = true;
          for (const a of act) {
            stages[s][sp.of[a]].est(X, a * d, out);
            m[a] = out[0]; w[a] = alpha * out[1] + Sv * out[2];
            if (w[a] > floor) allFloor = false;
            if (w[a] > thr) allThr = false;
          }
          if (allFloor) { chosenStage = -1; return pick(act, (a) => m[a] + w[a]); }
          if (!allThr) { chosenStage = s; return pick(act.filter((a) => w[a] > thr), (a) => w[a]); }
          let best = -Infinity;
          for (const a of act) if (m[a] + w[a] > best) best = m[a] + w[a];
          act = act.filter((a) => m[a] + w[a] >= best - 2 * thr);
        }
      },
      update(a, r) { if (stages && chosenStage >= 0) stages[chosenStage][sp.of[a]].add(X, a * d, r); },
    };
  }

  const suplinrel = {
    id: 'suplinrel', name: 'SupLinRel', family: 'linear',
    params: [P.delta, { key: 'S', label: 'S (weight on ‖ṽ‖)', default: 1, min: 0, max: 10, step: 0.1 }],
    formula: 'stages s = 1…⌊ln T⌋: explore any arm with width > 2⁻ˢ, else keep arms within 2·2⁻ˢ of the best and go deeper',
    regret: 'O(√(Td) · ln³ᐟ²(KT ln T/δ))',
    idea: 'Auer\'s master algorithm over LinRel: each stage only learns from the rounds it chose, which keeps rewards independent for the analysis. Provably near-optimal, but slow to exploit in practice.',
    create(K, ctx) { return supLin(K, ctx, LinRelBlock, ctx.params.S); },
  };

  const suplinucb = {
    id: 'suplinucb', name: 'SupLinUCB', family: 'linear',
    params: [P.delta],
    formula: 'as SupLinRel with BaseLinUCB widths α‖x‖_{Aₛ⁻¹}, α = σ√(2 ln(2TK/δ))',
    regret: 'O(√(Td ln³(KT ln T/δ)))',
    idea: 'Chu, Li, Reyzin & Schapire: SupLinRel\'s layered elimination with ridge-regression (LinUCB) widths instead of an eigen-decomposition. The first √(dT) regret bound for LinUCB-style methods; conservative in practice.',
    create(K, ctx) { return supLin(K, ctx, RidgeEstBlock, 0); },
  };

  // ------------------------------------------------------- Hybrid LinUCB -----
  const hybrid = {
    id: 'hybrid-linucb', name: 'Hybrid LinUCB', family: 'linear',
    params: [{ key: 'alpha', label: 'α (exploration × σ)', default: 1, min: 0, max: 5, step: 0.05 }],
    formula: 'r = zᵀβ + xᵀθₐ;  argmaxₐ zᵀβ̂ + xᵀθ̂ₐ + α√sₜ,ₐ   (Li et al. Alg. 2, z = x = row a)',
    regret: 'Õ(√(KdT))  (no tight bound published)',
    idea: 'Li, Chu, Langford & Schapire\'s hybrid model: a shared coefficient vector β pooled across arms plus an arm-specific θₐ. Pools data when arms are alike yet still fits per-arm effects.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, k = d, rng = ctx.rng, alpha = ctx.params.alpha * sigmaOf(ctx);
      const A0 = new Float64Array(k * k), A0inv = new Float64Array(k * k), b0 = new Float64Array(k), beta = new Float64Array(k);
      const Ainv = new Float64Array(K * d * d), B = new Float64Array(K * d * k), ba = new Float64Array(K * d);
      const Mq = new Float64Array(K * k * k), nv = new Float64Array(K * k), work = new Float64Array(k * k), work2 = new Float64Array(k * k);
      const u = new Float64Array(d), w = new Float64Array(k), g = new Float64Array(k), zw = new Float64Array(k);
      const idx = new Float64Array(K), terms = new Float64Array(4 * k), smu = new Float64Array(k);
      let sinceInv = 0;
      for (let i = 0; i < k; i++) { A0[i * k + i] = 1; A0inv[i * k + i] = 1; }
      for (let a = 0; a < K; a++) for (let i = 0; i < d; i++) Ainv[a * d * d + i * d + i] = 1;
      let X = null;
      const uOf = (a, o) => {  // u = A_a⁻¹x, w = B_aᵀu, returns xᵀu
        const oA = a * d * d, oB = a * d * k;
        let q = 0;
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += Ainv[oA + i * d + j] * X[o + j]; u[i] = s; q += X[o + i] * s; }
        for (let j = 0; j < k; j++) { let s = 0; for (let i = 0; i < d; i++) s += B[oB + i * k + j] * u[i]; w[j] = s; }
        return q;
      };
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          for (let i = 0; i < k; i++) { let s = 0; for (let j = 0; j < k; j++) s += A0inv[i * k + j] * b0[j]; beta[i] = s; }
          for (let a = 0; a < K; a++) {
            const o = a * d;
            const q = uOf(a, o);
            // xᵀθ̂ₐ = xᵀAₐ⁻¹(bₐ − Bₐβ̂) = uᵀbₐ − wᵀβ̂
            let mean = 0;
            for (let i = 0; i < d; i++) mean += u[i] * ba[a * d + i];
            for (let j = 0; j < k; j++) { mean += (X[o + j] - w[j]) * beta[j]; zw[j] = X[o + j] - w[j]; }
            // s = (z − w)ᵀA₀⁻¹(z − w) + xᵀAₐ⁻¹x  with w = BₐᵀAₐ⁻¹x
            let s2 = q;
            for (let i = 0; i < k; i++) { let s = 0; for (let j = 0; j < k; j++) s += A0inv[i * k + j] * zw[j]; s2 += zw[i] * s; }
            idx[a] = mean + alpha * Math.sqrt(Math.max(0, s2));
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          if (!X) return;
          const o = a * d, oA = a * d * d, oB = a * d * k, oM = a * k * k;
          const q = uOf(a, o), c = 1 + q;
          let ub = 0;
          for (let i = 0; i < d; i++) ub += u[i] * ba[a * d + i];
          for (let j = 0; j < k; j++) g[j] = w[j] + X[o + j] * q;
          // Mₐ = BₐᵀAₐ⁻¹Bₐ and nₐ = BₐᵀAₐ⁻¹bₐ after the rank-one update, in O(k² + dk)
          for (let i = 0; i < k; i++) {
            const zi = X[o + i];
            for (let j = 0; j < k; j++) {
              const zj = X[o + j];
              const mNew = Mq[oM + i * k + j] + w[i] * zj + zi * w[j] + q * zi * zj - (g[i] * g[j]) / c;
              A0[i * k + j] += Mq[oM + i * k + j] - mNew + zi * zj;
              Mq[oM + i * k + j] = mNew;
            }
            const nNew = nv[a * k + i] + zi * ub + r * g[i] - (g[i] * (ub + r * q)) / c;
            b0[i] += nv[a * k + i] - nNew + r * zi;
            nv[a * k + i] = nNew;
          }
          for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Ainv[oA + i * d + j] -= (u[i] * u[j]) / c;
          for (let i = 0; i < d; i++) { for (let j = 0; j < k; j++) B[oB + i * k + j] += X[o + i] * X[o + j]; ba[a * d + i] += r * X[o + i]; }
          for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) A0[i * k + j] = A0[j * k + i] = 0.5 * (A0[i * k + j] + A0[j * k + i]);
          // A₀ changed by (1−q)zzᵀ − zwᵀ − wzᵀ + ggᵀ/c (rank ≤ 3): apply it to A₀⁻¹ as Sherman–Morrison
          // steps (positive weights first) and refresh exactly every 50 updates or if a step is rejected.
          let ok = ++sinceInv < 50;
          if (ok) {
            for (let j = 0; j < k; j++) { const z = X[o + j]; terms[j] = z; terms[k + j] = z - w[j]; terms[2 * k + j] = z + w[j]; terms[3 * k + j] = g[j]; }
            const wt = [1 - q, 0.5, -0.5, 1 / c];
            for (const pass of [1, -1]) for (let l = 0; l < 4 && ok; l++) if (wt[l] * pass > 0 && !shermanMorrison(A0inv, k, terms, l * k, wt[l], smu)) ok = false;
          }
          if (!ok) { invertSPD(A0, k, A0inv, work, work2); sinceInv = 0; }
        },
      };
    },
  };

  // ------------------------------------------- G-optimal phased elimination -----
  const gopt = {
    id: 'g-optimal', name: 'Phased elimination (G-optimal)', family: 'linear',
    params: [P.delta, { key: 'c', label: 'c (phase-length scale; theory 1)', default: 0.25, min: 0.05, max: 5, step: 0.05 }],
    formula: 'phase ℓ: εₗ = 2⁻ˡ, π = G-optimal design on Aₗ, pull a ⌈c·2d·π(a)σ²/εₗ² · ln(|Aₗ|ℓ(ℓ+1)/δ)⌉ times, drop a if maxᵦ θ̂·(xᵦ − xₐ) > 2εₗ',
    regret: 'O(√(dT ln(KT)))',
    idea: 'Lattimore & Szepesvári (ch. 22): in each phase spread pulls by a Kiefer–Wolfowitz (G-optimal) design, found by Frank–Wolfe, so least squares is equally accurate for every surviving arm, then eliminate. Built for fixed arm sets; with changing contexts it eliminates arm indices.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, R = sigmaOf(ctx), { delta, c } = ctx.params;
      const rem = new Float64Array(K), mhat = new Float64Array(K);
      let active = null, phase = 0, ptr = 0, X = null, G = null, cv = null, left = 0;
      const tmp = new Float64Array(d * d), inv = new Float64Array(d * d), th = new Float64Array(d);
      function startPhase() {
        phase++;
        const eps = Math.pow(2, -phase);
        const { pi, deff } = gOptimalDesign(active, X, sp);
        const logt = Math.log((active.length * phase * (phase + 1)) / delta);
        left = 0;
        rem.fill(0);
        for (const a of active) {
          if (pi[a] <= 1e-6) continue;
          rem[a] = Math.ceil((c * 2 * deff * pi[a] * R * R * logt) / (eps * eps));
          left += rem[a];
        }
        if (left === 0) for (const a of active) { rem[a] = 1; left++; }
        G.fill(0); cv.fill(0);
      }
      function endPhase() {
        const eps = Math.pow(2, -phase);
        for (let b = 0; b < sp.B; b++) {
          const o = b * d * d;
          let tr = 0;
          for (let i = 0; i < d; i++) tr += G[o + i * d + i];
          for (let i = 0; i < d * d; i++) tmp[i] = G[o + i];
          for (let i = 0; i < d; i++) tmp[i * d + i] += 1e-6 * (tr / d) + 1e-9;
          invert(tmp, d, inv);
          for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += inv[i * d + j] * cv[b * d + j]; th[i] = s; }
          for (const a of active) if (sp.of[a] === b) mhat[a] = dotRow(th, d, X, a * d);
        }
        let best = -Infinity;
        for (const a of active) if (mhat[a] > best) best = mhat[a];
        active = active.filter((a) => mhat[a] >= best - 2 * eps);
      }
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!active) {
            sp.init(X);
            active = Array.from({ length: K }, (_, i) => i);
            G = new Float64Array(sp.B * d * d); cv = new Float64Array(sp.B * d);
            startPhase();
          }
          if (left <= 0) { endPhase(); startPhase(); }
          if (active.length === 1) return active[0];
          for (let s = 0; s < K; s++) {
            const a = (ptr + s) % K;
            if (rem[a] > 0) { rem[a]--; left--; ptr = (a + 1) % K; return a; }
          }
          return active[0];
        },
        update(a, r) {
          if (!G) return;
          const o = sp.of[a] * d * d, ob = sp.of[a] * d, row = a * d;
          for (let i = 0; i < d; i++) { cv[ob + i] += r * X[row + i]; for (let j = 0; j < d; j++) G[o + i * d + j] += X[row + i] * X[row + j]; }
        },
      };
    },
  };

  // ---------------------------------------------------------- SpectralUCB -----
  const spectral = {
    id: 'spectral-ucb', name: 'SpectralUCB', family: 'linear',
    params: [
      { key: 'lambda', label: 'λ (ridge)', default: 1, min: 0.001, max: 10, step: 0.001 },
      P.delta,
      { key: 'C', label: 'C (bound on ‖α‖_Λ)', default: 1, min: 0, max: 10, step: 0.1 },
    ],
    formula: 'argmaxₐ α̂·xₐ + cₜ‖xₐ‖_{Vₜ⁻¹},  Vₜ = Λ_L + λI + Σxxᵀ,  cₜ = σ√(2 ln(1/δ) + ln(|Vₜ|/|Λ|)) + C  ≤ 2σ√(d_eff ln(1+t/λ) + 2 ln(1/δ)) + C',
    regret: 'Õ(d_eff √T),  d_eff ≪ number of nodes',
    idea: 'Valko, Munos, Kveton & Kocák: arms are graph nodes and the reward is smooth on the graph, so regress on Laplacian eigenvectors with a prior Λ that penalises each by its eigenvalue. Regret scales with the small effective dimension d_eff. Uses the data-dependent radius that the paper\'s closed form upper-bounds; without a graph it is OFUL with ridge λ.',
    create(K, ctx) {
      const { lambda, delta, C } = ctx.params, R = sigmaOf(ctx), T = horizon(ctx), d = space(K, ctx).d;
      const info = ctx.info && ctx.info.spectral, prior = new Float64Array(d);
      const ev = info && info.d === d && info.eigenvalues ? info.eigenvalues : null;
      let maxEv = 0;
      if (ev) for (let i = 0; i < Math.min(d, ev.length); i++) maxEv = Math.max(maxEv, ev[i]);
      for (let i = 0; i < d; i++) prior[i] = lambda + (ev ? (i < ev.length && i < (info.m || d) ? ev[i] : maxEv) : 0);
      // d_eff = max{d′: (d′−1)λ_{d′} ≤ T/ln(1+T/λ)} caps the closed-form radius, which bounds ours
      const thr = T / Math.log(1 + T / lambda);
      let deff = 1;
      for (let j = 1; j <= d; j++) if ((j - 1) * prior[j - 1] <= thr) deff = j;
      return indexPolicy(K, ctx, () => Ridge(d, prior), (t, X, M, idx) => {
        for (let a = 0; a < K; a++) {
          const m = M[M.length === 1 ? 0 : a];
          const ct = Math.min(R * Math.sqrt(2 * Math.log(1 / delta) + Math.max(0, m.logdet)),
            2 * R * Math.sqrt(deff * Math.log(1 + t / lambda) + 2 * Math.log(1 / delta))) + C;
          idx[a] = m.mean(X, a * d) + ct * Math.sqrt(m.quad(X, a * d));
        }
      }, (m, X, o, r) => m.add(X, o, r));
    },
  };

  // ---------------------------------------------------------------- CLUCB -----
  const clucb = {
    id: 'clucb', name: 'CLUCB (conservative)', family: 'linear',
    params: [
      { key: 'alpha', label: 'α (allowed shortfall vs baseline)', default: 0.1, min: 0.01, max: 1, step: 0.01 },
      { key: 'base', label: 'baseline arm', default: 0, min: 0, max: 49, step: 1 },
      P.lambda, P.delta, P.S,
    ],
    formula: 'a′ = OFUL arm; play a′ iff min_{θ∈Cₜ} θ·(Σ_{S}x + x_{a′} + Σ_{Sᶜ}x_b − (1−α)Σ_{s≤t}x_b) ≥ 0, else baseline b',
    regret: 'Õ(d√T) + O(K/(α·r_b)) cost of conservatism',
    idea: 'Kazerouni, Ghavamzadeh, Abbasi-Yadkori & Van Roy: run OFUL but only take its arm if, even pessimistically, cumulative reward stays above (1 − α) times what always playing the default arm would earn. The baseline\'s reward is not known here, so it is bounded through θ as well (CLUCB2-style); Gaussian rewards are counted from a floor of −2σ so the budget stays positive.',
    create(K, ctx) {
      const { alpha, lambda, delta, S } = ctx.params, R = sigmaOf(ctx), rng = ctx.rng;
      const sp = space(K, ctx), d = sp.d, base = Math.min(K - 1, Math.max(0, Math.round(ctx.params.base)));
      const idx = new Float64Array(K);
      // Kazerouni et al. assume rewards ≥ 0, so α·(baseline reward) is a positive budget. Gaussian
      // rewards here can be negative, so they are measured from a floor of −2σ (adds α·2σ·t).
      const floor = isLogistic(ctx) ? 0 : 2 * R;
      let M = null, X = null, sumS, sumBaseNotS, sumBaseAll, w, rounds = 0;
      const lcb = () => {
        let mean = 0, q = 0, beta = 0;
        for (let b = 0; b < sp.B; b++) {
          const m = M[b];
          mean += dotRow(m.theta, d, w, b * d);
          q += quadForm(m.Vinv, d, w, b * d);
          beta = Math.max(beta, ofulBeta(m, R, lambda, delta, S));
        }
        return mean - beta * Math.sqrt(q);
      };
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!M) {
            sp.init(X); M = sp.blocks(() => Ridge(d, lambda));
            sumS = new Float64Array(sp.B * d); sumBaseNotS = new Float64Array(sp.B * d); sumBaseAll = new Float64Array(sp.B * d); w = new Float64Array(sp.B * d);
          }
          const bo = sp.of[base] * d;
          for (let j = 0; j < d; j++) sumBaseAll[bo + j] += X[base * d + j];
          rounds++;
          for (let a = 0; a < K; a++) {
            const m = M[sp.of[a]];
            idx[a] = m.mean(X, a * d) + ofulBeta(m, R, lambda, delta, S) * Math.sqrt(m.quad(X, a * d));
          }
          const opt = argmax(idx, rng);
          if (opt === base) return base;
          for (let i = 0; i < w.length; i++) w[i] = sumS[i] + sumBaseNotS[i] - (1 - alpha) * sumBaseAll[i];
          const oo = sp.of[opt] * d;
          for (let j = 0; j < d; j++) w[oo + j] += X[opt * d + j];
          return lcb() + alpha * floor * rounds >= 0 ? opt : base;
        },
        update(a, r) {
          if (!M) return;
          const o = sp.of[a] * d, target = a === base ? sumBaseNotS : sumS;
          for (let j = 0; j < d; j++) target[o + j] += X[a * d + j];
          M[sp.of[a]].add(X, a * d, r);
        },
      };
    },
  };

  // ============================================================ GLM bandits =====
  const glmParams = [
    { key: 'c', label: 'c (confidence scale)', default: 1, min: 0, max: 5, step: 0.05 },
    P.lambda,
  ];
  /** Index policy over GLM blocks: score(m, X, o, t) gives arm a's index. */
  function glmPolicy(K, ctx, lam, score) {
    const logistic = isLogistic(ctx), d = space(K, ctx).d;
    return indexPolicy(K, ctx, () => GlmBlock(d, lam, logistic), (t, X, M, idx) => {
      for (let a = 0; a < K; a++) idx[a] = score(M[M.length === 1 ? 0 : a], X, a * d, t);
    }, (m, X, o, r) => m.push(X, o, r));
  }

  const glmucb = {
    id: 'glm-ucb', name: 'GLM-UCB', family: 'glm',
    params: glmParams,
    formula: 'argmaxₐ μ(xₐᵀθ̂) + ρ(t)‖xₐ‖_{V⁻¹},  ρ(t) = c·k_μ(σ/c_μ)·√(2 ln t),  θ̂ = regularised MLE',
    regret: 'Õ((k_μ/c_μ) d√T)',
    idea: 'Filippi, Cappé, Garivier & Szepesvári: fit θ by maximum likelihood under the link μ (logistic for 0/1 rewards, identity otherwise) and add a bonus from the design matrix inflated by 1/c_μ, the flattest slope of the link. Simplified: the theory\'s ρ(t) is replaced by a scaled √(2 ln t).',
    create(K, ctx) {
      const { c, lambda } = ctx.params, logistic = isLogistic(ctx);
      // k_μ = sup μ̇ (Lipschitz constant), c_μ = slope floor; for the logistic link k_μ = c_μ = 1/4 here
      const s = sigmaOf(ctx);
      return glmPolicy(K, ctx, lambda, (m, X, o, t) => {
        const rho = c * s * Math.sqrt(2 * Math.log(Math.max(2, t)));
        return m.mu(dotRow(m.theta, m.d, X, o)) + rho * Math.sqrt(quadForm(m.V.Vinv, m.d, X, o));
      });
    },
  };

  const ucbglm = {
    id: 'ucb-glm', name: 'UCB-GLM', family: 'glm',
    params: [
      P.delta,
      { key: 'kappa', label: 'κ (min slope of μ, logistic only)', default: 0.25, min: 0.01, max: 1, step: 0.01 },
      { key: 'lambda', label: 'λ (MLE stabiliser)', default: 0.1, min: 0.001, max: 10, step: 0.001 },
    ],
    formula: 'τ random rounds, then argmaxₐ xₐᵀθ̂ + αₜ‖xₐ‖_{V⁻¹},  αₜ = (σ/κ)√(d/2·ln(1+2t/d) + ln(1/δ))',
    regret: 'Õ(d√T)  (optimal up to logs)',
    idea: 'Li, Lu & Zhou: a short burst of random exploration makes the design well conditioned, then the plain MLE is accurate enough that a LinUCB-style bonus on the linear predictor suffices. Here the MLE carries a tiny ridge term for numerical stability.',
    create(K, ctx) {
      const { delta, kappa, lambda } = ctx.params, logistic = isLogistic(ctx), rng = ctx.rng, T = horizon(ctx);
      const s = sigmaOf(ctx) / (logistic ? kappa : 1);
      const pol = glmPolicy(K, ctx, lambda, (m, X, o, t) => {
        const dd = m.d, at = s * Math.sqrt((dd / 2) * Math.log(1 + (2 * t) / dd) + Math.log(1 / delta));
        return dotRow(m.theta, dd, X, o) + at * Math.sqrt(quadForm(m.V.Vinv, dd, X, o));
      });
      let tau = -1;
      return {
        select(t, Xt) {
          const a = pol.select(t, Xt);
          if (tau < 0) tau = Math.min(Math.floor(T / 4), 2 * pol.sp.B * pol.sp.d);
          return t <= tau ? rng.int(K) : a;
        },
        update: pol.update,
      };
    },
  };

  /** γₜ(δ) of Faury et al. (2020), with 1/2-sub-Gaussian (Bernoulli) noise. */
  function faurgamma(t, d, lambda, S, delta) {
    return Math.sqrt(lambda) * (S + 0.5) + (2 / Math.sqrt(lambda)) * (Math.log(1 / delta) + d * Math.LN2 + (d / 2) * Math.log(1 + t / (4 * d * lambda)));
  }
  const logParams = (c) => [
    { key: 'c', label: 'c (confidence scale; theory 1)', default: c, min: 0.001, max: 2, step: 0.001 },
    P.lambda, P.delta, P.S,
  ];
  const logucb1 = {
    id: 'logistic-ucb1', name: 'Logistic-UCB-1', family: 'glm',
    params: logParams(0.03),
    formula: 'argmaxₐ μ(xₐᵀθ̂) + c·L_μ√(4+8S)·γₜ‖xₐ‖_{Hₜ⁻¹(θ̂)},  Hₜ(θ) = λI + Σ μ̇(xᵀθ)xxᵀ',
    regret: 'Õ(d√(T/κ_*))  — no κ in the leading term',
    idea: 'Faury, Abeille, Calauzènes & Fercoq: measure uncertainty with the local Hessian Hₜ(θ̂) instead of the design matrix divided by κ, removing the exponential dependence on ‖θ‖ from the leading regret term. Simplified: no projection step, and γₜ is multiplied by c.',
    create(K, ctx) {
      const { c, lambda, delta, S } = ctx.params, logistic = isLogistic(ctx);
      const Lmu = logistic ? 0.25 : 2 * sigmaOf(ctx), f = Math.sqrt(4 + 8 * S);
      return glmPolicy(K, ctx, lambda, (m, X, o, t) => {
        const g = c * faurgamma(t, m.d, lambda, S, delta);
        return m.mu(dotRow(m.theta, m.d, X, o)) + Lmu * f * g * Math.sqrt(quadForm(m.Hinv, m.d, X, o));
      });
    },
  };
  const logucb2 = {
    id: 'logistic-ucb2', name: 'Logistic-UCB-2', family: 'glm',
    params: logParams(0.02),
    formula: 'argmaxₐ μ(z) + c·[2√(1+2S)·μ̇(z)·γₜ‖xₐ‖_{Hₜ⁻¹} + (4+8S)M_μκ·cγₜ²‖xₐ‖²_{V⁻¹}],  z = xₐᵀθ̂',
    regret: 'Õ(d√(T·μ̇(x*ᵀθ*)) + d²κ)',
    idea: 'Faury et al.\'s refined bonus: first-order uncertainty is scaled by the local slope μ̇ at the arm, so saturated arms are barely explored, plus a second-order term that fades like 1/t. Simplified: no projection step, κ = 1/μ̇(S), γₜ multiplied by c.',
    create(K, ctx) {
      const { c, lambda, delta, S } = ctx.params, logistic = isLogistic(ctx);
      const kappa = logistic ? 1 / (sigmoid(S) * (1 - sigmoid(S))) : 1, Mmu = logistic ? 0.25 : 0, scale = logistic ? 1 : 2 * sigmaOf(ctx);
      return glmPolicy(K, ctx, lambda, (m, X, o, t) => {
        const g = c * faurgamma(t, m.d, lambda, S, delta), z = dotRow(m.theta, m.d, X, o);
        return m.mu(z) + scale * (2 * Math.sqrt(1 + 2 * S) * m.dmu(z) * g * Math.sqrt(quadForm(m.Hinv, m.d, X, o))
          + (4 + 8 * S) * Mmu * kappa * g * g * quadForm(m.V.Vinv, m.d, X, o));
      });
    },
  };

  const ofuglb = {
    id: 'ofuglb', name: 'OFUGLB', family: 'glm',
    params: [P.delta, { key: 'S', label: 'S (bound on ‖θ‖)', default: 2, min: 0.1, max: 10, step: 0.1 }, { key: 'lambda', label: 'λ (MLE stabiliser)', default: 0.1, min: 0.001, max: 10, step: 0.001 }],
    formula: 'argmaxₐ max_{θ∈Cₜ} xₐᵀθ,  Cₜ = {θ: ℒₜ(θ) − ℒₜ(θ̂) ≤ βₜ²},  βₜ² = ln(1/δ) + d·ln(max(e, 2eSt/d))',
    regret: 'Õ(d√(T/κ_*))',
    idea: 'Lee, Yun & Jun: a likelihood-ratio confidence sequence valid for any GLM, with a radius that has no κ or exponential S dependence. Simplified: the likelihood set is replaced by its Laplace ellipsoid ½‖θ − θ̂‖²_{H(θ̂)} ≤ βₜ², so the optimistic value is xᵀθ̂ + √2·βₜ‖x‖_{H⁻¹}.',
    create(K, ctx) {
      const { delta, S, lambda } = ctx.params, logistic = isLogistic(ctx), s = logistic ? 1 : sigmaOf(ctx);
      return glmPolicy(K, ctx, lambda, (m, X, o, t) => {
        const dd = m.d, beta2 = Math.log(1 / delta) + dd * Math.log(Math.max(Math.E, (2 * Math.E * S * t) / dd));
        return dotRow(m.theta, dd, X, o) + Math.SQRT2 * Math.sqrt(beta2) * s * Math.sqrt(quadForm(m.Hinv, dd, X, o));
      });
    },
  };

  // ======================================================= sparse & low-rank =====
  const lassoBandit = {
    id: 'lasso-bandit', name: 'Lasso Bandit', family: 'sparse',
    params: [
      { key: 'q', label: 'q (forced pulls per arm per batch)', default: 1, min: 1, max: 20, step: 1 },
      { key: 'h', label: 'h (gap for the forced filter)', default: 1, min: 0, max: 10, step: 0.05 },
      { key: 'lambda1', label: 'λ₁ (forced-sample Lasso)', default: 0.2, min: 0, max: 2, step: 0.005 },
      { key: 'lambda2', label: 'λ₂ (all-sample Lasso scale)', default: 0.3, min: 0, max: 2, step: 0.005 },
      { key: 'refit', label: 'refit every (steps)', default: 5, min: 1, max: 100, step: 1 },
    ],
    formula: 'forced pulls at t ∈ {(2ⁿ−1)Kq + j};  else Â = {a: xₐ·β̂_F ≥ max − h/2},  aₜ = argmax_{a∈Â} xₐ·β̂_A(λ₂√((ln t + ln d)/t))',
    regret: 'O(s² (ln d + ln T)²)',
    idea: 'Bastani & Bayati: two Lasso estimates, one on rare forced-exploration samples (unbiased, used to shortlist arms) and one on all samples (accurate, used to choose). Sparsity makes regret grow with log d, not d. Lasso is solved by coordinate descent on running Gram matrices and refitted every few steps.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, rng = ctx.rng, { h, lambda1, lambda2 } = ctx.params;
      const q = Math.max(1, Math.round(ctx.params.q)), every = Math.max(1, Math.round(ctx.params.refit));
      const mf = new Float64Array(K), ma = new Float64Array(K), sc = new Float64Array(K);
      let F = null, A = null, X = null, forcedArm = -1, fN = null, aN = null;
      const forcedAt = (t) => {
        for (let n = 0; ; n++) {
          const start = (Math.pow(2, n) - 1) * K * q;
          if (start >= t) return -1;
          if (t - start <= K * q) return Math.floor((t - start - 1) / q);
        }
      };
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!F) { sp.init(X); F = sp.blocks(() => GramBlock(d)); A = sp.blocks(() => GramBlock(d)); fN = new Int32Array(sp.B).fill(-1); aN = new Int32Array(sp.B).fill(-1); }
          forcedArm = forcedAt(t);
          if (forcedArm >= 0) return forcedArm;
          for (let b = 0; b < sp.B; b++) {
            if (F[b].n !== fN[b]) { lasso(F[b].G, F[b].c, F[b].n, d, lambda1, F[b].beta); fN[b] = F[b].n; }
            const n = A[b].n;
            if (aN[b] < 0 || n - aN[b] >= every) {
              lasso(A[b].G, A[b].c, n, d, lambda2 * Math.sqrt((Math.log(Math.max(2, n)) + Math.log(Math.max(2, d))) / Math.max(1, n)), A[b].beta);
              aN[b] = n;
            }
          }
          let best = -Infinity;
          for (let a = 0; a < K; a++) { mf[a] = dotRow(F[sp.of[a]].beta, d, X, a * d); if (mf[a] > best) best = mf[a]; }
          for (let a = 0; a < K; a++) { ma[a] = dotRow(A[sp.of[a]].beta, d, X, a * d); sc[a] = mf[a] >= best - h / 2 ? ma[a] : -Infinity; }
          return argmax(sc, rng);
        },
        update(a, r) {
          if (!F) return;
          if (forcedArm >= 0) F[sp.of[a]].add(X, a * d, r);
          A[sp.of[a]].add(X, a * d, r);
        },
      };
    },
  };

  const drLasso = {
    id: 'dr-lasso', name: 'DR-Lasso Bandit', family: 'sparse',
    params: [
      { key: 'lambda1', label: 'λ₁ (exploration scale)', default: 2, min: 0, max: 5, step: 0.05 },
      { key: 'lambda2', label: 'λ₂ (Lasso scale)', default: 0.5, min: 0, max: 2, step: 0.005 },
      { key: 'z', label: 'z (initial random rounds)', default: 10, min: 1, max: 200, step: 1 },
      { key: 'refit', label: 'refit every (steps)', default: 5, min: 1, max: 100, step: 1 },
    ],
    formula: 'r̂ₜ = x̄ₜᵀβ̂ + (rₜ − x_{aₜ}ᵀβ̂)/(Kπ_{aₜ});  β̂ = Lasso({x̄, r̂}, λ₂√((ln t + 2 ln d)/t))',
    regret: 'O(s √(T ln(dT)))',
    idea: 'Kim & Paik: mostly greedy on a Lasso estimate, with exploration probability λ₁√((ln t + ln d)/t). A doubly-robust pseudo-reward imputes every arm\'s outcome from the one observed, so the Lasso is fitted on the average context x̄ₜ and uses every round (needs arms\' features to be correlated, as in the sparse env). Inverse-propensity weights 1/π are capped at 10K. With one shared context (disjoint per-arm models) x̄ would be uninformative, so each arm\'s pseudo-reward goes to its own Lasso instead.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, rng = ctx.rng, { lambda1, lambda2 } = ctx.params;
      const z = Math.max(1, Math.round(ctx.params.z)), every = Math.max(1, Math.round(ctx.params.refit));
      const sc = new Float64Array(K), pred = new Float64Array(K);
      let M = null, fitN = null, xbar = null, X = null, pChosen = 1;
      const refit = (g, b) => {
        const n = g.n;
        if (n === fitN[b] || (fitN[b] >= 0 && n - fitN[b] < every)) return;
        lasso(g.G, g.c, n, d, lambda2 * Math.sqrt((Math.log(Math.max(2, n)) + 2 * Math.log(Math.max(2, d * sp.B))) / Math.max(1, n)), g.beta);
        fitN[b] = n;
      };
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!M) { sp.init(X); M = sp.blocks(() => GramBlock(d)); fitN = new Int32Array(sp.B).fill(-1); xbar = new Float64Array(d); }
          M.forEach(refit);
          for (let a = 0; a < K; a++) sc[a] = dotRow(M[sp.of[a]].beta, d, X, a * d);
          const greedy = argmax(sc, rng);
          if (t <= z) { pChosen = 1 / K; return rng.int(K); }
          const eps = Math.min(1, lambda1 * Math.sqrt((Math.log(t) + Math.log(Math.max(2, d * sp.B))) / t));
          const a = rng.uniform() < eps ? rng.int(K) : greedy;
          pChosen = eps / K + (a === greedy ? 1 - eps : 0);
          return a;
        },
        update(a, r) {
          if (!M) return;
          for (let b = 0; b < K; b++) pred[b] = dotRow(M[sp.of[b]].beta, d, X, b * d);
          const ipw = Math.min(10 * K, 1 / pChosen);   // 1/πₐ, capped
          if (sp.shared) {
            // r̂ = (1/K)Σ_b [x_bᵀβ̂ + 1{b = a}(r − x_aᵀβ̂)/π_a],  regressed on x̄ = (1/K)Σ_b x_b
            xbar.fill(0);
            let rhat = 0;
            for (let b = 0; b < K; b++) { rhat += pred[b] / K; for (let j = 0; j < d; j++) xbar[j] += X[b * d + j] / K; }
            rhat += (ipw / K) * (r - pred[a]);
            M[0].add(xbar, 0, rhat);
          } else {
            for (let b = 0; b < K; b++) M[b].add(X, b * d, pred[b] + (b === a ? ipw * (r - pred[a]) : 0));
          }
        },
      };
    },
  };

  const estc = {
    id: 'estc', name: 'ESTC', family: 'sparse',
    params: [
      { key: 'c', label: 'c (exploration length scale)', default: 1, min: 0.1, max: 5, step: 0.05 },
      { key: 'lam', label: 'Lasso λ scale', default: 0.5, min: 0, max: 5, step: 0.05 },
    ],
    formula: 'explore n₁ = ⌈c·T^{2/3}(ln 2d)^{1/3}⌉ rounds, θ̂ = Lasso(λ = c_λσ√(2 ln(2d)/n₁)), then commit to argmaxₐ θ̂·xₐ',
    regret: 'Õ(s^{2/3} T^{2/3})  — dimension-free up to log d',
    idea: 'Hao, Lattimore & Wang: in data-poor high-dimensional problems, explore once with a well-conditioned distribution, fit a Lasso and then act greedily on it forever. Its T^{2/3} rate beats √(dT) when d is large. Simplified: exploration samples arms uniformly instead of solving the C_min-optimal design.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, rng = ctx.rng, R = sigmaOf(ctx), T = horizon(ctx), { c, lam } = ctx.params;
      const n1 = Math.min(T, Math.max(2, Math.ceil(c * Math.pow(T, 2 / 3) * Math.cbrt(Math.log(2 * d)))));
      const sc = new Float64Array(K);
      let M = null, X = null, step = 0, fitted = false;
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!M) { sp.init(X); M = sp.blocks(() => GramBlock(d)); }
          step++;
          if (step <= n1) return rng.int(K);
          if (!fitted) {
            for (const g of M) lasso(g.G, g.c, g.n, d, lam * R * Math.sqrt((2 * Math.log(2 * d)) / Math.max(1, g.n)), g.beta, 500);
            fitted = true;
          }
          for (let a = 0; a < K; a++) sc[a] = dotRow(M[sp.of[a]].beta, d, X, a * d);
          return argmax(sc, rng);
        },
        update(a, r) { if (M && !fitted) M[sp.of[a]].add(X, a * d, r); },
      };
    },
  };

  /** Nuclear-norm regularised least squares on Gram statistics (proximal gradient with SVT). */
  function nuclearLS(G, cvec, n, d1, d2, lam, iters = 300) {
    const d = d1 * d2, Th = new Float64Array(d), Y = new Float64Array(d), grad = new Float64Array(d);
    const L = lambdaMax(G, d) / Math.max(1, n);
    if (!(L > 0)) return Th;
    const step = 1 / L, YtY = new Float64Array(d2 * d2), vals = new Float64Array(d2), vecs = new Float64Array(d2 * d2), W = new Float64Array(d1 * d2);
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < d; i++) { let s = -cvec[i]; for (let j = 0; j < d; j++) s += G[i * d + j] * Th[j]; grad[i] = s / n; }
      for (let i = 0; i < d; i++) Y[i] = Th[i] - step * grad[i];
      // singular value thresholding of Y (d1×d2): Y·V·diag(max(s−τ,0)/s)·Vᵀ
      YtY.fill(0);
      for (let i = 0; i < d1; i++) for (let a = 0; a < d2; a++) for (let b = 0; b < d2; b++) YtY[a * d2 + b] += Y[i * d2 + a] * Y[i * d2 + b];
      eigSym(YtY, d2, vals, vecs);
      W.fill(0);
      for (let k = 0; k < d2; k++) {
        const s = Math.sqrt(Math.max(0, vals[k])), f = s > 1e-12 ? Math.max(0, s - step * lam) / s : 0;
        if (f === 0) continue;
        for (let i = 0; i < d1; i++) { let yv = 0; for (let j = 0; j < d2; j++) yv += Y[i * d2 + j] * vecs[j * d2 + k]; W[i * d2 + k] = yv * f; }
      }
      let change = 0;
      for (let i = 0; i < d1; i++) for (let j = 0; j < d2; j++) {
        let s = 0;
        for (let k = 0; k < d2; k++) s += W[i * d2 + k] * vecs[j * d2 + k];
        change = Math.max(change, Math.abs(s - Th[i * d2 + j]));
        Th[i * d2 + j] = s;
      }
      if (change < 1e-7) break;
    }
    return Th;
  }

  const estr = {
    id: 'estr', name: 'ESTR (LowOFUL)', family: 'sparse',
    params: [
      { key: 'r', label: 'r (assumed rank)', default: 2, min: 1, max: 10, step: 1 },
      { key: 'c', label: 'c (stage-1 length scale)', default: 0.3, min: 0.1, max: 5, step: 0.05 },
      { key: 'mu', label: 'nuclear-norm λ scale', default: 0.5, min: 0, max: 5, step: 0.05 },
      { key: 'sperp', label: 'S⊥ (bound on the irrelevant part)', default: 0.1, min: 0, max: 2, step: 0.01 },
      P.lambda, P.delta, P.S,
    ],
    formula: 'stage 1: T₁ = ⌈c√(dT)⌉ random pulls, Θ̂ = argmin ½‖y − ⟨Θ, xzᵀ⟩‖²/n + λ‖Θ‖_*;  stage 2: LowOFUL in the basis [Û Û⊥]⊗[V̂ V̂⊥], λ⊥ = T/(k ln(1+T/λ)) on the Û⊥/V̂⊥ block',
    regret: 'Õ((d₁+d₂)^{3/2}√(rT))  vs Õ(d₁d₂√T) for OFUL',
    idea: 'Jun, Willett, Wright & Nowak\'s Explore-Subspace-Then-Refine for bilinear rewards xᵀΘz with low-rank Θ: estimate the row/column spaces by nuclear-norm regression, rotate features into that basis and run OFUL with a much heavier ridge on the (d₁−r)(d₂−r) almost-irrelevant coordinates. Here x is a context, so stage 1 pulls random arms; stage-1 data is reused. Needs info.bilinear (the low-rank env); elsewhere it is plain OFUL.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, rng = ctx.rng, R = sigmaOf(ctx), T = horizon(ctx);
      const { c, mu, sperp, lambda, delta, S } = ctx.params;
      const bil = ctx.info && ctx.info.bilinear;
      const [d1, d2] = bil && bil.d1 * bil.d2 === d ? [bil.d1, bil.d2] : [1, d];
      const r = Math.max(1, Math.min(Math.round(ctx.params.r), d1, d2));
      const lowRank = r < Math.min(d1, d2);
      const k = (d1 + d2) * r - r * r;
      const T1 = lowRank ? Math.min(Math.floor(T / 2), Math.ceil(c * Math.sqrt(d * T))) : 0;
      const lamPerp = lowRank ? Math.max(lambda, (T - T1) / (k * Math.log(1 + (T - T1) / lambda))) : lambda;
      const Sperp = lowRank ? sperp : 0;
      const prior = new Float64Array(d), Xr = new Float64Array(K * d), idx = new Float64Array(K);
      for (let i = 0; i < d1; i++) for (let j = 0; j < d2; j++) prior[i * d2 + j] = i < r || j < r ? lambda : lamPerp;
      let stats = null, M = null, Q = null, X = null, step = 0;
      const rotate = (src, o, dst, od) => {   // dst = vec(Uᵀ Φ V), Φ = reshape(src row, d1×d2)
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += Q[i * d + j] * src[o + j]; dst[od + i] = s; }
      };
      function enterStage2() {
        Q = new Float64Array(d * d);
        M = sp.blocks((b) => {
          const st = stats[b], m = Ridge(d, prior);
          if (!lowRank) { m.setFrom(addDiag(st.G, prior), st.c, logdetRatio(addDiag(st.G, prior), prior), st.n); return m; }
          const Th = nuclearLS(st.G, st.c, Math.max(1, st.n), d1, d2, mu * R * Math.sqrt((d1 + d2) / Math.max(1, st.n)));
          const TT = new Float64Array(d1 * d1), TtT = new Float64Array(d2 * d2), uv = new Float64Array(d1), U = new Float64Array(d1 * d1), vv = new Float64Array(d2), V = new Float64Array(d2 * d2);
          for (let i = 0; i < d1; i++) for (let j = 0; j < d1; j++) { let s = 0; for (let l = 0; l < d2; l++) s += Th[i * d2 + l] * Th[j * d2 + l]; TT[i * d1 + j] = s; }
          for (let i = 0; i < d2; i++) for (let j = 0; j < d2; j++) { let s = 0; for (let l = 0; l < d1; l++) s += Th[l * d2 + i] * Th[l * d2 + j]; TtT[i * d2 + j] = s; }
          eigSym(TT, d1, uv, U); eigSym(TtT, d2, vv, V);
          // Q[(i,j),(k,l)] = U[k,i]·V[l,j]  so (QΦ)[i,j] = (UᵀΦV)[i,j]
          for (let i = 0; i < d1; i++) for (let j = 0; j < d2; j++) for (let kk = 0; kk < d1; kk++) for (let l = 0; l < d2; l++) Q[(i * d2 + j) * d + kk * d2 + l] = U[kk * d1 + i] * V[l * d2 + j];
          const Gr = new Float64Array(d * d), tmpm = new Float64Array(d * d), cr = new Float64Array(d);
          for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = 0; for (let l = 0; l < d; l++) s += Q[i * d + l] * st.G[l * d + j]; tmpm[i * d + j] = s; }
          for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = 0; for (let l = 0; l < d; l++) s += tmpm[i * d + l] * Q[j * d + l]; Gr[i * d + j] = s; }
          for (let i = 0; i < d; i++) { let s = 0; for (let l = 0; l < d; l++) s += Q[i * d + l] * st.c[l]; cr[i] = s; }
          const V0 = addDiag(Gr, prior);
          m.setFrom(V0, cr, logdetRatio(V0, prior), st.n);
          return m;
        });
        if (!lowRank) for (let i = 0; i < d; i++) Q[i * d + i] = 1;
      }
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!stats) { sp.init(X); stats = sp.blocks(() => GramBlock(d)); }
          step++;
          if (step <= T1) return rng.int(K);
          if (!M) enterStage2();
          if (lowRank) for (let a = 0; a < K; a++) rotate(X, a * d, Xr, a * d);
          else Xr.set(X);
          for (let a = 0; a < K; a++) {
            const m = M[sp.of[a]];
            const beta = R * Math.sqrt(Math.max(0, m.logdet) + 2 * Math.log(1 / delta)) + Math.sqrt(lambda) * S + Math.sqrt(lamPerp) * Sperp;
            idx[a] = m.mean(Xr, a * d) + beta * Math.sqrt(m.quad(Xr, a * d));
          }
          return argmax(idx, rng);
        },
        update(a, r2) {
          if (!stats) return;
          if (!M) stats[sp.of[a]].add(X, a * d, r2);
          else M[sp.of[a]].add(Xr, a * d, r2);
        },
      };
    },
  };
  function addDiag(G, diag) {
    const d = diag.length, out = new Float64Array(G);
    for (let i = 0; i < d; i++) out[i * d + i] += diag[i];
    return out;
  }
  function logdetRatio(V, diag) {
    const d = diag.length, L = new Float64Array(d * d);
    if (!cholesky(V, d, L)) return 0;
    let s = 0;
    for (let i = 0; i < d; i++) s += 2 * Math.log(L[i * d + i]) - Math.log(diag[i]);
    return s;
  }

  // ===================================================== non-stationary linear =====
  const dlinucb = {
    id: 'dlinucb', name: 'dLinUCB', family: 'nslinear',
    params: [
      { key: 'tau', label: 'τ (badness window)', default: 200, min: 10, max: 2000, step: 10 },
      { key: 'delta1', label: 'δ₁ (tolerated error rate)', default: 0.01, min: 0.001, max: 0.5, step: 0.001 },
      { key: 'delta2', label: 'δ₂ (discard confidence)', default: 0.05, min: 0.001, max: 0.5, step: 0.001 },
    ],
    formula: 'eₘ = 1{|r − θ̂ₘ·x| > βₘ‖x‖_{Aₘ⁻¹} + ε};  drop m if êₘ > δ₁ + √(ln(1/δ₂)/2τₘ);  new model if all eₘ = 1;  act with argminₘ êₘ − √(ln τ/τₘ)',
    regret: 'Õ(Γ_T · d√(S_max))  with Γ_T change points',
    idea: 'Wu, Iyer & Wang\'s dynamic LinUCB: a pool of OFUL "slave" models, each tracking how often recent rewards fall outside its confidence band. Models that keep failing are discarded, a new one starts when none fits, and the most reliable model chooses the arm.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, rng = ctx.rng, R = sigmaOf(ctx), lambda = 1, delta = 0.05, S = 1;
      const tau = Math.max(1, Math.round(ctx.params.tau)), { delta1, delta2 } = ctx.params;
      const eps = R * normInv(1 - delta1 / 2), maxModels = 8, idx = new Float64Array(K);
      let pool = null, X = null, active = null, born = 0;
      const newModel = () => ({ M: sp.blocks(() => Ridge(d, lambda)), win: new Uint8Array(tau), wn: 0, wp: 0, bad: 0, born: born++ });
      const push = (s, e) => {
        if (s.wn === tau) s.bad -= s.win[s.wp]; else s.wn++;
        s.win[s.wp] = e; s.bad += e; s.wp = (s.wp + 1) % tau;
      };
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!pool) { sp.init(X); pool = [newModel()]; }
          let best = Infinity;
          active = pool[0];
          for (const s of pool) {
            const v = s.wn === 0 ? -Infinity : s.bad / s.wn - Math.sqrt(Math.log(tau) / s.wn);
            if (v < best) { best = v; active = s; }
          }
          for (let a = 0; a < K; a++) {
            const m = active.M[sp.of[a]];
            idx[a] = m.mean(X, a * d) + ofulBeta(m, R, lambda, delta, S) * Math.sqrt(m.quad(X, a * d));
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          if (!pool) return;
          const o = a * d;
          let allOut = true;
          for (const s of pool) {
            const m = s.M[sp.of[a]];
            const e = Math.abs(r - m.mean(X, o)) > ofulBeta(m, R, lambda, delta, S) * Math.sqrt(m.quad(X, o)) + eps ? 1 : 0;
            push(s, e);
            if (!e) { allOut = false; m.add(X, o, r); }
          }
          pool = pool.filter((s) => !(s.wn > 0 && s.bad / s.wn > delta1 + Math.sqrt(Math.log(1 / delta2) / (2 * s.wn))));
          if (allOut || pool.length === 0) {
            const s = newModel();
            s.M[sp.of[a]].add(X, o, r);
            pool.push(s);
          }
          if (pool.length > maxModels) {
            let worst = 0;
            for (let i = 1; i < pool.length - 1; i++) if (pool[i].bad / Math.max(1, pool[i].wn) > pool[worst].bad / Math.max(1, pool[worst].wn)) worst = i;
            pool.splice(worst, 1);
          }
        },
      };
    },
  };

  const swlinucb = {
    id: 'sw-linucb', name: 'SW-LinUCB', family: 'nslinear',
    params: [
      { key: 'w', label: 'w (window, steps)', default: 300, min: 10, max: 5000, step: 10 },
      P.lambda, P.delta, P.S,
    ],
    formula: 'Vₜ = λI + Σ_{s=t−w}^{t−1} xₛxₛᵀ;  argmaxₐ θ̂·xₐ + βₜ‖xₐ‖_{V⁻¹},  βₜ = σ√(d ln((1 + wL²/λ)/δ)) + √λ·S',
    regret: 'Õ(d^{2/3} B_T^{1/3} T^{2/3})  with variation budget B_T',
    idea: 'Cheung, Simchi-Levi & Zhu: OFUL on the last w observations only, so θ̂ tracks a drifting or switching parameter. Old points leave the window by a Sherman–Morrison downdate; statistics are rebuilt every w steps to stop round-off from accumulating.',
    create(K, ctx) {
      const sp = space(K, ctx), d = sp.d, rng = ctx.rng, R = sigmaOf(ctx), { lambda, delta, S } = ctx.params;
      const w = Math.max(1, Math.round(ctx.params.w)), idx = new Float64Array(K);
      const qa = new Int32Array(w), qx = new Float64Array(w * d), qr = new Float64Array(w);
      let M = null, X = null, qn = 0, qh = 0, L2 = 1, since = 0;
      return {
        select(t, Xt) {
          X = sp.view(Xt);
          if (!M) { sp.init(X); M = sp.blocks(() => Ridge(d, lambda)); }
          const beta = R * Math.sqrt(d * Math.log((1 + (Math.min(qn, w) * L2) / lambda) / delta)) + Math.sqrt(lambda) * S;
          for (let a = 0; a < K; a++) {
            const m = M[sp.of[a]];
            idx[a] = m.mean(X, a * d) + beta * Math.sqrt(m.quad(X, a * d));
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          if (!M) return;
          let nx = 0;
          for (let j = 0; j < d; j++) nx += X[a * d + j] * X[a * d + j];
          if (nx > L2) L2 = nx;
          if (qn === w) {   // evict the oldest observation (ring buffer head)
            M[sp.of[qa[qh]]].add(qx, qh * d, qr[qh], -1);
            qh = (qh + 1) % w; qn--;
          }
          const slot = (qh + qn) % w;
          qa[slot] = a; qr[slot] = r;
          for (let j = 0; j < d; j++) qx[slot * d + j] = X[a * d + j];
          qn++;
          M[sp.of[a]].add(X, a * d, r);
          if (++since >= w) {   // rebuild from the window
            since = 0;
            for (const m of M) m.reset();
            for (let i = 0; i < qn; i++) { const s = (qh + i) % w; M[sp.of[qa[s]]].add(qx, s * d, qr[s]); }
          }
        },
      };
    },
  };

  MAB.register({
    families,
    environments,
    algorithms: [oful, lints, linrel, suplinrel, suplinucb, hybrid, gopt, spectral, clucb,
      glmucb, ucbglm, logucb1, logucb2, ofuglb, lassoBandit, drLasso, estc, estr, dlinucb, swlinucb],
  });

  MAB.linear = { invert, cholesky, eigSym, lasso, nuclearLS, gOptimalDesign, Ridge, GlmBlock, space };
  return MAB;
});
