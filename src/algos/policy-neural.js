/*
 * Policy-class (oracle-based), kernel and neural contextual bandits, plus non-linear
 * contextual environments, for the K-armed & contextual arena.
 *
 * Contexts follow the arena contract: select(t, X) gets X = Float64Array(K*d) whose row a is
 * arm a's feature vector (context-free environments pass d = 1 and X = ones). Every learner
 * here keeps arm-specific parameters (a separate head / regression / dictionary per arm), so
 * it works whether the rows are a shared context repeated K times or genuine per-arm features.
 *
 * Everything is self-contained: small ridge regressions, a tiny multi-head MLP with Adam,
 * RBF kernel regression with an incremental inverse, and histogram regression trees.
 * All randomness comes from ctx.rng.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { util } = MAB;
  const { argmax, sampleIndex } = util;

  // ================================================================ helpers ====

  /** Copies the round's context into a private buffer; falls back to all-ones rows. */
  function contextView(K, d) {
    const cur = new Float64Array(K * d).fill(1);
    return (X) => {
      if (X && X.length === K * d) cur.set(X); else cur.fill(1);
      return cur;
    };
  }

  /** Width of one unit of raw reward on the env's [0, 1] normalized scale. */
  function rewardSlope(env) {
    const s = env.normalize(1) - env.normalize(0);
    return s > 1e-9 && Number.isFinite(s) ? s : 1 / (8 * (env.sigma || 1));
  }

  const clampAbs = (x, c) => (x > c ? c : x < -c ? -c : x);

  /**
   * Reward minus its running mean, in units of 2σ (σ = the env's noise scale) and clipped to
   * [−2, 2]. For Bernoulli arms this is r − r̄; for unit Gaussians it is 4× less compressed than
   * env.normalize. Importance-weighted estimates built from it shift every deterministic policy
   * by the same baseline, so rankings stay unbiased while the variance drops sharply.
   */
  function Centered(env) {
    const u = 1 / (2 * (env.sigma || 1));
    let m = 0, n = 0;
    return (r) => { const c = clampAbs((r - m) * u, 2); n++; m += (r - m) / n; return c; };
  }

  /** Raw reward gaps are measured in units of 2σ (= 1 for Bernoulli rewards). */
  const gapUnit = (env) => 1 / (2 * (env.sigma || 1));

  /**
   * K disjoint ridge regressions (one per arm), updated with Sherman–Morrison.
   * A_a = λI + Σ w x xᵀ,  b_a = Σ w y x,  θ_a = A_a⁻¹ b_a.
   */
  function Ridge(K, d, lambda) {
    const Ainv = new Float64Array(K * d * d), b = new Float64Array(K * d), theta = new Float64Array(K * d);
    const tmp = new Float64Array(d);
    function reset() {
      Ainv.fill(0); b.fill(0); theta.fill(0);
      for (let a = 0; a < K; a++) for (let i = 0; i < d; i++) Ainv[a * d * d + i * d + i] = 1 / lambda;
    }
    reset();
    function solve(a) {
      const o = a * d * d;
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += Ainv[o + i * d + j] * b[a * d + j];
        theta[a * d + i] = s;
      }
    }
    return {
      Ainv, b, theta, d, reset, solve,
      predict(a, x, off) {
        let s = 0;
        for (let j = 0; j < d; j++) s += theta[a * d + j] * x[off + j];
        return s;
      },
      /** xᵀ A_a⁻¹ x */
      quad(a, x, off) {
        const o = a * d * d;
        let q = 0;
        for (let i = 0; i < d; i++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += Ainv[o + i * d + j] * x[off + j];
          q += s * x[off + i];
        }
        return q > 0 ? q : 0;
      },
      add(a, x, off, y, w = 1) {
        const o = a * d * d;
        let den = 1 / w;
        for (let i = 0; i < d; i++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += Ainv[o + i * d + j] * x[off + j];
          tmp[i] = s; den += x[off + i] * s;
        }
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Ainv[o + i * d + j] -= (tmp[i] * tmp[j]) / den;
        for (let i = 0; i < d; i++) b[a * d + i] += w * y * x[off + i];
        solve(a);
      },
    };
  }

  /** In-place inverse of an n×n SPD matrix via Cholesky; returns false if not PD. */
  function invertSPD(A, n, L, out) {
    L.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) { if (s <= 1e-12) return false; L[i * n + i] = Math.sqrt(s); } else L[i * n + j] = s / L[j * n + j];
      }
    }
    // out = L⁻ᵀ L⁻¹, column by column
    const col = new Float64Array(n);
    for (let c = 0; c < n; c++) {
      for (let i = 0; i < n; i++) {
        let s = i === c ? 1 : 0;
        for (let k = 0; k < i; k++) s -= L[i * n + k] * col[k];
        col[i] = s / L[i * n + i];
      }
      for (let i = n - 1; i >= 0; i--) {
        let s = col[i];
        for (let k = i + 1; k < n; k++) s -= L[k * n + i] * out[k * n + c];
        out[i * n + c] = s / L[i * n + i];
      }
    }
    return true;
  }

  /** Lower Cholesky factor of a small SPD matrix (with jitter). */
  function cholesky(A, n, L) {
    L.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) L[i * n + i] = Math.sqrt(Math.max(s, 1e-12)); else L[i * n + j] = s / L[j * n + j];
      }
    }
    return L;
  }

  /**
   * A small multi-head MLP: input din → hidden layers (ReLU) → K linear heads.
   * Arm a's prediction is head a applied to row a of X. Trained with Adam on squared error.
   */
  function MLP(din, hidden, K, rng, { outScale = 1 } = {}) {
    const L = hidden.length, sizes = [din, ...hidden];
    const Woff = [], boff = [];
    let P = 0;
    for (let l = 0; l < L; l++) { Woff.push(P); P += sizes[l + 1] * sizes[l]; boff.push(P); P += sizes[l + 1]; }
    const hL = sizes[L], WoOff = P; P += K * hL;
    const boOff = P; P += K;
    const w = new Float64Array(P), g = new Float64Array(P), m1 = new Float64Array(P), m2 = new Float64Array(P);
    for (let l = 0; l < L; l++) {
      const s = Math.sqrt(2 / sizes[l]);
      for (let i = 0; i < sizes[l + 1] * sizes[l]; i++) w[Woff[l] + i] = s * rng.normal();
      for (let i = 0; i < sizes[l + 1]; i++) w[boff[l] + i] = 0.1 * rng.normal();
    }
    for (let i = 0; i < K * hL; i++) w[WoOff + i] = (outScale / Math.sqrt(hL)) * rng.normal();
    const w0 = Float64Array.from(w);
    const z = sizes.slice(1).map((n) => new Float64Array(n));    // pre-activations
    const h = sizes.map((n) => new Float64Array(n));              // h[0] = input, h[l+1] = relu(z[l])
    const dh = sizes.map((n) => new Float64Array(n));
    let adamT = 0, version = 0, cached = -1;

    function forward(x, off, arm) {
      const h0 = h[0];
      // shared contexts repeat the same row for every arm: reuse the hidden layers
      let same = cached === version;
      if (same) for (let j = 0; j < din; j++) if (h0[j] !== x[off + j]) { same = false; break; }
      if (!same) {
      for (let j = 0; j < din; j++) h0[j] = x[off + j];
      cached = version;
      for (let l = 0; l < L; l++) {
        const nin = sizes[l], nout = sizes[l + 1], zo = z[l], hi = h[l], ho = h[l + 1], W = Woff[l], B = boff[l];
        for (let i = 0; i < nout; i++) {
          let s = w[B + i];
          const r = W + i * nin;
          for (let j = 0; j < nin; j++) s += w[r + j] * hi[j];
          zo[i] = s; ho[i] = s > 0 ? s : 0;
        }
      }
      }
      const hl = h[L], r = WoOff + arm * hL;
      let o = w[boOff + arm];
      for (let j = 0; j < hL; j++) o += w[r + j] * hl[j];
      return o;
    }

    /**
     * Back-propagate dout through the last forward pass. visit(index, gradient) is called for
     * every parameter with a non-zero gradient, or (visit omitted) the gradient is added to g.
     */
    function backward(arm, dout, visit) {
      const hl = h[L], r = WoOff + arm * hL;
      const dl = dh[L];
      for (let j = 0; j < hL; j++) {
        const gj = dout * hl[j];
        if (visit) visit(r + j, gj); else g[r + j] += gj;
        dl[j] = dout * w[r + j];
      }
      if (visit) visit(boOff + arm, dout); else g[boOff + arm] += dout;
      for (let l = L - 1; l >= 0; l--) {
        const nin = sizes[l], nout = sizes[l + 1], zo = z[l], hi = h[l], dho = dh[l + 1], dhi = dh[l];
        const W = Woff[l], B = boff[l];
        if (l > 0) dhi.fill(0);
        for (let i = 0; i < nout; i++) {
          const dz = zo[i] > 0 ? dho[i] : 0;
          if (dz === 0) continue;
          const rr = W + i * nin;
          for (let j = 0; j < nin; j++) {
            const gj = dz * hi[j];
            if (visit) visit(rr + j, gj); else g[rr + j] += gj;
            if (l > 0) dhi[j] += dz * w[rr + j];
          }
          if (visit) visit(B + i, dz); else g[B + i] += dz;
        }
      }
    }

    /** Σ gᵢ²/Zᵢ (mode 0) or Zᵢ += gᵢ² (mode 1) for g = ∇θ f_arm(x) of the last forward pass. */
    let acc = 0, Zref = null;
    const visitNorm = (i, gi) => { acc += (gi * gi) / Zref[i]; };
    const visitAdd = (i, gi) => { Zref[i] += gi * gi; };
    function gradZ(arm, Z, mode) {
      acc = 0; Zref = Z;
      backward(arm, 1, mode ? visitAdd : visitNorm);
      return acc;
    }

    function adam(lr, wd = 0) {
      adamT++; version++;
      const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, adamT), c2 = 1 - Math.pow(b2, adamT);
      for (let i = 0; i < P; i++) {
        const gi = g[i] + wd * (w[i] - w0[i]);
        m1[i] = b1 * m1[i] + (1 - b1) * gi;
        m2[i] = b2 * m2[i] + (1 - b2) * gi * gi;
        w[i] -= (lr * (m1[i] / c1)) / (Math.sqrt(m2[i] / c2) + 1e-8);
        g[i] = 0;
      }
    }

    return { P, w, g, hL, forward, backward, gradZ, adam, last: () => h[L], zeroGrad: () => g.fill(0) };
  }

  /** Replay buffer of (row features, arm, reward, extra per-sample numbers); a ring when full. */
  function Buffer(d, cap, extra = 0) {
    const X = new Float64Array(cap * d), A = new Int32Array(cap), Y = new Float64Array(cap);
    const E = new Float64Array(cap * Math.max(1, extra));
    let n = 0, head = 0;
    return {
      X, A, Y, E, d, extra,
      get n() { return n; },
      push(x, off, a, y) {
        const i = head;
        for (let j = 0; j < d; j++) X[i * d + j] = x[off + j];
        A[i] = a; Y[i] = y;
        head = (head + 1) % cap; if (n < cap) n++;
        return i;
      },
    };
  }

  /** A few Adam steps of minibatch squared-error regression of net on buf. */
  function trainNet(net, buf, rng, steps, batch, lr, clip, wd, targetFn) {
    const n = buf.n;
    if (!n) return;
    const d = buf.d;
    for (let s = 0; s < steps; s++) {
      const B = Math.min(batch, n);
      for (let b = 0; b < B; b++) {
        const i = rng.int(n), a = buf.A[i];
        const y = targetFn ? targetFn(i) : buf.Y[i];
        const e = clampAbs(net.forward(buf.X, i * d, a) - y, clip);
        net.backward(a, e / B);
      }
      net.adam(lr, wd);
    }
  }

  /** Round-robin over arms for the first K rounds (pull each arm once). */
  const warmArm = (t, K) => (t <= K ? t - 1 : -1);

  /**
   * A fixed finite policy class: nLin random linear policies π(x) = argmaxₐ wₐ·xₐ with
   * wₐ ~ N(0, I), plus the K constant policies. Drawn from ctx.rng at creation.
   */
  function PolicyClass(K, d, nLin, rng) {
    const N = nLin + K, W = new Float64Array(nLin * K * d), acts = new Int32Array(N);
    for (let i = 0; i < W.length; i++) W[i] = rng.normal();
    for (let c = 0; c < K; c++) acts[nLin + c] = c;
    return {
      N, acts,
      actions(X) {
        for (let i = 0; i < nLin; i++) {
          let best = -Infinity, ba = 0;
          for (let a = 0; a < K; a++) {
            let s = 0;
            const o = (i * K + a) * d;
            for (let j = 0; j < d; j++) s += W[o + j] * X[a * d + j];
            if (s > best) { best = s; ba = a; }
          }
          acts[i] = ba;
        }
        return acts;
      },
    };
  }

  /** Policy-class parameters shared by the oracle-based methods. */
  const pPolicies = { key: 'policies', label: 'random linear policies', default: 64, min: 8, max: 256, step: 8 };

  /** History of policy actions and IPS data for the optimisation-based policy methods. */
  function PolicyHistory(K, N, cap) {
    const acts = new Uint16Array(N * cap), arm = new Int32Array(cap), prob = new Float64Array(cap);
    const rew = new Float64Array(cap), val = new Float64Array(N);
    let n = 0;
    return {
      acts, arm, prob, rew, val,
      get n() { return n; },
      push(pa, a, p, r) {
        if (n >= cap) return;                       // horizon exceeded: stop recording
        for (let i = 0; i < N; i++) acts[n * N + i] = pa[i];
        arm[n] = a; prob[n] = p; rew[n] = r;
        const ips = r / p;
        for (let i = 0; i < N; i++) if (pa[i] === a) val[i] += ips;
        n++;
      },
      erm() {
        let b = 0;
        for (let i = 1; i < N; i++) if (val[i] > val[b]) b = i;
        return b;
      },
    };
  }

  // ========================================================== environments ====

  /*
   * Shared context x ~ U(−1, 1)^5 repeated on every arm's row. Arm a's mean is
   *   f_a(x) = cos(3 θₐ·x) + βₐ·x,   θₐ uniform on the unit sphere, βₐ ~ N(0, 0.3² I),
   * so a linear model captures only the small linear part of each arm's reward.
   */
  function nonlinearFill(binary) {
    return function fill(K, T, rng, means, rewards, ctx) {
      const d = 5, th = new Float64Array(K * d), be = new Float64Array(K * d), x = new Float64Array(d);
      for (let a = 0; a < K; a++) {
        let nn = 0;
        for (let j = 0; j < d; j++) { const v = rng.normal(); th[a * d + j] = v; nn += v * v; }
        nn = Math.sqrt(nn) || 1;
        for (let j = 0; j < d; j++) { th[a * d + j] /= nn; be[a * d + j] = 0.3 * rng.normal(); }
      }
      for (let t = 0; t < T; t++) {
        for (let j = 0; j < d; j++) x[j] = 2 * rng.uniform() - 1;
        for (let a = 0; a < K; a++) {
          let s = 0, l = 0;
          const row = (t * K + a) * d;
          for (let j = 0; j < d; j++) { s += th[a * d + j] * x[j]; l += be[a * d + j] * x[j]; ctx[row + j] = x[j]; }
          const f = Math.cos(3 * s) + l;
          if (binary) {
            const p = 1 / (1 + Math.exp(-2.5 * f));
            means[t * K + a] = p;
            rewards[t * K + a] = rng.uniform() < p ? 1 : 0;
          } else {
            means[t * K + a] = f;
            rewards[t * K + a] = f + 0.5 * rng.normal();
          }
        }
      }
    };
  }

  const environments = [
    {
      id: 'nonlinear', name: 'Non-linear contextual', short: 'Non-linear',
      type: 'gaussian', stationary: true, contextual: true, d: 5, sigma: 0.5,
      tags: ['context', 'non-linear'],
      description: 'Each round shows a shared context x ~ U(−1, 1)⁵. Arm a pays N(cos(3θₐ·x) + βₐ·x, 0.5²) with θₐ a random unit vector and βₐ ~ N(0, 0.3²I). A linear model sees only the small linear part, so the best arm per context needs a non-linear learner.',
      fill: nonlinearFill(false),
    },
    {
      id: 'nonlinear-binary', name: 'Non-linear contextual, binary', short: 'Non-linear 0/1',
      type: 'bernoulli', stationary: true, contextual: true, d: 5,
      tags: ['context', 'non-linear'],
      description: 'Same hidden functions as the non-linear environment, but arm a pays 1 with probability σ(2.5·f_a(x)), a logistic link on f_a(x) = cos(3θₐ·x) + βₐ·x.',
      fill: nonlinearFill(true),
    },
  ];

  const families = [
    { id: 'policy', name: 'Policy-class & oracle-based', blurb: 'Compete with the best policy in a class using only an argmax (cost-sensitive) or regression oracle; exploration comes from carefully shaped action distributions.' },
    { id: 'kernel', name: 'Kernel', blurb: 'Non-parametric reward models: kernel ridge regression with an RBF kernel and confidence widths from the kernel posterior variance.' },
    { id: 'neural', name: 'Neural & ensembles', blurb: 'Learn non-linear reward functions with small neural networks, trees or ensembles, and explore with gradient-based bonuses, posterior samples or bootstraps.' },
  ];

  // ======================================================= policy class ====

  const algorithms = [];

  algorithms.push({
    id: 'exp4', name: 'EXP4', family: 'policy',
    params: [pPolicies, { key: 'c', label: 'η multiplier', default: 3, min: 0.1, max: 50, step: 0.1 }],
    formula: 'pₜ = (1−γ)·Σᵢ wᵢ ξᵢ(xₜ)/W + γ/K,   wᵢ ← wᵢ·exp(η ξᵢ·r̂),  r̂ₐ = (r − r̄)·𝟙[a=aₜ]/pₜ(a),  η = c√(2 ln N/(TK))',
    regret: 'O(√(TK ln N)) against the best of N experts, any reward sequence',
    idea: 'Exponential weights over N experts (here random linear policies plus the K constant arms, drawn once from the run\'s RNG), each scored on importance-weighted rewards. Learns which policy is best without modelling rewards. Rewards are normalized and centred on their running mean (a common baseline that cuts IPS variance), with γ at half its theory value and η scaled up by c.',
    create(K, ctx) {
      const center = Centered(ctx.env);
      const { rng, params, env, T } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const pc = PolicyClass(K, d, Math.round(params.policies), rng), N = pc.N;
      const lw = new Float64Array(N), wt = new Float64Array(N), p = new Float64Array(K);
      const TT = Math.max(T, 1), gamma = Math.min(0.5, 0.5 * Math.sqrt((K * Math.log(N)) / ((Math.E - 1) * TT)));
      const eta = params.c * Math.sqrt((2 * Math.log(N)) / (TT * K));
      let acts = pc.acts;
      return {
        select(t, X) {
          acts = pc.actions(view(X));
          let m = -Infinity;
          for (let i = 0; i < N; i++) if (lw[i] > m) m = lw[i];
          let W = 0;
          for (let i = 0; i < N; i++) { wt[i] = Math.exp(lw[i] - m); W += wt[i]; }
          p.fill(0);
          for (let i = 0; i < N; i++) p[acts[i]] += wt[i] / W;
          for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma / K;
          return sampleIndex(p, rng);
        },
        update(a, r) {
          const rhat = center(r) / p[a];
          for (let i = 0; i < N; i++) if (acts[i] === a) lw[i] += eta * rhat;
        },
      };
    },
  });

  algorithms.push({
    id: 'exp4p', name: 'EXP4.P', family: 'policy',
    params: [pPolicies, { key: 'delta', label: 'δ (confidence)', default: 0.05, min: 0.001, max: 0.5, step: 0.001 },
      { key: 'c', label: 'learning-rate multiplier', default: 10, min: 0.1, max: 50, step: 0.1 }],
    formula: 'pₜ = (1−K·p_min)·Σᵢ wᵢξᵢ/W + p_min,   wᵢ ← wᵢ·exp(p_min/2·(ξᵢ·r̂ + v̂ᵢ√(ln(N/δ)/KT))),  v̂ᵢ = Σₐ ξᵢ(a)/pₜ(a)',
    regret: 'O(√(KT ln(N/δ))) with probability 1−δ',
    idea: 'Beygelzimer et al.\'s high-probability EXP4: each expert\'s score gets an upper-confidence bonus proportional to the variance of its importance-weighted estimate, so regret holds with high probability, not just in expectation.',
    create(K, ctx) {
      const center = Centered(ctx.env);
      const { rng, params, env, T } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const pc = PolicyClass(K, d, Math.round(params.policies), rng), N = pc.N;
      const lw = new Float64Array(N), wt = new Float64Array(N), p = new Float64Array(K);
      const TT = Math.max(T, 1), pmin = Math.min(1 / (2 * K), Math.sqrt(Math.log(N) / (K * TT)));
      const bonus = Math.sqrt(Math.log(N / params.delta) / (K * TT));
      let acts = pc.acts;
      return {
        select(t, X) {
          acts = pc.actions(view(X));
          let m = -Infinity;
          for (let i = 0; i < N; i++) if (lw[i] > m) m = lw[i];
          let W = 0;
          for (let i = 0; i < N; i++) { wt[i] = Math.exp(lw[i] - m); W += wt[i]; }
          p.fill(0);
          for (let i = 0; i < N; i++) p[acts[i]] += wt[i] / W;
          for (let a = 0; a < K; a++) p[a] = (1 - K * pmin) * p[a] + pmin;
          return sampleIndex(p, rng);
        },
        update(a, r) {
          const rhat = center(r) / p[a];
          for (let i = 0; i < N; i++) {
            const yi = acts[i] === a ? rhat : 0, vi = 1 / p[acts[i]];
            lw[i] += params.c * (pmin / 2) * (yi + vi * bonus);
          }
        },
      };
    },
  });

  algorithms.push({
    id: 'epoch-greedy', name: 'Epoch-Greedy', family: 'policy',
    params: [pPolicies, { key: 'c', label: 'exploit-length scale', default: 1, min: 0.1, max: 10, step: 0.1 }],
    formula: 'explore 1 round uniformly, then exploit π̂ = argmax_π Σ K·r·𝟙[π(x)=a] for s(ℓ) = ⌈c√(ℓ / ln N)⌉ rounds',
    regret: 'O(T^{2/3} (K ln N)^{1/3})',
    idea: 'Langford & Zhang\'s epoch scheme: each epoch spends one uniformly random round to collect an unbiased sample, then plays the empirically best policy of the class on all exploration data for a growing number of rounds.',
    create(K, ctx) {
      const center = Centered(ctx.env);
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const pc = PolicyClass(K, d, Math.round(params.policies), rng), N = pc.N;
      const val = new Float64Array(N), lnN = Math.log(N);
      let acts = pc.acts, exploit = 0, ell = 0, exploring = false;
      return {
        select(t, X) {
          acts = pc.actions(view(X));
          if (exploit > 0) {
            exploit--; exploring = false;
            let b = 0;
            for (let i = 1; i < N; i++) if (val[i] > val[b]) b = i;
            return acts[b];
          }
          exploring = true;
          return rng.int(K);
        },
        update(a, r) {
          if (!exploring) return;
          const ips = K * center(r);
          for (let i = 0; i < N; i++) if (acts[i] === a) val[i] += ips;
          ell++;
          exploit = Math.ceil(params.c * Math.sqrt(ell / lnN));
        },
      };
    },
  });

  /**
   * Shared machinery for Randomized UCB and ILOVETOCONBANDITS: a sparse distribution Q over the
   * finite class, and mass[τ·K + a] = Σ_π Q(π)𝟙[π(x_τ) = a] on every logged context.
   */
  function PolicyOP(K, N, cap) {
    const H = PolicyHistory(K, N, cap), Q = new Float64Array(N), mass = new Float64Array(cap * K);
    const V = new Float64Array(N), S = new Float64Array(N);
    let filled = 0;   // contexts whose mass row is up to date
    function syncMass() {
      const n = H.n;
      for (let tau = filled; tau < n; tau++) {
        const o = tau * K, ao = tau * N;
        for (let a = 0; a < K; a++) mass[o + a] = 0;
        for (let i = 0; i < N; i++) if (Q[i] > 0) mass[o + H.acts[ao + i]] += Q[i];
      }
      filled = n;
    }
    function addMass(i, alpha) {
      Q[i] += alpha;
      const n = H.n;
      for (let tau = 0; tau < n; tau++) mass[tau * K + H.acts[tau * N + i]] += alpha;
    }
    function scale(c) {
      for (let i = 0; i < N; i++) Q[i] *= c;
      const n = H.n * K;
      for (let i = 0; i < n; i++) mass[i] *= c;
    }
    /** V_π = mean_τ 1/Q^μ(π(x_τ)|x_τ) and S_π = mean of its square, for every π. */
    function variances(mu) {
      const n = H.n, sm = 1 - K * mu;
      V.fill(0); S.fill(0);
      for (let tau = 0; tau < n; tau++) {
        const o = tau * K, ao = tau * N;
        for (let i = 0; i < N; i++) {
          const q = 1 / (sm * mass[o + H.acts[ao + i]] + mu);
          V[i] += q; S[i] += q * q;
        }
      }
      for (let i = 0; i < N; i++) { V[i] /= n; S[i] /= n; }
    }
    /** V_π for a single policy after mixing Q ← (1−α)Q + α·δ_π. */
    function variance1(i, alpha, mu) {
      const n = H.n, sm = 1 - K * mu;
      let v = 0;
      for (let tau = 0; tau < n; tau++) {
        const m = (1 - alpha) * mass[tau * K + H.acts[tau * N + i]] + alpha;
        v += 1 / (sm * m + mu);
      }
      return v / n;
    }
    /** Arm distribution at the current context: Q plus leftover mass on the ERM policy. */
    function play(acts, mu, p) {
      const erm = H.erm();
      let tot = 0;
      p.fill(0);
      for (let i = 0; i < N; i++) if (Q[i] > 0) { p[acts[i]] += Q[i]; tot += Q[i]; }
      if (tot < 1) p[acts[erm]] += 1 - tot;
      else for (let a = 0; a < K; a++) p[a] /= tot;
      for (let a = 0; a < K; a++) p[a] = (1 - K * mu) * p[a] + mu;
      return p;
    }
    return { H, Q, mass, V, S, syncMass, addMass, scale, variances, variance1, play };
  }

  algorithms.push({
    id: 'ilovetoconbandits', name: 'ILOVETOCONBANDITS', family: 'policy',
    params: [pPolicies, { key: 'psi', label: 'ψ (regret slack)', default: 1, min: 0.01, max: 100, step: 0.01 }],
    formula: 'find Q: Σ_π Q(π)·R̂eg(π) ≤ 2Kψμ  and  Ê_x[1/Q^μ(π(x)|x)] ≤ 2K + R̂eg(π)/(ψμ) ∀π;  aₜ ~ Q^μ(·|xₜ)',
    regret: 'Õ(√(KT ln N)) with Õ(√(KT/ln N)) oracle calls',
    idea: 'Agarwal et al.\'s "taming the monster": a sparse distribution over policies that has low empirical regret yet keeps every policy\'s IPS variance small, found by coordinate descent with argmax-oracle calls. Simplified: re-solved on a geometric schedule (every ×1.3 rounds) over the finite class, with μ = min(1/2K, √(ln N / Kt)).',
    create(K, ctx) {
      const center = Centered(ctx.env);
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const T = Math.max(ctx.T | 0, 1);
      const pc = PolicyClass(K, d, Math.round(params.policies), rng), N = pc.N;
      const op = PolicyOP(K, N, T), H = op.H, p = new Float64Array(K), b = new Float64Array(N);
      const lnN = Math.log(N);
      let acts = pc.acts, nextSolve = K, mu = 1 / (2 * K);
      function solve() {
        const t = H.n;
        mu = Math.min(1 / (2 * K), Math.sqrt(lnN / (K * t)));
        op.syncMass();
        let best = -Infinity;
        for (let i = 0; i < N; i++) if (H.val[i] > best) best = H.val[i];
        for (let i = 0; i < N; i++) b[i] = (best - H.val[i]) / t / (params.psi * mu);
        for (let it = 0; it < 20; it++) {
          let s = 0;
          for (let i = 0; i < N; i++) if (op.Q[i] > 0) s += op.Q[i] * (2 * K + b[i]);
          if (s > 2 * K) op.scale((2 * K) / s);
          op.variances(mu);
          let bi = -1, bd = 1e-9;
          for (let i = 0; i < N; i++) { const D = op.V[i] - (2 * K + b[i]); if (D > bd) { bd = D; bi = i; } }
          if (bi < 0) break;
          const alpha = (op.V[bi] + bd) / (2 * (1 - K * mu) * op.S[bi]);
          op.addMass(bi, alpha);
        }
        let tot = 0;
        for (let i = 0; i < N; i++) tot += op.Q[i];
        if (tot > 1) op.scale(1 / tot);
      }
      return {
        select(t, X) {
          acts = pc.actions(view(X));
          if (H.n >= nextSolve) { solve(); nextSolve = Math.ceil(H.n * 1.3) + 1; }
          return sampleIndex(op.play(acts, mu, p), rng);
        },
        update(a, r) { H.push(acts, a, p[a], center(r)); },
      };
    },
  });

  algorithms.push({
    id: 'rucb', name: 'Randomized UCB', family: 'policy',
    params: [pPolicies, { key: 'c', label: 'C (confidence scale)', default: 1, min: 0.05, max: 20, step: 0.05 }],
    formula: 'P ← argmin Σ_π P(π)Δ̂(π)  s.t.  Ê_x[1/P^μ(π(x)|x)] ≤ max{2K, t·Δ̂(π)² / (C ln Nt)} ∀π',
    regret: 'Õ(√(KT ln N)), polynomial-time with an argmax oracle',
    idea: 'Dudík et al.\'s Randomized UCB plays a distribution over policies whose IPS variance is small for every policy that could still be optimal. Simplified: start from the empirical best policy and mix in the most-violating policy (bisection on its weight) up to 10 times, re-solved every ×1.5 rounds.',
    create(K, ctx) {
      const center = Centered(ctx.env);
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const T = Math.max(ctx.T | 0, 1);
      const pc = PolicyClass(K, d, Math.round(params.policies), rng), N = pc.N;
      const op = PolicyOP(K, N, T), H = op.H, p = new Float64Array(K), bound = new Float64Array(N);
      let acts = pc.acts, nextSolve = K, mu = 1 / (2 * K);
      function solve() {
        const t = H.n;
        mu = Math.min(1 / (2 * K), 1 / Math.sqrt(K * t));
        const C = params.c * Math.log(N * t + 1);
        let best = -Infinity, erm = 0;
        for (let i = 0; i < N; i++) if (H.val[i] > best) { best = H.val[i]; erm = i; }
        for (let i = 0; i < N; i++) { const gap = (best - H.val[i]) / t; bound[i] = Math.max(2 * K, (t * gap * gap) / C); }
        // restart from the empirical best policy, then fix violated constraints greedily
        op.Q.fill(0); op.Q[erm] = 1;
        op.mass.fill(0, 0, t * K);
        for (let tau = 0; tau < t; tau++) op.mass[tau * K + H.acts[tau * N + erm]] = 1;
        for (let it = 0; it < 10; it++) {
          op.variances(mu);
          let bi = -1, bv = 1e-9;
          for (let i = 0; i < N; i++) { const v = op.V[i] - bound[i]; if (v > bv) { bv = v; bi = i; } }
          if (bi < 0) break;
          let lo = 0, hi = 1;
          for (let k = 0; k < 10; k++) {
            const m = (lo + hi) / 2;
            if (op.variance1(bi, m, mu) > bound[bi]) lo = m; else hi = m;
          }
          op.scale(1 - hi); op.addMass(bi, hi);
        }
      }
      return {
        select(t, X) {
          acts = pc.actions(view(X));
          if (H.n >= nextSolve) { solve(); nextSolve = Math.ceil(H.n * 1.5) + 1; }
          return sampleIndex(op.play(acts, mu, p), rng);
        },
        update(a, r) { H.push(acts, a, p[a], center(r)); },
      };
    },
  });

  algorithms.push({
    id: 'online-cover', name: 'Online Cover', family: 'policy',
    params: [
      { key: 'm', label: 'cover size m', default: 4, min: 1, max: 16, step: 1 },
      { key: 'psi', label: 'ψ (diversity bonus)', default: 0.1, min: 0, max: 2, step: 0.01 },
    ],
    formula: 'aₜ ~ (1−Kμ)·(1/m)Σᵢ𝟙[πᵢ(x)=a] + μ;   πᵢ ← CSC(r̂ + ψμ / Q^μ_{i−1}(·|x))',
    regret: 'Õ(√(KT ln N)) for the batch version; online version is a heuristic',
    idea: 'Agarwal et al.\'s Online Cover (the contextual-bandit "cover" in Vowpal Wabbit): m cost-sensitive learners are trained online on IPS reward vectors; learner i also gets a bonus for arms the first i−1 learners rarely choose, so together they cover the plausible actions. The CSC oracle here is a per-arm ridge regression on the IPS targets.',
    create(K, ctx) {
      const center = Centered(ctx.env);
      const { rng, params } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const m = Math.max(1, Math.round(params.m));
      // All learners regress on the same rows every round, so they share each arm's Gram
      // inverse A_a⁻¹ and differ only in b; prediction = (A_a⁻¹xₐ)·b_{i,a}.
      const G = Ridge(K, d, 1), bs = new Float64Array(m * K * d), u = new Float64Array(K * d);
      const choice = new Int32Array(m), p = new Float64Array(K), qPrev = new Float64Array(K), jit = new Float64Array(K);
      let X = null, mu = 0.5 / K;
      function project() {
        for (let a = 0; a < K; a++) {
          const o = a * d * d;
          for (let i = 0; i < d; i++) {
            let s = 0;
            for (let j = 0; j < d; j++) s += G.Ainv[o + i * d + j] * X[a * d + j];
            u[a * d + i] = s;
          }
        }
      }
      const pick = (i) => {
        let best = -Infinity, ba = 0;
        for (let a = 0; a < K; a++) {
          let v = jit[a];
          const o = (i * K + a) * d;
          for (let j = 0; j < d; j++) v += u[a * d + j] * bs[o + j];
          if (v > best) { best = v; ba = a; }
        }
        return ba;
      };
      return {
        select(t, Xt) {
          X = view(Xt);
          mu = Math.min(1 / (2 * K), 1 / Math.sqrt(K * t));
          for (let a = 0; a < K; a++) jit[a] = 1e-9 * rng.uniform();
          project();
          p.fill(0);
          for (let i = 0; i < m; i++) { choice[i] = pick(i); p[choice[i]] += 1 / m; }
          for (let a = 0; a < K; a++) p[a] = (1 - K * mu) * p[a] + mu;
          return sampleIndex(p, rng);
        },
        update(a, r) {
          const rhat = center(r) / p[a];
          // Sherman–Morrison with u = A⁻¹x from select:  A⁻¹ ← A⁻¹ − u uᵀ/(1+xᵀu),  and A_new⁻¹x = u/(1+xᵀu)
          for (let b = 0; b < K; b++) {
            const o = b * d * d, uo = b * d;
            let q = 1;
            for (let j = 0; j < d; j++) q += X[uo + j] * u[uo + j];
            for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) G.Ainv[o + i * d + j] -= (u[uo + i] * u[uo + j]) / q;
            for (let j = 0; j < d; j++) u[uo + j] /= q;
          }
          qPrev.fill(0);
          for (let i = 0; i < m; i++) {
            for (let b = 0; b < K; b++) {
              let target = b === a ? rhat : 0;
              if (i > 0) target += (params.psi * mu) / ((1 - K * mu) * (qPrev[b] / i) + mu);
              const o = (i * K + b) * d;
              for (let j = 0; j < d; j++) bs[o + j] += target * X[b * d + j];
            }
            qPrev[pick(i)] += 1;
          }
        },
      };
    },
  });

  algorithms.push({
    id: 'bootstrap-ts', name: 'Bootstrap Thompson', family: 'policy',
    params: [
      { key: 'J', label: 'J (bootstrap replicates)', default: 10, min: 2, max: 50, step: 1 },
      { key: 'prior', label: 'prior scale', default: 1, min: 0, max: 5, step: 0.05 },
    ],
    formula: 'j ~ U{1..J},  aₜ = argmaxₐ θ̂ⱼ,ₐ·xₐ;  each replicate refits with weight w ~ 2·Bernoulli(½)',
    regret: 'No general guarantee; approximates Thompson sampling',
    idea: 'Eckles & Kaptein\'s bootstrap Thompson sampling (and Vowpal Wabbit\'s "bag"): J per-arm ridge regressions each see every observation with a double-or-nothing weight; playing greedily on a random replicate approximates a posterior sample. Each replicate starts from a random prior mean so the replicates disagree before data arrives.',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const J = Math.max(2, Math.round(params.J)), s = params.prior * env.sigma;
      const reps = Array.from({ length: J }, () => {
        const R = Ridge(K, d, 1);
        for (let a = 0; a < K; a++) { for (let j = 0; j < d; j++) R.b[a * d + j] = s * rng.normal(); R.solve(a); }
        return R;
      });
      const idx = new Float64Array(K);
      let X = null;
      return {
        select(t, Xt) {
          X = view(Xt);
          const R = reps[rng.int(J)];
          for (let a = 0; a < K; a++) idx[a] = R.predict(a, X, a * d);
          return argmax(idx, rng);
        },
        update(a, r) {
          for (let j = 0; j < J; j++) if (rng.uniform() < 0.5) reps[j].add(a, X, a * d, r, 2);
        },
      };
    },
  });

  /** Inverse-gap weighting: p_a = 1/(K + γ·gap_a) off the greedy arm, the rest on it. */
  function igw(yhat, gamma, p, K, rng) {
    const b = argmax(yhat, rng);
    let rest = 0;
    for (let a = 0; a < K; a++) if (a !== b) { p[a] = 1 / (K + gamma * (yhat[b] - yhat[a])); rest += p[a]; }
    p[b] = 1 - rest;
    return p;
  }

  algorithms.push({
    id: 'squarecb', name: 'SquareCB', family: 'policy',
    params: [{ key: 'c', label: 'γ scale', default: 2, min: 0.05, max: 20, step: 0.05 }],
    formula: 'pₜ(a) = 1/(K + γ(ŷ_b − ŷₐ)) for a ≠ b = argmax ŷ,   γ = c√(Kt),  ŷ from an online square-loss oracle',
    regret: 'O(√(KT·Reg_sq(T)))',
    idea: 'Foster & Rakhlin\'s reduction from contextual bandits to online regression: play inverse-gap weighting on the oracle\'s predictions. The oracle here is a per-arm online ridge regression; gaps are measured in units of 2σ (the reward range for Bernoulli arms).',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const R = Ridge(K, d, 1), yhat = new Float64Array(K), p = new Float64Array(K), unit = gapUnit(env);
      let X = null;
      return {
        select(t, Xt) {
          X = view(Xt);
          for (let a = 0; a < K; a++) yhat[a] = R.predict(a, X, a * d);
          return sampleIndex(igw(yhat, params.c * Math.sqrt(K * t) * unit, p, K, rng), rng);
        },
        update(a, r) { R.add(a, X, a * d, r); },
      };
    },
  });

  algorithms.push({
    id: 'fastcb', name: 'FastCB', family: 'policy',
    params: [{ key: 'c', label: 'γ scale', default: 2, min: 0.05, max: 20, step: 0.05 }],
    formula: 'pₜ(a) = ℓ̂_b / (K ℓ̂_b + γ(ℓ̂ₐ − ℓ̂_b)) for a ≠ b = argmin ℓ̂,   ℓ̂ from an online log-loss oracle',
    regret: 'Õ(√(L* K·Reg_log)) — first-order (small-loss) bound',
    idea: 'Foster & Krishnamurthy\'s first-order variant of SquareCB: losses ℓ = 1 − r (normalized) are predicted by a log-loss oracle, and exploration scales with the best arm\'s predicted loss, so it nearly stops exploring when the best arm is almost perfect. Oracle: per-arm online-Newton logistic regression.',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const R = Ridge(K, d, 1), th = new Float64Array(K * d), ell = new Float64Array(K), neg = new Float64Array(K);
      const p = new Float64Array(K), slope = rewardSlope(env), g = new Float64Array(d);
      let X = null;
      const sig = (z) => 1 / (1 + Math.exp(-z));
      return {
        select(t, Xt) {
          X = view(Xt);
          for (let a = 0; a < K; a++) {
            let z = 0;
            for (let j = 0; j < d; j++) z += th[a * d + j] * X[a * d + j];
            ell[a] = Math.min(1 - 1e-3, Math.max(1e-3, sig(z)));
            neg[a] = -ell[a];
          }
          const b = argmax(neg, rng), gamma = (params.c * Math.sqrt(K * t) * gapUnit(env)) / slope;
          let rest = 0;
          for (let a = 0; a < K; a++) if (a !== b) { p[a] = ell[b] / (K * ell[b] + gamma * (ell[a] - ell[b])); rest += p[a]; }
          p[b] = 1 - rest;
          return sampleIndex(p, rng);
        },
        update(a, r) {
          const y = 1 - env.normalize(r);
          let z = 0;
          for (let j = 0; j < d; j++) z += th[a * d + j] * X[a * d + j];
          const q = sig(z), w = Math.max(q * (1 - q), 0.02);
          R.add(a, X, a * d, 0, w);                     // A ← A + q(1−q)·x xᵀ  (online Newton step)
          const o = a * d * d;
          for (let j = 0; j < d; j++) g[j] = (q - y) * X[a * d + j];
          for (let i = 0; i < d; i++) {
            let s = 0;
            for (let j = 0; j < d; j++) s += R.Ainv[o + i * d + j] * g[j];
            th[a * d + i] -= s;
          }
        },
      };
    },
  });

  algorithms.push({
    id: 'smoothcb', name: 'SmoothCB', family: 'policy',
    params: [
      { key: 'h', label: 'h (smoothness, ×1/K)', default: 2, min: 1, max: 10, step: 0.5 },
      { key: 'c', label: 'γ scale', default: 2, min: 0.05, max: 20, step: 0.05 },
    ],
    formula: 'pₜ(a) = μ(a) / (1 + hγ(ŷ_b − ŷₐ)) for a ≠ b,  μ = Uniform(K),  γ = c√(t/h)',
    regret: 'Õ(√(T·Reg_sq / h)) smooth regret',
    idea: 'Zhu & Mineiro\'s SmoothIGW, built for huge or continuous action spaces, competes with "smoothed" policies whose density is at most 1/h w.r.t. a base measure μ. Finite-action adaptation: μ is uniform over the K arms and h is given in multiples of 1/K (h = 1/K recovers SquareCB); the comparator is then the best mix over ⌈hK⌉ arms. Oracle: per-arm online ridge.',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const R = Ridge(K, d, 1), yhat = new Float64Array(K), p = new Float64Array(K), unit = gapUnit(env);
      const h = Math.min(1, params.h / K);
      let X = null;
      return {
        select(t, Xt) {
          X = view(Xt);
          for (let a = 0; a < K; a++) yhat[a] = R.predict(a, X, a * d);
          const b = argmax(yhat, rng), gamma = params.c * Math.sqrt(t / h) * unit;
          let rest = 0;
          for (let a = 0; a < K; a++) if (a !== b) { p[a] = (1 / K) / (1 + h * gamma * (yhat[b] - yhat[a])); rest += p[a]; }
          p[b] = 1 - rest;
          return sampleIndex(p, rng);
        },
        update(a, r) { R.add(a, X, a * d, r); },
      };
    },
  });

  algorithms.push({
    id: 'fgts', name: 'Feel-Good TS', family: 'policy',
    params: [
      { key: 'lambda', label: 'λ (feel-good weight)', default: 0.1, min: 0, max: 2, step: 0.01 },
      { key: 'eta', label: 'η (likelihood, ×1/σ²)', default: 0.5, min: 0.05, max: 2, step: 0.05 },
    ],
    formula: 'p(θ|Hₜ) ∝ p₀(θ)·exp(Σ_τ [−η(θ_{aτ}·x_τ − r_τ)² + λ·min(b, maxₐ θₐ·x_τ,ₐ)]),  aₜ = argmax θ̃ₐ·xₐ',
    regret: 'Õ(d√T) frequentist (Zhang 2021)',
    idea: 'Zhang\'s Feel-Good Thompson sampling adds an optimism term λ·max f(x) to the log-likelihood so the posterior favours models that predict high rewards, fixing TS\'s worst-case frequentist gap. Approximate: per-arm linear model sampled by 2 preconditioned Langevin (SGLD) steps per round (1 when d > 12), warm-started, with the feel-good gradient estimated on 8 past contexts.',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const T = Math.max(ctx.T | 0, 1), s2 = env.sigma * env.sigma;
      const two = (2 * params.eta) / s2;   // 2η
      const R = Ridge(K, d, 1);           // A_a = I + 2η Σ x xᵀ  (via weight 2η), mean = A⁻¹ 2η Σ r x
      const th = new Float64Array(K * d), gfg = new Float64Array(K * d), fgOn = new Uint8Array(K);
      const Lc = new Float64Array(K * d * d), L = new Float64Array(d * d), Ablk = new Float64Array(d * d);
      const dirty = new Uint8Array(K).fill(1), xi = new Float64Array(d), step = new Float64Array(d);
      const HC = Math.min(T, 2000), hist = new Float64Array(HC * K * d), idx = new Float64Array(K);
      const cap = env.type === 'bernoulli' ? 1 : 2 * env.sigma;
      const B = 8;
      let n = 0, head = 0, tot = 0, X = null;
      function sgld() {
        gfg.fill(0); fgOn.fill(0);
        if (n > 0 && params.lambda > 0) {
          const scale = (params.lambda * tot) / B;   // unbiased for the sum over all past rounds
          for (let k = 0; k < B; k++) {
            const o = rng.int(n) * K * d;
            let best = -Infinity, ba = 0;
            for (let a = 0; a < K; a++) {
              let s = 0;
              for (let j = 0; j < d; j++) s += th[a * d + j] * hist[o + a * d + j];
              if (s > best) { best = s; ba = a; }
            }
            if (best < cap) { fgOn[ba] = 1; for (let j = 0; j < d; j++) gfg[ba * d + j] += scale * hist[o + ba * d + j]; }
          }
        }
        for (let a = 0; a < K; a++) {
          const o = a * d * d;
          if (dirty[a]) {
            for (let i = 0; i < d * d; i++) Ablk[i] = R.Ainv[o + i];
            Lc.set(cholesky(Ablk, d, L), o); dirty[a] = 0;
          }
          for (let j = 0; j < d; j++) xi[j] = rng.normal();
          // θ ← θ + ½(m − θ + A⁻¹ g_fg) + L ξ   (ε = 1, preconditioner A⁻¹)
          for (let i = 0; i < d; i++) {
            let pre = 0, noise = 0;
            for (let j = 0; j <= i; j++) noise += Lc[o + i * d + j] * xi[j];
            if (fgOn[a]) for (let j = 0; j < d; j++) pre += R.Ainv[o + i * d + j] * gfg[a * d + j];
            step[i] = 0.5 * (R.theta[a * d + i] - th[a * d + i] + pre) + noise;
          }
          for (let i = 0; i < d; i++) th[a * d + i] += step[i];
        }
      }
      return {
        select(t, Xt) {
          X = view(Xt);
          sgld(); if (d <= 12) sgld();
          for (let a = 0; a < K; a++) {
            let s = 0;
            for (let j = 0; j < d; j++) s += th[a * d + j] * X[a * d + j];
            idx[a] = s;
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          // ridge with weight 2η stores A = I + 2η Σxxᵀ and b = 2η Σ r x, so θ_ridge is the posterior mean
          R.add(a, X, a * d, r, two); dirty[a] = 1;
          hist.set(X, head * K * d); head = (head + 1) % HC; if (n < HC) n++; tot++;
        },
      };
    },
  });

  algorithms.push({
    id: 'falcon', name: 'FALCON', family: 'policy',
    params: [{ key: 'c', label: 'γ scale', default: 2, min: 0.05, max: 20, step: 0.05 }],
    formula: 'epoch m: f̂ₘ = least squares on epoch m−1;  pₜ(a) = 1/(K + γₘ(f̂ₘ(x,b) − f̂ₘ(x,a))),  γₘ = c√(K τₘ₋₁)',
    regret: 'O(√(KT log|F|)) with O(log T) offline oracle calls',
    idea: 'Simchi-Levi & Xu\'s FALCON needs only an offline regression oracle, called once per doubling epoch on the previous epoch\'s data; within an epoch it plays inverse-gap weighting on that fixed model. Oracle: per-arm ridge regression.',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      let cur = Ridge(K, d, 1), fit = Ridge(K, d, 1);
      const yhat = new Float64Array(K), p = new Float64Array(K), unit = gapUnit(env);
      let X = null, epochEnd = 2 * K, epochLen = 2 * K, gamma = 0;
      return {
        select(t, Xt) {
          X = view(Xt);
          if (t > epochEnd) {
            const tmp = fit; fit = cur; cur = tmp; cur.reset();
            gamma = params.c * Math.sqrt(K * epochLen) * unit;
            epochLen *= 2; epochEnd = t - 1 + epochLen;
          }
          for (let a = 0; a < K; a++) yhat[a] = fit.predict(a, X, a * d);
          return sampleIndex(igw(yhat, gamma, p, K, rng), rng);
        },
        update(a, r) { cur.add(a, X, a * d, r); },
      };
    },
  });

  algorithms.push({
    id: 'corral', name: 'CORRAL', family: 'policy',
    params: [{ key: 'eta', label: 'η scale', default: 10, min: 0.1, max: 50, step: 0.1 }],
    formula: 'i ~ p̄,  play baseᵢ;  1/p_{t+1,i} = 1/p_{t,i} + ηᵢ(ℓ̂ᵢ − λ) (log-barrier OMD),  ηᵢ ← βηᵢ when 1/p̄ᵢ > ρᵢ',
    regret: 'Õ(√(MT) + M^α T^{1−α} R_i) — inherits the best base\'s rate',
    idea: 'Agarwal et al.\'s CORRAL masters a set of base bandit algorithms (UCB1, a LinUCB-style ridge, EXP3) with log-barrier mirror descent and growing learning rates, so a base that becomes good can win back probability quickly. Simplified: only the chosen base observes the reward (no importance-weighted feedback to bases).',
    create(K, ctx) {
      const center = Centered(ctx.env);
      const { rng, params, env } = ctx, T = Math.max(ctx.T | 0, 1);
      const ids = ['ucb1', 'linucb', 'exp3'].filter((id) => MAB.ALGO_BY_ID[id]);
      const bases = ids.map((id) => MAB.ALGO_BY_ID[id].create(K, Object.assign({}, ctx, { params: MAB.defaultParams(id) })));
      if (!bases.length) bases.push({ select: () => rng.int(K), update() {} });
      const Mb = bases.length, p = new Float64Array(Mb).fill(1 / Mb), pb = new Float64Array(Mb);
      const eta = new Float64Array(Mb).fill(params.eta * Math.sqrt(Mb / T)), rho = new Float64Array(Mb).fill(2 * Mb);
      const beta = Math.exp(1 / Math.log(Math.max(T, 3))), gamma = 1 / T;
      let i = 0;
      function omd(lossI) {
        // find λ with Σ 1/(1/pᵢ + ηᵢ(ℓᵢ − λ)) = 1 by bisection; ℓ is zero except at the chosen base
        const inv = (k, lam) => 1 / p[k] + eta[k] * ((k === i ? lossI : 0) - lam);
        // every denominator must stay positive: λ < min_k ℓ_k + 1/(η_k p_k); Σ increases in λ
        let lo = -1e6, hi = Infinity;
        for (let k = 0; k < Mb; k++) hi = Math.min(hi, (k === i ? lossI : 0) + 1 / (eta[k] * p[k]));
        for (let it = 0; it < 60; it++) {
          const lam = (lo + hi) / 2;
          let s = 0;
          for (let k = 0; k < Mb; k++) { const v = inv(k, lam); s += v > 0 ? 1 / v : Infinity; }
          if (s > 1) hi = lam; else lo = lam;
        }
        let tot = 0;
        for (let k = 0; k < Mb; k++) { const v = inv(k, lo); p[k] = v > 0 ? 1 / v : 1; tot += p[k]; }
        for (let k = 0; k < Mb; k++) p[k] /= tot;
      }
      return {
        select(t, X) {
          for (let k = 0; k < Mb; k++) pb[k] = (1 - gamma) * p[k] + gamma / Mb;
          i = sampleIndex(pb, rng);
          return bases[i].select(t, X);
        },
        update(a, r, t) {
          bases[i].update(a, r, t);
          omd(-center(r) / pb[i]);
          for (let k = 0; k < Mb; k++) {
            const q = (1 - gamma) * p[k] + gamma / Mb;
            if (1 / q > rho[k]) { rho[k] = 2 / q; eta[k] *= beta; }
          }
        },
      };
    },
  });

  // ============================================================== kernel ====

  /**
   * Per-arm RBF kernel ridge regression with an incrementally updated inverse
   * M = (K_a + λI)⁻¹ over a sliding dictionary of the last `cap` points of each arm.
   * Targets are centred on the running mean of all rewards.
   */
  function KernelArms(K, d, cap, ell, lambda) {
    const inv2l2 = 1 / (2 * ell * ell);
    const arms = Array.from({ length: K }, () => ({
      n: 0, X: new Float64Array(cap * d), y: new Float64Array(cap),
      M: new Float64Array(cap * cap), u: new Float64Array(cap), v: new Float64Array(cap),
    }));
    const kv = new Float64Array(cap), Mk = new Float64Array(cap);
    let ySum = 0, yN = 0;
    const kern = (A, ao, B, bo) => {
      let s = 0;
      for (let j = 0; j < d; j++) { const z = A[ao + j] - B[bo + j]; s += z * z; }
      return Math.exp(-s * inv2l2);
    };
    function refresh(S) {
      const n = S.n;
      for (let i = 0; i < n; i++) {
        let su = 0, sv = 0;
        for (let j = 0; j < n; j++) { const m = S.M[i * cap + j]; su += m * S.y[j]; sv += m; }
        S.u[i] = su; S.v[i] = sv;
      }
    }
    function removeOldest(S) {
      const n = S.n, M = S.M, c = M[0];
      for (let j = 1; j < n; j++) Mk[j] = M[j];          // row 0 of the old inverse
      for (let i = 1; i < n; i++) {
        const bi = Mk[i] / c;
        for (let j = 1; j < n; j++) M[(i - 1) * cap + (j - 1)] = M[i * cap + j] - bi * Mk[j];
      }
      S.X.copyWithin(0, d, n * d); S.y.copyWithin(0, 1, n);
      S.n = n - 1;
    }
    return {
      /** Posterior mean and variance (unit-amplitude kernel) of arm a at x. */
      post(a, x, off, out) {
        const S = arms[a], n = S.n, m0 = yN ? ySum / yN : 0;
        let mu = m0, q = 0;
        for (let i = 0; i < n; i++) { kv[i] = kern(S.X, i * d, x, off); mu += kv[i] * (S.u[i] - m0 * S.v[i]); }
        for (let i = 0; i < n; i++) {
          let s = 0;
          const r = i * cap;
          for (let j = 0; j < n; j++) s += S.M[r + j] * kv[j];
          q += kv[i] * s;
        }
        out.mu = mu; out.var = Math.max(0, 1 - q);
        return out;
      },
      add(a, x, off, y) {
        ySum += y; yN++;
        const S = arms[a];
        if (S.n >= cap) removeOldest(S);
        const n = S.n, M = S.M;
        for (let i = 0; i < n; i++) kv[i] = kern(S.X, i * d, x, off);
        let s = 1 + lambda;
        for (let i = 0; i < n; i++) {
          let z = 0;
          for (let j = 0; j < n; j++) z += M[i * cap + j] * kv[j];
          Mk[i] = z; s -= kv[i] * z;
        }
        s = Math.max(s, 1e-9);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i * cap + j] += (Mk[i] * Mk[j]) / s;
        for (let i = 0; i < n; i++) { M[i * cap + n] = -Mk[i] / s; M[n * cap + i] = -Mk[i] / s; }
        M[n * cap + n] = 1 / s;
        for (let j = 0; j < d; j++) S.X[n * d + j] = x[off + j];
        S.y[n] = y; S.n = n + 1;
        refresh(S);
      },
    };
  }

  const kernelParams = [
    { key: 'ell', label: 'ℓ (RBF length-scale)', default: 1, min: 0.1, max: 5, step: 0.05 },
    { key: 'lambda', label: 'λ (regularization)', default: 1, min: 0.01, max: 10, step: 0.01 },
    { key: 'eta', label: 'η (exploration)', default: 1, min: 0, max: 5, step: 0.05 },
    { key: 'cap', label: 'dictionary size per arm', default: 80, min: 10, max: 300, step: 10 },
  ];

  algorithms.push({
    id: 'kernelucb', name: 'KernelUCB', family: 'kernel',
    params: kernelParams,
    formula: 'argmaxₐ k̃ₐᵀ(Kₐ+λI)⁻¹y + (η/√λ)·√(k(x,x) − k̃ₐᵀ(Kₐ+λI)⁻¹k̃ₐ),  k = RBF(ℓ)',
    regret: 'Õ(√(d̃T)) with d̃ the effective dimension of the kernel',
    idea: 'Valko et al.\'s kernelised LinUCB: kernel ridge regression gives the mean, the kernel posterior variance gives the bonus. Here each arm has its own RBF kernel regression, its inverse is updated incrementally (Schur complement) and the dictionary slides once it holds `cap` points, keeping each step O(cap²).',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const KA = KernelArms(K, d, Math.round(params.cap), params.ell, params.lambda);
      const idx = new Float64Array(K), o = { mu: 0, var: 0 }, w = (params.eta * env.sigma) / Math.sqrt(params.lambda);
      let X = null;
      return {
        select(t, Xt) {
          X = view(Xt);
          for (let a = 0; a < K; a++) { KA.post(a, X, a * d, o); idx[a] = o.mu + w * Math.sqrt(o.var); }
          return argmax(idx, rng);
        },
        update(a, r) { KA.add(a, X, a * d, r); },
      };
    },
  });

  algorithms.push({
    id: 'supkernelucb', name: 'SupKernelUCB', family: 'kernel',
    params: kernelParams.map((q) => (q.key === 'eta' ? Object.assign({}, q, { default: 0.5 }) : q)),
    formula: 'stage s: if some wₐ > 2⁻ˢs₀ play it; else keep {a : μ̂ₐ+wₐ ≥ max(μ̂+w) − 2¹⁻ˢs₀}; once all wₐ ≤ s₀/√T play argmax μ̂  (s₀ = reward scale)',
    regret: 'Õ(√(d̃T log³ T)) for the full stage-wise version',
    idea: 'The Sup- wrapper of Valko et al. eliminates arms in stages of halving confidence width, exploring an arm only when its width is still large at the current stage. Simplified: all stages share one per-arm kernel regression (the original keeps independent data per stage for the proof).',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, view = contextView(K, d);
      const T = Math.max(ctx.T | 0, 2);
      const KA = KernelArms(K, d, Math.round(params.cap), params.ell, params.lambda);
      const mu = new Float64Array(K), wd = new Float64Array(K), alive = new Uint8Array(K), o = { mu: 0, var: 0 };
      const wk = (params.eta * env.sigma) / Math.sqrt(params.lambda), w0 = env.scale;   // widths vs. reward scale
      const S = Math.max(1, Math.ceil(Math.log2(Math.sqrt(T))));
      let X = null;
      const pickMax = (vals, mask) => {
        let best = -Infinity, ba = -1, ties = 0;
        for (let a = 0; a < K; a++) {
          if (!mask[a]) continue;
          if (vals[a] > best) { best = vals[a]; ba = a; ties = 1; } else if (vals[a] === best && rng.uniform() * ++ties < 1) ba = a;
        }
        return ba;
      };
      const tmp = new Float64Array(K);
      return {
        select(t, Xt) {
          X = view(Xt);
          for (let a = 0; a < K; a++) { KA.post(a, X, a * d, o); mu[a] = o.mu; wd[a] = wk * Math.sqrt(o.var); }
          alive.fill(1);
          for (let s = 1; s <= S; s++) {
            const thr = w0 * Math.pow(2, -s);
            let allSmall = true, wide = false;
            for (let a = 0; a < K; a++) if (alive[a]) { if (wd[a] > w0 / Math.sqrt(T)) allSmall = false; if (wd[a] > thr) wide = true; }
            if (allSmall) return pickMax(mu, alive);
            if (wide) {
              for (let a = 0; a < K; a++) tmp[a] = alive[a] && wd[a] > thr ? wd[a] : -Infinity;
              return pickMax(tmp, alive);
            }
            let m = -Infinity;
            for (let a = 0; a < K; a++) if (alive[a] && mu[a] + wd[a] > m) m = mu[a] + wd[a];
            for (let a = 0; a < K; a++) if (alive[a] && mu[a] + wd[a] < m - 2 * thr) alive[a] = 0;
          }
          for (let a = 0; a < K; a++) tmp[a] = mu[a] + wd[a];
          return pickMax(tmp, alive);
        },
        update(a, r) { KA.add(a, X, a * d, r); },
      };
    },
  });

  // ============================================================== neural ====

  const pHidden = { key: 'hidden', label: 'hidden units', default: 32, min: 8, max: 64, step: 4 };
  const pLr = { key: 'lr', label: 'learning rate (Adam)', default: 0.01, min: 0.0005, max: 0.1, step: 0.0005 };
  const BATCH = 16, STEPS = 2;

  /** Common scaffolding: one multi-head MLP regressor trained online on a replay buffer. */
  function neuralCore(K, ctx, hidden) {
    const { rng, params, env } = ctx, d = ctx.d || 1, T = Math.max(ctx.T | 0, 1);
    const net = MLP(d, hidden, K, rng);
    const cap = Math.min(T, 4000), buf = Buffer(d, cap);
    const clip = 4 * env.scale, steps0 = d > 12 ? 1 : STEPS;     // wide inputs: one step per round
    return {
      net, buf, d, cap, view: contextView(K, d), clip,
      train(steps = steps0) { trainNet(net, buf, rng, steps, BATCH, params.lr, clip, 1e-4); },
    };
  }

  algorithms.push({
    id: 'neuralucb', name: 'NeuralUCB', family: 'neural',
    params: [pHidden, pLr, { key: 'nu', label: 'ν (exploration)', default: 0.5, min: 0, max: 5, step: 0.05 }],
    formula: 'argmaxₐ f(xₐ; θ) + ν·√(gₐᵀ Z⁻¹ gₐ),   gₐ = ∇θ f(xₐ; θ),  Z ← Z + g gᵀ (diagonal)',
    regret: 'Õ(d̃√T) in the neural-tangent-kernel regime',
    idea: 'Zhou et al.\'s NeuralUCB: a neural network predicts rewards and the gradient at the input plays the role of LinUCB\'s feature vector for the confidence bonus. As in the paper\'s experiments Z is kept diagonal. Network: one ReLU hidden layer with K heads, 2 Adam steps per round.',
    create(K, ctx) {
      const { rng, params, env } = ctx;
      const core = neuralCore(K, ctx, [Math.round(params.hidden)]);
      const { net, buf, d } = core, Z = new Float64Array(net.P).fill(1), idx = new Float64Array(K);
      const nu = params.nu * env.sigma;
      let X = null;
      return {
        select(t, Xt) {
          X = core.view(Xt);
          const w = warmArm(t, K); if (w >= 0) return w;
          for (let a = 0; a < K; a++) {
            const f = net.forward(X, a * d, a);
            idx[a] = f + nu * Math.sqrt(net.gradZ(a, Z, 0));
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          net.forward(X, a * d, a); net.gradZ(a, Z, 1);
          buf.push(X, a * d, a, r);
          core.train();
        },
      };
    },
  });

  algorithms.push({
    id: 'neuralts', name: 'NeuralTS', family: 'neural',
    params: [pHidden, pLr, { key: 'nu', label: 'ν (posterior scale)', default: 0.5, min: 0, max: 5, step: 0.05 }],
    formula: 'r̃ₐ ~ N(f(xₐ; θ), ν²·gₐᵀ Z⁻¹ gₐ),   aₜ = argmaxₐ r̃ₐ',
    regret: 'Õ(d̃√T) in the neural-tangent-kernel regime',
    idea: 'Zhang et al.\'s NeuralTS samples each arm\'s reward from a Gaussian centred on the network\'s prediction with the NeuralUCB width as standard deviation, trading the deterministic bonus for randomized exploration. Diagonal Z, same network as NeuralUCB.',
    create(K, ctx) {
      const { rng, params, env } = ctx;
      const core = neuralCore(K, ctx, [Math.round(params.hidden)]);
      const { net, buf, d } = core, Z = new Float64Array(net.P).fill(1), idx = new Float64Array(K);
      const nu = params.nu * env.sigma;
      let X = null;
      return {
        select(t, Xt) {
          X = core.view(Xt);
          const w = warmArm(t, K); if (w >= 0) return w;
          for (let a = 0; a < K; a++) {
            const f = net.forward(X, a * d, a);
            idx[a] = f + nu * Math.sqrt(net.gradZ(a, Z, 0)) * rng.normal();
          }
          return argmax(idx, rng);
        },
        update(a, r) {
          net.forward(X, a * d, a); net.gradZ(a, Z, 1);
          buf.push(X, a * d, a, r);
          core.train();
        },
      };
    },
  });

  algorithms.push({
    id: 'neural-linucb', name: 'Neural-LinUCB', family: 'neural',
    params: [pLr, { key: 'alpha', label: 'α (exploration)', default: 0.5, min: 0, max: 5, step: 0.05 }],
    formula: 'argmaxₐ θₐᵀφ(xₐ) + α√(φ(xₐ)ᵀAₐ⁻¹φ(xₐ)),   φ = last hidden layer, retrained every H rounds',
    regret: 'Õ(√T) with deep representation and shallow exploration',
    idea: 'Xu et al.\'s Neural-LinUCB learns a representation φ with a deep network but explores only in the last layer: LinUCB runs on φ(x), and every H rounds the network is retrained and the per-arm ridge statistics are rebuilt in the new features. Network: 32→16 ReLU, retrained with 100 Adam steps every H = max(50, t/5) rounds.',
    create(K, ctx) {
      const { rng, params, env } = ctx;
      const core = neuralCore(K, ctx, [32, 16]);
      const { net, buf, d } = core, p = net.hL + 1;
      const R = Ridge(K, p, 1), phi = new Float64Array(K * p), idx = new Float64Array(K);
      const A = new Float64Array(p * p), Lw = new Float64Array(p * p), Ai = new Float64Array(p * p), f1 = new Float64Array(p);
      const alpha = params.alpha * env.sigma;
      let X = null, next = 2 * K;
      const feat = (x, off, a, out, o) => { net.forward(x, off, a); const h = net.last(); for (let j = 0; j < net.hL; j++) out[o + j] = h[j]; out[o + net.hL] = 1; };
      function rebuild() {
        const n = buf.n;
        for (let a = 0; a < K; a++) {
          A.fill(0);
          for (let j = 0; j < p; j++) A[j * p + j] = 1;
          const bo = a * p;
          for (let j = 0; j < p; j++) R.b[bo + j] = 0;
          for (let i = 0; i < n; i++) {
            if (buf.A[i] !== a) continue;
            feat(buf.X, i * d, a, f1, 0);
            const y = buf.Y[i];
            for (let j = 0; j < p; j++) { R.b[bo + j] += y * f1[j]; for (let k = 0; k <= j; k++) A[j * p + k] += f1[j] * f1[k]; }
          }
          for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) A[k * p + j] = A[j * p + k];
          if (invertSPD(A, p, Lw, Ai)) R.Ainv.set(Ai, a * p * p);
          R.solve(a);
        }
      }
      return {
        select(t, Xt) {
          X = core.view(Xt);
          const w = warmArm(t, K); if (w >= 0) return w;
          for (let a = 0; a < K; a++) {
            feat(X, a * d, a, phi, a * p);
            idx[a] = R.predict(a, phi, a * p) + alpha * Math.sqrt(R.quad(a, phi, a * p));
          }
          return argmax(idx, rng);
        },
        update(a, r, t) {
          buf.push(X, a * d, a, r);
          feat(X, a * d, a, phi, a * p);
          R.add(a, phi, a * p, r);
          if (t >= next) {
            core.train(100);
            rebuild();
            next = t + Math.max(50, Math.floor(t / 5));
          }
        },
      };
    },
  });

  algorithms.push({
    id: 'ee-net', name: 'EE-Net', family: 'neural',
    params: [pHidden, pLr],
    formula: 'argmaxₐ f₁(xₐ; θ₁) + f₂(∇θ₁f₁(xₐ); θ₂),   f₂ trained on r − f₁(xₐ)',
    regret: 'Õ(√T) under NTK-style assumptions',
    idea: 'Ban et al.\'s EE-Net uses a second "exploration" network that reads the exploitation network\'s gradient and learns the potential gain r − f₁ directly, instead of a hand-built confidence bonus. Simplified: f₂ sees only the last-layer gradient (the hidden activations of f₁) and the decision maker is the sum f₁ + f₂.',
    create(K, ctx) {
      const { rng, params } = ctx;
      const core = neuralCore(K, ctx, [Math.round(params.hidden)]);
      const { net, buf, d } = core, hL = net.hL, gd = hL + 1;
      const net2 = MLP(gd, [16], K, rng, { outScale: 1 });
      const buf2 = Buffer(gd, buf.X.length / d | 0);
      const g = new Float64Array(K * gd), idx = new Float64Array(K), f1 = new Float64Array(K);
      let X = null;
      const grad = (a) => {
        const h = net.last();
        let nn = 0;
        for (let j = 0; j < hL; j++) nn += h[j] * h[j];
        const s = 1 / Math.sqrt(nn / hL + 1);
        for (let j = 0; j < hL; j++) g[a * gd + j] = h[j] * s;
        g[a * gd + hL] = 1;
      };
      return {
        select(t, Xt) {
          X = core.view(Xt);
          const w = warmArm(t, K);
          for (let a = 0; a < K; a++) {
            f1[a] = net.forward(X, a * d, a);
            grad(a);
            idx[a] = f1[a] + net2.forward(g, a * gd, a);
          }
          return w >= 0 ? w : argmax(idx, rng);
        },
        update(a, r) {
          buf.push(X, a * d, a, r);
          buf2.push(g, a * gd, a, r - f1[a]);
          core.train();
          trainNet(net2, buf2, rng, 1, BATCH, params.lr, core.clip, 1e-4);
        },
      };
    },
  });

  algorithms.push({
    id: 'enn-ts', name: 'Epinet TS', family: 'neural',
    params: [pHidden, pLr, { key: 'prior', label: 'prior scale', default: 1, min: 0, max: 5, step: 0.05 },
      { key: 'noise', label: 'target perturbation (×σ)', default: 0.3, min: 0, max: 2, step: 0.05 }],
    formula: 'f(x, z) = μ(x; θ) + zᵀ[Uₐφ̃(x) + β·Pₐ(x)],  z ~ N(0, I₈);  aₜ = argmaxₐ f(xₐ, z)',
    regret: 'No regret bound; approximates Thompson sampling',
    idea: 'Osband et al.\'s epistemic neural network: a base network plus a small "epinet" that reads a random index z and the base network\'s stop-gradient features φ̃, next to a fixed random prior network P; one z per round gives a posterior-like sample. Simplified: the learnable epinet is linear in φ̃, so the z-averaged ENN loss (with target perturbations σ·cᵢᵀz) splits into one ridge regression per index coordinate, solved in closed form and rebuilt whenever the features are refreshed. The perturbation defaults to 0.3σ: the full σ over-explores in the 33-dimensional feature space.',
    create(K, ctx) {
      const { rng, params, env } = ctx;
      const core = neuralCore(K, ctx, [Math.round(params.hidden)]);
      const { net, buf, d } = core, hL = net.hL, F = hL + 1, DZ = 8, PH = 16;
      const Cz = new Float64Array(core.cap * DZ);
      const R = Ridge(K, F, 1), Bz = new Float64Array(K * DZ * F);
      // fixed random prior network x → tanh(R1 x + c1) → K·DZ outputs
      const R1 = new Float64Array(PH * d), c1 = new Float64Array(PH), R2 = new Float64Array(K * DZ * PH);
      for (let i = 0; i < R1.length; i++) R1[i] = (1.5 / Math.sqrt(d)) * rng.normal();
      for (let i = 0; i < PH; i++) c1[i] = rng.normal();
      for (let i = 0; i < R2.length; i++) R2[i] = rng.normal() / Math.sqrt(PH * DZ);
      const beta = 2 * params.prior * env.sigma, sig = params.noise * env.sigma;
      const z = new Float64Array(DZ), phi = new Float64Array(F), ph = new Float64Array(PH), pk = new Float64Array(DZ);
      const v = new Float64Array(F), idx = new Float64Array(K);
      let X = null, next = 2 * K;
      function features(x, off, a) {          // base forward; fills φ̃ and the prior outputs P_{a,·}(x)
        const mu = net.forward(x, off, a), h = net.last();
        for (let j = 0; j < hL; j++) phi[j] = h[j];
        phi[hL] = 1;
        for (let i = 0; i < PH; i++) { let s = c1[i]; for (let j = 0; j < d; j++) s += R1[i * d + j] * x[off + j]; ph[i] = Math.tanh(s); }
        for (let k = 0; k < DZ; k++) { let s = 0; const ro = (a * DZ + k) * PH; for (let j = 0; j < PH; j++) s += R2[ro + j] * ph[j]; pk[k] = s; }
        return mu;
      }
      function absorb(i, a) {                 // add sample i (features already computed) to arm a's ridge
        R.add(a, phi, 0, 0);
        for (let k = 0; k < DZ; k++) {
          const tgt = sig * Cz[i * DZ + k] - beta * pk[k], o = (a * DZ + k) * F;
          for (let j = 0; j < F; j++) Bz[o + j] += tgt * phi[j];
        }
      }
      function rebuild() {
        R.reset(); Bz.fill(0);
        for (let i = 0; i < buf.n; i++) { features(buf.X, i * d, buf.A[i]); absorb(i, buf.A[i]); }
      }
      return {
        select(t, Xt) {
          X = core.view(Xt);
          const w = warmArm(t, K); if (w >= 0) return w;
          for (let k = 0; k < DZ; k++) z[k] = rng.normal();
          for (let a = 0; a < K; a++) {
            const mu = features(X, a * d, a);
            v.fill(0);
            let prior = 0;
            for (let k = 0; k < DZ; k++) {
              prior += z[k] * pk[k];
              const o = (a * DZ + k) * F;
              for (let j = 0; j < F; j++) v[j] += z[k] * Bz[o + j];
            }
            // zᵀUₐφ̃ = φ̃ᵀAₐ⁻¹(Σₖ zₖ B_{a,k})
            let epi = 0;
            const ao = a * F * F;
            for (let i = 0; i < F; i++) { let s = 0; for (let j = 0; j < F; j++) s += R.Ainv[ao + i * F + j] * v[j]; epi += phi[i] * s; }
            idx[a] = mu + epi + beta * prior;
          }
          return argmax(idx, rng);
        },
        update(a, r, t) {
          const i = buf.push(X, a * d, a, r);
          for (let k = 0; k < DZ; k++) Cz[i * DZ + k] = rng.normal();
          features(X, a * d, a); absorb(i, a);
          core.train();
          if (t >= next) { rebuild(); next = t + Math.max(50, Math.floor(t / 5)); }
        },
      };
    },
  });

  algorithms.push({
    id: 'ensemble-sampling', name: 'Ensemble sampling', family: 'neural',
    params: [
      { key: 'M', label: 'M (ensemble heads)', default: 6, min: 2, max: 20, step: 1 },
      pHidden, pLr,
      { key: 'prior', label: 'prior scale', default: 0.5, min: 0, max: 5, step: 0.05 },
      { key: 'noise', label: 'target perturbation (×σ)', default: 0.3, min: 0, max: 2, step: 0.05 },
    ],
    formula: 'j ~ U{1..M},  aₜ = argmaxₐ fⱼ(xₐ) + β·pⱼ(xₐ);  fⱼ fit to r + σ·ξⱼ,ᵢ (perturbed targets)',
    regret: 'Approximates Thompson sampling; Õ(d√T) for linear ensembles with M ≳ log T',
    idea: 'Lu & Van Roy\'s ensemble sampling in the bootstrapped-DQN form of Osband et al.: M heads on a shared ReLU trunk, each with its own fixed random prior function and its own noise on every target, trained on the shared data; acting greedily on a random head approximates a posterior sample. Perturbations default to 0.3σ (the Bayesian σ over-explores at these horizons).',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, T = Math.max(ctx.T | 0, 1), view = contextView(K, d);
      const M = Math.max(2, Math.round(params.M)), sig = params.noise * env.sigma, beta = params.prior * env.sigma;
      // head (j, a) = output j·K + a of one network; the prior is a fixed random network of the same shape
      const net = MLP(d, [Math.round(params.hidden)], K * M, rng), prior = MLP(d, [16], K * M, rng);
      const buf = Buffer(d, Math.min(T, 4000), M), idx = new Float64Array(K), clip = 4 * env.scale;
      const steps = d > 12 ? 1 : STEPS;
      let X = null;
      return {
        select(t, Xt) {
          X = view(Xt);
          const w = warmArm(t, K); if (w >= 0) return w;
          const j = rng.int(M);
          for (let a = 0; a < K; a++) idx[a] = net.forward(X, a * d, j * K + a) + beta * prior.forward(X, a * d, j * K + a);
          return argmax(idx, rng);
        },
        update(a, r) {
          const i = buf.push(X, a * d, a, r);
          for (let j = 0; j < M; j++) buf.E[i * M + j] = rng.normal();
          const n = buf.n, B = Math.min(BATCH, n);
          for (let s = 0; s < steps; s++) {
            for (let b = 0; b < B; b++) {
              const k = rng.int(n), ak = buf.A[k], j = rng.int(M), head = j * K + ak;
              const y = buf.Y[k] + sig * buf.E[k * M + j] - beta * prior.forward(buf.X, k * d, head);
              net.backward(head, clampAbs(net.forward(buf.X, k * d, head) - y, clip) / B);
            }
            net.adam(params.lr, 1e-4);
          }
        },
      };
    },
  });

  /**
   * Histogram regression tree on rows idx[lo..hi) of (X, y). Leaves carry one random
   * pseudo-observation u: value = (Σy + u)/(n + 1).
   */
  function TreeFitter(d, maxNodes) {
    const feat = new Int32Array(maxNodes), thr = new Float64Array(maxNodes), left = new Int32Array(maxNodes);
    const right = new Int32Array(maxNodes), val = new Float64Array(maxNodes);
    const BINS = 12, hc = new Float64Array(BINS), hs = new Float64Array(BINS);
    let count = 0;
    function grow(X, y, idx, lo, hi, depth, maxDepth, minLeaf, u) {
      const node = count++;
      let s = 0;
      for (let i = lo; i < hi; i++) s += y[idx[i]];
      const n = hi - lo;
      val[node] = (s + u) / (n + 1); feat[node] = -1;
      if (depth >= maxDepth || n < 2 * minLeaf || count + 2 > maxNodes) return node;
      let bestGain = 1e-9, bf = -1, bt = 0;
      const base = (s * s) / n;
      for (let j = 0; j < d; j++) {
        let mn = Infinity, mx = -Infinity;
        for (let i = lo; i < hi; i++) { const v = X[idx[i] * d + j]; if (v < mn) mn = v; if (v > mx) mx = v; }
        if (mx - mn < 1e-12) continue;
        const wdt = (mx - mn) / BINS;
        hc.fill(0); hs.fill(0);
        for (let i = lo; i < hi; i++) {
          const k = Math.min(BINS - 1, Math.floor((X[idx[i] * d + j] - mn) / wdt));
          hc[k]++; hs[k] += y[idx[i]];
        }
        let cl = 0, sl = 0;
        for (let k = 0; k < BINS - 1; k++) {
          cl += hc[k]; sl += hs[k];
          const cr = n - cl;
          if (cl < minLeaf || cr < minLeaf) continue;
          const sr = s - sl, gain = (sl * sl) / cl + (sr * sr) / cr - base;
          if (gain > bestGain) { bestGain = gain; bf = j; bt = mn + (k + 1) * wdt; }
        }
      }
      if (bf < 0) return node;
      let i = lo, k = hi - 1;
      while (i <= k) {
        if (X[idx[i] * d + bf] < bt) i++;
        else { const tmp = idx[i]; idx[i] = idx[k]; idx[k] = tmp; k--; }
      }
      if (i === lo || i === hi) return node;
      feat[node] = bf; thr[node] = bt;
      left[node] = grow(X, y, idx, lo, i, depth + 1, maxDepth, minLeaf, u);
      right[node] = grow(X, y, idx, i, hi, depth + 1, maxDepth, minLeaf, u);
      return node;
    }
    return {
      /** Fit into a fresh tree object {feat, thr, left, right, val}. */
      fit(X, y, idx, n, maxDepth, minLeaf, u) {
        count = 0;
        grow(X, y, idx, 0, n, 0, maxDepth, minLeaf, u);
        return { feat: feat.slice(0, count), thr: thr.slice(0, count), left: left.slice(0, count), right: right.slice(0, count), val: val.slice(0, count) };
      },
    };
  }
  function treePredict(tr, x, off) {
    let k = 0;
    while (tr.feat[k] >= 0) k = x[off + tr.feat[k]] < tr.thr[k] ? tr.left[k] : tr.right[k];
    return tr.val[k];
  }

  algorithms.push({
    id: 'tree-bootstrap', name: 'TreeBootstrap', family: 'neural',
    params: [
      { key: 'depth', label: 'max tree depth', default: 4, min: 1, max: 8, step: 1 },
      { key: 'minLeaf', label: 'min samples per leaf', default: 5, min: 1, max: 50, step: 1 },
    ],
    formula: 'aₜ = argmaxₐ treeₐ(xₐ),   treeₐ = regression tree on a bootstrap resample of arm a\'s data',
    regret: 'No regret bound; approximates Thompson sampling',
    idea: 'Elmachtoub et al.\'s TreeBootstrap: one decision tree per arm, each fit to a bootstrap resample so its prediction behaves like a posterior sample. Here trees are regression trees with histogram splits, refit periodically (every pull while an arm has < 20 points, then every n/20 pulls, plus one stale arm per round) on at most 300 resampled points, with one random pseudo-observation per leaf as a prior.',
    create(K, ctx) {
      const { rng, params, env } = ctx, d = ctx.d || 1, T = Math.max(ctx.T | 0, 1), view = contextView(K, d);
      const cap = Math.min(T, 4000), MAXB = 300;
      const data = Array.from({ length: K }, () => ({ X: new Float64Array(cap * d), y: new Float64Array(cap), n: 0, head: 0, since: 0, tree: null }));
      const fitter = TreeFitter(d, 2 * (1 << Math.min(8, Math.round(params.depth))) + 4);
      const bx = new Float64Array(MAXB * d), by = new Float64Array(MAXB), bi = new Int32Array(MAXB), idx = new Float64Array(K);
      const bern = env.type === 'bernoulli';
      let ySum = 0, yN = 0, X = null;
      function prior() { return bern ? rng.uniform() : (yN ? ySum / yN : 0) + 2 * env.sigma * rng.normal(); }
      function refit(a) {
        const D = data[a], n = D.n, m = Math.min(n, MAXB);
        for (let i = 0; i < m; i++) {
          const k = rng.int(n);
          for (let j = 0; j < d; j++) bx[i * d + j] = D.X[k * d + j];
          by[i] = D.y[k]; bi[i] = i;
        }
        D.tree = fitter.fit(bx, by, bi, m, Math.round(params.depth), Math.round(params.minLeaf), prior());
        D.since = 0;
      }
      return {
        select(t, Xt) {
          X = view(Xt);
          refit((t - 1) % K);                                     // keep every arm's sample fresh
          for (let a = 0; a < K; a++) idx[a] = data[a].tree ? treePredict(data[a].tree, X, a * d) : Infinity;
          return argmax(idx, rng);
        },
        update(a, r) {
          const D = data[a];
          for (let j = 0; j < d; j++) D.X[D.head * d + j] = X[a * d + j];
          D.y[D.head] = r; D.head = (D.head + 1) % cap; if (D.n < cap) D.n++;
          ySum += r; yN++; D.since++;
          if (D.n < 20 || D.since >= D.n / 20) refit(a);
        },
      };
    },
  });

  MAB.register({ families, environments, algorithms });
  return { environments, algorithms, families };
});
