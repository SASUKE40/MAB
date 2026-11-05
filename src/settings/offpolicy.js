/*
 * offpolicy.js — setting 'offpolicy': off-policy evaluation and offline policy learning from
 * logged contextual-bandit data.
 *
 * Each run draws a synthetic contextual bandit with a finite pool of P context vectors (so every
 * true value is exact), a logging (behaviour) policy π₀ and a target policy π, and logs n_max
 * rounds (x, a ~ π₀(·|x), π₀(a|x), r). Every method sees the same logged rounds (common random
 * numbers) and is scored on nested prefixes n = n₀ … n_max (the curve x-axis), refitting its
 * models only at those grid points.
 *
 * Two kinds of methods share one set of metrics:
 *   kind 'eval'  — estimators of V(π), the value of the fixed target policy.
 *                  loss = RMSE of V̂ across runs, bias = mean(V̂ − V), sd = spread of V̂ − V.
 *   kind 'learn' — learners that output a policy π̂ from the logged data.
 *                  loss = mean suboptimality V(π*) − V(π̂) (true values, exact over the pool),
 *                  bias = mean(V̂(π̂) − V(π̂)): how far the learner's own estimate of its
 *                  policy's value is off (greedy methods are optimistic, pessimistic ones are not),
 *                  sd = spread of the suboptimality.
 * Both losses are in reward units and lower is better; the presets keep the two kinds apart.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;

  // ------------------------------------------------------- linear algebra ----
  /** In-place Cholesky factorisation (lower triangle) of a symmetric positive definite d×d matrix. */
  function cholesky(M, d) {
    for (let j = 0; j < d; j++) {
      let s = M[j * d + j];
      for (let k = 0; k < j; k++) s -= M[j * d + k] * M[j * d + k];
      const l = Math.sqrt(Math.max(s, 1e-12));
      M[j * d + j] = l;
      for (let i = j + 1; i < d; i++) {
        let t = M[i * d + j];
        for (let k = 0; k < j; k++) t -= M[i * d + k] * M[j * d + k];
        M[i * d + j] = t / l;
      }
    }
  }
  /** Solve L Lᵀ x = b, writing x into out[oo..oo+d). */
  function cholSolve(L, d, b, bo, out, oo, tmp) {
    for (let i = 0; i < d; i++) {
      let s = b[bo + i];
      for (let k = 0; k < i; k++) s -= L[i * d + k] * tmp[k];
      tmp[i] = s / L[i * d + i];
    }
    for (let i = d - 1; i >= 0; i--) {
      let s = tmp[i];
      for (let k = i + 1; k < d; k++) s -= L[k * d + i] * out[oo + k];
      out[oo + i] = s / L[i * d + i];
    }
  }

  /** Per-action (weighted) ridge regression with incremental sufficient statistics. */
  function ArmRidge(K, d) {
    const A = new Float64Array(K * d * d), b = new Float64Array(K * d);
    const wsum = new Float64Array(K), cnt = new Int32Array(K);
    const theta = new Float64Array(K * d), Ainv = new Float64Array(K * d * d);
    const L = new Float64Array(d * d), tmp = new Float64Array(d), e = new Float64Array(d), col = new Float64Array(d);
    return {
      theta, Ainv, cnt,
      add(X, xo, a, r, w) {
        const o = a * d * d;
        for (let i = 0; i < d; i++) {
          const wx = w * X[xo + i];
          for (let j = 0; j < d; j++) A[o + i * d + j] += wx * X[xo + j];
          b[a * d + i] += wx * r;
        }
        wsum[a] += w; cnt[a]++;
      },
      /** θₐ = (λ·w̄ₐ I + Aₐ)⁻¹ bₐ; λ is scaled by the mean weight so weighted fits regularise alike. */
      fit(lam, inv) {
        for (let a = 0; a < K; a++) {
          const reg = lam * (cnt[a] && wsum[a] > 0 ? wsum[a] / cnt[a] : 1);
          const o = a * d * d;
          for (let i = 0; i < d * d; i++) L[i] = A[o + i];
          for (let i = 0; i < d; i++) L[i * d + i] += reg;
          cholesky(L, d);
          cholSolve(L, d, b, a * d, theta, a * d, tmp);
          if (inv) {
            for (let j = 0; j < d; j++) {
              e.fill(0); e[j] = 1;
              cholSolve(L, d, e, 0, col, 0, tmp);
              for (let i = 0; i < d; i++) Ainv[o + i * d + j] = col[i];
            }
          }
        }
      },
    };
  }

  function dot(X, xo, Y, yo, d) {
    let s = 0;
    for (let j = 0; j < d; j++) s += X[xo + j] * Y[yo + j];
    return s;
  }
  /** q[p*K + a] = x_p · θ_a over the whole context pool. */
  function predictPool(data, theta, q) {
    const { K, d, P, X } = data;
    for (let p = 0; p < P; p++) for (let a = 0; a < K; a++) q[p * K + a] = dot(X, p * d, theta, a * d, d);
    return q;
  }
  /** Greedy (one-hot) policy over the pool from a P×K score table; ties go to the lowest index. */
  function greedyPolicy(score, P, K, pol) {
    pol.fill(0);
    for (let p = 0; p < P; p++) {
      let b = 0;
      for (let a = 1; a < K; a++) if (score[p * K + a] > score[p * K + b]) b = a;
      pol[p * K + b] = 1;
    }
    return pol;
  }
  /** Mean over the first n logged contexts of maxₐ score(x_i, a): a learner's own value estimate. */
  function meanMaxLogged(data, score, n) {
    const { K, cx } = data;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const o = cx[i] * K;
      let m = -Infinity;
      for (let a = 0; a < K; a++) if (score[o + a] > m) m = score[o + a];
      s += m;
    }
    return s / n;
  }

  // -------------------------------------------------------- environments ----
  /*
   * Mean rewards: x = (1, x₁ … x_{d−1}), xⱼ ~ N(0,1), θₐ ~ N(0, I/d), zₐ = x·θₐ.
   *   linear:    μ(x,a) = 1 + 0.5·zₐ
   *   nonlinear: μ(x,a) = 1 + 0.6·sin(2zₐ) + 0.4·cₐ·x₁·x_{d−1}
   * Rewards r = μ + σ·ε. Policies: softmax(τ · s/sd(s)) mixed with ε/K, where s = x·(θₐ + ν·ξₐ)
   * are noisy linear scores (ν = how far the policy is from the truth's linear part).
   */
  const OFFSET = 1;   // rewards are centred near 1, not 0, as most logged rewards are positive
  const ENVS = [
    {
      id: 'linear', name: 'Linear, good overlap', short: 'Linear',
      description: 'Linear mean rewards, so ridge reward models are well specified. The logging policy is a soft, noisy version of the truth; the target policy is sharper and closer to it.',
      tags: ['well specified', 'good overlap'],
      sigma: 0.5, nonlinear: false,
      logging: { tau: 1, nu: 1, eps: 0.05 }, target: { tau: 2.5, nu: 0.3, eps: 0 },
    },
    {
      id: 'misspecified', name: 'Non-linear rewards (misspecified model)', short: 'Misspecified',
      description: 'Mean rewards are a sine of a linear score plus an interaction term, so every linear reward model is biased. Overlap is as in the linear case.',
      tags: ['misspecified', 'good overlap'],
      sigma: 0.5, nonlinear: true,
      logging: { tau: 1, nu: 1, eps: 0.05 }, target: { tau: 2.5, nu: 0.3, eps: 0 },
    },
    {
      id: 'lowoverlap', name: 'Low overlap (near-deterministic logging)', short: 'Low overlap',
      description: 'The logging policy is almost deterministic (temperature 5, 5% uniform floor), and the target policy often prefers other actions, so importance weights reach about 20·K and many actions are rarely logged.',
      tags: ['low overlap', 'well specified'],
      sigma: 0.5, nonlinear: false,
      logging: { tau: 5, nu: 0.8, eps: 0.05 }, target: { tau: 2.5, nu: 0.3, eps: 0 },
    },
    {
      id: 'small', name: 'Small sample (n ≤ 150, noisy)', short: 'Small n',
      description: 'Linear rewards with twice the noise and at most 150 logged rounds (whatever the T control says), grid from n = 10: shows which methods degrade gracefully when data are scarce.',
      tags: ['small sample', 'well specified'],
      sigma: 1, nonlinear: false, nCap: 150,
      logging: { tau: 1, nu: 1, eps: 0.05 }, target: { tau: 2.5, nu: 0.3, eps: 0 },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of ENVS) ENV_BY_ID[e.id] = e;

  /** Draw one problem instance and its logged data. Truth (mu, values) is kept apart from `data`. */
  function makeInstance(env, K, d, P, nMax, rng) {
    const X = new Float64Array(P * d);
    for (let p = 0; p < P; p++) {
      X[p * d] = 1;
      for (let j = 1; j < d; j++) X[p * d + j] = rng.normal();
    }
    const sd = 1 / Math.sqrt(d);
    const theta = new Float64Array(K * d), coef = new Float64Array(K);
    for (let i = 0; i < K * d; i++) theta[i] = rng.normal() * sd;
    for (let a = 0; a < K; a++) coef[a] = rng.normal();
    const mu = new Float64Array(P * K);
    for (let p = 0; p < P; p++) {
      for (let a = 0; a < K; a++) {
        const z = dot(X, p * d, theta, a * d, d);
        mu[p * K + a] = OFFSET + (env.nonlinear ? 0.6 * Math.sin(2 * z) + 0.4 * coef[a] * X[p * d + 1] * X[p * d + d - 1] : 0.5 * z);
      }
    }
    function policy({ tau, nu, eps }) {
      const w = new Float64Array(K * d);
      for (let i = 0; i < K * d; i++) w[i] = theta[i] + nu * rng.normal() * sd;
      const s = new Float64Array(P * K);
      let m = 0, m2 = 0;
      for (let p = 0; p < P; p++) for (let a = 0; a < K; a++) {
        const v = dot(X, p * d, w, a * d, d);
        s[p * K + a] = v; m += v; m2 += v * v;
      }
      m /= P * K;
      const scale = tau / Math.sqrt(Math.max(1e-12, m2 / (P * K) - m * m));
      const pol = new Float64Array(P * K);
      for (let p = 0; p < P; p++) {
        const row = pol.subarray(p * K, p * K + K);
        util.softmaxInto(row, s.subarray(p * K, p * K + K), scale);
        for (let a = 0; a < K; a++) row[a] = (1 - eps) * row[a] + eps / K;
      }
      return pol;
    }
    const pb = policy(env.logging), pe = policy(env.target);
    let Ve = 0, Vstar = 0, Vb = 0;
    for (let p = 0; p < P; p++) {
      let best = -Infinity;
      for (let a = 0; a < K; a++) {
        const m = mu[p * K + a];
        Ve += pe[p * K + a] * m; Vb += pb[p * K + a] * m;
        if (m > best) best = m;
      }
      Vstar += best;
    }
    const cx = new Int32Array(nMax), act = new Int32Array(nMax);
    const prop = new Float64Array(nMax), rew = new Float64Array(nMax), w = new Float64Array(nMax);
    for (let i = 0; i < nMax; i++) {
      const c = rng.int(P);
      const a = util.sampleIndex(pb.subarray(c * K, c * K + K), rng);
      cx[i] = c; act[i] = a; prop[i] = pb[c * K + a];
      rew[i] = mu[c * K + a] + env.sigma * rng.normal();
      w[i] = pe[c * K + a] / prop[i];
    }
    return {
      data: { K, d, P, nMax, X, pb, pe, cx, act, prop, rew, w },
      mu, Ve: Ve / P, Vstar: Vstar / P, Vb: Vb / P,
    };
  }

  // ---------------------------------------------------------- estimators ----
  /** Ridge reward model q̂(x,a) on the first n logged rounds, optionally weighted. */
  function rewardModel(data, lam, weight) {
    const { K, d, P, X, cx, act, rew } = data;
    const R = ArmRidge(K, d), q = new Float64Array(P * K), qe = new Float64Array(P);
    let upto = 0;
    return {
      q, qe,
      fit(n) {
        for (; upto < n; upto++) R.add(X, cx[upto] * d, act[upto], rew[upto], weight ? weight(upto) : 1);
        R.fit(lam, false);
        predictPool(data, R.theta, q);
        for (let p = 0; p < P; p++) {
          let s = 0;
          for (let a = 0; a < K; a++) s += data.pe[p * K + a] * q[p * K + a];
          qe[p] = s;   // Σₐ π(a|x) q̂(x,a)
        }
      },
    };
  }

  function drEstimate(data, model, n) {
    const { K, cx, act, rew, w } = data;
    let s = 0;
    for (let i = 0; i < n; i++) s += model.qe[cx[i]] + w[i] * (rew[i] - model.q[cx[i] * K + act[i]]);
    return s / n;
  }

  const lamParam = { key: 'lambda', label: 'Ridge λ', default: 1, min: 0.01, max: 20, step: 0.01 };
  const clipParam = { key: 'M', label: 'Clip M', default: 10, min: 1, max: 200, step: 1 };

  const ESTIMATORS = [
    {
      id: 'ips', name: 'IPS', family: 'ips', kind: 'eval', params: [],
      formula: 'V̂ = (1/n) Σᵢ wᵢ rᵢ,  wᵢ = π(aᵢ|xᵢ) / π₀(aᵢ|xᵢ)',
      regret: 'Unbiased; variance ∝ E[w² r²] / n, explodes when overlap is poor',
      idea: 'Estimator. Reweights each logged reward by how much more often the target policy would have taken that action. No model, no bias, but large weights make it noisy.',
      create(data) {
        const { rew, w } = data;
        return { estimate(n) { let s = 0; for (let i = 0; i < n; i++) s += w[i] * rew[i]; return s / n; } };
      },
    },
    {
      id: 'snips', name: 'SNIPS', family: 'ips', kind: 'eval', params: [],
      formula: 'V̂ = Σᵢ wᵢ rᵢ / Σᵢ wᵢ',
      regret: 'Bias O(1/n); usually much lower variance than IPS; invariant to reward shifts',
      idea: 'Estimator. Self-normalised IPS (Swaminathan & Joachims 2015): divides by the realised sum of weights instead of n, trading a small bias for stability.',
      create(data) {
        const { rew, w } = data;
        return {
          estimate(n) {
            let s = 0, sw = 0;
            for (let i = 0; i < n; i++) { s += w[i] * rew[i]; sw += w[i]; }
            return sw > 0 ? s / sw : 0;
          },
        };
      },
    },
    {
      id: 'cips', name: 'Clipped IPS', family: 'ips', kind: 'eval', params: [clipParam],
      formula: 'V̂ = (1/n) Σᵢ min(wᵢ, M) rᵢ',
      regret: 'Bias ≤ E[(w − M)₊ |r|]; variance ≤ M² E[r²] / n',
      idea: 'Estimator. IPS with importance weights capped at M (Bottou et al. 2013): bounded variance, at the price of a bias towards the logging policy.',
      create(data, { params }) {
        const { rew, w } = data, M = params.M;
        return { estimate(n) { let s = 0; for (let i = 0; i < n; i++) s += Math.min(w[i], M) * rew[i]; return s / n; } };
      },
    },
    {
      id: 'dm', name: 'DM (ridge)', family: 'model-based', kind: 'eval', params: [lamParam],
      formula: 'V̂ = (1/n) Σᵢ Σₐ π(a|xᵢ) q̂(xᵢ,a),  q̂ = ridge regression per action',
      regret: 'Low variance; bias = reward-model error, which does not vanish if the model is misspecified',
      idea: 'Estimator. Direct method: fit a reward model on the logged data, then average its predictions under the target policy. Ignores the propensities entirely.',
      create(data, { params }) {
        const model = rewardModel(data, params.lambda);
        return {
          estimate(n) {
            model.fit(n);
            let s = 0;
            for (let i = 0; i < n; i++) s += model.qe[data.cx[i]];
            return s / n;
          },
        };
      },
    },
    {
      id: 'dr', name: 'DR', family: 'doubly-robust', kind: 'eval', params: [lamParam],
      formula: 'V̂ = (1/n) Σᵢ [ Σₐ π(a|xᵢ) q̂(xᵢ,a) + wᵢ (rᵢ − q̂(xᵢ,aᵢ)) ]',
      regret: 'Unbiased with known π₀; variance ∝ E[w² (r − q̂)²] / n, small when q̂ is good',
      idea: 'Estimator. Doubly robust (Dudík, Langford & Li 2011): DM plus an importance-weighted correction of the model residual. Accurate if either the model or the propensities are right.',
      create(data, { params }) {
        const model = rewardModel(data, params.lambda);
        return { estimate(n) { model.fit(n); return drEstimate(data, model, n); } };
      },
    },
    {
      id: 'mrdr', name: 'MRDR', family: 'doubly-robust', kind: 'eval', params: [lamParam],
      formula: 'DR with q̂ = argmin Σᵢ π(aᵢ|xᵢ)(1 − π₀ᵢ)/π₀ᵢ² · (rᵢ − q(xᵢ,aᵢ))²',
      regret: 'Unbiased; minimises the DR variance within the model class',
      idea: 'Estimator. More robust DR (Farajtabar, Chow & Ghavamzadeh 2018): the reward model is fitted to minimise the variance of DR rather than the squared error. Simplified: the per-sample weights of the deterministic-target derivation (as in Open Bandit Pipeline), weighted ridge.',
      create(data, { params }) {
        const { K, cx, act, prop, pe } = data;
        const model = rewardModel(data, params.lambda, (i) => (pe[cx[i] * K + act[i]] * (1 - prop[i])) / (prop[i] * prop[i]));
        return { estimate(n) { model.fit(n); return drEstimate(data, model, n); } };
      },
    },
    {
      id: 'switch-dr', name: 'SWITCH-DR', family: 'doubly-robust', kind: 'eval',
      params: [lamParam, { key: 'tau', label: 'Threshold τ (0 = tuned)', default: 0, min: 0, max: 200, step: 1 }],
      formula: 'V̂ = (1/n) Σᵢ [ Σₐ π(a|xᵢ) q̂(xᵢ,a) + 1{wᵢ ≤ τ} wᵢ (rᵢ − q̂(xᵢ,aᵢ)) ]',
      regret: 'Minimax-optimal MSE up to constants (Wang, Agarwal & Dudík 2017)',
      idea: 'Estimator. Uses DR where the importance weight is at most τ and the model alone where it is larger. With τ = 0 the threshold is picked from a grid by minimising estimated variance + (σ̂ · target mass above τ)², a simplified version of the paper\'s tuning.',
      create(data, { params }) {
        const { K, cx, act, rew, w, pe, pb } = data;
        const model = rewardModel(data, params.lambda);
        const grid = [1, 2, 3, 5, 10, 20, 50, 100, Infinity];
        function value(n, tau, wantVar) {
          let s = 0, s2 = 0;
          for (let i = 0; i < n; i++) {
            const res = rew[i] - model.q[cx[i] * K + act[i]];
            const y = model.qe[cx[i]] + (w[i] <= tau ? w[i] * res : 0);
            s += y; s2 += y * y;
          }
          const m = s / n;
          return wantVar ? { m, v: Math.max(0, s2 / n - m * m) / n } : m;
        }
        return {
          estimate(n) {
            model.fit(n);
            if (params.tau > 0) return value(n, params.tau, false);
            let rss = 0;
            for (let i = 0; i < n; i++) { const e = rew[i] - model.q[cx[i] * K + act[i]]; rss += e * e; }
            const sig = Math.sqrt(rss / n);
            let best = Infinity, out = 0;
            for (const tau of grid) {
              const { m, v } = value(n, tau, true);
              let mass = 0;   // target probability on actions whose weight exceeds τ
              for (let i = 0; i < n; i++) {
                const o = cx[i] * K;
                for (let a = 0; a < K; a++) if (pe[o + a] > tau * pb[o + a]) mass += pe[o + a];
              }
              const b = (sig * mass) / n, mse = v + b * b;
              if (mse < best) { best = mse; out = m; }
            }
            return out;
          },
        };
      },
    },
    {
      id: 'cab', name: 'CAB', family: 'doubly-robust', kind: 'eval', params: [lamParam, { ...clipParam, default: 2 }],
      formula: 'V̂ = (1/n) Σᵢ [ min(wᵢ, M) rᵢ + Σₐ (π(a|xᵢ) − M π₀(a|xᵢ))₊ q̂(xᵢ,a) ]',
      regret: 'Bias only from q̂ on the clipped mass; variance ≤ M² / n; sub-differentiable in π',
      idea: 'Estimator. Continuous adaptive blending (Su, Wang, Srinath & Joachims 2019): clipped IPS, with the probability mass removed by clipping filled in by the reward model. Blends IPS and DM per action rather than switching.',
      create(data, { params }) {
        const { K, P, cx, rew, w, pe, pb } = data, M = params.M;
        const model = rewardModel(data, params.lambda), fill = new Float64Array(P);
        return {
          estimate(n) {
            model.fit(n);
            for (let p = 0; p < P; p++) {
              let s = 0;
              for (let a = 0; a < K; a++) {
                const ex = pe[p * K + a] - M * pb[p * K + a];
                if (ex > 0) s += ex * model.q[p * K + a];
              }
              fill[p] = s;
            }
            let s = 0;
            for (let i = 0; i < n; i++) s += Math.min(w[i], M) * rew[i] + fill[cx[i]];
            return s / n;
          },
        };
      },
    },
  ];

  // ------------------------------------------------------------ learners ----
  const betaParam = { key: 'beta', label: 'Pessimism β', default: 1, min: 0, max: 5, step: 0.05 };

  /** Ridge per action, greedy w.r.t. q̂ − β σ̂ ‖x‖_{Aₐ⁻¹} (β = 0: DM-greedy). */
  function linLcb(data, lam, beta) {
    const { K, d, P, X, cx, act, rew } = data;
    const R = ArmRidge(K, d), q = new Float64Array(P * K), score = new Float64Array(P * K);
    const pol = new Float64Array(P * K);
    let upto = 0;
    return {
      learn(n) {
        for (; upto < n; upto++) R.add(X, cx[upto] * d, act[upto], rew[upto], 1);
        R.fit(lam, beta > 0);
        predictPool(data, R.theta, q);
        if (beta > 0) {
          let rss = 0;
          for (let i = 0; i < n; i++) { const e = rew[i] - q[cx[i] * K + act[i]]; rss += e * e; }
          const sig = Math.max(0.05, Math.sqrt(rss / Math.max(n / 2, n - K * d)));
          for (let p = 0; p < P; p++) {
            for (let a = 0; a < K; a++) {
              const o = a * d * d;
              let w2 = 0;
              for (let i = 0; i < d; i++) {
                let t = 0;
                for (let j = 0; j < d; j++) t += R.Ainv[o + i * d + j] * X[p * d + j];
                w2 += X[p * d + i] * t;
              }
              score[p * K + a] = q[p * K + a] - beta * sig * Math.sqrt(Math.max(0, w2));
            }
          }
        } else score.set(q);
        return { policy: greedyPolicy(score, P, K, pol), estimate: meanMaxLogged(data, score, n) };
      },
    };
  }

  /*
   * Counterfactual risk minimisation over softmax-linear policies π_W(a|x) ∝ exp(W_a·x):
   * maximise  mean(uᵢ) − λ √(var(u)/n) − (ρ/2)‖W‖²,  uᵢ = min(π_W(aᵢ|xᵢ)/π₀ᵢ, M) (rᵢ − r̄).
   * r̄ (mean logged reward) is a fixed baseline / control variate. Full-batch Adam on the exact
   * gradient, warm-started from the previous grid point.
   */
  function crm(data, { M, lambda, rho, iters, lr }) {
    const { K, d, P, X, cx, act, prop, rew } = data;
    const W = new Float64Array(K * d), G = new Float64Array(K * d);
    const m1 = new Float64Array(K * d), m2 = new Float64Array(K * d);
    const pol = new Float64Array(P * K), C = new Float64Array(P * K);
    const u = new Float64Array(data.nMax), om = new Float64Array(data.nMax);
    let step = 0;
    // Contexts come from the finite pool, so π_W is computed once per pool context per step and
    // per-sample gradient coefficients are aggregated by context before touching the features.
    function policyOnPool() {
      for (let p = 0; p < P; p++) {
        const o = p * K;
        let mx = -Infinity;
        for (let a = 0; a < K; a++) { const v = dot(X, p * d, W, a * d, d); pol[o + a] = v; if (v > mx) mx = v; }
        let z = 0;
        for (let a = 0; a < K; a++) { const e = Math.exp(pol[o + a] - mx); pol[o + a] = e; z += e; }
        for (let a = 0; a < K; a++) pol[o + a] /= z;
      }
    }
    function forward(n, base) {
      policyOnPool();
      let s = 0;
      for (let i = 0; i < n; i++) {
        const o = pol[cx[i] * K + act[i]] / prop[i];
        om[i] = o;                                  // unclipped ratio (clipping tested against M)
        u[i] = Math.min(o, M) * (rew[i] - base);
        s += u[i];
      }
      return s / n;
    }
    return {
      learn(n) {
        let base = 0;
        for (let i = 0; i < n; i++) base += rew[i];
        base /= n;
        const sqn = Math.sqrt(n), n1 = Math.max(1, n - 1);
        for (let it = 0; it < iters; it++) {
          const mean = forward(n, base);
          let v = 0;
          for (let i = 0; i < n; i++) v += (u[i] - mean) * (u[i] - mean);
          const sdv = Math.sqrt(v / n1) + 1e-9;
          C.fill(0);
          for (let i = 0; i < n; i++) {
            if (om[i] >= M) continue;               // clipped: no gradient
            const dJdu = 1 / n - ((lambda / sqn) * (u[i] - mean)) / (n1 * sdv);
            const g = dJdu * (rew[i] - base) * om[i], o = cx[i] * K;
            // ∂π(aᵢ|x)/∂W_b = π(aᵢ|x)(1{b = aᵢ} − π(b|x)) x; om already carries π(aᵢ|x)/π₀ᵢ
            for (let b = 0; b < K; b++) C[o + b] -= g * pol[o + b];
            C[o + act[i]] += g;
          }
          G.fill(0);
          for (let p = 0; p < P; p++) {
            for (let b = 0; b < K; b++) {
              const c = C[p * K + b];
              if (c !== 0) for (let j = 0; j < d; j++) G[b * d + j] += c * X[p * d + j];
            }
          }
          step++;
          const c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
          for (let k = 0; k < K * d; k++) {
            const g = G[k] - rho * W[k];
            m1[k] = 0.9 * m1[k] + 0.1 * g;
            m2[k] = 0.999 * m2[k] + 0.001 * g * g;
            W[k] += (lr * (m1[k] / c1)) / (Math.sqrt(m2[k] / c2) + 1e-8);
          }
        }
        const est = forward(n, base) + base;      // also leaves π_W on the pool in `pol`
        return { policy: pol, estimate: est };
      },
    };
  }

  const LEARNERS = [
    {
      id: 'dm-greedy', name: 'DM-greedy', family: 'learning', kind: 'learn', params: [lamParam],
      formula: 'π̂(x) = argmaxₐ q̂(x,a),  q̂ = ridge regression per action',
      regret: 'Consistent only with a well-specified model and full coverage; overestimates its own value',
      idea: 'Learner. Plug-in baseline: fit the reward model and act greedily on it. Rarely logged actions with lucky estimates win the argmax (the winner\'s curse).',
      create(data, { params }) { return linLcb(data, params.lambda, 0); },
    },
    {
      id: 'ips-greedy', name: 'IPS-greedy', family: 'learning', kind: 'learn', params: [lamParam],
      formula: 'π̂(x) = argmaxₐ x·θ̂ₐ,  θ̂ₐ = ridge fit of x ↦ rᵢ 1{aᵢ = a} / π₀ᵢ',
      regret: 'Unbiased reward targets but no variance control: unreliable under poor overlap',
      idea: 'Learner. Regresses importance-weighted pseudo-rewards (unbiased for every action) on the context and acts greedily. Its own value estimate is the IPS estimate of the greedy policy.',
      create(data, { params }) {
        const { K, d, P, X, cx, act, prop, rew } = data;
        const A = new Float64Array(d * d), b = new Float64Array(K * d), L = new Float64Array(d * d), tmp = new Float64Array(d);
        const theta = new Float64Array(K * d), q = new Float64Array(P * K), pol = new Float64Array(P * K);
        let upto = 0;
        return {
          learn(n) {
            for (; upto < n; upto++) {
              const xo = cx[upto] * d, z = rew[upto] / prop[upto];
              for (let i = 0; i < d; i++) {
                for (let j = 0; j < d; j++) A[i * d + j] += X[xo + i] * X[xo + j];
                b[act[upto] * d + i] += z * X[xo + i];
              }
            }
            L.set(A);
            for (let i = 0; i < d; i++) L[i * d + i] += params.lambda;
            cholesky(L, d);
            for (let a = 0; a < K; a++) cholSolve(L, d, b, a * d, theta, a * d, tmp);
            greedyPolicy(predictPool(data, theta, q), P, K, pol);
            let s = 0;
            for (let i = 0; i < n; i++) if (pol[cx[i] * K + act[i]] === 1) s += rew[i] / prop[i];
            return { policy: pol, estimate: s / n };
          },
        };
      },
    },
    {
      id: 'poem', name: 'POEM', family: 'learning', kind: 'learn',
      params: [
        { key: 'lambda', label: 'Variance penalty λ', default: 1, min: 0, max: 10, step: 0.05 },
        clipParam,
        { key: 'rho', label: 'L2 ρ', default: 0.001, min: 0, max: 1, step: 0.001 },
        { key: 'iters', label: 'Steps per refit', default: 40, min: 5, max: 200, step: 5 },
      ],
      formula: 'max_W  R̂ᴹ(π_W) − λ √(V̂ar/n) − (ρ/2)‖W‖²,  π_W(a|x) ∝ exp(W_a·x)',
      regret: 'CRM bound: R(π) ≥ R̂ᴹ(π) − O(√(V̂ar·C/n) + C/n) uniformly over the class',
      idea: 'Learner. Policy optimiser for exponential models (Swaminathan & Joachims 2015): counterfactual risk minimisation, i.e. clipped IPS with a sample-variance penalty, on a softmax-linear policy. Simplified: full-batch Adam on the exact objective instead of the majorise-minimise SGD.',
      create(data, { params }) {
        return crm(data, { M: params.M, lambda: params.lambda, rho: params.rho, iters: params.iters, lr: 0.1 });
      },
    },
    {
      id: 'svp', name: 'SVP', family: 'learning', kind: 'learn',
      params: [
        { key: 'lambda', label: 'Variance penalty λ', default: 1, min: 0, max: 10, step: 0.05 },
        { key: 'iters', label: 'Steps per refit', default: 40, min: 5, max: 200, step: 5 },
      ],
      formula: 'max_π  R̂_IPS(π) − λ √(V̂ar(π)/n)',
      regret: 'Excess risk O(√(Var*·C/n) + C/n) (Maurer & Pontil 2009)',
      idea: 'Learner. Plain sample-variance penalisation (Maurer & Pontil 2009) of the unclipped IPS objective over the same softmax-linear class. POEM is SVP applied to clipped IPS with L2; the two coincide when M = ∞ and ρ = 0, so the difference here isolates clipping and regularisation.',
      create(data, { params }) {
        return crm(data, { M: Infinity, lambda: params.lambda, rho: 0, iters: params.iters, lr: 0.1 });
      },
    },
    {
      id: 'pevi', name: 'PEVI / LCB (tabular)', family: 'pessimism', kind: 'learn',
      params: [betaParam],
      formula: 'π̂(x) = argmaxₐ r̄(x,a) − β σ̂ / √n(x,a)',
      regret: 'V* − V(π̂) = Õ(√(C*·|X| / n)) under single-policy coverage C* (Rashidinejad et al. 2021)',
      idea: 'Learner. Pessimism (PEVI for one step, Jin, Yang & Wang 2021; LCB, Rashidinejad et al. 2021) on the tabular problem over the finite context pool: act on lower confidence bounds, so unlogged actions are never chosen. Contexts never logged fall back to the context-free K-armed LCB.',
      create(data, { params }) {
        const { K, P, cx, act, rew } = data;
        const cnt = new Float64Array(P * K), sum = new Float64Array(P * K);
        const acnt = new Float64Array(K), asum = new Float64Array(K);
        const score = new Float64Array(P * K), pol = new Float64Array(P * K);
        let upto = 0, s1 = 0, s2 = 0;
        return {
          learn(n) {
            for (; upto < n; upto++) {
              const k = cx[upto] * K + act[upto], r = rew[upto];
              cnt[k]++; sum[k] += r; acnt[act[upto]]++; asum[act[upto]] += r; s1 += r; s2 += r * r;
            }
            // σ̂: pooled within-cell residual SD, falling back to the overall SD.
            let rss = 0, cells = 0;
            for (let i = 0; i < n; i++) { const k = cx[i] * K + act[i], e = rew[i] - sum[k] / cnt[k]; rss += e * e; }
            for (let k = 0; k < P * K; k++) if (cnt[k] > 0) cells++;
            const overall = Math.sqrt(Math.max(0, s2 / n - (s1 / n) * (s1 / n)));
            const sig = Math.max(0.05, n - cells >= 5 ? Math.sqrt(rss / (n - cells)) : overall);
            let fb = 0, fbv = -Infinity;
            for (let a = 0; a < K; a++) {
              if (!acnt[a]) continue;
              const v = asum[a] / acnt[a] - (params.beta * sig) / Math.sqrt(acnt[a]);
              if (v > fbv) { fbv = v; fb = a; }
            }
            for (let p = 0; p < P; p++) {
              let seen = false;
              for (let a = 0; a < K; a++) {
                const k = p * K + a;
                if (cnt[k] > 0) { seen = true; score[k] = sum[k] / cnt[k] - (params.beta * sig) / Math.sqrt(cnt[k]); }
                else score[k] = -Infinity;
              }
              if (!seen) { for (let a = 0; a < K; a++) score[p * K + a] = -Infinity; score[p * K + fb] = fbv; }
            }
            return { policy: greedyPolicy(score, P, K, pol), estimate: meanMaxLogged(data, score, n) };
          },
        };
      },
    },
    {
      id: 'linlcb', name: 'LinLCB', family: 'pessimism', kind: 'learn', params: [lamParam, betaParam],
      formula: 'π̂(x) = argmaxₐ x·θ̂ₐ − β σ̂ ‖x‖_{Aₐ⁻¹}',
      regret: 'V* − V(π̂) = Õ(d·√(C*/n)) under single-policy coverage (Jin, Yang & Wang 2021)',
      idea: 'Learner. Pessimistic ridge regression: subtract the confidence width of each action\'s linear model, so poorly covered actions need strong evidence. With β = 0 it is exactly DM-greedy.',
      create(data, { params }) { return linLcb(data, params.lambda, params.beta); },
    },
    {
      id: 'neurallcb', name: 'NeuralLCB', family: 'pessimism', kind: 'learn',
      params: [
        betaParam,
        { key: 'hidden', label: 'Hidden units', default: 16, min: 4, max: 64, step: 4 },
        { key: 'iters', label: 'SGD steps per refit', default: 60, min: 10, max: 500, step: 10 },
      ],
      formula: 'π̂(x) = argmaxₐ f(x,a;θ̂) − β σ̂ √(gᵀ diag(Λ)⁻¹ g),  g = ∇_θ f(x,a;θ̂)',
      regret: 'Õ(√(d̃ / n)) with effective dimension d̃ under coverage (Nguyen-Tang et al. 2022)',
      idea: 'Learner. Pessimism with a neural reward model (Nguyen-Tang, Gupta, Tran & Venkatesh 2022). Simplified: one ReLU hidden layer with a head per action, mini-batch Adam warm-started between refits, and a diagonal Λ = λI + Σ g gᵀ of gradient features.',
      create(data, { params, rng }) {
        const { K, d, P, X, cx, act, rew } = data;
        const H = Math.max(1, params.hidden | 0), iters = params.iters | 0, B = 32, lr = 0.01, lam = 1;
        // parameter layout: W1 (H×d) | b1 (H) | V (K×H) | c (K)
        const oB1 = H * d, oV = oB1 + H, oC = oV + K * H, N = oC + K;
        const th = new Float64Array(N), G = new Float64Array(N), m1 = new Float64Array(N), m2 = new Float64Array(N);
        const Lam = new Float64Array(N), pre = new Float64Array(H), h = new Float64Array(H);
        const score = new Float64Array(P * K), fq = new Float64Array(P * K), pol = new Float64Array(P * K);
        const s1 = Math.sqrt(2 / d), s2 = Math.sqrt(1 / H);
        for (let k = 0; k < oB1; k++) th[k] = rng.normal() * s1;
        for (let k = oV; k < oC; k++) th[k] = rng.normal() * s2;
        const cnt = new Float64Array(P * K);
        let step = 0, upto = 0;
        function hidden(xo) {
          for (let j = 0; j < H; j++) {
            let s = th[oB1 + j];
            for (let k = 0; k < d; k++) s += th[j * d + k] * X[xo + k];
            pre[j] = s; h[j] = s > 0 ? s : 0;
          }
        }
        function head(a) {
          let s = th[oC + a];
          for (let j = 0; j < H; j++) s += th[oV + a * H + j] * h[j];
          return s;
        }
        /** Σₖ gₖ² / Λₖ for output a at the current hidden state (or Λ += acc·g² when acc > 0). */
        function gradNorm(xo, a, acc) {
          let s = 0;
          for (let j = 0; j < H; j++) {
            const v = th[oV + a * H + j], on = pre[j] > 0;
            if (on) {
              for (let k = 0; k < d; k++) {
                const g = v * X[xo + k];
                if (acc) Lam[j * d + k] += acc * g * g; else s += (g * g) / Lam[j * d + k];
              }
              if (acc) Lam[oB1 + j] += acc * v * v; else s += (v * v) / Lam[oB1 + j];
            }
            if (acc) Lam[oV + a * H + j] += acc * h[j] * h[j]; else s += (h[j] * h[j]) / Lam[oV + a * H + j];
          }
          if (acc) Lam[oC + a] += acc; else s += 1 / Lam[oC + a];
          return s;
        }
        return {
          learn(n) {
            for (let it = 0; it < iters; it++) {
              G.fill(0);
              for (let bi = 0; bi < B; bi++) {
                const i = rng.int(n), xo = cx[i] * d, a = act[i];
                hidden(xo);
                const err = (head(a) - rew[i]) / B;
                G[oC + a] += err;
                for (let j = 0; j < H; j++) {
                  G[oV + a * H + j] += err * h[j];
                  if (pre[j] > 0) {
                    const dh = err * th[oV + a * H + j];
                    G[oB1 + j] += dh;
                    for (let k = 0; k < d; k++) G[j * d + k] += dh * X[xo + k];
                  }
                }
              }
              step++;
              const c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
              for (let k = 0; k < N; k++) {
                m1[k] = 0.9 * m1[k] + 0.1 * G[k];
                m2[k] = 0.999 * m2[k] + 0.001 * G[k] * G[k];
                th[k] -= (lr * (m1[k] / c1)) / (Math.sqrt(m2[k] / c2) + 1e-8);
              }
            }
            // Predictions on the pool, residual scale and the diagonal design matrix.
            for (let p = 0; p < P; p++) { hidden(p * d); for (let a = 0; a < K; a++) fq[p * K + a] = head(a); }
            for (; upto < n; upto++) cnt[cx[upto] * K + act[upto]]++;
            Lam.fill(lam);
            let rss = 0;
            for (let i = 0; i < n; i++) { const e = rew[i] - fq[cx[i] * K + act[i]]; rss += e * e; }
            for (let p = 0; p < P; p++) {   // Λ = λI + Σᵢ gᵢ², grouped by (context, action)
              hidden(p * d);
              for (let a = 0; a < K; a++) if (cnt[p * K + a]) gradNorm(p * d, a, cnt[p * K + a]);
            }
            const sig = Math.max(0.05, Math.sqrt(rss / n));
            for (let p = 0; p < P; p++) {
              hidden(p * d);
              for (let a = 0; a < K; a++) score[p * K + a] = fq[p * K + a] - params.beta * sig * Math.sqrt(gradNorm(p * d, a, 0));
            }
            return { policy: greedyPolicy(score, P, K, pol), estimate: meanMaxLogged(data, score, n) };
          },
        };
      },
    },
  ];

  const ALGORITHMS = ESTIMATORS.concat(LEARNERS);
  const ALGO_BY_ID = {};
  for (const a of ALGORITHMS) ALGO_BY_ID[a.id] = a;

  function defaultParams(id) {
    const out = {};
    for (const p of ALGO_BY_ID[id].params) out[p.key] = p.default;
    return out;
  }

  /** Log-spaced grid of sample sizes n₀ … nMax (nested prefixes of the logged data). */
  function sampleGrid(nMax, points) {
    const n0 = Math.min(nMax, Math.max(10, Math.round(nMax / 20)));
    const out = [];
    for (let i = 0; i < points; i++) {
      const v = Math.round(n0 * Math.pow(nMax / n0, points > 1 ? i / (points - 1) : 1));
      if (!out.length || v > out[out.length - 1]) out.push(v);
    }
    if (out[out.length - 1] !== nMax) out.push(nMax);
    return out;
  }

  // -------------------------------------------------------------- runner ----
  function createRunner(config) {
    const env = ENV_BY_ID[config.env];
    if (!env) throw new Error('Unknown environment: ' + config.env);
    const K = config.K | 0, T = config.T | 0, runs = config.runs | 0, seed = config.seed >>> 0;
    if (K < 2) throw new Error('K must be at least 2');
    const extra = config.extra || {};
    const d = Math.max(2, (extra.d ?? 5) | 0), P = Math.max(2, (extra.P ?? 100) | 0);
    const nMax = Math.max(10, env.nCap ? Math.min(T, env.nCap) : T);
    const grid = sampleGrid(nMax, 9), L = grid.length;

    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaultParams(spec.id), spec.params || {}),
        // loss-quantity moments (error e for estimators, suboptimality s for learners)
        l1: new Float64Array(L), l2: new Float64Array(L), l4: new Float64Array(L),
        b1: new Float64Array(L), finals: [], timeMs: 0,
      };
    });
    let done = 0;

    function runOne(run) {
      const inst = makeInstance(env, K, d, P, nMax, Rng(hash32(seed, run, 'env')));
      const { data, mu } = inst;
      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const t0 = util.now();
        const agent = slot.def.create(data, { params: slot.params, rng, env });
        let last = 0;
        for (let l = 0; l < L; l++) {
          const n = grid[l];
          let loss, bias;
          if (slot.def.kind === 'eval') {
            let v = agent.estimate(n);
            if (!Number.isFinite(v)) v = 0;
            loss = v - inst.Ve; bias = loss;
            slot.l1[l] += loss; slot.l2[l] += loss * loss; slot.l4[l] += loss * loss * loss * loss;
            last = Math.abs(loss);
          } else {
            const out = agent.learn(n);
            let val = 0;
            for (let k = 0; k < P * K; k++) val += out.policy[k] * mu[k];
            val /= P;
            loss = inst.Vstar - val;
            bias = Number.isFinite(out.estimate) ? out.estimate - val : 0;
            slot.l1[l] += loss; slot.l2[l] += loss * loss;
            last = loss;
          }
          slot.b1[l] += bias;
        }
        slot.timeMs += util.now() - t0;
        slot.finals.push(last);
      }
    }

    return {
      config: { env: env.id, K, T, runs, seed, d, P, nMax },
      grid,
      get done() { return done; },
      get finished() { return done >= runs; },
      step(n = 1) {
        const stop = Math.min(runs, done + n);
        while (done < stop) runOne(done++);
        return done;
      },
      results() {
        const R = Math.max(1, done), x = Float64Array.from(grid);
        return slots.map((s) => {
          const evalKind = s.def.kind === 'eval';
          const loss = new Float64Array(L), lossSE = new Float64Array(L), bias = new Float64Array(L), sd = new Float64Array(L);
          for (let l = 0; l < L; l++) {
            const m1 = s.l1[l] / R, m2 = s.l2[l] / R, v = Math.max(0, m2 - m1 * m1);
            sd[l] = R > 1 ? Math.sqrt((v * R) / (R - 1)) : 0;
            bias[l] = s.b1[l] / R;
            if (evalKind) {
              loss[l] = Math.sqrt(m2);
              const v2 = Math.max(0, s.l4[l] / R - m2 * m2);   // delta method for √(mean e²)
              lossSE[l] = R > 1 && loss[l] > 0 ? Math.sqrt(v2 / (R - 1)) / (2 * loss[l]) : 0;
            } else {
              loss[l] = m1;
              lossSE[l] = R > 1 ? Math.sqrt(v / (R - 1)) : 0;
            }
          }
          const finals = s.finals.slice().sort((a, b) => a - b);
          const summary = {
            loss: loss[L - 1], lossSE: lossSE[L - 1],
            lossMedian: finals.length ? finals[Math.floor(finals.length / 2)] : 0,
            bias: bias[L - 1], absBias: Math.abs(bias[L - 1]), sd: sd[L - 1], lossStart: loss[0],
            msPerRun: done ? s.timeMs / done : 0,
          };
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, kind: s.def.kind, params: s.params, runs: done,
            x, curves: { loss, bias, sd }, curvesSE: { loss: lossSE }, summary, finals,
          };
        });
      },
    };
  }

  const EST_IDS = ESTIMATORS.map((a) => a.id), LEARN_IDS = LEARNERS.map((a) => a.id);
  const NOTES = 'Estimators (kind "eval") and learners (kind "learn") share the metrics but read them differently. ' +
    'Error / suboptimality: RMSE of V̂(π) across runs for estimators; mean V(π*) − V(π̂) of the learned policy for learners. ' +
    'Value-estimate bias: mean V̂ − V for estimators; for learners, how far the method\'s own estimate of its policy\'s value is from the truth (positive = optimistic). ' +
    'True values are exact because contexts come from a finite pool. All methods see the same logged rounds; curves use nested prefixes of them.';

  const setting = MAB.registerSetting({
    id: 'offpolicy',
    name: 'Off-policy & offline',
    short: 'Off-policy',
    blurb: 'Learn from logged contextual-bandit data you did not collect: estimate a target policy\'s value (estimators, scored by RMSE) or learn a new policy (learners, scored by suboptimality).',
    notes: NOTES,
    families: [
      { id: 'ips', name: 'Importance weighting', blurb: 'Reweight logged rewards by π/π₀. Unbiased or nearly so, but noisy when the policies disagree.' },
      { id: 'model-based', name: 'Model-based', blurb: 'Fit a reward model and plug it in. Low variance, biased when the model is wrong.' },
      { id: 'doubly-robust', name: 'Doubly robust & blends', blurb: 'Combine a reward model with importance weights to get the best of both.' },
      { id: 'learning', name: 'Policy learning', blurb: 'Learn a new policy from the log: greedy plug-ins and counterfactual risk minimisation.' },
      { id: 'pessimism', name: 'Pessimism', blurb: 'Act on lower confidence bounds so that poorly covered actions are not chosen by luck.' },
    ],
    environments: ENVS.map(({ id, name, short, description, tags }) => ({ id, name, short, description, tags })),
    algorithms: ALGORITHMS,
    defaultEnv: 'linear',
    defaultLineup: EST_IDS.slice(),
    presets: [
      { label: 'Estimators', algos: EST_IDS.slice() },
      { label: 'Estimators, low overlap', algos: ['ips', 'snips', 'cips', 'dm', 'dr', 'switch-dr', 'cab'], env: 'lowoverlap' },
      { label: 'Estimators, misspecified', algos: ['ips', 'snips', 'dm', 'dr', 'mrdr', 'switch-dr', 'cab'], env: 'misspecified' },
      { label: 'Policy learning', algos: LEARN_IDS.slice() },
      { label: 'Pessimism vs greedy', algos: ['dm-greedy', 'ips-greedy', 'linlcb', 'pevi', 'neurallcb'], env: 'lowoverlap' },
      { label: 'CRM: POEM vs SVP', algos: ['ips-greedy', 'poem', 'svp', 'dm-greedy'] },
    ],
    controls: {
      K: { label: 'Actions (K)', min: 2, max: 20, default: 5 },
      T: { label: 'Logged rounds (max n)', options: [200, 500, 1000, 2000, 5000], default: 1000 },
      runs: { label: 'Runs averaged', options: [20, 50, 100, 200, 500], default: 100 },
      extra: [
        { key: 'd', label: 'Context dimension d', min: 2, max: 12, step: 1, default: 5 },
        { key: 'P', label: 'Distinct contexts', min: 10, max: 1000, step: 10, default: 100 },
      ],
    },
    xLabel: 'logged samples n',
    metrics: [
      { key: 'loss', label: 'Error / suboptimality', better: 'lower', fmt: 'fixed3', band: true,
        note: 'Estimators: RMSE of the value estimate of the target policy. Learners: V(π*) − V(π̂), true value lost by the learned policy. Reward units; bands are 95% CIs.' },
      { key: 'bias', label: 'Value-estimate bias', better: 'lower', fmt: 'fixed3',
        note: 'Mean signed error V̂ − V (closer to 0 is better). For learners: the method\'s estimate of its own policy\'s value minus the truth; greedy is optimistic, pessimism is not.' },
      { key: 'sd', label: 'Spread across runs', better: 'lower', fmt: 'fixed3',
        note: 'Standard deviation across runs of the estimation error (estimators) or of the suboptimality (learners).' },
    ],
    final: { key: 'loss', se: 'lossSE', median: 'lossMedian', label: 'RMSE / suboptimality at max n', better: 'lower' },
    tiles: [
      { key: 'loss', label: 'Lowest error / suboptimality', better: 'lower', fmt: 'fixed3', sub: 'at the largest n' },
      { key: 'lossStart', label: 'Best with little data', better: 'lower', fmt: 'fixed3', sub: 'at the smallest n' },
      { key: 'absBias', label: 'Least biased value estimate', better: 'lower', fmt: 'fixed3', sub: '|mean V̂ − V| at the largest n' },
    ],
    columns: [
      { key: 'loss', label: 'RMSE / subopt.', fmt: 'fixed3', better: 'lower' },
      { key: 'lossSE', label: '95% CI', fmt: 'ci' },
      { key: 'lossMedian', label: 'Median |err| / subopt.', fmt: 'fixed3', better: 'lower' },
      { key: 'bias', label: 'Bias of V̂', fmt: 'fixed3' },
      { key: 'sd', label: 'SD', fmt: 'fixed3', better: 'lower' },
      { key: 'lossStart', label: 'At smallest n', fmt: 'fixed3', better: 'lower' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: false,
    createRunner,
  });

  // Exposed for tests and tooling.
  setting.makeInstance = (envId, K, d, P, nMax, seed) => makeInstance(ENV_BY_ID[envId], K, d, P, nMax, Rng(seed));
  setting.sampleGrid = sampleGrid;
  return setting;
});
