/*
 * dueling.js — the dueling-bandit setting.
 *
 * Each round the learner picks a pair (i, j) and sees only which one won; the winner is drawn
 * from a preference matrix P with P[i][j] = P(i beats j) and P[j][i] = 1 − P[i][j].
 *
 * Design notes
 *  - Each run draws one instance (preference matrix, arm labels shuffled) and a table of
 *    uniforms U[t][pair] for every unordered pair, plus one uniform S[t] per step. The duel
 *    (i, j) at step t is won by the lower-indexed arm iff U[t][pair] < P[lo][hi]; a self-duel
 *    (i, i) and the continuous duels of DBGD are decided by S[t]. Every algorithm in the run
 *    reads the same table, so equal choices see equal outcomes (common random numbers).
 *  - Regret is measured against the instance's winner. Per arm, rᵢ is
 *      Condorcet: P(w ≻ i) − ½        Copeland: (Cpld(w) − Cpld(i)) / (K − 1)
 *      Borda:     b(w) − b(i), b(i) = Σⱼ≠ᵢ P(i ≻ j) / (K − 1)
 *    Strong regret of a duel (i, j) is (rᵢ + rⱼ)/2, weak regret is min(rᵢ, rⱼ).
 *  - Algorithm protocol: create(K, ctx) with ctx = { T, params, rng, env, info } returns
 *    { select(t) → [a, b], update(a, b, aWon, t) }, t = 1..T, aWon ∈ {0, 1}. a and b are arm
 *    indices; on the continuous environment (info.space = { d }) an agent may instead return two
 *    points (Float64Array(d) in the unit ball), as DBGD does.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;
  const { klBern, exp3Gamma, sampleIndex, softmaxInto, now } = util;

  // ------------------------------------------------------------ helpers ----
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const clampP = (p) => Math.min(0.98, Math.max(0.02, p));
  const loglog = (t) => Math.log(Math.log(Math.max(t, 3)));

  /** Index of the largest value(j) over allowed j, random tie-break; -1 if none allowed. */
  function bestOf(K, rng, value, allow) {
    let best = -Infinity, idx = -1, ties = 0;
    for (let j = 0; j < K; j++) {
      if (allow && !allow(j)) continue;
      const v = value(j);
      if (v > best || idx < 0) { best = v; idx = j; ties = 1; }
      else if (v === best) { ties++; if (rng.uniform() * ties < 1) idx = j; }
    }
    return idx;
  }

  /** Pairwise win counts: W[i*K+j] = wins of i over j, N[i*K+j] = N[j*K+i] = duels of the pair. */
  function Duels(K) {
    const W = new Float64Array(K * K), N = new Float64Array(K * K);
    return {
      W, N,
      add(i, j, iWon) {
        if (i === j) return;
        if (iWon) W[i * K + j]++; else W[j * K + i]++;
        N[i * K + j]++; N[j * K + i]++;
      },
      mean(i, j) { const n = N[i * K + j]; return n > 0 ? W[i * K + j] / n : 0.5; },
    };
  }

  /** RUCB-style bounds P̂ᵢⱼ ± √(α ln t / Nᵢⱼ); x/0 := 1 (upper) and 0 (lower); diagonal ½. */
  function bounds(D, K, alpha, t, U, L) {
    const lt = Math.log(Math.max(t, 1));
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        const k = i * K + j;
        if (i === j) { U[k] = 0.5; if (L) L[k] = 0.5; continue; }
        const n = D.N[k];
        if (n === 0) { U[k] = 1; if (L) L[k] = 0; continue; }
        const m = D.W[k] / n, r = Math.sqrt((alpha * lt) / n);
        U[k] = m + r;
        if (L) L[k] = m - r;
      }
    }
  }

  /** UCB1 single-bandit machine for rewards in [0, 1]. */
  function Ucb1(K, c, rng) {
    const n = new Float64Array(K), s = new Float64Array(K);
    let t = 0;
    return {
      select() {
        t++;
        const lt = Math.log(t);
        return bestOf(K, rng, (a) => (n[a] === 0 ? Infinity : s[a] / n[a] + c * Math.sqrt(lt / n[a])));
      },
      update(a, r) { n[a]++; s[a] += r; },
      reset() { n.fill(0); s.fill(0); t = 0; },
    };
  }

  /** EXP3 (Auer et al. 2002) for rewards in [0, 1]. */
  function Exp3(K, gamma, rng) {
    const lw = new Float64Array(K), p = new Float64Array(K);
    return {
      select() {
        softmaxInto(p, lw, 1);
        for (let a = 0; a < K; a++) p[a] = (1 - gamma) * p[a] + gamma / K;
        return sampleIndex(p, rng);
      },
      update(a, r) { lw[a] += (gamma * r) / (p[a] * K); },
    };
  }

  /** Pairs i < j in a fixed order, for warm-up phases. */
  function pairList(K) {
    const pi = [], pj = [];
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) { pi.push(i); pj.push(j); }
    return { pi, pj, n: pi.length };
  }

  // ----------------------------------------------------------- families ----
  const families = [
    { id: 'baseline', name: 'Baseline', blurb: 'Ignores the feedback; the cost of pure exploration.' },
    { id: 'explore-then-exploit', name: 'Explore then exploit', blurb: 'Run a tournament until one arm is confidently best, then duel it against itself forever.' },
    { id: 'confidence', name: 'Relative confidence & divergence', blurb: 'Optimism on pairwise win rates (RUCB) or empirical KL evidence that an arm is beaten (RMED).' },
    { id: 'thompson', name: 'Thompson sampling', blurb: 'Sample a preference matrix from Beta posteriors and act on the sample.' },
    { id: 'reduction', name: 'Reductions to MAB', blurb: 'Run ordinary bandit learners against each other or against themselves; the duel outcome is the reward.' },
    { id: 'copeland-borda', name: 'Copeland & Borda', blurb: 'Target the Copeland winner (most pairwise wins) or the Borda winner (best average win rate); works without a Condorcet winner.' },
    { id: 'gradient', name: 'Gradient', blurb: 'Treat arms as points of a parameter space and climb the preference by random-direction comparisons.' },
  ];

  // --------------------------------------------------------- algorithms ----
  const ALPHA = { key: 'alpha', label: 'α (exploration)', default: 0.51, min: 0.1, max: 2, step: 0.01 };
  const FCOEF = { key: 'f', label: 'f(K) coefficient', default: 0.3, min: 0, max: 2, step: 0.05 };
  const UCBC = { key: 'c', label: 'UCB1 c', default: 1.414, min: 0, max: 3, step: 0.01 };

  /** RMED1 / RMED2 / RMED2FH (Komiyama, Honda, Kashima & Nakagawa 2015). */
  function makeRmed(variant) {
    return function create(K, { T, params, rng }) {
      const D = Duels(K), fK = params.f * Math.pow(K, 1.01), alpha = params.alpha || 0;
      const P = pairList(K);
      const reps = variant === 'fh' ? Math.max(1, Math.ceil(alpha * loglog(T))) : 1;
      const nInit = reps * P.n;
      const I = new Float64Array(K), inR = new Uint8Array(K), inN = new Uint8Array(K);
      const fixedOpp = new Int32Array(K).fill(-1);
      let initPos = 0, started = false, LC = [], pos = 0, cur = -1, istar = 0, Imin = 0;

      function computeI() {
        Imin = Infinity;
        for (let i = 0; i < K; i++) {
          let s = 0;
          for (let j = 0; j < K; j++) {
            const n = D.N[i * K + j];
            if (j === i || n === 0) continue;
            const m = D.W[i * K + j] / n;
            if (m < 0.5) s += n * klBern(m, 0.5);
          }
          I[i] = s;
          if (s < Imin) Imin = s;
        }
        istar = bestOf(K, rng, (i) => -I[i]);
      }
      function startLoop(list) {
        LC = list; pos = 0; inR.fill(0); inN.fill(0);
        for (const l of LC) inR[l] = 1;
      }
      const beaten = (l) => { for (let j = 0; j < K; j++) if (j !== l && D.mean(l, j) <= 0.5) return true; return false; };
      const rmed1Opp = (l) => bestOf(K, rng, (j) => -D.mean(l, j), (j) => j !== l);
      // RMED2's LP puts all comparison mass on the j minimising regret per unit of evidence.
      const gap = (j) => (j === istar ? 0 : Math.max(0, D.mean(istar, j) - 0.5));
      function ratioOpp(l) {
        const dl = gap(l);
        return bestOf(K, rng, (j) => {
          const kl = klBern(D.mean(l, j), 0.5);
          return kl > 0 ? -(dl + gap(j)) / kl : -Infinity;
        }, (j) => j !== l && D.mean(l, j) <= 0.5);
      }
      function opponent(l, t) {
        if (!beaten(l)) return l;
        if (variant === 1) return rmed1Opp(l);
        if (variant === 2) {
          if (istar !== l && D.mean(l, istar) <= 0.5 && D.N[l * K + istar] < alpha * loglog(t)) return istar;
          return ratioOpp(l);
        }
        const f = fixedOpp[l];
        return f >= 0 && D.mean(l, f) <= 0.5 ? f : rmed1Opp(l);
      }
      return {
        select(t) {
          if (initPos < nInit) { const p = initPos++ % P.n; return [P.pi[p], P.pj[p]]; }
          if (!started) {
            started = true;
            computeI();
            if (variant === 'fh') for (let l = 0; l < K; l++) fixedOpp[l] = l === istar ? -1 : ratioOpp(l);
            startLoop(Array.from({ length: K }, (_, i) => i));
          }
          while (pos < LC.length && !inR[LC[pos]]) pos++;
          cur = LC[pos];
          return [cur, opponent(cur, t)];
        },
        update(a, b, won, t) {
          D.add(a, b, won);
          if (!started) return;
          inR[cur] = 0; pos++;
          computeI();
          const thr = Math.log(t) + fK;
          for (let j = 0; j < K; j++) if (!inR[j] && !inN[j] && I[j] - Imin <= thr) inN[j] = 1;
          if (pos >= LC.length) {
            const next = [];
            for (let j = 0; j < K; j++) if (inN[j]) next.push(j);
            startLoop(next.length ? next : [istar]);
          }
        },
      };
    };
  }

  const algorithms = [
    // ---- baseline
    {
      id: 'uniform', name: 'Uniform pairs', family: 'baseline', params: [],
      formula: '(i, j) ~ Uniform([K] × [K])',
      regret: 'Θ(T)',
      idea: 'Duels two arms drawn uniformly at random every round and never learns. Its regret is the price of not adapting.',
      create(K, { rng }) {
        return { select: () => [rng.int(K), rng.int(K)], update() {} };
      },
    },

    // ---- explore then exploit
    {
      id: 'if', name: 'Interleaved Filter', family: 'explore-then-exploit', params: [],
      formula: 'duel b̂ vs each b ∈ W in turn; drop b if P̂_b̂b − c > ½, switch b̂ ← b if P̂_b̂b + c < ½; c = √(ln(TK²)/n)',
      regret: 'O(K log T / ε₁,₂) (total order)',
      idea: 'Yue et al. (2012), IF2 with pruning: a candidate meets every survivor round-robin and is replaced by the first arm that confidently beats it. Assumes a total order with strong stochastic transitivity, so cycles can mislead it. Simplified: checks after every comparison rather than after each round.',
      create(K, { T, rng }) {
        const logInv = Math.log(T * K * K);
        const active = new Uint8Array(K).fill(1), w = new Float64Array(K), n = new Float64Array(K);
        let cand = rng.int(K), left = K - 1, ptr = cand;
        active[cand] = 0;
        return {
          select() {
            if (left === 0) return [cand, cand];
            do { ptr = (ptr + 1) % K; } while (!active[ptr]);
            return [cand, ptr];
          },
          update(a, b, won) {
            if (left === 0 || a === b || a !== cand || !active[b]) return;
            n[b]++; w[b] += won;
            const p = w[b] / n[b], c = Math.sqrt(logInv / n[b]);
            if (p - c > 0.5) { active[b] = 0; left--; }
            else if (p + c < 0.5) {
              for (let k = 0; k < K; k++) {
                if (active[k] && k !== b && n[k] > 0 && w[k] / n[k] > 0.5) { active[k] = 0; left--; }
              }
              active[b] = 0; left--;
              cand = b; w.fill(0); n.fill(0);
            }
          },
        };
      },
    },
    {
      id: 'btm', name: 'Beat the Mean', family: 'explore-then-exploit',
      params: [{ key: 'gamma', label: 'γ (relaxed transitivity)', default: 1, min: 1, max: 3, step: 0.1 }],
      formula: 'duel b = argmin_W n_b vs b′ ~ U(W); drop argmin P̂_b once min P̂ + c ≤ max P̂ − c, c = 3γ²√(ln(2TK)/n*)',
      regret: 'O(γ⁷ K log T / ε²)',
      idea: 'Yue & Joachims (2011): each survivor is scored by its win rate against a random survivor (the "mean bandit"); the worst is dropped once separated, and its duels are erased from everyone\'s record. Robust to relaxed transitivity but its confidence radius is very conservative.',
      create(K, { T, params, rng }) {
        const logInv = Math.log(2 * T * K), g = 3 * params.gamma * params.gamma;
        const act = Array.from({ length: K }, (_, i) => i);
        const wins = new Float64Array(K * K), cmp = new Float64Array(K * K);
        const nb = new Float64Array(K), wb = new Float64Array(K);
        return {
          select() {
            if (act.length === 1) return [act[0], act[0]];
            let b = act[0], bn = Infinity, ties = 0;
            for (const x of act) {
              if (nb[x] < bn) { bn = nb[x]; b = x; ties = 1; }
              else if (nb[x] === bn) { ties++; if (rng.uniform() * ties < 1) b = x; }
            }
            return [b, act[rng.int(act.length)]];
          },
          update(b, o, won) {
            if (act.length === 1) return;
            wins[b * K + o] += won; cmp[b * K + o]++; wb[b] += won; nb[b]++;
            let nmin = Infinity;
            for (const x of act) if (nb[x] < nmin) nmin = nb[x];
            if (nmin === 0) return;
            let pmin = Infinity, pmax = -Infinity, worst = -1;
            for (const x of act) {
              const p = wb[x] / nb[x];
              if (p > pmax) pmax = p;
              if (p < pmin) { pmin = p; worst = x; }
            }
            const c = g * Math.sqrt(logInv / nmin);
            if (pmin + c <= pmax - c) {
              act.splice(act.indexOf(worst), 1);
              for (const x of act) { nb[x] -= cmp[x * K + worst]; wb[x] -= wins[x * K + worst]; }
            }
          },
        };
      },
    },
    {
      id: 'savage', name: 'SAVAGE', family: 'explore-then-exploit',
      params: [{ key: 'delta', label: 'Confidence δ', default: 0.1, min: 0.001, max: 0.5, step: 0.001 }],
      formula: 'duel the least-sampled undecided pair touching a candidate; drop i once ∃ j: P̂ᵢⱼ + c < ½, c = √(ln(K²T/δ)/2n)',
      regret: 'O(K² log(KT/δ) / ε²) samples',
      idea: 'Urvoy et al. (2013), Condorcet version: explores only pairs whose outcome could still change who the winner is, then exploits. Without a Condorcet winner every arm gets eliminated and it falls back to the arm with the most confident wins (a crude Copeland SAVAGE).',
      create(K, { T, params, rng }) {
        const D = Duels(K), cand = new Uint8Array(K).fill(1);
        const logTerm = Math.log((K * (K - 1) * T) / params.delta);
        const rad = (n) => Math.sqrt(logTerm / (2 * n));
        let nc = K, exploit = -1;
        const undecided = (k) => { const n = D.N[k]; return n === 0 || Math.abs(D.W[k] / n - 0.5) <= rad(n); };
        const fallback = () => bestOf(K, rng, (i) => {
          let s = 0;
          for (let j = 0; j < K; j++) { const n = D.N[i * K + j]; if (j !== i && n > 0 && D.W[i * K + j] / n - rad(n) > 0.5) s++; }
          return s;
        });
        return {
          select() {
            if (exploit >= 0) return [exploit, exploit];
            let bi = -1, bj = -1, bn = Infinity, ties = 0;
            for (let i = 0; i < K; i++) {
              for (let j = i + 1; j < K; j++) {
                if (!(cand[i] || cand[j]) || !undecided(i * K + j)) continue;
                const n = D.N[i * K + j];
                if (n < bn) { bn = n; bi = i; bj = j; ties = 1; }
                else if (n === bn) { ties++; if (rng.uniform() * ties < 1) { bi = i; bj = j; } }
              }
            }
            if (bi < 0) { exploit = nc === 1 ? cand.indexOf(1) : fallback(); return [exploit, exploit]; }
            return [bi, bj];
          },
          update(i, j, won) {
            if (exploit >= 0 || i === j) return;
            D.add(i, j, won);
            const k = i * K + j, n = D.N[k], m = D.W[k] / n, r = rad(n);
            if (m - r > 0.5 && cand[j]) { cand[j] = 0; nc--; }
            else if (m + r < 0.5 && cand[i]) { cand[i] = 0; nc--; }
            if (nc === 1) exploit = cand.indexOf(1);
            else if (nc === 0) exploit = fallback();
          },
        };
      },
    },

    // ---- confidence / divergence
    {
      id: 'rucb', name: 'RUCB', family: 'confidence', params: [ALPHA],
      formula: 'c ∈ {i : Uᵢⱼ ≥ ½ ∀j}, d = argmaxⱼ Uⱼ꜀, Uᵢⱼ = P̂ᵢⱼ + √(α ln t / Nᵢⱼ)',
      regret: 'O(K² + K log T)',
      idea: 'Relative UCB (Zoghi et al. 2014): the champion is an arm that optimistically beats everyone, and its challenger is the arm most likely to beat it. Keeps the previous champion with probability ½.',
      create(K, { params, rng }) {
        const D = Duels(K), U = new Float64Array(K * K), C = new Int32Array(K), inC = new Uint8Array(K);
        let B = -1;
        return {
          select(t) {
            bounds(D, K, params.alpha, t, U, null);
            let nc = 0;
            for (let c = 0; c < K; c++) {
              let ok = 1;
              for (let j = 0; j < K; j++) if (U[c * K + j] < 0.5) { ok = 0; break; }
              inC[c] = ok;
              if (ok) C[nc++] = c;
            }
            let c;
            if (nc === 0) { B = -1; c = rng.int(K); }
            else {
              if (B >= 0 && !inC[B]) B = -1;
              if (nc === 1) { B = C[0]; c = B; }
              else if (B >= 0) {
                if (rng.uniform() < 0.5) c = B;
                else { do { c = C[rng.int(nc)]; } while (c === B); }
              } else c = C[rng.int(nc)];
            }
            const d = bestOf(K, rng, (j) => U[j * K + c]);
            return [c, d];
          },
          update(i, j, won) { D.add(i, j, won); },
        };
      },
    },
    {
      id: 'rmed1', name: 'RMED1', family: 'confidence', params: [FCOEF],
      formula: 'loop over {i : Îᵢ − Î* ≤ ln t + f(K)}, Îᵢ = Σ_{j: P̂ᵢⱼ<½} Nᵢⱼ·kl(P̂ᵢⱼ, ½); opponent argminⱼ P̂ᵢⱼ',
      regret: 'O(Σᵢ ln T / minⱼ kl(Pᵢⱼ, ½))',
      idea: 'Relative Minimum Empirical Divergence (Komiyama et al. 2015): an arm keeps being played only while the KL evidence that it is beaten is below ln t; it is matched with the arm that beats it most, or with itself once nothing seems to beat it.',
      create: makeRmed(1),
    },
    {
      id: 'rmed2', name: 'RMED2', family: 'confidence',
      params: [{ key: 'alpha', label: 'α (forced exploration)', default: 3, min: 0.5, max: 10, step: 0.1 }, FCOEF],
      formula: 'RMED1 loop; opponent argmin_{j: P̂ᵢⱼ ≤ ½} (Δ̂ᵢ + Δ̂ⱼ)/kl(P̂ᵢⱼ, ½), or î* while Nᵢî* < α ln ln t',
      regret: 'asymptotically optimal',
      idea: 'Komiyama et al. (2015): like RMED1 but picks the opponent that buys the most evidence per unit of regret, mostly the estimated winner î*, which reaches the lower bound. The comparison-ratio LP is solved in closed form (all mass on the best ratio).',
      create: makeRmed(2),
    },
    {
      id: 'rmed2fh', name: 'RMED2FH', family: 'confidence',
      params: [{ key: 'alpha', label: 'α (warm-up)', default: 3, min: 0.5, max: 10, step: 0.1 }, FCOEF],
      formula: 'warm-up ⌈α ln ln T⌉ duels per pair; then RMED loop with opponents fixed from the warm-up estimates',
      regret: 'asymptotically optimal (known T)',
      idea: 'The fixed-horizon RMED2 (Komiyama et al. 2015): a longer warm-up replaces the forced exploration, and each arm\'s low-regret opponent is chosen once from the warm-up data (falling back to RMED1\'s rule if that opponent no longer beats it).',
      create: makeRmed('fh'),
    },

    // ---- Thompson
    {
      id: 'rcs', name: 'RCS', family: 'thompson', params: [ALPHA],
      formula: 'θᵢⱼ ~ Beta(Wᵢⱼ+1, Wⱼᵢ+1); c = Condorcet winner of θ; d = argmaxⱼ Uⱼ꜀',
      regret: 'no proof (empirical)',
      idea: 'Relative Confidence Sampling (Zoghi et al. 2014b): the champion comes from a posterior sample, the challenger from RUCB\'s optimistic bounds. Simplified: when the sample has no Condorcet winner, its Copeland winner is used.',
      create(K, { params, rng }) {
        const D = Duels(K), U = new Float64Array(K * K), th = new Float64Array(K * K);
        const win = (i) => { let s = 0; for (let j = 0; j < K; j++) if (j !== i && th[i * K + j] > 0.5) s++; return s; };
        return {
          select(t) {
            bounds(D, K, params.alpha, t, U, null);
            for (let i = 0; i < K; i++) {
              for (let j = i + 1; j < K; j++) {
                const x = rng.beta(D.W[i * K + j] + 1, D.W[j * K + i] + 1);
                th[i * K + j] = x; th[j * K + i] = 1 - x;
              }
            }
            const c = bestOf(K, rng, win);
            return [c, bestOf(K, rng, (j) => U[j * K + c])];
          },
          update(i, j, won) { D.add(i, j, won); },
        };
      },
    },
    {
      id: 'dts', name: 'Double Thompson Sampling', family: 'thompson', params: [ALPHA],
      formula: 'a₁ = argmax_{i∈C} #{j : θ¹ᵢⱼ > ½}, C = argmax upper Copeland; a₂ = argmax_{i: Lᵢₐ₁ ≤ ½} θ²ᵢₐ₁',
      regret: 'O(K² log T) Copeland',
      idea: 'Wu & Liu (2016): confidence bounds prune the candidates, one posterior sample picks the first arm among them (by Copeland score), and a second sample picks its most dangerous challenger. Works for Copeland winners too.',
      create(K, { params, rng }) {
        const D = Duels(K), U = new Float64Array(K * K), L = new Float64Array(K * K), th = new Float64Array(K * K);
        const up = new Int32Array(K), inC = new Uint8Array(K);
        return {
          select(t) {
            bounds(D, K, params.alpha, t, U, L);
            let mx = -1;
            for (let i = 0; i < K; i++) {
              let s = 0;
              for (let j = 0; j < K; j++) if (j !== i && U[i * K + j] > 0.5) s++;
              up[i] = s; if (s > mx) mx = s;
            }
            let nC = 0, only = 0;
            for (let i = 0; i < K; i++) { inC[i] = up[i] === mx ? 1 : 0; if (inC[i]) { nC++; only = i; } }
            let a1 = only;
            if (nC > 1) { // the first sample only matters when several candidates remain
              for (let i = 0; i < K; i++) {
                for (let j = i + 1; j < K; j++) {
                  if (!inC[i] && !inC[j]) continue;
                  const x = rng.beta(D.W[i * K + j] + 1, D.W[j * K + i] + 1);
                  th[i * K + j] = x; th[j * K + i] = 1 - x;
                }
              }
              a1 = bestOf(K, rng, (i) => {
                let s = 0;
                for (let j = 0; j < K; j++) if (j !== i && th[i * K + j] > 0.5) s++;
                return s;
              }, (i) => inC[i]);
            }
            const a2 = bestOf(K, rng,
              (i) => (i === a1 ? 0.5 : rng.beta(D.W[i * K + a1] + 1, D.W[a1 * K + i] + 1)),
              (i) => L[i * K + a1] <= 0.5);
            return [a1, a2 < 0 ? a1 : a2];
          },
          update(i, j, won) { D.add(i, j, won); },
        };
      },
    },

    // ---- reductions
    {
      id: 'sparring-ucb', name: 'Sparring UCB1', family: 'reduction', params: [UCBC],
      formula: 'i = UCB1_left, j = UCB1_right; rewards 1{i ≻ j} and 1{j ≻ i}',
      regret: 'O(K log T) empirically (no general proof)',
      idea: 'Ailon, Karnin & Joachims (2014): two independent UCB1 learners play each other and each is rewarded when its arm wins. Simple and strong under utility-based preferences.',
      create(K, { params, rng }) {
        const A = Ucb1(K, params.c, rng), B = Ucb1(K, params.c, rng);
        return {
          select: () => [A.select(), B.select()],
          update(i, j, won) { A.update(i, won); B.update(j, 1 - won); },
        };
      },
    },
    {
      id: 'sparring-exp3', name: 'Sparring EXP3', family: 'reduction',
      params: [{ key: 'g', label: 'γ multiplier', default: 1, min: 0.1, max: 5, step: 0.1 }],
      formula: 'i ~ EXP3_left, j ~ EXP3_right; rewards 1{i ≻ j} and 1{j ≻ i}, γ = √(K ln K / ((e−1)T))',
      regret: 'O(√(KT ln K)) (utility-based)',
      idea: 'Ailon, Karnin & Joachims (2014): the same sparring game with two adversarial EXP3 learners; their regret bounds transfer to dueling regret when preferences come from utilities.',
      create(K, { T, params, rng }) {
        const g = Math.min(1, params.g * exp3Gamma(K, T));
        const A = Exp3(K, g, rng), B = Exp3(K, g, rng);
        return {
          select: () => [A.select(), B.select()],
          update(i, j, won) { A.update(i, won); B.update(j, 1 - won); },
        };
      },
    },
    {
      id: 'vdb', name: 'Versatile-DB (Tsallis-INF)', family: 'reduction',
      params: [{ key: 'eta', label: 'η multiplier', default: 1, min: 0.1, max: 5, step: 0.1 }],
      formula: 'i, j ~ pₜ i.i.d., pₜₖ = 4/(ηₜ(L̂ₖ − x))², ηₜ = 2/√t; L̂ᵢ += (1 − 1{i≻j})/(2pᵢ), L̂ⱼ += 1{i≻j}/(2pⱼ)',
      regret: 'O(√(KT)) adversarial, O(Σ log T/Δ) stochastic',
      idea: 'Self-play reduction of Saha & Gaillard (2022): one ½-Tsallis-INF learner (Zimmert & Seldin) draws both arms independently and is fed importance-weighted losses. Both sides of the duel are used (averaged), a variance-reducing variant.',
      create(K, { params, rng }) {
        const Lh = new Float64Array(K), p = new Float64Array(K);
        function probs(eta) {
          let mn = Infinity;
          for (let k = 0; k < K; k++) if (Lh[k] < mn) mn = Lh[k];
          let x = mn - 2 / eta;
          for (let it = 0; it < 50; it++) {
            let s = 0, s3 = 0;
            for (let k = 0; k < K; k++) {
              const z = eta * (Lh[k] - x), q = 4 / (z * z);
              p[k] = q; s += q; s3 += q * Math.sqrt(q);
            }
            if (Math.abs(s - 1) < 1e-10) break;
            x -= (s - 1) / (eta * s3);
          }
          let s = 0;
          for (let k = 0; k < K; k++) s += p[k];
          for (let k = 0; k < K; k++) p[k] /= s;
        }
        return {
          select(t) {
            probs((params.eta * 2) / Math.sqrt(t));
            return [sampleIndex(p, rng), sampleIndex(p, rng)];
          },
          update(i, j, won) { Lh[i] += (1 - won) / (2 * p[i]); Lh[j] += won / (2 * p[j]); },
        };
      },
    },
    {
      id: 'doubler', name: 'Doubler', family: 'reduction', params: [UCBC],
      formula: 'epoch e (2ᵉ rounds): i ~ uniform over epoch e−1\'s right arms, j = UCB1 on 1{j ≻ i} (reset each epoch)',
      regret: 'O(K log² T) (linear utility)',
      idea: 'Ailon, Karnin & Joachims (2014): the left arm replays the right learner\'s choices from the previous epoch, so the right learner faces a fixed opponent distribution that keeps improving.',
      create(K, { params, rng }) {
        const S = Ucb1(K, params.c, rng);
        let pool = [rng.int(K)], next = [], len = 2, pos = 0;
        return {
          select: () => [pool[rng.int(pool.length)], S.select()],
          update(x, y, won) {
            S.update(y, 1 - won);
            next.push(y);
            if (++pos >= len) { pool = next; next = []; len *= 2; pos = 0; S.reset(); }
          },
        };
      },
    },
    {
      id: 'multisbm', name: 'MultiSBM', family: 'reduction', params: [UCBC],
      formula: 'iₜ = jₜ₋₁, jₜ = UCB1⁽ⁱ⁾ on rewards 1{j ≻ i}',
      regret: 'O(K log T) (utility-based)',
      idea: 'Ailon, Karnin & Joachims (2014): one UCB1 learner per left arm; the left arm is the previous round\'s right arm, so play converges to the winner duelling itself.',
      create(K, { params, rng }) {
        const n = new Float64Array(K * K), s = new Float64Array(K * K), tx = new Float64Array(K), c = params.c;
        let x = rng.int(K);
        return {
          select() {
            tx[x]++;
            const lt = Math.log(tx[x]), o = x * K;
            return [x, bestOf(K, rng, (a) => (n[o + a] === 0 ? Infinity : s[o + a] / n[o + a] + c * Math.sqrt(lt / n[o + a])))];
          },
          update(a, b, won) { n[a * K + b]++; s[a * K + b] += 1 - won; x = b; },
        };
      },
    },

    // ---- Copeland & Borda
    {
      id: 'ccb', name: 'CCB', family: 'copeland-borda', params: [ALPHA],
      formula: 'c ~ C = argmax Cpld̄ (¼ of the time from hypotheses B); d = argmax_{j ∈ Bᶜ: Lⱼ꜀ ≤ ½} Uⱼ꜀',
      regret: 'O(K² + K log T) Copeland',
      idea: 'Copeland Confidence Bound (Zoghi et al. 2015): keeps hypotheses about which arms are Copeland winners and which arms beat each one, and spends duels confirming or refuting them; no Condorcet winner needed. Close to the paper\'s Algorithm 1, slightly simplified in how d is chosen.',
      create(K, { params, rng }) {
        const D = Duels(K), U = new Float64Array(K * K), L = new Float64Array(K * K);
        const Bset = new Uint8Array(K), Bi = new Uint8Array(K * K), Bsz = new Int32Array(K);
        const cu = new Int32Array(K), cl = new Int32Array(K), C = new Int32Array(K), BC = new Int32Array(K);
        let LC = K;
        const clearRow = (i) => { for (let k = 0; k < K; k++) Bi[i * K + k] = 0; Bsz[i] = 0; };
        const resetAll = () => { Bset.fill(1); Bi.fill(0); Bsz.fill(0); LC = K; };
        resetAll();
        return {
          select(t) {
            bounds(D, K, params.alpha, t, U, L);
            let mx = -1, mxl = -1;
            for (let i = 0; i < K; i++) {
              let u = 0, l = 0;
              for (let k = 0; k < K; k++) {
                if (k === i) continue;
                if (U[i * K + k] >= 0.5) u++;
                if (L[i * K + k] > 0.5) l++;
              }
              cu[i] = u; cl[i] = l;
              if (u > mx) mx = u;
              if (l > mxl) mxl = l;
            }
            // 1. reset disproven hypotheses
            let reset = false;
            for (let i = 0; i < K && !reset; i++) {
              for (let k = 0; k < K; k++) if (Bi[i * K + k] && L[i * K + k] > 0.5) { reset = true; break; }
            }
            if (reset) resetAll();
            // 2. remove non-Copeland winners
            let any = false;
            for (let i = 0; i < K; i++) {
              if (!Bset[i]) continue;
              if (cu[i] < mxl) {
                Bset[i] = 0;
                if (Bsz[i] !== LC + 1) {
                  Bsz[i] = 0;
                  for (let k = 0; k < K; k++) { const v = k !== i && U[i * K + k] < 0.5 ? 1 : 0; Bi[i * K + k] = v; Bsz[i] += v; }
                }
              } else any = true;
            }
            if (!any) resetAll();
            // 3. add confirmed Copeland winners
            for (let i = 0; i < K; i++) {
              if (cu[i] !== mx || cl[i] !== cu[i]) continue;
              Bset[i] = 1; clearRow(i);
              LC = K - 1 - cu[i];
              for (let j = 0; j < K; j++) {
                if (j === i) continue;
                if (Bsz[j] < LC + 1) clearRow(j);
                else if (Bsz[j] > LC + 1) {
                  const mem = [];
                  for (let k = 0; k < K; k++) if (Bi[j * K + k]) mem.push(k);
                  for (let a = mem.length - 1; a > 0; a--) { const b = rng.int(a + 1); const z = mem[a]; mem[a] = mem[b]; mem[b] = z; }
                  for (let a = LC + 1; a < mem.length; a++) Bi[j * K + mem[a]] = 0;
                  Bsz[j] = LC + 1;
                }
              }
            }
            // 4. champion c
            let nc = 0, nb = 0;
            for (let i = 0; i < K; i++) if (cu[i] === mx) { C[nc++] = i; if (Bset[i]) BC[nb++] = i; }
            const c = nb > 0 && rng.uniform() < 0.25 ? BC[rng.int(nb)] : C[rng.int(nc)];
            // 5. challenger d: an arm hypothesised to beat c that is not yet confirmed to
            let d = bestOf(K, rng, (j) => U[j * K + c], (j) => Bi[c * K + j] && L[j * K + c] <= 0.5);
            if (d < 0) d = bestOf(K, rng, (j) => U[j * K + c], (j) => L[j * K + c] <= 0.5);
            return [c, d];
          },
          update(i, j, won) { D.add(i, j, won); },
        };
      },
    },
    {
      id: 'cw-rmed', name: 'CW-RMED', family: 'copeland-borda', params: [FCOEF],
      formula: 'loop over {i : Îᵢ ≤ ln t + f(K)}, Îᵢ = Σ of the (Cpld̂* − Cpld̂ᵢ) smallest Nᵢⱼ·kl(P̂ᵢⱼ, ½) over j with P̂ᵢⱼ ≤ ½',
      regret: 'asymptotically optimal Copeland (original)',
      idea: 'Copeland Winners RMED (Komiyama et al. 2016), simplified in the spirit of their efficient ECW-RMED: an arm is explored while the cheapest KL cost of flipping enough of its losses to make it a Copeland winner is below ln t, against the most informative of those opponents. The empirical winner duels itself, or a close rival whose loss to it is still uncertain.',
      create(K, { T, params, rng }) {
        const D = Duels(K), fK = params.f * Math.pow(K, 1.01), P = pairList(K);
        const wins = new Int32Array(K), opp = new Int32Array(K), inJ = new Uint8Array(K);
        const cost = new Float64Array(K), order = new Int32Array(K), pick = new Uint8Array(K);
        let initPos = 0, ptr = -1;
        const ev = (i, j) => { const n = D.N[i * K + j]; return n > 0 ? n * klBern(D.W[i * K + j] / n, 0.5) : 0; };
        function analyze(t) {
          let mx = -1, nw = 0;
          for (let i = 0; i < K; i++) {
            let s = 0;
            for (let j = 0; j < K; j++) if (j !== i && D.mean(i, j) > 0.5) s++;
            wins[i] = s;
            if (s > mx) { mx = s; nw = 1; } else if (s === mx) nw++;
          }
          const thr = Math.log(t) + fK;
          for (let i = 0; i < K; i++) {
            if (wins[i] === mx) {
              inJ[i] = 1;
              if (nw > 1) { opp[i] = bestOf(K, rng, (j) => -ev(i, j), (j) => j !== i); continue; }
              // single winner: check its wins over close rivals
              let best = Infinity, o = i;
              for (let j = 0; j < K; j++) {
                if (j === i || D.mean(i, j) <= 0.5 || wins[j] < mx - 2) continue;
                const e = ev(i, j);
                if (e < best) { best = e; o = j; }
              }
              opp[i] = best < thr ? o : i;
              continue;
            }
            // non-winner: cheapest KL cost of turning enough non-wins (losses, or ties at cost 0)
            // and duel the most informative pair (largest kl per duel) among those flips
            let m = 0;
            for (let j = 0; j < K; j++) if (j !== i && D.mean(i, j) <= 0.5) { cost[j] = ev(i, j); order[m++] = j; }
            const need = Math.min(m, mx - wins[i]);
            const sub = order.subarray(0, m).sort((x, y) => cost[x] - cost[y]);
            let I = 0;
            for (let a = 0; a < need; a++) { I += cost[sub[a]]; pick[sub[a]] = 1; }
            inJ[i] = I <= thr ? 1 : 0;
            opp[i] = bestOf(K, rng, (j) => klBern(D.mean(i, j), 0.5), (j) => pick[j] === 1);
            pick.fill(0);
          }
        }
        return {
          select(t) {
            if (initPos < P.n) { const p = initPos++; return [P.pi[p], P.pj[p]]; }
            analyze(t);
            do { ptr = (ptr + 1) % K; } while (!inJ[ptr]);
            return [ptr, opp[ptr]];
          },
          update(i, j, won) { D.add(i, j, won); },
        };
      },
    },
    {
      id: 'borda-se', name: 'Borda successive elimination', family: 'copeland-borda',
      params: [{ key: 'beta', label: 'Confidence scale β', default: 0.5, min: 0.1, max: 3, step: 0.05 }],
      formula: 'duel (i, j), i = least-sampled survivor, j ~ U([K]∖{i}); drop i if b̂ᵢ + cᵢ < maxₖ (b̂ₖ − cₖ), cᵢ = β√(ln(KT)/2nᵢ)',
      regret: 'O(Σᵢ log(KT) / Δᵢ²) Borda',
      idea: 'An honest Borda-winner method: a duel against a uniformly random opponent is an unbiased sample of the Borda score, so plain successive elimination on those samples finds the Borda winner, which then duels itself. In the spirit of Busa-Fekete et al. (2013) and Jamieson et al. (2015), without SECS\'s sparsity tricks.',
      create(K, { T, params, rng }) {
        const n = new Float64Array(K), w = new Float64Array(K), act = new Uint8Array(K).fill(1);
        const logT = Math.log(K * T);
        let left = K, winner = -1;
        return {
          select() {
            if (winner >= 0) return [winner, winner];
            const i = bestOf(K, rng, (a) => -n[a], (a) => act[a]);
            let j = rng.int(K - 1);
            if (j >= i) j++;
            return [i, j];
          },
          update(i, j, won) {
            if (winner >= 0 || i === j) return;
            n[i]++; w[i] += won;
            let lcb = -Infinity;
            for (let a = 0; a < K; a++) {
              if (!act[a]) continue;
              if (n[a] === 0) return;
              const v = w[a] / n[a] - params.beta * Math.sqrt(logT / (2 * n[a]));
              if (v > lcb) lcb = v;
            }
            for (let a = 0; a < K; a++) {
              if (act[a] && w[a] / n[a] + params.beta * Math.sqrt(logT / (2 * n[a])) < lcb) { act[a] = 0; left--; }
            }
            if (left === 1) winner = act.indexOf(1);
          },
        };
      },
    },

    // ---- gradient
    {
      id: 'dbgd', name: 'DBGD', family: 'gradient',
      params: [
        { key: 'delta', label: 'Exploration radius δ', default: 0.3, min: 0.05, max: 1, step: 0.01 },
        { key: 'gamma', label: 'Step size γ', default: 0.02, min: 0.005, max: 0.5, step: 0.005 },
      ],
      formula: 'w′ = Π(w + δu), u ~ Unif(S^{d−1}); duel (w, w′); if w′ wins: w ← Π(w + γu)',
      regret: 'O(√d T^{3/4}) (smooth concave utility)',
      idea: 'Dueling Bandit Gradient Descent (Yue & Joachims 2009) climbs a utility over a continuous parameter space by comparing the current point with a random perturbation. On the continuous environment it plays points of the unit ball; elsewhere it runs over a linear scorer with one-hot arm features (w ∈ ℝᴷ, arm = argmax w), a stated discrete adaptation.',
      create(K, { params, rng, info }) {
        const space = info && info.space;
        const d = space ? space.d : K;
        const w = new Float64Array(d), w2 = new Float64Array(d), u = new Float64Array(d);
        const proj = (v) => {
          let s = 0;
          for (let k = 0; k < d; k++) s += v[k] * v[k];
          if (s > 1) { s = 1 / Math.sqrt(s); for (let k = 0; k < d; k++) v[k] *= s; }
        };
        let a = -1, b = -1;
        return {
          select() {
            let s = 0;
            for (let k = 0; k < d; k++) { u[k] = rng.normal(); s += u[k] * u[k]; }
            s = 1 / Math.sqrt(s);
            for (let k = 0; k < d; k++) { u[k] *= s; w2[k] = w[k] + params.delta * u[k]; }
            proj(w2);
            if (space) return [w, w2];
            a = bestOf(K, rng, (k) => w[k]);
            b = bestOf(K, rng, (k) => w2[k]);
            return [a, b];
          },
          update(x, y, won) {
            if (won || (!space && a === b)) return;
            for (let k = 0; k < d; k++) w[k] += params.gamma * u[k];
            proj(w);
          },
        };
      },
    },
  ];

  // ------------------------------------------------------- environments ----
  // build(K, s, rng) fills P (canonical labels; arm 0 is the intended winner) with strength s.
  const environments = [
    {
      id: 'bt', name: 'Bradley–Terry (Condorcet)', target: 'condorcet', tags: ['condorcet', 'utility'],
      description: 'Utility-based preferences P(i ≻ j) = σ(vᵢ − vⱼ) (the pairwise marginals of a Plackett–Luce model). The highest-utility arm is the Condorcet winner and preferences are transitive. Regret: P(w ≻ i) − ½.',
      build(K, s, rng, P) {
        const v = new Float64Array(K);
        for (let i = 1; i < K; i++) v[i] = -s * (0.3 + 1.2 * rng.uniform());
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) P[i * K + j] = i === j ? 0.5 : sigmoid(v[i] - v[j]);
      },
    },
    {
      id: 'cyclic', name: 'Condorcet with cycles', target: 'condorcet', tags: ['condorcet', 'non-transitive'],
      description: 'A Condorcet winner beats every arm by a narrow margin (5–20 points) while the other arms form a strong rock-paper-scissors tournament among themselves. Breaks the transitivity that IF and BTM rely on. Regret: P(w ≻ i) − ½.',
      build(K, s, rng, P) {
        const m = K - 1;
        P.fill(0.5);
        for (let j = 1; j < K; j++) { const p = clampP(0.5 + s * (0.05 + 0.15 * rng.uniform())); P[j] = p; P[j * K] = 1 - p; }
        for (let a = 1; a < K; a++) {
          for (let b = a + 1; b < K; b++) {
            const k = (b - a) % m, margin = s * (0.15 + 0.25 * rng.uniform());
            const aWins = 2 * k < m ? true : 2 * k > m ? false : rng.uniform() < 0.5;
            const p = clampP(0.5 + (aWins ? 1 : -1) * (2 * k === m ? margin / 4 : margin));
            P[a * K + b] = p; P[b * K + a] = 1 - p;
          }
        }
      },
    },
    {
      id: 'copeland', name: 'Copeland (no Condorcet winner)', target: 'copeland', tags: ['copeland', 'non-transitive'],
      description: 'Synthetic version of the MSLR/Yandex ranker cycles: three top arms A ≻ B ≻ C ≻ A beat the rest, but the two weakest arms ("spoilers") beat B and C by 15–25 points, so A has the most wins without beating everyone. Regret: normalised Copeland-score gap (Cpld(w) − Cpld(i))/(K − 1).',
      build(K, s, rng, P) {
        P.fill(0.5);
        const set = (i, j, p) => { p = clampP(p); P[i * K + j] = p; P[j * K + i] = 1 - p; };
        const rest = [];
        for (let i = 3; i < K; i++) rest.push(i);
        const v = new Float64Array(K);
        for (const i of rest) v[i] = -s * 2 * rng.uniform();
        rest.sort((x, y) => v[x] - v[y]); // weakest first
        for (let a = 0; a < rest.length; a++) for (let b = a + 1; b < rest.length; b++) set(rest[a], rest[b], sigmoid(v[rest[a]] - v[rest[b]]));
        const top = () => 0.5 + s * (0.15 + 0.15 * rng.uniform());
        const beat = () => 0.5 + s * (0.2 + 0.15 * rng.uniform());
        const spoil = () => 0.5 - s * (0.15 + 0.1 * rng.uniform());
        set(0, 1, top()); set(1, 2, top()); set(2, 0, top());
        const spoilB = rest.length ? rest[0] : -1;
        const spoilC = rest.length > 1 ? [rest[0], rest[1]] : rest.slice(0, 1);
        for (const r of rest) {
          set(0, r, beat());
          set(1, r, r === spoilB ? spoil() : beat());
          set(2, r, spoilC.includes(r) ? spoil() : beat());
        }
      },
    },
    {
      id: 'borda', name: 'Borda ≠ Condorcet', target: 'borda', tags: ['borda'],
      description: 'A Condorcet winner beats everyone by only 2–8 points, while a second arm loses to it narrowly but crushes all others (80–90%), so the Borda winner (best average win rate) differs from the Condorcet winner. Regret: Borda-score gap b(w) − b(i), b(i) = Σⱼ P(i ≻ j)/(K − 1).',
      build(K, s, rng, P) {
        P.fill(0.5);
        const set = (i, j, p) => { p = clampP(p); P[i * K + j] = p; P[j * K + i] = 1 - p; };
        const v = new Float64Array(K);
        for (let i = 2; i < K; i++) v[i] = -s * 1.5 * rng.uniform();
        for (let a = 2; a < K; a++) for (let b = a + 1; b < K; b++) set(a, b, sigmoid(v[a] - v[b]));
        set(0, 1, 0.5 + s * (0.02 + 0.06 * rng.uniform()));
        for (let r = 2; r < K; r++) {
          set(0, r, 0.5 + s * (0.02 + 0.06 * rng.uniform()));
          set(1, r, 0.5 + s * (0.3 + 0.1 * rng.uniform()));
        }
      },
    },
    {
      id: 'continuous', name: 'Continuous concave utility', target: 'condorcet', continuous: true, d: 5,
      tags: ['condorcet', 'continuous', 'utility'],
      description: 'Arms are points in the unit ball of ℝ⁵ with utility v(x) = −κ‖x − x*‖² and P(x ≻ y) = σ(v(x) − v(y)). One of the K arms sits at the optimum x*. DBGD searches the whole ball; every other method duels the K arms. Regret: P(x* ≻ x) − ½.',
      build(K, s, rng, P, space) {
        const d = space.d, X = space.arms, c = space.center;
        const unit = (out, o, r) => {
          let q = 0;
          for (let k = 0; k < d; k++) { out[o + k] = rng.normal(); q += out[o + k] * out[o + k]; }
          q = r / Math.sqrt(q);
          for (let k = 0; k < d; k++) out[o + k] *= q;
        };
        space.kappa = 2 * s;
        unit(c, 0, 0.3 + 0.6 * rng.uniform());
        for (let k = 0; k < d; k++) X[k] = c[k];
        for (let i = 1; i < K; i++) unit(X, i * d, Math.pow(rng.uniform(), 1 / d));
        const v = new Float64Array(K);
        for (let i = 0; i < K; i++) {
          let q = 0;
          for (let k = 0; k < d; k++) { const z = X[i * d + k] - c[k]; q += z * z; }
          v[i] = -space.kappa * q;
        }
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) P[i * K + j] = i === j ? 0.5 : sigmoid(v[i] - v[j]);
      },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of environments) ENV_BY_ID[e.id] = e;
  const ALGO_BY_ID = {};
  for (const a of algorithms) ALGO_BY_ID[a.id] = a;

  /** Per-arm regret rᵢ ≥ 0 against the target winner (0 for winners). */
  function armRegret(P, K, target, r) {
    if (target === 'condorcet') {
      for (let w = 0; w < K; w++) {
        let ok = true;
        for (let j = 0; j < K; j++) if (j !== w && P[w * K + j] <= 0.5) { ok = false; break; }
        if (ok) { for (let i = 0; i < K; i++) r[i] = P[w * K + i] - 0.5; return; }
      }
      target = 'copeland'; // no Condorcet winner drawn: score as Copeland
    }
    let best = -Infinity;
    for (let i = 0; i < K; i++) {
      let s = 0;
      for (let j = 0; j < K; j++) if (j !== i) s += target === 'borda' ? P[i * K + j] : P[i * K + j] > 0.5 ? 1 : 0;
      r[i] = s / (K - 1);
      if (r[i] > best) best = r[i];
    }
    for (let i = 0; i < K; i++) r[i] = Math.max(0, best - r[i]);
  }

  function defaultParams(def) {
    const out = {};
    for (const p of def.params) out[p.key] = p.default;
    return out;
  }

  // ------------------------------------------------------------- runner ----
  function createRunner(config) {
    const env = ENV_BY_ID[config.env];
    if (!env) throw new Error('Unknown dueling environment: ' + config.env);
    const K = config.K | 0, T = config.T | 0, runs = config.runs | 0, seed = config.seed >>> 0;
    if (K < 3) throw new Error('K must be at least 3');
    const extra = config.extra || {};
    const strength = extra.strength != null ? +extra.strength : 1;
    const NP = (K * (K - 1)) / 2;
    const pidx = new Int32Array(K * K);
    for (let i = 0, p = 0; i < K; i++) for (let j = i + 1; j < K; j++) { pidx[i * K + j] = p; pidx[j * K + i] = p; p++; }
    const Utab = new Float32Array(T * NP), S = new Float64Array(T);
    const P0 = new Float64Array(K * K), P = new Float64Array(K * K), r0 = new Float64Array(K), r = new Float64Array(K);
    const strong = new Float64Array(K * K), weak = new Float64Array(K * K), winPair = new Uint8Array(K * K);
    const D = env.d || 0;
    const space = env.continuous ? { d: D, arms: new Float64Array(K * D), center: new Float64Array(D), kappa: 1 } : null;
    const envView = { id: env.id, continuous: !!env.continuous };
    const info = space ? { space: { d: D } } : {};

    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown dueling algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaultParams(def), spec.params || {}),
        regSum: new Float64Array(T), regSq: new Float64Array(T),
        weakSum: new Float64Array(T), weakSq: new Float64Array(T), winSum: new Float64Array(T),
        finals: [], weakFinals: [], timeMs: 0,
      };
    });
    let done = 0;

    // Continuous points: regret and preference from the utility.
    function pointGap(x) {
      let q = 0;
      for (let k = 0; k < D; k++) { const z = x[k] - space.center[k]; q += z * z; }
      return space.kappa * q;
    }

    function runOne(run) {
      const rng = Rng(hash32(seed, run, 'env'));
      env.build(K, strength, rng, P0, space);
      armRegret(P0, K, env.target, r0);
      // shuffle labels so arm indices carry no information
      const perm = Array.from({ length: K }, (_, i) => i);
      for (let i = K - 1; i > 0; i--) { const j = rng.int(i + 1); const z = perm[i]; perm[i] = perm[j]; perm[j] = z; }
      for (let i = 0; i < K; i++) {
        r[perm[i]] = r0[i];
        for (let j = 0; j < K; j++) P[perm[i] * K + perm[j]] = P0[i * K + j];
      }
      let minGap = Infinity;
      for (let i = 0; i < K; i++) if (r[i] > 0 && r[i] < minGap) minGap = r[i];
      const tol = Number.isFinite(minGap) ? minGap / 2 : 0;
      for (let i = 0; i < K; i++) {
        for (let j = 0; j < K; j++) {
          strong[i * K + j] = (r[i] + r[j]) / 2;
          weak[i * K + j] = Math.min(r[i], r[j]);
          winPair[i * K + j] = i === j && r[i] === 0 ? 1 : 0;
        }
      }
      for (let k = 0; k < Utab.length; k++) Utab[k] = rng.uniform();
      for (let t = 0; t < T; t++) S[t] = rng.uniform();

      for (const slot of slots) {
        const arng = Rng(hash32(seed, run, slot.def.id));
        const agent = slot.def.create(K, { T, params: slot.params, rng: arng, env: envView, info });
        const { regSum, regSq, weakSum, weakSq, winSum } = slot;
        let cum = 0, cumW = 0;
        const t0 = now();
        for (let t = 0; t < T; t++) {
          const pair = agent.select(t + 1);
          const a = pair[0], b = pair[1];
          let won, rs, rw, wp;
          if (typeof a === 'number') {
            if (a === b) won = S[t] < 0.5 ? 1 : 0;
            else {
              const u = Utab[t * NP + pidx[a * K + b]];
              won = a < b ? (u < P[a * K + b] ? 1 : 0) : (u < P[b * K + a] ? 0 : 1);
            }
            const k = a * K + b;
            rs = strong[k]; rw = weak[k]; wp = winPair[k];
          } else {
            const ga = pointGap(a), gb = pointGap(b);
            won = S[t] < sigmoid(gb - ga) ? 1 : 0;
            const ra = sigmoid(ga) - 0.5, rb = sigmoid(gb) - 0.5;
            rs = (ra + rb) / 2; rw = Math.min(ra, rb); wp = ra <= tol && rb <= tol ? 1 : 0;
          }
          agent.update(a, b, won, t + 1);
          cum += rs; cumW += rw;
          regSum[t] += cum; regSq[t] += cum * cum;
          weakSum[t] += cumW; weakSq[t] += cumW * cumW;
          winSum[t] += wp;
        }
        slot.timeMs += now() - t0;
        slot.finals.push(cum);
        slot.weakFinals.push(cumW);
      }
    }

    function meanSE(sum, sq, R) {
      const m = new Float64Array(T), se = new Float64Array(T);
      for (let t = 0; t < T; t++) {
        const mu = sum[t] / R;
        m[t] = mu;
        se[t] = R > 1 ? Math.sqrt(Math.max(0, sq[t] / R - mu * mu) / (R - 1)) : 0;
      }
      return [m, se];
    }

    return {
      config: { env: env.id, K, T, runs, seed, extra: { strength } },
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
          const [regret, regretSE] = meanSE(s.regSum, s.regSq, R);
          const [weakR, weakSE] = meanSE(s.weakSum, s.weakSq, R);
          const winner = new Float64Array(T);
          let winAll = 0, winTail = 0;
          const tail = Math.max(1, Math.floor(T / 10));
          for (let t = 0; t < T; t++) {
            winner[t] = s.winSum[t] / R;
            winAll += winner[t] / T;
            if (t >= T - tail) winTail += winner[t] / tail;
          }
          const sorted = s.finals.slice().sort((x, y) => x - y);
          const summary = {
            finalRegret: regret[T - 1], finalSE: regretSE[T - 1],
            finalMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            finalWeak: weakR[T - 1], finalWeakSE: weakSE[T - 1],
            winnerTail: winTail, winnerShare: winAll,
            msPerRun: done ? s.timeMs / done : 0,
          };
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, params: s.params, runs: done,
            curves: { regret, weak: weakR, winner },
            curvesSE: { regret: regretSE, weak: weakSE },
            summary, finals: sorted,
          };
        });
      },
    };
  }

  // ----------------------------------------------------------- register ----
  return MAB.registerSetting({
    id: 'dueling',
    name: 'Dueling bandits',
    short: 'Dueling',
    blurb: 'Pick two arms each round and see only which one won. Regret is measured against the Condorcet, Copeland or Borda winner.',
    families, environments, algorithms,
    defaultEnv: 'bt',
    defaultLineup: ['uniform', 'if', 'btm', 'rucb', 'rmed2', 'dts', 'sparring-ucb', 'ccb'],
    presets: [
      { label: 'Condorcet classics', algos: ['if', 'btm', 'savage', 'rucb', 'rmed1', 'dts', 'uniform'], env: 'bt' },
      { label: 'RMED family', algos: ['rmed1', 'rmed2', 'rmed2fh', 'rucb', 'dts'] },
      { label: 'Reductions to MAB', algos: ['sparring-ucb', 'sparring-exp3', 'vdb', 'doubler', 'multisbm', 'rucb', 'uniform'] },
      { label: 'Cycles', algos: ['if', 'btm', 'savage', 'rucb', 'rmed1', 'dts', 'sparring-ucb', 'uniform'], env: 'cyclic' },
      { label: 'Copeland winners', algos: ['ccb', 'cw-rmed', 'dts', 'rucb', 'rmed1', 'savage', 'uniform'], env: 'copeland' },
      { label: 'Borda winners', algos: ['borda-se', 'vdb', 'sparring-exp3', 'rucb', 'dts', 'ccb', 'uniform'], env: 'borda' },
      { label: 'Gradient (continuous)', algos: ['dbgd', 'rucb', 'dts', 'sparring-ucb', 'vdb', 'uniform'], env: 'continuous' },
    ],
    controls: {
      K: { label: 'Arms (K)', min: 4, max: 20, default: 8 },
      T: { label: 'Horizon (duels)', options: [500, 1000, 2000, 5000], default: 2000 },
      runs: { label: 'Runs averaged', options: [10, 20, 50, 100, 200], default: 50 },
      extra: [{ key: 'strength', label: 'Preference strength', min: 0.3, max: 3, step: 0.1, default: 1 }],
    },
    xLabel: 'duel t',
    metrics: [
      {
        key: 'regret', label: 'Strong regret', better: 'lower', fmt: 'num', band: true,
        note: 'Cumulative Σ (rᵢ + rⱼ)/2 over the played pairs, with rᵢ = P(w ≻ i) − ½ against the Condorcet winner w, the normalised Copeland-score gap (Copeland env) or the Borda-score gap (Borda env).',
      },
      {
        key: 'weak', label: 'Weak regret', better: 'lower', fmt: 'num', band: true,
        note: 'Cumulative Σ min(rᵢ, rⱼ): zero whenever either played arm is the winner.',
      },
      {
        key: 'winner', label: 'Winner-vs-winner rate', better: 'higher', fmt: 'pct', smooth: true, zeroToOne: true,
        note: 'Share of runs that duelled the winner against itself at each step (for DBGD\'s points: both within half the smallest arm gap of x*).',
      },
    ],
    final: { key: 'finalRegret', se: 'finalSE', median: 'finalMedian', label: 'Final strong regret', better: 'lower' },
    tiles: [
      { key: 'finalWeak', label: 'Lowest weak regret', better: 'lower', fmt: 'num', sub: 'cumulative, at T' },
      { key: 'winnerTail', label: 'Duels the winner most', better: 'higher', fmt: 'pct', sub: '(w, w) in the last 10% of duels' },
    ],
    columns: [
      { key: 'finalRegret', label: 'Strong regret', fmt: 'num', better: 'lower' },
      { key: 'finalSE', label: '95% CI', fmt: 'ci' },
      { key: 'finalMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'finalWeak', label: 'Weak regret', fmt: 'num', better: 'lower' },
      { key: 'winnerTail', label: '(w, w), last 10%', fmt: 'pct', better: 'higher' },
      { key: 'winnerShare', label: '(w, w), all', fmt: 'pct', better: 'higher' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: false,
    createRunner,
  });
});
