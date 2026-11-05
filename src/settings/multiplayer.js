/*
 * Setting: multi-player & federated bandits.
 *
 * M players (agents) act simultaneously on the same K arms every round.
 *
 * Collision environments (decentralized multi-player MAB, collision-sensing variant): players
 * that pick the same arm all get reward 0 and each is told it collided. Players do not talk
 * except through collisions. System regret(t) = Σ_{s≤t} [OPT − Σ_p μ_{p,a_p}·1{no collision}],
 * where OPT is the best assignment of players to distinct arms: the sum of the top-M means for
 * homogeneous players, a max-weight matching for player-specific means.
 *
 * Federated environments: there are no collisions; each agent has its own reward stream and the
 * agents (or a server) may exchange statistics. Regret(t) = Σ_{s≤t} Σ_p [max_a μ_{p,a,s} − μ_{p,a_p,s}],
 * the sum over agents of each agent's regret against its own best arm.
 *
 * In both cases regret is pseudo-regret (expected rewards), and "collisions" counts the
 * player-pulls that ended in a collision.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;

  // ------------------------------------------------------------ helpers ----
  /** Max-weight assignment of n rows to distinct columns (n ≤ m), Hungarian algorithm. */
  function assignMax(w, n, m) {
    const INF = 1e18;
    const u = new Float64Array(n + 1), v = new Float64Array(m + 1), p = new Int32Array(m + 1), way = new Int32Array(m + 1);
    const minv = new Float64Array(m + 1), used = new Uint8Array(m + 1);
    for (let i = 1; i <= n; i++) {
      p[0] = i;
      let j0 = 0;
      minv.fill(INF); used.fill(0);
      do {
        used[j0] = 1;
        const i0 = p[j0];
        let delta = INF, j1 = 0;
        for (let j = 1; j <= m; j++) {
          if (used[j]) continue;
          const cur = -w[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= m; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    let tot = 0;
    for (let j = 1; j <= m; j++) if (p[j]) tot += w[(p[j] - 1) * m + j - 1];
    return tot;
  }

  /** Mark the top-M entries of g (ties broken by index) into `inSet`; returns the M-th largest value. */
  const sortBuf = { a: new Float64Array(0) };
  function topM(g, K, M, inSet) {
    if (sortBuf.a.length !== K) sortBuf.a = new Float64Array(K);
    const s = sortBuf.a;
    s.set(g);
    s.sort();
    const thr = s[K - Math.min(M, K)];
    let cnt = 0;
    for (let a = 0; a < K; a++) { inSet[a] = g[a] > thr ? 1 : 0; cnt += inSet[a]; }
    for (let a = 0; a < K && cnt < M; a++) if (!inSet[a] && g[a] === thr) { inSet[a] = 1; cnt++; }
    return thr;
  }
  function pickFrom(inSet, K, rng, filter) {
    let cnt = 0, pick = -1;
    for (let a = 0; a < K; a++) if (inSet[a] && (!filter || filter(a))) { cnt++; if (rng.uniform() * cnt < 1) pick = a; }
    return pick;
  }
  /** kl-UCB upper bound max{q : n·kl(p, q) ≤ bound} by Newton steps from the Pinsker bound (monotone from above). */
  function klUcbFast(p, n, bound) {
    const delta = bound / n;
    let q = Math.min(p + Math.sqrt(delta / 2), 1 - 1e-9);
    if (q <= p) return p;
    for (let i = 0; i < 6; i++) {
      const f = util.klBern(p, q) - delta;
      if (f <= 1e-7) break;
      q -= f / ((q - p) / (q * (1 - q)));
      if (q <= p) return p;
    }
    return q;
  }
  function klucbIndex(n, s, t, out, off, K) {
    const lt = Math.log(Math.max(t, 2));
    for (let a = 0; a < K; a++) {
      const na = n[off + a];
      out[a] = na > 0 ? klUcbFast(s[off + a] / na, na, lt) : 1e9 + a * 1e-6;
    }
    return out;
  }
  function ucbIndex(n, s, t, c, out, off, K) {
    const lt = Math.log(Math.max(t, 2));
    for (let a = 0; a < K; a++) {
      const na = n[off + a];
      out[a] = na > 0 ? s[off + a] / na + c * Math.sqrt(lt / na) : 1e9 + a * 1e-6;
    }
    return out;
  }

  // ------------------------------------------------------------ families ----
  const families = [
    { id: 'baseline', name: 'Selfish baseline', blurb: 'Each player runs a single-player algorithm and ignores the others.' },
    { id: 'orthogonal', name: 'Orthogonalization', blurb: 'Randomize until every player sits on its own arm among the estimated top M.' },
    { id: 'communication', name: 'Implicit communication', blurb: 'Use deliberate collisions as bits to share statistics or decisions between players.' },
    { id: 'matching', name: 'Distributed matching', blurb: 'Reach the max-weight player–arm matching when means differ per player.' },
    { id: 'federated', name: 'Federated', blurb: 'No collisions: agents learn the same arms in parallel and periodically pool their statistics.' },
  ];

  // --------------------------------------------------------- environments ----
  // setup(K, M, T, rng) → { collide, dx, dynamic, mu: Float64Array(M*K), opt }
  // draw(inst, srng, Y, MT, X) fills this round's realized rewards Y[p*K+a]; dynamic
  // environments also fill means MT[p*K+a] and contexts X[(p*K+a)*dx+i] and return OPT.
  const environments = [
    {
      id: 'bernoulli', name: 'Bernoulli, homogeneous players', short: 'Homogeneous',
      description: 'K Bernoulli arms with means in [0.1, 0.9], the same for every player. Colliding players get 0 (and are told). Regret is against the M best arms each held by one player: OPT = sum of the top-M means.',
      tags: ['collisions', 'stochastic'],
      setup(K, M, T, rng) {
        const mu = new Float64Array(M * K), m1 = new Float64Array(K);
        for (let a = 0; a < K; a++) m1[a] = 0.1 + 0.8 * rng.uniform();
        for (let p = 0; p < M; p++) mu.set(m1, p * K);
        return { collide: true, dx: 0, dynamic: false, mu, opt: assignMax(mu, M, K), shared: true };
      },
      draw(inst, srng, Y, K, M) {
        const mu = inst.mu;
        for (let a = 0; a < K; a++) {
          const y = srng.uniform() < mu[a] ? 1 : 0;
          for (let p = 0; p < M; p++) Y[p * K + a] = y;
        }
      },
    },
    {
      id: 'hetero', name: 'Player-specific means', short: 'Heterogeneous',
      description: 'Each player has its own mean for each arm, drawn uniformly in [0.05, 0.95]. Collisions give 0. OPT is the max-weight matching of players to distinct arms (Hungarian algorithm) — the target of Game of Thrones.',
      tags: ['collisions', 'heterogeneous'],
      setup(K, M, T, rng) {
        const mu = new Float64Array(M * K);
        for (let i = 0; i < M * K; i++) mu[i] = 0.05 + 0.9 * rng.uniform();
        return { collide: true, dx: 0, dynamic: false, mu, opt: assignMax(mu, M, K) };
      },
      draw(inst, srng, Y, K, M) {
        const mu = inst.mu;
        for (let i = 0; i < M * K; i++) Y[i] = srng.uniform() < mu[i] ? 1 : 0;
      },
    },
    {
      id: 'fed-bernoulli', name: 'Federated Bernoulli (no collisions)', short: 'Federated',
      description: 'M agents each face the same K Bernoulli arms with independent reward draws and never collide. Regret is summed over agents, each against its own best arm (here the same arm for all), so pooling statistics helps everyone.',
      tags: ['federated', 'stochastic'],
      setup(K, M, T, rng) {
        const mu = new Float64Array(M * K), m1 = new Float64Array(K);
        let best = 0;
        for (let a = 0; a < K; a++) { m1[a] = 0.1 + 0.8 * rng.uniform(); if (m1[a] > best) best = m1[a]; }
        for (let p = 0; p < M; p++) mu.set(m1, p * K);
        return { collide: false, dx: 0, dynamic: false, mu, opt: M * best };
      },
      draw(inst, srng, Y, K, M) {
        const mu = inst.mu;
        for (let i = 0; i < M * K; i++) Y[i] = srng.uniform() < mu[i] ? 1 : 0;
      },
    },
    {
      id: 'fed-linear', name: 'Federated linear contextual', short: 'Fed. linear',
      description: 'Each round every agent sees K fresh 5-dimensional contexts x ∈ [0,1]⁵; reward = θ·x + N(0, 0.1²) with one θ shared by all agents (Σθ = 1, so means lie in [0, 1]). No collisions. Regret is summed over agents against each round\'s best context.',
      tags: ['federated', 'context'],
      contextual: true,
      setup(K, M, T, rng) {
        const dx = 5, theta = new Float64Array(dx);
        let s = 0;
        for (let i = 0; i < dx; i++) { theta[i] = rng.uniform(); s += theta[i]; }
        for (let i = 0; i < dx; i++) theta[i] /= s;
        return { collide: false, dx, dynamic: true, mu: null, opt: 0, theta };
      },
      draw(inst, srng, Y, K, M, MT, X) {
        const dx = inst.dx, th = inst.theta;
        let opt = 0;
        for (let p = 0; p < M; p++) {
          let best = -Infinity;
          for (let a = 0; a < K; a++) {
            const o = (p * K + a) * dx;
            let m = 0;
            for (let i = 0; i < dx; i++) { const v = srng.uniform(); X[o + i] = v; m += th[i] * v; }
            MT[p * K + a] = m;
            Y[p * K + a] = m + 0.1 * srng.normal();
            if (m > best) best = m;
          }
          opt += best;
        }
        return opt;
      },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of environments) ENV_BY_ID[e.id] = e;

  // ---------------------------------------------------------- algorithms ----
  // create(K, M, ctx) → { select(t, X, arms), update(t, arms, rew, coll) }
  // The object simulates M decentralized players; each player's choices depend only on its own
  // observations (and, for federated methods, on the statistics shared at synchronizations).
  const algorithms = [
    {
      id: 'selfish-ucb', name: 'Selfish UCB1', family: 'baseline',
      params: [{ key: 'c', label: 'Exploration c', default: 1.414, min: 0, max: 3, step: 0.01 }],
      formula: 'aₜᵖ = argmaxₐ μ̂ₐᵖ + c·√(ln t / nₐᵖ)  (collisions count as reward 0)',
      regret: 'no guarantee; can be linear',
      idea: 'Every player runs UCB1 on its own rewards, treating a collision as a 0 reward. Collisions push players apart, but slowly and with no coordination, so they keep colliding.',
      create(K, M, { params }) {
        const n = new Float64Array(M * K), s = new Float64Array(M * K), idx = new Float64Array(K);
        return {
          select(t, X, arms, rng) {
            for (let p = 0; p < M; p++) arms[p] = util.argmax(ucbIndex(n, s, t, params.c, idx, p * K, K), rng);
          },
          update(t, arms, rew) {
            for (let p = 0; p < M; p++) { n[p * K + arms[p]] += 1; s[p * K + arms[p]] += rew[p]; }
          },
        };
      },
    },
    {
      id: 'musical-chairs', name: 'Musical Chairs', family: 'orthogonal',
      params: [{ key: 'T0', label: 'Exploration rounds T₀', default: 300, min: 20, max: 5000, step: 10 }],
      formula: 't ≤ T₀: aₜ ~ U[K];  M̂ = 1 + ln((T₀−C)/T₀)/ln(1−1/K);  then sit on a random top-M̂ arm until no collision',
      regret: 'O(K log T / Δ²) w.h.p. (explore-then-commit)',
      idea: 'Rosenski, Shamir & Szlak (2016). Explore uniformly for T₀ rounds, estimating the means and, from the collision rate, the number of players. Then each player jumps between its top-M̂ arms until it gets a round without collision, and stays there forever.',
      create(K, M, { params }) {
        const T0 = Math.max(1, Math.round(params.T0));
        const n = new Float64Array(M * K), s = new Float64Array(M * K), C = new Float64Array(M);
        const fixed = new Int32Array(M).fill(-1), top = new Uint8Array(M * K), mean = new Float64Array(K);
        let ready = false;
        return {
          select(t, X, arms, rng) {
            if (t <= T0) { for (let p = 0; p < M; p++) arms[p] = rng.int(K); return; }
            if (!ready) {
              ready = true;
              for (let p = 0; p < M; p++) {
                const free = T0 - C[p];
                let Mh = free <= 0 ? K : Math.round(1 + Math.log(free / T0) / Math.log(1 - 1 / K));
                Mh = Math.min(K, Math.max(1, Mh));
                for (let a = 0; a < K; a++) mean[a] = n[p * K + a] ? s[p * K + a] / n[p * K + a] : 0;
                topM(mean, K, Mh, top.subarray(p * K, p * K + K));
              }
            }
            for (let p = 0; p < M; p++) arms[p] = fixed[p] >= 0 ? fixed[p] : pickFrom(top.subarray(p * K, p * K + K), K, rng);
          },
          update(t, arms, rew, coll) {
            for (let p = 0; p < M; p++) {
              if (t <= T0) {
                if (coll[p]) C[p] += 1;
                else { n[p * K + arms[p]] += 1; s[p * K + arms[p]] += rew[p]; }
              } else if (fixed[p] < 0 && !coll[p]) fixed[p] = arms[p];
            }
          },
        };
      },
    },
    {
      id: 'mctopm', name: 'MCTopM', family: 'orthogonal',
      params: [],
      formula: 'g = kl-UCB;  aₜ₊₁ ~ U(M̂ₜ ∩ {g_m(t−1) ≤ g_a(t−1)}) if aₜ ∉ M̂ₜ;  ~U(M̂ₜ) on collision unless seated;  else stay (seated)',
      regret: 'O(M³ Σ log T / Δ²)-type, log T',
      idea: 'Besson & Kaufmann (2018). Each player keeps kl-UCB indices g (on its non-colliding samples) and the estimated top-M set M̂. A player that collides re-draws uniformly in M̂ unless it is already "seated"; once seated it only leaves if its arm drops out of M̂. Knows M.',
      create(K, M) {
        const n = new Float64Array(M * K), s = new Float64Array(M * K);
        const g = new Float64Array(M * K), gPrev = new Float64Array(M * K), set = new Uint8Array(K);
        const cur = new Int32Array(M).fill(-1), seated = new Uint8Array(M), lastColl = new Uint8Array(M);
        const idx = new Float64Array(K);
        return {
          select(t, X, arms, rng) {
            gPrev.set(g);
            for (let p = 0; p < M; p++) {
              klucbIndex(n, s, t, idx, p * K, K);
              g.set(idx, p * K);
              topM(idx, K, M, set);
              const a = cur[p];
              if (a < 0) { cur[p] = rng.int(K); }
              else if (!set[a]) {
                const ga = gPrev[p * K + a];
                let b = pickFrom(set, K, rng, (m) => gPrev[p * K + m] <= ga);
                if (b < 0) b = pickFrom(set, K, rng);
                cur[p] = b; seated[p] = 0;
              } else if (lastColl[p] && !seated[p]) {
                cur[p] = pickFrom(set, K, rng); seated[p] = 0;
              } else seated[p] = 1;
              arms[p] = cur[p];
            }
          },
          update(t, arms, rew, coll) {
            for (let p = 0; p < M; p++) {
              lastColl[p] = coll[p];
              if (!coll[p]) { n[p * K + arms[p]] += 1; s[p * K + arms[p]] += rew[p]; }
            }
          },
        };
      },
    },
    {
      id: 'randtopm', name: 'RandTopM', family: 'orthogonal',
      params: [],
      formula: 'on collision: aₜ₊₁ ~ U(M̂ₜ);  if aₜ ∉ M̂ₜ: aₜ₊₁ ~ U(M̂ₜ ∩ {g_m(t−1) ≤ g_a(t−1)});  else stay',
      regret: 'O(Σ log T / Δ²)-type, log T',
      idea: 'Besson & Kaufmann (2018), with kl-UCB indices. Like MCTopM without the "seated" state: every collision triggers a uniform redraw inside the estimated top-M set. Simpler, but keeps colliding a bit longer. Knows M.',
      create(K, M) {
        const n = new Float64Array(M * K), s = new Float64Array(M * K);
        const g = new Float64Array(M * K), gPrev = new Float64Array(M * K), set = new Uint8Array(K);
        const cur = new Int32Array(M).fill(-1), lastColl = new Uint8Array(M), idx = new Float64Array(K);
        return {
          select(t, X, arms, rng) {
            gPrev.set(g);
            for (let p = 0; p < M; p++) {
              klucbIndex(n, s, t, idx, p * K, K);
              g.set(idx, p * K);
              topM(idx, K, M, set);
              const a = cur[p];
              if (a < 0) cur[p] = rng.int(K);
              else if (lastColl[p]) cur[p] = pickFrom(set, K, rng);
              else if (!set[a]) {
                const ga = gPrev[p * K + a];
                let b = pickFrom(set, K, rng, (m) => gPrev[p * K + m] <= ga);
                if (b < 0) b = pickFrom(set, K, rng);
                cur[p] = b;
              }
              arms[p] = cur[p];
            }
          },
          update(t, arms, rew, coll) {
            for (let p = 0; p < M; p++) {
              lastColl[p] = coll[p];
              if (!coll[p]) { n[p * K + arms[p]] += 1; s[p * K + arms[p]] += rew[p]; }
            }
          },
        };
      },
    },
    {
      id: 'sic-mmab', name: 'SIC-MMAB (simplified)', family: 'communication',
      params: [{ key: 'T0c', label: 'Init length T₀ / (K ln T)', default: 1, min: 0.1, max: 5, step: 0.1 }],
      formula: 'phase p: hop over active arms K_p·2ᵖ rounds → send (p+1)-bit means by collisions → accept/reject with √(2 ln T / s)',
      regret: 'O(Σ_{k>M} log T / Δₖ + M K log T)',
      idea: 'Boursier & Perchet (2019). After musical chairs, players explore active arms by sequential hopping (no collisions), then send their quantized means to each other bit by bit: a deliberate collision is a 1. Everyone then accepts sure-optimal arms (a player sits on each) and rejects sure-suboptimal ones. Simplified: the M/rank estimation is charged as 2K collision-free rounds.',
      create(K, M, { params, T }) {
        const T0 = Math.max(K, Math.ceil(params.T0c * K * Math.log(Math.max(T, 3))));
        const lnT = Math.log(Math.max(T, 3));
        const n = new Float64Array(M * K), s = new Float64Array(M * K);
        const home = new Int32Array(M).fill(-1);      // musical-chairs seat
        const fixedArm = new Int32Array(M).fill(-1);  // exploitation arm once accepted
        let stage = 'init', clock = 0, initEnd = -1;
        let active = [], players = [], p = 0, L = 0, CL = 0, Q = null;
        const sig = new Uint8Array(M);
        const sumG = new Float64Array(K), cntG = new Float64Array(K);
        function startPhase() {
          p += 1; clock = 0;
          L = active.length * Math.pow(2, p);
          const Mp = players.length;
          CL = Mp > 1 ? Mp * (Mp - 1) * active.length * (p + 1) : 0;
          stage = 'explore';
        }
        function decide() {
          const Kp = active.length, Mp = players.length;
          const levels = Math.pow(2, p + 1) - 1;
          sumG.fill(0); cntG.fill(0);
          for (const q of players) for (let i = 0; i < Kp; i++) {
            const k = active[i], nn = n[q * K + k];
            if (!nn) continue;
            const qm = Math.round((s[q * K + k] / nn) * levels) / levels;   // what was transmitted
            sumG[k] += qm * nn; cntG[k] += nn;
          }
          const mean = (k) => (cntG[k] ? sumG[k] / cntG[k] : 0.5);
          const B = (k) => (cntG[k] ? Math.sqrt(2 * lnT / cntG[k]) : 1);
          const acc = [], rej = [];
          for (const k of active) {
            let above = 0, below = 0;
            for (const i of active) {
              if (i === k) continue;
              if (mean(i) - B(i) >= mean(k) + B(k)) above++;
              if (mean(i) + B(i) <= mean(k) - B(k)) below++;
            }
            if (above >= Mp) rej.push(k);
            else if (below >= Kp - Mp) acc.push(k);
          }
          if (Kp - rej.length <= Mp) {           // every surviving arm is optimal
            for (const k of active) if (!rej.includes(k) && !acc.includes(k)) acc.push(k);
          }
          acc.sort((x, y) => mean(y) - mean(x));
          for (const k of acc) { const q = players.pop(); if (q === undefined) break; fixedArm[q] = k; }
          active = active.filter((k) => !acc.includes(k) && !rej.includes(k));
          if (players.length && active.length) startPhase();
          else stage = 'exploit';
        }
        return {
          signal: sig,
          select(t, X, arms, rng) {
            if (stage === 'init') {
              let all = true;
              for (let q = 0; q < M; q++) if (home[q] < 0) all = false;
              if (all && initEnd < 0 && t > T0) initEnd = t + 2 * K;   // rank / M estimation by sequential hopping
              if (initEnd >= 0 && t >= initEnd) {
                // ranks: order of the seats; the active arms are all K arms
                players = Array.from({ length: M }, (_, q) => q).sort((a, b) => home[a] - home[b]);
                active = Array.from({ length: K }, (_, k) => k);
                startPhase();
              } else {
                for (let q = 0; q < M; q++) arms[q] = home[q] >= 0 ? home[q] : rng.int(K);
                return;
              }
            }
            for (let q = 0; q < M; q++) if (fixedArm[q] >= 0) arms[q] = fixedArm[q];
            if (stage === 'exploit') return;
            const Kp = active.length, Mp = players.length;
            if (stage === 'explore' && clock >= L) {
              stage = CL > 0 ? 'comm' : 'decide'; clock = 0;
              if (stage === 'comm') {
                const levels = Math.pow(2, p + 1) - 1;
                Q = players.map((q) => active.map((k) => (n[q * K + k] ? Math.round((s[q * K + k] / n[q * K + k]) * levels) : 0)));
              }
            }
            if (stage === 'comm' && clock >= CL) { stage = 'decide'; }
            if (stage === 'decide') {
              decide();
              for (let q = 0; q < M; q++) if (fixedArm[q] >= 0) arms[q] = fixedArm[q];
              if (stage === 'exploit') return;
            }
            const K2 = active.length, M2 = players.length;
            if (stage === 'explore') {
              for (let j = 0; j < M2; j++) arms[players[j]] = active[(j + clock) % K2];
            } else {
              // communication round: sender i sends bit b of arm k's quantized sum to receiver r
              const per = Kp * (p + 1);
              const pair = Math.floor(clock / per), rem = clock % per, ki = Math.floor(rem / (p + 1)), bit = rem % (p + 1);
              const i = Math.floor(pair / (Mp - 1));
              let r = pair % (Mp - 1);
              if (r >= i) r++;
              for (let j = 0; j < Mp; j++) arms[players[j]] = active[j];
              if ((Q[i][ki] >> bit) & 1) { arms[players[i]] = active[r]; sig[players[i]] = 1; sig[players[r]] = 1; }
            }
            clock++;
          },
          update(t, arms, rew, coll) {
            for (let q = 0; q < M; q++) {
              if (stage === 'init' && home[q] < 0 && !coll[q]) home[q] = arms[q];
              if (!coll[q] && (stage === 'init' || stage === 'explore')) { n[q * K + arms[q]] += 1; s[q * K + arms[q]] += rew[q]; }
            }
          },
        };
      },
    },
    {
      id: 'dpe', name: 'DPE (simplified)', family: 'communication',
      params: [],
      formula: 'followers hop over the leader\'s top-M set Ŝ;  leader, in the slot of min Ŝ, explores k ∉ Ŝ w.p. ½ if kl-UCBₖ ≥ μ̂_{min Ŝ}',
      regret: 'asymptotically optimal: Σ_{k>M} (μ_M − μₖ) log T / kl(μₖ, μ_M)',
      idea: 'Wang, Proutière, Ariu, Jedra & Russo (2020), DPE1. After an orthogonalization phase a leader does all the (parsimonious) exploration with kl-UCB while followers just rotate over the leader\'s current top-M set; when that set changes the leader tells each follower with a deliberate collision. Simplified: one signalling collision per follower per change; decoding the new set is free.',
      create(K, M, { T }) {
        const T0 = Math.max(K, Math.ceil(K * Math.log(Math.max(T, 3))));
        const nq = new Float64Array(M * K), sq = new Float64Array(M * K);   // per-player stats during init
        const n = new Float64Array(K), s = new Float64Array(K), mean = new Float64Array(K);  // the leader's stats
        const home = new Int32Array(M).fill(-1), set = new Uint8Array(K);
        let stage = 'init', initEnd = -1, order = null, S = null, tau = 0, comm = 0, pending = null, leader = 0, leaderArm = -1;
        const sig = new Uint8Array(M);
        function topSet() {
          for (let a = 0; a < K; a++) mean[a] = n[a] ? s[a] / n[a] : 0;
          topM(mean, K, M, set);
          const out = [];
          for (let a = 0; a < K; a++) if (set[a]) out.push(a);
          return out;
        }
        return {
          signal: sig,
          select(t, X, arms, rng) {
            if (stage === 'init') {
              let all = true;
              for (let q = 0; q < M; q++) if (home[q] < 0) all = false;
              if (all && initEnd < 0 && t > T0) initEnd = t + 2 * K;   // rank estimation, charged as 2K rounds
              if (initEnd < 0 || t < initEnd) {
                for (let q = 0; q < M; q++) arms[q] = home[q] >= 0 ? home[q] : rng.int(K);
                return;
              }
              order = Array.from({ length: M }, (_, q) => q).sort((x, y) => home[x] - home[y]);
              leader = order[0];
              for (let a = 0; a < K; a++) { n[a] = nq[leader * K + a]; s[a] = sq[leader * K + a]; }
              S = topSet();
              stage = 'run';
            }
            const Ms = S.length;
            for (let j = 0; j < M; j++) arms[order[j]] = S[(j + tau) % Ms];
            leaderArm = -1;
            if (comm > 0) {                         // the leader collides with follower number comm
              arms[leader] = arms[order[comm]];
              sig[leader] = 1; sig[order[comm]] = 1;
              comm--;
              if (comm === 0 && pending) { S = pending; pending = null; }
              tau++;
              return;
            }
            // the leader's exploration slot: its turn on the weakest arm of S
            let worst = S[0];
            for (const a of S) if (mean[a] < mean[worst]) worst = a;
            if (arms[leader] === worst && K > Ms) {
              const lt = Math.log(Math.max(tau, 2));
              const cand = [];
              for (let k = 0; k < K; k++) {
                if (set[k]) continue;
                const ix = n[k] ? util.klUcbBern(mean[k], n[k], lt) : 1;
                if (ix >= mean[worst]) cand.push(k);
              }
              if (cand.length && rng.uniform() < 0.5) arms[leader] = cand[rng.int(cand.length)];
            }
            leaderArm = arms[leader];
            tau++;
          },
          update(t, arms, rew, coll) {
            if (stage === 'init') {
              for (let q = 0; q < M; q++) {
                if (coll[q]) continue;
                if (home[q] < 0) home[q] = arms[q];
                nq[q * K + arms[q]] += 1; sq[q * K + arms[q]] += rew[q];
              }
              return;
            }
            if (leaderArm >= 0 && !coll[leader]) { n[leaderArm] += 1; s[leaderArm] += rew[leader]; }
            if (pending || comm > 0) return;
            const S2 = topSet();
            let same = S2.length === S.length;
            if (same) for (let i = 0; i < S2.length; i++) if (S2[i] !== S[i]) same = false;
            if (!same) {
              if (M > 1) { pending = S2; comm = M - 1; } else S = S2;
            }
          },
        };
      },
    },
    {
      id: 'got', name: 'Game of Thrones (simplified)', family: 'matching',
      params: [
        { key: 'eps', label: 'Noise ε', default: 0.05, min: 0.001, max: 0.5, step: 0.001 },
        { key: 'kappa', label: 'Sharpness κ (1 = paper)', default: 3, min: 1, max: 20, step: 0.5 },
        { key: 'c1', label: 'Explore rounds / epoch', default: 100, min: 10, max: 2000, step: 10 },
        { key: 'c2', label: 'GoT rounds / epoch', default: 500, min: 10, max: 5000, step: 10 },
        { key: 'c3', label: 'First exploit length', default: 200, min: 10, max: 5000, step: 10 },
      ],
      formula: 'epoch k: explore c₁ → GoT dynamics c₂ (content: keep ā w.p. 1−ε^M; else content w.p. (u/u_max)·ε^{κ(u_max−u)}) → exploit c₃·2^{k−1}',
      regret: 'O(log^{1+δ} T)',
      idea: 'Bistritz & Leshem (2018). Fully distributed max-weight matching when each player has its own means. After an exploration phase, players run noisy "content/discontent" dynamics whose stationary distribution concentrates on the optimal matching, then exploit the arm they were most often content with. Simplified: fixed phase lengths, content counts over the last half of each GoT phase, and a sharper acceptance ε^{κ(u_max−u)} (κ = 3; the paper\'s κ = 1 needs far longer horizons to mix).',
      create(K, M, { params }) {
        const n = new Float64Array(M * K), s = new Float64Array(M * K), mu = new Float64Array(M * K), umax = new Float64Array(M);
        const content = new Uint8Array(M), base = new Int32Array(M), baseU = new Float64Array(M);
        const count = new Float64Array(M * K), exploit = new Int32Array(M);
        const c1 = Math.round(params.c1), c2 = Math.round(params.c2), c3 = Math.round(params.c3);
        const kappa = params.kappa;
        const stayP = 1 - Math.pow(params.eps, M);   // c = M experimentation exponent
        let epoch = 1, phase = 'explore', clock = 0;
        function advance() {
          clock++;
          if (phase === 'explore' && clock >= c1) {
            phase = 'got'; clock = 0;
            for (let i = 0; i < M * K; i++) mu[i] = n[i] ? s[i] / n[i] : 0;
            for (let p = 0; p < M; p++) { let m = 1e-9; for (let a = 0; a < K; a++) m = Math.max(m, mu[p * K + a]); umax[p] = m; content[p] = 0; }
            count.fill(0);
          } else if (phase === 'got' && clock >= c2) {
            phase = 'exploit'; clock = 0;
            for (let p = 0; p < M; p++) {
              let b = base[p], bv = 0;
              for (let a = 0; a < K; a++) if (count[p * K + a] > bv) { bv = count[p * K + a]; b = a; }
              exploit[p] = b;
            }
          } else if (phase === 'exploit' && clock >= c3 * Math.pow(2, epoch - 1)) {
            phase = 'explore'; clock = 0; epoch++;
          }
        }
        return {
          select(t, X, arms, rng) {
            for (let p = 0; p < M; p++) {
              if (phase === 'explore') arms[p] = rng.int(K);
              else if (phase === 'exploit') arms[p] = exploit[p];
              else if (content[p] && rng.uniform() < stayP) arms[p] = base[p];
              else if (content[p] && K > 1) { let a = rng.int(K - 1); if (a >= base[p]) a++; arms[p] = a; }
              else arms[p] = rng.int(K);
            }
          },
          update(t, arms, rew, coll, rng) {
            for (let p = 0; p < M; p++) {
              const a = arms[p];
              if (phase === 'explore' || phase === 'exploit') {
                if (!coll[p]) { n[p * K + a] += 1; s[p * K + a] += rew[p]; }
                continue;
              }
              const u = coll[p] ? 0 : mu[p * K + a];
              if (content[p] && a === base[p] && u === baseU[p]) { /* stays content */ }
              else {
                const pr = (u / umax[p]) * Math.pow(params.eps, kappa * (umax[p] - u));
                if (u > 0 && rng.uniform() < pr) { content[p] = 1; base[p] = a; baseU[p] = u; }
                else content[p] = 0;
              }
              if (clock >= c2 / 2 && content[p] && !coll[p]) count[p * K + a] += 1;
            }
            advance();
          },
        };
      },
    },
    {
      id: 'fed-ucb', name: 'FedUCB (periodic sync)', family: 'federated',
      params: [
        { key: 'tau', label: 'Sync period τ', default: 10, min: 1, max: 500, step: 1 },
        { key: 'c', label: 'Exploration c', default: 1.414, min: 0, max: 3, step: 0.01 },
      ],
      formula: 'aₜᵖ = argmaxₐ (Sₐ + sₐᵖ)/(Nₐ + nₐᵖ) + c·√(ln(Mt) / (Nₐ + nₐᵖ));  every τ rounds: (S, N) += Σₚ (sᵖ, nᵖ)',
      regret: 'O(Σ log(MT)/Δ) total + O(MK·T/τ) communication',
      idea: 'Federated UCB in the spirit of Shi & Shen (2021, Fed-UCB), homogeneous case: every agent runs UCB on the server\'s pooled statistics plus its own unsent samples, and uploads every τ rounds. Simplified: no local/global heterogeneity and no growing exploration schedule. In collision environments it herds all agents onto the same arm.',
      create(K, M, { params }) {
        const N = new Float64Array(K), S = new Float64Array(K), n = new Float64Array(M * K), s = new Float64Array(M * K), idx = new Float64Array(K);
        const tau = Math.max(1, Math.round(params.tau));
        return {
          select(t, X, arms, rng) {
            const lt = Math.log(Math.max(2, M * t));
            for (let p = 0; p < M; p++) {
              for (let a = 0; a < K; a++) {
                const nn = N[a] + n[p * K + a];
                idx[a] = nn > 0 ? (S[a] + s[p * K + a]) / nn + params.c * Math.sqrt(lt / nn) : 1e9 + rng.uniform();
              }
              arms[p] = util.argmax(idx, rng);
            }
          },
          update(t, arms, rew) {
            for (let p = 0; p < M; p++) { n[p * K + arms[p]] += 1; s[p * K + arms[p]] += rew[p]; }
            if (t % tau === 0) {
              for (let p = 0; p < M; p++) for (let a = 0; a < K; a++) { N[a] += n[p * K + a]; S[a] += s[p * K + a]; }
              n.fill(0); s.fill(0);
            }
          },
        };
      },
    },
    {
      id: 'fed-linucb', name: 'FedLinUCB (periodic sync)', family: 'federated',
      params: [
        { key: 'tau', label: 'Sync period τ', default: 10, min: 1, max: 500, step: 1 },
        { key: 'alpha', label: 'Confidence width α', default: 0.5, min: 0, max: 3, step: 0.05 },
      ],
      formula: 'aₜᵖ = argmaxₐ θ̂ₚ·xₐ + α√ln(1+Mt)·‖xₐ‖_{Vₚ⁻¹},  Vₚ = λI + V_sync + ΔVₚ;  every τ rounds V_sync += Σₚ ΔVₚ',
      regret: 'Õ(d√(MT)) total for suitable sync schedules',
      idea: 'The federated LinUCB of Dubey & Pentland (2020) without the privacy noise: agents run LinUCB on the pooled Gram matrix and response vector plus their own unsent data, and synchronize. Simplified: a fixed period τ instead of the log-det trigger. On K-armed environments the features are one-hot, which makes it a ridge-regularised FedUCB.',
      create(K, M, { params, dx }) {
        const tau = Math.max(1, Math.round(params.tau)), al = params.alpha;
        const idx = new Float64Array(K);
        if (!dx) {
          const N = new Float64Array(K), S = new Float64Array(K), n = new Float64Array(M * K), s = new Float64Array(M * K);
          return {
            select(t, X, arms, rng) {
              for (let p = 0; p < M; p++) {
                const w = al * Math.sqrt(Math.log(1 + M * t));
                for (let a = 0; a < K; a++) { const v = 1 + N[a] + n[p * K + a]; idx[a] = (S[a] + s[p * K + a]) / v + w / Math.sqrt(v); }
                arms[p] = util.argmax(idx, rng);
              }
            },
            update(t, arms, rew) {
              for (let p = 0; p < M; p++) { n[p * K + arms[p]] += 1; s[p * K + arms[p]] += rew[p]; }
              if (t % tau === 0) {
                for (let p = 0; p < M; p++) for (let a = 0; a < K; a++) { N[a] += n[p * K + a]; S[a] += s[p * K + a]; }
                n.fill(0); s.fill(0);
              }
            },
          };
        }
        const d = dx, dd = d * d;
        const Vg = new Float64Array(dd), bg = new Float64Array(d);
        for (let i = 0; i < d; i++) Vg[i * d + i] = 1;
        const Vinv = new Float64Array(M * dd), b = new Float64Array(M * d), dV = new Float64Array(M * dd), db = new Float64Array(M * d);
        for (let p = 0; p < M; p++) for (let i = 0; i < d; i++) Vinv[p * dd + i * d + i] = 1;
        const th = new Float64Array(d), z = new Float64Array(d), xs = new Float64Array(M * d);
        function invertInto(A, out, o) {   // Gauss-Jordan on a copy
          const m = A.slice(), I = new Float64Array(dd);
          for (let i = 0; i < d; i++) I[i * d + i] = 1;
          for (let c = 0; c < d; c++) {
            let piv = c;
            for (let r = c + 1; r < d; r++) if (Math.abs(m[r * d + c]) > Math.abs(m[piv * d + c])) piv = r;
            for (let k = 0; k < d; k++) { let x = m[c * d + k]; m[c * d + k] = m[piv * d + k]; m[piv * d + k] = x; x = I[c * d + k]; I[c * d + k] = I[piv * d + k]; I[piv * d + k] = x; }
            const pv = m[c * d + c];
            for (let k = 0; k < d; k++) { m[c * d + k] /= pv; I[c * d + k] /= pv; }
            for (let r = 0; r < d; r++) {
              if (r === c) continue;
              const f = m[r * d + c];
              if (!f) continue;
              for (let k = 0; k < d; k++) { m[r * d + k] -= f * m[c * d + k]; I[r * d + k] -= f * I[c * d + k]; }
            }
          }
          out.set(I, o);
        }
        return {
          select(t, X, arms, rng) {
            const w = al * Math.sqrt(Math.log(1 + M * t));
            for (let p = 0; p < M; p++) {
              const o = p * dd;
              for (let i = 0; i < d; i++) { let v = 0; for (let k = 0; k < d; k++) v += Vinv[o + i * d + k] * b[p * d + k]; th[i] = v; }
              for (let a = 0; a < K; a++) {
                const xo = (p * K + a) * d;
                let mean = 0, q = 0;
                for (let i = 0; i < d; i++) {
                  let v = 0;
                  for (let k = 0; k < d; k++) v += Vinv[o + i * d + k] * X[xo + k];
                  q += v * X[xo + i]; mean += th[i] * X[xo + i];
                }
                idx[a] = mean + w * Math.sqrt(Math.max(0, q));
              }
              const a = util.argmax(idx, rng);
              arms[p] = a;
              for (let i = 0; i < d; i++) xs[p * d + i] = X[(p * K + a) * d + i];
            }
          },
          update(t, arms, rew) {
            for (let p = 0; p < M; p++) {
              const o = p * dd;
              let q = 0;
              for (let i = 0; i < d; i++) { let v = 0; for (let k = 0; k < d; k++) v += Vinv[o + i * d + k] * xs[p * d + k]; z[i] = v; q += v * xs[p * d + i]; }
              for (let i = 0; i < d; i++) for (let k = 0; k < d; k++) Vinv[o + i * d + k] -= (z[i] * z[k]) / (1 + q);
              for (let i = 0; i < d; i++) {
                b[p * d + i] += rew[p] * xs[p * d + i]; db[p * d + i] += rew[p] * xs[p * d + i];
                for (let k = 0; k < d; k++) dV[o + i * d + k] += xs[p * d + i] * xs[p * d + k];
              }
            }
            if (t % tau === 0) {
              for (let p = 0; p < M; p++) {
                for (let i = 0; i < dd; i++) Vg[i] += dV[p * dd + i];
                for (let i = 0; i < d; i++) bg[i] += db[p * d + i];
              }
              dV.fill(0); db.fill(0);
              invertInto(Vg, Vinv, 0);
              for (let p = 1; p < M; p++) Vinv.copyWithin(p * dd, 0, dd);
              for (let p = 0; p < M; p++) b.set(bg, p * d);
            }
          },
        };
      },
    },
  ];
  const ALGO_BY_ID = {};
  for (const a of algorithms) ALGO_BY_ID[a.id] = a;
  function defaults(def) {
    const o = {};
    for (const p of def.params) o[p.key] = p.default;
    return o;
  }

  // -------------------------------------------------------------- runner ----
  function createRunner(config) {
    const env = ENV_BY_ID[config.env];
    if (!env) throw new Error('Unknown environment: ' + config.env);
    const K = config.K | 0, T = config.T | 0, runs = config.runs | 0, seed = config.seed >>> 0;
    if (K < 2) throw new Error('K must be at least 2');
    const ex = config.extra || {};
    let M = Math.max(1, Math.round(ex.M != null ? +ex.M : 3));
    if (env.id === 'bernoulli' || env.id === 'hetero') M = Math.min(M, K);
    const Y = new Float64Array(M * K), MT = new Float64Array(M * K), X = new Float64Array(M * K * 5);
    const arms = new Int32Array(M), rew = new Float64Array(M), coll = new Uint8Array(M), occ = new Int32Array(K);
    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaults(def), spec.params || {}),
        reg: new Float64Array(T), reg2: new Float64Array(T), col: new Float64Array(T), rw: new Float64Array(T),
        finals: [], totColl: 0, totSig: 0, totRew: 0, timeMs: 0,
      };
    });
    let done = 0, cacheBuf = null;

    function runOne(run) {
      const envRng = Rng(hash32(seed, run, 'env'));
      const inst = env.setup(K, M, T, envRng);
      const streamSeed = hash32(seed, run, 'env-stream');
      if (!inst.dynamic) MT.set(inst.mu);
      // Contextual streams are costly to draw: draw them once per run when they fit in memory.
      const MK = M * K, blk = MK * (2 + inst.dx) + 1;
      let cache = null;
      if (inst.dynamic && T * blk <= 3e6) {
        if (!cacheBuf || cacheBuf.length < T * blk) cacheBuf = new Float64Array(T * blk);
        cache = cacheBuf;
        const srng = Rng(streamSeed);
        for (let t = 0; t < T; t++) {
          const o = t * blk;
          cache[o + blk - 1] = env.draw(inst, srng, Y, K, M, MT, X);
          for (let i = 0; i < MK; i++) { cache[o + i] = Y[i]; cache[o + MK + i] = MT[i]; }
          for (let i = 0; i < MK * inst.dx; i++) cache[o + 2 * MK + i] = X[i];
        }
      }
      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const srng = Rng(streamSeed);   // the same reward/context stream for every algorithm
        const agent = slot.def.create(K, M, { T, params: slot.params, rng, dx: inst.dx, env });
        const sig = agent.signal || null;
        let cum = 0, ncoll = 0, got = 0, nsig = 0;
        const t0 = util.now();
        for (let t = 0; t < T; t++) {
          let opt = inst.opt;
          if (cache) {
            const o = t * blk;
            opt = cache[o + blk - 1];
            for (let i = 0; i < MK; i++) { Y[i] = cache[o + i]; MT[i] = cache[o + MK + i]; }
            for (let i = 0; i < MK * inst.dx; i++) X[i] = cache[o + 2 * MK + i];
          } else {
            const o = env.draw(inst, srng, Y, K, M, MT, X);
            if (inst.dynamic) opt = o;
          }
          if (sig) sig.fill(0);
          agent.select(t + 1, inst.dx ? X : null, arms, rng);
          occ.fill(0);
          for (let p = 0; p < M; p++) {
            let a = arms[p];
            if (!(a >= 0 && a < K)) a = arms[p] = 0;
            occ[a]++;
          }
          let exp = 0, nc = 0;
          for (let p = 0; p < M; p++) {
            const a = arms[p], c = inst.collide && occ[a] > 1;
            coll[p] = c ? 1 : 0;
            rew[p] = c ? 0 : Y[p * K + a];
            if (c) { nc++; if (sig && sig[p]) nsig++; } else exp += MT[p * K + a];
          }
          agent.update(t + 1, arms, rew, coll, rng);
          cum += opt - exp; ncoll += nc; got += exp;
          slot.reg[t] += cum; slot.reg2[t] += cum * cum; slot.col[t] += ncoll; slot.rw[t] += exp;
        }
        slot.timeMs += util.now() - t0;
        slot.finals.push(cum);
        slot.totColl += ncoll;
        slot.totSig += nsig;
        slot.totRew += got / T;
      }
    }

    return {
      config: { env: env.id, K, T, runs, seed, extra: { M } },
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
          const regret = new Float64Array(T), regretSE = new Float64Array(T), collisions = new Float64Array(T), reward = new Float64Array(T);
          for (let t = 0; t < T; t++) {
            const m = s.reg[t] / R;
            regret[t] = m;
            regretSE[t] = R > 1 ? Math.sqrt(Math.max(0, s.reg2[t] / R - m * m) / (R - 1)) : 0;
            collisions[t] = s.col[t] / R;
            reward[t] = s.rw[t] / R;
          }
          const sorted = s.finals.slice().sort((a, b) => a - b);
          const summary = {
            regret: regret[T - 1], regretSE: regretSE[T - 1],
            regretMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            collisions: s.totColl / R, collRate: s.totColl / R / T,
            accidental: (s.totColl - s.totSig) / R, signalling: s.totSig / R,
            meanReward: s.totRew / R, msPerRun: done ? s.timeMs / done : 0,
          };
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, params: s.params, runs: done,
            curves: { regret, collisions, reward }, curvesSE: { regret: regretSE },
            summary, finals: sorted,
          };
        });
      },
    };
  }

  return MAB.registerSetting({
    id: 'multiplayer',
    name: 'Multi-player & federated',
    short: 'Multi-player',
    blurb: 'M players pull arms at the same time: colliding players get nothing, or, in the federated case, agents share what they learn.',
    families, environments, algorithms,
    defaultEnv: 'bernoulli',
    defaultLineup: ['sic-mmab', 'mctopm', 'randtopm', 'dpe', 'musical-chairs', 'got', 'selfish-ucb'],
    presets: [
      { label: 'Collision avoidance', algos: ['mctopm', 'randtopm', 'musical-chairs', 'selfish-ucb'], env: 'bernoulli' },
      { label: 'Implicit communication', algos: ['sic-mmab', 'dpe', 'mctopm', 'selfish-ucb'], env: 'bernoulli' },
      { label: 'Player-specific means', algos: ['got', 'mctopm', 'sic-mmab', 'selfish-ucb'], env: 'hetero' },
      { label: 'Federated', algos: ['fed-ucb', 'fed-linucb', 'selfish-ucb'], env: 'fed-bernoulli' },
      { label: 'Federated linear', algos: ['fed-linucb', 'fed-ucb', 'selfish-ucb'], env: 'fed-linear' },
    ],
    controls: {
      K: { label: 'Arms (K)', min: 2, max: 20, default: 8 },
      T: { label: 'Horizon (T rounds)', options: [500, 1000, 2000, 5000, 10000], default: 2000 },
      runs: { label: 'Runs averaged', options: [20, 50, 100, 200], default: 50 },
      extra: [{ key: 'M', label: 'Players (M)', min: 1, max: 10, step: 1, default: 3 }],
    },
    xLabel: 'round t',
    metrics: [
      { key: 'regret', label: 'System regret', better: 'lower', fmt: 'num', band: true, note: 'Collision environments: OPT (top-M means, or the max-weight matching) minus the expected reward of non-colliding players. Federated environments: sum over agents of each agent\'s regret against its own best arm.' },
      { key: 'collisions', label: 'Collisions (cumulative)', better: 'lower', fmt: 'num', note: 'Player-pulls that ended in a collision so far, including deliberate signalling collisions (SIC-MMAB, DPE); the table splits accidental from signalling. Always 0 in federated environments.' },
      { key: 'reward', label: 'System reward per round', better: 'higher', fmt: 'fixed3', smooth: true, note: 'Expected total reward of all players in each round.' },
    ],
    final: { key: 'regret', se: 'regretSE', median: 'regretMedian', label: 'System regret', better: 'lower' },
    tiles: [
      { key: 'collRate', label: 'Fewest collisions', better: 'lower', fmt: 'fixed3', sub: 'colliding players per round' },
      { key: 'accidental', label: 'Fewest accidental collisions', better: 'lower', fmt: 'num', sub: 'not used as signals' },
      { key: 'meanReward', label: 'Highest system reward', better: 'higher', fmt: 'fixed3', sub: 'per round' },
    ],
    columns: [
      { key: 'regret', label: 'Regret', fmt: 'num', better: 'lower' },
      { key: 'regretSE', label: '95% CI', fmt: 'ci' },
      { key: 'regretMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'collisions', label: 'Collisions', fmt: 'num', better: 'lower' },
      { key: 'collRate', label: 'Coll. / round', fmt: 'fixed3', better: 'lower' },
      { key: 'accidental', label: 'Accidental coll.', fmt: 'num', better: 'lower' },
      { key: 'signalling', label: 'Signalling coll.', fmt: 'num' },
      { key: 'meanReward', label: 'Reward / round', fmt: 'fixed3', better: 'higher' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: false,
    createRunner,
    _internal: { assignMax },
  });
});
