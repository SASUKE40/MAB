/*
 * continuum.js — continuum-armed bandits, Lipschitz tree search, bandit convex optimisation
 * and Gaussian-process bandits.
 *
 * The action is a point x in [0,1] (or [0,1]² for the 2-D environment); the reward is
 * f(x) + noise. Regret is pseudo-regret T·f* − Σ f(xₜ) (for the contextual environment
 * Σ f*(cₜ) − f(xₜ, cₜ)), and simple regret f* − f(recommended x).
 *
 * Internal algorithm interface (only this file's runner uses it):
 *   create(ctx)  ctx = { T, d, sigma, threshold, safeSeed, contextual, params, rng }
 *     -> { select(t, c) -> point (array-like of length d),
 *          update(x, y, t, c),            // x is a copy of the clamped point that was played
 *          recommend(c) -> point }         // optional; default: best observed point
 *   `sigma` is the (known) noise standard deviation, `threshold` the safety threshold h
 *   (−∞ unless the environment is the safe one), `safeSeed` a point known to be safe,
 *   and `c` the context in [0,1] (0.5 on non-contextual environments).
 *
 * Fairness: per run, the environment instance, the noise stream (indexed by step) and the
 * context stream are drawn from Rng(hash32(seed, run, 'env')), so every algorithm sees the
 * same noise at step t (common random numbers). Algorithms draw from Rng(hash32(seed, run, id)).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, hash32, util } = MAB;
  const now = util.now;

  // ------------------------------------------------------------- helpers ----
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const gauss = (x, m, w) => Math.exp(-((x - m) * (x - m)) / (2 * w * w));

  /** Maximise a 1-D function on [0,1]: dense grid, then golden-section refinement. */
  function maximize1D(f) {
    const N = 4000;
    let bi = 0, bv = -Infinity;
    for (let i = 0; i <= N; i++) { const v = f(i / N); if (v > bv) { bv = v; bi = i; } }
    let a = Math.max(0, (bi - 1) / N), b = Math.min(1, (bi + 1) / N);
    const g = (Math.sqrt(5) - 1) / 2;
    for (let k = 0; k < 60; k++) {
      const c = b - g * (b - a), d = a + g * (b - a);
      if (f(c) > f(d)) b = d; else a = c;
    }
    return Math.max(bv, f((a + b) / 2));
  }

  /** Uniform direction on the unit sphere of R^d. */
  function unitVector(rng, d, out) {
    if (d === 1) { out[0] = rng.uniform() < 0.5 ? -1 : 1; return out; }
    let s = 0;
    for (let i = 0; i < d; i++) { out[i] = rng.normal(); s += out[i] * out[i]; }
    s = Math.sqrt(s) || 1;
    for (let i = 0; i < d; i++) out[i] /= s;
    return out;
  }

  /** argmax over a Float64Array with random tie-breaking, restricted by an optional mask. */
  function argmaxMasked(v, rng, mask) {
    let best = -Infinity, idx = -1, ties = 0;
    for (let i = 0; i < v.length; i++) {
      if (mask && !mask[i]) continue;
      const x = v[i];
      if (x > best) { best = x; idx = i; ties = 1; }
      else if (x === best) { ties++; if (rng.uniform() * ties < 1) idx = i; }
    }
    return idx;
  }

  // ---------------------------------------------------------------- grids ----
  /** Regular grid on [0,1]^d with n1 points per axis, including the end points. */
  const GRIDS = {};
  function grid(d, n1) {
    const key = d + ':' + n1;
    if (GRIDS[key]) return GRIDS[key];
    const G = Math.pow(n1, d), pts = new Float64Array(G * d);
    for (let g = 0; g < G; g++) {
      let r = g;
      for (let j = 0; j < d; j++) { pts[g * d + j] = (r % n1) / (n1 - 1); r = Math.floor(r / n1); }
    }
    return (GRIDS[key] = { key, d, n1, G, pts });
  }
  /** GP acquisition grid: 128 points in 1-D, 14×14 in 2-D (the full posterior covariance is kept). */
  const gpGrid = (d) => (d === 1 ? grid(1, 128) : grid(2, 14));
  function nearestIndex(gr, x) {
    let g = 0, mul = 1;
    for (let j = 0; j < gr.d; j++) { g += Math.round(clamp(x[j], 0, 1) * (gr.n1 - 1)) * mul; mul *= gr.n1; }
    return g;
  }
  function gridPoint(gr, g, out) {
    for (let j = 0; j < gr.d; j++) out[j] = gr.pts[g * gr.d + j];
    return out;
  }
  function dist2(gr, a, b) {
    let s = 0;
    for (let j = 0; j < gr.d; j++) { const z = gr.pts[a * gr.d + j] - gr.pts[b * gr.d + j]; s += z * z; }
    return s;
  }

  // ------------------------------------------------------- GP on a grid ----
  /*
   * Exact GP posterior on a fixed grid of G candidate points, kept as the full G×G posterior
   * covariance (upper triangle) and updated with one rank-1 (Kalman) step per observation:
   * O(G²/2) per step,
   * independent of the number of observations, and repeated points cost nothing extra.
   * RBF kernel sf2·exp(−‖x−x′‖²/(2ℓ²)) with fixed hyperparameters, constant prior mean m0.
   */
  const COV = {};
  function unitCov(gr, ell) {
    const key = gr.key + '|' + ell;
    if (COV[key]) return COV[key];
    const G = gr.G, K = new Float64Array(G * G), s = 1 / (2 * ell * ell);
    for (let a = 0; a < G; a++) for (let b = a; b < G; b++) {
      const v = Math.exp(-dist2(gr, a, b) * s);
      K[a * G + b] = v; K[b * G + a] = v;
    }
    return (COV[key] = K);
  }
  function gridGP(gr, ell, sf2, noise2, m0) {
    const G = gr.G, S = new Float64Array(G * G), mu = new Float64Array(G).fill(m0);
    const K = unitCov(gr, ell);
    for (let i = 0; i < G * G; i++) S[i] = sf2 * K[i];
    const col = new Float64Array(G);
    return {
      G, mu, S, noise2, n: 0,
      sd(g) { const v = S[g * G + g]; return v > 1e-12 ? Math.sqrt(v) : 1e-6; },
      cov(a, b) { return a <= b ? S[a * G + b] : S[b * G + a]; },
      /** Condition on y observed at grid index i. Returns ½ ln(1 + σ²ₜ₋₁(xᵢ)/noise²). */
      observe(i, y) {
        const prior = Math.max(S[i * G + i], 0), s = prior + noise2;
        for (let g = 0; g < i; g++) col[g] = S[g * G + i];
        for (let g = i; g < G; g++) col[g] = S[i * G + g];
        const r = (y - mu[i]) / s;
        for (let g = 0; g < G; g++) mu[g] += col[g] * r;
        for (let a = 0; a < G; a++) {
          const ca = col[a] / s;
          if (ca === 0) continue;
          const off = a * G;
          for (let b = a; b < G; b++) S[off + b] -= ca * col[b];
        }
        this.n++;
        return 0.5 * Math.log(1 + prior / noise2);
      },
    };
  }

  // ----------------------------------------------------------- tree nodes ----
  function node(lo, hi, depth, parent) {
    const d = lo.length, x = new Float64Array(d);
    for (let j = 0; j < d; j++) x[j] = (lo[j] + hi[j]) / 2;
    return { lo, hi, x, depth, parent, kids: null, n: 0, sum: 0, sq: 0, U: Infinity, B: Infinity };
  }
  function rootNode(d) { return node(new Float64Array(d), new Float64Array(d).fill(1), 0, null); }
  /** Split a cell into k equal parts along its longest side (cycles through dimensions). */
  function split(nd, k) {
    const d = nd.lo.length;
    let j = 0, w = -1;
    for (let i = 0; i < d; i++) { const wi = nd.hi[i] - nd.lo[i]; if (wi > w + 1e-15) { w = wi; j = i; } }
    nd.kids = [];
    for (let m = 0; m < k; m++) {
      const lo = Float64Array.from(nd.lo), hi = Float64Array.from(nd.hi);
      lo[j] = nd.lo[j] + (w * m) / k; hi[j] = nd.lo[j] + (w * (m + 1)) / k;
      nd.kids.push(node(lo, hi, nd.depth + 1, nd));
    }
    return nd.kids;
  }
  /** Follow the most-played child from the root; return the cell centre where it stops. */
  function mostPlayedPath(rt) {
    let nd = rt;
    while (nd.kids) {
      let nx = nd.kids[0];
      for (const k of nd.kids) if (k.n > nx.n) nx = k;
      if (nx.n === 0) break;
      nd = nx;
    }
    return nd.x;
  }
  const MAX_DEPTH = 40;

  /** HOO (Bubeck, Munos, Stoltz & Szepesvári 2011) on the binary tree of the box. */
  function makeHOO(d, sigma, nu, rho, c, rng) {
    const rt = rootNode(d), nodes = [rt], path = [];
    const rp = Math.pow(rho, 1 / d);
    const smooth = []; for (let h = 0; h <= MAX_DEPTH + 1; h++) smooth.push(nu * Math.pow(rp, h));
    let steps = 0;
    return {
      root: rt,
      select() {
        path.length = 0;
        let nd = rt;
        path.push(nd);
        while (nd.n > 0 && nd.depth < MAX_DEPTH) {
          if (!nd.kids) { split(nd, 2); nodes.push(nd.kids[0], nd.kids[1]); }
          const [a, b] = nd.kids;
          nd = a.B > b.B ? a : b.B > a.B ? b : (rng.uniform() < 0.5 ? a : b);
          path.push(nd);
        }
        return nd.x;
      },
      update(y) {
        steps++;
        for (const nd of path) { nd.n++; nd.sum += y; }
        const lt = Math.log(Math.max(2, steps)), cs = c * sigma;
        for (let i = nodes.length - 1; i >= 0; i--) {
          const nd = nodes[i];
          if (nd.n === 0) { nd.U = nd.B = Infinity; continue; }
          nd.U = nd.sum / nd.n + cs * Math.sqrt((2 * lt) / nd.n) + smooth[nd.depth];
          nd.B = nd.kids ? Math.min(nd.U, Math.max(nd.kids[0].B, nd.kids[1].B)) : nd.U;
        }
      },
      recommend() { return mostPlayedPath(rt); },
    };
  }

  /** HCT (Azar, Lazaric & Brunskill 2014); variance = true gives the VHCT variant (Li et al. 2023). */
  function makeHCT(ctx, variance) {
    const { d, sigma, rng, params } = ctx;
    const { nu, rho, c, delta } = params;
    const rt = rootNode(d), nodes = [rt], path = [];
    const rp = Math.pow(rho, 1 / d), cs = c * sigma, s2 = sigma * sigma;
    const smooth = []; for (let h = 0; h <= MAX_DEPTH + 1; h++) smooth.push(nu * Math.pow(rp, h));
    let Lc = -1;
    const Lof = (t) => Math.log(Math.pow(2, Math.floor(Math.log2(Math.max(1, t))) + 1) / delta);
    function bonus(nd) {
      if (nd.n === 0) return Infinity;
      if (!variance) return cs * Math.sqrt((2 * Lc) / nd.n);
      const m = nd.sum / nd.n, v = (Math.max(0, nd.sq - nd.n * m * m) + s2) / (nd.n + 1);
      return Math.sqrt((2 * v * Lc) / nd.n) + (3 * cs * Lc) / nd.n;
    }
    // HCT: a node is refined once its pull count reaches τₕ = 2c²σ² L / (νρʰ)²; VHCT: once its
    // variance-aware confidence width falls below the smoothness term νρʰ.
    const ready = (nd) => (variance ? bonus(nd) <= smooth[nd.depth] : nd.n >= Math.ceil((2 * cs * cs * Lc) / (smooth[nd.depth] * smooth[nd.depth])));
    function setU(nd) { nd.U = nd.n ? nd.sum / nd.n + smooth[nd.depth] + bonus(nd) : Infinity; }
    function setB(nd) { nd.B = nd.kids ? Math.min(nd.U, Math.max(nd.kids[0].B, nd.kids[1].B)) : nd.U; }
    return {
      select(t) {
        const L = Lof(t);
        if (L !== Lc) { // t reached a power of two: refresh every U and B
          Lc = L;
          for (let i = nodes.length - 1; i >= 0; i--) { setU(nodes[i]); setB(nodes[i]); }
        }
        path.length = 0;
        let nd = rt;
        path.push(nd);
        while (nd.kids && ready(nd)) {
          const [a, b] = nd.kids;
          nd = a.B > b.B ? a : b.B > a.B ? b : (rng.uniform() < 0.5 ? a : b);
          path.push(nd);
        }
        return nd.x;
      },
      update(x, y) {
        const nd = path[path.length - 1];
        nd.n++; nd.sum += y; nd.sq += y * y;
        setU(nd);
        if (!nd.kids && nd.depth < MAX_DEPTH && ready(nd)) { split(nd, 2); nodes.push(nd.kids[0], nd.kids[1]); }
        for (let i = path.length - 1; i >= 0; i--) setB(path[i]);
      },
      recommend() {
        let best = rt, bv = -Infinity;
        for (const nd of nodes) {
          if (!nd.n) continue;
          const v = nd.sum / nd.n - cs * Math.sqrt((2 * Math.max(Lc, 1)) / nd.n);
          if (v > bv) { bv = v; best = nd; }
        }
        return best.x;
      },
    };
  }

  /*
   * DOO / SOO (Munos 2011) and StoSOO (Valko, Carpentier & Munos 2013) on the ternary tree of
   * the box (the middle child keeps the parent's centre and its samples).
   */
  function makeSOO(ctx, mode) {
    const { d, T, sigma, rng, params } = ctx;
    const rt = rootNode(d), all = [rt], byDepth = [[rt]], queue = [];
    const cap = d === 1 ? 14 : 22;
    const rp = Math.pow(params.rho || 1 / 3, 1 / d), nu = params.nu || 0;
    const k = Math.max(1, Math.round(params.k || 1));
    const conf = (params.c || 0) * sigma * Math.sqrt(2 * Math.log((T * k) / (params.delta || 0.1)));
    const hmaxSto = Math.max(1, Math.floor(Math.sqrt(T / k)));
    let h = 0, vmax = -Infinity, cur = null;

    function value(nd) {
      if (mode === 'stosoo') return nd.n ? nd.sum / nd.n + conf / Math.sqrt(nd.n) : Infinity;
      if (!nd.n) return -Infinity;
      return nd.sum / nd.n + (mode === 'doo' ? nu * Math.pow(rp, nd.depth) : 0);
    }
    function expand(nd) {
      const lv = byDepth[nd.depth];
      lv.splice(lv.indexOf(nd), 1);
      const kids = split(nd, 3);
      if (!byDepth[nd.depth + 1]) byDepth[nd.depth + 1] = [];
      for (const kd of kids) { byDepth[nd.depth + 1].push(kd); all.push(kd); }
      kids[1].n = nd.n; kids[1].sum = nd.sum;            // same centre: inherit samples
      if (mode !== 'stosoo') queue.push(kids[0], kids[2]);
      if (mode === 'doo' && kids[1].n) hpush(kids[1]);
    }
    // DOO keeps its evaluated leaves in a max-heap of b-values (stale entries are skipped)
    const heap = [];
    function hpush(nd) {
      nd.ver = (nd.ver || 0) + 1;
      const e = [value(nd) + 1e-12 * rng.uniform(), nd, nd.ver];
      let i = heap.push(e) - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] >= e[0]) break; heap[i] = heap[p]; i = p; }
      heap[i] = e;
    }
    function hbest() {
      while (heap.length) {
        const top = heap[0];
        if (!top[1].kids && top[2] === top[1].ver) return top[1];
        const e = heap.pop();
        if (!heap.length) break;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i, mv = e[0];
          if (l < heap.length && heap[l][0] > mv) { m = l; mv = heap[l][0]; }
          if (r < heap.length && heap[r][0] > mv) m = r;
          if (m === i) break;
          heap[i] = heap[m]; i = m;
        }
        heap[i] = e;
      }
      return rt;
    }
    function bestAt(depth) {
      const lv = byDepth[depth];
      if (!lv || !lv.length) return null;
      let best = null, bv = -Infinity, ties = 0;
      for (const nd of lv) {
        const v = value(nd);
        if (v === -Infinity) continue;
        if (v > bv) { bv = v; best = nd; ties = 1; } else if (v === bv) { ties++; if (rng.uniform() * ties < 1) best = nd; }
      }
      return best;
    }
    function bestLeaf() {
      let best = rt, bv = -Infinity;
      for (const lv of byDepth) for (const nd of lv || []) { const v = value(nd); if (v > bv) { bv = v; best = nd; } }
      return best;
    }
    function choose(t) {
      if (!rt.n) return rt;
      if (mode === 'doo') {
        const nd = hbest();
        if (nd.depth >= cap) return nd;
        expand(nd);
        return queue.shift();
      }
      for (let guard = 0; guard < 4 * (cap + 2); guard++) {
        const hm = Math.min(byDepth.length - 1, cap, mode === 'soo' ? Math.floor(Math.sqrt(t)) : hmaxSto);
        if (h > hm) { h = 0; vmax = -Infinity; }
        const nd = bestAt(h);
        h++;
        if (!nd) continue;
        const v = value(nd);
        if (v < vmax) continue;
        if (mode === 'stosoo') {
          if (nd.n < k || nd.depth >= cap) return nd;
          expand(nd); vmax = v;
        } else {
          vmax = v;
          if (nd.depth >= cap) return nd;
          expand(nd);
          return queue.shift();
        }
      }
      return bestLeaf();
    }
    return {
      select(t) {
        cur = queue.length ? queue.shift() : choose(t);
        return cur.x;
      },
      update(x, y) { cur.n++; cur.sum += y; if (mode === 'doo') hpush(cur); },
      recommend() {
        let best = rt, bv = -Infinity;
        const need = mode === 'stosoo' ? k : 1;
        for (const nd of all) if (nd.n >= need && nd.sum / nd.n > bv) { bv = nd.sum / nd.n; best = nd; }
        if (bv === -Infinity) for (const nd of all) if (nd.n > best.n) best = nd;
        return best.x;
      },
    };
  }

  // ------------------------------------------------------------ algorithms ----
  const P = (key, label, def, min, max, step) => ({ key, label, default: def, min, max, step });

  const ALGORITHMS = [
    {
      id: 'uniform', name: 'Uniform random', family: 'baseline',
      params: [],
      formula: 'xₜ ~ U([0,1]ᵈ)',
      regret: 'Θ(T)',
      idea: 'Samples the domain uniformly and recommends the best point it observed. Its regret is the cost of pure exploration.',
      create({ d, rng }) {
        const x = new Float64Array(d);
        return { select() { for (let j = 0; j < d; j++) x[j] = rng.uniform(); return x; }, update() {} };
      },
    },

    // ---- discretization
    {
      id: 'ucb-grid', name: 'Discretized UCB (Kleinberg)', family: 'discretization',
      params: [P('mult', 'Grid size × (T/ln T)^{1/(d+2)}', 1, 0.5, 4, 0.1), P('c', 'Bonus scale c', 1, 0, 3, 0.05)],
      formula: 'K = ⌈(T/ln T)^{1/(d+2)}⌉ᵈ cells; argmaxₖ μ̂ₖ + cσ√(2 ln t / nₖ)',
      regret: 'Õ(T^{2/3}) for 1-D Lipschitz f',
      idea: 'Kleinberg (2004): fix a uniform grid sized for the horizon and run UCB1 on its cell centres. Simple and robust, but the grid ignores where the good region is. In 2-D the grid is a product grid with T^{1/4}-ish points per axis.',
      create({ T, d, sigma, params, rng }) {
        const n1 = Math.max(2, Math.ceil(params.mult * Math.pow(T / Math.log(T), 1 / (d + 2))));
        const K = Math.pow(n1, d), n = new Float64Array(K), s = new Float64Array(K), idx = new Float64Array(K);
        const pt = (k, out) => { let r = k; for (let j = 0; j < d; j++) { out[j] = ((r % n1) + 0.5) / n1; r = Math.floor(r / n1); } return out; };
        const x = new Float64Array(d), cs = params.c * sigma;
        let last = 0;
        return {
          select(t) {
            const lt = Math.log(Math.max(2, t));
            for (let k = 0; k < K; k++) idx[k] = n[k] ? s[k] / n[k] + cs * Math.sqrt((2 * lt) / n[k]) : Infinity;
            last = util.argmax(idx, rng);
            return pt(last, x);
          },
          update(_, y) { n[last]++; s[last] += y; },
          recommend() { let b = 0; for (let k = 1; k < K; k++) if (n[k] > n[b]) b = k; return pt(b, new Float64Array(d)); },
        };
      },
    },
    {
      id: 'zooming', name: 'Zooming', family: 'discretization',
      params: [P('L', 'Lipschitz constant L', 3, 0.5, 20, 0.5), P('c', 'Radius scale c', 1, 0.1, 3, 0.05)],
      formula: 'activate an uncovered x; play argmaxᵥ μ̂ᵥ + 2rᵥ, rᵥ = cσ√(2 ln T/(1+nᵥ))',
      regret: 'Õ(T^{(z+1)/(z+2)}), z = zooming dimension',
      idea: 'Kleinberg, Slivkins & Upfal (2008): keep a set of active points, each covering a ball of radius rᵥ/L that shrinks as it is played; when some point is uncovered, activate it. Arms concentrate where the function is high. Coverage is checked on a fine candidate grid; unplayed arms get an infinite index.',
      create({ T, d, sigma, params, rng }) {
        const cand = d === 1 ? grid(1, 257) : grid(2, 17), C = cand.G;
        const ax = [], an = [], as = [];
        const cs = params.c * sigma * Math.sqrt(2 * Math.log(T)), L = params.L;
        const rad = (i) => cs / Math.sqrt(1 + an[i]);
        const coverer = new Int32Array(C).fill(-1); // last active arm known to cover each candidate
        let last = 0;
        const x = new Float64Array(d);
        function covers(i, g) {
          let dm = 0;
          for (let j = 0; j < d; j++) dm = Math.max(dm, Math.abs(cand.pts[g * d + j] - ax[i][j]));
          return L * dm <= rad(i);
        }
        function covered(g) {
          const c0 = coverer[g];
          if (c0 >= 0 && covers(c0, g)) return true;
          for (let i = ax.length - 1; i >= 0; i--) if (covers(i, g)) { coverer[g] = i; return true; }
          return false;
        }
        return {
          select() {
            const off = rng.int(C);
            for (let q = 0; q < C; q++) {
              const g = (off + q) % C;
              if (!covered(g)) { ax.push(gridPoint(cand, g, new Float64Array(d))); an.push(0); as.push(0); break; }
            }
            let best = -Infinity, ties = 0;
            for (let i = 0; i < ax.length; i++) {
              const v = an[i] ? as[i] / an[i] + 2 * rad(i) : Infinity;
              if (v > best) { best = v; last = i; ties = 1; } else if (v === best) { ties++; if (rng.uniform() * ties < 1) last = i; }
            }
            x.set(ax[last]);
            return x;
          },
          update(_, y) { an[last]++; as[last] += y; },
          recommend() { let b = 0; for (let i = 1; i < ax.length; i++) if (an[i] > an[b]) b = i; return ax[b]; },
        };
      },
    },
    {
      id: 'contextual-zooming', name: 'Contextual Zooming', family: 'discretization',
      params: [P('L', 'Lipschitz constant L', 3, 0.5, 20, 0.5), P('c', 'Confidence scale c', 1, 0.1, 3, 0.05)],
      formula: 'argmax_B L·r_B + min_B′ (μ̂_B′ + L·r_B′ + conf_B′ + L·D(B,B′)), B relevant to cₜ',
      regret: 'Õ(T^{(z+1)/(z+2)}), z = contextual zooming dim.',
      idea: 'Slivkins (2011): adaptively partition the context × action space into balls; play the relevant ball with the best Lipschitz-propagated upper bound and split a ball in half once its confidence radius drops below its size. Simplified: a ball\'s domain is probed at its centre plus a few random points, and the Lipschitz minimum in the index runs over the ball and its child balls only (a looser but valid upper bound), and at most 400 balls are activated. On non-contextual environments the context is constant and it reduces to zooming with balls.',
      create({ T, d, sigma, params, rng }) {
        const L = params.L, cs = params.c * sigma * Math.sqrt(2 * Math.log(T)), MAX_BALLS = 400;
        const balls = [{ c: 0.5, x: new Float64Array(d).fill(0.5), r: 1, n: 0, s: 0, kids: [], pre: 0, rel: -1, deadC: NaN }];
        const conf = (b) => cs / Math.sqrt(1 + b.n);
        const pre = (b) => (b.n ? b.s / b.n : 1) + L * b.r + conf(b);
        const D = (a, b) => { let m = Math.abs(a.c - b.c); for (let j = 0; j < d; j++) m = Math.max(m, Math.abs(a.x[j] - b.x[j])); return m; };
        const inside = (b, c, x) => { if (Math.abs(c - b.c) > b.r) return false; for (let j = 0; j < d; j++) if (Math.abs(x[j] - b.x[j]) > b.r) return false; return true; };
        const tryX = new Float64Array(d), x = new Float64Array(d);
        let chosen = null;
        // Only balls whose context range contains c (the relevant set) can host (c, x).
        const rel = [], bySize = [];
        // bySize is sorted by decreasing radius, so B's own children (radius r_B/2) are tried first
        function inDomain(B, c, xx) {
          let lo = 0, hi = bySize.length;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (bySize[mid].r >= B.r) lo = mid + 1; else hi = mid; }
          for (let i = lo; i < bySize.length; i++) if (inside(bySize[i], c, xx)) return false;
          return true;
        }
        const order = [], bound = [];
        return {
          select(t, c) {
            rel.length = 0; order.length = 0;
            for (const b of balls) {
              b.rel = -1;
              if (Math.abs(c - b.c) > b.r) continue;
              b.rel = rel.length; b.pre = pre(b);
              order.push(rel.length); rel.push(b);
            }
            // index: I(B) = L·r_B + min over B and its relevant children of I_pre + L·D; visit balls
            // in index order and play the first one whose domain is non-empty at context c
            for (const b of rel) {
              let m = b.pre;
              for (const k of b.kids) if (k.rel >= 0) m = Math.min(m, k.pre + L * D(b, k));
              bound[b.rel] = L * b.r + m + 1e-9 * rng.uniform();
            }
            order.sort((p, q) => bound[q] - bound[p]);
            bySize.length = 0;
            for (const b of rel) bySize.push(b);
            bySize.sort((p, q) => q.r - p.r);
            chosen = null;
            for (const oi of order) {
              const B = rel[oi];
              if (B.deadC === c) continue; // coverage only grows: a ball found covered at c stays covered
              let ok = false;
              for (let a = 0; a < 4 && !ok; a++) {
                for (let j = 0; j < d; j++) tryX[j] = a === 0 ? B.x[j] : clamp(B.x[j] + B.r * (2 * rng.uniform() - 1), 0, 1);
                ok = inDomain(B, c, tryX);
              }
              if (ok) { chosen = B; x.set(tryX); break; }
              B.deadC = c;
            }
            if (!chosen) { chosen = balls[0]; for (let j = 0; j < d; j++) x[j] = rng.uniform(); }
            return x;
          },
          update(xx, y, t, c) {
            const B = chosen;
            B.n++; B.s += y;
            if (conf(B) <= L * B.r && B.r > 1 / 1024 && balls.length < MAX_BALLS) {
              const nb = { c, x: Float64Array.from(xx), r: B.r / 2, n: 0, s: 0, kids: [], pre: 0, rel: -1, deadC: NaN };
              balls.push(nb); B.kids.push(nb);
            }
          },
          recommend(c) {
            let best = null, bv = -Infinity;
            for (const b of balls) {
              if (!b.n || Math.abs(c - b.c) > b.r) continue;
              const v = b.s / b.n - conf(b);
              if (v > bv) { bv = v; best = b; }
            }
            return best ? best.x : balls[0].x;
          },
        };
      },
    },

    // ---- tree search
    {
      id: 'hoo', name: 'HOO', family: 'tree-search',
      params: [P('nu', 'Smoothness ν', 1, 0.05, 10, 0.05), P('rho', 'Shrink rate ρ', 0.5, 0.1, 0.95, 0.01), P('c', 'Bonus scale c', 1, 0, 3, 0.05)],
      formula: 'B = min(μ̂ + cσ√(2 ln t/n) + νρʰ, max B_children)',
      regret: 'Õ(T^{(d′+1)/(d′+2)}), d′ = near-optimality dim.',
      idea: 'Bubeck, Munos, Stoltz & Szepesvári (2011): grow a binary tree of cells, descending by optimistic B-values that combine a UCB bonus with a smoothness term νρʰ, and play the centre of the new leaf. In 2-D cells are split along their longest side and ρ applies per halving of the diameter (ρ^{h/d}).',
      create({ d, sigma, params, rng }) {
        const H = makeHOO(d, sigma, params.nu, params.rho, params.c, rng);
        return { select: () => H.select(), update: (x, y) => H.update(y), recommend: () => H.recommend() };
      },
    },
    {
      id: 'hct', name: 'HCT', family: 'tree-search',
      params: [P('nu', 'Smoothness ν', 1, 0.05, 10, 0.05), P('rho', 'Shrink rate ρ', 0.5, 0.1, 0.95, 0.01), P('c', 'Bonus scale c', 1, 0.1, 3, 0.05), P('delta', 'Confidence δ', 0.05, 0.001, 0.5, 0.001)],
      formula: 'U = μ̂ + νρʰ + cσ√(2 ln(t⁺/δ)/n); refine after τₕ ∝ ρ⁻²ʰ pulls',
      regret: 'Õ(T^{(d′+1)/(d′+2)}), O(log T)-time updates',
      idea: 'Azar, Lazaric & Brunskill (2014): like HOO, but a node is only refined after it has been pulled τₕ times, enough for its estimate to be as accurate as its smoothness term. Each node keeps its own statistics, B-values are refreshed only when t doubles, so it is fast and works with correlated feedback.',
      create(ctx) { return makeHCT(ctx, false); },
    },
    {
      id: 'vhct', name: 'VHCT', family: 'tree-search',
      params: [P('nu', 'Smoothness ν', 1, 0.05, 10, 0.05), P('rho', 'Shrink rate ρ', 0.5, 0.1, 0.95, 0.01), P('c', 'Bonus scale c', 1, 0.1, 3, 0.05), P('delta', 'Confidence δ', 0.05, 0.001, 0.5, 0.001)],
      formula: 'U = μ̂ + νρʰ + √(2V̂ ln(t⁺/δ)/n) + 3cσ ln(t⁺/δ)/n',
      regret: 'Õ(T^{(d′+1)/(d′+2)}), variance-adaptive',
      idea: 'Li et al. (2023): HCT with an empirical-Bernstein bonus, refining a node once its variance-aware width drops below νρʰ, so low-noise regions are refined sooner. Simplified: the variance estimate is shrunk toward the known noise level.',
      create(ctx) { return makeHCT(ctx, true); },
    },
    {
      id: 'poo', name: 'POO (with HOO)', family: 'tree-search',
      params: [P('nu', 'Smoothness ν_max', 1, 0.05, 10, 0.05), P('rhoMax', 'Largest ρ', 0.9, 0.3, 0.99, 0.01), P('c', 'Bonus scale c', 1, 0, 3, 0.05)],
      formula: 'N ≈ ½D ln(T/ln T) HOOs with ρᵢ = ρ_max^{2N/(2i+1)}, round robin; keep the best',
      regret: 'simple regret Õ(√ln T · T^{−1/(d′+2)}), ρ unknown',
      idea: 'Grill, Valko & Munos (2015) — "POOO" in the request is read as POO, Parallel Optimistic Optimization. Runs several HOO instances with different smoothness ρ in round robin and recommends from the one with the best average reward, so ρ need not be known. N follows the paper, capped at 8.',
      create({ T, d, sigma, params, rng }) {
        const Dmax = Math.log(2) / Math.log(1 / params.rhoMax);
        const N = clamp(Math.ceil(0.5 * Dmax * Math.log(T / Math.log(T))), 2, 8);
        const H = [];
        for (let i = 0; i < N; i++) H.push(makeHOO(d, sigma, params.nu, Math.pow(params.rhoMax, (2 * N) / (2 * i + 1)), params.c, rng));
        let cur = 0;
        return {
          select(t) { cur = (t - 1) % N; return H[cur].select(); },
          update(x, y) { H[cur].update(y); },
          recommend() {
            let b = 0, bv = -Infinity;
            for (let i = 0; i < N; i++) { const r = H[i].root; const v = r.n ? r.sum / r.n : -Infinity; if (v > bv) { bv = v; b = i; } }
            return H[b].recommend();
          },
        };
      },
    },
    {
      id: 'uct', name: 'UCT (dyadic tree)', family: 'tree-search',
      params: [P('c', 'Bonus scale c', 1, 0, 3, 0.05)],
      formula: 'child = argmax μ̂ + cσ√(2 ln n_parent / n_child); expand one leaf per step',
      regret: 'no smoothness term: worst case can be very bad',
      idea: 'Kocsis & Szepesvári (2006): treat every node of the dyadic tree of the domain as a two-armed UCB1 bandit and descend, adding one leaf per step. Without a smoothness term it can be overconfident (Coquelin & Munos 2007), which is why HOO adds νρʰ.',
      create({ d, sigma, params, rng }) {
        const rt = rootNode(d), path = [], cs = params.c * sigma;
        return {
          select() {
            path.length = 0;
            let nd = rt;
            path.push(nd);
            while (nd.n > 0 && nd.depth < MAX_DEPTH) {
              if (!nd.kids) split(nd, 2);
              const [a, b] = nd.kids;
              if (!a.n || !b.n) nd = !a.n && !b.n ? (rng.uniform() < 0.5 ? a : b) : (!a.n ? a : b);
              else {
                const lb = 2 * Math.log(nd.n);
                const ia = a.sum / a.n + cs * Math.sqrt(lb / a.n), ib = b.sum / b.n + cs * Math.sqrt(lb / b.n);
                nd = ia > ib ? a : ib > ia ? b : (rng.uniform() < 0.5 ? a : b);
              }
              path.push(nd);
            }
            return nd.x;
          },
          update(x, y) { for (const nd of path) { nd.n++; nd.sum += y; } },
          recommend() { return mostPlayedPath(rt); },
        };
      },
    },
    {
      id: 'doo', name: 'DOO', family: 'tree-search',
      params: [P('nu', 'Smoothness ν', 1, 0.05, 10, 0.05), P('rho', 'Shrink rate ρ', 0.333, 0.1, 0.9, 0.01)],
      formula: 'expand argmax_leaves f̂(xₕ,ᵢ) + νρʰ (one evaluation per cell)',
      regret: 'simple regret O(n^{−1/d′}) for deterministic f',
      idea: 'Munos (2011): Deterministic Optimistic Optimization trisects the leaf whose value plus smoothness bound νρʰ is largest. Designed for noiseless f: here it trusts single noisy evaluations, which is its weakness. Depth is capped; a capped leaf is re-evaluated.',
      create(ctx) { return makeSOO(ctx, 'doo'); },
    },
    {
      id: 'soo', name: 'SOO', family: 'tree-search',
      params: [],
      formula: 'for h = 0…√t: expand the best leaf at depth h if it beats v_max',
      regret: 'simple regret, adapts to unknown smoothness (noiseless)',
      idea: 'Munos (2011): Simultaneous Optimistic Optimization sweeps depths and expands at each depth the best leaf if no shallower expanded leaf was better, which covers every smoothness at once without knowing ν or ρ. Noiseless by design: one noisy evaluation per cell.',
      create(ctx) { return makeSOO(ctx, 'soo'); },
    },
    {
      id: 'stosoo', name: 'StoSOO', family: 'tree-search',
      params: [P('k', 'Evaluations per node k', 2, 1, 20, 1), P('c', 'Bonus scale c', 1, 0, 3, 0.05), P('delta', 'Confidence δ', 0.1, 0.001, 0.5, 0.001)],
      formula: 'b = μ̂ + cσ√(2 ln(Tk/δ)/n); sample a leaf until n = k, then expand',
      regret: 'simple regret Õ(T^{−1/2}) when d′ = 0',
      idea: 'Valko, Carpentier & Munos (2013): the stochastic version of SOO. Each leaf is evaluated k times (paper: k ≈ T/ln³T, ≈ 2 at T = 500) and compared by upper confidence bounds before it may be trisected.',
      create(ctx) { return makeSOO(ctx, 'stosoo'); },
    },

    // ---- zeroth-order / bandit convex optimisation
    {
      id: 'fkm', name: 'FKM (one-point gradient)', family: 'zeroth-order',
      params: [P('eta', 'Step η × T^{−3/4}', 1, 0.05, 10, 0.05), P('delta', 'Radius δ × T^{−1/4}', 1, 0.1, 2, 0.05)],
      formula: 'xₜ₊₁ = Π_{(1−δ)K}(xₜ + η·(d/δ)·yₜ·uₜ), yₜ = f(xₜ + δuₜ) + ε',
      regret: 'O(T^{3/4}) for convex losses',
      idea: 'Flaxman, Kalai & McMahan (2005): one random perturbation per step gives an unbiased gradient of a smoothed f, then take a projected gradient step. Needs no model of f but the estimate has variance ∝ 1/δ², so it is slow. Recommends its current iterate.',
      create({ T, d, params, rng }) {
        const delta = Math.min(0.25, params.delta * Math.pow(T, -0.25)), eta = params.eta * Math.pow(T, -0.75);
        const x = new Float64Array(d).fill(0.5), u = new Float64Array(d), y = new Float64Array(d);
        return {
          select() { unitVector(rng, d, u); for (let j = 0; j < d; j++) y[j] = x[j] + delta * u[j]; return y; },
          update(_, r) { for (let j = 0; j < d; j++) x[j] = clamp(x[j] + ((eta * d) / delta) * r * u[j], delta, 1 - delta); },
          recommend() { return x; },
        };
      },
    },
    {
      id: 'bgd', name: 'BGD (two-point)', family: 'zeroth-order',
      params: [P('eta', 'Step η × T^{−1/2}', 1, 0.05, 10, 0.05), P('delta', 'Radius δ × T^{−1/4}', 0.5, 0.05, 2, 0.05)],
      formula: 'xₜ₊₁ = Π(xₜ + η·(d/2δ)·(f(x+δu) − f(x−δu))·u)',
      regret: 'O(d²√T) convex, O(d² log T) strongly convex',
      idea: 'Agarwal, Dekel & Xiao (2010): query x + δu and x − δu and difference the two values, which cancels the level of f and makes the gradient estimate far less noisy than FKM. Here the two queries are two consecutive steps, and each counts toward regret.',
      create({ T, d, params, rng }) {
        const delta = Math.min(0.25, params.delta * Math.pow(T, -0.25)), eta = params.eta / Math.sqrt(T);
        const x = new Float64Array(d).fill(0.5), u = new Float64Array(d), y = new Float64Array(d);
        let phase = 0, yA = 0;
        return {
          select() {
            if (phase === 0) unitVector(rng, d, u);
            const s = phase === 0 ? 1 : -1;
            for (let j = 0; j < d; j++) y[j] = x[j] + s * delta * u[j];
            return y;
          },
          update(_, r) {
            if (phase === 0) { yA = r; phase = 1; return; }
            phase = 0;
            const g = (d / (2 * delta)) * (yA - r);
            for (let j = 0; j < d; j++) x[j] = clamp(x[j] + eta * g * u[j], delta, 1 - delta);
          },
          recommend() { return x; },
        };
      },
    },
    {
      id: 'scr', name: 'SCRiBLe-style (log-barrier)', family: 'zeroth-order',
      params: [P('eta', 'Step η × T^{−1/2}', 0.5, 0.01, 5, 0.01), P('lam', 'Smoothness λ', 5, 0, 50, 0.5)],
      formula: 'xₜ₊₁ = argmax η⟨Σĝₛ, x⟩ − R(x), yₜ = xₜ + (∇²R(xₜ) + ηλt)^{−½}uₜ',
      regret: 'Õ(√T) linear (SCRiBLe); Õ(T^{2/3}) smooth convex',
      idea: 'Abernethy, Hazan & Rakhlin (2008): follow-the-regularized-leader with the self-concordant log-barrier R(x) = −Σ ln xᵢ(1−xᵢ) of the box, exploring along the Dikin ellipsoid so queries stay feasible without projection. The extra ηλt term in the exploration matrix is the smooth-convex extension of Saha & Tewari (2011); the FTRL step has a closed form per coordinate.',
      create({ T, d, params, rng }) {
        const eta = params.eta / Math.sqrt(T), lam = params.lam;
        const x = new Float64Array(d).fill(0.5), G = new Float64Array(d), A = new Float64Array(d);
        const u = new Float64Array(d), y = new Float64Array(d);
        return {
          select(t) {
            unitVector(rng, d, u);
            for (let j = 0; j < d; j++) {
              const h = 1 / (x[j] * x[j]) + 1 / ((1 - x[j]) * (1 - x[j])) + eta * lam * t;
              A[j] = 1 / Math.sqrt(h);
              y[j] = x[j] + A[j] * u[j];
            }
            return y;
          },
          update(_, r) {
            for (let j = 0; j < d; j++) {
              G[j] += (d * r * u[j]) / A[j];
              const s = eta * G[j];
              x[j] = 2 / (2 - s + Math.sqrt(s * s + 4)); // solves R′(x) = s for R = −ln x − ln(1−x)
            }
          },
          recommend() { return x; },
        };
      },
    },
    {
      id: 'kernel-bco', name: 'Kernelized exp. weights (simplified)', family: 'zeroth-order',
      params: [P('eta', 'Learning rate × √(2 ln K/(TK))', 1, 0.1, 10, 0.1), P('h', 'Kernel width h', 0.05, 0.01, 0.3, 0.01), P('gamma', 'Uniform mixing γ', 0.05, 0, 0.5, 0.01)],
      formula: 'w(y) ∝ exp(−η Σₛ ℓₛ K(xₛ,y) / Σ_z q(z)K(z,y)), xₜ ~ q = (1−γ)w + γ·U',
      regret: 'BLE 2017: Õ(√T); this simplified version: no guarantee',
      idea: 'A simplified stand-in for Bubeck, Lee & Eldan (2017), whose kernel method is impractical: exponential weights over a grid (100 points; 15×15 in 2-D), with each importance-weighted loss spread to nearby points by a triangular kernel. It keeps the "exponential weights + kernel smoothing" idea but not BLE\'s convexity-exploiting kernel.',
      create({ T, d, params, rng }) {
        const gr = d === 1 ? grid(1, 100) : grid(2, 15), G = gr.G, n1 = gr.n1;
        const R = Math.max(1, Math.round(params.h * (n1 - 1)));
        const nb = [], nw = [];
        for (let i = 0; i < G; i++) {
          const li = [], lw = [];
          for (let j = 0; j < G; j++) {
            let m = 0;
            for (let k = 0; k < d; k++) m = Math.max(m, Math.abs(gr.pts[i * d + k] - gr.pts[j * d + k]) * (n1 - 1));
            if (m <= R + 1e-9) { li.push(j); lw.push(1 - m / (R + 1)); }
          }
          nb.push(Int32Array.from(li)); nw.push(Float64Array.from(lw));
        }
        const Keff = Math.max(2, G / Math.pow(2 * R + 1, d));
        const eta = params.eta * Math.sqrt((2 * Math.log(Keff)) / (T * Keff)), gamma = params.gamma;
        const logw = new Float64Array(G), q = new Float64Array(G), x = new Float64Array(d);
        let last = 0;
        return {
          select() {
            util.softmaxInto(q, logw, 1);
            for (let i = 0; i < G; i++) q[i] = (1 - gamma) * q[i] + gamma / G;
            last = util.sampleIndex(q, rng);
            return gridPoint(gr, last, x);
          },
          update(_, r) {
            const loss = 1 - r, li = nb[last], lw = nw[last];
            for (let a = 0; a < li.length; a++) {
              const j = li[a];
              let D = 0;
              const lj = nb[j], wj = nw[j];
              for (let b = 0; b < lj.length; b++) D += q[lj[b]] * wj[b];
              logw[j] -= (eta * loss * lw[a]) / D;
            }
          },
          recommend() { return gridPoint(gr, argmaxMasked(logw, rng), new Float64Array(d)); },
        };
      },
    },

    // ---- Gaussian processes
    {
      id: 'gp-ucb', name: 'GP-UCB', family: 'gp',
      params: [P('ell', 'Length-scale ℓ', 0.1, 0.02, 0.5, 0.01), P('scale', 'β scale', 0.2, 0.01, 1, 0.01), P('delta', 'Confidence δ', 0.1, 0.001, 0.5, 0.001)],
      formula: 'xₜ = argmax μₜ₋₁(x) + √βₜ σₜ₋₁(x), βₜ = 2 ln(|D|t²π²/6δ)',
      regret: 'O*(√(T βₜ γₜ)), γₜ = max information gain',
      idea: 'Srinivas, Krause, Kakade & Seeger (2010): fit a GP (RBF kernel, fixed hyperparameters, exact posterior on a 128-point grid, 14×14 in 2-D) and play the upper confidence bound. As in the paper\'s experiments, βₜ is scaled down (default ×0.2).',
      create({ d, sigma, params, rng }) {
        const gr = gpGrid(d), gp = gridGP(gr, params.ell, 0.25, sigma * sigma, 0.5), G = gr.G;
        const acq = new Float64Array(G), x = new Float64Array(d);
        let last = 0;
        return {
          select(t) {
            const sb = Math.sqrt(params.scale * 2 * Math.log((G * t * t * Math.PI * Math.PI) / (6 * params.delta)));
            for (let g = 0; g < G; g++) acq[g] = gp.mu[g] + sb * gp.sd(g);
            last = argmaxMasked(acq, rng);
            return gridPoint(gr, last, x);
          },
          update(_, y) { gp.observe(last, y); },
          recommend() { return gridPoint(gr, argmaxMasked(gp.mu, rng), new Float64Array(d)); },
        };
      },
    },
    {
      id: 'igp-ucb', name: 'IGP-UCB', family: 'gp',
      params: [P('ell', 'Length-scale ℓ', 0.1, 0.02, 0.5, 0.01), P('B', 'RKHS norm bound B', 0.5, 0.05, 3, 0.05), P('delta', 'Confidence δ', 0.1, 0.001, 0.5, 0.001)],
      formula: 'argmax μₜ(x) + (B + R√(2(γₜ + 1 + ln 1/δ)))·σₜ(x), λ = 1 + 2/T',
      regret: 'O(√T (B√γₜ + γₜ))',
      idea: 'Chowdhury & Gopalan (2017): an agnostic (RKHS) version of GP-UCB with regularizer λ = 1 + 2/T in place of the noise variance and a confidence width driven by the information gain γₜ, here the plug-in Σ ½ ln(1 + σ²ₛ₋₁(xₛ)/λ). R is the noise level σ.',
      create({ T, d, sigma, params, rng }) {
        const lam = 1 + 2 / T, gr = gpGrid(d), gp = gridGP(gr, params.ell, 1, lam, 0.5), G = gr.G;
        const acq = new Float64Array(G), x = new Float64Array(d);
        let last = 0, gamma = 0;
        return {
          select() {
            const sb = params.B + sigma * Math.sqrt(2 * (gamma + 1 + Math.log(1 / params.delta)));
            for (let g = 0; g < G; g++) acq[g] = gp.mu[g] + sb * gp.sd(g);
            last = argmaxMasked(acq, rng);
            return gridPoint(gr, last, x);
          },
          update(_, y) { gamma += gp.observe(last, y); },
          recommend() { return gridPoint(gr, argmaxMasked(gp.mu, rng), new Float64Array(d)); },
        };
      },
    },
    {
      id: 'gp-ts', name: 'GP-TS', family: 'gp',
      params: [P('ell', 'Length-scale ℓ', 0.1, 0.02, 0.5, 0.01), P('B', 'RKHS norm bound B', 0.5, 0.05, 3, 0.05), P('delta', 'Confidence δ', 0.1, 0.001, 0.5, 0.001), P('v', 'Inflation × vₜ', 1, 0.05, 2, 0.05)],
      formula: 'f̃ₜ ~ GP(μₜ₋₁, vₜ² kₜ₋₁), vₜ = B + R√(2(γₜ + 1 + ln 2/δ)); xₜ = argmax f̃ₜ',
      regret: 'Õ(γₜ√(dT)) (finite D: d = |D|… log factors)',
      idea: 'Chowdhury & Gopalan (2017): Thompson sampling from an inflated GP posterior (λ = 1 + 2/T). Each step it draws an exact joint posterior sample on a finite set Dₜ (as in the paper\'s analysis): here the 16 most optimistic grid points plus 16 random ones, which keeps the cost independent of t. The theory inflation vₜ explores a lot; lower "Inflation" for plain posterior sampling.',
      create({ T, d, sigma, params, rng }) {
        const lam = 1 + 2 / T, gr = gpGrid(d), G = gr.G, gp = gridGP(gr, params.ell, 1, lam, 0.5);
        const m = Math.min(G, 32), half = m >> 1, cand = new Int32Array(m), Ls = new Float64Array(m * m);
        const e = new Float64Array(m), score = new Float64Array(G), order = new Int32Array(G), used = new Uint8Array(G);
        const x = new Float64Array(d);
        let last = 0, gamma = 0;
        return {
          select() {
            const v = params.v * (params.B + sigma * Math.sqrt(2 * (gamma + 1 + Math.log(2 / params.delta))));
            // candidate set Dₜ: the m/2 most optimistic grid points plus m/2 random others
            for (let g = 0; g < G; g++) { score[g] = gp.mu[g] + 2 * v * gp.sd(g) + 1e-9 * rng.uniform(); order[g] = g; }
            order.sort((a, b) => score[b] - score[a]);
            used.fill(0);
            for (let i = 0; i < half; i++) { cand[i] = order[i]; used[order[i]] = 1; }
            for (let i = half; i < m; i++) { let g; do g = rng.int(G); while (used[g]); used[g] = 1; cand[i] = g; }
            // exact joint posterior sample on Dₜ via Cholesky of the posterior covariance block
            for (let i = 0; i < m; i++) {
              for (let j = 0; j <= i; j++) {
                let s = gp.cov(cand[i], cand[j]) + (i === j ? 1e-9 : 0);
                for (let k = 0; k < j; k++) s -= Ls[i * m + k] * Ls[j * m + k];
                Ls[i * m + j] = i === j ? Math.sqrt(Math.max(s, 1e-14)) : s / Ls[j * m + j];
              }
            }
            for (let i = 0; i < m; i++) e[i] = rng.normal();
            let best = -Infinity;
            for (let i = 0; i < m; i++) {
              let z = 0;
              for (let k = 0; k <= i; k++) z += Ls[i * m + k] * e[k];
              const f = gp.mu[cand[i]] + v * z;
              if (f > best) { best = f; last = cand[i]; }
            }
            return gridPoint(gr, last, x);
          },
          update(_, y) { gamma += gp.observe(last, y); },
          recommend() { return gridPoint(gr, argmaxMasked(gp.mu, rng), new Float64Array(d)); },
        };
      },
    },
    {
      id: 'bkb', name: 'BKB (Nyström GP-UCB)', family: 'gp',
      params: [P('ell', 'Length-scale ℓ', 0.1, 0.02, 0.5, 0.01), P('scale', 'β scale', 0.2, 0.01, 1, 0.01), P('qbar', 'Oversampling q̄', 2, 0.1, 20, 0.1), P('mmax', 'Max dictionary size', 128, 8, 256, 1)],
      formula: 'argmax μ̃ₜ + √βₜ σ̃ₜ; xₜ joins the dictionary w.p. min(1, q̄σ̃²ₜ₋₁(xₜ)/σ²)',
      regret: 'same order as GP-UCB, Õ(T·d_eff²) time',
      idea: 'Calandriello et al. (2019): GP-UCB on a Nyström approximation whose inducing points are sampled with probability proportional to posterior variance, so the dictionary stays near the effective dimension and each step costs O(grid × dictionary). Simplified: the dictionary only grows (points are never resampled away) and is capped (default 128), which allows exact incremental updates.',
      create({ d, sigma, params, rng }) {
        const gr = gpGrid(d), G = gr.G, sf2 = 0.25, s2 = sigma * sigma, m0 = 0.5;
        const Kmat = unitCov(gr, params.ell), cap = Math.round(params.mmax);
        const cnt = new Float64Array(G), sum = new Float64Array(G), inDict = new Uint8Array(G);
        const Phi = new Float64Array(G * cap), Ainv = new Float64Array(cap * cap), w = new Float64Array(cap), bv = new Float64Array(cap);
        const mu = new Float64Array(G).fill(m0), q = new Float64Array(G), resid = new Float64Array(G).fill(sf2);
        const acq = new Float64Array(G), psi = new Float64Array(G), v = new Float64Array(cap), hv = new Float64Array(cap), av = new Float64Array(cap);
        const x = new Float64Array(d);
        let M = 0, last = 0;
        const S = cap; // row stride of Phi and Ainv
        function observe(i, y) {
          // Kalman / Sherman–Morrison update of the weight posterior A⁻¹ = (I + ΦᵀΦ/σ²)⁻¹ and of μ̃, q on the grid
          if (!M) return;
          const off = i * S;
          let den = s2, pred = 0;
          for (let a = 0; a < M; a++) {
            let z = 0;
            for (let c = 0; c < M; c++) z += Ainv[a * S + c] * Phi[off + c];
            v[a] = z; den += Phi[off + a] * z; pred += Phi[off + a] * w[a];
          }
          const r = (y - m0 - pred) / den;
          for (let a = 0; a < M; a++) w[a] += v[a] * r;
          for (let a = 0; a < M; a++) for (let c = 0; c < M; c++) Ainv[a * S + c] -= (v[a] * v[c]) / den;
          for (let g = 0; g < G; g++) {
            const og = g * S;
            let z = 0;
            for (let a = 0; a < M; a++) z += Phi[og + a] * v[a];
            mu[g] += z * r;
            q[g] = Math.max(0, q[g] - (z * z) / den);
          }
        }
        function addToDictionary(i) {
          // new Nyström feature ψ(x) = (k(x, xᵢ) − φ(x)·φ(xᵢ)) / √resid(xᵢ); old features are unchanged
          const li = Math.sqrt(resid[i]), oi = i * S;
          for (let g = 0; g < G; g++) {
            let z = sf2 * Kmat[g * G + i];
            const og = g * S;
            for (let a = 0; a < M; a++) z -= Phi[og + a] * Phi[oi + a];
            psi[g] = z / li;
          }
          // grow A = I + Σ c φφᵀ/σ² by one row/column and invert by the Schur complement
          let alpha = 1, beta = 0;
          av.fill(0, 0, M);
          for (let g = 0; g < G; g++) {
            if (!cnt[g]) continue;
            const cw = (cnt[g] * psi[g]) / s2, og = g * S;
            for (let a = 0; a < M; a++) av[a] += cw * Phi[og + a];
            alpha += cw * psi[g];
            beta += ((sum[g] - cnt[g] * m0) * psi[g]) / s2;
          }
          let sch = alpha;
          for (let a = 0; a < M; a++) { let z = 0; for (let c = 0; c < M; c++) z += Ainv[a * S + c] * av[c]; hv[a] = z; sch -= av[a] * z; }
          sch = Math.max(sch, 1e-12);
          for (let a = 0; a < M; a++) {
            for (let c = 0; c < M; c++) Ainv[a * S + c] += (hv[a] * hv[c]) / sch;
            Ainv[a * S + M] = -hv[a] / sch; Ainv[M * S + a] = -hv[a] / sch;
          }
          Ainv[M * S + M] = 1 / sch;
          bv[M] = beta;
          for (let g = 0; g < G; g++) {
            const og = g * S;
            let z = 0;
            for (let a = 0; a < M; a++) z += hv[a] * Phi[og + a];
            const e = psi[g] - z;
            q[g] += (e * e) / sch;
            Phi[og + M] = psi[g];
            resid[g] = Math.max(0, resid[g] - psi[g] * psi[g]);
          }
          M++;
          inDict[i] = 1;
          for (let a = 0; a < M; a++) { let z = 0; for (let c = 0; c < M; c++) z += Ainv[a * S + c] * bv[c]; w[a] = z; }
          for (let g = 0; g < G; g++) { const og = g * S; let z = m0; for (let a = 0; a < M; a++) z += Phi[og + a] * w[a]; mu[g] = z; }
        }
        return {
          select(t) {
            const sb = Math.sqrt(params.scale * 2 * Math.log((G * t * t * Math.PI * Math.PI) / 0.6));
            for (let g = 0; g < G; g++) acq[g] = mu[g] + sb * Math.sqrt(resid[g] + q[g]);
            last = argmaxMasked(acq, rng);
            return gridPoint(gr, last, x);
          },
          update(_, y) {
            const prior = resid[last] + q[last];
            cnt[last]++; sum[last] += y;
            if (M) { const off = last * S; for (let a = 0; a < M; a++) bv[a] += ((y - m0) * Phi[off + a]) / s2; }
            observe(last, y);
            if (!inDict[last] && M < cap && resid[last] > 1e-10 * sf2 && rng.uniform() < Math.min(1, (params.qbar * prior) / s2)) addToDictionary(last);
          },
          recommend() { return gridPoint(gr, argmaxMasked(mu, rng), new Float64Array(d)); },
        };
      },
    },

    // ---- safe exploration
    {
      id: 'safeopt', name: 'SafeOpt', family: 'safe',
      params: [P('ell', 'Length-scale ℓ', 0.1, 0.02, 0.5, 0.01), P('beta', 'Confidence √β', 2, 0.5, 5, 0.1), P('L', 'Lipschitz constant L', 5, 0.5, 30, 0.5)],
      formula: 'xₜ = argmax_{Mₜ ∪ Gₜ} uₜ(x) − lₜ(x), Sₜ = {x : lₜ(x) ≥ h}',
      regret: 'safe w.h.p.; ε-optimal on the reachable safe set',
      idea: 'Sui, Gotovos, Burdick & Krause (2015): only evaluate points whose GP lower bound clears the threshold h, starting from a known safe seed. Among safe points it samples the most uncertain potential maximizer or expander (a point whose optimistic value could certify a new point via the Lipschitz constant L). Recommends the safe point with the best lower bound. Its uncertainty-driven sampling explores the whole reachable safe set, so its cumulative regret is high by design.',
      create(ctx) { return makeSafe(ctx, false); },
    },
    {
      id: 'stageopt', name: 'StageOpt', family: 'safe',
      params: [P('ell', 'Length-scale ℓ', 0.1, 0.02, 0.5, 0.01), P('beta', 'Safety √β', 2, 0.5, 5, 0.1), P('L', 'Lipschitz constant L', 5, 0.5, 30, 0.5), P('stage', 'Expansion share of T', 0.3, 0.05, 0.9, 0.05), P('betaF', 'UCB √β (stage 2)', 2, 0.5, 5, 0.1)],
      formula: 'stage 1: argmax_{Gₜ} uₜ − lₜ; stage 2: argmax_{Sₜ} μₜ + β_f σₜ',
      regret: 'same safety guarantee; staged sample complexity',
      idea: 'Sui, Zhuang, Burdick & Yue (2018): separate safe expansion from optimisation. First grow the safe set by sampling uncertain expanders; then run GP-UCB restricted to the safe set. Here the safety function is f itself.',
      create(ctx) { return makeSafe(ctx, true); },
    },
  ];

  /** SafeOpt (stage = false) and StageOpt (stage = true) on the GP grid; safety function = f. */
  function makeSafe({ T, d, sigma, threshold, safeSeed, params, rng }, staged) {
    const gr = gpGrid(d), G = gr.G, gp = gridGP(gr, params.ell, 0.25, sigma * sigma, 0.5);
    const h = threshold, beta = params.beta, L = params.L;
    const lo = new Float64Array(G).fill(-Infinity), up = new Float64Array(G).fill(Infinity);
    const safe = new Uint8Array(G), cand = new Uint8Array(G), score = new Float64Array(G), dmin = new Float64Array(G);
    const x = new Float64Array(d);
    const seed = nearestIndex(gr, safeSeed || new Float64Array(d).fill(0.5));
    safe[seed] = 1;
    if (h > -Infinity) lo[seed] = h;
    let last = 0;
    const T0 = Math.round(params.stage ? params.stage * T : 0);
    function refreshSets() {
      for (let g = 0; g < G; g++) {
        const s = gp.sd(g), m = gp.mu[g];
        lo[g] = Math.max(lo[g], m - beta * s);
        up[g] = Math.min(up[g], m + beta * s);
        if (h === -Infinity || lo[g] >= h) safe[g] = 1;
      }
      // distance from each safe point to the nearest unsafe grid point
      let anyUnsafe = false;
      for (let g = 0; g < G; g++) if (!safe[g]) { anyUnsafe = true; break; }
      if (!anyUnsafe) { dmin.fill(Infinity); return; }
      for (let g = 0; g < G; g++) {
        if (!safe[g]) { dmin[g] = 0; continue; }
        let m = Infinity;
        for (let q = 0; q < G; q++) if (!safe[q]) { const z = dist2(gr, g, q); if (z < m) m = z; }
        dmin[g] = Math.sqrt(m);
      }
    }
    function expanders() {
      let any = false;
      for (let g = 0; g < G; g++) {
        cand[g] = safe[g] && dmin[g] < Infinity && up[g] - L * dmin[g] >= h ? 1 : 0;
        if (cand[g]) any = true;
      }
      return any;
    }
    return {
      select(t) {
        refreshSets();
        let pick = -1;
        if (staged) {
          if (t <= T0 && expanders()) {
            for (let g = 0; g < G; g++) score[g] = up[g] - lo[g];
            pick = argmaxMasked(score, rng, cand);
          } else {
            const bf = params.betaF;
            for (let g = 0; g < G; g++) score[g] = gp.mu[g] + bf * gp.sd(g);
            pick = argmaxMasked(score, rng, safe);
          }
        } else {
          let lmax = -Infinity;
          for (let g = 0; g < G; g++) if (safe[g] && lo[g] > lmax) lmax = lo[g];
          expanders();
          for (let g = 0; g < G; g++) {
            if (safe[g] && up[g] >= lmax) cand[g] = 1;
            score[g] = up[g] - lo[g];
          }
          pick = argmaxMasked(score, rng, cand);
        }
        if (pick < 0) pick = argmaxMasked(lo, rng, safe);
        if (pick < 0) pick = seed;
        last = pick;
        return gridPoint(gr, last, x);
      },
      update(_, y) { gp.observe(last, y); },
      recommend() {
        let b = -1, bv = -Infinity;
        for (let g = 0; g < G; g++) {
          if (!safe[g]) continue;
          const v = Math.max(lo[g], gp.mu[g] - beta * gp.sd(g));
          if (v > bv) { bv = v; b = g; }
        }
        return gridPoint(gr, b < 0 ? seed : b, new Float64Array(d));
      },
    };
  }

  // ---------------------------------------------------------- environments ----
  /*
   * Each environment's make(rng) draws one instance:
   *   { f(x, c), fstar(c), h (safety threshold, −∞ if none), seed (a known safe point) }
   * All objective values lie roughly in [0, 1].
   */
  const NO_SAFETY = -Infinity;
  function instance1D(f, extra) {
    const fs = maximize1D(f);
    return Object.assign({ f: (x) => f(x[0]), fstar: () => fs, h: NO_SAFETY, seed: null }, extra || {});
  }

  function branin(u, v) {
    const b = 5.1 / (4 * Math.PI * Math.PI), c = 5 / Math.PI, t = 1 / (8 * Math.PI);
    const q = v - b * u * u + c * u - 6;
    return q * q + 10 * (1 - t) * Math.cos(u) + 10;
  }
  const BRANIN_MIN = 0.397887, BRANIN_MAX = branin(-5, 0);

  const ENVIRONMENTS = [
    {
      id: 'smooth', name: 'Smooth unimodal (1-D)', short: 'Smooth',
      description: 'f(x) = 0.05 + 0.9·exp(−(x−a)²/(2·0.15²)) with the peak a ~ U(0.2, 0.8). Every method should find it.',
      tags: ['1-D', 'smooth'], d: 1,
      make(rng) {
        const a = 0.2 + 0.6 * rng.uniform();
        return instance1D((x) => 0.05 + 0.9 * gauss(x, a, 0.15));
      },
    },
    {
      id: 'multimodal', name: 'Multimodal bumps (1-D)', short: 'Multimodal',
      description: 'A sum of four narrow Gaussian bumps (widths 0.025–0.06), one of height 0.9 and three decoys of height 0.45–0.75, at random positions.',
      tags: ['1-D', 'multimodal'], d: 1,
      make(rng) {
        const best = rng.int(4), cs = [], ws = [], hs = [];
        for (let k = 0; k < 4; k++) {
          cs.push((k + 0.15 + 0.7 * rng.uniform()) / 4);
          ws.push(0.025 + 0.035 * rng.uniform());
          hs.push(k === best ? 0.9 : 0.45 + 0.3 * rng.uniform());
        }
        return instance1D((x) => { let s = 0.05; for (let k = 0; k < 4; k++) s += hs[k] * gauss(x, cs[k], ws[k]); return s; });
      },
    },
    {
      id: 'needle', name: 'Needle in a haystack (1-D)', short: 'Needle',
      description: 'A broad decoy hill (peak 0.55) plus a needle of height +0.4 and half-width 0.015 elsewhere. Smoothness-based methods must keep exploring to find it.',
      tags: ['1-D', 'needle', 'hard'], d: 1,
      make(rng) {
        const b = 0.2 + 0.6 * rng.uniform();
        let a = rng.uniform();
        while (Math.abs(a - b) < 0.3 || a < 0.05 || a > 0.95) a = rng.uniform();
        return instance1D((x) => 0.3 + 0.25 * gauss(x, b, 0.2) + 0.4 * Math.max(0, 1 - Math.abs(x - a) / 0.015));
      },
    },
    {
      id: 'branin', name: 'Branin, rescaled (2-D)', short: 'Branin 2-D',
      description: 'The Branin function on [0,1]², log-rescaled to f = 1 − ln(1 + (B − B_min)/10)/ln(1 + (B_max − B_min)/10) and randomly mirrored; three global optima with f* = 1.',
      tags: ['2-D', 'multimodal'], d: 2,
      make(rng) {
        const fx = rng.uniform() < 0.5, fy = rng.uniform() < 0.5, den = Math.log(1 + (BRANIN_MAX - BRANIN_MIN) / 10);
        const f = (x) => {
          const a = fx ? 1 - x[0] : x[0], b = fy ? 1 - x[1] : x[1];
          return Math.max(0, 1 - Math.log(1 + Math.max(0, branin(15 * a - 5, 15 * b) - BRANIN_MIN) / 10) / den);
        };
        return { f, fstar: () => 1, h: NO_SAFETY, seed: null };
      },
    },
    {
      id: 'convex', name: 'Concave quadratic (1-D, convex loss)', short: 'Convex',
      description: 'f(x) = 0.9 − 1.2(x − a)² with a ∈ [0.15, 0.3] ∪ [0.7, 0.85]: a convex loss for bandit convex optimisation (FKM, BGD, SCR) started at x = 0.5.',
      tags: ['1-D', 'convex'], d: 1,
      make(rng) {
        const u = rng.uniform(), a = u < 0.5 ? 0.15 + 0.3 * u : 0.7 + 0.3 * (u - 0.5);
        return { f: (x) => 0.9 - 1.2 * (x[0] - a) * (x[0] - a), fstar: () => 0.9, h: NO_SAFETY, seed: null };
      },
    },
    {
      id: 'safe', name: 'Safe optimisation (1-D, h = 0.35)', short: 'Safe',
      description: 'A main bump (0.9) and a secondary bump (0.65) separated by a valley below the safety threshold h = 0.35. Evaluations with f(x) < h count as unsafe; a safe seed near the main bump is given.',
      tags: ['1-D', 'safety'], d: 1, safe: true,
      make(rng) {
        const mirror = rng.uniform() < 0.5;
        let a = 0.25 + 0.1 * rng.uniform(), b = a + 0.45 + 0.1 * rng.uniform();
        const s = rng.uniform() < 0.5 ? -1 : 1;
        let seed = a + 0.1 * s;
        if (mirror) { a = 1 - a; b = 1 - b; seed = 1 - seed; }
        return instance1D((x) => 0.05 + 0.85 * gauss(x, a, 0.12) + 0.6 * gauss(x, b, 0.06), { h: 0.35, seed: [seed] });
      },
    },
    {
      id: 'contextual', name: 'Contextual peak (1-D, context c)', short: 'Contextual',
      description: 'Each step reveals c ~ U(0,1); f(x, c) = 0.05 + 0.9·exp(−(x − m(c))²/(2·0.1²)), m(c) = 0.5 + 0.3 sin(2πc + φ). Only Contextual Zooming uses c; the others chase the best fixed x.',
      tags: ['1-D', 'context'], d: 1, contextual: true,
      make(rng) {
        const phi = 2 * Math.PI * rng.uniform();
        const m = (c) => 0.5 + 0.3 * Math.sin(2 * Math.PI * c + phi);
        return { f: (x, c) => 0.05 + 0.9 * gauss(x[0], m(c), 0.1), fstar: () => 0.95, h: NO_SAFETY, seed: null };
      },
    },
  ];
  const ENV_BY_ID = {};
  for (const e of ENVIRONMENTS) ENV_BY_ID[e.id] = e;
  const ALGO_BY_ID = {};
  for (const a of ALGORITHMS) ALGO_BY_ID[a.id] = a;

  function defaultParams(id) {
    const out = {};
    for (const p of ALGO_BY_ID[id].params) out[p.key] = p.default;
    return out;
  }

  // ---------------------------------------------------------------- runner ----
  const EVAL_CONTEXTS = [0.1, 0.3, 0.5, 0.7, 0.9];

  function createRunner(config) {
    const env = ENV_BY_ID[config.env];
    if (!env) throw new Error('Unknown environment: ' + config.env);
    const T = config.T | 0, runs = config.runs | 0, seed = config.seed >>> 0, d = env.d;
    if (T < 4) throw new Error('T must be at least 4');
    const extra = config.extra || {};
    const sigma = extra.sigma != null ? +extra.sigma : 0.1;
    const L = Math.min(T, 100), checkpoints = new Int32Array(L), xs = new Float64Array(L);
    for (let k = 0; k < L; k++) { checkpoints[k] = Math.max(1, Math.round((T * (k + 1)) / L)); xs[k] = checkpoints[k]; }
    const noise = new Float64Array(T), ctxs = new Float64Array(T), xb = new Float64Array(d), bestX = new Float64Array(d);

    const slots = config.algorithms.map((spec) => {
      const def = ALGO_BY_ID[spec.id];
      if (!def) throw new Error('Unknown algorithm: ' + spec.id);
      return {
        def, params: Object.assign(defaultParams(spec.id), spec.params || {}),
        cumSum: new Float64Array(L), cumSq: new Float64Array(L), simSum: new Float64Array(L), simSq: new Float64Array(L),
        finals: [], simFinals: [], unsafe: 0, timeMs: 0,
      };
    });
    let done = 0;

    function simpleRegret(inst, agent, bestY) {
      const rec = (c) => {
        const r = agent.recommend ? agent.recommend(c) : bestY > -Infinity ? bestX : null;
        if (!r) return 0.5;
        for (let j = 0; j < d; j++) xb[j] = clamp(r[j], 0, 1);
        return inst.fstar(c) - inst.f(xb, c);
      };
      if (!env.contextual) return rec(0.5);
      let s = 0;
      for (const c of EVAL_CONTEXTS) s += rec(c);
      return s / EVAL_CONTEXTS.length;
    }

    function runOne(run) {
      const envRng = Rng(hash32(seed, run, 'env'));
      const inst = env.make(envRng);
      for (let t = 0; t < T; t++) noise[t] = sigma * envRng.normal();
      for (let t = 0; t < T; t++) ctxs[t] = env.contextual ? envRng.uniform() : 0.5;
      for (const slot of slots) {
        const rng = Rng(hash32(seed, run, slot.def.id));
        const t0 = now();
        const agent = slot.def.create({
          T, d, sigma, threshold: inst.h, safeSeed: inst.seed, contextual: !!env.contextual, params: slot.params, rng,
        });
        let cum = 0, unsafe = 0, k = 0, bestY = -Infinity, sr = 0, evalMs = 0;
        for (let t = 0; t < T; t++) {
          const c = ctxs[t];
          const x = agent.select(t + 1, c);
          for (let j = 0; j < d; j++) { const v = +x[j]; xb[j] = v >= 0 ? (v <= 1 ? v : 1) : 0; }
          const v = inst.f(xb, c), y = v + noise[t];
          cum += inst.fstar(c) - v;
          if (v < inst.h) unsafe++;
          if (!agent.recommend && y > bestY) { bestY = y; bestX.set(xb); }
          agent.update(Float64Array.from(xb), y, t + 1, c);
          if (t + 1 === checkpoints[k]) {
            const e0 = now();
            sr = simpleRegret(inst, agent, bestY);
            evalMs += now() - e0;
            while (k < L && checkpoints[k] === t + 1) {
              slot.cumSum[k] += cum; slot.cumSq[k] += cum * cum;
              slot.simSum[k] += sr; slot.simSq[k] += sr * sr;
              k++;
            }
          }
        }
        slot.timeMs += now() - t0 - evalMs;
        slot.finals.push(cum);
        slot.simFinals.push(sr);
        slot.unsafe += unsafe;
      }
    }

    return {
      config: { env: env.id, T, runs, seed, extra: { sigma } },
      get done() { return done; },
      get finished() { return done >= runs; },
      step(n = 1) {
        const stop = Math.min(runs, done + n);
        while (done < stop) runOne(done++);
        return done;
      },
      results() {
        const R = Math.max(1, done);
        const se = (s, q, i) => { const m = s[i] / R; return R > 1 ? Math.sqrt(Math.max(0, q[i] / R - m * m) / (R - 1)) : 0; };
        return slots.map((s) => {
          const regret = new Float64Array(L), regretSE = new Float64Array(L), simple = new Float64Array(L), simpleSE = new Float64Array(L);
          for (let k = 0; k < L; k++) {
            regret[k] = s.cumSum[k] / R; regretSE[k] = se(s.cumSum, s.cumSq, k);
            simple[k] = s.simSum[k] / R; simpleSE[k] = se(s.simSum, s.simSq, k);
          }
          const sorted = s.finals.slice().sort((a, b) => a - b);
          const summary = {
            finalRegret: regret[L - 1], finalSE: regretSE[L - 1],
            finalMedian: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
            simpleRegret: simple[L - 1], simpleSE: simpleSE[L - 1],
            unsafe: s.unsafe / R,
            msPerRun: done ? s.timeMs / done : 0,
          };
          return {
            id: s.def.id, name: s.def.name, family: s.def.family, params: s.params, runs: done,
            x: xs, curves: { regret, simple }, curvesSE: { regret: regretSE, simple: simpleSE },
            summary, finals: sorted,
          };
        });
      },
    };
  }

  function simulate(config) {
    const r = createRunner(config);
    r.step(config.runs);
    return r.results();
  }

  // ---------------------------------------------------------- registration ----
  const setting = MAB.registerSetting({
    id: 'continuum',
    name: 'Continuum-armed & GP',
    short: 'Continuum',
    blurb: 'Pick a point x in [0,1] (or [0,1]²) and observe f(x) + noise: Lipschitz discretization, optimistic tree search, bandit convex optimisation, Gaussian-process and safe bandits.',
    families: [
      { id: 'baseline', name: 'Baseline', blurb: 'No learning: shows the cost of pure exploration.' },
      { id: 'discretization', name: 'Discretization & zooming', blurb: 'Turn the continuum into arms: a fixed grid sized for the horizon, or an adaptive set of active points that zooms in on good regions.' },
      { id: 'tree-search', name: 'Optimistic tree search', blurb: 'Recursively partition the domain and refine the cells whose optimistic value, statistical bonus plus smoothness, is highest.' },
      { id: 'zeroth-order', name: 'Bandit convex optimisation', blurb: 'Estimate a gradient from function values at perturbed points and take gradient steps; built for convex losses.' },
      { id: 'gp', name: 'Gaussian-process bandits', blurb: 'Model f with a Gaussian process and use its posterior to trade off exploration and exploitation.' },
      { id: 'safe', name: 'Safe exploration', blurb: 'Gaussian-process methods that only evaluate points whose value is certified above a safety threshold.' },
    ],
    environments: ENVIRONMENTS.map((e) => ({ id: e.id, name: e.name, short: e.short, description: e.description, tags: e.tags, d: e.d })),
    algorithms: ALGORITHMS,
    defaultEnv: 'multimodal',
    defaultLineup: ['uniform', 'ucb-grid', 'zooming', 'hoo', 'stosoo', 'bgd', 'gp-ucb', 'gp-ts'],
    presets: [
      { label: 'Tree search', algos: ['doo', 'soo', 'stosoo', 'hoo', 'hct', 'vhct', 'poo', 'uct'] },
      { label: 'Discretization', algos: ['uniform', 'ucb-grid', 'zooming', 'contextual-zooming', 'hoo', 'gp-ucb'] },
      { label: 'Convex (BCO)', algos: ['uniform', 'fkm', 'bgd', 'scr', 'kernel-bco', 'gp-ucb'], env: 'convex' },
      { label: 'Gaussian processes', algos: ['uniform', 'gp-ucb', 'igp-ucb', 'gp-ts', 'bkb', 'hoo'] },
      { label: 'Safe', algos: ['uniform', 'gp-ucb', 'safeopt', 'stageopt'], env: 'safe' },
      { label: 'Contextual', algos: ['uniform', 'zooming', 'contextual-zooming', 'hoo', 'gp-ucb'], env: 'contextual' },
    ],
    controls: {
      T: { label: 'Horizon (T steps)', options: [200, 500, 1000, 2000], default: 500 },
      runs: { label: 'Runs averaged', options: [10, 20, 50, 100], default: 20 },
      extra: [{ key: 'sigma', label: 'Noise σ', min: 0.01, max: 0.5, step: 0.01, default: 0.1 }],
    },
    xLabel: 'step t',
    metrics: [
      { key: 'regret', label: 'Cumulative regret', better: 'lower', fmt: 'num', band: true, note: 'Pseudo-regret T·f* − Σ f(xₜ); shaded bands are 95% confidence intervals across runs.' },
      { key: 'simple', label: 'Simple regret', better: 'lower', fmt: 'fixed3', band: true, note: 'f* − f(recommended x): how good the point each method would recommend now is (contextual: averaged over 5 contexts).' },
    ],
    final: { key: 'finalRegret', se: 'finalSE', median: 'finalMedian', label: 'Cumulative regret', better: 'lower' },
    tiles: [
      { key: 'simpleRegret', label: 'Best recommendation', better: 'lower', fmt: 'fixed3', sub: 'simple regret at T' },
    ],
    columns: [
      { key: 'finalRegret', label: 'Regret', fmt: 'num', better: 'lower' },
      { key: 'finalSE', label: '95% CI', fmt: 'ci' },
      { key: 'finalMedian', label: 'Median', fmt: 'num', better: 'lower' },
      { key: 'simpleRegret', label: 'Simple regret', fmt: 'fixed3', better: 'lower' },
      { key: 'unsafe', label: 'Unsafe evals', fmt: 'num', better: 'lower' },
      { key: 'msPerRun', label: 'ms / run', fmt: 'fixed2', better: 'lower' },
    ],
    heatmap: false,
    createRunner,
  });
  // Extras for tests and scripts.
  setting.ENV_BY_ID = ENV_BY_ID;
  setting.ALGO_BY_ID = ALGO_BY_ID;
  setting.envDefs = ENVIRONMENTS;
  setting.defaultParams = defaultParams;
  setting.simulate = simulate;
  return setting;
});
