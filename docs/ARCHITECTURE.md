# Architecture

Everything is plain JavaScript with no dependencies. The same files run in the browser
(as classic `<script>` tags) and in Node (as CommonJS modules).

```
src/mab.js              core: RNG, helpers, the K-armed & contextual arena, registries
src/algos/*.js          add-ons that register more algorithms/environments into the arena
src/settings/*.js       other problem settings (best-arm identification, dueling, ...)
src/index.js            Node entry point: loads mab.js and every add-on, exports MAB
dashboard/              the web UI; loads the same files with <script> tags
test/*.test.js          node --test suites
```

## Module pattern

Every add-on file uses this wrapper so it works in both environments:

```js
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../mab.js'));
  else factory(root.MAB);
})(typeof self !== 'undefined' ? self : this, function (MAB) {
  'use strict';
  const { Rng, util } = MAB;
  // ...
  MAB.register({ families, environments, algorithms });   // arena add-on
  // or
  MAB.registerSetting({ ... });                          // new setting
});
```

`MAB.util` exposes shared helpers: `stats(K)`, `record(s, a, r)`, `argmax(values, rng)` (random
tie-break), `untried(n)`, `sampleIndex(p, rng)`, `softmaxInto(out, x, scale)`, `klBern(p, q)`,
`klUcbBern(p, n, bound)`, `klIndex(bern, mean, n, bound, s2)`, `posterior(bern, s2, sum, n)`,
`samplePosterior(rng, bern, s2, sum, n)`, `normCdf`, `normPdf`, `normInv`, `logPlus`,
`binomHalf(rng, m)`, `exp3Gamma(K, T)`, `now()`.
`MAB.Rng(seed)` gives `uniform()`, `normal()`, `gamma(k)`, `beta(a, b)`, `int(n)`.
`MAB.hash32(...parts)` derives seeds from numbers and strings.

## The arena (setting `kab`)

### Algorithms

```js
{
  id: 'ucb1',                 // unique across the arena
  name: 'UCB1',               // display name
  family: 'optimism',         // a registered family id
  params: [{ key: 'c', label: 'c', default: 1.414, min: 0, max: 5, step: 0.01 }],
  formula: 'argmaxₐ Q̂ₐ + c·s·√(ln t / nₐ)',   // one line, Unicode math
  regret: 'O(Σ log T / Δₐ)',                   // guarantee, short
  idea: 'One or two plain sentences on how it works and when it shines.',
  create(K, ctx) {            // ctx = { T, env, params, rng, d, info }
    return {
      select(t, X) { return arm; },   // t = 1..T; X = Float64Array(K*d), row a = arm a's features
      update(arm, reward, t) {},
      observe(arm, reward, t) {},     // optional: side observations (graph-feedback envs)
    };
  },
}
```

- `ctx.env.type` is `'bernoulli'` or `'gaussian'`; `ctx.env.sigma` is the noise scale, `ctx.env.scale`
  the bonus scale (1 for Bernoulli, √2 for unit Gaussian), and `ctx.env.normalize(r)` maps a
  reward into [0, 1] for methods that need bounded rewards.
- `X` is the per-round context. Context-free environments pass `d = 1` and `X` all ones.
  Algorithms must work (not crash) on every environment, contextual or not.
- `ctx.info` is whatever the environment's `fill` returned for this run (e.g. `{ graph }`).
- All randomness must come from `ctx.rng`. Never use `Math.random()`.

### Environments

```js
{
  id, name, short, description,
  type: 'bernoulli' | 'gaussian',
  stationary: true,
  tags: ['changes' | 'context' | 'rested' | 'adversarial' | 'side observations' | 'heavy tails' | ...],
  d: 5,                        // context dimension (omit for 1)
  contextual: true,            // optional
  rested: true,                // optional, see below
  sigma, scale, normalize,     // optional overrides
  fill(K, T, rng, means, rewards, contexts) { return info; },
}
```

`fill` writes `means[t*K + a]` (expected reward) and `rewards[t*K + a]` (the realized reward,
shared by every algorithm in the run), and `contexts[(t*K + a)*d + j]`. It may return an `info`
object handed to every algorithm's `create` (e.g. `{ graph: Uint8Array(K*K) }` where
`graph[a*K + b] = 1` means pulling `a` also reveals `b`'s reward; the runner then calls
`observe(b, rewards[t*K + b], t)`).

**Rested environments** (`rested: true`): arms change when pulled, not with time. `fill` writes
`means[n*K + a]` and `rewards[n*K + a]` for the arm's (n+1)-th pull. Regret is measured against
the better of the greedy oracle and the best fixed arm.

Regret is pseudo-regret against the best mean at each step.

### Registering

`MAB.register({ families, environments, algorithms })`. A family is `{ id, name, blurb, before? }`;
`before` places it ahead of an existing family in listings. Ids must be unique (registration throws).

## Settings

A setting is a different problem type with its own environments, algorithms, measures and runner.
The dashboard renders any setting from this description.

```js
MAB.registerSetting({
  id: 'bai', name: 'Best-arm identification', short: 'Best arm', blurb: 'One sentence.',
  families: [{ id, name, blurb }],
  environments: [{ id, name, description, tags }],
  algorithms: [{ id, name, family, params, formula, regret, idea, ...anything the runner needs }],
  defaultEnv: 'id', defaultLineup: ['ids', '≤ 8'],
  presets: [{ label, algos: [...], env: 'optional env id' }],
  controls: {
    K: { label: 'Arms (K)', min: 2, max: 50, default: 10 },              // omit if not used
    T: { label: 'Budget (pulls)', options: [500, 1000, 2000], default: 1000 }, // omit if not used
    runs: { label: 'Runs averaged', options: [20, 50, 100, 200], default: 100 },
    extra: [{ key: 'delta', label: 'Confidence δ', min: 0.001, max: 0.5, step: 0.001, default: 0.05 }],
  },
  xLabel: 'pulls',             // label of the curve x-axis
  metrics: [                   // curves; the first is shown by default
    { key: 'error', label: 'Error rate', better: 'lower', fmt: 'pct', band: false, smooth: false,
      zeroToOne: true, note: 'One sentence shown under the chart.' },
  ],
  final: { key: 'samples', se: 'samplesSE', median: 'samplesMedian', label: 'Samples to stop', better: 'lower' },
  tiles: [{ key: 'error', label: 'Most accurate', better: 'lower', fmt: 'pct', sub: 'wrong answers' }],
  columns: [{ key: 'samples', label: 'Samples', fmt: 'num', better: 'lower' }, { key: 'samplesSE', label: '95% CI', fmt: 'ci' }],
  heatmap: false,              // true only if results carry pullShare (share by arm rank, best first)
  createRunner(config) { ... },
});
```

Formats: `num`, `int`, `pct` (0..1 shown as %), `fixed2`, `fixed3`, `ci` (shown as ± 1.96 × value).

`config = { env, K, T, runs, seed, extra: { key: value }, algorithms: [{ id, params }] }`.
Fill missing params from each algorithm's `params[].default`.

The runner must be incremental so the dashboard can redraw while it works:

```js
{
  get done() {}, get finished() {},
  step(n) {},        // run n more Monte-Carlo runs of every algorithm
  results() {        // summary of the runs so far
    return [{
      id, name, family, params, runs,
      curves: { metricKey: Float64Array(L) },     // one per metric, same length L
      curvesSE: { metricKey: Float64Array(L) },   // optional standard errors (for band: true)
      x: Float64Array(L),                          // optional x values (default 1..L)
      summary: { [key]: number, msPerRun },        // every key used by final/tiles/columns
      finals: [],                                  // optional sorted per-run values of final.key
      pullShare: [],                               // optional, with heatmap: true
    }];
  },
}
```

Fairness rules for every runner: seed each run's environment from `hash32(seed, run, 'env')` and
each algorithm from `hash32(seed, run, algorithmId)`, so results do not depend on which other
algorithms are selected; share environment randomness across algorithms within a run wherever
the setting allows (common random numbers).

Performance budget: with default controls and an 8-algorithm lineup, one `step(1)` should take
well under 100 ms in Node, because the dashboard runs on the page's main thread.
