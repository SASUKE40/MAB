/* Bandit Algorithm Arena — dashboard UI. Depends on window.MAB (src/mab.js plus add-ons). */
(function () {
  'use strict';
  const M = window.MAB;
  const $ = (s, r = document) => r.querySelector(s);
  const SLOTS = 8;

  // Presets for the K-armed arena; other settings bring their own.
  const KAB_LINEUP = ['greedy', 'egreedy', 'ucb1', 'ucb-tuned', 'klucb', 'thompson', 'gradient', 'exp3'];
  const KAB_PRESETS = [
    { label: 'Classic lineup', algos: KAB_LINEUP },
    { label: 'Optimism vs Bayes', algos: ['ucb1', 'ucb-tuned', 'klucb', 'moss', 'bayes-ucb', 'thompson'] },
    { label: 'Undirected exploration', algos: ['greedy', 'egreedy', 'egreedy-decay', 'etc', 'softmax', 'optimistic', 'thompson'] },
    { label: 'KL & divergence', algos: ['klucb', 'klucb-plus', 'klucb-pp', 'imed', 'bayes-ucb', 'thompson', 'ucb1'] },
    { label: 'Bayesian decision', algos: ['thompson', 'bayes-ucb', 'gittins', 'kg', 'ids', 'phe', 'ucb1'] },
    { label: 'UCB variants', algos: ['ucb1', 'ucb2', 'ucb-tuned', 'ucb-v', 'ucb-normal', 'moss', 'moss-anytime', 'klucb'] },
    { label: 'Elimination', algos: ['successive-elim', 'improved-ucb', 'seq-halving', 'etc', 'ucb1', 'thompson'] },
    { label: 'Learning automata', algos: ['pursuit', 'lri', 'reinforcement-comparison', 'gradient', 'softmax', 'softmax-decay', 'egreedy'] },
    { label: 'Adversarial', algos: ['exp3', 'exp3-ix', 'exp3p', 'tsallis-inf', 'thompson', 'klucb'] },
    { label: 'Contextual', algos: ['linucb', 'thompson', 'klucb', 'ucb1', 'egreedy', 'gittins'], env: 'contextual' },
    { label: 'Tracking change', algos: ['egreedy-const', 'ucb1', 'thompson', 'sw-ucb', 'd-ucb', 'dts', 'm-ucb', 'exp3s'], env: 'abrupt' },
  ];
  const EXTRA_PRESETS = window.MAB_PRESETS || {};

  // ------------------------------------------------------------ utils ----
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function fmtNum(v) {
    if (!Number.isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a >= 100) return Math.round(v).toLocaleString('en-US');
    if (a >= 10) return v.toFixed(1);
    if (a >= 0.01 || v === 0) return v.toFixed(2);
    return v.toPrecision(2);
  }
  function fmtPct(v) { return Number.isFinite(v) ? (v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0) + '%' : '—'; }
  function fmtBy(kind, v) {
    if (v == null || !Number.isFinite(v)) return '—';
    switch (kind) {
      case 'pct': return fmtPct(v);
      case 'int': return Math.round(v).toLocaleString('en-US');
      case 'fixed2': return v.toFixed(2);
      case 'fixed3': return v.toFixed(3);
      case 'ci': return '± ' + fmtNum(1.96 * v);
      default: return fmtNum(v);
    }
  }
  function fmtTick(v) {
    if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString('en-US');
    return +v.toFixed(6) + '';
  }
  function niceStep(range, count) {
    const raw = range / Math.max(1, count);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
  }
  function ticks(min, max, count) {
    const s = niceStep(max - min || 1, count);
    const out = [];
    for (let v = Math.ceil(min / s - 1e-9) * s; v <= max + s * 1e-6; v += s) out.push(+v.toFixed(10));
    return out;
  }
  const measureCtx = document.createElement('canvas').getContext('2d');
  function textWidth(s, font) { measureCtx.font = font; return measureCtx.measureText(s).width; }
  const LABEL_FONT = '12px "IBM Plex Sans", system-ui, sans-serif';
  const NUM_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];

  // ------------------------------------------------------------ state ----
  const SETTINGS = M.SETTINGS;
  const views = {};          // per-setting UI state, kept when switching
  let S = null, V = null;    // current setting and its view state

  function byId(setting) {
    if (!setting._byId) {
      setting._byId = {};
      for (const a of setting.algorithms) setting._byId[a.id] = a;
    }
    return setting._byId;
  }
  function algo(id) { return byId(S)[id]; }
  function envOf(id) { return S.environments.find((e) => e.id === id); }
  function defaults(a) { const o = {}; for (const p of a.params || []) o[p.key] = p.default; return o; }
  function presetsOf(setting) {
    const list = setting.id === 'kab' ? KAB_PRESETS : (setting.presets || []).concat(EXTRA_PRESETS[setting.id] || []);
    const ids = byId(setting);
    return list
      .map((p) => ({ ...p, algos: p.algos.filter((id) => ids[id]).slice(0, SLOTS) }))
      .filter((p) => p.algos.length && (!p.env || setting.environments.some((e) => e.id === p.env)));
  }
  function viewFor(setting) {
    if (views[setting.id]) return views[setting.id];
    const c = setting.controls || {};
    const v = {
      env: setting.defaultEnv && setting.environments.some((e) => e.id === setting.defaultEnv) ? setting.defaultEnv : setting.environments[0].id,
      K: c.K ? c.K.default : 10,
      T: c.T ? c.T.default : 1000,
      runs: c.runs ? c.runs.default : 100,
      seed: 7,
      extra: {},
      selected: [], slot: {}, params: {},
      metric: setting.metrics[0].key, logx: false,
      sortKey: setting.final.key, sortDir: setting.final.better === 'higher' ? -1 : 1,
      filter: '',
      results: null, cfg: null,
    };
    for (const x of c.extra || []) v.extra[x.key] = x.default;
    for (const a of setting.algorithms) v.params[a.id] = defaults(a);
    const ids = byId(setting);
    let lineup = (setting.id === 'kab' ? KAB_LINEUP : setting.defaultLineup || []).filter((id) => ids[id]);
    if (!lineup.length) lineup = setting.algorithms.slice(0, Math.min(6, setting.algorithms.length)).map((a) => a.id);
    setLineup(v, lineup);
    views[setting.id] = v;
    return v;
  }
  function setLineup(v, ids) {
    v.selected = []; v.slot = {};
    ids.slice(0, SLOTS).forEach((id, i) => { v.selected.push(id); v.slot[id] = i; });
  }
  function toggleAlgo(id, on) {
    if (on && !V.selected.includes(id)) {
      if (V.selected.length >= SLOTS) return false;
      const used = new Set(Object.values(V.slot));
      let s = 0; while (used.has(s)) s++;
      V.slot[id] = s;
      V.selected.push(id);
    } else if (!on) {
      V.selected = V.selected.filter((x) => x !== id);
      delete V.slot[id];
    }
    return true;
  }
  function color(id) { return `var(--s${(V.slot[id] ?? 0) + 1})`; }
  function sorter(better) { return better === 'higher' ? (a, b) => b - a : (a, b) => a - b; }
  function sumVal(r, key) { return r.summary ? r.summary[key] : r[key]; }
  function finalOf(r) { return sumVal(r, S.final.key); }
  function finalSE(r) { return S.final.se ? sumVal(r, S.final.se) || 0 : 0; }
  function envTags(e) {
    const t = (e.tags || []).slice();
    if (!e.tags) {
      if (e.stationary === false) t.push('changes');
      if (e.contextual) t.push('context');
      if (e.rested) t.push('rested');
    }
    return t;
  }

  // ---------------------------------------------------------- controls ----
  function buildSettingsNav() {
    const nav = $('#settings-nav');
    nav.innerHTML = '';
    for (const s of SETTINGS) {
      const b = document.createElement('button');
      b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.setting = s.id;
      const name = document.createElement('span'); name.textContent = s.short || s.name;
      const count = document.createElement('span'); count.className = 'count'; count.textContent = s.algorithms.length;
      b.append(name, count);
      b.title = s.blurb || s.name;
      b.addEventListener('click', () => switchSetting(s.id, true));
      nav.appendChild(b);
    }
  }

  function switchSetting(id, pushHash) {
    const next = SETTINGS.find((s) => s.id === id) || SETTINGS[0];
    if (S && S.id === next.id && V) return;
    if (S) job++; // cancel any run of the previous setting
    S = next; V = viewFor(S);
    if (pushHash) { try { history.replaceState(null, '', '#' + S.id); } catch (e) { location.hash = S.id; } }
    for (const b of $('#settings-nav').children) b.setAttribute('aria-selected', String(b.dataset.setting === S.id));
    $('#setting-blurb').textContent = S.blurb || '';
    $('#catalog-title').textContent = `Every algorithm in ${S.name}`;
    buildControls(); buildEnvs(); buildPresets(); buildMetricTabs(); buildAlgoList(); buildCatalog();
    $('#heat-card').hidden = !S.heatmap;
    $('#split').classList.toggle('single', !S.heatmap);
    $('#board-title').textContent = S.final.label;
    $('#board-sub').textContent = `${S.final.better === 'higher' ? 'Higher' : 'Lower'} is better.${S.final.se ? ' Whiskers show a 95% confidence interval of the mean.' : ''}`;
    if (V.results) renderAll();
    else { $('#tiles').innerHTML = ''; ['#curve-chart', '#board-chart', '#heat-chart'].forEach((q) => { $(q).innerHTML = ''; }); $('#legend').innerHTML = ''; $('#results-table').innerHTML = ''; }
    scheduleRun(0);
  }

  function buildControls() {
    const c = S.controls || {};
    const kField = $('#k-field');
    kField.hidden = !c.K;
    if (c.K) {
      const k = $('#k-range');
      $('#k-label').textContent = c.K.label || 'Arms (K)';
      k.min = c.K.min; k.max = c.K.max; k.step = c.K.step || 1; k.value = V.K;
      $('#k-out').textContent = V.K;
    }
    const tField = $('#t-field');
    tField.hidden = !c.T;
    if (c.T) {
      $('#t-label').textContent = c.T.label || 'Horizon (T steps)';
      $('#t-select').innerHTML = c.T.options.map((o) => `<option value="${o}" ${o === V.T ? 'selected' : ''}>${o.toLocaleString('en-US')}</option>`).join('');
    }
    const runsOpts = (c.runs && c.runs.options) || [20, 50, 100, 200];
    $('#runs-label').textContent = (c.runs && c.runs.label) || 'Runs averaged';
    $('#runs-select').innerHTML = runsOpts.map((o) => `<option value="${o}" ${o === V.runs ? 'selected' : ''}>${o.toLocaleString('en-US')}</option>`).join('');
    $('#seed-input').value = V.seed;
    const ex = c.extra || [];
    $('#extra-controls').innerHTML = ex.map((x) => `
      <div class="field">
        <label for="x-${esc(x.key)}">${esc(x.label)}</label>
        <input type="number" id="x-${esc(x.key)}" data-key="${esc(x.key)}" value="${V.extra[x.key]}" min="${x.min}" max="${x.max}" step="${x.step || 'any'}">
      </div>`).join('');
    $('#extra-controls').hidden = !ex.length;
  }

  function buildEnvs() {
    const sel = $('#env-select');
    sel.innerHTML = S.environments.map((e) => `<option value="${esc(e.id)}" ${e.id === V.env ? 'selected' : ''}>${esc(e.name)}</option>`).join('');
    describeEnv();
  }
  function describeEnv() {
    const e = envOf(V.env);
    $('#env-desc').textContent = e ? e.description || '' : '';
    $('#env-tags').innerHTML = e ? envTags(e).map((t) => `<span class="tag">${esc(t)}</span>`).join('') : '';
  }

  function buildPresets() {
    const box = $('#presets');
    box.innerHTML = '';
    for (const p of presetsOf(S)) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip'; b.textContent = p.label;
      if (p.env) b.title = `Also switches to the ${envOf(p.env).name.toLowerCase()} environment`;
      b.addEventListener('click', () => {
        setLineup(V, p.algos);
        if (p.env) { V.env = p.env; $('#env-select').value = p.env; describeEnv(); }
        buildAlgoList(); buildCatalog(); scheduleRun(0);
      });
      box.appendChild(b);
    }
    box.hidden = !box.children.length;
  }

  function buildMetricTabs() {
    if (!S.metrics.some((m) => m.key === V.metric)) V.metric = S.metrics[0].key;
    $('#metric-tabs').innerHTML = S.metrics.map((m) => `<button role="tab" type="button" data-metric="${esc(m.key)}" aria-selected="${m.key === V.metric}">${esc(m.label)}</button>`).join('');
    $('#logx').checked = V.logx;
  }

  function buildAlgoList() {
    const active = document.activeElement && document.activeElement.id;
    const full = V.selected.length >= SLOTS;
    $('#algo-filter').value = V.filter;
    const q = V.filter.trim().toLowerCase();
    const match = (a) => !q || a.name.toLowerCase().includes(q) || a.id.includes(q) || V.selected.includes(a.id);
    const html = S.families.map((f) => {
      const algos = S.algorithms.filter((a) => a.family === f.id && (match(a) || f.name.toLowerCase().includes(q)));
      if (!algos.length) return '';
      return `<div class="algo-group"><div class="eyebrow">${esc(f.name)}</div>${algos.map((a) => {
        const on = V.selected.includes(a.id);
        const dis = !on && full;
        const params = on && (a.params || []).length ? `<div class="params">${a.params.map((p) => `
            <label for="p-${esc(a.id)}-${esc(p.key)}">${esc(p.label)}
              <input type="number" id="p-${esc(a.id)}-${esc(p.key)}" data-algo="${esc(a.id)}" data-key="${esc(p.key)}"
                value="${V.params[a.id][p.key]}" min="${p.min}" max="${p.max}" step="${p.step || 'any'}">
            </label>`).join('')}</div>` : '';
        return `<div class="algo${dis ? ' disabled' : ''}">
          <label for="a-${esc(a.id)}">
            <input type="checkbox" id="a-${esc(a.id)}" data-algo="${esc(a.id)}" ${on ? 'checked' : ''} ${dis ? 'disabled' : ''}>
            <span class="sw" style="background:${on ? color(a.id) : 'transparent'}"></span>
            <span>${esc(a.name)}</span>
          </label>${params}</div>`;
      }).join('')}</div>`;
    }).join('');
    $('#algo-list').innerHTML = html;
    $('#algo-empty').hidden = html.trim() !== '';
    $('#algo-count').textContent = `${V.selected.length} of ${SLOTS}`;
    if (active) { const el = document.getElementById(active); if (el) el.focus(); }
  }

  function bindControls() {
    $('#algo-list').addEventListener('change', (ev) => {
      const t = ev.target;
      if (t.type === 'checkbox') {
        if (!t.checked && V.selected.length <= 1) { t.checked = true; return; }
        toggleAlgo(t.dataset.algo, t.checked);
        buildAlgoList(); buildCatalog(); scheduleRun();
      } else if (t.type === 'number') {
        const p = algo(t.dataset.algo).params.find((x) => x.key === t.dataset.key);
        let v = parseFloat(t.value);
        if (!Number.isFinite(v)) v = p.default;
        v = Math.min(p.max, Math.max(p.min, v));
        t.value = v;
        V.params[t.dataset.algo][t.dataset.key] = v;
        scheduleRun();
      }
    });
    $('#algo-filter').addEventListener('input', (e) => { V.filter = e.target.value; buildAlgoList(); });
    $('#env-select').addEventListener('change', (e) => { V.env = e.target.value; describeEnv(); scheduleRun(); });
    const k = $('#k-range');
    k.addEventListener('input', () => { V.K = +k.value; $('#k-out').textContent = k.value; scheduleRun(); });
    $('#t-select').addEventListener('change', (e) => { V.T = +e.target.value; scheduleRun(); });
    $('#runs-select').addEventListener('change', (e) => { V.runs = +e.target.value; scheduleRun(); });
    $('#extra-controls').addEventListener('change', (e) => {
      const t = e.target; if (!t.dataset.key) return;
      const x = (S.controls.extra || []).find((y) => y.key === t.dataset.key);
      let v = parseFloat(t.value);
      if (!Number.isFinite(v)) v = x.default;
      v = Math.min(x.max, Math.max(x.min, v));
      t.value = v; V.extra[x.key] = v; scheduleRun();
    });
    const seed = $('#seed-input');
    seed.addEventListener('change', () => { V.seed = Math.max(0, parseInt(seed.value, 10) || 0); seed.value = V.seed; scheduleRun(); });
    $('#seed-roll').addEventListener('click', () => { V.seed = Math.floor(Math.random() * 100000); seed.value = V.seed; scheduleRun(0); });
    $('#run-btn').addEventListener('click', () => scheduleRun(0));
    $('#metric-tabs').addEventListener('click', (ev) => {
      const b = ev.target.closest('button[data-metric]'); if (!b) return;
      V.metric = b.dataset.metric;
      for (const x of $('#metric-tabs').children) x.setAttribute('aria-selected', String(x === b));
      renderCurves();
    });
    $('#logx').addEventListener('change', (e) => { V.logx = e.target.checked; renderCurves(); });
    window.addEventListener('hashchange', () => {
      const id = location.hash.replace('#', '');
      if (SETTINGS.some((s) => s.id === id)) switchSetting(id, false);
    });
  }

  // ------------------------------------------------------------ runner ----
  let runTimer = null, job = 0;
  function scheduleRun(delay = 250) {
    clearTimeout(runTimer);
    runTimer = setTimeout(startRun, delay);
  }
  function startRun() {
    const my = ++job, setting = S, view = V;
    const c = setting.controls || {};
    const cfg = {
      env: view.env, K: c.K ? view.K : undefined, T: c.T ? view.T : undefined, runs: view.runs, seed: view.seed,
      extra: Object.assign({}, view.extra),
      algorithms: view.selected.map((id) => ({ id, params: Object.assign({}, view.params[id]) })),
    };
    let runner;
    try { runner = setting.createRunner(cfg); } catch (err) { setStatus('error', `Could not start: ${err.message}`, 0); return; }
    const t0 = performance.now();
    let lastDraw = 0, firstDraw = true;
    document.querySelectorAll('.chart').forEach((el) => el.classList.add('stale'));
    setStatus('running', `Running 0 / ${cfg.runs}`, 0);
    function tick() {
      if (my !== job) return;
      const start = performance.now();
      try {
        do { runner.step(1); } while (!runner.finished && performance.now() - start < 30);
      } catch (err) {
        setStatus('error', `Stopped: ${err.message}`, runner.done / cfg.runs);
        console.error(err);
        return;
      }
      const now = performance.now();
      if (runner.finished || now - lastDraw > 300 || firstDraw) {
        view.results = runner.results();
        view.cfg = cfg;
        if (firstDraw) document.querySelectorAll('.chart').forEach((el) => el.classList.remove('stale'));
        firstDraw = false;
        renderAll();
        lastDraw = performance.now();
      }
      if (runner.finished) {
        const secs = ((performance.now() - t0) / 1000).toFixed(1);
        setStatus('done', `${cfg.runs} runs · ${secs} s`, 1);
      } else {
        setStatus('running', `Running ${runner.done} / ${cfg.runs}`, runner.done / cfg.runs);
        setTimeout(tick, 0);
      }
    }
    setTimeout(tick, 0);
  }
  function setStatus(s, text, frac) {
    $('#status-pill').dataset.state = s;
    $('#status-text').textContent = text;
    $('#progress-bar').style.width = (frac * 100).toFixed(1) + '%';
  }

  // ------------------------------------------------------------ render ----
  function renderAll() {
    if (!V.results || !V.results.length) return;
    renderTiles(); renderCurves(); renderBoard(); if (S.heatmap) renderHeat(); renderTable();
  }

  function renderTiles() {
    const R = V.results, cfg = V.cfg, env = envOf(cfg.env), F = S.final;
    const ord = R.slice().sort((a, b) => sorter(F.better)(finalOf(a), finalOf(b)));
    const best = ord[0], worst = ord[ord.length - 1];
    const cheap = R.slice().sort((a, b) => (sumVal(a, 'msPerRun') || 0) - (sumVal(b, 'msPerRun') || 0))[0];
    const tiles = [{ label: F.better === 'higher' ? `Highest ${F.label.toLowerCase()}` : `Lowest ${F.label.toLowerCase()}`, id: best.id, sub: `${fmtNum(finalOf(best))}${F.se ? ' ± ' + fmtNum(1.96 * finalSE(best)) : ''}` }];
    for (const t of (S.tiles || []).slice(0, 1)) {
      const top = R.slice().sort((a, b) => sorter(t.better)(sumVal(a, t.key), sumVal(b, t.key)))[0];
      tiles.push({ label: t.label, id: top.id, sub: `${fmtBy(t.fmt, sumVal(top, t.key))} ${t.sub || ''}`.trim() });
    }
    if (R.length > 1) {
      const a = finalOf(best), b = finalOf(worst), ratio = a > 0 && b > 0 ? (F.better === 'higher' ? a / b : b / a) : 0;
      tiles.push({ label: F.better === 'higher' ? `Lowest ${F.label.toLowerCase()}` : `Highest ${F.label.toLowerCase()}`, id: worst.id, sub: `${fmtNum(b)}${ratio > 1 ? `, ${ratio.toFixed(1)}× the leader` : ''}` });
    }
    tiles.push({ label: 'Cheapest to run', id: cheap.id, sub: `${(sumVal(cheap, 'msPerRun') || 0).toFixed(2)} ms per run` });
    $('#tiles').innerHTML = tiles.map((t) => `
      <div class="tile">
        <div class="label">${esc(t.label)}</div>
        <div class="value"><span class="key" style="background:${color(t.id)}"></span><span class="txt">${esc(algo(t.id).name)}</span></div>
        <div class="sub num">${esc(t.sub)}</div>
      </div>`).join('');
    const parts = [env ? env.name : cfg.env];
    if (cfg.K != null) parts.push(`${(S.controls.K && S.controls.K.label) ? S.controls.K.label.replace(/\s*\(.*\)/, '') : 'K'} = ${cfg.K}`);
    for (const x of S.controls.extra || []) parts.push(`${x.label.replace(/\s*\(.*\)/, '')} = ${cfg.extra[x.key]}`);
    parts.push(`${R[0].runs} of ${cfg.runs} runs`, `seed ${cfg.seed}`);
    $('#curve-sub').textContent = parts.join(' · ');
  }

  // ---- curves ----
  function smooth(arr, w) {
    if (w <= 1) return arr;
    const n = arr.length, pre = new Float64Array(n + 1), out = new Float64Array(n);
    for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + arr[i];
    const h = Math.floor(w / 2);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - h), hi = Math.min(n, i + h + 1);
      out[i] = (pre[hi] - pre[lo]) / (hi - lo);
    }
    return out;
  }
  function sampleIdx(xs, n, log) {
    const L = xs.length;
    if (L <= n) return Array.from({ length: L }, (_, i) => i);
    const out = [];
    if (!log) {
      for (let i = 0; i < n; i++) out.push(Math.round((i * (L - 1)) / (n - 1)));
    } else {
      const a = Math.log(xs[0]), b = Math.log(xs[L - 1]);
      let j = 0;
      for (let i = 0; i < n; i++) {
        const target = Math.exp(a + ((b - a) * i) / (n - 1));
        while (j < L - 1 && xs[j] < target) j++;
        out.push(j);
      }
    }
    return [...new Set(out)];
  }

  let curveGeo = null;
  function renderCurves() {
    const el = $('#curve-chart');
    if (!V.results) return;
    const R = V.results, mdef = S.metrics.find((m) => m.key === V.metric) || S.metrics[0];
    const L = R[0].curves[mdef.key] ? R[0].curves[mdef.key].length : 0;
    if (!L) { el.innerHTML = ''; return; }
    const xs = R[0].x ? Array.from(R[0].x) : Array.from({ length: L }, (_, i) => i + 1);
    const log = V.logx && xs[0] > 0;
    const W = Math.max(280, el.clientWidth);
    const H = Math.round(Math.min(440, Math.max(260, W * 0.46)));
    const smoothW = mdef.smooth ? Math.max(1, Math.round(L / 100)) : 1;
    const series = R.map((r) => ({
      id: r.id, name: r.name,
      v: smooth(r.curves[mdef.key], smoothW),
      se: mdef.band && r.curvesSE ? r.curvesSE[mdef.key] : null,
    }));
    const fmtM = (v) => fmtBy(mdef.fmt, v);
    const endVal = (s) => s.v[L - 1];
    const winner = series.slice().sort((a, b) => sorter(mdef.better)(endVal(a), endVal(b)))[0];
    const winnerLabel = `${winner.name} ${fmtM(endVal(winner))}`;
    const showLabel = W >= 560;
    const m = { l: 56, r: showLabel ? Math.ceil(textWidth(winnerLabel, '600 ' + LABEL_FONT)) + 22 : 14, t: 10, b: 34 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const idx = sampleIdx(xs, Math.max(60, Math.round(pw / 2)), log);

    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const i of idx) {
      const v = s.v[i], e = s.se ? 1.96 * s.se[i] : 0;
      if (!Number.isFinite(v)) continue;
      if (v + e > hi) hi = v + e;
      if (v - e < lo) lo = v - e;
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (mdef.zeroToOne) { lo = 0; hi = 1; }
    else if (lo >= 0 && mdef.better === 'lower') lo = 0;
    if (hi - lo < 1e-12) hi = lo + 1;
    const yt = ticks(lo, hi, Math.max(3, Math.round(ph / 60)));
    const step = yt.length > 1 ? yt[1] - yt[0] : 1;
    if (!mdef.zeroToOne) {
      if (yt[yt.length - 1] < hi - 1e-12) yt.push(+(yt[yt.length - 1] + step).toFixed(10));
      if (yt[0] > lo + 1e-12) yt.unshift(+(yt[0] - step).toFixed(10));
    }
    const y0 = yt[0], y1 = yt[yt.length - 1];
    const Y = (v) => m.t + ph - ((Math.min(y1, Math.max(y0, v)) - y0) / (y1 - y0)) * ph;
    const xa = xs[0], xb = xs[L - 1];
    const X = log
      ? (x) => m.l + ((Math.log(x) - Math.log(xa)) / Math.max(1e-12, Math.log(xb) - Math.log(xa))) * pw
      : (x) => m.l + ((x - xa) / Math.max(1e-12, xb - xa)) * pw;
    let xt;
    if (log) {
      xt = [];
      for (let d = Math.pow(10, Math.ceil(Math.log10(xa))); d <= xb; d *= 10) xt.push(d);
      if (!xt.length || X(xb) - X(xt[xt.length - 1]) > 40) xt.push(xb);
    } else {
      xt = ticks(xa, xb, Math.max(2, Math.round(pw / 110))).filter((x) => x >= xa && x <= xb);
      if (!xt.length) xt = [xa, xb];
    }
    const yLab = (v) => (mdef.fmt === 'pct' ? Math.round(v * 100) + '%' : fmtTick(v));

    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(mdef.label)} for ${R.length} algorithms">`;
    svg += '<g class="grid">' + yt.map((v) => `<line x1="${m.l}" x2="${m.l + pw}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/>`).join('') + '</g>';
    const base = y0 <= 0 && y1 >= 0 ? Y(0) : m.t + ph;
    svg += `<line class="baseline" x1="${m.l}" x2="${m.l + pw}" y1="${base.toFixed(1)}" y2="${base.toFixed(1)}"/>`;
    svg += '<g class="axis">' + yt.map((v) => `<text x="${m.l - 8}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${esc(yLab(v))}</text>`).join('');
    svg += xt.map((x) => `<text x="${X(x).toFixed(1)}" y="${m.t + ph + 18}" text-anchor="middle">${esc(fmtTick(x))}</text>`).join('');
    svg += `<text x="${m.l + pw}" y="${H - 2}" text-anchor="end">${esc(S.xLabel || 'step')}${log ? ' (log scale)' : ''}</text></g>`;

    for (const s of series) {
      if (!s.se) continue;
      const up = idx.map((i) => `${X(xs[i]).toFixed(1)},${Y(s.v[i] + 1.96 * s.se[i]).toFixed(1)}`);
      const dn = idx.slice().reverse().map((i) => `${X(xs[i]).toFixed(1)},${Y(s.v[i] - 1.96 * s.se[i]).toFixed(1)}`);
      svg += `<polygon class="series" data-id="${esc(s.id)}" points="${up.join(' ')} ${dn.join(' ')}" style="fill:${color(s.id)};fill-opacity:0.10;stroke:none"/>`;
    }
    const order = series.slice().sort((a, b) => (a.id === winner.id) - (b.id === winner.id));
    for (const s of order) {
      const pts = idx.filter((i) => Number.isFinite(s.v[i])).map((i) => `${X(xs[i]).toFixed(1)},${Y(s.v[i]).toFixed(1)}`).join(' ');
      svg += `<polyline class="series" data-id="${esc(s.id)}" points="${pts}" style="fill:none;stroke:${color(s.id)};stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>`;
    }
    for (const s of order) {
      if (!Number.isFinite(endVal(s))) continue;
      svg += `<circle class="series" data-id="${esc(s.id)}" cx="${X(xb).toFixed(1)}" cy="${Y(endVal(s)).toFixed(1)}" r="4" style="fill:${color(s.id)};stroke:var(--surface);stroke-width:2"/>`;
    }
    if (showLabel && Number.isFinite(endVal(winner))) {
      svg += `<text class="lbl" x="${m.l + pw + 10}" y="${(Y(endVal(winner)) + 4).toFixed(1)}">${esc(winnerLabel)}</text>`;
    }
    svg += '<g id="curve-hover"></g>';
    svg += `<rect id="curve-hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="transparent" tabindex="0" aria-label="Chart readout. Use left and right arrow keys to move along the x-axis."/>`;
    svg += '</svg>';
    el.innerHTML = svg + '<div class="tooltip" hidden></div>';
    curveGeo = { X, Y, idx, xs, series, m, ph, W, fmtM, better: mdef.better };

    const leg = $('#legend');
    leg.innerHTML = '';
    for (const s of series) {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.id = s.id;
      const k = document.createElement('span'); k.className = 'key'; k.style.background = color(s.id);
      b.append(k, document.createTextNode(s.name));
      leg.appendChild(b);
    }
    const svgEl = el.querySelector('svg');
    const setFocus = (id) => {
      svgEl.classList.toggle('focusing', !!id);
      svgEl.querySelectorAll('.series').forEach((n) => n.classList.toggle('on', n.dataset.id === id));
      leg.querySelectorAll('button').forEach((n) => n.classList.toggle('on', n.dataset.id === id));
    };
    leg.onpointerover = (e) => { const b = e.target.closest('button'); setFocus(b ? b.dataset.id : null); };
    leg.onpointerleave = () => setFocus(null);
    leg.onfocusin = (e) => { const b = e.target.closest('button'); setFocus(b ? b.dataset.id : null); };
    leg.onfocusout = () => setFocus(null);

    let note = mdef.note || '';
    if (smoothW > 1) note += `${note ? ', ' : ''}smoothed over ${smoothW} points.`;
    else if (note && !/[.]$/.test(note)) note += '.';
    $('#curve-note').textContent = note;

    const hit = el.querySelector('#curve-hit');
    let cur = idx.length - 1;
    const show = (i) => { cur = Math.max(0, Math.min(idx.length - 1, i)); drawHover(el, cur); };
    hit.addEventListener('pointermove', (e) => {
      const rect = svgEl.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      let a = 0, b = idx.length - 1;
      while (b - a > 1) { const mid = (a + b) >> 1; if (X(xs[idx[mid]]) < px) a = mid; else b = mid; }
      show(Math.abs(X(xs[idx[a]]) - px) <= Math.abs(X(xs[idx[b]]) - px) ? a : b);
    });
    hit.addEventListener('pointerleave', () => clearHover(el));
    hit.addEventListener('focus', () => show(cur));
    hit.addEventListener('blur', () => clearHover(el));
    hit.addEventListener('keydown', (e) => {
      const jump = e.shiftKey ? 10 : 1;
      if (e.key === 'ArrowLeft') { show(cur - jump); e.preventDefault(); }
      if (e.key === 'ArrowRight') { show(cur + jump); e.preventDefault(); }
    });
  }
  function clearHover(el) {
    const g = el.querySelector('#curve-hover'); if (g) g.innerHTML = '';
    const tip = el.querySelector('.tooltip'); if (tip) tip.hidden = true;
  }
  function drawHover(el, k) {
    const { X, Y, idx, xs, series, m, ph, W, fmtM, better } = curveGeo;
    const i = idx[k], x = X(xs[i]);
    let g = `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ph}" style="stroke:var(--ink-2);stroke-width:1"/>`;
    for (const s of series) if (Number.isFinite(s.v[i])) g += `<circle cx="${x}" cy="${Y(s.v[i])}" r="4" style="fill:${color(s.id)};stroke:var(--surface);stroke-width:2"/>`;
    el.querySelector('#curve-hover').innerHTML = g;
    const tip = el.querySelector('.tooltip');
    tip.innerHTML = '';
    const head = document.createElement('div'); head.className = 't-head';
    head.textContent = `${S.xLabel || 'x'} = ${fmtTick(xs[i])}`;
    tip.appendChild(head);
    const rows = series.map((s) => ({ s, v: s.v[i] })).sort((a, b) => sorter(better)(a.v, b.v));
    for (const r of rows) {
      const row = document.createElement('div'); row.className = 't-row';
      const kk = document.createElement('span'); kk.className = 'key'; kk.style.background = color(r.s.id);
      const v = document.createElement('b'); v.textContent = fmtM(r.v);
      const n = document.createElement('span'); n.textContent = r.s.name;
      row.append(kk, v, n); tip.appendChild(row);
    }
    tip.hidden = false;
    const scale = el.querySelector('svg').getBoundingClientRect().width / W;
    const tw = tip.offsetWidth;
    let left = x * scale + 14;
    if (left + tw > el.clientWidth) left = x * scale - tw - 14;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = (m.t * scale + 4) + 'px';
  }

  // ---- leaderboard ----
  function barPath(xBase, xEnd, y, h, r) {
    const w = Math.abs(xEnd - xBase);
    if (w < 0.5) return '';
    r = Math.min(r, w, h / 2);
    if (xEnd >= xBase) return `M${xBase},${y}H${xEnd - r}Q${xEnd},${y} ${xEnd},${y + r}V${y + h - r}Q${xEnd},${y + h} ${xEnd - r},${y + h}H${xBase}Z`;
    return `M${xBase},${y}H${xEnd + r}Q${xEnd},${y} ${xEnd},${y + r}V${y + h - r}Q${xEnd},${y + h} ${xEnd + r},${y + h}H${xBase}Z`;
  }
  function renderBoard() {
    const el = $('#board-chart'), F = S.final;
    const R = V.results.slice().sort((a, b) => sorter(F.better)(finalOf(a), finalOf(b)));
    const W = Math.max(260, el.clientWidth);
    const rowH = 42, barH = 12, top = 4;
    const H = top + R.length * rowH + 22;
    const valW = Math.max(...R.map((r) => textWidth(fmtNum(finalOf(r)), '500 11px "IBM Plex Mono", monospace'))) + 10;
    let lo = 0, hi = 0;
    for (const r of R) {
      const v = finalOf(r), ci = 1.96 * finalSE(r);
      if (!Number.isFinite(v)) continue;
      hi = Math.max(hi, v + ci); lo = Math.min(lo, v - ci);
    }
    if (hi - lo < 1e-12) hi = lo + 1;
    const xt = ticks(lo, hi, Math.max(2, Math.round((W - valW) / 90)));
    const dlo = Math.min(lo, xt[0]), dhi = Math.max(hi, xt[xt.length - 1]);
    const padL = lo < 0 ? valW : 2, x1 = W - valW - 4;
    const X = (v) => padL + ((v - dlo) / (dhi - dlo)) * (x1 - padL);
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(F.label)} by algorithm">`;
    svg += '<g class="grid">' + xt.map((v) => `<line x1="${X(v).toFixed(1)}" x2="${X(v).toFixed(1)}" y1="${top}" y2="${H - 20}"/>`).join('') + '</g>';
    svg += '<g class="axis">' + xt.map((v, i) => `<text x="${X(v).toFixed(1)}" y="${H - 4}" text-anchor="${i === 0 ? 'start' : 'middle'}">${esc(fmtTick(v))}</text>`).join('') + '</g>';
    R.forEach((r, i) => {
      const y = top + i * rowH, by = y + 20, v = finalOf(r), ci = 1.96 * finalSE(r);
      svg += `<g class="bar-row" data-id="${esc(r.id)}" tabindex="0" role="img" aria-label="${esc(r.name)}: ${fmtNum(v)}">`;
      svg += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
      svg += `<text class="lbl-2" x="${padL}" y="${y + 14}">${esc(r.name)}</text>`;
      if (Number.isFinite(v)) {
        svg += `<path d="${barPath(X(0), X(v), by, barH, 4)}" style="fill:${color(r.id)}"/>`;
        const a = X(v - ci), b = X(v + ci), cy = by + barH / 2;
        if (b - a > 2) svg += `<path d="M${a},${cy}H${b}M${a},${cy - 4}V${cy + 4}M${b},${cy - 4}V${cy + 4}" style="stroke:var(--ink-2);stroke-width:1.25;fill:none"/>`;
        const tx = v >= 0 ? Math.max(b, X(v)) + 6 : Math.min(a, X(v)) - 6;
        svg += `<text x="${tx}" y="${cy + 4}" text-anchor="${v >= 0 ? 'start' : 'end'}" style="fill:var(--ink);font-weight:500">${esc(fmtNum(v))}</text>`;
      }
      svg += '</g>';
    });
    svg += `<line class="baseline" x1="${X(0)}" x2="${X(0)}" y1="${top}" y2="${H - 20}"/>`;
    svg += '</svg>';
    el.innerHTML = svg + '<div class="tooltip" hidden></div>';
    const tip = el.querySelector('.tooltip');
    const showTip = (g) => {
      const r = V.results.find((x) => x.id === g.dataset.id);
      tip.innerHTML = '';
      const h = document.createElement('div'); h.className = 't-head'; h.textContent = r.name;
      tip.appendChild(h);
      const rows = [[F.label, fmtNum(finalOf(r))]];
      if (F.se) rows.push(['95% CI', '± ' + fmtNum(1.96 * finalSE(r))]);
      if (F.median) rows.push(['Median', fmtNum(sumVal(r, F.median))]);
      if (r.finals && r.finals.length) rows.push(['Best run', fmtNum(r.finals[0])], ['Worst run', fmtNum(r.finals[r.finals.length - 1])]);
      for (const [k, v] of rows) {
        const row = document.createElement('div'); row.className = 't-row';
        const sp = document.createElement('span'); sp.className = 'key';
        const b = document.createElement('b'); b.textContent = v;
        const n = document.createElement('span'); n.textContent = k;
        row.append(sp, b, n); tip.appendChild(row);
      }
      tip.hidden = false;
      const box = g.getBoundingClientRect(), host = el.getBoundingClientRect();
      tip.style.left = Math.max(0, Math.min(el.clientWidth - tip.offsetWidth, box.width * 0.45)) + 'px';
      tip.style.top = (box.bottom - host.top + 2) + 'px';
    };
    el.querySelectorAll('.bar-row').forEach((g) => {
      g.addEventListener('pointerenter', () => showTip(g));
      g.addEventListener('focus', () => showTip(g));
      g.addEventListener('pointerleave', () => { tip.hidden = true; });
      g.addEventListener('blur', () => { tip.hidden = true; });
    });
  }

  // ---- heatmap ----
  const BINS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8];
  function bin(v) { let i = 0; while (i < BINS.length && v >= BINS[i]) i++; return i; }
  function renderHeat() {
    const el = $('#heat-chart');
    const R = V.results.slice().filter((r) => r.pullShare).sort((a, b) => sorter(S.final.better)(finalOf(a), finalOf(b)));
    if (!R.length) { el.innerHTML = ''; return; }
    const K = R[0].pullShare.length;
    const W = Math.max(260, el.clientWidth);
    const labW = Math.min(W * 0.42, Math.max(...R.map((r) => textWidth(r.name, LABEL_FONT))) + 12);
    const gx = labW, gw = W - labW;
    const gap = gw / K > 10 ? 2 : 1;
    const cw = gw / K;
    const rowH = 26, top = 22;
    const H = top + R.length * rowH + 4;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Share of pulls per arm, arms ranked best to worst">`;
    const every = Math.ceil(28 / cw);
    for (let k = 0; k < K; k++) {
      if (k === 0 || k === K - 1 || (k % every === 0 && K - 1 - k >= every)) {
        const label = k === 0 ? 'best' : k === K - 1 ? 'worst' : String(k + 1);
        const anchor = k === 0 ? 'start' : k === K - 1 ? 'end' : 'middle';
        const x = k === 0 ? gx : k === K - 1 ? gx + gw : gx + (k + 0.5) * cw;
        svg += `<text x="${x.toFixed(1)}" y="12" text-anchor="${anchor}">${label}</text>`;
      }
    }
    R.forEach((r, i) => {
      const y = top + i * rowH;
      svg += `<text class="lbl-2" x="0" y="${y + rowH / 2 + 3}">${esc(r.name)}</text>`;
      for (let k = 0; k < K; k++) {
        const v = r.pullShare[k] || 0;
        svg += `<rect class="cell" data-i="${i}" data-k="${k}" x="${(gx + k * cw + gap / 2).toFixed(2)}" y="${y + 2}" width="${Math.max(1, cw - gap).toFixed(2)}" height="${rowH - 4}" rx="2" style="fill:var(--q${bin(v)})"/>`;
      }
    });
    svg += '</svg>';
    el.innerHTML = svg + '<div class="tooltip" hidden></div>';
    const tip = el.querySelector('.tooltip');
    const svgEl = el.querySelector('svg');
    svgEl.addEventListener('pointermove', (e) => {
      const c = e.target.closest('.cell');
      if (!c) { tip.hidden = true; return; }
      const r = R[+c.dataset.i], k = +c.dataset.k;
      tip.innerHTML = '';
      const h = document.createElement('div'); h.className = 't-head';
      h.textContent = `Arm ranked ${k + 1} of ${K}`;
      const row = document.createElement('div'); row.className = 't-row';
      const sp = document.createElement('span'); sp.className = 'key'; sp.style.background = color(r.id);
      const b = document.createElement('b'); b.textContent = fmtPct(r.pullShare[k]);
      const n = document.createElement('span'); n.textContent = r.name;
      row.append(sp, b, n); tip.append(h, row);
      tip.hidden = false;
      const host = el.getBoundingClientRect();
      let left = e.clientX - host.left + 12;
      if (left + tip.offsetWidth > el.clientWidth) left = e.clientX - host.left - tip.offsetWidth - 12;
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = (e.clientY - host.top + 14) + 'px';
    });
    svgEl.addEventListener('pointerleave', () => { tip.hidden = true; });
    const labels = ['under 1%', '1%', '2%', '5%', '10%', '20%', '40%', '60%', '80%+'];
    $('#heat-scale').innerHTML = `<span>Share of pulls</span><span class="bins">${labels.map((l, i) => `<span title="${l}" style="background:var(--q${i})"></span>`).join('')}</span><span>under 1% → 80%+</span>`;
  }

  // ---- table ----
  function tableCols() {
    return [
      { key: '@name', label: 'Algorithm', get: (r) => r.name },
      { key: '@family', label: 'Family', get: (r) => (S.families.find((f) => f.id === r.family) || { name: r.family }).name },
      { key: '@params', label: 'Settings', get: (r) => Object.entries(r.params || {}).map(([k, v]) => `${k}=${v}`).join(' ') || '—', sortable: false },
    ].concat(S.columns.map((c) => ({ key: c.key, label: c.label, n: true, fmt: c.fmt, better: c.better, get: (r) => sumVal(r, c.key) })));
  }
  function renderTable() {
    const tbl = $('#results-table'), COLS = tableCols();
    const col = COLS.find((c) => c.key === V.sortKey) || COLS[3] || COLS[0];
    const rows = V.results.slice().sort((a, b) => {
      const x = col.get(a), y = col.get(b);
      return (typeof x === 'string' ? x.localeCompare(y) : (x ?? Infinity) - (y ?? Infinity)) * V.sortDir;
    });
    const winner = V.results.slice().sort((a, b) => sorter(S.final.better)(finalOf(a), finalOf(b)))[0].id;
    tbl.innerHTML = `<thead><tr>${COLS.map((c) => {
      const sorted = c.key === col.key;
      const aria = sorted ? (V.sortDir > 0 ? 'ascending' : 'descending') : 'none';
      const arrow = sorted ? (V.sortDir > 0 ? ' ↑' : ' ↓') : '';
      return `<th class="${c.n ? 'n' : ''}" aria-sort="${aria}" scope="col">${c.sortable === false ? esc(c.label) : `<button type="button" data-key="${esc(c.key)}">${esc(c.label)}${arrow}</button>`}</th>`;
    }).join('')}</tr></thead><tbody>${rows.map((r) => `<tr class="${r.id === winner ? 'winner' : ''}">${COLS.map((c, i) => {
      const v = c.get(r);
      if (i === 0) return `<td><span class="key" style="background:${color(r.id)}"></span>${esc(v)}</td>`;
      if (c.key === '@params') return `<td class="params-cell">${esc(v)}</td>`;
      return `<td class="${c.n ? 'n' : ''}">${esc(c.n ? fmtBy(c.fmt, v) : v)}</td>`;
    }).join('')}</tr>`).join('')}</tbody>`;
    tbl.querySelectorAll('th button').forEach((b) => b.addEventListener('click', () => {
      const c = COLS.find((x) => x.key === b.dataset.key);
      if (V.sortKey === c.key) V.sortDir *= -1;
      else { V.sortKey = c.key; V.sortDir = c.better === 'higher' ? -1 : 1; }
      renderTable();
      const again = tbl.querySelector(`th button[data-key="${CSS.escape(c.key)}"]`); if (again) again.focus();
    }));
  }

  // ------------------------------------------------------------ catalog ----
  function buildCatalog() {
    const full = V.selected.length >= SLOTS;
    $('#catalog').innerHTML = S.families.map((f) => {
      const list = S.algorithms.filter((a) => a.family === f.id);
      if (!list.length) return '';
      return `
      <section class="family">
        <div class="family-head"><h3>${esc(f.name)}</h3><p>${esc(f.blurb || '')}</p></div>
        <div class="cards">${list.map((a) => {
          const on = V.selected.includes(a.id);
          const params = (a.params || []).map((p) => `${p.label.split(' ')[0]} = ${p.default}`).join(', ');
          return `<article class="acard">
            <div class="top"><h3>${esc(a.name)}</h3>${on ? `<span class="key" style="display:inline-block;width:16px;height:3px;border-radius:2px;background:${color(a.id)}"></span>` : ''}</div>
            <div class="formula">${esc(a.formula || '')}</div>
            <p class="idea">${esc(a.idea || '')}</p>
            <p class="bound">${a.regret ? `Guarantee: <b>${esc(a.regret)}</b>` : ''}${params ? `<br>Defaults: <b>${esc(params)}</b>` : ''}</p>
            <div><button type="button" class="chip" data-algo="${esc(a.id)}" aria-pressed="${on}" ${!on && full ? 'disabled title="Remove an algorithm first: eight is the limit"' : ''}>${on ? 'In comparison' : 'Add to comparison'}</button></div>
          </article>`;
        }).join('')}</div>
      </section>`;
    }).join('');
  }
  function bindCatalog() {
    $('#catalog').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-algo]'); if (!b) return;
      const id = b.dataset.algo, on = V.selected.includes(id);
      if (on && V.selected.length <= 1) return;
      toggleAlgo(id, !on);
      buildAlgoList(); buildCatalog(); scheduleRun();
      const again = $(`#catalog button[data-algo="${CSS.escape(id)}"]`); if (again) again.focus();
    });
  }

  // ----------------------------------------------------------- diagrams ----
  function drawLoopDiagram() {
    const svg = $('#loop-diagram');
    const arrow = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="currentColor"/></marker>
      <marker id="ah-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" style="fill:var(--muted)"/></marker></defs>`;
    const means = [0.35, 0.72, 0.5, 0.2, 0.61];
    let arms = '';
    means.forEach((mu, i) => {
      const x = 600 + i * 62, h = 90 * mu;
      arms += `<rect x="${x}" y="${180 - h}" width="34" height="${h}" rx="3" style="fill:var(--surface-2);stroke:var(--muted);stroke-dasharray:3 3"/>`;
      arms += `<text class="mono ink2" x="${x + 17}" y="200" text-anchor="middle">μ${'₁₂₃₄₅'[i]}</text>`;
    });
    svg.innerHTML = arrow + `
      <rect class="box" x="40" y="24" width="360" height="200" rx="10"/>
      <text class="head" x="60" y="52">Agent</text>
      <rect class="box-2" x="60" y="66" width="320" height="52" rx="6"/>
      <text x="76" y="88">Estimates</text>
      <text class="mono ink2" x="76" y="106">Q̂ₐ and nₐ, or a posterior per arm</text>
      <rect x="60" y="150" width="320" height="56" rx="6" style="fill:var(--surface);stroke:var(--s1);stroke-width:2"/>
      <text x="76" y="172" style="font-weight:600">Selection rule</text>
      <text class="mono ink2" x="76" y="192">ε-greedy · UCB · Thompson · EXP3 · …</text>
      <path class="edge" d="M220,118 V148" marker-end="url(#ah)"/>
      <text class="dim" x="228" y="138">reads</text>

      <rect class="box" x="560" y="24" width="360" height="200" rx="10"/>
      <text class="head" x="580" y="52">Environment</text>
      <text class="dim" x="900" y="52" text-anchor="end">K arms, means hidden</text>
      ${arms}

      <path class="edge" d="M380,178 H556" marker-end="url(#ah)"/>
      <text x="468" y="170" text-anchor="middle">pull arm aₜ</text>
      <path class="edge" d="M560,92 H384" marker-end="url(#ah)"/>
      <text x="472" y="84" text-anchor="middle">reward rₜ</text>
      <text class="dim" x="472" y="110" text-anchor="middle">from arm aₜ only</text>

      <rect x="560" y="252" width="360" height="44" rx="6" style="fill:var(--bg);stroke:var(--muted);stroke-dasharray:4 4"/>
      <text class="mono" x="580" y="279">regret += μ* − μ(aₜ)</text>
      <text class="dim" x="900" y="279" text-anchor="end">simulator only</text>
      <path class="edge-dim" d="M740,224 V248" marker-end="url(#ah-dim)"/>
      <text class="dim" x="60" y="279">The agent never sees μ or its own regret.</text>`;
  }

  function drawDecideDiagram() {
    const svg = $('#decide-diagram');
    const t = 200;
    const arms = [
      { k: 'A', n: 150, q: 0.62 },
      { k: 'B', n: 30, q: 0.55 },
      { k: 'C', n: 20, q: 0.30 },
    ];
    const eps = 0.1;
    arms.forEach((a) => {
      a.s = Math.round(a.q * a.n);
      a.bonus = Math.sqrt((2 * Math.log(t)) / a.n);
      a.ucb = a.q + a.bonus;
    });
    // P(pick) for Thompson, by Monte Carlo with a fixed seed
    const rng = M.Rng(42), wins = [0, 0, 0], N = 20000;
    for (let i = 0; i < N; i++) {
      let bi = 0, bv = -1;
      arms.forEach((a, j) => { const v = rng.beta(1 + a.s, 1 + a.n - a.s); if (v > bv) { bv = v; bi = j; } });
      wins[bi]++;
    }
    const draws = [0.60, 0.66, 0.33]; // one illustrative posterior draw per arm
    const ucbBest = arms.reduce((b, a, i) => (a.ucb > arms[b].ucb ? i : b), 0);
    const panels = [
      { title: 'ε-greedy', rule: 'explore w.p. ε = 0.1, else argmax Q̂', max: 1, pick: 0,
        p: arms.map((a, i) => (i === 0 ? 1 - eps : 0) + eps / 3), foot: 'Ignores how many samples each arm has.' },
      { title: 'UCB1', rule: 'argmax Q̂ + √(2 ln t / n)', max: 1.5, pick: ucbBest,
        p: arms.map((a, i) => (i === ucbBest ? 1 : 0)), foot: 'Fewest samples, widest bonus: B wins.' },
      { title: 'Thompson sampling', rule: 'argmax of one draw θ ~ Beta(1+s, 1+f)', max: 1, pick: 1,
        p: wins.map((w) => w / N), foot: 'This draw picks B; over many draws, A wins most.' },
    ];
    const pw = 290, gap = 25, rowY = [104, 168, 232], axisY = 266;
    let out = '';
    panels.forEach((P, pi) => {
      const px = 20 + pi * (pw + gap);
      const ax0 = px + 26, ax1 = px + pw - 64;
      const X = (v) => ax0 + (Math.min(v, P.max) / P.max) * (ax1 - ax0);
      out += `<text class="head" x="${px}" y="26">${P.title}</text>`;
      out += `<text class="mono ink2" x="${px}" y="46">${esc(P.rule)}</text>`;
      out += `<text class="dim" x="${px + pw}" y="72" text-anchor="end">P(pick)</text>`;
      // gridlines
      for (let v = 0; v <= P.max + 1e-9; v += 0.5) {
        out += `<line class="gridl" x1="${X(v)}" x2="${X(v)}" y1="80" y2="${axisY}"/>`;
        out += `<text class="mono dim" x="${X(v)}" y="${axisY + 16}" text-anchor="middle">${v}</text>`;
      }
      out += `<line x1="${ax0}" x2="${ax1}" y1="${axisY}" y2="${axisY}" style="stroke:var(--baseline)"/>`;
      arms.forEach((a, i) => {
        const y = rowY[i];
        const picked = i === P.pick;
        const col = picked ? 'var(--s1)' : 'currentColor';
        out += `<text x="${px}" y="${y + 4}" style="font-weight:${picked ? 700 : 500}">${a.k}</text>`;
        if (pi === 0) {
          out += `<circle cx="${X(a.q)}" cy="${y}" r="6" style="fill:${col};stroke:var(--surface);stroke-width:2"/>`;
          out += `<text class="mono dim" x="${ax0}" y="${y + 24}">Q̂ ${a.q.toFixed(2)} · n ${a.n}</text>`;
        } else if (pi === 1) {
          out += `<line x1="${X(a.q)}" x2="${X(a.ucb)}" y1="${y}" y2="${y}" style="stroke:${col};stroke-width:2"/>`;
          out += `<line x1="${X(a.ucb)}" x2="${X(a.ucb)}" y1="${y - 7}" y2="${y + 7}" style="stroke:${col};stroke-width:2"/>`;
          out += `<circle cx="${X(a.q)}" cy="${y}" r="5" style="fill:${col};stroke:var(--surface);stroke-width:2"/>`;
          out += `<text class="mono dim" x="${ax0}" y="${y + 24}">${a.q.toFixed(2)} + ${a.bonus.toFixed(2)} = ${a.ucb.toFixed(2)}</text>`;
        } else {
          // Beta posterior density, scaled to a 34px tall ridge
          const al = 1 + a.s, be = 1 + a.n - a.s, pts = [];
          let peak = 0; const dens = [];
          for (let j = 0; j <= 120; j++) {
            const x = j / 120;
            const d = x <= 0 || x >= 1 ? 0 : Math.exp((al - 1) * Math.log(x) + (be - 1) * Math.log(1 - x));
            dens.push(d); if (d > peak) peak = d;
          }
          dens.forEach((d, j) => pts.push(`${X(j / 120).toFixed(1)},${(y + 10 - (d / peak) * 34).toFixed(1)}`));
          out += `<polygon points="${X(0)},${y + 10} ${pts.join(' ')} ${X(1)},${y + 10}" style="fill:${col};fill-opacity:0.12;stroke:${col};stroke-width:1.5"/>`;
          out += `<line x1="${X(draws[i])}" x2="${X(draws[i])}" y1="${y - 28}" y2="${y + 12}" style="stroke:${col};stroke-width:2"/>`;
          out += `<text class="mono dim" x="${ax0}" y="${y + 24}">θ = ${draws[i].toFixed(2)} · n ${a.n}</text>`;
        }
        out += `<text class="mono" x="${px + pw}" y="${y + 4}" text-anchor="end" style="font-weight:${picked ? 600 : 400}">${P.p[i] >= 0.995 ? '1' : P.p[i] < 0.005 ? '0' : P.p[i].toFixed(2)}</text>`;
      });
      out += `<text class="ink2" x="${px}" y="${axisY + 44}">${esc(P.foot)}</text>`;
    });
    svg.innerHTML = out;
  }

  // ------------------------------------------------------------- boot ----
  const total = SETTINGS.reduce((n, s) => n + s.algorithms.length, 0);
  $('#algo-total').textContent = total.toLocaleString('en-US');
  $('#setting-total').textContent = SETTINGS.length < NUM_WORDS.length ? NUM_WORDS[SETTINGS.length] : String(SETTINGS.length);
  $('#setting-word').textContent = SETTINGS.length === 1 ? 'problem setting' : 'problem settings';
  buildSettingsNav();
  bindControls();
  bindCatalog();
  drawLoopDiagram();
  drawDecideDiagram();
  const fromHash = location.hash.replace('#', '');
  switchSetting(SETTINGS.some((s) => s.id === fromHash) ? fromHash : SETTINGS[0].id, false);
  let rz = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(rz);
    rz = requestAnimationFrame(() => { if (V && V.results) { renderCurves(); renderBoard(); if (S.heatmap) renderHeat(); } });
  }).observe($('.main'));
})();
