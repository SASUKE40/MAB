/* Bandit Algorithm Arena — dashboard UI. Depends on window.MAB (src/mab.js). */
(function () {
  'use strict';
  const M = window.MAB;
  const $ = (s, r = document) => r.querySelector(s);
  const SLOTS = 8;
  const DEFAULT_LINEUP = ['greedy', 'egreedy', 'ucb1', 'ucb-tuned', 'klucb', 'thompson', 'gradient', 'exp3'];
  const PRESETS = [
    { label: 'Classic lineup', algos: DEFAULT_LINEUP },
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
  const METRICS = {
    regret: { label: 'Cumulative regret', better: 'lower', fmt: fmtNum },
    optimal: { label: 'Best-arm rate', better: 'higher', fmt: fmtPct },
    reward: { label: 'Reward per step', better: 'higher', fmt: (v) => v.toFixed(3) },
  };

  const state = {
    env: 'bernoulli', K: 10, T: 1000, runs: 200, seed: 7,
    selected: [], slot: {}, params: {},
    metric: 'regret', logx: false, focus: null,
    sortKey: 'finalRegret', sortDir: 1,
    results: null, cfg: null, runner: null,
  };
  for (const a of M.ALGORITHMS) state.params[a.id] = M.defaultParams(a.id);
  setLineup(DEFAULT_LINEUP);

  // ------------------------------------------------------------ utils ----
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function fmtNum(v) {
    const a = Math.abs(v);
    if (a >= 100) return Math.round(v).toLocaleString('en-US');
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
  function fmtPct(v) { return (v * 100).toFixed(v < 0.1 ? 1 : 0) + '%'; }
  function fmtTick(v) {
    if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString('en-US');
    return +v.toFixed(6) + '';
  }
  function color(id) { return `var(--s${state.slot[id] + 1})`; }
  function algoName(id) { return M.ALGO_BY_ID[id].name; }
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

  function setLineup(ids) {
    state.selected = [];
    state.slot = {};
    ids.slice(0, SLOTS).forEach((id, i) => { state.selected.push(id); state.slot[id] = i; });
  }
  function toggleAlgo(id, on) {
    if (on && !state.selected.includes(id)) {
      if (state.selected.length >= SLOTS) return false;
      const used = new Set(Object.values(state.slot));
      let s = 0; while (used.has(s)) s++;
      state.slot[id] = s;
      state.selected.push(id);
    } else if (!on) {
      state.selected = state.selected.filter((x) => x !== id);
      delete state.slot[id];
    }
    return true;
  }

  // ---------------------------------------------------------- controls ----
  function buildEnvs() {
    const box = $('#envs');
    box.innerHTML = M.ENVIRONMENTS.map((e) => `
      <label class="env-opt">
        <input type="radio" name="env" id="env-${e.id}" value="${e.id}" ${e.id === state.env ? 'checked' : ''}>
        <b>${esc(e.name)}${e.stationary ? '' : '<span class="tag">changes</span>'}${e.contextual ? '<span class="tag">context</span>' : ''}</b>
        <span>${esc(e.description)}</span>
      </label>`).join('');
    box.addEventListener('change', (ev) => { state.env = ev.target.value; scheduleRun(); });
  }

  function buildPresets() {
    const box = $('#presets');
    PRESETS.forEach((p) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip'; b.textContent = p.label;
      if (p.env) b.title = `Also switches to the ${M.ENV_BY_ID[p.env].name.toLowerCase()} environment`;
      b.addEventListener('click', () => {
        setLineup(p.algos);
        if (p.env) { state.env = p.env; const r = $(`#env-${p.env}`); if (r) r.checked = true; }
        buildAlgoList(); buildCatalog(); scheduleRun(0);
      });
      box.appendChild(b);
    });
  }

  function buildAlgoList() {
    const active = document.activeElement && document.activeElement.id;
    const full = state.selected.length >= SLOTS;
    const q = ($('#algo-filter').value || '').trim().toLowerCase();
    const match = (a) => !q || a.name.toLowerCase().includes(q) || a.id.includes(q) || state.selected.includes(a.id);
    const html = M.FAMILIES.map((f) => {
      const algos = M.ALGORITHMS.filter((a) => a.family === f.id && (match(a) || f.name.toLowerCase().includes(q)));
      if (!algos.length) return '';
      return `<div class="algo-group"><div class="eyebrow">${esc(f.name)}</div>${algos.map((a) => {
        const on = state.selected.includes(a.id);
        const dis = !on && full;
        const params = on && a.params.length ? `<div class="params">${a.params.map((p) => `
            <label for="p-${a.id}-${p.key}">${esc(p.label)}
              <input type="number" id="p-${a.id}-${p.key}" data-algo="${a.id}" data-key="${p.key}"
                value="${state.params[a.id][p.key]}" min="${p.min}" max="${p.max}" step="${p.step}">
            </label>`).join('')}</div>` : '';
        return `<div class="algo${dis ? ' disabled' : ''}">
          <label for="a-${a.id}">
            <input type="checkbox" id="a-${a.id}" data-algo="${a.id}" ${on ? 'checked' : ''} ${dis ? 'disabled' : ''}>
            <span class="sw" style="background:${on ? color(a.id) : 'transparent'}"></span>
            <span>${esc(a.name)}</span>
          </label>${params}</div>`;
      }).join('')}</div>`;
    }).join('');
    $('#algo-list').innerHTML = html;
    $('#algo-empty').hidden = html.trim() !== '';
    $('#algo-count').textContent = `${state.selected.length} of ${SLOTS}`;
    if (active) { const el = document.getElementById(active); if (el) el.focus(); }
  }

  function bindControls() {
    $('#algo-list').addEventListener('change', (ev) => {
      const t = ev.target;
      if (t.type === 'checkbox') {
        if (!t.checked && state.selected.length <= 1) { t.checked = true; return; }
        toggleAlgo(t.dataset.algo, t.checked);
        buildAlgoList(); buildCatalog(); scheduleRun();
      } else if (t.type === 'number') {
        const p = M.ALGO_BY_ID[t.dataset.algo].params.find((x) => x.key === t.dataset.key);
        let v = parseFloat(t.value);
        if (!Number.isFinite(v)) v = p.default;
        v = Math.min(p.max, Math.max(p.min, v));
        t.value = v;
        state.params[t.dataset.algo][t.dataset.key] = v;
        scheduleRun();
      }
    });
    $('#algo-filter').addEventListener('input', buildAlgoList);
    const k = $('#k-range');
    k.addEventListener('input', () => { state.K = +k.value; $('#k-out').textContent = k.value; scheduleRun(); });
    $('#t-select').addEventListener('change', (e) => { state.T = +e.target.value; scheduleRun(); });
    $('#runs-select').addEventListener('change', (e) => { state.runs = +e.target.value; scheduleRun(); });
    const seed = $('#seed-input');
    seed.addEventListener('change', () => { state.seed = Math.max(0, parseInt(seed.value, 10) || 0); seed.value = state.seed; scheduleRun(); });
    $('#seed-roll').addEventListener('click', () => { state.seed = Math.floor(Math.random() * 100000); seed.value = state.seed; scheduleRun(0); });
    $('#run-btn').addEventListener('click', () => scheduleRun(0));
    $('#metric-tabs').addEventListener('click', (ev) => {
      const b = ev.target.closest('button[data-metric]'); if (!b) return;
      state.metric = b.dataset.metric;
      for (const x of $('#metric-tabs').children) x.setAttribute('aria-selected', String(x === b));
      renderCurves();
    });
    $('#logx').addEventListener('change', (e) => { state.logx = e.target.checked; renderCurves(); });
  }

  // ------------------------------------------------------------ runner ----
  let runTimer = null, job = 0;
  function scheduleRun(delay = 250) {
    clearTimeout(runTimer);
    runTimer = setTimeout(startRun, delay);
  }
  function startRun() {
    const my = ++job;
    const cfg = {
      env: state.env, K: state.K, T: state.T, runs: state.runs, seed: state.seed,
      algorithms: state.selected.map((id) => ({ id, params: Object.assign({}, state.params[id]) })),
    };
    const runner = M.createRunner(cfg);
    const t0 = performance.now();
    let lastDraw = 0, firstDraw = true;
    document.querySelectorAll('.chart').forEach((c) => c.classList.add('stale'));
    setStatus('running', `Running 0 / ${cfg.runs}`, 0);
    function tick() {
      if (my !== job) return;
      const start = performance.now();
      do { runner.step(1); } while (!runner.finished && performance.now() - start < 30);
      const now = performance.now();
      if (runner.finished || now - lastDraw > 300 || firstDraw) {
        state.results = runner.results();
        state.cfg = cfg; state.runner = runner;
        if (firstDraw) document.querySelectorAll('.chart').forEach((c) => c.classList.remove('stale'));
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
    renderTiles(); renderCurves(); renderBoard(); renderHeat(); renderTable();
  }

  function renderTiles() {
    const R = state.results, cfg = state.cfg, env = M.ENV_BY_ID[cfg.env];
    const byReg = R.slice().sort((a, b) => a.finalRegret - b.finalRegret);
    const best = byReg[0], worst = byReg[byReg.length - 1];
    const bestOpt = R.slice().sort((a, b) => b.optimalTail - a.optimalTail)[0];
    const cheap = R.slice().sort((a, b) => a.msPerRun - b.msPerRun)[0];
    const runs = R[0].runs;
    const ratio = best.finalRegret > 0 ? worst.finalRegret / best.finalRegret : 0;
    const tiles = [
      { label: 'Lowest regret', id: best.id, sub: `${fmtNum(best.finalRegret)} ± ${fmtNum(1.96 * best.finalSE)} after ${cfg.T.toLocaleString('en-US')} steps` },
      { label: 'Picks the best arm most', id: bestOpt.id, sub: `${fmtPct(bestOpt.optimalTail)} of the last 10% of steps` },
      { label: 'Highest regret', id: worst.id, sub: `${fmtNum(worst.finalRegret)}${R.length > 1 && ratio ? `, ${ratio.toFixed(1)}× the leader` : ''}` },
      { label: 'Cheapest to run', id: cheap.id, sub: `${cheap.msPerRun.toFixed(2)} ms per ${cfg.T.toLocaleString('en-US')}-step run` },
    ];
    $('#tiles').innerHTML = tiles.map((t) => `
      <div class="tile">
        <div class="label">${esc(t.label)}</div>
        <div class="value"><span class="key" style="background:${color(t.id)}"></span><span class="txt">${esc(algoName(t.id))}</span></div>
        <div class="sub num">${esc(t.sub)}</div>
      </div>`).join('');
    $('#curve-sub').textContent = `${env.name} · K = ${cfg.K} · ${runs} of ${cfg.runs} runs averaged · seed ${cfg.seed}`;
  }

  // ---- learning curves ----
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
  function sampleSteps(T, n, log) {
    const out = [];
    if (!log) {
      const step = Math.max(1, T / n);
      for (let x = 1; x < T; x += step) out.push(Math.round(x));
    } else {
      const l = Math.log(T);
      for (let i = 0; i < n; i++) out.push(Math.round(Math.exp((l * i) / (n - 1))));
    }
    out.push(T);
    return [...new Set(out)].filter((t) => t >= 1 && t <= T);
  }

  let curveGeo = null;
  function renderCurves() {
    const el = $('#curve-chart');
    if (!state.results) return;
    const R = state.results, T = state.cfg.T, metric = state.metric, log = state.logx;
    const W = Math.max(280, el.clientWidth);
    const H = Math.round(Math.min(440, Math.max(260, W * 0.46)));
    const winner = metric === 'regret'
      ? R.slice().sort((a, b) => a.finalRegret - b.finalRegret)[0]
      : R.slice().sort((a, b) => (metric === 'optimal' ? b.optimalTail - a.optimalTail : b.meanReward - a.meanReward))[0];
    const smoothW = metric === 'regret' ? 1 : Math.max(1, Math.round(T / 100));
    const series = R.map((r) => ({
      id: r.id, name: r.name,
      v: metric === 'regret' ? r.regret : smooth(metric === 'optimal' ? r.optimal : r.reward, smoothW),
      se: metric === 'regret' ? r.regretSE : null,
    }));
    const endVal = (s) => s.v[T - 1];
    const winnerLabel = `${winner.name} ${METRICS[metric].fmt(endVal(series.find((s) => s.id === winner.id)))}`;
    const showLabel = W >= 560;
    const m = { l: 52, r: showLabel ? Math.ceil(textWidth(winnerLabel, '600 ' + LABEL_FONT)) + 22 : 14, t: 10, b: 34 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const steps = sampleSteps(T, Math.min(T, Math.max(60, Math.round(pw / 2))), log);

    // y domain
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const t of steps) {
      const v = s.v[t - 1], e = s.se ? 1.96 * s.se[t - 1] : 0;
      if (v + e > hi) hi = v + e;
      if (v - e < lo) lo = v - e;
    }
    if (metric === 'regret') lo = 0;
    if (metric === 'optimal') { lo = 0; hi = 1; }
    if (hi - lo < 1e-9) hi = lo + 1;
    const yt = ticks(lo, hi, Math.max(3, Math.round(ph / 60)));
    const step = yt.length > 1 ? yt[1] - yt[0] : 1;
    if (metric !== 'optimal') {
      if (yt[yt.length - 1] < hi) yt.push(+(yt[yt.length - 1] + step).toFixed(10));
      if (yt[0] > lo) yt.unshift(+(yt[0] - step).toFixed(10));
    }
    const y0 = yt[0], y1 = yt[yt.length - 1];
    const Y = (v) => m.t + ph - ((v - y0) / (y1 - y0)) * ph;
    const X = log
      ? (t) => m.l + (Math.log(t) / Math.log(T)) * pw
      : (t) => m.l + ((t - 1) / Math.max(1, T - 1)) * pw;
    let xt;
    if (log) { xt = []; for (let d = 1; d <= T; d *= 10) xt.push(d); if (xt[xt.length - 1] !== T && X(T) - X(xt[xt.length - 1]) > 40) xt.push(T); }
    else { xt = ticks(0, T, Math.max(2, Math.round(pw / 110))).filter((t) => t >= 1 || t === 0).map((t) => Math.max(1, t)); }

    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(METRICS[metric].label)} over time for ${R.length} algorithms">`;
    svg += '<g class="grid">' + yt.map((v) => `<line x1="${m.l}" x2="${m.l + pw}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/>`).join('') + '</g>';
    svg += `<line class="baseline" x1="${m.l}" x2="${m.l + pw}" y1="${m.t + ph}" y2="${m.t + ph}"/>`;
    svg += '<g class="axis">' + yt.map((v) => `<text x="${m.l - 8}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${esc(metric === 'optimal' ? Math.round(v * 100) + '%' : fmtTick(v))}</text>`).join('');
    svg += xt.map((t) => `<text x="${X(t).toFixed(1)}" y="${m.t + ph + 18}" text-anchor="middle">${esc(fmtTick(t))}</text>`).join('');
    svg += `<text x="${m.l + pw}" y="${H - 2}" text-anchor="end">step t${log ? ' (log scale)' : ''}</text></g>`;

    // bands then lines
    if (metric === 'regret') {
      for (const s of series) {
        const up = steps.map((t) => `${X(t).toFixed(1)},${Y(s.v[t - 1] + 1.96 * s.se[t - 1]).toFixed(1)}`);
        const dn = steps.slice().reverse().map((t) => `${X(t).toFixed(1)},${Y(Math.max(y0, s.v[t - 1] - 1.96 * s.se[t - 1])).toFixed(1)}`);
        svg += `<polygon class="series" data-id="${s.id}" points="${up.join(' ')} ${dn.join(' ')}" style="fill:${color(s.id)};fill-opacity:0.10;stroke:none"/>`;
      }
    }
    // draw winner last so it sits on top
    const order = series.slice().sort((a, b) => (a.id === winner.id) - (b.id === winner.id));
    for (const s of order) {
      const pts = steps.map((t) => `${X(t).toFixed(1)},${Y(s.v[t - 1]).toFixed(1)}`).join(' ');
      svg += `<polyline class="series" data-id="${s.id}" points="${pts}" style="fill:none;stroke:${color(s.id)};stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>`;
    }
    for (const s of order) {
      svg += `<circle class="series" data-id="${s.id}" cx="${X(T).toFixed(1)}" cy="${Y(endVal(s)).toFixed(1)}" r="4" style="fill:${color(s.id)};stroke:var(--surface);stroke-width:2"/>`;
    }
    if (showLabel) {
      const wy = Y(endVal(series.find((s) => s.id === winner.id)));
      svg += `<text class="lbl" x="${m.l + pw + 10}" y="${(wy + 4).toFixed(1)}">${esc(winnerLabel)}</text>`;
    }
    svg += '<g id="curve-hover"></g>';
    svg += `<rect id="curve-hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="transparent" tabindex="0" aria-label="Chart readout. Use left and right arrow keys to move through time."/>`;
    svg += '</svg>';
    el.innerHTML = svg + '<div class="tooltip" hidden></div>';
    curveGeo = { X, Y, steps, series, m, pw, ph, T, metric, log, W };

    // legend
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

    const notes = {
      regret: 'Shaded bands: 95% confidence interval of the mean across runs. Logarithmic growth shows up as a straight line on the log time axis.',
      optimal: `Share of runs that pulled the best arm at each step, smoothed over ${smoothW} steps.`,
      reward: `Average reward received at each step, smoothed over ${smoothW} steps.`,
    };
    $('#curve-note').textContent = notes[metric];

    const hit = el.querySelector('#curve-hit');
    let cur = steps.length - 1;
    const show = (i) => { cur = Math.max(0, Math.min(steps.length - 1, i)); drawHover(el, cur); };
    hit.addEventListener('pointermove', (e) => {
      const rect = svgEl.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      let best = 0, bd = Infinity;
      // binary search on x
      let a = 0, b = steps.length - 1;
      while (b - a > 1) { const mid = (a + b) >> 1; if (X(steps[mid]) < px) a = mid; else b = mid; }
      for (const i of [a, b]) { const d = Math.abs(X(steps[i]) - px); if (d < bd) { bd = d; best = i; } }
      show(best);
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
  function drawHover(el, i) {
    const { X, Y, steps, series, m, ph, metric, W } = curveGeo;
    const t = steps[i], x = X(t);
    let g = `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ph}" style="stroke:var(--ink-2);stroke-width:1"/>`;
    for (const s of series) g += `<circle cx="${x}" cy="${Y(s.v[t - 1])}" r="4" style="fill:${color(s.id)};stroke:var(--surface);stroke-width:2"/>`;
    el.querySelector('#curve-hover').innerHTML = g;
    const tip = el.querySelector('.tooltip');
    tip.innerHTML = '';
    const head = document.createElement('div'); head.className = 't-head'; head.textContent = `t = ${t.toLocaleString('en-US')}`;
    tip.appendChild(head);
    const dir = metric === 'regret' ? 1 : -1;
    const rows = series.map((s) => ({ s, v: s.v[t - 1] })).sort((a, b) => dir * (a.v - b.v));
    for (const r of rows) {
      const row = document.createElement('div'); row.className = 't-row';
      const k = document.createElement('span'); k.className = 'key'; k.style.background = color(r.s.id);
      const v = document.createElement('b'); v.textContent = METRICS[metric].fmt(r.v);
      const n = document.createElement('span'); n.textContent = r.s.name;
      row.append(k, v, n); tip.appendChild(row);
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
  function barPath(x0, x1, y, h, r) {
    const w = x1 - x0;
    if (w <= 0) return '';
    r = Math.min(r, w, h / 2);
    return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
  }
  function renderBoard() {
    const el = $('#board-chart');
    const R = state.results.slice().sort((a, b) => a.finalRegret - b.finalRegret);
    const W = Math.max(260, el.clientWidth);
    const rowH = 42, barH = 12, top = 4;
    const H = top + R.length * rowH + 22;
    const valW = Math.max(...R.map((r) => textWidth(fmtNum(r.finalRegret), '500 11px "IBM Plex Mono", monospace'))) + 10;
    const x0 = 2, x1 = W - valW - 4;
    const max = Math.max(...R.map((r) => r.finalRegret + 1.96 * r.finalSE), 1e-9);
    const xt = ticks(0, max, Math.max(2, Math.round((x1 - x0) / 90)));
    const xmax = Math.max(xt[xt.length - 1], max);
    const X = (v) => x0 + (Math.max(0, v) / xmax) * (x1 - x0);
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Final cumulative regret by algorithm, lowest first">`;
    svg += '<g class="grid">' + xt.map((v) => `<line x1="${X(v).toFixed(1)}" x2="${X(v).toFixed(1)}" y1="${top}" y2="${H - 20}"/>`).join('') + '</g>';
    svg += '<g class="axis">' + xt.map((v, i) => `<text x="${X(v).toFixed(1)}" y="${H - 4}" text-anchor="${i === 0 ? 'start' : 'middle'}">${esc(fmtTick(v))}</text>`).join('') + '</g>';
    R.forEach((r, i) => {
      const y = top + i * rowH, by = y + 20;
      const ci = 1.96 * r.finalSE;
      svg += `<g class="bar-row" data-id="${r.id}" tabindex="0" role="img" aria-label="${esc(r.name)}: ${fmtNum(r.finalRegret)} plus or minus ${fmtNum(ci)}">`;
      svg += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
      svg += `<text class="lbl-2" x="${x0}" y="${y + 14}">${esc(r.name)}</text>`;
      svg += `<path d="${barPath(X(0), X(r.finalRegret), by, barH, 4)}" style="fill:${color(r.id)}"/>`;
      const lo = X(r.finalRegret - ci), hi = X(r.finalRegret + ci), cy = by + barH / 2;
      if (hi - lo > 2) svg += `<path d="M${lo},${cy}H${hi}M${lo},${cy - 4}V${cy + 4}M${hi},${cy - 4}V${cy + 4}" style="stroke:var(--ink-2);stroke-width:1.25;fill:none"/>`;
      svg += `<text x="${Math.max(hi, X(r.finalRegret)) + 6}" y="${cy + 4}" style="fill:var(--ink);font-weight:500">${esc(fmtNum(r.finalRegret))}</text>`;
      svg += '</g>';
    });
    svg += `<line class="baseline" x1="${X(0)}" x2="${X(0)}" y1="${top}" y2="${H - 20}"/>`;
    svg += '</svg>';
    el.innerHTML = svg + '<div class="tooltip" hidden></div>';
    const tip = el.querySelector('.tooltip');
    const showTip = (g) => {
      const r = state.results.find((x) => x.id === g.dataset.id);
      tip.innerHTML = '';
      const h = document.createElement('div'); h.className = 't-head'; h.textContent = r.name;
      tip.appendChild(h);
      const rows = [
        ['Mean regret', fmtNum(r.finalRegret)], ['95% CI', '± ' + fmtNum(1.96 * r.finalSE)],
        ['Median', fmtNum(r.finalMedian)], ['Best run', fmtNum(r.finals[0])], ['Worst run', fmtNum(r.finals[r.finals.length - 1])],
      ];
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
    const R = state.results.slice().sort((a, b) => a.finalRegret - b.finalRegret);
    const K = state.cfg.K;
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
  const COLS = [
    { key: 'name', label: 'Algorithm', get: (r) => r.name },
    { key: 'family', label: 'Family', get: (r) => M.FAMILIES.find((f) => f.id === r.family).name },
    { key: 'params', label: 'Settings', get: (r) => Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join(' ') || '—', sortable: false },
    { key: 'finalRegret', label: 'Regret', n: true, get: (r) => r.finalRegret, fmt: fmtNum },
    { key: 'finalSE', label: '95% CI', n: true, get: (r) => 1.96 * r.finalSE, fmt: (v) => '± ' + fmtNum(v) },
    { key: 'finalMedian', label: 'Median', n: true, get: (r) => r.finalMedian, fmt: fmtNum },
    { key: 'optimalTail', label: 'Best arm, last 10%', n: true, get: (r) => r.optimalTail, fmt: fmtPct },
    { key: 'best', label: 'Pulls on best arm', n: true, get: (r) => r.pullShare[0], fmt: fmtPct },
    { key: 'meanReward', label: 'Mean reward', n: true, get: (r) => r.meanReward, fmt: (v) => v.toFixed(3) },
    { key: 'msPerRun', label: 'ms / run', n: true, get: (r) => r.msPerRun, fmt: (v) => v.toFixed(2) },
  ];
  function renderTable() {
    const tbl = $('#results-table');
    const col = COLS.find((c) => c.key === state.sortKey) || COLS[3];
    const rows = state.results.slice().sort((a, b) => {
      const x = col.get(a), y = col.get(b);
      return (typeof x === 'string' ? x.localeCompare(y) : x - y) * state.sortDir;
    });
    const winner = state.results.slice().sort((a, b) => a.finalRegret - b.finalRegret)[0].id;
    tbl.innerHTML = `<thead><tr>${COLS.map((c) => {
      const sorted = c.key === state.sortKey;
      const aria = sorted ? (state.sortDir > 0 ? 'ascending' : 'descending') : 'none';
      const arrow = sorted ? (state.sortDir > 0 ? ' ↑' : ' ↓') : '';
      return `<th class="${c.n ? 'n' : ''}" aria-sort="${aria}" scope="col">${c.sortable === false ? esc(c.label) : `<button type="button" data-key="${c.key}">${esc(c.label)}${arrow}</button>`}</th>`;
    }).join('')}</tr></thead><tbody>${rows.map((r) => `<tr class="${r.id === winner ? 'winner' : ''}">${COLS.map((c, i) => {
      const v = c.get(r);
      if (i === 0) return `<td><span class="key" style="background:${color(r.id)}"></span>${esc(v)}</td>`;
      if (c.key === 'params') return `<td class="params-cell">${esc(v)}</td>`;
      return `<td class="${c.n ? 'n' : ''}">${esc(c.fmt ? c.fmt(v) : v)}</td>`;
    }).join('')}</tr>`).join('')}</tbody>`;
    tbl.querySelectorAll('th button').forEach((b) => b.addEventListener('click', () => {
      if (state.sortKey === b.dataset.key) state.sortDir *= -1;
      else { state.sortKey = b.dataset.key; state.sortDir = COLS.find((c) => c.key === b.dataset.key).n && !['finalRegret', 'finalSE', 'finalMedian', 'msPerRun'].includes(b.dataset.key) ? -1 : 1; }
      renderTable();
      const again = tbl.querySelector(`th button[data-key="${b.dataset.key}"]`); if (again) again.focus();
    }));
  }

  // ------------------------------------------------------------ catalog ----
  function buildCatalog() {
    const full = state.selected.length >= SLOTS;
    $('#catalog').innerHTML = M.FAMILIES.map((f) => `
      <section class="family">
        <div class="family-head"><h3>${esc(f.name)}</h3><p>${esc(f.blurb)}</p></div>
        <div class="cards">${M.ALGORITHMS.filter((a) => a.family === f.id).map((a) => {
          const on = state.selected.includes(a.id);
          const params = a.params.map((p) => `${p.label.split(' ')[0]} = ${p.default}`).join(', ');
          return `<article class="acard">
            <div class="top"><h3>${esc(a.name)}</h3>${on ? `<span class="key" style="display:inline-block;width:16px;height:3px;border-radius:2px;background:${color(a.id)}"></span>` : ''}</div>
            <div class="formula">${esc(a.formula)}</div>
            <p class="idea">${esc(a.idea)}</p>
            <p class="bound">Regret: <b>${esc(a.regret)}</b>${params ? `<br>Defaults: <b>${esc(params)}</b>` : ''}</p>
            <div><button type="button" class="chip" data-algo="${a.id}" aria-pressed="${on}" ${!on && full ? 'disabled title="Remove an algorithm first: eight is the limit"' : ''}>${on ? 'In comparison' : 'Add to comparison'}</button></div>
          </article>`;
        }).join('')}</div>
      </section>`).join('');
  }
  function bindCatalog() {
    $('#catalog').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-algo]'); if (!b) return;
      const id = b.dataset.algo, on = state.selected.includes(id);
      if (on && state.selected.length <= 1) return;
      toggleAlgo(id, !on);
      buildAlgoList(); buildCatalog(); scheduleRun();
      const again = $(`#catalog button[data-algo="${id}"]`); if (again) again.focus();
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
  const WORDS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
  const TENS = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty'];
  const n = M.ALGORITHMS.length;
  $('#algo-total').textContent = n >= 20 && n < 70 ? TENS[Math.floor(n / 10)] + (n % 10 ? '-' + WORDS[n % 10].toLowerCase() : '') : String(n);
  buildEnvs();
  buildPresets();
  buildAlgoList();
  bindControls();
  buildCatalog();
  bindCatalog();
  drawLoopDiagram();
  drawDecideDiagram();
  let rz = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(rz);
    rz = requestAnimationFrame(() => { if (state.results) { renderCurves(); renderBoard(); renderHeat(); } });
  }).observe($('.main'));
  scheduleRun(0);
})();
