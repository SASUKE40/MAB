#!/usr/bin/env node
/*
 * Command-line benchmark. Runs a setting's algorithms on its environments and prints
 * Markdown leaderboards (and optionally JSON).
 *
 *   node scripts/benchmark.js                                   # K-armed arena, every env, every algorithm
 *   node scripts/benchmark.js --env gaussian --T 2000 --runs 500
 *   node scripts/benchmark.js --algos ucb1,thompson,klucb --json results/run.json
 *   node scripts/benchmark.js --setting dueling --runs 50
 *   node scripts/benchmark.js --list                            # settings, environments and algorithm ids
 */
'use strict';
const fs = require('fs');
const path = require('path');
const M = require('../src/index.js');

function parseArgs(argv) {
  const o = { setting: 'kab', env: 'all', algos: 'all', K: null, T: null, runs: null, seed: 7, json: null, md: null, extra: {}, list: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i], next = () => argv[++i];
    if (a === '--setting') o.setting = next();
    else if (a === '--env') o.env = next();
    else if (a === '--algos') o.algos = next();
    else if (a === '--K') o.K = +next();
    else if (a === '--T') o.T = +next();
    else if (a === '--runs') o.runs = +next();
    else if (a === '--seed') o.seed = +next();
    else if (a === '--json') o.json = next();
    else if (a === '--md') o.md = next();
    else if (a === '--set') { const [k, v] = next().split('='); o.extra[k] = +v; }
    else if (a === '--list') o.list = true;
    else if (a === '-h' || a === '--help') {
      console.log('Usage: node scripts/benchmark.js [--setting id] [--env id[,id]|all] [--algos id,id|all] [--K n] [--T n] [--runs n] [--seed n] [--set key=value] [--json file] [--md file] [--list]');
      process.exit(0);
    } else throw new Error('Unknown argument: ' + a);
  }
  return o;
}

const fmt = (v) => {
  if (v == null || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  return a >= 100 ? Math.round(v).toLocaleString('en-US') : a >= 10 ? v.toFixed(1) : a >= 0.01 || v === 0 ? v.toFixed(2) : v.toPrecision(2);
};
const fmtBy = (kind, v) => {
  if (v == null || !Number.isFinite(v)) return '—';
  if (kind === 'pct') return (v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0) + '%';
  if (kind === 'int') return Math.round(v).toLocaleString('en-US');
  if (kind === 'fixed2') return v.toFixed(2);
  if (kind === 'fixed3') return v.toFixed(3);
  if (kind === 'ci') return fmt(1.96 * v);
  return fmt(v);
};

function main() {
  const o = parseArgs(process.argv.slice(2));
  if (o.list) {
    for (const s of M.SETTINGS) {
      console.log(`\n# ${s.id}: ${s.name} (${s.algorithms.length} algorithms)`);
      console.log('environments: ' + s.environments.map((e) => e.id).join(', '));
      console.log('algorithms:   ' + s.algorithms.map((a) => a.id).join(', '));
    }
    return;
  }
  const S = M.SETTINGS.find((s) => s.id === o.setting);
  if (!S) throw new Error(`Unknown setting ${o.setting}. Settings: ${M.SETTINGS.map((s) => s.id).join(', ')}`);
  const c = S.controls || {};
  const envs = o.env === 'all' ? S.environments.map((e) => e.id) : o.env.split(',');
  const algos = o.algos === 'all' ? S.algorithms.map((a) => a.id) : o.algos.split(',');
  const extra = {};
  for (const x of c.extra || []) extra[x.key] = x.key in o.extra ? o.extra[x.key] : x.default;
  const cfgBase = {
    K: c.K ? o.K || c.K.default : undefined,
    T: c.T ? o.T || c.T.default : undefined,
    runs: o.runs || (c.runs && c.runs.default) || 100,
    seed: o.seed, extra,
  };
  const F = S.final;
  const out = { setting: S.id, config: cfgBase, environments: {} };
  const md = [];
  const parts = [`Setting: ${S.name}.`];
  if (cfgBase.K != null) parts.push(`${c.K.label || 'K'} = ${cfgBase.K}.`);
  if (cfgBase.T != null) parts.push(`${c.T.label || 'T'} = ${cfgBase.T.toLocaleString('en-US')}.`);
  for (const x of c.extra || []) parts.push(`${x.label} = ${extra[x.key]}.`);
  parts.push(`${cfgBase.runs} runs, seed ${o.seed}.`);
  md.push(parts.join(' '));
  md.push(`Ranked by ${F.label.toLowerCase()} (${F.better} is better).\n`);

  for (const envId of envs) {
    const env = S.environments.find((e) => e.id === envId);
    if (!env) throw new Error(`Unknown environment ${envId} in ${S.id}`);
    const t0 = Date.now();
    const runner = S.createRunner({ ...cfgBase, env: envId, algorithms: algos.map((id) => ({ id })) });
    runner.step(cfgBase.runs);
    const res = runner.results();
    const val = (r, k) => (r.summary ? r.summary[k] : r[k]);
    res.sort((a, b) => (F.better === 'higher' ? val(b, F.key) - val(a, F.key) : val(a, F.key) - val(b, F.key)));
    process.stderr.write(`${S.id}/${envId}: ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
    md.push(`### ${env.name}\n`);
    if (env.description) md.push(`${env.description}\n`);
    const cols = S.columns;
    md.push(`| # | Algorithm | ${cols.map((x) => x.label).join(' | ')} |`);
    md.push(`|--:|---|${cols.map(() => '--:').join('|')}|`);
    res.forEach((r, i) => md.push(`| ${i + 1} | ${r.name} | ${cols.map((x) => fmtBy(x.fmt, val(r, x.key))).join(' | ')} |`));
    md.push('');
    out.environments[envId] = res.map((r) => ({ id: r.id, name: r.name, family: r.family, params: r.params, summary: r.summary || {} }));
  }

  const text = md.join('\n');
  console.log(text);
  if (o.md) { fs.mkdirSync(path.dirname(o.md), { recursive: true }); fs.writeFileSync(o.md, text + '\n'); }
  if (o.json) { fs.mkdirSync(path.dirname(o.json), { recursive: true }); fs.writeFileSync(o.json, JSON.stringify(out, null, 1)); }
}

main();
