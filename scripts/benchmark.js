#!/usr/bin/env node
/*
 * Command-line benchmark. Runs every (or selected) algorithm on every (or selected)
 * environment and prints a Markdown leaderboard per environment.
 *
 *   node scripts/benchmark.js                       # all algorithms, all environments
 *   node scripts/benchmark.js --env gaussian --T 2000 --runs 500
 *   node scripts/benchmark.js --algos ucb1,thompson,klucb --json results/run.json
 */
'use strict';
const fs = require('fs');
const path = require('path');
const M = require('../src/mab.js');

function parseArgs(argv) {
  const opts = { env: 'all', algos: 'all', K: 10, T: 1000, runs: 200, seed: 7, json: null, md: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--env') opts.env = next();
    else if (a === '--algos') opts.algos = next();
    else if (a === '--K') opts.K = +next();
    else if (a === '--T') opts.T = +next();
    else if (a === '--runs') opts.runs = +next();
    else if (a === '--seed') opts.seed = +next();
    else if (a === '--json') opts.json = next();
    else if (a === '--md') opts.md = next();
    else if (a === '-h' || a === '--help') {
      console.log(`Usage: node scripts/benchmark.js [--env id|all] [--algos id,id|all] [--K n] [--T n] [--runs n] [--seed n] [--json file] [--md file]

Environments: ${M.ENVIRONMENTS.map((e) => e.id).join(', ')}
Algorithms:   ${M.ALGORITHMS.map((a) => a.id).join(', ')}`);
      process.exit(0);
    } else throw new Error('Unknown argument: ' + a);
  }
  return opts;
}

function fmt(v) {
  const a = Math.abs(v);
  return a >= 100 ? Math.round(v).toLocaleString('en-US') : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function main() {
  const o = parseArgs(process.argv.slice(2));
  const envs = o.env === 'all' ? M.ENVIRONMENTS.map((e) => e.id) : o.env.split(',');
  const algos = o.algos === 'all' ? M.ALGORITHMS.map((a) => a.id) : o.algos.split(',');
  const out = { config: { K: o.K, T: o.T, runs: o.runs, seed: o.seed }, environments: {} };
  const md = [];
  md.push(`K = ${o.K} arms, T = ${o.T.toLocaleString('en-US')} steps, ${o.runs} runs per cell, seed ${o.seed}.`);
  md.push('Regret is mean cumulative pseudo-regret at T, ± a 95% confidence interval. "Best arm" is the share of the last 10% of steps spent on the best arm.\n');

  for (const envId of envs) {
    const env = M.ENV_BY_ID[envId];
    const t0 = Date.now();
    const res = M.simulate({ env: envId, K: o.K, T: o.T, runs: o.runs, seed: o.seed, algorithms: algos.map((id) => ({ id })) });
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    res.sort((a, b) => a.finalRegret - b.finalRegret);
    md.push(`### ${env.name}\n`);
    md.push(`${env.description}\n`);
    md.push('| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |');
    md.push('|--:|---|--:|--:|--:|--:|');
    res.forEach((r, i) => {
      md.push(`| ${i + 1} | ${r.name} | ${fmt(r.finalRegret)} | ${fmt(1.96 * r.finalSE)} | ${(r.optimalTail * 100).toFixed(0)}% | ${r.msPerRun.toFixed(2)} |`);
    });
    md.push('');
    process.stderr.write(`${envId}: ${secs}s\n`);

    // Keep the JSON small: summary numbers plus 100-point regret curves.
    const idx = Array.from({ length: 100 }, (_, i) => Math.round(((i + 1) * o.T) / 100) - 1);
    out.environments[envId] = res.map((r) => ({
      id: r.id, name: r.name, family: r.family, params: r.params,
      finalRegret: r.finalRegret, ci95: 1.96 * r.finalSE, median: r.finalMedian,
      bestArmLast10pct: r.optimalTail, meanReward: r.meanReward, msPerRun: r.msPerRun,
      pullShareByRank: r.pullShare,
      curve: { t: idx.map((i) => i + 1), regret: idx.map((i) => r.regret[i]) },
    }));
  }

  const text = md.join('\n');
  console.log(text);
  if (o.md) { fs.mkdirSync(path.dirname(o.md), { recursive: true }); fs.writeFileSync(o.md, text + '\n'); }
  if (o.json) { fs.mkdirSync(path.dirname(o.json), { recursive: true }); fs.writeFileSync(o.json, JSON.stringify(out, null, 1)); }
}

main();
