#!/usr/bin/env node
/*
 * Keeps dashboard/index.html's add-on <script> tags in sync with src/algos and src/settings,
 * and (with --single <file>) writes a self-contained single-file copy with every script inlined.
 *
 *   node scripts/build-dashboard.js
 *   node scripts/build-dashboard.js --single dist/bandit-arena.html
 */
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const htmlPath = path.join(root, 'dashboard', 'index.html');
const addons = [];
for (const dir of ['algos', 'settings']) {
  const full = path.join(root, 'src', dir);
  if (!fs.existsSync(full)) continue;
  for (const f of fs.readdirSync(full).filter((x) => x.endsWith('.js')).sort()) addons.push(`../src/${dir}/${f}`);
}

let html = fs.readFileSync(htmlPath, 'utf8');
const tags = addons.map((src) => `<script src="${src}"></script>`).join('\n');
html = html.replace(/<!-- addons:start -->[\s\S]*?<!-- addons:end -->/, `<!-- addons:start -->\n${tags}${tags ? '\n' : ''}<!-- addons:end -->`);
fs.writeFileSync(htmlPath, html);
console.log(`dashboard/index.html: ${addons.length} add-on scripts`);

const i = process.argv.indexOf('--single');
if (i > 0) {
  const out = process.argv[i + 1];
  const inline = (src) => fs.readFileSync(path.join(root, 'dashboard', src), 'utf8').replace(/<\/script/gi, '<\\/script');
  let single = html.replace(/<script src="([^"]+)"><\/script>/g, (_, src) => `<script>\n${inline(src)}\n</script>`);
  if (process.argv.includes('--fragment')) {
    // For hosts that supply their own document skeleton: drop doctype/html/head/body wrappers.
    single = single.replace(/<!doctype html>\s*<html[^>]*>\s*<head>\s*/i, '')
      .replace(/<meta charset="utf-8">\s*/, '').replace(/<meta name="viewport"[^>]*>\s*/, '')
      .replace(/<\/head>\s*<body>\s*/, '\n').replace(/\s*<\/body>\s*<\/html>\s*$/, '\n');
  }
  fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
  fs.writeFileSync(out, single);
  console.log(`wrote ${out} (${(single.length / 1024).toFixed(0)} KB)`);
}
