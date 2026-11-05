/*
 * Node entry point: the core arena plus every add-on and setting.
 * The dashboard loads the same files, in the same order, with <script> tags.
 * Set MAB_STRICT=1 to make a broken add-on throw instead of being skipped with a warning.
 */
'use strict';
const MAB = require('./mab.js');
const fs = require('fs');
const path = require('path');

const FILES = [];
for (const dir of ['algos', 'settings']) {
  const full = path.join(__dirname, dir);
  if (!fs.existsSync(full)) continue;
  for (const f of fs.readdirSync(full).filter((x) => x.endsWith('.js')).sort()) FILES.push(path.join(full, f));
}
for (const file of FILES) {
  try {
    require(file);
  } catch (err) {
    if (process.env.MAB_STRICT) throw err;
    process.stderr.write(`[mab] skipped ${path.relative(__dirname, file)}: ${err.message}\n`);
  }
}

module.exports = MAB;
