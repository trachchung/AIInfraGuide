#!/usr/bin/env node
// Apply scripts/path-map.json: git-mv Chinese guide paths to English across
// docs/guides and docs/i18n/{en,vi}/guides. Idempotent — skips missing sources
// and already-renamed targets.
//
// Usage:
//   node scripts/rename-guides.mjs [--dry-run]

import { spawnSync } from 'node:child_process';
import { access, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { GUIDE_ROOTS, ROOT, loadPathMap } from './path-map-lib.mjs';

const { values: opts } = parseArgs({
  options: { 'dry-run': { type: 'boolean', default: false } },
});

const { files } = await loadPathMap();
let moved = 0;
let skipped = 0;

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

function gitMv(from, to) {
  if (opts['dry-run']) {
    console.log(`git mv ${path.relative(ROOT, from)} → ${path.relative(ROOT, to)}`);
    return true;
  }
  const r = spawnSync('git', ['mv', from, to], { cwd: ROOT, encoding: 'utf8' });
  if (r.status !== 0) {
    // Fallback when path isn't tracked yet
    const r2 = spawnSync('mv', [from, to], { cwd: ROOT, encoding: 'utf8' });
    if (r2.status !== 0) {
      console.error(r.stderr || r2.stderr || `failed: ${from} → ${to}`);
      return false;
    }
  }
  return true;
}

for (const root of GUIDE_ROOTS) {
  // Move deepest paths first so parent dirs stay valid during rename
  const ordered = [...files].sort((a, b) => b.zh.split('/').length - a.zh.split('/').length || b.zh.localeCompare(a.zh));

  for (const { zh, en } of ordered) {
    if (zh === en) {
      skipped++;
      continue;
    }
    const from = path.join(root, zh);
    const to = path.join(root, en);
    if (!(await exists(from))) {
      if (await exists(to)) {
        skipped++;
        continue;
      }
      console.warn(`missing: ${path.relative(ROOT, from)}`);
      skipped++;
      continue;
    }
    if (await exists(to)) {
      console.warn(`target exists, skip: ${path.relative(ROOT, to)}`);
      skipped++;
      continue;
    }
    if (!opts['dry-run']) await mkdir(path.dirname(to), { recursive: true });
    if (gitMv(from, to)) moved++;
  }
}

// Remove empty Chinese leftover directories (best-effort)
if (!opts['dry-run']) {
  spawnSync(
    'bash',
    [
      '-lc',
      `find docs/guides docs/i18n/en/guides docs/i18n/vi/guides -type d -empty -delete 2>/dev/null || true`,
    ],
    { cwd: ROOT },
  );
}

console.log(`${opts['dry-run'] ? 'dry-run: would move' : 'moved'} ${moved}, skipped ${skipped}`);
