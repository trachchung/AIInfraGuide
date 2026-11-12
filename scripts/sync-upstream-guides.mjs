#!/usr/bin/env node
// Pull guide changes from upstream (origin) and land them on English paths
// using scripts/path-map.json.
//
// Workflow:
//   1. git fetch <remote>
//   2. Diff upstream guides vs our guides (by remapped English path)
//   3. Copy/add/update files into docs/guides/<en>
//   4. Fail loudly listing any upstream Chinese paths missing from the map
//
// Does NOT auto-update i18n translations — only the Chinese source tree under
// docs/guides. Re-run scripts/translate.mjs afterward for en/vi.
//
// Usage:
//   node scripts/sync-upstream-guides.mjs [--remote origin] [--ref origin/main] [--dry-run]

import { spawnSync } from 'node:child_process';
import { mkdir, writeFile, readFile, access } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { ROOT, loadPathMap, remapGuidesRelative } from './path-map-lib.mjs';

const { values: opts } = parseArgs({
  options: {
    remote: { type: 'string', default: 'origin' },
    ref: { type: 'string', default: '' },
    'dry-run': { type: 'boolean', default: false },
  },
});

const ref = opts.ref || `${opts.remote}/main`;
const map = await loadPathMap();
const guidesRoot = path.join(ROOT, 'docs', 'guides');

function git(args, encoding = 'utf8') {
  const r = spawnSync('git', args, { cwd: ROOT, encoding, maxBuffer: 64 * 1024 * 1024 });
  if (r.status !== 0) {
    throw new Error(`git ${args.join(' ')} failed:\n${r.stderr || r.stdout}`);
  }
  return r.stdout;
}

console.log(`fetching ${opts.remote}...`);
git(['fetch', opts.remote]);

// core.quotepath=false so CJK paths are not octal-escaped
const upstreamFiles = git(['-c', 'core.quotepath=false', 'ls-tree', '-r', '--name-only', ref, '--', 'docs/guides'])
  .split('\n')
  .map((l) => l.trim().replace(/^"|"$/g, ''))
  .filter((l) => l.endsWith('.md'));
const unmapped = [];
const planned = [];

for (const repoPath of upstreamFiles) {
  const rel = repoPath.replace(/^docs\/guides\//, '');
  const hasCjk = /[\u4e00-\u9fff]/.test(rel);
  let enRel;
  if (map.zhToEn.has(rel)) {
    enRel = map.zhToEn.get(rel);
  } else if (!hasCjk) {
    enRel = rel; // already English / latin path upstream
  } else {
    // Try remap; if still contains CJK, it's unmapped
    enRel = remapGuidesRelative(rel, map);
    if (/[\u4e00-\u9fff]/.test(enRel) || enRel.includes('MISSING')) {
      unmapped.push(rel);
      continue;
    }
  }
  planned.push({ upstream: repoPath, enRel });
}

if (unmapped.length) {
  console.error(`\n${unmapped.length} upstream guide path(s) are not in scripts/path-map.json:\n`);
  for (const p of unmapped) console.error(`  - ${p}`);
  console.error(`\nAdd zh→en entries to the map, then re-run.`);
  process.exit(1);
}

let written = 0;
let unchanged = 0;

for (const { upstream, enRel } of planned) {
  const blob = spawnSync('git', ['show', `${ref}:${upstream}`], {
    cwd: ROOT,
    encoding: 'buffer',
    maxBuffer: 64 * 1024 * 1024,
  });
  if (blob.status !== 0) {
    console.warn(`skip (missing blob): ${upstream}`);
    continue;
  }
  const target = path.join(guidesRoot, enRel);
  let same = false;
  try {
    await access(target);
    const cur = await readFile(target);
    same = Buffer.compare(cur, blob.stdout) === 0;
  } catch {
    same = false;
  }
  if (same) {
    unchanged++;
    continue;
  }
  console.log(`${same ? '=' : opts['dry-run'] ? 'would write' : 'write'} docs/guides/${enRel}  ←  ${upstream}`);
  if (!opts['dry-run']) {
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, blob.stdout);
  }
  written++;
}

console.log(`\ndone: ${written} updated, ${unchanged} unchanged, ${planned.length} total upstream guides`);
console.log(`Next: node scripts/translate.mjs --check   # then translate new/changed guides`);
