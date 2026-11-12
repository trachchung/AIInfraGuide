import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(HERE, '..');
export const PATH_MAP_FILE = path.join(HERE, 'path-map.json');

export const GUIDE_ROOTS = [
  path.join(ROOT, 'docs', 'guides'),
  path.join(ROOT, 'docs', 'i18n', 'en', 'guides'),
  path.join(ROOT, 'docs', 'i18n', 'vi', 'guides'),
];

/** Load path-map.json and return lookup tables. */
export async function loadPathMap(mapFile = PATH_MAP_FILE) {
  const raw = JSON.parse(await readFile(mapFile, 'utf8'));
  const files = raw.files ?? [];
  const directories = raw.directories ?? [];
  const zhToEn = new Map(files.map((e) => [e.zh, e.en]));
  const enToZh = new Map(files.map((e) => [e.en, e.zh]));
  const dirZhToEn = new Map(directories.map((e) => [e.zh, e.en]));
  return { raw, files, directories, zhToEn, enToZh, dirZhToEn };
}

/** Remap a guides-relative path using longest directory prefix + file map. */
export function remapGuidesRelative(relPath, { zhToEn, dirZhToEn }) {
  const norm = relPath.replaceAll('\\', '/');
  if (zhToEn.has(norm)) return zhToEn.get(norm);

  // Directory-only path
  if (dirZhToEn.has(norm)) return dirZhToEn.get(norm);

  // Try translating each segment via directory map + file map for the leaf
  const parts = norm.split('/');
  const out = [];
  for (let i = 0; i < parts.length; i++) {
    const prefix = parts.slice(0, i + 1).join('/');
    if (i === parts.length - 1 && zhToEn.has(prefix)) {
      return zhToEn.get(prefix);
    }
    if (dirZhToEn.has(prefix)) {
      // Replace accumulated path with mapped prefix, continue from next segment
      const mapped = dirZhToEn.get(prefix).split('/');
      out.length = 0;
      out.push(...mapped);
      continue;
    }
    // Fall back: if this single segment is a known dir name
    const seg = parts[i];
    let mappedSeg = null;
    for (const [zh, en] of dirZhToEn) {
      if (zh === seg || zh.endsWith('/' + seg)) {
        mappedSeg = en.includes('/') ? en.split('/').pop() : en;
        break;
      }
    }
    out.push(mappedSeg ?? seg);
  }
  return out.join('/');
}

/** category top-level dir → public URL prefix (matches CATEGORY_URL_PREFIX). */
export const CATEGORY_URL_PREFIX = {
  prerequisites: '/prerequisites',
  'cuda-optimization': '/cuda',
  'distributed-training': '/distributed',
  'inference-optimization': '/inference',
};

/**
 * Build Astro redirects from Chinese guide ids to English ones.
 * Default lang (en) has no prefix; zh/vi use /zh and /vi.
 */
export function buildRedirects(files, locales = ['', '/zh', '/vi']) {
  const redirects = {};
  for (const { zh, en } of files) {
    const fromId = zh.replace(/\.md$/, '');
    const toId = en.replace(/\.md$/, '');
    if (fromId === toId) continue;
    const top = en.split('/')[0];
    const modPrefix = CATEGORY_URL_PREFIX[top] ?? null;
    for (const loc of locales) {
      redirects[`${loc}/guides/${fromId}`] = `${loc}/guides/${toId}`;
      if (modPrefix) {
        redirects[`${loc}${modPrefix}/${fromId}`] = `${loc}${modPrefix}/${toId}`;
      }
    }
  }
  return redirects;
}
