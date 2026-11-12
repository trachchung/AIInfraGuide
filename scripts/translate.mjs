#!/usr/bin/env node
// Translate the Chinese markdown in docs/ into other languages with Claude.
//
// Each translation lands at docs/i18n/<lang>/<same path>.md and records a hash of its source,
// so re-running only translates new or changed articles. The prose is rewritten with the
// humanizer style guide (scripts/humanizer/SKILL.md) so it reads like a person wrote it.
//
// Usage:
//   node scripts/translate.mjs [--lang en,vi] [--concurrency 4] [--effort high] [--force] [--check] [files...]
//
//   files          Source files or directories under docs/ (default: all of docs/guides, docs/interview, docs/posts)
//   --check        List missing or outdated translations and exit 1 if any. Makes no API calls.
//   --force        Translate even when the existing translation is up to date.
//
// Credentials: ANTHROPIC_API_KEY, or a profile from `ant auth login`.

import Anthropic from '@anthropic-ai/sdk';
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const DOCS = path.join(ROOT, 'docs');
const SOURCE_DIRS = ['guides', 'interview', 'posts'];
const LANGS = {
  en: 'English',
  vi: 'Vietnamese',
};

const { values: opts, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    lang: { type: 'string', default: Object.keys(LANGS).join(',') },
    concurrency: { type: 'string', default: '4' },
    effort: { type: 'string', default: 'high' },
    force: { type: 'boolean', default: false },
    check: { type: 'boolean', default: false },
  },
});

const langs = opts.lang.split(',').map((l) => l.trim());
for (const l of langs) if (!LANGS[l]) throw new Error(`Unknown language "${l}". Known: ${Object.keys(LANGS).join(', ')}`);

async function listMarkdown(p) {
  const s = await stat(p);
  if (s.isFile()) return p.endsWith('.md') ? [p] : [];
  const out = [];
  for (const e of await readdir(p, { withFileTypes: true })) out.push(...(await listMarkdown(path.join(p, e.name))));
  return out;
}

const sources = (
  await Promise.all(
    (positionals.length ? positionals.map((p) => path.resolve(p)) : SOURCE_DIRS.map((d) => path.join(DOCS, d))).map(listMarkdown),
  )
)
  .flat()
  .filter((f) => !f.startsWith(path.join(DOCS, 'i18n') + path.sep))
  .sort();

const hashOf = (text) => createHash('sha256').update(text).digest('hex').slice(0, 16);
const targetOf = (src, lang) => path.join(DOCS, 'i18n', lang, path.relative(DOCS, src));

async function existingHash(file) {
  try {
    return (await readFile(file, 'utf8')).match(/^sourceHash:\s*"?([0-9a-f]+)"?\s*$/m)?.[1];
  } catch {
    return undefined;
  }
}

// Work out which (source, language) pairs need translating.
const jobs = [];
for (const src of sources) {
  const text = await readFile(src, 'utf8');
  const hash = hashOf(text);
  for (const lang of langs) {
    const target = targetOf(src, lang);
    const current = await existingHash(target);
    if (opts.force || current !== hash) jobs.push({ src, text, hash, lang, target, stale: current !== undefined });
  }
}

if (opts.check) {
  for (const j of jobs) console.log(`${j.stale ? 'outdated' : 'missing '} ${j.lang} ${path.relative(ROOT, j.src)}`);
  console.log(`${jobs.length} translation(s) missing or outdated.`);
  process.exit(jobs.length ? 1 : 0);
}
if (!jobs.length) {
  console.log('All translations are up to date.');
  process.exit(0);
}

const humanizer = await readFile(path.join(ROOT, 'scripts/humanizer/SKILL.md'), 'utf8');

const systemPrompt = (language) => `You translate articles from AIInfraGuide, an open-source Chinese tutorial on AI infrastructure (GPUs, CUDA kernels, distributed training, LLM inference), into ${language}. Readers are engineers learning the field.

Translate the meaning, not word by word. The result should read as if a ${language}-speaking engineer wrote it for other engineers: clear, plain and specific. While you translate, apply the style guide below in its embedded mode to the ${language} prose. The source is often casual and explains ideas in plain words before giving the terms; keep that voice, the author's opinions and the first person. Do not add facts, summaries, closing remarks or notes of your own, and do not drop any fact, number, caveat or step.

Rules for the markdown:
- Keep the document structure: the same headings in the same order, lists, tables, blockquotes, images and horizontal rules. Headings use sentence case.
- Copy fenced code blocks, inline code, math ($...$ and $$...$$), HTML tags and attributes, URLs, link targets and image paths exactly. You may translate Chinese comments and strings inside code blocks, but change nothing else in them. Mermaid diagrams: translate only the label text.
- Translate link text and image alt text.
- Keep established technical terms in their usual English form (CUDA, kernel, warp, AllReduce, KV Cache, PagedAttention, prefill, decode, tensor parallelism and so on).${
    language === 'Vietnamese'
      ? ' Vietnamese engineers usually keep these English terms; add a Vietnamese gloss in parentheses only the first time a term appears, and only when it helps.'
      : ''
  }
- Product and company names keep their official spelling; use the official English name of Chinese companies when one exists (字节跳动 is ByteDance).
- If the source has a mistake, translate it as written.

Reply in exactly this format, with nothing before or after it:
TITLE: <translated title on one line>
DESCRIPTION: <translated description on one line>
TAGS: <JSON array of translated tags, or [] if the source has none>
---BODY---
<the translated markdown body, without frontmatter>

<style_guide>
${humanizer}
</style_guide>`;

function splitFrontmatter(text) {
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!m) throw new Error('no frontmatter');
  return { frontmatter: m[1], body: m[2] };
}

const count = (s, re) => (s.match(re) ?? []).length;

/** Relative links and image paths still point at the right file from the deeper i18n folder. */
function rebaseRelative(body, src, target) {
  const fix = (url) => {
    if (/^([a-z]+:|\/|#)/i.test(url)) return url;
    const abs = path.resolve(path.dirname(src), url);
    return path.relative(path.dirname(target), abs).split(path.sep).join('/');
  };
  return body
    .replace(/(\]\()([^)\s]+)/g, (_, pre, url) => pre + fix(url))
    .replace(/(\bsrc=")([^"]+)/g, (_, pre, url) => pre + fix(url));
}


function parseReply(reply) {
  const m = reply.match(/^\s*TITLE:\s*(.+)\nDESCRIPTION:\s*(.*)\nTAGS:\s*(.*)\n---BODY---\n([\s\S]*)$/);
  if (!m) throw new Error('reply is not in the expected format');
  const tags = JSON.parse(m[3].trim());
  if (!Array.isArray(tags)) throw new Error('TAGS is not a JSON array');
  return { title: m[1].trim(), description: m[2].trim(), tags, body: m[4].replace(/^\n+/, '') };
}

const client = new Anthropic();

async function translate(job) {
  const { frontmatter, body } = splitFrontmatter(job.text);
  const hasTags = /^tags:/m.test(frontmatter);

  const stream = client.beta.messages.stream({
    model: 'claude-opus-5-5',
    max_tokens: 64000,
    betas: ['server-side-fallback-2026-07-01'],
    fallbacks: 'default',
    thinking: { type: 'adaptive' },
    output_config: { effort: opts.effort },
    system: [{ type: 'text', text: systemPrompt(LANGS[job.lang]), cache_control: { type: 'ephemeral' } }],
    messages: [
      {
        role: 'user',
        content: `Translate this article into ${LANGS[job.lang]}.\n\n<frontmatter>\n${frontmatter}\n</frontmatter>\n\n<body>\n${body}\n</body>`,
      },
    ],
  });
  const message = await stream.finalMessage();
  if (message.stop_reason !== 'end_turn') throw new Error(`stopped with ${message.stop_reason}`);

  const reply = message.content.filter((b) => b.type === 'text').map((b) => b.text).join('');
  const out = parseReply(reply);

  // Cheap guards against a truncated or mangled translation.
  for (const [what, re] of [
    ['code fences', /^\s*```/gm],
    ['$$ math blocks', /\$\$/g],
    ['images', /!\[/g],
  ]) {
    const a = count(body, re);
    const b = count(out.body, re);
    if (a !== b) throw new Error(`${what}: source has ${a}, translation has ${b}`);
  }

  const translatedBody = rebaseRelative(out.body, job.src, job.target);
  const lines = [
    '---',
    `title: ${JSON.stringify(out.title)}`,
    `description: ${JSON.stringify(out.description)}`,
    ...(hasTags ? [`tags: ${JSON.stringify(out.tags)}`] : []),
    `sourceHash: "${job.hash}"`,
    '---',
    '',
  ];
  await mkdir(path.dirname(job.target), { recursive: true });
  await writeFile(job.target, lines.join('\n') + translatedBody.replace(/\s*$/, '\n'));
  return message.usage;
}

console.log(`Translating ${jobs.length} file(s) with concurrency ${opts.concurrency}...`);
let done = 0;
let failed = 0;
const queue = [...jobs];
await Promise.all(
  Array.from({ length: Number(opts.concurrency) }, async () => {
    for (let job; (job = queue.shift()); ) {
      const name = `${job.lang} ${path.relative(DOCS, job.src)}`;
      try {
        const usage = await translate(job);
        done++;
        console.log(`[${done + failed}/${jobs.length}] ok   ${name} (${usage.output_tokens} output tokens)`);
      } catch (err) {
        failed++;
        console.error(`[${done + failed}/${jobs.length}] FAIL ${name}: ${err.message}`);
      }
    }
  }),
);
console.log(`Done: ${done} translated, ${failed} failed.`);
process.exit(failed ? 1 : 0);
