import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const guides = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/guides' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    category: z.enum([
      'learning-path',
      'prerequisites',
      'cuda-optimization',
      'distributed-training',
      'inference-optimization',
      'performance-analysis',
      'course',
    ]),
    order: z.number().default(0),
    chapter: z.number().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    heroImage: z.string().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    ref: z.string().optional(),
  }),
});

const interview = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/interview' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    company: z.string(),
    tier: z.enum(['T0', 'T1', 'T2', 'T3', 'T4', 'T5', '综合']),
    interviewType: z.enum(['实习', '校招', '社招', '未知']).default('未知'),
    round: z.string().optional(),
    order: z.number().default(0),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

// Translations of the collections above, mirrored by path:
// docs/i18n/<lang>/<collection>/<same path as the Chinese source>.md
// Frontmatter holds only the translated fields; everything else comes from the source file.
const translations = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/i18n' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    tags: z.array(z.string()).optional(),
    sourceHash: z.string().optional(),
  }),
});

export const collections = { guides, posts, interview, translations };
