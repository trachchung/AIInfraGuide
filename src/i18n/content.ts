import { getCollection, type CollectionEntry } from 'astro:content';
import { DEFAULT_LANG, SOURCE_LANG, type Lang } from './ui';

type Source = 'guides' | 'interview' | 'posts';

/** A Chinese source entry with translated frontmatter and body swapped in when a translation exists. */
export type Localized<C extends Source> = CollectionEntry<C> & { translated: boolean };

// Links in articles to other pages of this site, e.g. https://caomaolufei.github.io/AIInfraGuide/cuda/...
const SITE_LINK = /(href="(?:https:\/\/caomaolufei\.github\.io)?\/AIInfraGuide)\/(?=(?:guides|prerequisites|cuda|distributed|inference|interview|blog|about)[/#?"])/g;

/** Point site links in rendered HTML at the page's language, so readers stay in it. */
function localizeLinks<E extends { rendered?: { html: string } }>(entry: E, lang: Lang): E {
  if (lang === DEFAULT_LANG || !entry.rendered) return entry;
  return { ...entry, rendered: { ...entry.rendered, html: entry.rendered.html.replace(SITE_LINK, `$1/${lang}/`) } };
}

let translationsByPath: Map<string, CollectionEntry<'translations'>> | undefined;

async function translationFor(lang: Lang, filePath: string | undefined) {
  if (!filePath) return undefined;
  translationsByPath ??= new Map((await getCollection('translations')).map((t) => [t.filePath!, t]));
  // docs/guides/x.md -> docs/i18n/<lang>/guides/x.md
  return translationsByPath.get(filePath.replace(/^docs\//, `docs/i18n/${lang}/`));
}

/**
 * Non-draft entries of `collection` in `lang`. Ids, ordering and metadata always come from the
 * Chinese source, so URLs and navigation are the same in every language. Untranslated entries
 * keep the Chinese text and get `translated: false`.
 */
export async function getLocalized<C extends Source>(collection: C, lang: Lang): Promise<Localized<C>[]> {
  const entries = (await getCollection(collection, ({ data }: { data: { draft: boolean } }) => !data.draft)) as CollectionEntry<C>[];
  return Promise.all(
    entries.map(async (entry) => {
      const tr = lang === SOURCE_LANG ? undefined : await translationFor(lang, entry.filePath);
      if (!tr) return localizeLinks({ ...entry, translated: lang === SOURCE_LANG }, lang);
      const { sourceHash: _, ...fields } = tr.data;
      const overrides = Object.fromEntries(Object.entries(fields).filter(([, v]) => v !== undefined));
      return localizeLinks({
        ...entry,
        data: { ...entry.data, ...overrides },
        body: tr.body,
        rendered: tr.rendered,
        filePath: tr.filePath,
        translated: true,
      }, lang);
    }),
  );
}
