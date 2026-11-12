import type { CollectionEntry } from 'astro:content';

type Guide = CollectionEntry<'guides'>;

export interface ChapterGroup {
  /** The chapter entry itself (第N章 file) */
  chapter: Guide;
  /** Child articles belonging to this chapter */
  children: Guide[];
}

export interface GroupedResult {
  chapters: ChapterGroup[];
  /** Articles that don't belong to any chapter */
  ungrouped: Guide[];
}

/**
 * Detect chapter files by file name ("第N章-....md"), not by title, so grouping still works
 * when the title is translated.
 */
const CHAPTER_FILE_RE = /(?:^|\/)第(\d+)章[^/]*$/;

/** Check if an item is a chapter file */
function isChapterItem(item: Guide): boolean {
  return CHAPTER_FILE_RE.test(item.id);
}

/** Extract chapter number from the file name */
function getChapterNumber(item: Guide): number {
  const m = item.id.match(CHAPTER_FILE_RE);
  return m ? parseInt(m[1], 10) : 0;
}

/**
 * Group a flat list of articles into chapter groups.
 * - Chapter files (title matches "第N章") become group headers.
 * - Articles with a `chapter` field are nested under the matching chapter.
 * - Articles without a `chapter` field and non-chapter files go to `ungrouped`.
 */
export function groupByChapter(items: Guide[]): GroupedResult {
  const sorted = [...items].sort((a, b) => a.data.order - b.data.order);

  // Separate chapter entries and regular articles
  const chapterEntries = sorted.filter(isChapterItem);
  const regularArticles = sorted.filter((item) => !isChapterItem(item));

  // Build a map: chapterNumber -> ChapterGroup
  const chapterMap = new Map<number, ChapterGroup>();
  for (const ch of chapterEntries) {
    const num = getChapterNumber(ch);
    chapterMap.set(num, { chapter: ch, children: [] });
  }

  const ungrouped: Guide[] = [];

  for (const article of regularArticles) {
    const chNum = article.data.chapter;
    if (chNum !== undefined && chapterMap.has(chNum)) {
      chapterMap.get(chNum)!.children.push(article);
    } else {
      ungrouped.push(article);
    }
  }

  // Sort chapters by their chapter number
  const chapters = [...chapterMap.values()].sort(
    (a, b) => getChapterNumber(a.chapter) - getChapterNumber(b.chapter)
  );

  return { chapters, ungrouped };
}

/**
 * Flatten chapter groups into reading order:
 * ungrouped first, then chapter1 → children → chapter2 → children → ...
 */
export function flattenChapterOrder(items: Guide[]): Guide[] {
  const { chapters, ungrouped } = groupByChapter(items);
  const result: Guide[] = [...ungrouped];
  for (const group of chapters) {
    result.push(group.chapter);
    result.push(...group.children);
  }
  return result;
}
