// Apply heuristic relabeling from segmentation outputs to taxonomy labels.
// Writes data/segments-labeled.ndjson.

import fs from 'node:fs/promises';
import path from 'node:path';

const DATA_DIR = path.join(process.cwd(), 'data');
const SRC = path.join(DATA_DIR, 'segments.ndjson');
const OUT = path.join(DATA_DIR, 'segments-labeled.ndjson');
const TAXONOMY = JSON.parse(await fs.readFile(path.join(DATA_DIR, 'label-taxonomy.json'), 'utf-8')).labels;

const loadSegments = async () => {
  const content = await fs.readFile(SRC, 'utf-8');
  return content
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l));
};

const isPageNumber = (t) => t.trim().match(/^\d{1,4}$/);
const isLikelyTOC = (t) => t.includes('...') || t.match(/\.+\s*\d+$/);
const extractTrailingPage = (t) => {
  const m = t.match(/(\d{1,4})\s*$/);
  return m ? Number(m[1]) : null;
};

const buildTOCMap = (segments) => {
  const map = new Map(); // book -> [{title,page}]
  segments.forEach((s) => {
    const text = s.text.trim();
    if (isLikelyTOC(text)) {
      const pageNum = extractTrailingPage(text);
      if (pageNum !== null) {
        const title = text.replace(/\.+\s*\d+$/, '').trim();
        const entry = { title, page: pageNum };
        if (!map.has(s.book)) map.set(s.book, []);
        map.get(s.book).push(entry);
      }
    }
  });
  return map;
};

const mapLabel = (seg, tocMap) => {
  const t = seg.text?.trim() ?? '';
  const lower = t.toLowerCase();
  const hasDots = t.includes('....') || t.includes('..');
  const looksList = /^[-•\d]/.test(t);
  const short = t.length < 80;

  // Page number near edges -> footer/header
  if (isPageNumber(t)) return seg.box?.y > 1200 ? 'footer' : 'header';

  // TOC-like
  if (isLikelyTOC(t)) return 'toc.entry';

  if (t.startsWith('Chapter') || t.match(/^chapter\s+\d+/i)) return 'chapter.heading';
  if (lower.startsWith('appendix')) return 'appendix.heading';
  if (looksList) return 'list.bullet';
  if (lower.includes('supplemental modules')) return 'toc.entry';
  if (lower.includes('exercise') || lower.includes('question')) return 'question.prompt';
  if (lower.includes('figure') || seg.label === 'callout') return 'figure.caption';
  if (seg.label === 'title') return 'frontmatter.title';
  if (seg.label === 'subtitle') return 'frontmatter.subtitle';
  if (seg.label === 'heading') return 'section.heading';
  if (seg.label === 'list') return 'list.bullet';
  if (seg.label === 'footer') return 'footer';

  // TOC proximity to set headings
  const pageEntries = tocMap.get(seg.book) || [];
  const maybeEntry = pageEntries.find((e) => e.page === seg.page || e.page === seg.page - 1);
  if (maybeEntry && t.length < 120) return 'section.heading';

  return 'paragraph.body';
};

const main = async () => {
  const segments = await loadSegments();
  const tocMap = buildTOCMap(segments);
  const labeled = segments.map((seg) => {
    const newLabel = mapLabel(seg, tocMap);
    if (!TAXONOMY.includes(newLabel)) return seg;
    return { ...seg, label: newLabel };
  });
  const ndjson = labeled.map((l) => JSON.stringify(l)).join('\n');
  await fs.writeFile(OUT, ndjson, 'utf-8');
  console.log(`Wrote ${labeled.length} labeled records to ${OUT}`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
