// Label segments using OpenAI with strong pre/post rules, richer context (neighbors + TOC), and stratified random sampling.
// Requires OPENAI_API_KEY. Writes data/segments-labeled-openai.ndjson.

import fs from 'node:fs/promises';
import path from 'node:path';

const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const MAX_PAGES = process.env.MAX_PAGES ? Number(process.env.MAX_PAGES) : null; // pages to label in this run
const PAGES_PER_BOOK = process.env.PAGES_PER_BOOK ? Number(process.env.PAGES_PER_BOOK) : 20; // pages per book (front/back + interior)
const NEIGHBOR_PAGES = 1; // include prev/next text blocks for context
const MAX_SEGMENTS_PER_PAGE = 50;
const MAX_TOC_ENTRIES = 20;
const QUIET = process.env.QUIET === '1';
const FORCE_BOOKS_ENV = process.env.FORCE_BOOKS || ''; // comma-separated basenames to force include
const FORCE_RICH_ENV =
  process.env.FORCE_RICH ||
  [
    // figure/table rich
    'OrganicChemistryOpenStax-full',
    'BiochemistryFreeForAllAhernRajagopalAndTan-full',
    'BookGeneralChemistryAnAtomsFirstApproachHalpern-full',
    'Chemistry1eOpenSTAX-full',
    'Chemistry2eOpenStax-full',
    'OrganicChemistryLabTechniquesNichols-full',
    'OrganicChemistryMorschEtAl-full',
    // exercises/questions
    'ChemistryForChangingTimesHillAndMcCreary-full',
    'ChemistryAtomsFirst2eOpenStax-full',
    'ChemistryAtomsFirst1eOpenSTAX-full',
    'ChemistryForAlliedHealthSoult-full',
    'APEnvironmentalScience-full',
    // appendices/glossary/bibliography
    'BotanyLabManualMorrow-full',
    'BasicPrinciplesOfOrganicChemistryRobertsAndCaserio-full',
    'OrganicChemistryOpenStax-full',
    'BookBasicCellAndMolecularBiologyBergtrom-full',
    'BookComputationalBiologyGenomesNetworksAndEvolutionKellisEtAl-full',
    'BookCellsMoleculesAndMechanismsWong-full'
  ].join(',');

if (!API_KEY) {
  console.error('Set OPENAI_API_KEY in your environment.');
  process.exit(1);
}

const DATA_DIR = path.join(process.cwd(), 'data');
const SRC = path.join(DATA_DIR, 'segments.ndjson');
const OUT = path.join(DATA_DIR, 'segments-labeled-openai.ndjson');
const TAXONOMY = JSON.parse(await fs.readFile(path.join(DATA_DIR, 'label-taxonomy.json'), 'utf-8')).labels;
const OUTPUT_ROOT = path.join(process.cwd(), 'textbooks', 'output');

const shuffle = (arr) => {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

const isLikelyTOC = (t) => {
  if (t.includes('...')) return true;
  if (t.match(/\.+\s*\d{1,4}$/)) return true;
  return false;
};
const extractTrailingPage = (t) => {
  const m = t.match(/\.+\s*(\d{1,4})\s*$/);
  return m ? Number(m[1]) : null;
};

const isTocOutline = (t) => {
  if (!t) return false;
  if (t.length > 150) return false;
  if (/^\d+(\.\d+)+/.test(t.trim())) return true;
  if (/chapter\s*\d+/i.test(t) && t.includes(':')) return true;
  return false;
};

const buildTOCMap = (segments) => {
  const map = new Map();
  segments.forEach((s) => {
    const text = (s.text || '').trim();
    if (isLikelyTOC(text) || isTocOutline(text)) {
      const pageNum = extractTrailingPage(text);
      if (pageNum !== null) {
        const title = text.replace(/\.+\s*\d+$/, '').trim();
        if (!map.has(s.book)) map.set(s.book, { entries: [] });
        map.get(s.book).entries.push({ title, page: pageNum });
      }
    }
  });
  return map;
};

const positionBucket = (y, pageHeight) => {
  const r = y / Math.max(pageHeight ?? 2000, 1);
  if (r < 0.33) return 'top';
  if (r < 0.66) return 'middle';
  return 'bottom';
};

const rareContentHit = (text = '') => {
  const t = text.toLowerCase();
  return (
    t.includes('figure ') ||
    t.includes('fig.') ||
    t.includes('table ') ||
    t.includes('exhibit ') ||
    t.includes('equation') ||
    t.includes('formula') ||
    t.includes('question') ||
    t.includes('quiz') ||
    t.includes('exercise') ||
    t.includes('problem') ||
    t.includes('glossary') ||
    t.includes('appendix') ||
    t.includes('bibliograph') ||
    t.includes('reference')
  );
};

const normalizeLabel = (raw) => {
  const label = (raw || '').trim();
  if (label === 'title') return 'frontmatter.title';
  if (label === 'callout') return 'callout.note';
  if (label === 'list') return 'list.bullet';
  if (TAXONOMY.includes(label)) return label;
  return 'paragraph.body';
};

const collectTargetedPages = (byPage) => {
  const candidates = [];
  const perBookLimit = 60;
  const seenPerBook = new Map();
  for (const [key, segs] of byPage.entries()) {
    const [book, pageStr] = key.split('::');
    const pageNum = Number(pageStr);
    const hits = segs.some((s) => rareContentHit(s.text));
    const headingLike = segs.some((s) => (s.box?.fontSize ?? 0) >= 34 && (s.box?.lines ?? 1) <= 2 && (s.text || '').length < 120);
    if (!hits && !headingLike) continue;
    const count = seenPerBook.get(book) || 0;
    if (count >= perBookLimit) continue;
    candidates.push(key);
    seenPerBook.set(book, count + 1);
  }
  return candidates;
};

const basePrompt = (labels) => `
You label textbook page segments into exactly one label per segment.
Labels: ${labels.join(', ')}
Rules (hard):
- Never default to paragraph.body if a more specific label fits; when cues exist, choose headings/figures/tables/questions/glossary/appendix/bibliography over generic labels.
- Single numbers at page edges -> header/footer (not title).
- Long prose (multi-line or >120 chars) -> paragraph.body (not list/callout).
- TOC rows (dot leaders, trailing page numbers, or outlines like "Chapters ... 1.1:", "1.2:" etc.) -> toc.entry.
- Author/affiliation/brand: names/commas/affiliations/brands (OpenStax, CK-12, LibreTexts) -> frontmatter.author or metadata.link (not title).
- Headings: chapter.heading/section.heading/subsection.heading; Titles: frontmatter.title/subtitle.
- Do NOT emit metadata.link unless there is an actual URL/doi/handle/@go/page or an explicit license/brand attribution; otherwise prefer content labels.
- Short, high-font lines near the top/middle that look like section names -> section.heading/subsection.heading (not paragraph.body).
- If text mentions Figure/Fig./Table/Exhibit/Equation/Formula, use figure.caption/table.title/table.body/formula.
- If text mentions Question/Quiz/Exercise/Problem, use question.prompt/choice/answer or callout.exercise.
- If text mentions Glossary/Appendix/Bibliography/References, use glossary.* / appendix.* / bibliography.entry.
- Lists: bullets/numbers -> list.bullet or list.numbered; glossary term/body for definition lists.
- Callouts: boxed/asides only -> callout.note/warning/example/exercise (do not use for normal links).
- Questions: question.prompt/choice/answer for assessments.
- Media: figure.image/figure.caption; table.title/table.body.
- Glossary/appendix/bibliography/footer/header/nav.breadcrumb/metadata.link as appropriate.
Return a JSON array, index-aligned with inputs: [{"label": "<one label>"}...]
`;

const systemPrompt = basePrompt(TAXONOMY);

const RARE_LABELS = [
  'figure.image',
  'figure.caption',
  'table.title',
  'table.body',
  'formula',
  'question.prompt',
  'question.choice',
  'question.answer',
  'callout.exercise',
  'callout.warning',
  'callout.example',
  'glossary.term',
  'glossary.definition',
  'appendix.heading',
  'appendix.body',
  'bibliography.entry',
  'chapter.heading',
  'section.heading',
  'subsection.heading',
  'frontmatter.title',
  'frontmatter.author',
  'toc.entry',
  'list.bullet',
  'paragraph.body'
];

const rarePrompt = basePrompt(RARE_LABELS) + `
Examples (informal):
- "Figure 3.2: Energy diagram..." -> figure.caption
- Large box with little/no text near images -> figure.image or table.body
- "Table 5.1: Reaction yields" -> table.title; rows of values -> table.body
- "Question 4. (A)...", "Exercises", "Problems" -> question.prompt/choice/answer or callout.exercise
- "Glossary", "Term — definition" -> glossary.term / glossary.definition
- "Appendix A", "Appendix: ..." -> appendix.heading/appendix.body
- "References" / numbered citations -> bibliography.entry
`;

const looksLikeTocLink = (text) => {
  if (isLikelyTOC(text)) return true;
  if (text.length < 80 && text.match(/\.+\s*\d{1,4}$/)) return true;
  return false;
};

const segmentCues = (seg, pageHeight) => {
  const cues = [];
  const text = (seg.text || '').trim();
  const lower = text.toLowerCase();
  const y = seg.box?.y ?? 0;
  const h = seg.box?.height ?? 0;
  const w = seg.box?.width ?? 0;
  const fontSize = seg.box?.fontSize ?? 0;
  const lines = seg.box?.lines ?? 1;
  const area = (w || 1) * (h || 1);
  const aspect = h > 0 ? w / h : 0;
  const pos = positionBucket(y, pageHeight ?? 2000);
  if (pos) cues.push(`pos:${pos}`);
  if (fontSize) cues.push(`font:${fontSize}`);
  cues.push(`lines:${lines}`);
  cues.push(`area:${Math.round(area)}`);
  cues.push(`aspect:${aspect.toFixed(2)}`);
  if (rareContentHit(text)) cues.push('keywords:rare');
  if (fontSize >= 18 && lines <= 2 && text.length < 160) cues.push('heading_like');
  return cues.join('|');
};

const loadSegmentsByPage = async () => {
  const content = await fs.readFile(SRC, 'utf-8');
  const items = content
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l));
  const byPage = new Map();
  items.forEach((seg) => {
    const key = `${seg.book}::${seg.page}`;
    if (!byPage.has(key)) byPage.set(key, []);
    byPage.get(key).push(seg);
  });
  return { byPage, items };
};

const loadPageTexts = async (book) => {
  const dir = path.join(OUTPUT_ROOT, book);
  const pagesTextPath = path.join(dir, 'pages-text.json');
  try {
    const data = JSON.parse(await fs.readFile(pagesTextPath, 'utf-8'));
    const map = new Map();
    data.forEach((p) => map.set(Number(p.page), p.blocks || []));
    return map;
  } catch {
    return new Map();
  }
};

const gatherContext = async (book, pageNum, pageHeight, tocMap) => {
  const pageTexts = await loadPageTexts(book);
  const neighbors = [];
  for (let d = -NEIGHBOR_PAGES; d <= NEIGHBOR_PAGES; d++) {
    if (d === 0) continue;
    const blocks = pageTexts.get(pageNum + d);
    if (blocks && blocks.length) {
      neighbors.push({ page: pageNum + d, text: blocks.join(' ').slice(0, 400) });
    }
  }
  const tocEntries = (tocMap?.entries || [])
    .filter((e) => Math.abs((e.page ?? 0) - pageNum) <= 5)
    .slice(0, MAX_TOC_ENTRIES);
  return { neighbors, tocEntries };
};

const preLabel = (seg, prevSeg, pageHeight) => {
  const text = (seg.text || '').trim();
  const lower = text.toLowerCase();
  const y = seg.box?.y ?? 0;
  const fontSize = seg.box?.fontSize ?? 0;
  const topRatio = pageHeight ? y / pageHeight : 0;
  const lines = seg.box?.lines ?? 1;
  const isLikelyNameLine =
    text.length < 120 &&
    (/[A-Z][a-z]+\s+[A-Z][a-z]+/.test(text) || /university|college|institute/i.test(text) || text.includes(',') || text.includes('&'));
  const isBrandShort = text.length < 40 && /(openstax|ck-?12|libretexts)/i.test(text);
  const isUrl = /https?:\/\//i.test(text) || /@go\/page/i.test(text) || /\.libretexts\.org/i.test(text);
  const isLicenseAttribution = /is shared under/i.test(text) && /libretexts/i.test(text);
  const isBrandAttribution =
    /libretexts\s+libretexts/i.test(text) ||
    /anonymous\s+libretexts/i.test(text) ||
    /textmap\s+organized/i.test(text) ||
    /by\s+openstax/i.test(text) ||
    /map:\s*chemistry.*openstax/i.test(text) ||
    /openstax.*textmap/i.test(text) ||
    /flowers,?\s+theopold,?\s+and\s+langley/i.test(text) ||
    /paul\s+flowers.*openstax/i.test(text) ||
    /de\s+kluyver.*libretexts/i.test(text) ||
    /^de\s+kluyver\s+libretexts$/i.test(text);

  if (text.startsWith('?') || text.includes('?   ')) return 'list.bullet';
  if (text.match(/^\d{1,4}$/)) return y > 1200 ? 'footer' : 'header';
  if (rareContentHit(text)) return null;
  if (fontSize >= 18 && lines <= 2 && text.length < 160) return 'section.heading';
  if (text.length > 120 || lines > 2) return 'paragraph.body';
  if (isLikelyTOC(text)) return 'toc.entry';
  if (looksLikeTocLink(text)) return 'toc.entry';
  if (isTocOutline(text)) return 'toc.entry';
  if (isLicenseAttribution) return 'metadata.link';
  if (isBrandAttribution) return 'metadata.link';
  if (isUrl) return 'metadata.link';
  if (
    text.length < 80 &&
    (lower.includes('libretexts') || lower.match(/\b(ball|harvey|malik|moore|arif|hampton|university)\b/))
  ) {
    if (prevSeg && (prevSeg.label === 'title' || prevSeg.label === 'frontmatter.title')) {
      return 'frontmatter.author';
    }
    return 'frontmatter.author';
  }
  if (isLikelyNameLine && topRatio < 0.6 && lines <= 3) {
    if (prevSeg && (prevSeg.label === 'title' || prevSeg.label === 'frontmatter.title')) {
      return 'frontmatter.author';
    }
    return 'frontmatter.author';
  }
  if (isBrandShort && lines <= 2) {
    return 'metadata.link';
  }
  if (fontSize >= 40 && topRatio < 0.45 && text.length < 80 && lines <= 2) {
    return 'frontmatter.title';
  }
  if (/^[-•*0-9]/.test(text)) return 'list.bullet';
  return null;
};

const postFix = (label, seg, prevSeg) => {
  const text = (seg.text || '').trim();
  const lower = text.toLowerCase();
  const y = seg.box?.y ?? 0;
  const fontSize = seg.box?.fontSize ?? 0;
  const lines = seg.box?.lines ?? 1;
  const isLikelyNameLine =
    text.length < 120 &&
    (/[A-Z][a-z]+\s+[A-Z][a-z]+/.test(text) || /university|college|institute/i.test(text) || text.includes(',') || text.includes('&'));
  const isBrandShort = text.length < 40 && /(openstax|ck-?12|libretexts)/i.test(text);
  const isUrl = /https?:\/\//i.test(text) || /@go\/page/i.test(text) || /\.libretexts\.org/i.test(text);
  const isLicenseAttribution = /is shared under/i.test(text) && /libretexts/i.test(text);
  const isBrandAttribution =
    /libretexts\s+libretexts/i.test(text) ||
    /anonymous\s+libretexts/i.test(text) ||
    /textmap\s+organized/i.test(text) ||
    /by\s+openstax/i.test(text) ||
    /map:\s*chemistry.*openstax/i.test(text) ||
    /openstax.*textmap/i.test(text);

  // Keyword-first routing for rare labels
  if (/\bfigure\b|\bfig\.?\b/i.test(text)) return label === 'figure.image' ? label : 'figure.caption';
  if (/\btable\b|\bexhibit\b/i.test(text)) return label === 'table.body' ? label : 'table.title';
  if (/\bequation\b|\bformula\b/i.test(text)) return 'formula';
  if (/\bquestion\b|\bquiz\b|\bexercise\b|\bproblem\b/i.test(text)) {
    if (label.startsWith('question.')) return label;
    return 'question.prompt';
  }
  if (/\bglossary\b/.test(lower)) {
    if (label.startsWith('glossary.')) return label;
    return 'glossary.term';
  }
  if (/\bappendix\b/.test(lower)) {
    if (label.startsWith('appendix.')) return label;
    return 'appendix.heading';
  }
  if (/\bbibliograph|references\b/i.test(text)) {
    return 'bibliography.entry';
  }

  if (text.startsWith('?') || text.includes('?   ')) return 'list.bullet';
  if (text.match(/^\d{1,4}$/)) return y > 1200 ? 'footer' : 'header';
  if (isLikelyTOC(text)) return 'toc.entry';
  if (looksLikeTocLink(text)) return 'toc.entry';
  if (isTocOutline(text)) return 'toc.entry';
  if (isLicenseAttribution) return 'metadata.link';
  if (isBrandAttribution) return 'metadata.link';
  if (isUrl) return 'metadata.link';
  const rareLabels = new Set([
    'figure.image',
    'figure.caption',
    'table.title',
    'table.body',
    'formula',
    'question.prompt',
    'question.choice',
    'question.answer',
    'callout.exercise',
    'callout.warning',
    'callout.example',
    'glossary.term',
    'glossary.definition',
    'appendix.heading',
    'appendix.body',
    'bibliography.entry'
  ]);
  if (rareLabels.has(label)) return label;
  if (label === 'metadata.link' && !isUrl && !isLicenseAttribution && !isBrandAttribution) {
    return 'paragraph.body';
  }
  if (/^[-•*0-9]/.test(text) && text.length < 120) return 'list.bullet';
  if (text.length > 120 && label === 'list') return 'paragraph.body';
  if (
    label === 'paragraph.body' &&
    fontSize >= 18 &&
    lines <= 2 &&
    text.length < 160
  ) {
    return 'section.heading';
  }
  if (
    (label === 'title' || label === 'heading') &&
    text.length < 80 &&
    (lower.includes('libretexts') || lower.match(/\b(ball|harvey|malik|moore|arif|hampton|university)\b/))
  ) {
    return 'frontmatter.author';
  }
  if (isLikelyNameLine && y < 1400 && lines <= 3) {
    return 'frontmatter.author';
  }
  if (isBrandShort && lines <= 2) {
    return 'metadata.link';
  }
  if (label === 'callout' && !text.match(/(note|warning|example|exercise)/i) && text.length < 80) {
    return 'toc.entry';
  }
  if (
    (label === 'toc.entry' || label === 'heading') &&
    fontSize >= 40 &&
    lines <= 2 &&
    text.length < 80 &&
    y < 1200
  ) {
    return 'frontmatter.title';
  }
  return label;
};

const callOpenAI = async (segments, pageHeight, tocSummary, neighborSummary, prompt) => {
  const trimmed = segments.slice(0, MAX_SEGMENTS_PER_PAGE);
  const userContent =
    `TOC (nearby): ${tocSummary || 'none'}\nNeighbors:\n${neighborSummary || 'none'}\nSegments:\n` +
    trimmed
      .map((s, idx) => {
        const pos = positionBucket(s.box?.y ?? 0, pageHeight ?? 2000);
        const text = (s.text || '').trim().slice(0, 500);
        const prevText = idx > 0 ? (trimmed[idx - 1].text || '').trim().slice(0, 120) : '';
        const cues = segmentCues(s, pageHeight ?? 2000);
        return `${idx}: [pos=${pos}, font=${s.box?.fontSize ?? ''}, w=${s.box?.width ?? ''}, h=${
          s.box?.height ?? ''
        }, cues=${cues}, prev="${prevText}"] "${text}"`;
      })
      .join('\n');

  const body = {
    model: MODEL,
    messages: [
      { role: 'system', content: prompt },
      { role: 'user', content: userContent }
    ],
    response_format: { type: 'json_object' }
  };

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`OpenAI error ${res.status}: ${txt}`);
  }
  const json = await res.json();
  const content = json.choices?.[0]?.message?.content;
  return JSON.parse(content);
};

const labelPage = async (book, pageNum, segments, pageHeight, tocContext, neighbors) => {
  const preLabeled = new Array(segments.length).fill(null);
  const ambiguous = [];
  segments.forEach((s, idx) => {
    const prev = idx > 0 ? segments[idx - 1] : null;
    const lbl = preLabel(s, prev, pageHeight);
    if (lbl) preLabeled[idx] = lbl;
    else ambiguous.push({ seg: s, idx });
  });

  if (ambiguous.length === 0) {
    return segments.map((s, i) => ({ label: postFix(preLabeled[i] ?? s.label, s, i > 0 ? segments[i - 1] : null) }));
  }

  const tocEntries = (tocContext?.entries || []).slice(0, MAX_TOC_ENTRIES);
  const tocSummary = tocEntries.map((e) => `${e.title} -> ${e.page}`).join('; ');
  const neighborSummary = (neighbors || [])
    .map((n) => `p${n.page}: ${n.text}`)
    .join('\n')
    .slice(0, 1200);

  const parsed = await callOpenAI(segments, pageHeight, tocSummary, neighborSummary, systemPrompt);

  const merged = segments.map((s, i) => {
    if (preLabeled[i]) return { label: preLabeled[i] };
    const modelLabel = parsed[i]?.label ?? s.label;
    return { label: modelLabel };
  });
  // second pass for rare-cue segments: re-ask with rare-focused prompt/labels
  const rareIdx = segments
    .map((s, i) => ({ s, i }))
    .filter(({ s }) => rareContentHit(s.text) || ((s.box?.fontSize ?? 0) >= 18 && (s.box?.lines ?? 1) <= 2 && (s.text || '').length < 160))
    .map(({ i }) => i);
  if (rareIdx.length) {
    const rareSegments = rareIdx.map((i) => segments[i]);
    const rareParsed = await callOpenAI(rareSegments, pageHeight, tocSummary, neighborSummary, rarePrompt);
    rareIdx.forEach((idx, local) => {
      merged[idx] = { label: rareParsed[local]?.label ?? merged[idx].label };
    });
  }
  return merged.map((l, i) => {
    const post = postFix(normalizeLabel(l.label), segments[i], i > 0 ? segments[i - 1] : null);
    return { label: normalizeLabel(post) };
  });
};

const main = async () => {
  const { byPage, items } = await loadSegmentsByPage();
  const tocMap = buildTOCMap(items);
  const labeledLines = [];

  // If SAMPLE_WHOLE_BOOKS is set, process that many full books end-to-end. Otherwise, stratified sampling.
  const SAMPLE_WHOLE_BOOKS = process.env.SAMPLE_WHOLE_BOOKS
    ? Number(process.env.SAMPLE_WHOLE_BOOKS)
    : 0;

  // Stratified sampling: choose a wide set of books evenly spaced (target ~80), plus force-include list, then front/back plus random interior pages.
  const pagesByBook = new Map();
  for (const [key] of byPage.entries()) {
    const [book] = key.split('::');
    if (!pagesByBook.has(book)) pagesByBook.set(book, []);
    pagesByBook.get(book).push(key);
  }
  const forcedBooks = [
    ...FORCE_BOOKS_ENV.split(',').map((s) => s.trim()).filter(Boolean),
    ...FORCE_RICH_ENV.split(',').map((s) => s.trim()).filter(Boolean)
  ].filter(Boolean);
  const allBooks = Array.from(pagesByBook.keys()).sort();
  const selectedBooks = new Set();
  if (forcedBooks.length) {
    forcedBooks.forEach((b) => {
      if (pagesByBook.has(b)) selectedBooks.add(b);
    });
  } else {
    // pick ~80 books evenly spread when nothing forced
    const takeBooks = Math.min(80, allBooks.length);
    for (let i = 0; i < takeBooks; i++) {
      const idx = Math.floor((i * allBooks.length) / takeBooks);
      selectedBooks.add(allBooks[idx]);
    }
  }

  let selection = [];
  let targetedPages = collectTargetedPages(byPage);
  if (forcedBooks.length) {
    targetedPages = targetedPages.filter((k) => forcedBooks.includes(k.split('::')[0]));
  }
  if (SAMPLE_WHOLE_BOOKS > 0) {
    const books = Array.from(new Set([...forcedBooks, ...selectedBooks]));
    const pickBooks = books.slice(0, SAMPLE_WHOLE_BOOKS);
    pickBooks.forEach((book) => {
      const pages = pagesByBook.get(book) || [];
      selection.push(...pages);
    });
  } else {
    // prioritize targeted pages; if we have MAX_PAGES, fill from targeted first.
    if (targetedPages.length) {
      selection.push(...targetedPages);
    }
    const booksToUse = forcedBooks.length ? forcedBooks : Array.from(selectedBooks);
    booksToUse.forEach((book) => {
      const keys = pagesByBook.get(book) || [];
      if (!keys.length) return;
      const pageNums = keys.map((k) => Number(k.split('::')[1])).sort((a, b) => a - b);
      const minPage = pageNums[0];
      const maxPage = pageNums[pageNums.length - 1];
      const interior = pageNums.slice(1, -1);
      const pickedInterior = shuffle(interior).slice(0, Math.max(0, PAGES_PER_BOOK - 2));
      const targetPages = [minPage, maxPage, ...pickedInterior];
      targetPages.forEach((p) => selection.push(`${book}::${p}`));
    });
  }

  const finalList = MAX_PAGES ? selection.slice(0, MAX_PAGES) : selection;

  let processedPages = 0;
  for (const key of finalList) {
    const segments = byPage.get(key);
    if (!segments || !Array.isArray(segments) || segments.length === 0) {
      console.warn(`Skipping ${key} (no segments)`);
      continue;
    }
    const [book, pageStr] = key.split('::');
    const pageNum = Number(pageStr);
    if (MAX_PAGES && processedPages >= MAX_PAGES) break;
    if (!QUIET) console.log(`Labeling ${key} (${segments.length} segments)`);
    let labeled = false;
    for (let attempt = 1; attempt <= 3 && !labeled; attempt++) {
      try {
        const ctx = await gatherContext(book, pageNum, segments[0]?.pageHeight, tocMap.get(book));
        const labels = await labelPage(book, pageNum, segments, segments[0]?.pageHeight, tocMap.get(book), ctx.neighbors);
        segments.forEach((seg, idx) => {
          const mapped = labels[idx]?.label ?? seg.label;
          labeledLines.push(JSON.stringify({ ...seg, label: mapped }));
        });
        processedPages += 1;
        labeled = true;
        await new Promise((resolve) => setTimeout(resolve, 200)); // mild rate limit
      } catch (err) {
        const finalAttempt = attempt === 3;
        console.error(`Failed on ${key} (attempt ${attempt}/3)`, err.message);
        if (!finalAttempt) {
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      }
    }
    if (!labeled) continue;
  }
  await fs.writeFile(OUT, labeledLines.join('\n'), 'utf-8');
  console.log(`Wrote ${labeledLines.length} labeled records to ${OUT}`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
