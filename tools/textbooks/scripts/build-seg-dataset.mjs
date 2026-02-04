// Build an NDJSON dataset of segment features + labels from segmentation outputs.
// Reads textbooks/output/**/page-*.json and emits data/segments.ndjson with fields:
// { book, page, label, confidence, features: [...], box: {x,y,width,height,fontSize,lines} }

import fs from 'node:fs/promises';
import path from 'node:path';

const OUTPUT_ROOT = path.join(process.cwd(), 'textbooks', 'output');
const DATA_DIR = path.join(process.cwd(), 'data');
const DATA_FILE = path.join(DATA_DIR, 'segments.ndjson');

const ensureDir = async (dir) => fs.mkdir(dir, { recursive: true });

const buildFeatureVector = (box, pageHeight) => {
  const fontSizeNorm = Math.min(box.fontSize / 36, 1);
  const heightNorm = Math.min(box.height / 300, 1);
  const widthNorm = Math.min(box.width / 800, 1);
  const yNorm = Math.max(0, Math.min(1, 1 - box.y / pageHeight));
  const isAllCaps = box.text === box.text.toUpperCase() ? 1 : 0;
  const wordCount = Math.max(1, box.text.split(/\s+/).length);
  const density = Math.min(wordCount / Math.max(1, (box.width * box.height) / 10000), 1);
  const lineCountNorm = Math.min((box.lines ?? 1) / 12, 1);
  return [fontSizeNorm, heightNorm, widthNorm, yNorm, isAllCaps, density, lineCountNorm];
};

const collectPages = async () => {
  const entries = await fs.readdir(OUTPUT_ROOT, { withFileTypes: true });
  const books = entries.filter((e) => e.isDirectory()).map((e) => e.name);
  const pages = [];
  for (const book of books) {
    const dir = path.join(OUTPUT_ROOT, book);
    const files = await fs.readdir(dir);
    const jsons = files.filter((f) => /^page-\d+\.json$/i.test(f));
    for (const file of jsons) {
      const content = JSON.parse(await fs.readFile(path.join(dir, file), 'utf-8'));
      pages.push({ book, ...content });
    }
  }
  return pages;
};

const main = async () => {
  await ensureDir(DATA_DIR);
  const pages = await collectPages();
  const lines = [];
  pages.forEach((page) => {
    const pageHeight = page.pageHeight ?? 2000;
    page.boxes.forEach((box) => {
      lines.push({
        book: page.book,
        page: page.page,
        label: box.label,
        confidence: box.confidence,
        features: buildFeatureVector(box, pageHeight),
        box: {
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height,
          fontSize: box.fontSize,
          lines: box.lines
        },
        text: box.text
      });
    });
  });
  const ndjson = lines.map((l) => JSON.stringify(l)).join('\n');
  await fs.writeFile(DATA_FILE, ndjson, 'utf-8');
  console.log(`Wrote ${lines.length} records to ${DATA_FILE}`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
