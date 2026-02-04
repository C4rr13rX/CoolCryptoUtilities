// Verify page outputs and rewrite pages.json with full stats:
// - totalPages from PDF
// - processedPages from page-*.json count
// - nullPages from pages with no boxes
// - usablePages = processedPages - nullPages

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';

const TEXTBOOK_DIR = path.join(process.cwd(), 'textbooks');
const OUTPUT_DIR = path.join(process.cwd(), 'textbooks', 'output');
const WORKER_SRC = path.join(process.cwd(), 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.min.mjs');

pdfjsLib.GlobalWorkerOptions.workerSrc = pathToFileURL(WORKER_SRC).href;

const listDirs = async (dir) => {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  return entries.filter((e) => e.isDirectory()).map((e) => e.name);
};

const readPageFiles = async (dir) => {
  const entries = await fs.readdir(dir);
  const pageFiles = entries.filter((name) => /^page-(\d+)\.json$/i.test(name));
  let nullPages = 0;
  let maxPage = 0;
  for (const name of pageFiles) {
    try {
      const raw = await fs.readFile(path.join(dir, name), 'utf-8');
      const parsed = JSON.parse(raw);
      const boxes = Array.isArray(parsed.boxes) ? parsed.boxes : [];
      if (boxes.length === 0) nullPages += 1;
      const match = name.match(/^page-(\d+)\.json$/i);
      if (match) {
        const num = Number(match[1]);
        if (!Number.isNaN(num)) maxPage = Math.max(maxPage, num);
      }
    } catch {
      // ignore malformed page file
    }
  }
  return { count: pageFiles.length, nullPages, maxPage };
};

const main = async () => {
  const dirs = await listDirs(OUTPUT_DIR);
  let ok = 0;
  let mismatch = 0;
  let skipped = 0;

  for (const dir of dirs) {
    const pdfPath = path.join(TEXTBOOK_DIR, `${dir}.pdf`);
    const pagesJsonPath = path.join(OUTPUT_DIR, dir, 'pages.json');
    try {
      const pdfData = new Uint8Array(await fs.readFile(pdfPath));
      const doc = await pdfjsLib.getDocument({ data: pdfData }).promise;
      const totalPages = doc.numPages;
      await doc.destroy();

      const { count: processedPages, nullPages, maxPage } = await readPageFiles(path.join(OUTPUT_DIR, dir));
      const usablePages = Math.max(0, processedPages - nullPages);
      const status = processedPages >= totalPages ? 'completed' : 'incomplete';

      await fs.writeFile(
        pagesJsonPath,
        JSON.stringify(
          {
            status,
            totalPages,
            processedPages,
            nullPages,
            usablePages,
            lastPage: maxPage,
            timestamp: new Date().toISOString()
          },
          null,
          2
        ),
        'utf-8'
      );

      if (processedPages === totalPages) ok += 1;
      else mismatch += 1;
    } catch {
      skipped += 1;
      continue;
    }
  }

  console.log(`Verified: ${ok} fully processed, ${mismatch} incomplete, ${skipped} skipped (missing pdf or pages.json).`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
