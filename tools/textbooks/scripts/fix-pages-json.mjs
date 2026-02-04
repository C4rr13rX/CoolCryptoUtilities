// Rewrite pages.json files to store the total PDF page count (number) instead of an array.
// Uses the original PDFs to read numPages, so partial processing or prior truncation is ignored.

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';

const TEXTBOOK_DIR = path.join(process.cwd(), 'textbooks');
const OUTPUT_DIR = path.join(process.cwd(), 'textbooks', 'output');
const WORKER_SRC = path.join(process.cwd(), 'node_modules', 'pdfjs-dist', 'build', 'pdf.worker.min.mjs');

pdfjsLib.GlobalWorkerOptions.workerSrc = pathToFileURL(WORKER_SRC).href;

const main = async () => {
  const entries = await fs.readdir(TEXTBOOK_DIR);
  const pdfs = entries.filter((f) => f.toLowerCase().endsWith('.pdf'));
  if (!pdfs.length) {
    console.error('No PDFs found in textbooks/.');
    process.exit(1);
  }

  let updated = 0;
  const errors = [];

  for (const pdfName of pdfs) {
    const base = path.basename(pdfName, path.extname(pdfName));
    const pdfPath = path.join(TEXTBOOK_DIR, pdfName);
    const pagesJsonPath = path.join(OUTPUT_DIR, base, 'pages.json');

    try {
      await fs.access(pagesJsonPath);
    } catch {
      // No output yet; skip.
      continue;
    }

    try {
      const data = new Uint8Array(await fs.readFile(pdfPath));
      const doc = await pdfjsLib.getDocument({ data }).promise;
      const pageCount = doc.numPages;
      await doc.destroy();

      await fs.writeFile(pagesJsonPath, JSON.stringify(pageCount, null, 2), 'utf-8');
      updated += 1;
      console.log(`Updated ${base}: pages=${pageCount}`);
    } catch (err) {
      errors.push({ book: base, reason: err.message });
      console.warn(`Failed ${base}: ${err.message}`);
    }
  }

  console.log(`Done. Updated ${updated} books.` + (errors.length ? ` Failed ${errors.length}.` : ''));
  if (errors.length) {
    errors.forEach((e) => console.log(` - ${e.book}: ${e.reason}`));
  }
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
