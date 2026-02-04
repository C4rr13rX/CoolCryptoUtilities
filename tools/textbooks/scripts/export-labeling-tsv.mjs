// Export segments to TSV for manual labeling using the proposed taxonomy.
// Reads data/segments.ndjson and writes data/segments-label.tsv with columns:
// id	label	book	page	text	features	json

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DATA_DIR = path.join(process.cwd(), 'data');
const SRC = path.join(DATA_DIR, 'segments.ndjson');
const OUT = path.join(DATA_DIR, 'segments-label.tsv');

const main = async () => {
  const content = await fs.readFile(SRC, 'utf-8');
  const lines = content
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l, idx) => ({ id: idx, ...JSON.parse(l) }));
  const rows = ['id\tlabel\tbook\tpage\ttext\tfeatures\tjson'];
  lines.forEach((item) => {
    const jsonBlob = JSON.stringify(item);
    rows.push(
      `${item.id}\t${item.label}\t${item.book}\t${item.page}\t${item.text.replace(/\s+/g, ' ').slice(0, 500)}\t${item.features.join(',')}\t${jsonBlob}`
    );
  });
  await fs.writeFile(OUT, rows.join('\n'), 'utf-8');
  console.log(`Wrote ${lines.length} rows to ${OUT}`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
