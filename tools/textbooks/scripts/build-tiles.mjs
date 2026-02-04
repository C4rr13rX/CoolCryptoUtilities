// Build tile-level features from segmented page PNGs for the VisionCortex.
// Outputs data/tiles.ndjson: { book, page, tileId, features }

import fs from 'node:fs/promises';
import path from 'node:path';
import { createCanvas, loadImage } from '@napi-rs/canvas';

const OUTPUT_ROOT = path.join(process.cwd(), 'textbooks', 'output');
const DATA_DIR = path.join(process.cwd(), 'data');
const OUT = path.join(DATA_DIR, 'tiles.ndjson');

const TILE = 32;

const ensureDir = async (dir) => fs.mkdir(dir, { recursive: true });

const listPngs = async () => {
  const books = await fs.readdir(OUTPUT_ROOT);
  const entries = [];
  for (const book of books) {
    const dir = path.join(OUTPUT_ROOT, book);
    const files = await fs.readdir(dir);
    files
      .filter((f) => f.endsWith('.png'))
      .forEach((file) => entries.push({ book, file, path: path.join(dir, file) }));
  }
  return entries;
};

const computeTileFeatures = (img, x0, y0, size) => {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, x0, y0, size, size, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  let sum = 0;
  let edgeH = 0;
  let edgeV = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      sum += gray;
      if (x > 0) {
        const idxL = (y * size + (x - 1)) * 4;
        const grayL = 0.299 * data[idxL] + 0.587 * data[idxL + 1] + 0.114 * data[idxL + 2];
        edgeH += Math.abs(gray - grayL);
      }
      if (y > 0) {
        const idxU = ((y - 1) * size + x) * 4;
        const grayU = 0.299 * data[idxU] + 0.587 * data[idxU + 1] + 0.114 * data[idxU + 2];
        edgeV += Math.abs(gray - grayU);
      }
    }
  }
  const pixels = size * size;
  return [
    sum / (pixels * 255), // intensity
    edgeH / (pixels * 255),
    edgeV / (pixels * 255)
  ];
};

const main = async () => {
  await ensureDir(DATA_DIR);
  const pngs = await listPngs();
  const lines = [];
  for (const entry of pngs) {
    const img = await loadImage(entry.path);
    const cols = Math.ceil(img.width / TILE);
    const rows = Math.ceil(img.height / TILE);
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const x0 = col * TILE;
        const y0 = row * TILE;
        const features = computeTileFeatures(img, x0, y0, TILE);
        lines.push({
          book: entry.book,
          page: Number(entry.file.match(/page-(\d+)/)?.[1] ?? 0),
          tileId: `${entry.book}-p${entry.file}-r${row}-c${col}`,
          features
        });
      }
    }
  }
  await fs.writeFile(OUT, lines.map((l) => JSON.stringify(l)).join('\n'), 'utf-8');
  console.log(`Wrote ${lines.length} tiles to ${OUT}`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
