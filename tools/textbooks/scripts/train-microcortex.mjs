// Train a lightweight microcortex perceptron on labeled segments.
// Expects data/segments-labeled.ndjson (preferred) or data/segments.ndjson.
// Saves weights to data/microcortex-weights.json.

import fs from 'node:fs/promises';
import path from 'node:path';
const DATA_DIR = path.join(process.cwd(), 'data');
const LABELED_OPENAI = path.join(DATA_DIR, 'segments-labeled-openai.ndjson');
const LABELED = path.join(DATA_DIR, 'segments-labeled.ndjson');
const RAW = path.join(DATA_DIR, 'segments.ndjson');
const WEIGHTS_OUT = path.join(DATA_DIR, 'microcortex-weights.json');
const TAXONOMY = JSON.parse(await fs.readFile(path.join(DATA_DIR, 'label-taxonomy.json'), 'utf-8')).labels;

const loadSegments = async () => {
  const src = await fs
    .access(LABELED_OPENAI)
    .then(() => LABELED_OPENAI)
    .catch(async () =>
      fs
        .access(LABELED)
        .then(() => LABELED)
        .catch(() => RAW)
    );
  const content = await fs.readFile(src, 'utf-8');
  return content
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l));
};

const oneHot = (label) => {
  const idx = TAXONOMY.indexOf(label);
  if (idx === -1) return null;
  const arr = new Float32Array(TAXONOMY.length);
  arr[idx] = 1;
  return { idx, vec: arr };
};

const heuristicMap = {
  title: 'frontmatter.title',
  subtitle: 'frontmatter.subtitle',
  heading: 'section.heading',
  paragraph: 'paragraph.body',
  list: 'list.bullet',
  callout: 'callout.note',
  footer: 'footer'
};

const relu = (x) => (x > 0 ? x : 0);
const reluGrad = (x) => (x > 0 ? 1 : 0);

const train = (data, options = {}) => {
  const inputSize = 7;
  const hiddenSize = options.hiddenSize ?? 24;
  const outputSize = TAXONOMY.length;
  const lr = options.lr ?? 0.01;
  const epochs = options.epochs ?? 20;

  // Initialize weights small random.
  const rand = (n) => Float32Array.from({ length: n }, () => (Math.random() - 0.5) * 0.1);
  const hiddenWeights = rand(inputSize * hiddenSize);
  const hiddenBiases = rand(hiddenSize);
  const outputWeights = rand(hiddenSize * outputSize);
  const outputBiases = rand(outputSize);

  const forward = (features) => {
    const h = new Float32Array(hiddenSize);
    for (let j = 0; j < hiddenSize; j++) {
      let acc = hiddenBiases[j];
      const wOff = j * inputSize;
      for (let i = 0; i < inputSize; i++) acc += features[i] * hiddenWeights[wOff + i];
      h[j] = relu(acc);
    }
    const o = new Float32Array(outputSize);
    for (let k = 0; k < outputSize; k++) {
      let acc = outputBiases[k];
      const wOff = k * hiddenSize;
      for (let j = 0; j < hiddenSize; j++) acc += h[j] * outputWeights[wOff + j];
      o[k] = acc;
    }
    return { h, o };
  };

  const softmax = (logits) => {
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((acc, v) => acc + v, 0);
    return exps.map((v) => v / sum);
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    let loss = 0;
    for (const item of data) {
      const { features, target } = item;
      const { h, o } = forward(features);
      const probs = softmax(Array.from(o));
      const gradO = new Float32Array(outputSize);
      for (let k = 0; k < outputSize; k++) {
        const y = k === target ? 1 : 0;
        loss += -Math.log(Math.max(probs[k], 1e-8));
        gradO[k] = probs[k] - y;
      }
      // Backprop to output weights/bias
      for (let k = 0; k < outputSize; k++) {
        const wOff = k * hiddenSize;
        for (let j = 0; j < hiddenSize; j++) {
          outputWeights[wOff + j] -= lr * gradO[k] * h[j];
        }
        outputBiases[k] -= lr * gradO[k];
      }
      // Backprop to hidden
      const gradH = new Float32Array(hiddenSize);
      for (let j = 0; j < hiddenSize; j++) {
        let acc = 0;
        for (let k = 0; k < outputSize; k++) {
          const wOff = k * hiddenSize;
          acc += gradO[k] * outputWeights[wOff + j];
        }
        gradH[j] = acc * reluGrad(h[j]);
      }
      for (let j = 0; j < hiddenSize; j++) {
        const wOff = j * inputSize;
        for (let i = 0; i < inputSize; i++) {
          hiddenWeights[wOff + i] -= lr * gradH[j] * features[i];
        }
        hiddenBiases[j] -= lr * gradH[j];
      }
    }
    loss /= data.length;
    console.log(`epoch ${epoch + 1}/${epochs} loss=${loss.toFixed(4)}`);
  }

  return {
    inputSize,
    hiddenSize,
    outputSize,
    hiddenWeights: Array.from(hiddenWeights),
    hiddenBiases: Array.from(hiddenBiases),
    outputWeights: Array.from(outputWeights),
    outputBiases: Array.from(outputBiases)
  };
};

const main = async () => {
  const segments = await loadSegments();
  const data = [];
  for (const seg of segments) {
    const mappedLabel = heuristicMap[seg.label] ?? seg.label;
    const one = oneHot(mappedLabel);
    if (!one) continue;
    data.push({ features: seg.features, target: one.idx });
  }
  if (!data.length) {
    console.error('No labeled data found matching taxonomy.');
    process.exit(1);
  }
  const model = train(data, { epochs: 30, lr: 0.01, hiddenSize: 32 });
  await fs.writeFile(WEIGHTS_OUT, JSON.stringify({ taxonomy: TAXONOMY, model }, null, 2), 'utf-8');
  console.log(`Saved weights to ${WEIGHTS_OUT}`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
