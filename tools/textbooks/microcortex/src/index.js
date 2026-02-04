// Lightweight neuron primitives using plain typed arrays.
// Designed for CPU/mobile use without external tensor libraries.

export class DenseLayer {
  constructor(inputSize, outputSize, weights, biases, activation = 'relu') {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = weights; // Float32Array length inputSize * outputSize (row-major)
    this.biases = biases; // Float32Array length outputSize
    this.activation = activation;
  }

  forward(input, output) {
    const { inputSize, outputSize, weights, biases } = this;
    for (let j = 0; j < outputSize; j++) {
      let acc = biases[j] ?? 0;
      const wOffset = j * inputSize;
      for (let i = 0; i < inputSize; i++) {
        acc += input[i] * weights[wOffset + i];
      }
      output[j] = applyActivation(acc, this.activation);
    }
  }
}

export class MicroNetwork {
  constructor(layers) {
    this.layers = layers;
  }

  forward(input) {
    let current = input;
    for (let idx = 0; idx < this.layers.length; idx++) {
      const layer = this.layers[idx];
      const out = new Float32Array(layer.outputSize);
      layer.forward(current, out);
      current = out;
    }
    return current;
  }
}

export const applyActivation = (value, activation) => {
  switch (activation) {
    case 'relu':
      return value > 0 ? value : 0;
    case 'tanh':
      return Math.tanh(value);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-value));
    case 'identity':
    default:
      return value;
  }
};

export const softmax = (values) => {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sum);
};

// Helper to build a two-layer perceptron quickly from plain objects.
export const buildPerceptron = (spec) => {
  const { inputSize, hiddenSize, outputSize, hiddenWeights, hiddenBiases, outputWeights, outputBiases } = spec;
  const hidden = new DenseLayer(inputSize, hiddenSize, new Float32Array(hiddenWeights), new Float32Array(hiddenBiases), 'relu');
  const output = new DenseLayer(hiddenSize, outputSize, new Float32Array(outputWeights), new Float32Array(outputBiases), 'identity');
  return new MicroNetwork([hidden, output]);
};

// Neuro-inspired primitives (lightweight)
export class Synapse {
  constructor(targetId, weight = 0, inhibitory = false) {
    this.targetId = targetId;
    this.weight = weight;
    this.inhibitory = inhibitory;
  }
}

export class Neuron {
  constructor(id) {
    this.id = id;
    this.potential = 0;
    this.threshold = 1;
    this.incoming = [];
    this.outgoing = [];
  }

  connect(target, weight = 0.1, inhibitory = false) {
    const syn = new Synapse(target.id, weight, inhibitory);
    this.outgoing.push(syn);
    target.incoming.push(new Synapse(this.id, weight, inhibitory));
  }

  integrate(inputs) {
    // inputs: Map<neuronId, signal>
    let excit = 0;
    let inhib = 0;
    for (const syn of this.incoming) {
      const signal = inputs.get(syn.targetId) ?? 0;
      if (syn.inhibitory) inhib += signal * Math.abs(syn.weight);
      else excit += signal * syn.weight;
    }
    this.potential = excit - inhib;
    return this.potential;
  }

  fire() {
    const active = this.potential >= this.threshold ? 1 : 0;
    this.potential = 0;
    return active;
  }

  hebbianUpdate(inputs, lr = 0.01) {
    for (const syn of this.incoming) {
      const signal = inputs.get(syn.targetId) ?? 0;
      const delta = lr * signal;
      syn.weight += syn.inhibitory ? -delta : delta;
    }
  }
}

export class Column {
  constructor(id, size = 8) {
    this.id = id;
    this.neurons = Array.from({ length: size }, (_, idx) => new Neuron(`${id}-${idx}`));
  }

  inhibitAll(strength = 0.05) {
    this.neurons.forEach((n) => {
      n.threshold += strength;
    });
  }
}

export class MicroCortex {
  constructor(columns = []) {
    this.columns = columns;
    this.neuronIndex = new Map();
    columns.flatMap((c) => c.neurons).forEach((n) => this.neuronIndex.set(n.id, n));
  }

  step(activeIds, lr = 0.01) {
    const inputs = new Map();
    activeIds.forEach((id) => inputs.set(id, 1));
    const outputs = new Map();
    for (const neuron of this.neuronIndex.values()) {
      neuron.integrate(inputs);
      const fired = neuron.fire();
      outputs.set(neuron.id, fired);
      if (fired) neuron.hebbianUpdate(inputs, lr);
    }
    return outputs;
  }

  addColumn(column) {
    this.columns.push(column);
    column.neurons.forEach((n) => this.neuronIndex.set(n.id, n));
  }
}
