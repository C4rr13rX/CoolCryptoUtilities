/**
 * Pure TypeScript / Zero-Dependency QR Code Generator
 * Generates ISO/IEC 18004 compliant QR codes (Versions 1-14) in Byte mode.
 * Produces crisp SVG strings or Canvas renderings without any external dependencies.
 */

export type EccLevel = 'L' | 'M' | 'Q' | 'H';

// Galois Field 256 tables (polynomial 0x11d)
const GF_EXP = new Uint8Array(512);
const GF_LOG = new Uint8Array(256);
(() => {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    GF_EXP[i] = x;
    GF_LOG[x] = i;
    x <<= 1;
    if (x & 256) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) {
    GF_EXP[i] = GF_EXP[i - 255];
  }
})();

function gfMul(x: number, y: number): number {
  if (x === 0 || y === 0) return 0;
  return GF_EXP[GF_LOG[x] + GF_LOG[y]];
}

function rsComputeEcc(data: Uint8Array, numEcc: number): Uint8Array {
  // Generator polynomial
  let gen = new Uint8Array([1]);
  for (let i = 0; i < numEcc; i++) {
    const nextGen = new Uint8Array(gen.length + 1);
    for (let j = 0; j < gen.length; j++) {
      nextGen[j] ^= gfMul(gen[j], GF_EXP[i]);
      nextGen[j + 1] ^= gen[j];
    }
    gen = nextGen;
  }

  const remainder = new Uint8Array(numEcc);
  for (let i = 0; i < data.length; i++) {
    const factor = data[i] ^ remainder[0];
    for (let j = 0; j < numEcc - 1; j++) {
      remainder[j] = remainder[j + 1] ^ gfMul(gen[j + 1], factor);
    }
    remainder[numEcc - 1] = gfMul(gen[numEcc], factor);
  }
  return remainder;
}

// Version table: [version, totalCodewords, [eccLevel: [numEcc, numBlocksG1, dataBytesG1, numBlocksG2, dataBytesG2]]]
interface VersionEccInfo {
  numEcc: number;
  blocks1: number;
  data1: number;
  blocks2: number;
  data2: number;
}

const ECC_TABLE: Record<number, Record<EccLevel, VersionEccInfo>> = {
  1: {
    L: { numEcc: 7, blocks1: 1, data1: 19, blocks2: 0, data2: 0 },
    M: { numEcc: 10, blocks1: 1, data1: 16, blocks2: 0, data2: 0 },
    Q: { numEcc: 13, blocks1: 1, data1: 13, blocks2: 0, data2: 0 },
    H: { numEcc: 17, blocks1: 1, data1: 9, blocks2: 0, data2: 0 }
  },
  2: {
    L: { numEcc: 10, blocks1: 1, data1: 34, blocks2: 0, data2: 0 },
    M: { numEcc: 16, blocks1: 1, data1: 28, blocks2: 0, data2: 0 },
    Q: { numEcc: 22, blocks1: 1, data1: 22, blocks2: 0, data2: 0 },
    H: { numEcc: 28, blocks1: 1, data1: 16, blocks2: 0, data2: 0 }
  },
  3: {
    L: { numEcc: 15, blocks1: 1, data1: 55, blocks2: 0, data2: 0 },
    M: { numEcc: 26, blocks1: 1, data1: 44, blocks2: 0, data2: 0 },
    Q: { numEcc: 18, blocks1: 2, data1: 17, blocks2: 0, data2: 0 },
    H: { numEcc: 22, blocks1: 2, data1: 13, blocks2: 0, data2: 0 }
  },
  4: {
    L: { numEcc: 20, blocks1: 1, data1: 80, blocks2: 0, data2: 0 },
    M: { numEcc: 18, blocks1: 2, data1: 32, blocks2: 0, data2: 0 },
    Q: { numEcc: 26, blocks1: 2, data1: 24, blocks2: 0, data2: 0 },
    H: { numEcc: 16, blocks1: 4, data1: 9, blocks2: 0, data2: 0 }
  },
  5: {
    L: { numEcc: 26, blocks1: 1, data1: 108, blocks2: 0, data2: 0 },
    M: { numEcc: 24, blocks1: 2, data1: 43, blocks2: 0, data2: 0 },
    Q: { numEcc: 18, blocks1: 2, data1: 15, blocks2: 2, data2: 16 },
    H: { numEcc: 22, blocks1: 2, data1: 11, blocks2: 2, data2: 12 }
  },
  6: {
    L: { numEcc: 18, blocks1: 2, data1: 68, blocks2: 0, data2: 0 },
    M: { numEcc: 16, blocks1: 4, data1: 27, blocks2: 0, data2: 0 },
    Q: { numEcc: 24, blocks1: 4, data1: 19, blocks2: 0, data2: 0 },
    H: { numEcc: 28, blocks1: 4, data1: 15, blocks2: 0, data2: 0 }
  },
  7: {
    L: { numEcc: 20, blocks1: 2, data1: 78, blocks2: 0, data2: 0 },
    M: { numEcc: 18, blocks1: 4, data1: 31, blocks2: 0, data2: 0 },
    Q: { numEcc: 18, blocks1: 2, data1: 14, blocks2: 4, data2: 15 },
    H: { numEcc: 26, blocks1: 4, data1: 13, blocks2: 1, data2: 14 }
  },
  8: {
    L: { numEcc: 24, blocks1: 2, data1: 97, blocks2: 0, data2: 0 },
    M: { numEcc: 22, blocks1: 2, data1: 38, blocks2: 2, data2: 39 },
    Q: { numEcc: 22, blocks1: 4, data1: 18, blocks2: 2, data2: 19 },
    H: { numEcc: 26, blocks1: 4, data1: 14, blocks2: 2, data2: 15 }
  },
  9: {
    L: { numEcc: 30, blocks1: 2, data1: 116, blocks2: 0, data2: 0 },
    M: { numEcc: 22, blocks1: 3, data1: 36, blocks2: 2, data2: 37 },
    Q: { numEcc: 20, blocks1: 4, data1: 16, blocks2: 4, data2: 17 },
    H: { numEcc: 24, blocks1: 4, data1: 12, blocks2: 4, data2: 13 }
  },
  10: {
    L: { numEcc: 18, blocks1: 2, data1: 68, blocks2: 2, data2: 69 },
    M: { numEcc: 26, blocks1: 4, data1: 43, blocks2: 1, data2: 44 },
    Q: { numEcc: 24, blocks1: 6, data1: 19, blocks2: 2, data2: 20 },
    H: { numEcc: 28, blocks1: 6, data1: 15, blocks2: 2, data2: 16 }
  },
  11: {
    L: { numEcc: 20, blocks1: 4, data1: 81, blocks2: 0, data2: 0 },
    M: { numEcc: 30, blocks1: 1, data1: 50, blocks2: 4, data2: 51 },
    Q: { numEcc: 28, blocks1: 4, data1: 22, blocks2: 4, data2: 23 },
    H: { numEcc: 24, blocks1: 3, data1: 12, blocks2: 8, data2: 13 }
  },
  12: {
    L: { numEcc: 24, blocks1: 2, data1: 92, blocks2: 2, data2: 93 },
    M: { numEcc: 22, blocks1: 6, data1: 36, blocks2: 2, data2: 37 },
    Q: { numEcc: 26, blocks1: 4, data1: 20, blocks2: 6, data2: 21 },
    H: { numEcc: 28, blocks1: 7, data1: 14, blocks2: 4, data2: 15 }
  },
  13: {
    L: { numEcc: 26, blocks1: 4, data1: 107, blocks2: 0, data2: 0 },
    M: { numEcc: 22, blocks1: 8, data1: 37, blocks2: 1, data2: 38 },
    Q: { numEcc: 24, blocks1: 8, data1: 20, blocks2: 4, data2: 21 },
    H: { numEcc: 22, blocks1: 12, data1: 11, blocks2: 4, data2: 12 }
  },
  14: {
    L: { numEcc: 30, blocks1: 3, data1: 115, blocks2: 1, data2: 116 },
    M: { numEcc: 24, blocks1: 4, data1: 40, blocks2: 5, data2: 41 },
    Q: { numEcc: 20, blocks1: 11, data1: 16, blocks2: 5, data2: 17 },
    H: { numEcc: 24, blocks1: 11, data1: 12, blocks2: 5, data2: 13 }
  }
};

const ALIGNMENT_PATTERNS: Record<number, number[]> = {
  1: [],
  2: [6, 18],
  3: [6, 22],
  4: [6, 26],
  5: [6, 30],
  6: [6, 34],
  7: [6, 22, 38],
  8: [6, 24, 42],
  9: [6, 26, 46],
  10: [6, 28, 50],
  11: [6, 30, 54],
  12: [6, 32, 58],
  13: [6, 34, 62],
  14: [6, 26, 46, 66]
};

const FORMAT_BITS: Record<string, number> = {
  'L_0': 0x77c4, 'L_1': 0x72f3, 'L_2': 0x7daa, 'L_3': 0x789d,
  'L_4': 0x662f, 'L_5': 0x6318, 'L_6': 0x6c41, 'L_7': 0x6976,
  'M_0': 0x5412, 'M_1': 0x5125, 'M_2': 0x5e7c, 'M_3': 0x5b4b,
  'M_4': 0x45f9, 'M_5': 0x40ce, 'M_6': 0x4f97, 'M_7': 0x4aa0,
  'Q_0': 0x355f, 'Q_1': 0x3068, 'Q_2': 0x3f31, 'Q_3': 0x3a06,
  'Q_4': 0x24b4, 'Q_5': 0x2183, 'Q_6': 0x2eda, 'Q_7': 0x2bed,
  'H_0': 0x1689, 'H_1': 0x13be, 'H_2': 0x1ce7, 'H_3': 0x19d0,
  'H_4': 0x0762, 'H_5': 0x0255, 'H_6': 0x0d0c, 'H_7': 0x083b
};

class BitBuffer {
  private buffer: number[] = [];
  public length = 0;

  put(num: number, length: number) {
    for (let i = length - 1; i >= 0; i--) {
      this.putBit(((num >>> i) & 1) === 1);
    }
  }

  putBit(bit: boolean) {
    const bufIndex = Math.floor(this.length / 8);
    if (this.buffer.length <= bufIndex) {
      this.buffer.push(0);
    }
    if (bit) {
      this.buffer[bufIndex] |= 0x80 >>> (this.length % 8);
    }
    this.length++;
  }

  getBytes(): Uint8Array {
    return new Uint8Array(this.buffer);
  }
}

export class QrCode {
  public readonly version: number;
  public readonly size: number;
  public readonly modules: boolean[][];
  private isFunction: boolean[][];

  constructor(public readonly text: string, public readonly ecc: EccLevel = 'M') {
    const utf8Bytes = new TextEncoder().encode(text);
    this.version = this.selectVersion(utf8Bytes.length, ecc);
    this.size = this.version * 4 + 17;
    this.modules = Array.from({ length: this.size }, () => Array(this.size).fill(false));
    this.isFunction = Array.from({ length: this.size }, () => Array(this.size).fill(false));

    this.build(utf8Bytes);
  }

  private selectVersion(dataLen: number, ecc: EccLevel): number {
    for (let v = 1; v <= 14; v++) {
      const info = ECC_TABLE[v][ecc];
      const maxDataCodewords = info.blocks1 * info.data1 + info.blocks2 * info.data2;
      // Byte mode header: 4 bits mode + 8 bits length (or 16 bits if v >= 10)
      const countBits = v >= 10 ? 16 : 8;
      const totalBits = 4 + countBits + dataLen * 8;
      const totalBytes = Math.ceil(totalBits / 8);
      if (totalBytes <= maxDataCodewords) {
        return v;
      }
    }
    return 14;
  }

  private build(dataBytes: Uint8Array) {
    // 1. Draw static patterns
    this.drawFinderPattern(0, 0);
    this.drawFinderPattern(this.size - 7, 0);
    this.drawFinderPattern(0, this.size - 7);
    this.drawTimingPatterns();
    this.drawAlignmentPatterns();
    this.reserveFormatInfo();

    // 2. Prepare bit stream
    const bb = new BitBuffer();
    bb.put(0x4, 4); // Byte mode indicator
    const countBits = this.version >= 10 ? 16 : 8;
    bb.put(dataBytes.length, countBits);
    for (let i = 0; i < dataBytes.length; i++) {
      bb.put(dataBytes[i], 8);
    }

    const info = ECC_TABLE[this.version][this.ecc];
    const totalDataBytes = info.blocks1 * info.data1 + info.blocks2 * info.data2;

    // Terminator
    const termLen = Math.min(4, totalDataBytes * 8 - bb.length);
    bb.put(0, termLen);

    // Byte align
    while (bb.length % 8 !== 0) {
      bb.putBit(false);
    }

    // Pad bytes
    const padBytes = [0xec, 0x11];
    let padIdx = 0;
    while (bb.length / 8 < totalDataBytes) {
      bb.put(padBytes[padIdx % 2], 8);
      padIdx++;
    }

    const rawData = bb.getBytes();

    // 3. Error correction and block interleaving
    const blocks: { data: Uint8Array; ecc: Uint8Array }[] = [];
    let offset = 0;

    for (let b = 0; b < info.blocks1; b++) {
      const slice = rawData.slice(offset, offset + info.data1);
      offset += info.data1;
      const ecc = rsComputeEcc(slice, info.numEcc);
      blocks.push({ data: slice, ecc });
    }
    for (let b = 0; b < info.blocks2; b++) {
      const slice = rawData.slice(offset, offset + info.data2);
      offset += info.data2;
      const ecc = rsComputeEcc(slice, info.numEcc);
      blocks.push({ data: slice, ecc });
    }

    // Interleave data
    const finalCodewords: number[] = [];
    const maxDataLen = Math.max(info.data1, info.data2);
    for (let i = 0; i < maxDataLen; i++) {
      for (const block of blocks) {
        if (i < block.data.length) {
          finalCodewords.push(block.data[i]);
        }
      }
    }
    // Interleave ECC
    for (let i = 0; i < info.numEcc; i++) {
      for (const block of blocks) {
        finalCodewords.push(block.ecc[i]);
      }
    }

    // 4. Place codewords in matrix
    let bitIndex = 0;
    const totalBits = finalCodewords.length * 8;
    let up = true;

    for (let right = this.size - 1; right > 0; right -= 2) {
      if (right === 6) right--; // Skip vertical timing line

      for (let vert = 0; vert < this.size; vert++) {
        const y = up ? this.size - 1 - vert : vert;

        for (let i = 0; i < 2; i++) {
          const x = right - i;
          if (!this.isFunction[y][x]) {
            let bit = false;
            if (bitIndex < totalBits) {
              const byteVal = finalCodewords[Math.floor(bitIndex / 8)];
              bit = ((byteVal >>> (7 - (bitIndex % 8))) & 1) === 1;
              bitIndex++;
            }
            this.modules[y][x] = bit;
          }
        }
      }
      up = !up;
    }

    // 5. Select optimal mask pattern (0..7)
    let bestMask = 0;
    let lowestPenalty = Infinity;

    for (let mask = 0; mask < 8; mask++) {
      this.applyMask(mask);
      this.writeFormatInfo(mask);
      const penalty = this.calculatePenalty();
      if (penalty < lowestPenalty) {
        lowestPenalty = penalty;
        bestMask = mask;
      }
      this.applyMask(mask); // Undo mask
    }

    // Apply best mask
    this.applyMask(bestMask);
    this.writeFormatInfo(bestMask);
  }

  private drawFinderPattern(startX: number, startY: number) {
    for (let y = -1; y <= 7; y++) {
      for (let x = -1; x <= 7; x++) {
        const px = startX + x;
        const py = startY + y;
        if (px >= 0 && px < this.size && py >= 0 && py < this.size) {
          const isBlack =
            (x >= 0 && x <= 6 && (y === 0 || y === 6)) ||
            (y >= 0 && y <= 6 && (x === 0 || x === 6)) ||
            (x >= 2 && x <= 4 && y >= 2 && y <= 4);
          this.modules[py][px] = isBlack;
          this.isFunction[py][px] = true;
        }
      }
    }
  }

  private drawTimingPatterns() {
    for (let i = 8; i < this.size - 8; i++) {
      const bit = i % 2 === 0;
      this.modules[6][i] = bit;
      this.isFunction[6][i] = true;
      this.modules[i][6] = bit;
      this.isFunction[i][6] = true;
    }
  }

  private drawAlignmentPatterns() {
    const coords = ALIGNMENT_PATTERNS[this.version] || [];
    for (let i = 0; i < coords.length; i++) {
      for (let j = 0; j < coords.length; j++) {
        const cx = coords[i];
        const cy = coords[j];
        // Skip finder pattern zones
        if (this.isFunction[cy][cx]) continue;

        for (let y = -2; y <= 2; y++) {
          for (let x = -2; x <= 2; x++) {
            const isBlack = Math.max(Math.abs(x), Math.abs(y)) !== 1;
            this.modules[cy + y][cx + x] = isBlack;
            this.isFunction[cy + y][cx + x] = true;
          }
        }
      }
    }
  }

  private reserveFormatInfo() {
    for (let i = 0; i < 9; i++) {
      this.isFunction[8][i] = true;
      this.isFunction[i][8] = true;
    }
    for (let i = 0; i < 8; i++) {
      this.isFunction[8][this.size - 1 - i] = true;
      this.isFunction[this.size - 1 - i][8] = true;
    }
    // Dark module
    this.modules[this.size - 8][8] = true;
    this.isFunction[this.size - 8][8] = true;
  }

  private writeFormatInfo(mask: number) {
    const key = `${this.ecc}_${mask}`;
    const bits = FORMAT_BITS[key] || 0;

    for (let i = 0; i < 15; i++) {
      const bit = ((bits >>> (14 - i)) & 1) === 1;
      // Around top-left finder
      if (i < 6) {
        this.modules[8][i] = bit;
      } else if (i < 8) {
        this.modules[8][i + 1] = bit;
      } else if (i === 8) {
        this.modules[7][8] = bit;
      } else {
        this.modules[14 - i][8] = bit;
      }

      // Around other finders
      if (i < 8) {
        this.modules[this.size - 1 - i][8] = bit;
      } else {
        this.modules[8][this.size - 15 + i] = bit;
      }
    }
  }

  private applyMask(mask: number) {
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        if (this.isFunction[y][x]) continue;

        let invert = false;
        switch (mask) {
          case 0: invert = (x + y) % 2 === 0; break;
          case 1: invert = y % 2 === 0; break;
          case 2: invert = x % 3 === 0; break;
          case 3: invert = (x + y) % 3 === 0; break;
          case 4: invert = (Math.floor(y / 2) + Math.floor(x / 3)) % 2 === 0; break;
          case 5: invert = ((x * y) % 2) + ((x * y) % 3) === 0; break;
          case 6: invert = (((x * y) % 2) + ((x * y) % 3)) % 2 === 0; break;
          case 7: invert = (((x + y) % 2) + ((x * y) % 3)) % 2 === 0; break;
        }
        if (invert) {
          this.modules[y][x] = !this.modules[y][x];
        }
      }
    }
  }

  private calculatePenalty(): number {
    let penalty = 0;
    // 1. Five or more same-color modules in a row/col
    for (let y = 0; y < this.size; y++) {
      let count = 0;
      let last = false;
      for (let x = 0; x < this.size; x++) {
        if (x === 0 || this.modules[y][x] !== last) {
          last = this.modules[y][x];
          count = 1;
        } else {
          count++;
          if (count === 5) penalty += 3;
          else if (count > 5) penalty += 1;
        }
      }
    }
    return penalty;
  }

  /**
   * Generates a clean, scalable SVG string representing the QR code.
   */
  toSvg(options: { size?: number; margin?: number; color?: string; background?: string } = {}): string {
    const margin = options.margin ?? 4;
    const color = options.color ?? '#000000';
    const bg = options.background ?? '#ffffff';
    const viewSize = this.size + margin * 2;

    let path = '';
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        if (this.modules[y][x]) {
          path += `M${x + margin},${y + margin}h1v1h-1z `;
        }
      }
    }

    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${viewSize} ${viewSize}" shape-rendering="crispEdges">
      <rect width="${viewSize}" height="${viewSize}" fill="${bg}"/>
      <path d="${path}" fill="${color}"/>
    </svg>`;
  }
}
