export type AmbientSettings = {
  baseFreq: number;
  detuneHz: number;
  lfoRate: number;
  lfoDepth: number;
  gain: number;
  attack: number;
  decay: number;
  chordGain: number;
  chordDuration: number;
  chordPreset: string;
  keyRoot: string;
  keyMode: 'major' | 'minor';
};

export const DEFAULT_AMBIENT_SETTINGS: AmbientSettings = {
  baseFreq: 110,
  detuneHz: 6,
  lfoRate: 0.08,
  lfoDepth: 0.12,
  gain: 0.18,
  attack: 0.03,
  decay: 3.4,
  chordGain: 0.22,
  chordDuration: 3.2,
  chordPreset: 'dream_minor',
  keyRoot: 'C',
  keyMode: 'minor',
};

const CHORDS: Record<string, number[][]> = {
  dream_minor: [
    [0, 3, 7, 10],
    [0, 5, 7, 10],
    [0, 3, 8, 10],
  ],
  soft_major: [
    [0, 4, 7, 11],
    [0, 4, 9],
    [0, 5, 9],
  ],
  suspended: [
    [0, 5, 7, 10],
    [0, 7, 12],
    [0, 5, 10],
  ],
  shimmer: [
    [0, 7, 14],
    [0, 9, 16],
    [0, 5, 12],
  ],
};

class AmbientAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private leftOsc: OscillatorNode | null = null;
  private rightOsc: OscillatorNode | null = null;
  private leftPan: StereoPannerNode | null = null;
  private rightPan: StereoPannerNode | null = null;
  private lfo: OscillatorNode | null = null;
  private lfoGain: GainNode | null = null;
  private enabled = false;
  private settings: AmbientSettings = { ...DEFAULT_AMBIENT_SETTINGS };
  private lastChordAt = 0;

  isEnabled() {
    return this.enabled;
  }

  getSettings() {
    return { ...this.settings };
  }

  async enable(settings?: Partial<AmbientSettings>) {
    this.settings = { ...this.settings, ...(settings || {}) };
    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    this.enabled = true;
    this.ensureNodes();
    this.applySettings(this.settings);
  }

  disable() {
    this.enabled = false;
    this.fadeOut();
  }

  applySettings(settings: Partial<AmbientSettings>) {
    this.settings = { ...this.settings, ...settings };
    if (!this.enabled) {
      return;
    }
    this.ensureNodes();
    this.syncParams();
  }

  triggerChord() {
    if (!this.enabled || !this.ctx || !this.master) return;
    const now = this.ctx.currentTime;
    if (now - this.lastChordAt < 0.15) return;
    this.lastChordAt = now;

    const chordOptions = CHORDS[this.settings.chordPreset] || CHORDS.dream_minor;
    const chord = chordOptions[Math.floor(Math.random() * chordOptions.length)];
    const keyOffset = keyToSemitone(this.settings.keyRoot);
    const gain = this.ctx.createGain();
    const decayTime = Math.max(0.4, this.settings.decay);
    const stopTime = now + Math.max(this.settings.chordDuration, decayTime);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(this.settings.chordGain, now + this.settings.attack);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + decayTime);
    gain.connect(this.master);

    chord.forEach((interval) => {
      const osc = this.ctx!.createOscillator();
      osc.type = 'sine';
      const freq = this.settings.baseFreq * Math.pow(2, (interval + keyOffset) / 12);
      osc.frequency.setValueAtTime(freq, now);
      osc.connect(gain);
      osc.start(now);
      osc.stop(stopTime + 0.1);
    });
  }

  private ensureContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      this.ctx = new AudioCtx();
    }
    return this.ctx;
  }

  private ensureNodes() {
    if (!this.ctx) return;
    if (!this.master) {
      this.master = this.ctx.createGain();
      this.master.gain.setValueAtTime(0.0001, this.ctx.currentTime);
      this.master.connect(this.ctx.destination);
    }

    if (!this.leftOsc || !this.rightOsc) {
      this.leftOsc = this.ctx.createOscillator();
      this.rightOsc = this.ctx.createOscillator();
      this.leftOsc.type = 'sine';
      this.rightOsc.type = 'sine';

      this.leftPan = this.ctx.createStereoPanner();
      this.rightPan = this.ctx.createStereoPanner();
      this.leftPan.pan.value = -0.4;
      this.rightPan.pan.value = 0.4;

      this.leftOsc.connect(this.leftPan).connect(this.master);
      this.rightOsc.connect(this.rightPan).connect(this.master);
      this.leftOsc.start();
      this.rightOsc.start();
    }

    if (!this.lfo) {
      this.lfo = this.ctx.createOscillator();
      this.lfoGain = this.ctx.createGain();
      this.lfo.type = 'sine';
      this.lfo.connect(this.lfoGain!).connect(this.master.gain);
      this.lfo.start();
    }
  }

  private syncParams() {
    if (!this.ctx || !this.master || !this.leftOsc || !this.rightOsc || !this.lfo || !this.lfoGain) return;
    const now = this.ctx.currentTime;
    const base = this.settings.baseFreq;
    const detune = this.settings.detuneHz;
    this.leftOsc.frequency.setTargetAtTime(Math.max(30, base - detune / 2), now, 0.08);
    this.rightOsc.frequency.setTargetAtTime(Math.max(30, base + detune / 2), now, 0.08);
    this.master.gain.setTargetAtTime(this.settings.gain, now, 0.12);
    this.lfo.frequency.setTargetAtTime(this.settings.lfoRate, now, 0.2);
    this.lfoGain.gain.setTargetAtTime(this.settings.lfoDepth, now, 0.2);
  }

  private fadeOut() {
    if (!this.ctx || !this.master) return;
    const now = this.ctx.currentTime;
    this.master.gain.setTargetAtTime(0.0001, now, 0.2);
  }
}

export const ambientAudio = new AmbientAudio();

const KEY_MAP: Record<string, number> = {
  C: 0,
  'C#': 1,
  Db: 1,
  D: 2,
  Eb: 3,
  'D#': 3,
  E: 4,
  F: 5,
  'F#': 6,
  Gb: 6,
  G: 7,
  Ab: 8,
  'G#': 8,
  A: 9,
  Bb: 10,
  'A#': 10,
  B: 11,
};

function keyToSemitone(root: string) {
  return KEY_MAP[root] ?? 0;
}
