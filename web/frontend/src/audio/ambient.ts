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
  chordWaveform: OscillatorType;
  droneWaveform: OscillatorType;
  backgroundSource: 'drone' | 'midi';
  chordGateMode: 'off' | 'pattern';
  chordGateBpm: number;
  chordGateDepth: number;
  chordGatePattern: number[];
  droneGateMode: 'off' | 'pattern';
  droneGateBpm: number;
  droneGateDepth: number;
  droneGatePattern: number[];
  midiWaveform: OscillatorType;
  midiGain: number;
  midiGateMode: 'off' | 'pattern';
  midiGateBpm: number;
  midiGateDepth: number;
  midiGatePattern: number[];
  keyRoot: string;
  keyMode: 'major' | 'minor';
};

export type SoundMap = Record<string, string>;

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
  chordWaveform: 'sine',
  droneWaveform: 'sine',
  backgroundSource: 'drone',
  chordGateMode: 'off',
  chordGateBpm: 84,
  chordGateDepth: 1,
  chordGatePattern: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0.4, 1, 0, 1, 0.2, 1, 0],
  droneGateMode: 'off',
  droneGateBpm: 84,
  droneGateDepth: 1,
  droneGatePattern: [1, 0.2, 0.8, 0.2, 1, 0.3, 0.7, 0.2, 1, 0.4, 0.8, 0.2, 1, 0.3, 0.9, 0.2],
  midiWaveform: 'sine',
  midiGain: 0.2,
  midiGateMode: 'off',
  midiGateBpm: 84,
  midiGateDepth: 1,
  midiGatePattern: [1, 0, 1, 0, 1, 0.4, 1, 0, 1, 0.2, 1, 0, 1, 0.4, 1, 0],
  keyRoot: 'C',
  keyMode: 'minor',
};

export const MOTIFS: Record<string, number[][]> = {
  overview: [
    [0, 4, 7, 11],
    [2, 7, 11],
    [0, 5, 9],
  ],
  organism: [
    [0, 3, 7, 10],
    [5, 8, 12],
    [2, 7, 10],
  ],
  pipeline: [
    [0, 5, 7, 10],
    [0, 4, 9],
    [2, 7, 12],
  ],
  streams: [
    [0, 7, 12],
    [0, 5, 9],
    [0, 4, 7, 12],
  ],
  telemetry: [
    [0, 3, 7],
    [2, 5, 9],
    [0, 3, 8],
  ],
  wallet: [
    [0, 4, 7],
    [0, 9, 12],
    [2, 7, 11],
  ],
  c0d3r: [
    [0, 7, 10],
    [0, 3, 7, 10],
    [2, 5, 9],
  ],
  addressbook: [
    [0, 4, 9],
    [0, 5, 9],
    [2, 7, 11],
  ],
  advisories: [
    [0, 3, 6, 9],
    [0, 5, 8],
    [2, 6, 9],
  ],
  datalab: [
    [0, 4, 7, 10],
    [2, 5, 9],
    [0, 5, 10],
  ],
  lab: [
    [0, 4, 8],
    [0, 3, 7, 10],
    [2, 6, 9],
  ],
  guardian: [
    [0, 5, 10],
    [0, 3, 7],
    [2, 7, 10],
  ],
  codegraph: [
    [0, 4, 9, 12],
    [0, 7, 14],
    [2, 5, 9],
  ],
  integrations: [
    [0, 4, 7],
    [2, 6, 9],
    [0, 5, 9],
  ],
  settings: [
    [0, 3, 7],
    [0, 5, 8],
    [2, 5, 9],
  ],
  audiolab: [
    [0, 7, 12],
    [0, 4, 9],
    [0, 5, 10],
  ],
  u53rxr080t: [
    [0, 3, 7, 10],
    [2, 5, 9],
    [0, 7, 10],
  ],
  branddozer: [
    [0, 4, 7, 11],
    [0, 3, 8],
    [2, 7, 11],
  ],
  action: [
    [0, 7],
    [0, 4, 7],
    [2, 7, 11],
  ],
  link: [
    [0, 5, 9],
    [0, 4, 7],
    [2, 5, 9],
  ],
  toggle: [
    [0, 5, 7],
    [0, 3, 7],
    [2, 7, 10],
  ],
  danger: [
    [0, 3, 6],
    [0, 6, 9],
    [0, 3, 7],
  ],
  warning: [
    [0, 4, 8],
    [0, 5, 9],
    [2, 6, 10],
  ],
  confirm: [
    [0, 4, 7],
    [0, 5, 9],
    [2, 7, 11],
  ],
};

export const DEFAULT_SOUND_MAP: SoundMap = {
  'section:dashboard': 'overview',
  'section:organism': 'organism',
  'section:pipeline': 'pipeline',
  'section:streams': 'streams',
  'section:telemetry': 'telemetry',
  'section:wallet': 'wallet',
  'section:c0d3r': 'c0d3r',
  'section:addressbook': 'addressbook',
  'section:advisories': 'advisories',
  'section:datalab': 'datalab',
  'section:lab': 'lab',
  'section:guardian': 'guardian',
  'section:codegraph': 'codegraph',
  'section:integrations': 'integrations',
  'section:settings': 'settings',
  'section:audiolab': 'audiolab',
  'section:u53rxr080t': 'u53rxr080t',
  'section:branddozer': 'branddozer',
  'section:branddozer_solo': 'branddozer',
  action: 'action',
  link: 'link',
  toggle: 'toggle',
  warning: 'warning',
  danger: 'danger',
  confirm: 'confirm',
};

export const MOTIF_NAMES = Object.keys(MOTIFS).sort();

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
  private droneGate: GainNode | null = null;
  private midiMaster: GainNode | null = null;
  private midiGate: GainNode | null = null;
  private leftOsc: OscillatorNode | null = null;
  private rightOsc: OscillatorNode | null = null;
  private leftPan: StereoPannerNode | null = null;
  private rightPan: StereoPannerNode | null = null;
  private lfo: OscillatorNode | null = null;
  private lfoGain: GainNode | null = null;
  private enabled = false;
  private settings: AmbientSettings = { ...DEFAULT_AMBIENT_SETTINGS };
  private lastChordAt = 0;
  private soundMap: SoundMap = { ...DEFAULT_SOUND_MAP };
  private motifCursor: Record<string, number> = {};
  private droneGateTimer: number | null = null;
  private midiLoopTimer: number | null = null;
  private midiGateTimer: number | null = null;
  private midiData: MidiSequence | null = null;

  isEnabled() {
    return this.enabled;
  }

  getSettings() {
    return { ...this.settings };
  }

  getSoundMap() {
    return { ...this.soundMap };
  }

  setSoundMap(map: SoundMap) {
    this.soundMap = { ...DEFAULT_SOUND_MAP, ...(map || {}) };
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
    if (this.droneGateTimer) {
      window.clearTimeout(this.droneGateTimer);
      this.droneGateTimer = null;
    }
    if (this.midiLoopTimer) {
      window.clearTimeout(this.midiLoopTimer);
      this.midiLoopTimer = null;
    }
    if (this.midiGateTimer) {
      window.clearTimeout(this.midiGateTimer);
      this.midiGateTimer = null;
    }
    this.fadeOut();
  }

  applySettings(settings: Partial<AmbientSettings>) {
    this.settings = normalizeSettings({ ...this.settings, ...settings });
    if (!this.enabled) {
      return;
    }
    this.ensureNodes();
    this.syncParams();
  }

  loadMidi(buffer: ArrayBuffer) {
    const parsed = parseMidiSequence(buffer);
    this.midiData = parsed;
    if (this.enabled) {
      this.syncParams();
    }
  }

  clearMidi() {
    this.midiData = null;
    this.stopMidiPlayback();
  }

  triggerChord(soundId?: string) {
    if (!this.enabled || !this.ctx || !this.master) return;
    const now = this.ctx.currentTime;
    if (now - this.lastChordAt < 0.15) return;
    this.lastChordAt = now;

    const chord = this.pickChord(soundId);
    const keyOffset = keyToSemitone(this.settings.keyRoot);
    const gain = this.ctx.createGain();
    const decayTime = Math.max(0.4, this.settings.decay);
    const stopTime = now + Math.max(this.settings.chordDuration, decayTime);
    const useGate = this.settings.chordGateMode === 'pattern';
    gain.gain.setValueAtTime(0.0001, now);
    if (useGate) {
      const release = Math.min(0.6, Math.max(0.12, this.settings.decay));
      const sustainEnd = Math.max(now + this.settings.attack, stopTime - release);
      gain.gain.linearRampToValueAtTime(this.settings.chordGain, now + this.settings.attack);
      gain.gain.setValueAtTime(this.settings.chordGain, sustainEnd);
      gain.gain.linearRampToValueAtTime(0.0001, stopTime);
    } else {
      gain.gain.linearRampToValueAtTime(this.settings.chordGain, now + this.settings.attack);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + decayTime);
    }

    let outputNode: AudioNode = gain;
    if (useGate) {
      const gateGain = this.ctx.createGain();
      gateGain.gain.setValueAtTime(1, now);
      gain.connect(gateGain);
      gateGain.connect(this.master);
      this.scheduleGatePattern(
        gateGain.gain,
        now,
        stopTime,
        this.settings.chordGateBpm,
        this.settings.chordGateDepth,
        this.settings.chordGatePattern
      );
      outputNode = gateGain;
    } else {
      gain.connect(this.master);
    }

    chord.forEach((interval) => {
      const osc = this.ctx!.createOscillator();
      osc.type = this.settings.chordWaveform || 'sine';
      const freq = this.settings.baseFreq * Math.pow(2, (interval + keyOffset) / 12);
      osc.frequency.setValueAtTime(freq, now);
      osc.connect(outputNode);
      osc.start(now);
      osc.stop(stopTime + 0.1);
    });
  }

  private pickChord(soundId?: string): number[] {
    if (soundId) {
      const mapped = this.soundMap[soundId] || soundId;
      const motif = MOTIFS[mapped];
      if (motif && motif.length) {
        const index = this.motifCursor[mapped] ?? 0;
        this.motifCursor[mapped] = (index + 1) % motif.length;
        return motif[index];
      }
    }
    const chordOptions = CHORDS[this.settings.chordPreset] || CHORDS.dream_minor;
    return chordOptions[Math.floor(Math.random() * chordOptions.length)];
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

    if (!this.droneGate) {
      this.droneGate = this.ctx.createGain();
      this.droneGate.gain.setValueAtTime(1, this.ctx.currentTime);
      this.droneGate.connect(this.master);
    }

    if (!this.midiMaster) {
      this.midiMaster = this.ctx.createGain();
      this.midiMaster.gain.setValueAtTime(0.0001, this.ctx.currentTime);
      this.midiMaster.connect(this.master);
    }

    if (!this.midiGate) {
      this.midiGate = this.ctx.createGain();
      this.midiGate.gain.setValueAtTime(1, this.ctx.currentTime);
      this.midiGate.connect(this.midiMaster);
    }

    if (!this.leftOsc || !this.rightOsc) {
      this.leftOsc = this.ctx.createOscillator();
      this.rightOsc = this.ctx.createOscillator();
      this.leftOsc.type = this.settings.droneWaveform || 'sine';
      this.rightOsc.type = this.settings.droneWaveform || 'sine';

      this.leftPan = this.ctx.createStereoPanner();
      this.rightPan = this.ctx.createStereoPanner();
      this.leftPan.pan.value = -0.4;
      this.rightPan.pan.value = 0.4;

      this.leftOsc.connect(this.leftPan).connect(this.droneGate);
      this.rightOsc.connect(this.rightPan).connect(this.droneGate);
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
    if (
      !this.ctx ||
      !this.master ||
      !this.leftOsc ||
      !this.rightOsc ||
      !this.lfo ||
      !this.lfoGain ||
      !this.droneGate ||
      !this.midiMaster ||
      !this.midiGate
    ) {
      return;
    }
    const now = this.ctx.currentTime;
    const base = this.settings.baseFreq;
    const detune = this.settings.detuneHz;
    this.leftOsc.type = this.settings.droneWaveform || 'sine';
    this.rightOsc.type = this.settings.droneWaveform || 'sine';
    this.leftOsc.frequency.setTargetAtTime(Math.max(30, base - detune / 2), now, 0.08);
    this.rightOsc.frequency.setTargetAtTime(Math.max(30, base + detune / 2), now, 0.08);
    this.master.gain.setTargetAtTime(this.settings.gain, now, 0.12);
    this.midiMaster.gain.setTargetAtTime(this.settings.midiGain, now, 0.12);
    this.lfo.frequency.setTargetAtTime(this.settings.lfoRate, now, 0.2);
    this.lfoGain.gain.setTargetAtTime(this.settings.lfoDepth, now, 0.2);
    this.syncDroneGate();
    this.syncMidiGate();
    this.syncBackgroundMode();
  }

  private fadeOut() {
    if (!this.ctx || !this.master) return;
    const now = this.ctx.currentTime;
    this.master.gain.setTargetAtTime(0.0001, now, 0.2);
  }

  private scheduleGatePattern(
    param: AudioParam,
    now: number,
    stopTime: number,
    bpmValue: number,
    depthValue: number,
    patternValue: number[]
  ) {
    param.cancelScheduledValues(now);
    const bpm = Math.max(30, bpmValue);
    const stepSeconds = 60 / bpm / 4;
    const depth = clamp(depthValue, 0, 1);
    const pattern = normalizeGatePattern(patternValue);
    const steps = Math.max(1, Math.ceil((stopTime - now) / stepSeconds));
    const ramp = Math.min(0.005, stepSeconds * 0.2);
    let t = now;
    let lastValue = clamp(pattern[0] ?? 1, 0, 1);
    for (let i = 0; i < steps; i += 1) {
      const raw = clamp(pattern[i % pattern.length] ?? 1, 0, 1);
      const value = (1 - depth) + depth * raw;
      if (i === 0) {
        param.setValueAtTime(value, t);
      } else {
        param.setValueAtTime(lastValue, t);
        param.linearRampToValueAtTime(value, t + ramp);
      }
      lastValue = value;
      t += stepSeconds;
      if (t > stopTime) break;
    }
  }

  private syncDroneGate() {
    if (!this.ctx || !this.droneGate) return;
    if (this.droneGateTimer) {
      window.clearTimeout(this.droneGateTimer);
      this.droneGateTimer = null;
    }
    if (this.settings.droneGateMode !== 'pattern') {
      this.droneGate.gain.cancelScheduledValues(this.ctx.currentTime);
      this.droneGate.gain.setTargetAtTime(1, this.ctx.currentTime, 0.05);
      return;
    }
    this.scheduleDroneGateLoop();
  }

  private syncMidiGate() {
    if (!this.ctx || !this.midiGate) return;
    if (this.midiGateTimer) {
      window.clearTimeout(this.midiGateTimer);
      this.midiGateTimer = null;
    }
    if (this.settings.midiGateMode !== 'pattern') {
      this.midiGate.gain.cancelScheduledValues(this.ctx.currentTime);
      this.midiGate.gain.setTargetAtTime(1, this.ctx.currentTime, 0.05);
      return;
    }
    this.scheduleMidiGateLoop();
  }

  private scheduleDroneGateLoop() {
    if (!this.ctx || !this.droneGate) return;
    const now = this.ctx.currentTime;
    const bpm = Math.max(30, this.settings.droneGateBpm);
    const stepSeconds = 60 / bpm / 4;
    const steps = 64;
    const stopTime = now + stepSeconds * steps;
    this.droneGate.gain.cancelScheduledValues(now);
    this.scheduleGatePattern(
      this.droneGate.gain,
      now,
      stopTime,
      this.settings.droneGateBpm,
      this.settings.droneGateDepth,
      this.settings.droneGatePattern
    );
    const loopMs = stepSeconds * steps * 1000 * 0.9;
    this.droneGateTimer = window.setTimeout(() => this.scheduleDroneGateLoop(), Math.max(50, loopMs));
  }

  private scheduleMidiGateLoop() {
    if (!this.ctx || !this.midiGate) return;
    const now = this.ctx.currentTime;
    const bpm = Math.max(30, this.settings.midiGateBpm);
    const stepSeconds = 60 / bpm / 4;
    const steps = 64;
    const stopTime = now + stepSeconds * steps;
    this.midiGate.gain.cancelScheduledValues(now);
    this.scheduleGatePattern(
      this.midiGate.gain,
      now,
      stopTime,
      this.settings.midiGateBpm,
      this.settings.midiGateDepth,
      this.settings.midiGatePattern
    );
    const loopMs = stepSeconds * steps * 1000 * 0.9;
    this.midiGateTimer = window.setTimeout(() => this.scheduleMidiGateLoop(), Math.max(50, loopMs));
  }

  private syncBackgroundMode() {
    if (!this.ctx || !this.droneGate || !this.midiMaster || !this.midiGate) return;
    const now = this.ctx.currentTime;
    if (this.settings.backgroundSource === 'midi' && this.midiData) {
      this.droneGate.gain.cancelScheduledValues(now);
      this.droneGate.gain.setTargetAtTime(0.0001, now, 0.05);
      this.startMidiPlayback();
    } else {
      this.stopMidiPlayback();
      this.droneGate.gain.cancelScheduledValues(now);
      this.droneGate.gain.setTargetAtTime(1, now, 0.05);
    }
  }

  private startMidiPlayback() {
    if (!this.ctx || !this.midiData || !this.midiGate) return;
    if (this.midiLoopTimer) {
      window.clearTimeout(this.midiLoopTimer);
      this.midiLoopTimer = null;
    }
    const schedule = () => {
      if (!this.ctx || !this.midiData || !this.midiGate) return;
      const startAt = this.ctx.currentTime + 0.1;
      const duration = Math.max(1, this.midiData.duration);
      for (const note of this.midiData.notes) {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = this.settings.midiWaveform || 'sine';
        const freq = 440 * Math.pow(2, (note.note - 69) / 12);
        osc.frequency.setValueAtTime(freq, startAt + note.start);
        const velocity = clamp(note.velocity, 0, 1);
        gain.gain.setValueAtTime(0.0001, startAt + note.start);
        gain.gain.linearRampToValueAtTime(velocity, startAt + note.start + 0.01);
        gain.gain.setValueAtTime(velocity, startAt + note.end);
        gain.gain.linearRampToValueAtTime(0.0001, startAt + note.end + 0.02);
        osc.connect(gain).connect(this.midiGate);
        osc.start(startAt + note.start);
        osc.stop(startAt + note.end + 0.05);
      }
      const loopMs = duration * 1000;
      this.midiLoopTimer = window.setTimeout(schedule, Math.max(250, loopMs));
    };
    schedule();
  }

  private stopMidiPlayback() {
    if (this.midiLoopTimer) {
      window.clearTimeout(this.midiLoopTimer);
      this.midiLoopTimer = null;
    }
    if (this.midiGateTimer) {
      window.clearTimeout(this.midiGateTimer);
      this.midiGateTimer = null;
    }
    if (this.ctx && this.midiGate) {
      const now = this.ctx.currentTime;
      this.midiGate.gain.cancelScheduledValues(now);
      this.midiGate.gain.setTargetAtTime(0.0001, now, 0.05);
    }
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

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeGatePattern(pattern?: number[], fallback?: number[]): number[] {
  const base =
    Array.isArray(pattern) && pattern.length
      ? pattern
      : fallback ?? DEFAULT_AMBIENT_SETTINGS.chordGatePattern;
  const cleaned = base.map((value) => clamp(Number(value) || 0, 0, 1));
  if (cleaned.length === 16) return cleaned;
  const out: number[] = [];
  for (let i = 0; i < 16; i += 1) {
    out.push(cleaned[i % cleaned.length] ?? 0);
  }
  return out;
}

function normalizeSettings(settings: AmbientSettings): AmbientSettings {
  const legacy = settings as AmbientSettings & {
    gateEnabled?: boolean;
    gateBpm?: number;
    gateDepth?: number;
    gatePattern?: number[];
    midiEnabled?: boolean;
  };
  return {
    ...settings,
    backgroundSource: settings.backgroundSource ?? (legacy.midiEnabled ? 'midi' : 'drone'),
    chordGateMode: settings.chordGateMode ?? (legacy.gateEnabled ? 'pattern' : 'off'),
    chordGateBpm: settings.chordGateBpm ?? legacy.gateBpm ?? DEFAULT_AMBIENT_SETTINGS.chordGateBpm,
    chordGateDepth: settings.chordGateDepth ?? legacy.gateDepth ?? DEFAULT_AMBIENT_SETTINGS.chordGateDepth,
    chordGatePattern: normalizeGatePattern(
      settings.chordGatePattern ?? legacy.gatePattern,
      DEFAULT_AMBIENT_SETTINGS.chordGatePattern
    ),
    droneGateMode: settings.droneGateMode ?? DEFAULT_AMBIENT_SETTINGS.droneGateMode,
    droneGateBpm: settings.droneGateBpm ?? DEFAULT_AMBIENT_SETTINGS.droneGateBpm,
    droneGateDepth: settings.droneGateDepth ?? DEFAULT_AMBIENT_SETTINGS.droneGateDepth,
    droneGatePattern: normalizeGatePattern(
      settings.droneGatePattern,
      DEFAULT_AMBIENT_SETTINGS.droneGatePattern
    ),
    midiWaveform: settings.midiWaveform ?? DEFAULT_AMBIENT_SETTINGS.midiWaveform,
    midiGain: settings.midiGain ?? DEFAULT_AMBIENT_SETTINGS.midiGain,
    midiGateMode: settings.midiGateMode ?? DEFAULT_AMBIENT_SETTINGS.midiGateMode,
    midiGateBpm: settings.midiGateBpm ?? DEFAULT_AMBIENT_SETTINGS.midiGateBpm,
    midiGateDepth: settings.midiGateDepth ?? DEFAULT_AMBIENT_SETTINGS.midiGateDepth,
    midiGatePattern: normalizeGatePattern(
      settings.midiGatePattern,
      DEFAULT_AMBIENT_SETTINGS.midiGatePattern
    ),
  };
}

type MidiNote = { note: number; velocity: number; start: number; end: number };
type MidiSequence = { notes: MidiNote[]; duration: number };

function parseMidiSequence(buffer: ArrayBuffer): MidiSequence | null {
  const view = new DataView(buffer);
  let offset = 0;
  const readString = (len: number) => {
    let out = '';
    for (let i = 0; i < len; i += 1) {
      out += String.fromCharCode(view.getUint8(offset + i));
    }
    offset += len;
    return out;
  };
  const readUint32 = () => {
    const value = view.getUint32(offset);
    offset += 4;
    return value;
  };
  const readUint16 = () => {
    const value = view.getUint16(offset);
    offset += 2;
    return value;
  };
  const readVarLen = () => {
    let value = 0;
    while (true) {
      const byte = view.getUint8(offset++);
      value = (value << 7) | (byte & 0x7f);
      if ((byte & 0x80) === 0) break;
    }
    return value;
  };

  if (readString(4) !== 'MThd') return null;
  const headerLength = readUint32();
  const format = readUint16();
  const tracks = readUint16();
  const division = readUint16();
  offset += Math.max(0, headerLength - 6);
  if (division & 0x8000) return null;
  const ticksPerQuarter = division;

  const noteEvents: Array<{ tick: number; note: number; velocity: number; on: boolean }> = [];
  const tempoEvents: Array<{ tick: number; tempo: number }> = [{ tick: 0, tempo: 500000 }];

  for (let t = 0; t < tracks; t += 1) {
    if (readString(4) !== 'MTrk') break;
    const trackLength = readUint32();
    const trackEnd = offset + trackLength;
    let tick = 0;
    let runningStatus = 0;
    while (offset < trackEnd) {
      const delta = readVarLen();
      tick += delta;
      let status = view.getUint8(offset++);
      if (status < 0x80) {
        offset -= 1;
        status = runningStatus;
      } else {
        runningStatus = status;
      }
      if (status === 0xff) {
        const type = view.getUint8(offset++);
        const len = readVarLen();
        if (type === 0x51 && len === 3) {
          const tempo =
            (view.getUint8(offset) << 16) | (view.getUint8(offset + 1) << 8) | view.getUint8(offset + 2);
          tempoEvents.push({ tick, tempo });
        }
        offset += len;
        continue;
      }
      if (status === 0xf0 || status === 0xf7) {
        const len = readVarLen();
        offset += len;
        continue;
      }
      const type = status & 0xf0;
      const data1 = view.getUint8(offset++);
      const data2 = type === 0xc0 || type === 0xd0 ? 0 : view.getUint8(offset++);
      if (type === 0x90) {
        if (data2 === 0) {
          noteEvents.push({ tick, note: data1, velocity: 0, on: false });
        } else {
          noteEvents.push({ tick, note: data1, velocity: data2, on: true });
        }
      } else if (type === 0x80) {
        noteEvents.push({ tick, note: data1, velocity: data2, on: false });
      }
    }
    offset = trackEnd;
  }

  tempoEvents.sort((a, b) => a.tick - b.tick);
  const segments: Array<{ startTick: number; startTime: number; tempo: number }> = [];
  let currTempo = tempoEvents[0].tempo;
  let currTick = 0;
  let currTime = 0;
  segments.push({ startTick: 0, startTime: 0, tempo: currTempo });
  for (let i = 1; i < tempoEvents.length; i += 1) {
    const change = tempoEvents[i];
    if (change.tick < currTick) continue;
    const deltaTicks = change.tick - currTick;
    currTime += (deltaTicks * currTempo) / 1_000_000 / ticksPerQuarter;
    currTick = change.tick;
    currTempo = change.tempo;
    segments.push({ startTick: currTick, startTime: currTime, tempo: currTempo });
  }

  const tickToSeconds = (tick: number) => {
    let segment = segments[0];
    for (const seg of segments) {
      if (seg.startTick <= tick) segment = seg;
      else break;
    }
    return segment.startTime + ((tick - segment.startTick) * segment.tempo) / 1_000_000 / ticksPerQuarter;
  };

  noteEvents.sort((a, b) => a.tick - b.tick);
  const noteStacks: Record<number, Array<{ tick: number; velocity: number }>> = {};
  const notes: MidiNote[] = [];
  for (const evt of noteEvents) {
    if (evt.on && evt.velocity > 0) {
      noteStacks[evt.note] = noteStacks[evt.note] || [];
      noteStacks[evt.note].push({ tick: evt.tick, velocity: evt.velocity });
    } else {
      const stack = noteStacks[evt.note];
      if (stack && stack.length) {
        const start = stack.shift()!;
        const startTime = tickToSeconds(start.tick);
        const endTime = tickToSeconds(evt.tick);
        notes.push({
          note: evt.note,
          velocity: clamp(start.velocity / 127, 0, 1),
          start: startTime,
          end: Math.max(startTime + 0.05, endTime),
        });
      }
    }
  }

  const duration = notes.reduce((max, note) => Math.max(max, note.end), 0);
  return { notes, duration };
}
