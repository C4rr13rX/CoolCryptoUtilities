<template>
  <div class="audio-lab">
    <section class="panel">
      <header>
        <div>
          <h1>{{ t('audio.title') }}</h1>
          <p>{{ t('audio.subtitle') }}</p>
        </div>
        <div class="header-actions">
          <button class="btn" type="button" @click="toggleAudio">
            {{ enabled ? t('audio.disable') : t('audio.enable') }}
          </button>
          <button class="btn ghost" type="button" @click="playChord" :disabled="!enabled">
            {{ t('audio.trigger_chord') }}
          </button>
        </div>
      </header>

      <div class="status-grid">
        <div class="status-card">
          <span class="label">{{ t('common.status') }}</span>
          <span class="value">{{ enabled ? t('common.active') : t('common.idle') }}</span>
        </div>
        <div class="status-card">
          <span class="label">{{ t('audio.chord_preset') }}</span>
          <span class="value">{{ settings.chordPreset }}</span>
        </div>
        <div class="status-card">
          <span class="label">{{ t('audio.key') }}</span>
          <span class="value">{{ keyLabel }}</span>
        </div>
        <div class="status-card">
          <span class="label">{{ t('audio.genre') }}</span>
          <span class="value">{{ genre }}</span>
        </div>
      </div>
    </section>

    <section class="panel">
      <header>
        <h2>{{ t('audio.drone_controls') }}</h2>
        <p>{{ t('audio.drone_subtitle') }}</p>
      </header>
      <div class="controls-grid">
        <label>
          <span>{{ t('audio.base_frequency') }}</span>
          <input v-model.number="settings.baseFreq" type="range" min="48" max="196" step="1" />
          <span class="value">{{ settings.baseFreq.toFixed(0) }}</span>
        </label>
        <label>
          <span>{{ t('audio.binaural_detune') }}</span>
          <input v-model.number="settings.detuneHz" type="range" min="0" max="12" step="0.5" />
          <span class="value">{{ settings.detuneHz.toFixed(1) }}</span>
        </label>
        <label>
          <span>{{ t('audio.drone_gain') }}</span>
          <input v-model.number="settings.gain" type="range" min="0.04" max="0.4" step="0.01" />
          <span class="value">{{ settings.gain.toFixed(2) }}</span>
        </label>
        <label>
          <span>{{ t('audio.lfo_rate') }}</span>
          <input v-model.number="settings.lfoRate" type="range" min="0.02" max="0.4" step="0.01" />
          <span class="value">{{ settings.lfoRate.toFixed(2) }}</span>
        </label>
        <label>
          <span>{{ t('audio.lfo_depth') }}</span>
          <input v-model.number="settings.lfoDepth" type="range" min="0.02" max="0.35" step="0.01" />
          <span class="value">{{ settings.lfoDepth.toFixed(2) }}</span>
        </label>
        <label>
          <span>{{ t('audio.chord_gain') }}</span>
          <input v-model.number="settings.chordGain" type="range" min="0.05" max="0.5" step="0.01" />
          <span class="value">{{ settings.chordGain.toFixed(2) }}</span>
        </label>
        <label>
          <span>{{ t('audio.attack') }}</span>
          <input v-model.number="settings.attack" type="range" min="0.01" max="0.2" step="0.01" />
          <span class="value">{{ settings.attack.toFixed(2) }}</span>
        </label>
        <label>
          <span>{{ t('audio.decay') }}</span>
          <input v-model.number="settings.decay" type="range" min="1" max="6" step="0.1" />
          <span class="value">{{ settings.decay.toFixed(1) }}</span>
        </label>
        <label>
          <span>{{ t('audio.chord_duration') }}</span>
          <input v-model.number="settings.chordDuration" type="range" min="1" max="6" step="0.1" />
          <span class="value">{{ settings.chordDuration.toFixed(1) }}</span>
        </label>
        <label>
          <span>{{ t('audio.chord_preset') }}</span>
          <select v-model="settings.chordPreset">
            <option value="dream_minor">{{ t('audio.preset_dream_minor') }}</option>
            <option value="soft_major">{{ t('audio.preset_soft_major') }}</option>
            <option value="suspended">{{ t('audio.preset_suspended') }}</option>
            <option value="shimmer">{{ t('audio.preset_shimmer') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.background_source') }}</span>
          <select v-model="settings.backgroundSource">
            <option value="drone">{{ t('audio.source_drone') }}</option>
            <option value="midi" :disabled="!midiAvailable">{{ t('audio.source_midi') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.drone_waveform') }}</span>
          <select v-model="settings.droneWaveform">
            <option value="sine">{{ t('audio.wave_sine') }}</option>
            <option value="triangle">{{ t('audio.wave_triangle') }}</option>
            <option value="square">{{ t('audio.wave_square') }}</option>
            <option value="sawtooth">{{ t('audio.wave_sawtooth') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.chord_waveform') }}</span>
          <select v-model="settings.chordWaveform">
            <option value="sine">{{ t('audio.wave_sine') }}</option>
            <option value="triangle">{{ t('audio.wave_triangle') }}</option>
            <option value="square">{{ t('audio.wave_square') }}</option>
            <option value="sawtooth">{{ t('audio.wave_sawtooth') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.midi_waveform') }}</span>
          <select v-model="settings.midiWaveform">
            <option value="sine">{{ t('audio.wave_sine') }}</option>
            <option value="triangle">{{ t('audio.wave_triangle') }}</option>
            <option value="square">{{ t('audio.wave_square') }}</option>
            <option value="sawtooth">{{ t('audio.wave_sawtooth') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.midi_gain') }}</span>
          <input v-model.number="settings.midiGain" type="range" min="0" max="0.5" step="0.01" />
          <span class="value">{{ settings.midiGain.toFixed(2) }}</span>
        </label>
        <label>
          <span>{{ t('audio.chord_gate_mode') }}</span>
          <select v-model="settings.chordGateMode">
            <option value="off">{{ t('audio.gate_off') }}</option>
            <option value="pattern">{{ t('audio.gate_pattern') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.chord_gate_bpm') }}</span>
          <input v-model.number="settings.chordGateBpm" type="range" min="40" max="160" step="1" />
          <span class="value">{{ settings.chordGateBpm.toFixed(0) }}</span>
        </label>
        <label>
          <span>{{ t('audio.chord_gate_depth') }}</span>
          <input v-model.number="settings.chordGateDepth" type="range" min="0" max="1" step="0.05" />
          <span class="value">{{ settings.chordGateDepth.toFixed(2) }}</span>
        </label>
        <div class="gate-pattern">
          <div class="gate-pattern-header">
            <span>{{ t('audio.chord_gate_pattern') }}</span>
            <button class="btn ghost" type="button" @click="resetChordGatePattern">
              {{ t('audio.reset_pattern') }}
            </button>
          </div>
          <div class="gate-grid">
            <div v-for="(val, idx) in settings.chordGatePattern" :key="`chord-gate-${idx}`" class="gate-cell">
              <div class="gate-bar">
                <span class="fill" :style="{ height: `${Math.round(val * 100)}%` }"></span>
              </div>
              <input
                class="gate-slider"
                type="range"
                min="0"
                max="1"
                step="0.01"
                v-model.number="settings.chordGatePattern[idx]"
              />
              <span class="gate-label">{{ idx + 1 }}</span>
            </div>
          </div>
        </div>
        <label>
          <span>{{ t('audio.drone_gate_mode') }}</span>
          <select v-model="settings.droneGateMode">
            <option value="off">{{ t('audio.gate_off') }}</option>
            <option value="pattern">{{ t('audio.gate_pattern') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.drone_gate_bpm') }}</span>
          <input v-model.number="settings.droneGateBpm" type="range" min="40" max="160" step="1" />
          <span class="value">{{ settings.droneGateBpm.toFixed(0) }}</span>
        </label>
        <label>
          <span>{{ t('audio.drone_gate_depth') }}</span>
          <input v-model.number="settings.droneGateDepth" type="range" min="0" max="1" step="0.05" />
          <span class="value">{{ settings.droneGateDepth.toFixed(2) }}</span>
        </label>
        <div class="gate-pattern">
          <div class="gate-pattern-header">
            <span>{{ t('audio.drone_gate_pattern') }}</span>
            <button class="btn ghost" type="button" @click="resetDroneGatePattern">
              {{ t('audio.reset_pattern') }}
            </button>
          </div>
          <div class="gate-grid">
            <div v-for="(val, idx) in settings.droneGatePattern" :key="`drone-gate-${idx}`" class="gate-cell">
              <div class="gate-bar">
                <span class="fill" :style="{ height: `${Math.round(val * 100)}%` }"></span>
              </div>
              <input
                class="gate-slider"
                type="range"
                min="0"
                max="1"
                step="0.01"
                v-model.number="settings.droneGatePattern[idx]"
              />
              <span class="gate-label">{{ idx + 1 }}</span>
            </div>
          </div>
        </div>
        <label>
          <span>{{ t('audio.midi_gate_mode') }}</span>
          <select v-model="settings.midiGateMode">
            <option value="off">{{ t('audio.gate_off') }}</option>
            <option value="pattern">{{ t('audio.gate_pattern') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.midi_gate_bpm') }}</span>
          <input v-model.number="settings.midiGateBpm" type="range" min="40" max="160" step="1" />
          <span class="value">{{ settings.midiGateBpm.toFixed(0) }}</span>
        </label>
        <label>
          <span>{{ t('audio.midi_gate_depth') }}</span>
          <input v-model.number="settings.midiGateDepth" type="range" min="0" max="1" step="0.05" />
          <span class="value">{{ settings.midiGateDepth.toFixed(2) }}</span>
        </label>
        <div class="gate-pattern">
          <div class="gate-pattern-header">
            <span>{{ t('audio.midi_gate_pattern') }}</span>
            <button class="btn ghost" type="button" @click="resetMidiGatePattern">
              {{ t('audio.reset_pattern') }}
            </button>
          </div>
          <div class="gate-grid">
            <div v-for="(val, idx) in settings.midiGatePattern" :key="`midi-gate-${idx}`" class="gate-cell">
              <div class="gate-bar">
                <span class="fill" :style="{ height: `${Math.round(val * 100)}%` }"></span>
              </div>
              <input
                class="gate-slider"
                type="range"
                min="0"
                max="1"
                step="0.01"
                v-model.number="settings.midiGatePattern[idx]"
              />
              <span class="gate-label">{{ idx + 1 }}</span>
            </div>
          </div>
        </div>
        <label>
          <span>{{ t('audio.key_root') }}</span>
          <select v-model="settings.keyRoot">
            <option v-for="note in keyRoots" :key="note" :value="note">{{ note }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.key_mode') }}</span>
          <select v-model="settings.keyMode">
            <option value="major">{{ t('audio.key_major') }}</option>
            <option value="minor">{{ t('audio.key_minor') }}</option>
          </select>
        </label>
        <label>
          <span>{{ t('audio.genre_hint') }}</span>
          <input v-model="genre" type="text" :placeholder="t('audio.genre_placeholder')" />
        </label>
        <label>
          <span>{{ t('audio.midi_background') }}</span>
          <input type="file" accept=".mid,.midi" @change="handleMidi" />
          <span class="value">{{ midiLabel }}</span>
        </label>
      </div>
      <p class="muted note">
        {{ t('audio.engine_note') }}
      </p>
    </section>

    <section class="panel">
      <header>
        <h2>{{ t('audio.section_chords') }}</h2>
        <p>{{ t('audio.section_subtitle') }}</p>
      </header>
      <div class="sound-grid">
        <label v-for="target in soundTargets" :key="target.id">
          <span>{{ target.label }}</span>
          <select v-model="soundMap[target.id]">
            <option v-for="motif in motifOptions" :key="motif" :value="motif">{{ motif }}</option>
          </select>
        </label>
      </div>
      <div class="sound-actions">
        <button class="btn ghost" type="button" @click="resetSoundMap">{{ t('audio.reset_defaults') }}</button>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue';
import {
  ambientAudio,
  DEFAULT_AMBIENT_SETTINGS,
  DEFAULT_SOUND_MAP,
  MOTIF_NAMES,
  AmbientSettings,
  SoundMap,
} from '@/audio/ambient';
import { t } from '@/i18n';

const enabled = ref(false);
const settings = reactive<AmbientSettings>({
  ...DEFAULT_AMBIENT_SETTINGS,
  chordGatePattern: DEFAULT_AMBIENT_SETTINGS.chordGatePattern.slice(),
  droneGatePattern: DEFAULT_AMBIENT_SETTINGS.droneGatePattern.slice(),
  midiGatePattern: DEFAULT_AMBIENT_SETTINGS.midiGatePattern.slice(),
});
const genre = ref('ambient');
const midiFile = ref<File | null>(null);
const detectedMidiKey = ref<{ root: string; mode: 'major' | 'minor' } | null>(null);
const soundMap = reactive<SoundMap>({ ...DEFAULT_SOUND_MAP });
const keyRoots = ['C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const motifOptions = MOTIF_NAMES;
const midiAvailable = computed(() => Boolean(midiFile.value));

const soundTargets = computed(() => [
  { id: 'section:dashboard', label: t('nav.overview') },
  { id: 'section:organism', label: t('nav.organism') },
  { id: 'section:pipeline', label: t('nav.pipeline') },
  { id: 'section:streams', label: t('nav.streams') },
  { id: 'section:telemetry', label: t('nav.telemetry') },
  { id: 'section:wallet', label: t('nav.wallet') },
  { id: 'section:c0d3r', label: t('nav.c0d3r') },
  { id: 'section:addressbook', label: t('nav.addressbook') },
  { id: 'section:advisories', label: t('nav.advisories') },
  { id: 'section:datalab', label: t('nav.datalab') },
  { id: 'section:lab', label: t('nav.lab') },
  { id: 'section:guardian', label: t('nav.guardian') },
  { id: 'section:codegraph', label: t('nav.codegraph') },
  { id: 'section:integrations', label: t('nav.integrations') },
  { id: 'section:settings', label: t('nav.settings') },
  { id: 'section:audiolab', label: t('nav.audiolab') },
  { id: 'section:u53rxr080t', label: t('nav.u53rxr080t') },
  { id: 'section:branddozer', label: t('nav.branddozer') },
  { id: 'section:branddozer_solo', label: t('audio.branddozer_solo') },
  { id: 'action', label: t('audio.sound_primary') },
  { id: 'link', label: t('audio.sound_links') },
  { id: 'toggle', label: t('audio.sound_toggles') },
  { id: 'warning', label: t('audio.sound_warning') },
  { id: 'danger', label: t('audio.sound_danger') },
  { id: 'confirm', label: t('audio.sound_confirm') },
]);

const midiLabel = computed(() => (midiFile.value ? midiFile.value.name : t('audio.none_selected')));
const keyLabel = computed(() => `${settings.keyRoot} ${settings.keyMode}`);

const loadState = () => {
  try {
    const raw = localStorage.getItem('ccu:audio-settings');
    if (raw) {
      const data = JSON.parse(raw);
      const incoming = { ...(data.settings || {}) } as Record<string, any>;
      if (incoming.chordGateMode === undefined && incoming.gateEnabled !== undefined) {
        incoming.chordGateMode = incoming.gateEnabled ? 'pattern' : 'off';
        if (incoming.chordGateBpm === undefined && typeof incoming.gateBpm === 'number') {
          incoming.chordGateBpm = incoming.gateBpm;
        }
        if (incoming.chordGateDepth === undefined && typeof incoming.gateDepth === 'number') {
          incoming.chordGateDepth = incoming.gateDepth;
        }
        if (incoming.chordGatePattern === undefined && Array.isArray(incoming.gatePattern)) {
          incoming.chordGatePattern = incoming.gatePattern;
        }
      }
      Object.assign(settings, incoming);
      settings.chordGatePattern = normalizeGatePattern(
        settings.chordGatePattern,
        DEFAULT_AMBIENT_SETTINGS.chordGatePattern
      );
      settings.droneGatePattern = normalizeGatePattern(
        settings.droneGatePattern,
        DEFAULT_AMBIENT_SETTINGS.droneGatePattern
      );
      settings.midiGatePattern = normalizeGatePattern(
        settings.midiGatePattern,
        DEFAULT_AMBIENT_SETTINGS.midiGatePattern
      );
      if (typeof data.genre === 'string') genre.value = data.genre;
      Object.assign(soundMap, data.soundMap || {});
      enabled.value = Boolean(data.enabled);
    }
    if (settings.backgroundSource === 'midi' && !midiFile.value) {
      settings.backgroundSource = 'drone';
    }
  } catch {
    // ignore
  }
};

const persistState = () => {
  localStorage.setItem(
    'ccu:audio-settings',
    JSON.stringify({
      settings,
      soundMap,
      genre: genre.value,
      enabled: enabled.value,
    })
  );
};

const toggleAudio = async () => {
  if (enabled.value) {
    ambientAudio.disable();
    enabled.value = false;
    persistState();
    return;
  }
  await ambientAudio.enable(settings);
  enabled.value = true;
  persistState();
};

const playChord = () => {
  ambientAudio.triggerChord();
};

const resetSoundMap = () => {
  Object.assign(soundMap, DEFAULT_SOUND_MAP);
  ambientAudio.setSoundMap(soundMap);
  persistState();
};

const handleMidi = async (event: Event) => {
  const input = event.target as HTMLInputElement | null;
  midiFile.value = input?.files?.[0] || null;
  detectedMidiKey.value = null;
  if (!midiFile.value) {
    ambientAudio.clearMidi();
    if (settings.backgroundSource === 'midi') {
      settings.backgroundSource = 'drone';
    }
    return;
  }
  if (!midiFile.value) return;
  try {
    const buffer = await midiFile.value.arrayBuffer();
    ambientAudio.loadMidi(buffer);
    const parsed = parseMidiKey(buffer) || parseKeyFromName(midiFile.value.name);
    if (parsed) {
      detectedMidiKey.value = parsed;
      settings.keyRoot = parsed.root;
      settings.keyMode = parsed.mode;
    }
  } catch {
    // ignore parse errors
  }
};

watch(
  () => ({ ...settings }),
  () => {
    const chordNormalized = normalizeGatePattern(
      settings.chordGatePattern,
      DEFAULT_AMBIENT_SETTINGS.chordGatePattern
    );
    if (!gatePatternEquals(chordNormalized, settings.chordGatePattern)) {
      settings.chordGatePattern = chordNormalized;
    }
    const droneNormalized = normalizeGatePattern(
      settings.droneGatePattern,
      DEFAULT_AMBIENT_SETTINGS.droneGatePattern
    );
    if (!gatePatternEquals(droneNormalized, settings.droneGatePattern)) {
      settings.droneGatePattern = droneNormalized;
    }
    const midiNormalized = normalizeGatePattern(
      settings.midiGatePattern,
      DEFAULT_AMBIENT_SETTINGS.midiGatePattern
    );
    if (!gatePatternEquals(midiNormalized, settings.midiGatePattern)) {
      settings.midiGatePattern = midiNormalized;
    }
    persistState();
    if (enabled.value) {
      ambientAudio.applySettings(settings);
    }
  },
  { deep: true }
);

watch(genre, () => persistState());

watch(
  () => ({ ...soundMap }),
  () => {
    persistState();
    ambientAudio.setSoundMap(soundMap);
  },
  { deep: true }
);

onMounted(() => {
  loadState();
  ambientAudio.setSoundMap(soundMap);
  if (enabled.value) {
    ambientAudio.enable(settings);
  }
});

const normalizeGatePattern = (pattern?: number[], fallback?: number[]) => {
  const base =
    Array.isArray(pattern) && pattern.length
      ? pattern
      : fallback ?? DEFAULT_AMBIENT_SETTINGS.chordGatePattern;
  const cleaned = base.map((val) => Math.min(1, Math.max(0, Number(val) || 0)));
  if (cleaned.length === 16) return cleaned;
  const out: number[] = [];
  for (let i = 0; i < 16; i += 1) {
    out.push(cleaned[i % cleaned.length] ?? 0);
  }
  return out;
};

const gatePatternEquals = (left: number[], right: number[]) => {
  if (left.length !== right.length) return false;
  for (let i = 0; i < left.length; i += 1) {
    if (left[i] !== right[i]) return false;
  }
  return true;
};

const resetChordGatePattern = () => {
  settings.chordGatePattern = DEFAULT_AMBIENT_SETTINGS.chordGatePattern.slice();
};

const resetDroneGatePattern = () => {
  settings.droneGatePattern = DEFAULT_AMBIENT_SETTINGS.droneGatePattern.slice();
};

const resetMidiGatePattern = () => {
  settings.midiGatePattern = DEFAULT_AMBIENT_SETTINGS.midiGatePattern.slice();
};

const parseMidiKey = (buffer: ArrayBuffer): { root: string; mode: 'major' | 'minor' } | null => {
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length - 4; i += 1) {
    if (bytes[i] === 0xff && bytes[i + 1] === 0x59 && bytes[i + 2] === 0x02) {
      const sf = bytes[i + 3] > 127 ? bytes[i + 3] - 256 : bytes[i + 3];
      const mi = bytes[i + 4];
      const mode: 'major' | 'minor' = mi === 1 ? 'minor' : 'major';
      const root = mode === 'major' ? KEY_SIG_MAJOR[sf] : KEY_SIG_MINOR[sf];
      if (root) return { root, mode };
    }
  }
  return null;
};

const parseKeyFromName = (name: string): { root: string; mode: 'major' | 'minor' } | null => {
  const match = name.match(/([A-G])(#{1}|b{1})?(?:_|\s|-)?(major|minor|maj|min|m)?/i);
  if (!match) return null;
  const root = `${match[1].toUpperCase()}${match[2] || ''}` as string;
  const modeRaw = (match[3] || '').toLowerCase();
  const mode: 'major' | 'minor' = modeRaw.startsWith('m') && modeRaw !== 'maj' ? 'minor' : 'major';
  if (!keyRoots.includes(root)) return null;
  return { root, mode };
};

const KEY_SIG_MAJOR: Record<number, string> = {
  [-7]: 'Cb',
  [-6]: 'Gb',
  [-5]: 'Db',
  [-4]: 'Ab',
  [-3]: 'Eb',
  [-2]: 'Bb',
  [-1]: 'F',
  [0]: 'C',
  [1]: 'G',
  [2]: 'D',
  [3]: 'A',
  [4]: 'E',
  [5]: 'B',
  [6]: 'F#',
  [7]: 'C#',
};

const KEY_SIG_MINOR: Record<number, string> = {
  [-7]: 'Ab',
  [-6]: 'Eb',
  [-5]: 'Bb',
  [-4]: 'F',
  [-3]: 'C',
  [-2]: 'G',
  [-1]: 'D',
  [0]: 'A',
  [1]: 'E',
  [2]: 'B',
  [3]: 'F#',
  [4]: 'C#',
  [5]: 'G#',
  [6]: 'D#',
  [7]: 'A#',
};
</script>

<style scoped>
.audio-lab {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.status-card {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 1rem;
}

.status-card .label {
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.2rem;
  color: var(--text-muted);
}

.status-card .value {
  font-size: 1.1rem;
  font-weight: 600;
  color: #f8fbff;
}

.controls-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem 1.5rem;
  margin-top: 1.2rem;
}

.controls-grid label {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.08rem;
  color: var(--text-muted);
}

.controls-grid input[type="range"],
.controls-grid input[type="text"],
.controls-grid select {
  width: 100%;
}

.controls-grid .value {
  font-size: 0.9rem;
  color: #f8fbff;
}

.note {
  margin-top: 1rem;
}

.sound-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem 1.5rem;
  margin-top: 1.2rem;
}

.sound-grid label {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.08rem;
  color: var(--text-muted);
}

.sound-actions {
  margin-top: 1rem;
  display: flex;
  justify-content: flex-end;
}

.gate-pattern {
  grid-column: 1 / -1;
  background: rgba(10, 20, 36, 0.7);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 16px;
  padding: 1rem;
}

.gate-pattern-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.8rem;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  color: var(--text-muted);
}

.gate-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(42px, 1fr));
  gap: 0.6rem;
}

.gate-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.35rem;
}

.gate-bar {
  width: 100%;
  height: 60px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(111, 167, 255, 0.2);
  overflow: hidden;
  display: flex;
  align-items: flex-end;
}

.gate-bar .fill {
  width: 100%;
  background: linear-gradient(180deg, rgba(125, 179, 255, 0.9) 0%, rgba(45, 117, 196, 0.2) 100%);
  transition: height 0.15s ease;
}

.gate-slider {
  width: 100%;
}

.gate-label {
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.6);
}
</style>
