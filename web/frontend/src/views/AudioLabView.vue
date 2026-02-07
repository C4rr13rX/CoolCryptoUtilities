<template>
  <div class="audio-lab">
    <section class="panel">
      <header>
        <div>
          <h1>Audio Lab</h1>
          <p>Ambient drone + reactive chord pulses for the control tower.</p>
        </div>
        <div class="header-actions">
          <button class="btn" type="button" @click="toggleAudio">
            {{ enabled ? 'Disable Drone' : 'Enable Drone' }}
          </button>
          <button class="btn ghost" type="button" @click="playChord" :disabled="!enabled">
            Trigger Chord
          </button>
        </div>
      </header>

      <div class="status-grid">
        <div class="status-card">
          <span class="label">Status</span>
          <span class="value">{{ enabled ? 'Active' : 'Idle' }}</span>
        </div>
        <div class="status-card">
          <span class="label">Chord preset</span>
          <span class="value">{{ settings.chordPreset }}</span>
        </div>
        <div class="status-card">
          <span class="label">Key</span>
          <span class="value">{{ keyLabel }}</span>
        </div>
        <div class="status-card">
          <span class="label">Genre</span>
          <span class="value">{{ genre }}</span>
        </div>
      </div>
    </section>

    <section class="panel">
      <header>
        <h2>Drone Controls</h2>
        <p>Adjust the ambient layer and the reactive chord envelope.</p>
      </header>
      <div class="controls-grid">
        <label>
          <span>Base Frequency (Hz)</span>
          <input v-model.number="settings.baseFreq" type="range" min="48" max="196" step="1" />
          <span class="value">{{ settings.baseFreq.toFixed(0) }}</span>
        </label>
        <label>
          <span>Binaural Detune (Hz)</span>
          <input v-model.number="settings.detuneHz" type="range" min="0" max="12" step="0.5" />
          <span class="value">{{ settings.detuneHz.toFixed(1) }}</span>
        </label>
        <label>
          <span>Drone Gain</span>
          <input v-model.number="settings.gain" type="range" min="0.04" max="0.4" step="0.01" />
          <span class="value">{{ settings.gain.toFixed(2) }}</span>
        </label>
        <label>
          <span>Pulse LFO Rate (Hz)</span>
          <input v-model.number="settings.lfoRate" type="range" min="0.02" max="0.4" step="0.01" />
          <span class="value">{{ settings.lfoRate.toFixed(2) }}</span>
        </label>
        <label>
          <span>LFO Depth</span>
          <input v-model.number="settings.lfoDepth" type="range" min="0.02" max="0.35" step="0.01" />
          <span class="value">{{ settings.lfoDepth.toFixed(2) }}</span>
        </label>
        <label>
          <span>Chord Gain</span>
          <input v-model.number="settings.chordGain" type="range" min="0.05" max="0.5" step="0.01" />
          <span class="value">{{ settings.chordGain.toFixed(2) }}</span>
        </label>
        <label>
          <span>Attack (s)</span>
          <input v-model.number="settings.attack" type="range" min="0.01" max="0.2" step="0.01" />
          <span class="value">{{ settings.attack.toFixed(2) }}</span>
        </label>
        <label>
          <span>Decay (s)</span>
          <input v-model.number="settings.decay" type="range" min="1" max="6" step="0.1" />
          <span class="value">{{ settings.decay.toFixed(1) }}</span>
        </label>
        <label>
          <span>Chord Duration (s)</span>
          <input v-model.number="settings.chordDuration" type="range" min="1" max="6" step="0.1" />
          <span class="value">{{ settings.chordDuration.toFixed(1) }}</span>
        </label>
        <label>
          <span>Chord Preset</span>
          <select v-model="settings.chordPreset">
            <option value="dream_minor">dream_minor</option>
            <option value="soft_major">soft_major</option>
            <option value="suspended">suspended</option>
            <option value="shimmer">shimmer</option>
          </select>
        </label>
        <label>
          <span>Key Root</span>
          <select v-model="settings.keyRoot">
            <option v-for="note in keyRoots" :key="note" :value="note">{{ note }}</option>
          </select>
        </label>
        <label>
          <span>Key Mode</span>
          <select v-model="settings.keyMode">
            <option value="major">major</option>
            <option value="minor">minor</option>
          </select>
        </label>
        <label>
          <span>Genre Hint (stored only)</span>
          <input v-model="genre" type="text" placeholder="ambient, synthwave…" />
        </label>
        <label>
          <span>MIDI Background (stored only)</span>
          <input type="file" accept=".mid,.midi" @change="handleMidi" />
          <span class="value">{{ midiLabel }}</span>
        </label>
      </div>
      <p class="muted note">
        The current engine is a local Web Audio synth. You can still store genre/MIDI choices now,
        and we can wire them into a local AI model pipeline next.
      </p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue';
import { ambientAudio, DEFAULT_AMBIENT_SETTINGS, AmbientSettings } from '@/audio/ambient';

const enabled = ref(false);
const settings = reactive<AmbientSettings>({ ...DEFAULT_AMBIENT_SETTINGS });
const genre = ref('ambient');
const midiFile = ref<File | null>(null);
const detectedMidiKey = ref<{ root: string; mode: 'major' | 'minor' } | null>(null);
const keyRoots = ['C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];

const midiLabel = computed(() => (midiFile.value ? midiFile.value.name : 'none selected'));
const keyLabel = computed(() => `${settings.keyRoot} ${settings.keyMode}`);

const loadState = () => {
  try {
    const raw = localStorage.getItem('ccu:audio-settings');
    if (raw) {
      const data = JSON.parse(raw);
      Object.assign(settings, data.settings || {});
      if (typeof data.genre === 'string') genre.value = data.genre;
      enabled.value = Boolean(data.enabled);
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

const handleMidi = async (event: Event) => {
  const input = event.target as HTMLInputElement | null;
  midiFile.value = input?.files?.[0] || null;
  detectedMidiKey.value = null;
  if (!midiFile.value) return;
  try {
    const buffer = await midiFile.value.arrayBuffer();
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
    persistState();
    if (enabled.value) {
      ambientAudio.applySettings(settings);
    }
  },
  { deep: true }
);

watch(genre, () => persistState());

onMounted(() => {
  loadState();
  if (enabled.value) {
    ambientAudio.enable(settings);
  }
});

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
</style>
