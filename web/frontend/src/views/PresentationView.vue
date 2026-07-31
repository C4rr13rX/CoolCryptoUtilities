<template>
  <div class="deck-root" :style="rootStyle">
    <!-- Stage: one slide, large type, centred. Everything else is chrome. -->
    <div class="stage" @click="togglePlay">
      <Transition :name="transitionName" mode="out-in">
        <div v-if="current" :key="current.index" class="slide" :class="`kind-${current.kind}`">
          <p v-if="current.section && current.kind === 'body'" class="section">
            {{ current.section }}
          </p>

          <!-- Word-level rendering so narration can highlight in step. -->
          <p class="slide-text" :class="`anim-${wordAnimation}`">
            <!-- The trailing space must be inside the word span: Vue's
                 compiler collapses whitespace between sibling elements,
                 which ran every word together. -->
            <span
              v-for="(word, i) in currentWords"
              :key="i"
              class="word"
              :class="{ spoken: i < spokenIndex, active: i === spokenIndex }"
            >{{ word }}{{ i < currentWords.length - 1 ? ' ' : '' }}</span>
          </p>
        </div>
      </Transition>

      <div v-if="!current" class="empty">
        <p v-if="loading">Loading presentation…</p>
        <p v-else-if="error" class="error">{{ error }}</p>
        <p v-else>No slides.</p>
      </div>
    </div>

    <!-- Progress: thin, unobtrusive, tappable to seek. -->
    <div class="progress" @click.stop="seekFromEvent">
      <div class="progress-fill" :style="{ width: `${progressPct}%` }" />
    </div>

    <div class="controls">
      <button type="button" @click.stop="prev" :disabled="index === 0" aria-label="Previous">‹</button>
      <button type="button" class="play" @click.stop="togglePlay">
        {{ playing ? '❚❚' : '▶' }}
      </button>
      <button
        type="button"
        @click.stop="next"
        :disabled="index >= slides.length - 1"
        aria-label="Next"
      >›</button>
      <span class="counter">{{ index + 1 }} / {{ slides.length }}</span>
      <button type="button" class="ghost" @click.stop="panelOpen = !panelOpen" aria-label="Settings">⚙</button>
    </div>

    <!-- Settings live in a sheet so the stage stays clean on a phone. -->
    <div v-if="panelOpen" class="sheet" @click.stop>
      <label>
        <span>Transition</span>
        <select v-model="transition">
          <option v-for="t in options.transitions" :key="t" :value="t">{{ t }}</option>
        </select>
      </label>
      <label>
        <span>Word animation</span>
        <select v-model="wordAnimation">
          <option v-for="a in options.word_animations" :key="a" :value="a">{{ a }}</option>
        </select>
      </label>
      <label>
        <span>Voice</span>
        <select v-model="voiceId">
          <option v-for="v in voices" :key="v" :value="v">{{ v }}</option>
        </select>
      </label>
      <label>
        <span>Narrate first N slides</span>
        <input v-model.number="narrateLimit" type="number" min="0" max="500" />
      </label>
      <label class="row">
        <input v-model="withScore" type="checkbox" />
        <span>Compose background score</span>
      </label>
      <button type="button" class="btn" :disabled="generating" @click="generateMedia">
        {{ generating ? 'Generating…' : 'Generate narration' }}
      </button>
      <p v-if="mediaNote" class="note">{{ mediaNote }}</p>
      <p v-if="scoreNote" class="note">{{ scoreNote }}</p>
      <button type="button" class="btn ghost" @click="panelOpen = false">Close</button>
    </div>

    <audio ref="audioEl" @ended="onAudioEnded" />
    <audio ref="scoreEl" loop />
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useRoute } from 'vue-router';
import {
  fetchResearchPresentation,
  generateResearchPresentationMedia,
  researchPresentationAudioUrl,
  researchPresentationScoreUrl,
  type PresentationDeck,
  type PresentationSlide,
} from '@/api';

const route = useRoute();
const paperId = computed(() => String(route.params.paperId || ''));

const slides = ref<PresentationSlide[]>([]);
const options = ref<PresentationDeck['options']>({
  transitions: ['crossfade'],
  word_animations: ['highlight'],
  color_schemes: {},
  color_ratios: {},
});
const loading = ref(true);
const error = ref('');
const index = ref(0);
const playing = ref(false);
const panelOpen = ref(false);
const generating = ref(false);
const mediaNote = ref('');
const scoreNote = ref('');

const transition = ref('crossfade');
const wordAnimation = ref('highlight');
const voiceId = ref('Joanna');
const narrateLimit = ref(20);
const withScore = ref(false);
const voices = ['Joanna', 'Matthew', 'Ruth', 'Gregory', 'Danielle', 'Stephen'];

const audioEl = ref<HTMLAudioElement | null>(null);
const scoreEl = ref<HTMLAudioElement | null>(null);

// Elapsed time within the current slide, used to drive word highlighting
// when narration exists and to advance the slide when it does not.
const elapsedMs = ref(0);
let ticker: number | null = null;

const current = computed<PresentationSlide | null>(() => slides.value[index.value] || null);

const currentWords = computed(() => {
  const slide = current.value;
  if (!slide) return [];
  // Prefer Polly's own tokenisation so highlight indices line up exactly
  // with the spoken words rather than with a naive split.
  if (slide.words?.length) return slide.words.map((w) => w.word);
  return slide.text.split(/\s+/).filter(Boolean);
});

// Which word is being spoken right now. -1 before narration starts.
const spokenIndex = computed(() => {
  const slide = current.value;
  if (!slide?.words?.length) return -1;
  const t = elapsedMs.value;
  for (let i = slide.words.length - 1; i >= 0; i -= 1) {
    if (t >= slide.words[i].start_ms) return i;
  }
  return -1;
});

const transitionName = computed(() => `t-${transition.value}`);

const progressPct = computed(() => {
  if (!slides.value.length) return 0;
  return ((index.value + 1) / slides.value.length) * 100;
});

const rootStyle = computed(() => ({
  '--deck-bg': '#070d18',
  '--deck-fg': '#e8eeff',
  '--deck-accent': '#7ea8ff',
}));

async function load() {
  loading.value = true;
  error.value = '';
  try {
    const deck = await fetchResearchPresentation(paperId.value);
    slides.value = deck.slides || [];
    if (deck.options) options.value = deck.options;
    index.value = 0;
  } catch (err: any) {
    error.value = err?.message || 'Unable to load presentation.';
  } finally {
    loading.value = false;
  }
}

function stopTicker() {
  if (ticker !== null) {
    window.clearInterval(ticker);
    ticker = null;
  }
}

function startTicker() {
  stopTicker();
  const started = performance.now();
  const base = elapsedMs.value;
  ticker = window.setInterval(() => {
    elapsedMs.value = base + (performance.now() - started);
    const slide = current.value;
    if (!slide) return;
    // Slides without narration advance on their estimated duration; with
    // narration the audio's `ended` event drives the change instead, so
    // the deck never runs ahead of the voice.
    if (!slide.audio_url && elapsedMs.value >= slide.duration_ms) next();
  }, 50);
}

function playCurrent() {
  const slide = current.value;
  elapsedMs.value = 0;
  if (!slide) return;
  const el = audioEl.value;
  if (slide.audio_url && el) {
    el.src = researchPresentationAudioUrl(paperId.value, slide.index);
    el.currentTime = 0;
    el.play().catch(() => {
      /* autoplay blocked until the user interacts; the tap that started
         playback satisfies this on mobile */
    });
  }
  startTicker();
}

function togglePlay() {
  playing.value = !playing.value;
  if (playing.value) {
    playCurrent();
    scoreEl.value?.play().catch(() => {});
  } else {
    stopTicker();
    audioEl.value?.pause();
    scoreEl.value?.pause();
  }
}

function next() {
  if (index.value < slides.value.length - 1) {
    index.value += 1;
  } else {
    playing.value = false;
    stopTicker();
  }
}

function prev() {
  if (index.value > 0) index.value -= 1;
}

function onAudioEnded() {
  if (playing.value) next();
}

function seekFromEvent(event: MouseEvent) {
  const target = event.currentTarget as HTMLElement;
  const ratio = (event.clientX - target.getBoundingClientRect().left) / target.clientWidth;
  index.value = Math.max(
    0,
    Math.min(slides.value.length - 1, Math.round(ratio * slides.value.length)),
  );
}

async function generateMedia() {
  generating.value = true;
  mediaNote.value = '';
  scoreNote.value = '';
  try {
    const deck = await generateResearchPresentationMedia(paperId.value, {
      limit: narrateLimit.value,
      voice_id: voiceId.value,
      transition: transition.value,
      word_animation: wordAnimation.value,
      score: withScore.value,
    });
    slides.value = deck.slides || [];
    const narrated = slides.value.filter((s) => s.audio_url).length;
    mediaNote.value = `Narrated ${narrated} slide${narrated === 1 ? '' : 's'}.`;
    if (deck.media_failures?.length) {
      mediaNote.value += ` ${deck.media_failures.length} failed.`;
    }
    if (deck.score?.composed) {
      scoreNote.value = `Score: ${deck.score.key} at ${deck.score.bpm}bpm, ${Math.round(
        (deck.score.alignment?.alignment_rate || 0) * 100,
      )}% aligned to transitions.`;
      if (scoreEl.value) scoreEl.value.src = researchPresentationScoreUrl(paperId.value);
    } else if (deck.score?.error) {
      scoreNote.value = `Score failed: ${deck.score.error}`;
    }
  } catch (err: any) {
    mediaNote.value = err?.message || 'Media generation failed.';
  } finally {
    generating.value = false;
  }
}

watch(index, () => {
  if (playing.value) playCurrent();
  else elapsedMs.value = 0;
});

function onKey(event: KeyboardEvent) {
  if (event.key === 'ArrowRight') next();
  else if (event.key === 'ArrowLeft') prev();
  else if (event.key === ' ') {
    event.preventDefault();
    togglePlay();
  }
}

onMounted(() => {
  // Site header/footer live in the Django shell, outside Vue. Without this
  // the footer sits above the fixed controls and intercepts taps.
  document.body.classList.add('deck-fs');
  load();
  window.addEventListener('keydown', onKey);
});

onBeforeUnmount(() => {
  document.body.classList.remove('deck-fs');
  stopTicker();
  window.removeEventListener('keydown', onKey);
});
</script>

<style scoped>
/* Mobile-first: the stage owns the viewport, chrome is minimal and
   thumb-reachable at the bottom. */
.deck-root {
  position: fixed;
  inset: 0;
  display: flex;
  flex-direction: column;
  background: var(--deck-bg);
  color: var(--deck-fg);
  overflow: hidden;
}

.stage {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem 1.25rem;
  cursor: pointer;
  min-height: 0;
}

.slide {
  width: 100%;
  max-width: 34rem;
  text-align: center;
}

.section {
  margin: 0 0 0.75rem;
  font-size: 0.8rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  opacity: 0.5;
}

/* Large type that still fits a phone. clamp() keeps it readable from a
   360px phone up to a desktop without media queries. */
.slide-text {
  margin: 0;
  font-size: clamp(1.5rem, 7vw, 2.75rem);
  line-height: 1.28;
  font-weight: 500;
  /* Wrap between words only. `break-word` split words mid-syllable
     ("Eva/luating"), which is far harder to read than a short line. */
  overflow-wrap: normal;
  word-break: keep-all;
  hyphens: none;
}

/* Each word is an inline-block so it wraps as a unit rather than
   fragmenting across lines. */
.slide-text .word {
  display: inline-block;
  white-space: pre;
}

.kind-title .slide-text {
  font-size: clamp(1.75rem, 8.5vw, 3.5rem);
  font-weight: 700;
}

.kind-heading .slide-text,
.kind-subtitle .slide-text {
  font-size: clamp(1.35rem, 6vw, 2.25rem);
  color: var(--deck-accent);
  letter-spacing: 0.02em;
}

.kind-quote .slide-text {
  font-style: italic;
  opacity: 0.9;
}

.kind-citation .slide-text {
  /* URLs and keys are reference material: small, wrappable, never read. */
  font-size: clamp(0.8rem, 3.2vw, 1rem);
  opacity: 0.65;
  word-break: break-all;
  font-family: ui-monospace, monospace;
}

/* Word states drive every animation option. */
.word {
  transition: color 160ms ease, opacity 160ms ease, transform 160ms ease;
}

.anim-highlight .word.active { color: var(--deck-accent); }
.anim-highlight .word.spoken { opacity: 0.55; }

.anim-fade_in .word { opacity: 0.25; }
.anim-fade_in .word.spoken,
.anim-fade_in .word.active { opacity: 1; }

.anim-pop .word.active { display: inline-block; transform: scale(1.12); }

.anim-rise .word { display: inline-block; transform: translateY(0.12em); opacity: 0.4; }
.anim-rise .word.spoken,
.anim-rise .word.active { transform: translateY(0); opacity: 1; }

.anim-typewriter .word { opacity: 0; }
.anim-typewriter .word.spoken,
.anim-typewriter .word.active { opacity: 1; }

.anim-underline .word.active { border-bottom: 2px solid var(--deck-accent); }

.progress {
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  cursor: pointer;
  flex: 0 0 auto;
}

.progress-fill {
  height: 100%;
  background: var(--deck-accent);
  transition: width 200ms ease;
}

.controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem calc(0.75rem + env(safe-area-inset-bottom));
  background: rgba(0, 0, 0, 0.35);
  flex: 0 0 auto;
}

.controls button {
  min-width: 3rem;
  min-height: 3rem;   /* thumb-sized touch targets */
  border-radius: 10px;
  border: 1px solid rgba(126, 168, 255, 0.28);
  background: rgba(126, 168, 255, 0.1);
  color: var(--deck-fg);
  font-size: 1.15rem;
  cursor: pointer;
}

.controls button:disabled { opacity: 0.3; cursor: default; }
.controls .play { flex: 1; }
.controls .ghost { background: transparent; }

.counter {
  font-size: 0.8rem;
  opacity: 0.6;
  min-width: 5.5rem;
  text-align: center;
}

.sheet {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  max-height: 70vh;
  overflow-y: auto;
  padding: 1rem 1rem calc(1rem + env(safe-area-inset-bottom));
  background: #0b1424;
  border-top: 1px solid rgba(126, 168, 255, 0.3);
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}

.sheet label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.85rem;
}

.sheet label.row {
  flex-direction: row;
  align-items: center;
  gap: 0.5rem;
}

.sheet select,
.sheet input[type='number'] {
  padding: 0.6rem;
  border-radius: 8px;
  border: 1px solid rgba(126, 168, 255, 0.3);
  background: rgba(3, 8, 18, 0.9);
  color: var(--deck-fg);
  font-size: 1rem;   /* >=16px stops iOS zooming on focus */
}

.btn {
  padding: 0.8rem;
  border-radius: 10px;
  border: 1px solid rgba(126, 168, 255, 0.4);
  background: rgba(126, 168, 255, 0.16);
  color: var(--deck-fg);
  font-size: 1rem;
  cursor: pointer;
}

.btn.ghost { background: transparent; }
.note { margin: 0; font-size: 0.8rem; opacity: 0.75; }
.empty { opacity: 0.6; text-align: center; }
.error { color: #ff9c9c; }

/* --- Slide transitions -------------------------------------------------- */
.t-cut-enter-active, .t-cut-leave-active { transition: none; }

.t-fade-enter-active, .t-fade-leave-active,
.t-crossfade-enter-active, .t-crossfade-leave-active,
.t-dissolve-enter-active, .t-dissolve-leave-active { transition: opacity 320ms ease; }
.t-fade-enter-from, .t-fade-leave-to,
.t-crossfade-enter-from, .t-crossfade-leave-to,
.t-dissolve-enter-from, .t-dissolve-leave-to { opacity: 0; }

.t-slide_left-enter-active, .t-slide_left-leave-active,
.t-push-enter-active, .t-push-leave-active { transition: transform 320ms ease, opacity 320ms ease; }
.t-slide_left-enter-from, .t-push-enter-from { transform: translateX(28px); opacity: 0; }
.t-slide_left-leave-to, .t-push-leave-to { transform: translateX(-28px); opacity: 0; }

.t-slide_up-enter-active, .t-slide_up-leave-active { transition: transform 320ms ease, opacity 320ms ease; }
.t-slide_up-enter-from { transform: translateY(28px); opacity: 0; }
.t-slide_up-leave-to { transform: translateY(-28px); opacity: 0; }

.t-zoom_in-enter-active, .t-zoom_in-leave-active,
.t-zoom_out-enter-active, .t-zoom_out-leave-active { transition: transform 320ms ease, opacity 320ms ease; }
.t-zoom_in-enter-from { transform: scale(0.92); opacity: 0; }
.t-zoom_in-leave-to { transform: scale(1.06); opacity: 0; }
.t-zoom_out-enter-from { transform: scale(1.08); opacity: 0; }
.t-zoom_out-leave-to { transform: scale(0.94); opacity: 0; }

.t-blur_through-enter-active, .t-blur_through-leave-active { transition: filter 320ms ease, opacity 320ms ease; }
.t-blur_through-enter-from, .t-blur_through-leave-to { filter: blur(8px); opacity: 0; }

.t-wipe-enter-active, .t-wipe-leave-active { transition: clip-path 340ms ease, opacity 340ms ease; }
.t-wipe-enter-from { clip-path: inset(0 100% 0 0); opacity: 0.4; }
.t-wipe-leave-to { clip-path: inset(0 0 0 100%); opacity: 0.4; }

@media (prefers-reduced-motion: reduce) {
  .word,
  [class*='-enter-active'],
  [class*='-leave-active'] { transition-duration: 1ms !important; }
}
</style>
