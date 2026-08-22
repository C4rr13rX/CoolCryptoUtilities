<script setup lang="ts">
/**
 * Video Studio — compile a research deck into an MP4.
 *
 * The renderer (`branddozer/presentation_video.py`) could always do this;
 * nothing ever called it and there was no way in from the menu. This is that
 * way in: pick a paper, choose the format, render, watch it.
 *
 * Rendering is a background job on the server, so this polls rather than
 * waiting on the request — a full deck takes far longer than any HTTP timeout.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import {
  fetchVideoJob,
  fetchVideoStudio,
  startVideoRender,
  videoFileUrl,
} from '../api';

interface VideoState {
  exists: boolean;
  bytes: number;
  status: string;
  percent: number;
}

interface PaperItem {
  id: string;
  title: string;
  updated_at: string;
  has_deck: boolean;
  has_audio: boolean;
  has_score: boolean;
  video: VideoState;
}

interface Options {
  aspects: string[];
  transitions: string[];
  word_animations: string[];
  defaults: { aspect: string; transition: string; word_animation: string };
}

const items = ref<PaperItem[]>([]);
const options = ref<Options | null>(null);
const selected = ref<string>('');
const loading = ref(true);
const error = ref('');
const busy = ref(false);

const aspect = ref('');
const transition = ref('');
const wordAnimation = ref('');
const maxSlides = ref<number | null>(null);

const job = ref<Record<string, any>>({ status: 'idle' });
let poller: number | undefined;

const selectedPaper = computed(() =>
  items.value.find((p) => p.id === selected.value) || null);

const isRendering = computed(() => job.value?.status === 'rendering');

const videoUrl = computed(() =>
  selected.value ? videoFileUrl(selected.value) : '');

function formatBytes(n: number): string {
  if (!n) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(1)} ${units[i]}`;
}

async function load() {
  loading.value = true;
  error.value = '';
  try {
    const data = await fetchVideoStudio();
    items.value = data.items || [];
    options.value = data.options;
    if (options.value) {
      aspect.value = aspect.value || options.value.defaults.aspect;
      transition.value = transition.value || options.value.defaults.transition;
      wordAnimation.value = wordAnimation.value || options.value.defaults.word_animation;
    }
    if (!selected.value && items.value.length) {
      selected.value = items.value[0].id;
      await refreshJob();
    }
  } catch (err: any) {
    error.value = err?.response?.data?.detail || err?.message || 'Failed to load papers';
  } finally {
    loading.value = false;
  }
}

async function refreshJob() {
  if (!selected.value) return;
  try {
    const data = await fetchVideoJob(selected.value);
    job.value = data.job || { status: 'idle' };
    // Keep the list card in step so the badge does not contradict the panel.
    const paper = items.value.find((p) => p.id === selected.value);
    if (paper) {
      paper.video = { ...paper.video, ...data.video, status: job.value.status,
                      percent: job.value.percent ?? 0 };
    }
  } catch { /* transient; the next poll will catch up */ }
}

async function selectPaper(id: string) {
  selected.value = id;
  job.value = { status: 'idle' };
  await refreshJob();
}

async function render() {
  if (!selected.value || busy.value) return;
  busy.value = true;
  error.value = '';
  try {
    const body: Record<string, any> = {
      aspect: aspect.value,
      transition: transition.value,
      word_animation: wordAnimation.value,
    };
    if (maxSlides.value && maxSlides.value > 0) body.max_slides = maxSlides.value;
    await startVideoRender(selected.value, body);
    job.value = { status: 'rendering', percent: 0 };
  } catch (err: any) {
    error.value = err?.response?.data?.detail || 'Render failed to start';
  } finally {
    busy.value = false;
  }
}

onMounted(async () => {
  await load();
  // 3s: fast enough that progress looks live, slow enough that a long render
  // does not generate hundreds of pointless requests.
  poller = window.setInterval(refreshJob, 3000);
});

onBeforeUnmount(() => {
  if (poller) window.clearInterval(poller);
});
</script>

<template>
  <section class="video-studio">
    <header class="vs-head">
      <h1>Video Studio</h1>
      <p class="vs-sub">
        Compile a research deck into a narrated MP4 — 9:16 portrait by default,
        timed to the generated speech.
      </p>
    </header>

    <p v-if="error" class="vs-error" role="alert">{{ error }}</p>

    <div class="vs-grid">
      <!-- Papers -->
      <aside class="vs-list">
        <h2>Papers <span class="count">{{ items.length }}</span></h2>
        <p v-if="loading" class="vs-muted">Loading…</p>
        <p v-else-if="!items.length" class="vs-muted">
          No research papers yet. Generate one in BrandDozer first.
        </p>
        <button
          v-for="paper in items"
          :key="paper.id"
          type="button"
          class="vs-card"
          :class="{ active: paper.id === selected }"
          @click="selectPaper(paper.id)"
        >
          <span class="vs-card__title">{{ paper.title }}</span>
          <span class="vs-card__meta">
            <span v-if="paper.video.exists" class="badge ok">
              MP4 {{ formatBytes(paper.video.bytes) }}
            </span>
            <span v-else-if="paper.video.status === 'rendering'" class="badge busy">
              {{ paper.video.percent }}%
            </span>
            <span v-else class="badge">not rendered</span>
            <span v-if="!paper.has_audio" class="badge warn">no narration</span>
          </span>
        </button>
      </aside>

      <!-- Render + preview -->
      <div class="vs-main">
        <template v-if="selectedPaper">
          <h2 class="vs-title">{{ selectedPaper.title }}</h2>

          <div class="vs-controls" v-if="options">
            <label>
              <span>Aspect</span>
              <select v-model="aspect" :disabled="isRendering">
                <option v-for="a in options.aspects" :key="a" :value="a">{{ a }}</option>
              </select>
            </label>
            <label>
              <span>Transition</span>
              <select v-model="transition" :disabled="isRendering">
                <option v-for="tr in options.transitions" :key="tr" :value="tr">{{ tr }}</option>
              </select>
            </label>
            <label>
              <span>Word animation</span>
              <select v-model="wordAnimation" :disabled="isRendering">
                <option v-for="w in options.word_animations" :key="w" :value="w">{{ w }}</option>
              </select>
            </label>
            <label>
              <span>Max slides</span>
              <input
                v-model.number="maxSlides"
                type="number"
                min="0"
                placeholder="all"
                :disabled="isRendering"
              />
            </label>
          </div>

          <p v-if="!selectedPaper.has_audio" class="vs-note">
            This deck has no narration yet, so the video will be silent.
            Generate the presentation media first for voiced output.
          </p>

          <div class="vs-actions">
            <button
              class="vs-render"
              type="button"
              :disabled="busy || isRendering"
              @click="render"
            >
              {{ isRendering ? 'Rendering…' : 'Render video' }}
            </button>
            <a
              v-if="selectedPaper.video.exists && !isRendering"
              class="vs-download"
              :href="videoUrl"
              download
            >Download</a>
          </div>

          <div v-if="isRendering" class="vs-progress">
            <div class="vs-bar"><span :style="{ width: `${job.percent || 0}%` }" /></div>
            <span class="vs-progress__text">
              {{ job.percent || 0 }}%
              <template v-if="job.slides_total">
                — slide {{ job.slides_done }} of {{ job.slides_total }}
              </template>
            </span>
          </div>

          <p v-else-if="job.status === 'error'" class="vs-error">
            {{ job.error }}
          </p>

          <!-- key: force the element to reload when a new render lands -->
          <video
            v-if="selectedPaper.video.exists && !isRendering"
            :key="`${selected}-${selectedPaper.video.bytes}`"
            class="vs-player"
            :src="videoUrl"
            controls
            playsinline
          />
        </template>
        <p v-else class="vs-muted">Select a paper to begin.</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.video-studio { padding: 1.25rem; max-width: 1400px; margin: 0 auto; }
.vs-head h1 { margin: 0 0 0.25rem; font-size: 1.4rem; }
.vs-sub { margin: 0 0 1.25rem; opacity: 0.7; font-size: 0.9rem; }
.vs-grid { display: grid; grid-template-columns: minmax(240px, 340px) 1fr; gap: 1.25rem; }
@media (max-width: 900px) { .vs-grid { grid-template-columns: 1fr; } }

.vs-list h2 { font-size: 0.95rem; margin: 0 0 0.6rem; }
.vs-list .count { opacity: 0.5; font-weight: 400; }
.vs-card {
  display: flex; flex-direction: column; gap: 0.4rem; width: 100%;
  text-align: left; padding: 0.7rem 0.8rem; margin-bottom: 0.5rem;
  border: 1px solid rgba(128,128,128,0.28); border-radius: 0.5rem;
  background: transparent; color: inherit; font: inherit; cursor: pointer;
}
.vs-card.active { border-color: #4fc3f7; background: rgba(79,195,247,0.08); }
.vs-card__title {
  font-size: 0.85rem; line-height: 1.3;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden;
}
.vs-card__meta { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.badge {
  font-size: 0.68rem; padding: 0.12rem 0.4rem; border-radius: 0.25rem;
  border: 1px solid rgba(128,128,128,0.4); opacity: 0.85;
}
.badge.ok { border-color: #4caf50; color: #4caf50; }
.badge.busy { border-color: #4fc3f7; color: #4fc3f7; }
.badge.warn { border-color: #ffb300; color: #ffb300; }

.vs-title { font-size: 1rem; margin: 0 0 0.9rem; line-height: 1.35; }
.vs-controls { display: flex; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 0.9rem; }
.vs-controls label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.78rem; }
.vs-controls select, .vs-controls input {
  padding: 0.4rem 0.5rem; border-radius: 0.35rem; font: inherit;
  border: 1px solid rgba(128,128,128,0.4); background: transparent; color: inherit;
}
.vs-controls input { width: 6rem; }

.vs-note {
  font-size: 0.8rem; padding: 0.5rem 0.7rem; margin: 0 0 0.9rem;
  border-left: 3px solid #ffb300; background: rgba(255,179,0,0.08);
}
.vs-actions { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 1rem; }
.vs-render {
  padding: 0.55rem 1.1rem; border: 0; border-radius: 0.4rem;
  background: #2f6feb; color: #fff; font: inherit; cursor: pointer;
}
.vs-render:disabled { opacity: 0.55; cursor: default; }
.vs-download { font-size: 0.85rem; color: #4fc3f7; }

.vs-progress { margin-bottom: 1rem; }
.vs-bar {
  height: 6px; border-radius: 3px; overflow: hidden;
  background: rgba(128,128,128,0.25);
}
.vs-bar span { display: block; height: 100%; background: #4fc3f7; transition: width 0.4s; }
.vs-progress__text { font-size: 0.75rem; opacity: 0.75; }

.vs-player {
  width: 100%; max-width: 420px; border-radius: 0.6rem;
  background: #000; display: block;
}
.vs-muted { opacity: 0.6; font-size: 0.87rem; }
.vs-error { color: #ef5350; font-size: 0.85rem; }
</style>
