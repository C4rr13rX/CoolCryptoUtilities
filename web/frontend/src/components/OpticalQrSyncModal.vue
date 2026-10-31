<template>
  <div v-if="modelValue" class="modal-backdrop" @click.self="close">
    <div class="optical-sync-card">
      <header class="modal-header">
        <div class="header-titles">
          <div class="header-badge">
            <span class="live-dot" />
            <span class="badge-text">AIR-GAP OPTICAL SYNC</span>
          </div>
          <h2>Optical Credential Card Deck</h2>
          <p class="caption">
            Point your Rapid Diplomat phone scanner at each card on the monitor. A green toast cue on the phone signals when to flip to the next card.
          </p>
        </div>
        <button class="btn-close" type="button" aria-label="Close" @click="close">✕</button>
      </header>

      <!-- Loading State -->
      <div v-if="loading" class="deck-loading">
        <div class="loading-spinner" />
        <span>Decrypting and generating optical key deck...</span>
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="deck-error">
        <p>{{ error }}</p>
        <button class="btn" type="button" @click="loadDeck">Retry</button>
      </div>

      <!-- No Cards -->
      <div v-else-if="cards.length === 0" class="deck-empty">
        <p>No keys or secure settings configured in this account.</p>
      </div>

      <!-- Main Card Deck Stepper -->
      <div v-else class="deck-body">
        <!-- Progress Bar & Stepper Meta -->
        <div class="deck-stepper-meta">
          <div class="step-indicator">
            <span class="card-num">CARD {{ currentIndex + 1 }}</span>
            <span class="card-total">OF {{ cards.length }}</span>
          </div>
          <div class="card-category-badge" :class="categoryClass">
            {{ activeCard.category.replace('_', ' ').toUpperCase() }}
          </div>
          <div class="auto-advance-control">
            <label class="switch-label">
              <input type="checkbox" v-model="autoAdvance" />
              <span>Auto-Advance</span>
            </label>
            <select v-if="autoAdvance" v-model="autoAdvanceInterval" class="interval-select">
              <option :value="2000">2s</option>
              <option :value="3000">3s</option>
              <option :value="4000">4s</option>
              <option :value="5000">5s</option>
            </select>
          </div>
        </div>

        <!-- Animated Progress Line for Auto-Advance -->
        <div class="deck-progress-track">
          <div
            class="deck-progress-fill"
            :style="{ width: `${progressPercent}%` }"
          />
          <div
            v-if="autoAdvance"
            class="auto-timer-bar"
            :style="{ width: `${autoAdvancePercent}%` }"
          />
        </div>

        <!-- Card Stage -->
        <div class="card-stage">
          <!-- QR Code Scannable Surface -->
          <div class="qr-column">
            <div class="qr-container">
              <div
                class="qr-surface"
                v-html="qrSvg"
                :key="`qr-${activeCard.id}`"
              />
              <div class="qr-target-corners">
                <span class="c-tl" />
                <span class="c-tr" />
                <span class="c-bl" />
                <span class="c-br" />
              </div>
            </div>
            <button class="btn-maximize-qr" type="button" @click="fullscreenQr = true">
              ⛶ Enlarge QR to Screen Size
            </button>
          </div>

          <!-- Key Metadata & Verification -->
          <div class="card-info">
            <div class="card-headline">
              <h3 class="key-label">{{ activeCard.label }}</h3>
              <code class="key-name">{{ activeCard.name }}</code>
            </div>

            <p class="key-details" v-if="activeCard.details">{{ activeCard.details }}</p>

            <!-- Value Reveal / Mask -->
            <div class="value-inspector">
              <span class="value-text" :class="{ masked: !revealed }">
                {{ revealed ? activeCard.value : '••••••••••••••••••••••••••••••••' }}
              </span>
              <button class="btn-micro" type="button" @click="revealed = !revealed">
                {{ revealed ? 'Hide' : 'Reveal' }}
              </button>
              <button class="btn-micro" type="button" @click="copyValue">
                {{ copied ? 'Copied Secret!' : 'Copy Secret' }}
              </button>
            </div>

            <!-- Copy Full Card Payload for Instant Phone Auto-Intake -->
            <button class="btn-copy-payload" type="button" @click="copyPayload">
              {{ copiedPayload ? '✓ Card Payload Copied! Tap Auto-Intake on Phone' : '📋 Copy Card Payload (For Phone Auto-Intake)' }}
            </button>

            <!-- QR Module Density Toggle -->
            <div class="qr-mode-selector">
              <span class="qr-mode-label">QR Pattern Density:</span>
              <button
                class="btn-mode-toggle"
                :class="{ active: qrDensity === 'compact' }"
                type="button"
                @click="qrDensity = 'compact'"
              >
                Chunky High-Contrast (Compact)
              </button>
              <button
                class="btn-mode-toggle"
                :class="{ active: qrDensity === 'full' }"
                type="button"
                @click="qrDensity = 'full'"
              >
                Standard JSON
              </button>
            </div>

            <div class="optical-hint">
              <span class="hint-icon">📱</span>
              <span class="hint-text">
                Point phone camera at the screen or tap "Copy Card Payload" to intake instantly without glare.
              </span>
            </div>
          </div>
        </div>

        <!-- Navigation Controls -->
        <div class="deck-controls">
          <button
            class="btn ghost"
            type="button"
            :disabled="currentIndex === 0"
            @click="prevCard"
          >
            ◀ Previous
          </button>

          <!-- Hotkey navigation hints -->
          <div class="deck-nav-pills">
            <button
              v-for="(c, idx) in cards"
              :key="c.id"
              class="pill-dot"
              :class="{
                active: idx === currentIndex,
                visited: visitedCards.has(idx)
              }"
              :title="`Card ${idx + 1}: ${c.label}`"
              @click="goToCard(idx)"
            >
              {{ idx + 1 }}
            </button>
          </div>

          <button
            class="btn primary"
            type="button"
            :disabled="currentIndex === cards.length - 1"
            @click="nextCard"
          >
            Next ▶
          </button>
        </div>

        <footer class="deck-footer-keys">
          <small>Keyboard: <strong>[←]</strong> Previous • <strong>[→]</strong> or <strong>[Space]</strong> Next • <strong>[Esc]</strong> Close</small>
        </footer>
      </div>
    </div>

    <!-- Fullscreen Screen-Size QR Overlay -->
    <div
      v-if="fullscreenQr"
      class="fullscreen-qr-overlay"
      @click.self="fullscreenQr = false"
    >
      <div class="fullscreen-qr-card">
        <header class="fullscreen-header">
          <div class="fullscreen-title-box">
            <span class="fs-card-tag">CARD {{ currentIndex + 1 }} OF {{ cards.length }}</span>
            <h3>{{ activeCard.label }}</h3>
            <code>{{ activeCard.name }}</code>
          </div>
          <div class="fullscreen-actions">
            <button
              class="btn ghost"
              type="button"
              :disabled="currentIndex === 0"
              @click="prevCard"
            >
              ◀ Prev
            </button>
            <button
              class="btn primary"
              type="button"
              :disabled="currentIndex === cards.length - 1"
              @click="nextCard"
            >
              Next ▶
            </button>
            <button
              class="btn-close-fs"
              type="button"
              @click="fullscreenQr = false"
            >
              ✕ Exit Fullscreen
            </button>
          </div>
        </header>

        <div class="giant-qr-surface" v-html="qrSvg" />

        <footer class="fullscreen-footer">
          <small>Use <strong>[←]</strong> / <strong>[→]</strong> or <strong>[Space]</strong> to navigate cards • <strong>[Esc]</strong> to exit screen-size view</small>
        </footer>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import axios from 'axios';
import { QrCode } from '@/utils/qrCode';

interface KeyCard {
  id: string;
  category: string;
  name: string;
  label: string;
  value: string;
  format: string;
  details?: string;
  qr_payload: string;
  meta?: Record<string, any>;
  index?: number;
  total?: number;
}

const props = defineProps<{
  modelValue: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', val: boolean): void;
}>();

const cards = ref<KeyCard[]>([]);
const currentIndex = ref(0);
const loading = ref(false);
const error = ref<string | null>(null);
const revealed = ref(false);
const copied = ref(false);
const copiedPayload = ref(false);
const qrDensity = ref<'compact' | 'full'>('compact');
const fullscreenQr = ref(false);
const visitedCards = ref<Set<number>>(new Set([0]));

// Auto-advance
const autoAdvance = ref(false);
const autoAdvanceInterval = ref(3000);
const autoAdvancePercent = ref(0);
let autoAdvanceTimer: number | undefined;
let autoAdvanceFrame: number | undefined;
let autoAdvanceStartTime = 0;

const activeCard = computed<KeyCard>(() => {
  return cards.value[currentIndex.value] || {
    id: 'placeholder',
    category: 'unknown',
    name: '',
    label: '',
    value: '',
    format: '',
    qr_payload: '{}'
  };
});

const activePayload = computed(() => {
  const c = activeCard.value;
  if (!c || !c.value) return '';
  if (qrDensity.value === 'compact') {
    const catShort = c.category.includes('mnemonic') ? 'm' : (c.category.includes('private_key') ? 'p' : 's');
    return `ccu://k?n=${encodeURIComponent(c.name)}&v=${encodeURIComponent(c.value)}&c=${catShort}&i=${(c.index || currentIndex.value + 1)}&t=${(c.total || cards.value.length)}`;
  }
  return c.qr_payload;
});

const progressPercent = computed(() => {
  if (cards.value.length === 0) return 0;
  return ((currentIndex.value + 1) / cards.value.length) * 100;
});

const categoryClass = computed(() => {
  const cat = activeCard.value.category.toLowerCase();
  if (cat.includes('wallet')) return 'cat-wallet';
  if (cat.includes('key') || cat.includes('api')) return 'cat-api';
  return 'cat-vault';
});

// Generated QR SVG
const qrSvg = computed(() => {
  const payload = activePayload.value;
  if (!payload) return '';
  try {
    const qr = new QrCode(payload, 'M');
    return qr.toSvg({ margin: 3, color: '#000000', background: '#ffffff' });
  } catch (err) {
    console.warn('QR generation error:', err);
    return '<svg viewBox="0 0 100 100"><text x="10" y="50" fill="red">QR Error</text></svg>';
  }
});

async function loadDeck() {
  loading.value = true;
  error.value = null;
  try {
    const resp = await axios.get('/api/secure/settings/deck/');
    cards.value = resp.data.cards || [];
    currentIndex.value = 0;
    visitedCards.value = new Set([0]);
  } catch (err: any) {
    error.value = err?.response?.data?.detail || err?.message || 'Failed to load key deck';
  } finally {
    loading.value = false;
  }
}

function prevCard() {
  if (currentIndex.value > 0) {
    currentIndex.value--;
    onCardChanged();
  }
}

function nextCard() {
  if (currentIndex.value < cards.value.length - 1) {
    currentIndex.value++;
    onCardChanged();
  } else if (autoAdvance.value) {
    // Reached end; stop auto advance
    autoAdvance.value = false;
  }
}

function goToCard(idx: number) {
  if (idx >= 0 && idx < cards.value.length) {
    currentIndex.value = idx;
    onCardChanged();
  }
}

function onCardChanged() {
  revealed.value = false;
  copied.value = false;
  copiedPayload.value = false;
  visitedCards.value.add(currentIndex.value);
  resetAutoAdvanceTimer();
}

async function copyValue() {
  if (!activeCard.value?.value) return;
  try {
    await navigator.clipboard.writeText(activeCard.value.value);
    copied.value = true;
    setTimeout(() => { copied.value = false; }, 2000);
  } catch {
    // fallback
  }
}

async function copyPayload() {
  const payload = activePayload.value;
  if (!payload) return;
  try {
    await navigator.clipboard.writeText(payload);
    copiedPayload.value = true;
    setTimeout(() => { copiedPayload.value = false; }, 3000);
  } catch {
    // fallback
  }
}

function close() {
  stopAutoAdvance();
  emit('update:modelValue', false);
}

// Auto-advance loop
function startAutoAdvance() {
  resetAutoAdvanceTimer();
}

function stopAutoAdvance() {
  if (autoAdvanceTimer) {
    window.clearTimeout(autoAdvanceTimer);
    autoAdvanceTimer = undefined;
  }
  if (autoAdvanceFrame) {
    cancelAnimationFrame(autoAdvanceFrame);
    autoAdvanceFrame = undefined;
  }
  autoAdvancePercent.value = 0;
}

function resetAutoAdvanceTimer() {
  stopAutoAdvance();
  if (!autoAdvance.value || cards.value.length <= 1) return;

  autoAdvanceStartTime = performance.now();
  const updateProgress = () => {
    if (!autoAdvance.value) return;
    const elapsed = performance.now() - autoAdvanceStartTime;
    autoAdvancePercent.value = Math.min(100, (elapsed / autoAdvanceInterval.value) * 100);
    if (elapsed < autoAdvanceInterval.value) {
      autoAdvanceFrame = requestAnimationFrame(updateProgress);
    }
  };
  autoAdvanceFrame = requestAnimationFrame(updateProgress);

  autoAdvanceTimer = window.setTimeout(() => {
    nextCard();
  }, autoAdvanceInterval.value);
}

watch(autoAdvance, (enabled) => {
  if (enabled) startAutoAdvance();
  else stopAutoAdvance();
});

watch(autoAdvanceInterval, () => {
  if (autoAdvance.value) resetAutoAdvanceTimer();
});

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      loadDeck();
      window.addEventListener('keydown', handleKeyDown);
    } else {
      stopAutoAdvance();
      window.removeEventListener('keydown', handleKeyDown);
    }
  }
);

function handleKeyDown(e: KeyboardEvent) {
  if (!props.modelValue) return;
  if (e.key === 'ArrowRight' || e.key === ' ') {
    e.preventDefault();
    nextCard();
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    prevCard();
  } else if (e.key === 'Escape') {
    e.preventDefault();
    if (fullscreenQr.value) {
      fullscreenQr.value = false;
    } else {
      close();
    }
  }
}

onMounted(() => {
  if (props.modelValue) {
    loadDeck();
    window.addEventListener('keydown', handleKeyDown);
  }
});

onBeforeUnmount(() => {
  stopAutoAdvance();
  window.removeEventListener('keydown', handleKeyDown);
});
</script>

<style scoped>
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(2, 6, 12, 0.88);
  backdrop-filter: blur(8px);
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}

.optical-sync-card {
  width: 100%;
  max-width: 640px;
  background: #090f1b;
  border: 1px solid rgba(120, 180, 255, 0.35);
  box-shadow: 0 0 40px rgba(0, 220, 255, 0.15), 0 20px 40px rgba(0, 0, 0, 0.8);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-family: inherit;
}

.modal-header {
  padding: 1.25rem 1.5rem 1rem;
  border-bottom: 1px solid rgba(120, 180, 255, 0.15);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  background: rgba(14, 23, 40, 0.6);
}

.header-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  color: #34d399;
  font-weight: 700;
  margin-bottom: 0.3rem;
}

.live-dot {
  width: 8px;
  height: 8px;
  border-radius: 50% !important;
  background: #34d399;
  box-shadow: 0 0 8px #34d399;
  animation: pulse 1.8s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.85); }
}

.header-titles h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: 0.05em;
}

.caption {
  margin: 0.3rem 0 0;
  font-size: 0.8rem;
  color: #94a3b8;
  line-height: 1.4;
}

.btn-close {
  background: transparent;
  border: none;
  color: #94a3b8;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  transition: color 0.15s;
}

.btn-close:hover {
  color: #ff5a5f;
}

.deck-loading, .deck-error, .deck-empty {
  padding: 3rem 2rem;
  text-align: center;
  color: #94a3b8;
}

.deck-body {
  padding: 1.25rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.deck-stepper-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.step-indicator {
  display: flex;
  align-items: baseline;
  gap: 0.3rem;
}

.card-num {
  font-size: 1.1rem;
  font-weight: 800;
  color: #38bdf8;
  letter-spacing: 0.08em;
}

.card-total {
  font-size: 0.8rem;
  color: #64748b;
}

.card-category-badge {
  padding: 0.2rem 0.6rem;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  border-radius: 4px;
}

.cat-wallet {
  background: rgba(56, 189, 248, 0.15);
  color: #38bdf8;
  border: 1px solid rgba(56, 189, 248, 0.4);
}

.cat-api {
  background: rgba(245, 158, 11, 0.15);
  color: #fbbf24;
  border: 1px solid rgba(245, 158, 11, 0.4);
}

.cat-vault {
  background: rgba(168, 85, 247, 0.15);
  color: #c084fc;
  border: 1px solid rgba(168, 85, 247, 0.4);
}

.auto-advance-control {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.switch-label {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.75rem;
  color: #cbd5e1;
  cursor: pointer;
}

.interval-select {
  background: #0d1527;
  color: #38bdf8;
  border: 1px solid rgba(120, 180, 255, 0.3);
  padding: 0.15rem 0.3rem;
  font-size: 0.75rem;
}

.deck-progress-track {
  height: 4px;
  background: rgba(255, 255, 255, 0.08);
  position: relative;
  overflow: hidden;
}

.deck-progress-fill {
  height: 100%;
  background: #38bdf8;
  transition: width 0.25s ease;
}

.auto-timer-bar {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: #34d399;
  opacity: 0.8;
}

.card-stage {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 1.5rem;
  align-items: center;
  background: rgba(13, 22, 38, 0.5);
  border: 1px solid rgba(120, 180, 255, 0.15);
  padding: 1.25rem;
}

@media (max-width: 540px) {
  .card-stage {
    grid-template-columns: 1fr;
    justify-items: center;
  }
}

.qr-container {
  position: relative;
  width: 210px;
  height: 210px;
  background: #ffffff;
  padding: 8px;
  box-shadow: 0 0 25px rgba(56, 189, 248, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
}

.qr-surface {
  width: 100%;
  height: 100%;
  display: flex;
}

.qr-surface :deep(svg) {
  width: 100%;
  height: 100%;
}

.qr-target-corners span {
  position: absolute;
  width: 12px;
  height: 12px;
  border-color: #38bdf8;
  border-style: solid;
  pointer-events: none;
}
.c-tl { top: -4px; left: -4px; border-width: 2px 0 0 2px; }
.c-tr { top: -4px; right: -4px; border-width: 2px 2px 0 0; }
.c-bl { bottom: -4px; left: -4px; border-width: 0 0 2px 2px; }
.c-br { bottom: -4px; right: -4px; border-width: 0 2px 2px 0; }

.qr-column {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.btn-maximize-qr {
  background: rgba(56, 189, 248, 0.15);
  border: 1px solid rgba(56, 189, 248, 0.4);
  color: #38bdf8;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.35rem 0.6rem;
  cursor: pointer;
  width: 100%;
  text-align: center;
  transition: all 0.2s ease;
  letter-spacing: 0.03em;
}

.btn-maximize-qr:hover {
  background: rgba(56, 189, 248, 0.3);
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.3);
}

.fullscreen-qr-overlay {
  position: fixed;
  inset: 0;
  z-index: 10000;
  background: rgba(1, 4, 9, 0.96);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}

.fullscreen-qr-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  gap: 0.75rem;
}

.fullscreen-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: min(84vw, 84vh);
  color: #ffffff;
}

.fs-card-tag {
  font-size: 0.72rem;
  color: #38bdf8;
  font-weight: 700;
  letter-spacing: 0.1em;
  display: block;
}

.fullscreen-title-box h3 {
  margin: 0.1rem 0;
  font-size: 1.15rem;
  font-weight: 700;
  color: #ffffff;
}

.fullscreen-title-box code {
  font-size: 0.75rem;
  color: #94a3b8;
}

.fullscreen-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.btn-close-fs {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.4);
  color: #f87171;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.4rem 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-close-fs:hover {
  background: rgba(239, 68, 68, 0.3);
}

.giant-qr-surface {
  width: min(78vw, 78vh);
  height: min(78vw, 78vh);
  background: #ffffff;
  padding: 16px;
  box-shadow: 0 0 50px rgba(56, 189, 248, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
}

.giant-qr-surface :deep(svg) {
  width: 100%;
  height: 100%;
}

.fullscreen-footer {
  color: #94a3b8;
  font-size: 0.75rem;
}

.card-info {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  overflow: hidden;
}

.card-headline h3 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 700;
  color: #f8fafc;
}

.key-name {
  display: inline-block;
  font-size: 0.75rem;
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.1);
  padding: 0.15rem 0.4rem;
  margin-top: 0.3rem;
  word-break: break-all;
}

.key-details {
  margin: 0;
  font-size: 0.8rem;
  color: #94a3b8;
  word-break: break-all;
}

.value-inspector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: rgba(2, 6, 12, 0.6);
  border: 1px solid rgba(120, 180, 255, 0.2);
  padding: 0.4rem 0.6rem;
  font-size: 0.8rem;
}

.value-text {
  flex: 1;
  font-family: monospace;
  font-size: 0.75rem;
  color: #f1f5f9;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.value-text.masked {
  letter-spacing: 0.15em;
  color: #64748b;
}

.btn-micro {
  background: rgba(56, 189, 248, 0.15);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #38bdf8;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  cursor: pointer;
  white-space: nowrap;
}

.btn-micro:hover {
  background: rgba(56, 189, 248, 0.3);
}

.btn-copy-payload {
  background: rgba(52, 211, 153, 0.15);
  border: 1px solid rgba(52, 211, 153, 0.4);
  color: #34d399;
  font-size: 0.78rem;
  font-weight: 700;
  padding: 0.5rem 0.75rem;
  cursor: pointer;
  width: 100%;
  text-align: center;
  transition: all 0.2s ease;
}

.btn-copy-payload:hover {
  background: rgba(52, 211, 153, 0.3);
  box-shadow: 0 0 12px rgba(52, 211, 153, 0.3);
}

.qr-mode-selector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.72rem;
  color: #94a3b8;
  flex-wrap: wrap;
}

.qr-mode-label {
  font-weight: 600;
}

.btn-mode-toggle {
  background: rgba(148, 163, 184, 0.1);
  border: 1px solid rgba(148, 163, 184, 0.2);
  color: #94a3b8;
  font-size: 0.7rem;
  padding: 0.2rem 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-mode-toggle.active {
  background: rgba(56, 189, 248, 0.2);
  border-color: #38bdf8;
  color: #38bdf8;
  font-weight: 700;
}

.optical-hint {
  display: flex;
  align-items: flex-start;
  gap: 0.4rem;
  background: rgba(52, 211, 153, 0.08);
  border-left: 2px solid #34d399;
  padding: 0.4rem 0.6rem;
  font-size: 0.75rem;
  color: #a7f3d0;
  line-height: 1.35;
}

.hint-icon {
  font-size: 0.9rem;
}

.deck-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.btn {
  padding: 0.5rem 1.2rem;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.15s;
}

.btn.primary {
  background: #38bdf8;
  color: #020617;
  border-color: #38bdf8;
}

.btn.primary:hover:not(:disabled) {
  background: #7dd3fc;
  box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
}

.btn.ghost {
  background: transparent;
  color: #cbd5e1;
  border-color: rgba(148, 163, 184, 0.3);
}

.btn.ghost:hover:not(:disabled) {
  border-color: #38bdf8;
  color: #38bdf8;
}

.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.deck-nav-pills {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  overflow-x: auto;
  max-width: 320px;
  padding: 0.2rem;
}

.pill-dot {
  width: 24px;
  height: 24px;
  border-radius: 50% !important;
  border: 1px solid rgba(148, 163, 184, 0.3);
  background: transparent;
  color: #94a3b8;
  font-size: 0.65rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s;
  padding: 0;
  flex-shrink: 0;
}

.pill-dot:hover {
  border-color: #38bdf8;
  color: #38bdf8;
}

.pill-dot.visited {
  border-color: rgba(52, 211, 153, 0.5);
  color: #34d399;
}

.pill-dot.active {
  background: #38bdf8;
  border-color: #38bdf8;
  color: #020617;
  font-weight: 800;
  box-shadow: 0 0 8px rgba(56, 189, 248, 0.5);
}

.deck-footer-keys {
  text-align: center;
  color: #64748b;
  font-size: 0.7rem;
}

.deck-footer-keys strong {
  color: #94a3b8;
}
</style>
