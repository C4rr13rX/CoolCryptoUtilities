<template>
  <Teleport to="body">
    <div v-if="isOpen && steps.length" class="startup-wizard">
      <div class="wizard-backdrop" @click="close"></div>
      <section class="wizard-shell" role="dialog" aria-modal="true">
        <header class="wizard-header">
          <div>
            <span class="eyebrow">{{ eyebrow }}</span>
            <h2>{{ title }}</h2>
            <p>{{ subtitle }}</p>
          </div>
          <button class="close-btn" type="button" @click="close" :aria-label="t('wizard.close')">x</button>
        </header>

        <div class="wizard-status">
          <div class="status-pill">
            {{ t('wizard.step').replace('{current}', String(activeIndex + 1)).replace('{total}', String(steps.length)) }}
          </div>
          <div class="status-dots">
            <button
              v-for="(step, idx) in steps"
              :key="step.id"
              type="button"
              class="dot"
              :class="{ active: idx === activeIndex }"
              @click="goTo(idx)"
              :aria-label="t('wizard.jump_to').replace('{title}', step.title)"
            ></button>
          </div>
        </div>

        <div class="wizard-slider">
          <div class="wizard-track" :style="{ transform: `translateX(-${activeIndex * 100}%)` }">
            <article v-for="step in steps" :key="step.id" class="wizard-slide">
              <div class="slide-card" :class="step.tone ? `tone-${step.tone}` : ''">
                <h3>{{ step.title }}</h3>
                <p>{{ step.description }}</p>
                <small v-if="step.detail">{{ step.detail }}</small>
                <slot :name="`step-${step.id}`" :step="step">
                  <div v-if="step.ctaLabel" class="cta-row">
                    <button class="btn" type="button" @click="step.ctaAction?.()">
                      {{ step.ctaLabel }}
                    </button>
                  </div>
                </slot>
              </div>
            </article>
          </div>
        </div>

        <footer class="wizard-footer">
          <button class="btn ghost" type="button" @click="prev" :disabled="activeIndex === 0">
            {{ t('common.back') }}
          </button>
          <div class="rail">
            <span class="fill" :style="{ width: progressWidth }"></span>
          </div>
          <button class="btn" type="button" @click="next">
            {{ nextLabel }}
          </button>
        </footer>
      </section>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { t } from '@/i18n';

interface WizardStep {
  id: string;
  title: string;
  description: string;
  detail?: string;
  ctaLabel?: string;
  ctaAction?: () => void;
  tone?: 'info' | 'warning' | 'critical' | 'success';
}

const props = withDefaults(defineProps<{
  steps: WizardStep[];
  title?: string;
  subtitle?: string;
  eyebrow?: string;
  open?: boolean;
}>(), {
  steps: () => [],
  title: 'Trading Launch Checklist',
  subtitle: 'Complete the missing signals to unlock ghost + live trading.',
  eyebrow: 'Startup Wizard',
  open: true,
});

const emit = defineEmits<{
  (e: 'update:open', value: boolean): void;
}>();

const activeIndex = ref(0);

const isOpen = computed(() => props.open !== false);
const nextLabel = computed(() =>
  activeIndex.value >= props.steps.length - 1 ? t('common.done') : t('common.next')
);
const progressWidth = computed(() => {
  if (!props.steps.length) return '0%';
  return `${((activeIndex.value + 1) / props.steps.length) * 100}%`;
});

watch(
  () => props.steps,
  (steps) => {
    if (!steps.length) {
      activeIndex.value = 0;
      return;
    }
    if (activeIndex.value > steps.length - 1) {
      activeIndex.value = steps.length - 1;
    }
  },
  { deep: true }
);

function close() {
  emit('update:open', false);
}

function goTo(index: number) {
  activeIndex.value = Math.max(0, Math.min(index, props.steps.length - 1));
}

function prev() {
  goTo(activeIndex.value - 1);
}

function next() {
  if (activeIndex.value >= props.steps.length - 1) {
    close();
    return;
  }
  goTo(activeIndex.value + 1);
}
</script>

<style scoped>
.startup-wizard {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}

.wizard-backdrop {
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 20% 20%, rgba(60, 152, 255, 0.24), transparent 55%),
    radial-gradient(circle at 80% 10%, rgba(16, 210, 190, 0.25), transparent 48%),
    rgba(2, 6, 12, 0.75);
  backdrop-filter: blur(6px);
  animation: fade-in 0.35s ease-out;
}

.wizard-shell {
  position: relative;
  width: min(720px, 92vw);
  background: linear-gradient(135deg, rgba(10, 20, 36, 0.96), rgba(6, 10, 18, 0.98));
  border: 1px solid rgba(127, 176, 255, 0.28);
  border-radius: 22px;
  padding: 1.4rem 1.6rem 1.2rem;
  box-shadow: 0 40px 80px rgba(0, 0, 0, 0.55);
  animation: rise-in 0.4s ease-out;
  display: grid;
  gap: 1rem;
}

.wizard-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.wizard-header h2 {
  margin: 0.2rem 0 0.3rem;
  font-size: 1.3rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: #9bd2ff;
}

.wizard-header p {
  margin: 0;
  color: rgba(255, 255, 255, 0.7);
}

.eyebrow {
  display: inline-flex;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  border: 1px solid rgba(127, 176, 255, 0.35);
  background: rgba(44, 112, 204, 0.2);
  font-size: 0.7rem;
  letter-spacing: 0.2rem;
  text-transform: uppercase;
  color: rgba(233, 246, 255, 0.8);
}

.close-btn {
  border: 1px solid rgba(127, 176, 255, 0.3);
  background: rgba(5, 10, 18, 0.4);
  color: rgba(255, 255, 255, 0.7);
  width: 36px;
  height: 36px;
  border-radius: 12px;
  cursor: pointer;
}

.wizard-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.status-pill {
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: rgba(19, 35, 60, 0.7);
  border: 1px solid rgba(127, 176, 255, 0.2);
  font-size: 0.75rem;
  letter-spacing: 0.12rem;
  text-transform: uppercase;
  color: rgba(233, 246, 255, 0.75);
}

.status-dots {
  display: flex;
  gap: 0.4rem;
}

.status-dots .dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 1px solid rgba(127, 176, 255, 0.5);
  background: transparent;
  cursor: pointer;
}

.status-dots .dot.active {
  background: #36d1dc;
  border-color: #36d1dc;
  box-shadow: 0 0 8px rgba(54, 209, 220, 0.6);
}

.wizard-slider {
  overflow: hidden;
}

.wizard-track {
  display: flex;
  transition: transform 0.4s ease;
}

.wizard-slide {
  min-width: 100%;
  padding: 0.2rem;
}

.slide-card {
  background: rgba(7, 14, 25, 0.9);
  border: 1px solid rgba(127, 176, 255, 0.18);
  border-radius: 18px;
  padding: 1.2rem 1.3rem;
  display: grid;
  gap: 0.6rem;
}

.slide-card h3 {
  margin: 0;
  font-size: 1.05rem;
  color: #f7fbff;
}

.slide-card p {
  margin: 0;
  color: rgba(255, 255, 255, 0.75);
}

.slide-card small {
  color: rgba(255, 255, 255, 0.55);
}

.slide-card.tone-warning {
  border-color: rgba(246, 177, 67, 0.45);
}

.slide-card.tone-info {
  border-color: rgba(127, 176, 255, 0.45);
}

.slide-card.tone-critical {
  border-color: rgba(255, 90, 95, 0.55);
}

.slide-card.tone-success {
  border-color: rgba(52, 211, 153, 0.45);
}

.cta-row {
  margin-top: 0.4rem;
  display: flex;
  justify-content: flex-start;
}

.wizard-footer {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 0.8rem;
}

.rail {
  height: 6px;
  border-radius: 999px;
  background: rgba(127, 176, 255, 0.15);
  overflow: hidden;
}

.rail .fill {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #36d1dc, #5aa8ff);
  transition: width 0.4s ease;
}

@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes rise-in {
  from { opacity: 0; transform: translateY(12px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

@media (max-width: 720px) {
  .wizard-shell {
    padding: 1.1rem;
  }

  .wizard-header h2 {
    font-size: 1.05rem;
  }

  .wizard-footer {
    grid-template-columns: 1fr;
  }

  .wizard-footer .btn {
    width: 100%;
  }
}
</style>
