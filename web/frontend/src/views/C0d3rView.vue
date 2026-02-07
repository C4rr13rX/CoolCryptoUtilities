<template>
  <div class="c0d3r-view">
    <section class="panel">
      <header>
        <div>
          <h1>c0d3r Control</h1>
          <p>Send prompts to the Bedrock-backed c0d3r session and capture responses.</p>
        </div>
        <div class="header-actions">
          <button type="button" class="btn ghost" @click="resetSession" :disabled="sending">
            Reset Session
          </button>
        </div>
      </header>

      <form class="prompt-form" @submit.prevent="submit">
        <label>
          <span>Prompt</span>
          <textarea v-model="prompt" rows="5" placeholder="Ask c0d3r anything…" />
        </label>
        <div class="actions">
          <button type="submit" class="btn" :disabled="sending || !prompt.trim()">
            {{ sending ? 'Running…' : 'Send' }}
          </button>
          <label class="switch-row">
            <input type="checkbox" v-model="research" />
            <span>Research mode</span>
          </label>
          <span v-if="modelLabel" class="pill">Model: {{ modelLabel }}</span>
        </div>
      </form>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="panel">
      <header>
        <h2>Conversation</h2>
        <span class="caption">{{ messages.length }} messages</span>
      </header>
      <div class="conversation">
        <div v-for="item in messages" :key="item.id" :class="['message', item.role]">
          <div class="meta">{{ item.role === 'user' ? 'You' : 'c0d3r' }} · {{ item.time }}</div>
          <pre>{{ item.text }}</pre>
        </div>
        <div v-if="!messages.length" class="empty">No prompts yet.</div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { runC0d3rPrompt } from '@/api';

type MessageRole = 'user' | 'c0d3r';

interface Message {
  id: string;
  role: MessageRole;
  text: string;
  time: string;
}

const prompt = ref('');
const research = ref(false);
const sending = ref(false);
const error = ref('');
const modelLabel = ref('');
const messages = ref<Message[]>([]);

const nowStamp = () => new Date().toLocaleTimeString();

const submit = async () => {
  const text = prompt.value.trim();
  if (!text) return;
  error.value = '';
  messages.value.unshift({ id: `${Date.now()}-u`, role: 'user', text, time: nowStamp() });
  prompt.value = '';
  sending.value = true;
  try {
    const result = await runC0d3rPrompt({ prompt: text, research: research.value });
    if (result.model) modelLabel.value = result.model;
    messages.value.unshift({
      id: `${Date.now()}-a`,
      role: 'c0d3r',
      text: result.output || '(no response)',
      time: nowStamp(),
    });
  } catch (err: any) {
    error.value = err?.message || 'Unable to reach c0d3r.';
  } finally {
    sending.value = false;
  }
};

const resetSession = async () => {
  error.value = '';
  sending.value = true;
  try {
    await runC0d3rPrompt({ prompt: '', reset: true });
    messages.value = [];
  } catch (err: any) {
    error.value = err?.message || 'Unable to reset session.';
  } finally {
    sending.value = false;
  }
};
</script>

<style scoped>
.c0d3r-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
}

.prompt-form {
  display: grid;
  gap: 0.9rem;
  margin-top: 1rem;
}

.prompt-form label {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.prompt-form textarea {
  padding: 0.6rem 0.75rem;
  background: rgba(4, 10, 20, 0.8);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
  min-height: 140px;
}

.actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
}

.switch-row {
  display: flex;
  gap: 0.4rem;
  align-items: center;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.7);
}

.pill {
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  background: rgba(45, 117, 196, 0.2);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
}

.conversation {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.message {
  padding: 0.9rem 1rem;
  border-radius: 12px;
  background: rgba(10, 20, 34, 0.75);
  border: 1px solid rgba(111, 167, 255, 0.2);
}

.message.user {
  border-color: rgba(111, 167, 255, 0.4);
}

.message.c0d3r {
  border-color: rgba(34, 197, 94, 0.35);
}

.message pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'Fira Code', 'Source Code Pro', monospace;
  font-size: 0.9rem;
}

.meta {
  font-size: 0.7rem;
  letter-spacing: 0.12rem;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.55);
  margin-bottom: 0.35rem;
}

.caption {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.55);
}

.error {
  color: #ff6b6b;
}

.empty {
  text-align: center;
  color: rgba(255, 255, 255, 0.6);
  padding: 1.25rem;
}
</style>
