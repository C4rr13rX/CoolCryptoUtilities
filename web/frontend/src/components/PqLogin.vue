<script setup lang="ts">
/**
 * Quantum-safe login form.
 *
 * The password is sealed with ML-KEM-768 inside `pqAuth.login()` and the
 * plaintext never leaves this component. Do not bind it to a store, log it,
 * or include it in error reporting.
 */
import { ref } from 'vue';
import { pqAuth, type SessionUser } from '../services/pqAuth';

const emit = defineEmits<{ (e: 'authenticated', user: SessionUser): void }>();

const email = ref('');
const password = ref('');
const busy = ref(false);
const error = ref('');

async function submit() {
  error.value = '';
  if (!email.value || !password.value) {
    error.value = 'Enter your username and password.';
    return;
  }
  busy.value = true;
  try {
    const result = await pqAuth.login(email.value.trim(), password.value);
    // Clear immediately on success so the plaintext is not sitting in a
    // reactive ref for the lifetime of the page.
    password.value = '';
    emit('authenticated', result.user);
  } catch (err: any) {
    error.value = err?.message || 'Sign-in failed.';
    password.value = '';
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <form class="pq-login" @submit.prevent="submit">
    <h1 class="pq-login__title">Sign in</h1>

    <label class="pq-login__field">
      <span>Username</span>
      <input
        v-model="email"
        type="text"
        autocomplete="username"
        :disabled="busy"
        required
      />
    </label>

    <label class="pq-login__field">
      <span>Password</span>
      <input
        v-model="password"
        type="password"
        autocomplete="current-password"
        :disabled="busy"
        required
      />
    </label>

    <p v-if="error" class="pq-login__error" role="alert">{{ error }}</p>

    <button class="pq-login__submit" type="submit" :disabled="busy">
      {{ busy ? 'Signing in…' : 'Sign in' }}
    </button>

    <p class="pq-login__note">
      Protected with ML-KEM-768 post-quantum encryption.
    </p>
  </form>
</template>

<style scoped>
.pq-login {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-width: 22rem;
  margin: 4rem auto;
  padding: 2rem;
  border: 1px solid rgba(128, 128, 128, 0.3);
  border-radius: 0.75rem;
}
.pq-login__title {
  margin: 0;
  font-size: 1.25rem;
}
.pq-login__field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.875rem;
}
.pq-login__field input {
  padding: 0.6rem 0.7rem;
  border: 1px solid rgba(128, 128, 128, 0.4);
  border-radius: 0.4rem;
  background: transparent;
  color: inherit;
  font: inherit;
}
.pq-login__submit {
  padding: 0.65rem 1rem;
  border: 0;
  border-radius: 0.4rem;
  background: #2f6feb;
  color: #fff;
  font: inherit;
  cursor: pointer;
}
.pq-login__submit:disabled {
  opacity: 0.6;
  cursor: default;
}
.pq-login__error {
  margin: 0;
  color: #d33;
  font-size: 0.85rem;
}
.pq-login__note {
  margin: 0;
  opacity: 0.65;
  font-size: 0.75rem;
  text-align: center;
}
</style>
