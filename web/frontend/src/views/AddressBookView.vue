<template>
  <div class="addressbook-view">
    <section class="panel header-panel">
      <header>
        <div>
          <h1>Address Book</h1>
          <p>Save wallets, notes, and contact images for fast routing.</p>
        </div>
        <div class="header-actions">
          <button type="button" class="btn ghost" @click="refresh" :disabled="loading">
            {{ loading ? 'Refreshing…' : 'Refresh' }}
          </button>
          <button type="button" class="btn" @click="openCreate">New Entry</button>
        </div>
      </header>
    </section>

    <section class="panel">
      <header class="toolbar">
        <div class="search-row">
          <input v-model="query" type="text" placeholder="Search by name…" @keyup.enter="search" />
          <button type="button" class="btn ghost" @click="search" :disabled="loading || !query.trim()">
            Search
          </button>
          <button type="button" class="btn ghost" @click="clearSearch" :disabled="loading">
            Clear
          </button>
        </div>
        <div class="caption">{{ entries.length }} entries</div>
      </header>

      <p v-if="error" class="error">{{ error }}</p>
      <p v-else-if="loading" class="muted">Loading address book…</p>

      <div v-else class="entries-grid">
        <article v-for="entry in entries" :key="entry.id" class="entry-card">
          <div class="avatar">
            <img v-if="entry.image_url" :src="entry.image_url" :alt="entry.name" />
            <span v-else>{{ entry.name?.slice(0, 1) || '?' }}</span>
          </div>
          <div class="entry-body">
            <div class="entry-head">
              <div>
                <strong>{{ entry.name }}</strong>
                <small class="chain">{{ entry.chain || 'Unspecified chain' }}</small>
              </div>
              <div class="entry-actions">
                <button type="button" class="link" @click="openEdit(entry)">Edit</button>
                <button type="button" class="link danger" @click="remove(entry)">Delete</button>
              </div>
            </div>
            <div class="address">{{ entry.address }}</div>
            <p v-if="entry.notes" class="notes">{{ entry.notes }}</p>
          </div>
        </article>
        <div v-if="!entries.length" class="empty">No address book entries yet.</div>
      </div>
    </section>

    <div v-if="showForm" class="modal-backdrop" @click.self="closeForm">
      <div class="modal-card">
        <header>
          <div>
            <h2>{{ editing ? 'Edit Entry' : 'New Entry' }}</h2>
            <p class="muted">Store wallet addresses with optional images.</p>
          </div>
        </header>
        <form class="form-grid" @submit.prevent="save">
          <label>
            <span>Name</span>
            <input v-model="form.name" type="text" required />
          </label>
          <label>
            <span>Wallet Address</span>
            <input v-model="form.address" type="text" required />
          </label>
          <label>
            <span>Chain</span>
            <input v-model="form.chain" type="text" placeholder="Ethereum, Solana…" />
          </label>
          <label class="full">
            <span>Notes</span>
            <textarea v-model="form.notes" rows="3" placeholder="How you know them, risk notes…" />
          </label>
          <label class="full">
            <span>Contact Image</span>
            <input type="file" accept="image/*" @change="handleFile" />
            <div v-if="previewUrl" class="preview">
              <img :src="previewUrl" alt="Preview" />
            </div>
          </label>
          <div class="actions">
            <button type="submit" class="btn" :disabled="saving">
              {{ saving ? 'Saving…' : 'Save' }}
            </button>
            <button type="button" class="btn ghost" @click="closeForm" :disabled="saving">
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import {
  AddressBookEntry,
  createAddressBookEntry,
  deleteAddressBookEntry,
  fetchAddressBookEntries,
  lookupAddressBookEntries,
  updateAddressBookEntry,
} from '@/api';

const entries = ref<AddressBookEntry[]>([]);
const loading = ref(false);
const saving = ref(false);
const error = ref('');
const query = ref('');

const showForm = ref(false);
const editing = ref<AddressBookEntry | null>(null);
const previewUrl = ref<string | null>(null);

const form = ref({
  name: '',
  address: '',
  chain: '',
  notes: '',
  image: null as File | null,
});

const resetForm = () => {
  form.value = { name: '', address: '', chain: '', notes: '', image: null };
  previewUrl.value = null;
};

const refresh = async () => {
  loading.value = true;
  error.value = '';
  try {
    entries.value = await fetchAddressBookEntries();
  } catch (err: any) {
    error.value = err?.message || 'Unable to load address book entries.';
  } finally {
    loading.value = false;
  }
};

const search = async () => {
  const term = query.value.trim();
  if (!term) {
    await refresh();
    return;
  }
  loading.value = true;
  error.value = '';
  try {
    const results = await lookupAddressBookEntries(term, false, 50);
    entries.value = results.results || [];
  } catch (err: any) {
    error.value = err?.message || 'Search failed.';
  } finally {
    loading.value = false;
  }
};

const clearSearch = async () => {
  query.value = '';
  await refresh();
};

const openCreate = () => {
  editing.value = null;
  resetForm();
  showForm.value = true;
};

const openEdit = (entry: AddressBookEntry) => {
  editing.value = entry;
  form.value = {
    name: entry.name || '',
    address: entry.address || '',
    chain: entry.chain || '',
    notes: entry.notes || '',
    image: null,
  };
  previewUrl.value = entry.image_url || null;
  showForm.value = true;
};

const closeForm = () => {
  showForm.value = false;
  resetForm();
};

const handleFile = (event: Event) => {
  const input = event.target as HTMLInputElement | null;
  const file = input?.files?.[0] || null;
  form.value.image = file;
  previewUrl.value = file ? URL.createObjectURL(file) : editing.value?.image_url || null;
};

const save = async () => {
  if (!form.value.name.trim() || !form.value.address.trim()) {
    return;
  }
  saving.value = true;
  error.value = '';
  try {
    const payload = {
      name: form.value.name.trim(),
      address: form.value.address.trim(),
      chain: form.value.chain.trim(),
      notes: form.value.notes.trim(),
      image: form.value.image,
    };
    if (editing.value) {
      const updated = await updateAddressBookEntry(editing.value.id, payload);
      entries.value = entries.value.map((entry) => (entry.id === updated.id ? updated : entry));
    } else {
      const created = await createAddressBookEntry(payload);
      entries.value = [created, ...entries.value];
    }
    closeForm();
  } catch (err: any) {
    error.value = err?.message || 'Unable to save entry.';
  } finally {
    saving.value = false;
  }
};

const remove = async (entry: AddressBookEntry) => {
  if (!window.confirm(`Delete ${entry.name}?`)) return;
  error.value = '';
  try {
    await deleteAddressBookEntry(entry.id);
    entries.value = entries.value.filter((item) => item.id !== entry.id);
  } catch (err: any) {
    error.value = err?.message || 'Unable to delete entry.';
  }
};

onMounted(() => {
  refresh();
});
</script>

<style scoped>
.addressbook-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.header-panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}

.search-row {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.search-row input {
  min-width: 220px;
  padding: 0.5rem 0.75rem;
  background: rgba(4, 10, 20, 0.8);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
}

.entries-grid {
  display: grid;
  gap: 1rem;
}

.entry-card {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background: rgba(10, 20, 34, 0.75);
  border: 1px solid rgba(111, 167, 255, 0.2);
  border-radius: 16px;
}

.avatar {
  width: 64px;
  height: 64px;
  border-radius: 12px;
  background: rgba(45, 117, 196, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  font-weight: 700;
}

.avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.entry-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.entry-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
}

.entry-actions {
  display: flex;
  gap: 0.6rem;
}

.chain {
  display: block;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
  text-transform: uppercase;
  letter-spacing: 0.12rem;
}

.address {
  font-family: 'Fira Code', 'Source Code Pro', monospace;
  font-size: 0.9rem;
  word-break: break-all;
}

.notes {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.7);
}

.link {
  background: none;
  border: none;
  color: #7fb0ff;
  cursor: pointer;
  padding: 0;
  font: inherit;
}

.link.danger {
  color: #ff6b6b;
}

.error {
  color: #ff6b6b;
}

.empty {
  text-align: center;
  color: rgba(255, 255, 255, 0.55);
  padding: 1.5rem;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(5, 9, 15, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 1.5rem;
}

.modal-card {
  background: rgba(7, 14, 24, 0.95);
  border: 1px solid rgba(127, 176, 255, 0.2);
  border-radius: 18px;
  padding: 1.5rem;
  width: min(720px, 100%);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.form-grid {
  display: grid;
  gap: 0.9rem;
}

.form-grid label {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.form-grid input,
.form-grid textarea {
  padding: 0.55rem 0.75rem;
  background: rgba(4, 10, 20, 0.8);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
}

.form-grid .full {
  grid-column: 1 / -1;
}

.actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.preview {
  margin-top: 0.5rem;
  max-width: 180px;
  border: 1px solid rgba(127, 176, 255, 0.25);
  padding: 0.25rem;
}

.preview img {
  width: 100%;
  display: block;
}

.muted {
  color: rgba(255, 255, 255, 0.6);
}
</style>
