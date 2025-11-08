import { defineStore } from 'pinia';
import { fetchGuardianSettings, updateGuardianSettings, runGuardianJob, GuardianSettingsPayload } from '@/api';

interface GuardianState {
  settings: Record<string, any> | null;
  status: Record<string, any> | null;
  consoleStatus: Record<string, any> | null;
  loading: boolean;
  error: string | null;
  saving: boolean;
  running: boolean;
}

export const useGuardianStore = defineStore('guardian', {
  state: (): GuardianState => ({
    settings: null,
    status: null,
    consoleStatus: null,
    loading: false,
    error: null,
    saving: false,
    running: false,
  }),
  actions: {
    async load() {
      this.loading = true;
      try {
        const data = await fetchGuardianSettings();
        this.settings = data?.settings || null;
        this.status = data?.status || null;
        this.consoleStatus = data?.console || null;
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to load guardian settings';
      } finally {
        this.loading = false;
      }
    },
    async saveSettings(payload: GuardianSettingsPayload) {
      this.saving = true;
      try {
        const data = await updateGuardianSettings(payload);
        this.settings = data?.settings || this.settings;
        await this.load();
      } finally {
        this.saving = false;
      }
    },
    async runPrompt(prompt: string, saveDefault = false) {
      this.running = true;
      try {
        await runGuardianJob({ prompt, save_default: saveDefault });
      } finally {
        this.running = false;
      }
    },
  },
});
