import { defineStore } from 'pinia';
import {
  createSecureSetting,
  deleteSecureSetting,
  fetchSecureSettings,
  updateSecureSetting,
  importSecureSettings,
  clearSecureSettings,
  revealSecureSetting,
  SecureSettingPayload,
  SecureSettingImportPayload,
} from '@/api';

interface SecureSettingState {
  items: Record<string, any>[];
  loading: boolean;
  error: string | null;
}

export const useSecureSettingsStore = defineStore('secureSettings', {
  state: (): SecureSettingState => ({
    items: [],
    loading: false,
    error: null,
  }),
  actions: {
    async load() {
      this.loading = true;
      try {
        const data = await fetchSecureSettings();
        this.items = Array.isArray(data) ? data : [];
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to load settings';
      } finally {
        this.loading = false;
      }
    },
    async save(payload: SecureSettingPayload) {
      if (payload.id) {
        await updateSecureSetting(payload.id, payload);
      } else {
        await createSecureSetting(payload);
      }
      await this.load();
    },
    async remove(id: number) {
      await deleteSecureSetting(id);
      await this.load();
    },
    async clearAll(category?: string) {
      await clearSecureSettings(category);
      await this.load();
    },
    async importFromEnv(payload: SecureSettingImportPayload) {
      await importSecureSettings(payload);
      await this.load();
    },
    async reveal(id: number) {
      const data = await revealSecureSetting(id);
      return data;
    },
  },
});
