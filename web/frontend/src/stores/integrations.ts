import { defineStore } from 'pinia';
import { fetchIntegrations, updateIntegration, testIntegration, revealIntegrationValue } from '@/api';

interface IntegrationState {
  items: Record<string, any>[];
  loading: boolean;
  error: string | null;
  testing: Record<string, boolean>;
  testResult: Record<string, string>;
  revealState: Record<string, string>;
  revealVisible: Record<string, boolean>;
}

export const useIntegrationsStore = defineStore('integrations', {
  state: (): IntegrationState => ({
    items: [],
    loading: false,
    error: null,
    testing: {},
    testResult: {},
    revealState: {},
    revealVisible: {},
  }),
  actions: {
    async load() {
      this.loading = true;
      try {
        const data = await fetchIntegrations();
        this.items = data?.items || [];
        this.error = null;
      } catch (error: any) {
        this.error = error?.message || 'Unable to load integrations';
      } finally {
        this.loading = false;
      }
    },
    async save(name: string, value: string | null) {
      await updateIntegration(name, value);
      await this.load();
    },
    async test(name: string, value: string) {
      this.testing[name] = true;
      try {
        const result = await testIntegration(name, value);
        this.testResult[name] = result?.detail || 'OK';
      } catch (error: any) {
        this.testResult[name] = error?.response?.data?.detail || error?.message || 'Failed';
        throw error;
      } finally {
        this.testing[name] = false;
      }
    },
    async reveal(name: string) {
      if (this.revealVisible[name]) {
        this.revealVisible[name] = false;
        delete this.revealState[name];
        return;
      }
      const data = await revealIntegrationValue(name);
      if (data?.value) {
        this.revealState[name] = data.value;
        this.revealVisible[name] = true;
      }
    },
  },
});
