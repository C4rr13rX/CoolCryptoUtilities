import { defineStore } from 'pinia';
import {
  fetchCronStatus,
  updateCronProfile,
  setCronEnabled,
  runCronTask,
} from '@/api';

interface CronState {
  profile: Record<string, any> | null;
  status: Record<string, any> | null;
  tasks: Record<string, any>;
  loading: boolean;
  saving: boolean;
  running: boolean;
  error: string | null;
}

export const useCronStore = defineStore('cron', {
  state: (): CronState => ({
    profile: null,
    status: null,
    tasks: {},
    loading: false,
    saving: false,
    running: false,
    error: null,
  }),
  actions: {
    async load() {
      this.loading = true;
      try {
        const data = await fetchCronStatus();
        this.profile = data?.profile || null;
        this.status = data?.status || null;
        this.tasks = data?.tasks || {};
        this.error = null;
      } catch (err: any) {
        this.error = err?.message || 'Failed to load cron status';
      } finally {
        this.loading = false;
      }
    },
    async saveProfile(profile: Record<string, any>) {
      this.saving = true;
      try {
        const data = await updateCronProfile(profile);
        this.profile = data?.profile || profile;
        await this.load();
      } finally {
        this.saving = false;
      }
    },
    async toggle(enabled: boolean) {
      this.saving = true;
      try {
        await setCronEnabled(enabled);
        await this.load();
      } finally {
        this.saving = false;
      }
    },
    async runNow(taskId?: string) {
      this.running = true;
      try {
        await runCronTask(taskId);
      } finally {
        this.running = false;
      }
    },
  },
});
