import { defineStore } from 'pinia';

const STORAGE_KEY = 'ccu-ui-settings';

type UiSettingsState = {
  autoScrollEnabled: boolean;
};

const DEFAULTS: UiSettingsState = {
  autoScrollEnabled: true,
};

function readStorage(): UiSettingsState {
  if (typeof window === 'undefined') return { ...DEFAULTS };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    return {
      autoScrollEnabled:
        typeof parsed?.autoScrollEnabled === 'boolean' ? parsed.autoScrollEnabled : DEFAULTS.autoScrollEnabled,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

function writeStorage(state: UiSettingsState) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export const useUiSettingsStore = defineStore('uiSettings', {
  state: (): UiSettingsState => ({ ...DEFAULTS }),
  actions: {
    load() {
      Object.assign(this, readStorage());
    },
    save() {
      writeStorage({ autoScrollEnabled: this.autoScrollEnabled });
    },
    setAutoScrollEnabled(enabled: boolean) {
      this.autoScrollEnabled = enabled;
      this.save();
    },
  },
});
