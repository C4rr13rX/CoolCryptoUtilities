<template>
  <div class="branddozer">
    <section class="panel project-list-panel">
      <header>
        <div>
          <h2>{{ t('branddozer.current_projects') }}</h2>
          <p class="caption">{{ t('branddozer.current_projects_caption') }}</p>
        </div>
      </header>
      <div class="project-list">
        <article
          v-for="project in store.projects"
          :key="project.id"
          class="project-card"
          :class="{ selected: project.id === selectedId }"
          @click="selectProject(project.id)"
        >
          <header>
            <div>
              <strong>{{ project.name }}</strong>
              <small class="path">{{ project.root_path }}</small>
            </div>
            <span class="status-pill" :class="project.running ? 'ok' : 'warn'">
              {{ project.running ? t('branddozer.running') : t('branddozer.idle') }}
            </span>
          </header>
          <p class="meta">
            {{ t('branddozer.interval') }}: {{ project.interval_minutes }}m · {{ t('branddozer.interjections') }}:
            {{ project.interjections?.length || 0 }}
          </p>
          <p v-if="project.repo_url" class="meta">{{ t('branddozer.repo') }}: {{ project.repo_url }}</p>
          <p class="meta">{{ t('branddozer.last') }}: {{ formatTime(project.last_run) }} · {{ project.last_message || t('common.na') }}</p>
          <div class="card-actions">
            <button
              type="button"
              class="btn ghost"
              @click.stop="start(project.id)"
              :disabled="project.running || store.saving"
            >
              {{ t('common.start') }}
            </button>
            <button type="button" class="btn ghost" @click.stop="openPublish(project)">{{ t('branddozer.push') }}</button>
            <button
              type="button"
              class="btn ghost danger"
              @click.stop="stop(project.id)"
              :disabled="!project.running || store.saving"
            >
              {{ t('common.stop') }}
            </button>
            <button type="button" class="btn ghost" @click.stop="editProject(project)">{{ t('common.edit') }}</button>
            <button type="button" class="btn ghost danger" @click.stop="remove(project.id)">{{ t('common.delete') }}</button>
          </div>
        </article>
        <div v-if="!store.projects.length" class="empty">{{ t('branddozer.no_projects') }}</div>
      </div>
    </section>

    <section class="panel projects-panel">
      <header>
        <div>
          <h1>{{ t('branddozer.title') }}</h1>
          <p class="caption">{{ t('branddozer.subtitle') }}</p>
        </div>
        <button type="button" class="btn" @click="toggleForm">
          {{ showForm ? t('common.close') : t('branddozer.new_project') }}
        </button>
      </header>

      <div v-if="showForm" class="modal" @click.self="resetForm">
        <div class="modal-card wide">
          <header>
            <div>
              <h2>{{ form.id ? t('branddozer.edit_project') : t('branddozer.new_project') }}</h2>
              <p class="caption">{{ t('branddozer.project_form_caption') }}</p>
            </div>
          </header>
          <form class="project-form" @submit.prevent="saveProject">
            <div class="form-grid">
              <label>
                <span>{{ t('common.name') }}</span>
                <input v-model="form.name" type="text" required />
              </label>
              <label>
                <span>{{ t('branddozer.root_folder') }}</span>
                <div class="path-picker">
                  <input v-model="form.root_path" type="text" :placeholder="folderState.home || '/home'" readonly required />
                  <button type="button" class="btn ghost" @click="openFolderPicker">{{ t('branddozer.browse') }}</button>
                </div>
                <small class="caption">{{ t('branddozer.browse_hint') }}</small>
              </label>
              <label>
                <span>{{ t('branddozer.interval_minutes') }}</span>
                <input v-model.number="form.interval_minutes" type="number" min="5" max="720" />
              </label>
            </div>
            <label>
              <span>{{ t('branddozer.default_prompt') }}</span>
              <textarea v-model="form.default_prompt" rows="4" required />
            </label>
            <div class="interjections">
              <div class="interjections-header">
                <span>{{ t('branddozer.interjections_prompt') }}</span>
                <div class="interjection-actions">
                  <button type="button" class="btn ghost" @click="addInterjection">{{ t('branddozer.add_prompt') }}</button>
                  <button type="button" class="btn ghost" @click="openAiConfirm" :disabled="store.saving || !form.default_prompt.trim()">
                    {{ t('branddozer.ai_expand') }}
                  </button>
                </div>
              </div>
              <div v-if="!form.interjections.length" class="empty">{{ t('branddozer.no_interjections') }}</div>
              <div v-for="(prompt, idx) in form.interjections" :key="idx" class="interjection-row">
                <textarea v-model="form.interjections[idx]" rows="3" />
                <button type="button" class="btn danger ghost" @click="removeInterjection(idx)">{{ t('common.remove') }}</button>
              </div>
              <div v-if="interjectionError" class="error">{{ interjectionError }}</div>
            </div>
            <div class="actions">
              <button type="submit" class="btn" :disabled="store.saving">
                {{ store.saving ? t('common.saving') : form.id ? t('common.update') : t('common.create') }}
              </button>
              <button type="button" class="btn ghost" @click="resetForm">{{ t('common.cancel') }}</button>
            </div>
          </form>
        </div>
      </div>

      <div class="delivery-card">
        <div class="import-head">
          <div>
            <h3>{{ t('branddozer.delivery_system') }}</h3>
            <p class="caption">{{ t('branddozer.delivery_caption') }}</p>
          </div>
        <div class="import-actions">
          <span class="status-chip" :class="activeDelivery?.status === 'running' ? 'ok' : 'warn'">
            {{ activeDelivery?.status || t('branddozer.idle') }}
          </span>
          <button
            type="button"
            class="btn ghost small"
            @click="openDeliveryDesktop"
            :disabled="!activeDelivery?.id"
          >
            {{ t('branddozer.open_desktop') }}
          </button>
          <button type="button" class="btn ghost small" @click="refreshDelivery">{{ t('common.refresh') }}</button>
        </div>
        </div>
        <div class="form-grid compact">
          <label>
            <span>{{ t('branddozer.project') }}</span>
            <select v-model="deliveryForm.project_id">
              <option disabled value="">{{ t('branddozer.select_project') }}</option>
              <option v-for="project in store.projects" :key="project.id" :value="project.id">
                {{ project.name }}
              </option>
            </select>
          </label>
          <label>
            <span>{{ t('branddozer.start_mode') }}</span>
            <select v-model="deliveryForm.mode">
              <option value="auto">{{ t('branddozer.mode_auto') }}</option>
              <option value="new">{{ t('branddozer.mode_new') }}</option>
              <option value="existing">{{ t('branddozer.mode_existing') }}</option>
            </select>
          </label>
          <label>
            <span>{{ t('branddozer.team_mode') }}</span>
            <select v-model="deliveryForm.team_mode">
              <option value="full">{{ t('branddozer.team_full') }}</option>
              <option value="solo">{{ t('branddozer.team_solo') }}</option>
            </select>
          </label>
          <label>
            <span>{{ t('branddozer.session_provider') }}</span>
            <select v-model="deliveryForm.session_provider">
              <option value="codex">{{ t('branddozer.provider_codex') }}</option>
              <option value="c0d3r">{{ t('branddozer.provider_c0d3r') }}</option>
            </select>
          </label>
          <label>
            <span>{{ t('branddozer.codex_model') }}</span>
            <input v-model="deliveryForm.codex_model" placeholder="gpt-5.2-codex" />
          </label>
          <label>
            <span>{{ t('branddozer.codex_reasoning') }}</span>
            <select v-model="deliveryForm.codex_reasoning">
              <option value="medium">{{ t('branddozer.reasoning_medium') }}</option>
              <option value="high">{{ t('branddozer.reasoning_high') }}</option>
              <option value="extra_high">{{ t('branddozer.reasoning_extra_high') }}</option>
              <option value="low">{{ t('branddozer.reasoning_low') }}</option>
            </select>
          </label>
          <label>
            <span>{{ t('branddozer.c0d3r_model') }}</span>
            <input v-model="deliveryForm.c0d3r_model" placeholder="anthropic.claude-3-5-sonnet" />
          </label>
          <label>
            <span>{{ t('branddozer.c0d3r_reasoning') }}</span>
            <select v-model="deliveryForm.c0d3r_reasoning">
              <option value="medium">{{ t('branddozer.reasoning_medium') }}</option>
              <option value="high">{{ t('branddozer.reasoning_high') }}</option>
              <option value="extra_high">{{ t('branddozer.reasoning_extra_high') }}</option>
              <option value="low">{{ t('branddozer.reasoning_low') }}</option>
            </select>
          </label>
          <label class="full">
            <span>{{ t('branddozer.prompt') }}</span>
            <textarea v-model="deliveryForm.prompt" rows="3" :placeholder="t('branddozer.prompt_placeholder')" />
          </label>
          <label class="full">
            <span>{{ t('branddozer.smoke_test') }}</span>
            <input v-model="deliveryForm.smoke_test_cmd" :placeholder="t('branddozer.smoke_test_placeholder')" />
          </label>
        </div>
        <div class="actions">
          <button type="button" class="btn" @click="startDeliveryRun" :disabled="deliveryRunning">
            {{ deliveryRunning ? t('common.running') : t('branddozer.start_delivery') }}
          </button>
          <button
            type="button"
            class="btn danger"
            @click="stopDeliveryRun"
            :disabled="!deliveryRunning"
          >
            {{ t('branddozer.stop_delivery') }}
          </button>
          <button type="button" class="btn ghost" @click="acceptDeliveryRun" :disabled="activeDelivery?.status !== 'awaiting_acceptance'">
            {{ t('branddozer.record_acceptance') }}
          </button>
          <button type="button" class="btn ghost" @click="runUiCapture" :disabled="!activeDelivery?.id || store.uiCaptureLoading">
            {{ store.uiCaptureLoading ? t('branddozer.capturing') : t('branddozer.capture_ui') }}
          </button>
        </div>
        <div v-if="deliveryError" class="error">{{ deliveryError }}</div>
        <div v-if="uiCaptureError" class="error">{{ uiCaptureError }}</div>
        <div v-if="deliveryStatusNote" class="caption">{{ deliveryStatusNote }}</div>
        <div v-if="uiCaptureStatus" class="caption">{{ uiCaptureStatus }}</div>
        <div class="delivery-status">
          <div>
            <strong>{{ t('common.status') }}</strong>
            <p class="caption">{{ activeDelivery?.status || t('common.na') }} · {{ activeDelivery?.phase || t('common.na') }}</p>
          </div>
          <div>
            <strong>{{ t('branddozer.run_id') }}</strong>
            <p class="caption">{{ activeDelivery?.id || t('common.na') }}</p>
          </div>
          <div>
            <strong>{{ t('branddozer.sprints') }}</strong>
            <p class="caption">
              {{ activeDelivery?.sprint_count || 0 }} · {{ t('branddozer.iteration') }} {{ activeDelivery?.iteration || 0 }}
            </p>
          </div>
          <div>
            <strong>{{ t('branddozer.activity') }}</strong>
            <p class="caption">{{ deliveryActivity || t('common.na') }}</p>
            <p v-if="deliveryActivityDetail" class="caption muted">{{ deliveryActivityDetail }}</p>
            <p v-if="deliveryActivityTime" class="caption muted">{{ t('branddozer.last_update') }}: {{ deliveryActivityTime }}</p>
            <p v-if="deliveryEta" class="caption muted">{{ t('branddozer.eta') }}: {{ deliveryEta }}</p>
          </div>
        </div>
        <div class="delivery-grid">
          <div class="delivery-block">
            <h4>{{ t('branddozer.gates') }}</h4>
            <div v-if="!deliveryGateSummary.length" class="empty">{{ t('branddozer.no_gate_runs') }}</div>
            <div v-for="gate in deliveryGateSummary" :key="gate.name" class="delivery-row">
              <span>{{ gate.name }}</span>
              <span class="status-pill" :class="gate.tone">{{ gate.label }}</span>
            </div>
          </div>
          <div class="delivery-block">
            <h4>{{ t('branddozer.backlog') }}</h4>
            <div v-if="!store.deliveryBacklog.length" class="empty">{{ t('branddozer.no_backlog_items') }}</div>
            <div v-for="item in store.deliveryBacklog.slice(0, 6)" :key="item.id" class="delivery-row">
              <span>{{ item.title }}</span>
              <span class="status-pill" :class="item.status === 'done' ? 'ok' : 'warn'">{{ item.status }}</span>
            </div>
          </div>
          <div class="delivery-block">
            <h4>{{ t('branddozer.sessions') }}</h4>
            <div v-if="!store.deliverySessions.length" class="empty">{{ t('branddozer.no_sessions') }}</div>
            <div v-for="session in store.deliverySessions" :key="session.id" class="delivery-row">
              <span>{{ session.role }}</span>
              <span class="status-pill" :class="session.status === 'done' ? 'ok' : 'warn'">{{ session.status }}</span>
            </div>
          </div>
          <div class="delivery-block">
            <h4>{{ t('branddozer.governance') }}</h4>
            <div v-if="!store.deliveryGovernance.length" class="empty">{{ t('branddozer.no_governance') }}</div>
            <div v-for="artifact in store.deliveryGovernance.slice(0, 4)" :key="artifact.id" class="delivery-row">
              <span>{{ artifact.kind }}</span>
              <span class="caption">{{ artifact.summary }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="import-card">
        <div class="import-head">
          <div>
            <h3>{{ t('branddozer.import_github') }}</h3>
          <p class="caption">{{ t('branddozer.import_caption') }}</p>
          </div>
          <div class="import-actions">
            <span class="status-chip" :class="githubConnectionTone">
              {{ githubConnectionLabel }}
            </span>
            <button type="button" class="btn ghost" @click="resetGithubForm">{{ t('common.reset') }}</button>
          </div>
        </div>

        <div class="github-grid">
          <div class="github-block">
            <div class="block-head">
              <div>
                <p class="eyebrow">{{ t('branddozer.step').replace('{count}', '1') }}</p>
                <strong>{{ t('branddozer.accounts') }}</strong>
                <p class="caption">{{ t('branddozer.accounts_caption') }}</p>
              </div>
              <div class="connection-actions">
                <button
                  type="button"
                  class="btn small"
                  @click="connectGithub"
                  :disabled="store.githubAccountLoading || !githubAccountForm.token"
                >
                  {{
                    store.githubAccountLoading
                      ? t('common.saving')
                      : githubAccountSelection === 'new'
                        ? t('branddozer.add_account')
                        : t('branddozer.update_token')
                  }}
                </button>
                <button
                  type="button"
                  class="btn ghost small"
                  @click="refreshGithubRepos"
                  :disabled="store.githubRepoLoading"
                >
                  {{ store.githubRepoLoading ? t('common.refreshing') : t('branddozer.refresh_repos') }}
                </button>
              </div>
            </div>
            <div class="form-grid compact">
              <label>
                <span>{{ t('branddozer.account') }}</span>
                <select v-model="githubAccountSelection">
                  <option value="new">{{ t('branddozer.add_new_account') }}</option>
                  <option v-for="account in store.githubAccounts" :key="account.id" :value="account.id">
                    {{ account.label || account.username || t('branddozer.github_account') }}
                  </option>
                </select>
              </label>
              <label>
                <span>{{ t('branddozer.github_username') }}</span>
                <input v-model="githubAccountForm.username" type="text" :placeholder="store.githubUsername || t('branddozer.github_username_placeholder')" />
              </label>
              <label>
                <span>{{ t('branddozer.pat') }}</span>
                <input v-model="githubAccountForm.token" type="password" :placeholder="t('branddozer.pat_placeholder')" />
              </label>
            </div>
            <p class="caption muted">{{ t('branddozer.pat_hint') }}</p>
            <p v-if="githubTokenLocked" class="caption warn">
              {{ t('branddozer.token_locked') }}
            </p>
          </div>

          <div class="github-block">
            <div class="block-head">
              <div>
                <p class="eyebrow">{{ t('branddozer.step').replace('{count}', '2') }}</p>
                <strong>{{ t('branddozer.select_repo') }}</strong>
                <p class="caption">{{ t('branddozer.select_repo_caption') }}</p>
              </div>
            </div>
            <div class="form-grid compact">
              <label>
                <span>{{ t('branddozer.filter_repos') }}</span>
                <input v-model="githubRepoSearch" type="text" :placeholder="t('branddozer.filter_repos_placeholder')" />
              </label>
              <label>
                <span>{{ t('branddozer.repository') }}</span>
                <select
                  v-model="githubImportForm.repo_full_name"
                  :disabled="store.githubRepoLoading || !store.githubRepos.length"
                >
                  <option disabled value="">
                    {{ store.githubRepoLoading ? t('branddozer.loading_repos') : t('branddozer.select_from_github') }}
                  </option>
                  <option v-for="repo in filteredRepos" :key="repo.full_name" :value="repo.full_name">
                    {{ repo.full_name }} {{ repo.private ? t('branddozer.private_suffix') : '' }}
                  </option>
                </select>
              </label>
              <label>
                <span>{{ t('branddozer.branch') }}</span>
                <select v-model="githubImportForm.branch" :disabled="store.githubBranchLoading || !githubImportForm.repo_full_name">
                  <option value="">{{ t('branddozer.use_default_branch') }}</option>
                  <option v-for="branch in store.githubBranches" :key="branch.name" :value="branch.name">
                    {{ branch.name }}{{ branch.protected ? t('branddozer.protected_suffix') : '' }}
                  </option>
                </select>
              </label>
            </div>
            <div v-if="selectedRepo" class="repo-meta">
              <p class="caption">{{ selectedRepo.description || t('branddozer.no_description') }}</p>
              <p class="caption muted">
                {{ t('branddozer.default_branch') }}: {{ selectedRepo.default_branch || t('common.unknown') }} ·
                {{ selectedRepo.private ? t('branddozer.private') : t('branddozer.public') }}
              </p>
            </div>
            <div v-if="!store.githubRepoLoading && !store.githubRepos.length" class="empty">
              {{ t('branddozer.connect_github_hint') }}
            </div>
          </div>
        </div>

        <div class="github-block wide">
          <div class="block-head">
            <div>
              <p class="eyebrow">{{ t('branddozer.step').replace('{count}', '3') }}</p>
              <strong>{{ t('branddozer.import_settings') }}</strong>
              <p class="caption">{{ t('branddozer.import_settings_caption') }}</p>
            </div>
          </div>
          <div class="form-grid compact">
            <label>
              <span>{{ t('branddozer.destination_folder') }}</span>
              <input
                v-model="githubImportForm.destination"
                type="text"
                :placeholder="`~/BrandDozerProjects/${githubImportForm.repo_full_name.split('/').pop() || ''}`"
              />
              <small class="caption">{{ t('branddozer.destination_hint') }}</small>
            </label>
            <label>
              <span>{{ t('branddozer.project_name') }}</span>
              <input v-model="githubImportForm.name" type="text" :placeholder="t('branddozer.project_name_placeholder')" />
            </label>
            <label class="full">
              <span>{{ t('branddozer.default_prompt_optional') }}</span>
              <textarea
                v-model="githubImportForm.default_prompt"
                rows="2"
                :placeholder="form.default_prompt || t('branddozer.default_prompt_placeholder')"
              />
            </label>
          </div>
        </div>
        <div v-if="githubError" class="error">{{ githubError }}</div>
        <div class="import-status">
          <span class="caption">{{ t('branddozer.import_status') }}</span>
          <strong>{{ githubImportStatus }}</strong>
          <span v-if="githubImportDetail" class="caption">{{ githubImportDetail }}</span>
        </div>
        <div class="actions">
          <button
            type="button"
            class="btn"
            @click="runGithubImport"
            :disabled="store.importing || !!githubImportJobId || !githubImportForm.repo_full_name || store.githubRepoLoading"
          >
            {{ store.importing || githubImportJobId ? t('branddozer.importing') : t('branddozer.import_and_create') }}
          </button>
        </div>
      </div>
    </section>

    <section class="panel logs-panel">
      <header>
        <div>
          <h2>{{ t('branddozer.console_output') }}</h2>
          <p class="caption">{{ t('branddozer.console_caption') }}</p>
        </div>
        <div class="header-actions">
          <select v-model="selectedId">
            <option disabled value="">{{ t('branddozer.select_project') }}</option>
            <option v-for="project in store.projects" :key="project.id" :value="project.id">
              {{ project.name }}
            </option>
          </select>
          <button type="button" class="btn ghost" @click="refreshLogs" :disabled="store.logLoading">{{ t('common.refresh') }}</button>
        </div>
      </header>
      <pre class="console-output" ref="logBox">{{ logText }}</pre>
    </section>

    <q-dialog v-model="deliveryDesktopOpen" :persistent="deliveryRunning" maximized>
      <div class="delivery-desktop" :class="{ 'performance-mode': performanceMode }">
        <div class="desktop-topbar">
          <div class="topbar-left">
            <div class="desktop-brand">
              <h2>{{ t('branddozer.delivery_command_center') }}</h2>
              <span class="caption">
                {{ t('branddozer.project') }} {{ deliveryProjectName || activeDelivery?.project_id || deliveryForm.project_id || t('common.na') }}
              </span>
            </div>
            <div class="topbar-meta">
              <span class="status-chip" :class="deliveryRunning ? 'ok' : 'warn'">
                {{ activeDelivery?.status || t('branddozer.idle') }}
              </span>
              <span v-if="deliveryActivity" class="caption">{{ deliveryActivity }}</span>
              <span v-if="deliveryActivityDetail" class="caption muted">{{ deliveryActivityDetail }}</span>
            </div>
          </div>
          <div class="topbar-actions">
            <div class="layout-group">
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'free' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('free')"
              >
                {{ t('branddozer.layout_free') }}
              </q-btn>
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'grid' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('grid')"
              >
                {{ t('branddozer.layout_grid') }}
              </q-btn>
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'masonry' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('masonry')"
              >
                {{ t('branddozer.layout_masonry') }}
              </q-btn>
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'cascade' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('cascade')"
              >
                {{ t('branddozer.layout_cascade') }}
              </q-btn>
            </div>
            <q-btn size="sm" flat color="primary" @click="scatterWindows">{{ t('branddozer.scatter') }}</q-btn>
            <q-btn size="sm" flat color="secondary" @click="toggleDesktopLogs">
              {{ desktopLiveLogs ? t('branddozer.pause_logs') : t('branddozer.resume_logs') }}
            </q-btn>
            <q-toggle v-model="performanceMode" dense color="amber" :label="t('branddozer.performance')" />
            <q-btn size="sm" color="negative" outline @click="stopDeliveryRun" :disable="!deliveryRunning">
              {{ t('common.stop') }}
            </q-btn>
            <q-btn size="sm" color="primary" outline @click="minimizeDeliveryDesktop">{{ t('branddozer.minimize') }}</q-btn>
            <q-btn v-if="!deliveryRunning" size="sm" color="secondary" outline @click="closeDeliveryDesktop">
              {{ t('common.hide') }}
            </q-btn>
            <q-btn v-if="deliveryComplete" size="sm" color="positive" outline @click="closeDeliveryDesktop">
              {{ t('common.close') }}
            </q-btn>
          </div>
        </div>

        <div v-if="!desktopReady" class="desktop-loading">
          {{ t('branddozer.preparing_desktop') }}
        </div>
        <div v-else class="desktop-shell">
          <div class="desktop-canvas" ref="desktopRef" :class="`layout-${desktopLayout}`">
            <div
              v-for="session in desktopWindowSessions"
              :key="session.id"
              class="desktop-window"
              :class="[`role-${session.role}`, { active: session.id === focusedSessionId }]"
              :style="getWindowStyle(session.id)"
            >
              <div class="window-titlebar" @pointerdown="startDrag(session.id, $event)" @dblclick="focusSession(session.id)">
                <div class="window-title">
                  <span class="role-dot" />
                  <span class="window-name">{{ session.name || session.role }}</span>
                </div>
                <div class="window-status">
                  <span class="caption">{{ session.role }}</span>
                  <q-badge :color="session.status === 'done' ? 'positive' : 'warning'" outline>
                    {{ session.status }}
                  </q-badge>
                </div>
              </div>
              <div class="window-body" @pointerdown="bringToFront(session.id)">
                <q-virtual-scroll
                  :items="sessionLogLines(session.id)"
                  :virtual-scroll-item-size="18"
                  class="terminal-output"
                  :ref="setLogRef(session.id)"
                >
                  <template v-slot="{ item }">
                    <div class="terminal-line">{{ item }}</div>
                  </template>
                </q-virtual-scroll>
                <div v-if="!sessionLogLines(session.id).length" class="terminal-empty">
                  {{ t('branddozer.no_output') }}
                </div>
                <div v-if="!desktopLiveLogs" class="terminal-paused">
                  {{ t('branddozer.logs_paused') }}
                </div>
              </div>
            </div>
            <div v-if="hiddenDesktopSessionCount" class="desktop-empty">
              {{ t('branddozer.showing_first')
                .replace('{shown}', String(desktopWindowSessions.length))
                .replace('{hidden}', String(hiddenDesktopSessionCount)) }}
            </div>
            <div v-if="!store.deliverySessions.length" class="desktop-empty">
              {{ desktopEmptyMessage }}
            </div>
          </div>

          <aside class="desktop-panels">
            <div class="desktop-module">
              <div class="module-head">
                <h4>{{ t('branddozer.run_control') }}</h4>
                <q-badge :color="deliveryRunning ? 'warning' : 'positive'" outline>
                  {{ activeDelivery?.status || t('common.idle') }}
                </q-badge>
              </div>
              <div class="module-body">
                <div class="module-row">
                  <span class="caption">{{ t('branddozer.activity') }}</span>
                  <span>{{ deliveryActivity || t('common.na') }}</span>
                </div>
                <div v-if="deliveryActivityDetail" class="module-row">
                  <span class="caption">{{ t('branddozer.detail') }}</span>
                  <span class="muted">{{ deliveryActivityDetail }}</span>
                </div>
                <div v-if="deliveryActivityTime" class="module-row">
                  <span class="caption">{{ t('branddozer.last_update') }}</span>
                  <span class="muted">{{ deliveryActivityTime }}</span>
                </div>
                <div v-if="deliveryPromptSnippet" class="module-row stacked">
                  <span class="caption">{{ t('branddozer.prompt') }}</span>
                  <span class="muted">{{ deliveryPromptSnippet }}</span>
                </div>
              </div>
              <div class="module-actions">
                <q-btn size="sm" outline color="primary" @click="refreshDelivery">{{ t('common.refresh') }}</q-btn>
                <q-btn size="sm" outline color="secondary" @click="toggleDesktopLogs">
                  {{ desktopLiveLogs ? t('branddozer.pause_logs') : t('branddozer.resume_logs') }}
                </q-btn>
                <q-btn size="sm" outline color="primary" @click="runUiCapture" :disable="!activeDelivery?.id">
                  {{ t('branddozer.capture_ui') }}
                </q-btn>
                <q-btn size="sm" outline color="negative" @click="stopDeliveryRun" :disable="!deliveryRunning">
                  {{ t('branddozer.stop_run') }}
                </q-btn>
              </div>
            </div>

            <div class="desktop-module">
              <h4>{{ t('branddozer.gate_radar') }}</h4>
              <div v-if="!deliveryGateSummary.length" class="caption">{{ t('branddozer.no_gate_runs') }}</div>
              <div v-for="gate in deliveryGateSummary" :key="gate.name" class="module-row">
                <span>{{ gate.name }}</span>
                <span class="status-pill" :class="gate.tone">{{ gate.label }}</span>
              </div>
            </div>

            <div class="desktop-module">
              <h4>{{ t('branddozer.project_checklist') }}</h4>
              <q-scroll-area class="checklist-scroll">
                <div v-for="item in store.deliveryBacklog" :key="item.id" class="checklist-row">
                  <q-checkbox :model-value="item.status === 'done'" dense />
                  <div class="checklist-text">
                    <span>{{ item.title }}</span>
                    <small class="caption">{{ item.status }}</small>
                  </div>
                </div>
                <div v-if="!store.deliveryBacklog.length" class="caption">{{ t('branddozer.no_checklist_items') }}</div>
              </q-scroll-area>
            </div>

            <div class="desktop-module">
              <div class="module-head">
                <h4>{{ t('branddozer.ui_evidence') }}</h4>
                <q-btn
                  size="sm"
                  flat
                  color="primary"
                  @click="runUiCapture"
                  :loading="store.uiCaptureLoading"
                  :disable="!activeDelivery?.id"
                >
                  {{ t('branddozer.capture') }}
                </q-btn>
              </div>
              <div v-if="!uiSnapshots.length" class="caption">{{ t('branddozer.no_screenshots') }}</div>
              <div v-else class="evidence-grid">
                <button
                  v-for="shot in uiSnapshots.slice(0, 6)"
                  :key="shot.id"
                  type="button"
                  class="evidence-thumb"
                  @click="openScreenshot(shot)"
                >
                  <img :src="artifactUrl(shot)" :alt="shot.title || t('branddozer.ui_screenshot')" loading="lazy" />
                  <span class="caption">
                    {{ shot.title || shot.id.slice(0, 8) }}
                    <template v-if="shot.kind && shot.kind !== 'ui_screenshot'">
                      · {{ shot.kind.replace('ui_screenshot_', '') }}
                    </template>
                  </span>
                </button>
              </div>
            </div>

            <div class="desktop-module">
              <div class="module-head">
                <h4>{{ t('branddozer.ux_audit') }}</h4>
                <q-btn
                  v-if="uxReport"
                  size="sm"
                  flat
                  color="primary"
                  :href="artifactUrl(uxReport)"
                  target="_blank"
                  rel="noopener"
                >
                  {{ t('branddozer.open_report') }}
                </q-btn>
              </div>
              <div v-if="!uxReport && !conversionArtifacts.length && !uxGateStatuses.length" class="caption">
                {{ t('branddozer.no_ux_audit') }}
              </div>
              <div v-else>
                <div v-if="uxReport" class="module-row">
                  <div>
                    <div>{{ uxReport.title || t('branddozer.ux_audit_report') }}</div>
                    <div class="caption">{{ t('branddozer.ux_audit_caption') }}</div>
                  </div>
                  <a :href="artifactUrl(uxReport)" target="_blank" rel="noopener" class="caption">{{ t('common.view') }}</a>
                </div>
                <div v-for="gate in uxGateStatuses" :key="gate.name" class="module-row">
                  <span>{{ gate.name }}</span>
                  <span class="status-pill" :class="gate.tone">{{ gate.label }}</span>
                </div>
                <div v-for="artifact in conversionArtifacts.slice(0, 4)" :key="artifact.id" class="module-row">
                  <div>
                    <div>{{ artifact.title || t('branddozer.conversion_check') }}</div>
                    <div class="caption">
                      {{ artifact.data?.detail || artifact.data?.status || t('branddozer.conversion_recorded') }}
                    </div>
                  </div>
                  <a :href="artifactUrl(artifact)" target="_blank" rel="noopener" class="caption">{{ t('common.open') }}</a>
                </div>
              </div>
            </div>

            <div class="desktop-module">
              <div class="module-head">
                <h4>{{ t('branddozer.worker_intents') }}</h4>
              </div>
              <div v-if="!taskIntents.length" class="caption">{{ t('branddozer.no_worker_intents') }}</div>
              <div v-else>
                <div v-for="intent in taskIntents.slice(0, 6)" :key="intent.id" class="module-row">
                  <div>
                    <div>{{ intent.title || intent.meta?.title || t('branddozer.task_intent') }}</div>
                    <div class="caption">
                      {{ intent.meta?.codex_role || intent.meta?.role || t('branddozer.worker') }} ·
                      {{ intent.meta?.codex_model || intent.meta?.model || t('branddozer.model_unknown') }}
                      <span v-if="intent.meta?.priority !== undefined"> · {{ t('branddozer.priority') }} {{ intent.meta?.priority }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </q-dialog>

    <div v-if="deliveryDesktopMinimized && activeDelivery?.id" class="delivery-minibar" @click="restoreDeliveryDesktop">
      <span>{{ t('branddozer.delivery_desktop') }} · {{ activeDelivery?.status || t('common.idle') }}</span>
      <span class="caption">{{ t('branddozer.click_restore') }}</span>
    </div>

    <div v-if="confirmOpen" class="modal">
      <div class="modal-card">
        <h3>{{ t('branddozer.generate_interjections_title') }}</h3>
        <p>{{ t('branddozer.generate_interjections_desc') }}</p>
        <div class="actions">
          <button type="button" class="btn" @click="generateInterjections" :disabled="store.saving">
            {{ store.saving ? t('branddozer.generating') : t('branddozer.generate_yes') }}
          </button>
          <button type="button" class="btn ghost" @click="confirmOpen = false">{{ t('common.cancel') }}</button>
        </div>
      </div>
    </div>

    <q-dialog v-model="publishOpen">
      <q-card class="publish-card">
        <q-btn class="publish-close" flat dense label="X" @click="closePublishModal" />
        <div class="publish-header">
          <div>
            <h3>{{ t('branddozer.push_github') }}</h3>
            <p class="caption">
              {{ t('branddozer.push_caption') }}
              <span v-if="activeGithubLabel">{{ t('branddozer.active_account') }} {{ activeGithubLabel }}</span>
            </p>
          </div>
          <q-badge v-if="publishTarget?.name" color="primary" outline>{{ publishTarget.name }}</q-badge>
        </div>
        <div class="publish-grid">
          <q-input v-model="publishForm.message" :label="t('branddozer.commit_message')" dense outlined />
          <q-input v-model="publishForm.repo_name" :label="t('branddozer.repo_name_new')" dense outlined />
          <q-toggle v-model="publishForm.private" :label="t('branddozer.private_repo')" />
        </div>
        <div v-if="publishStatus" class="caption">{{ publishStatus }}</div>
        <div v-if="publishError" class="error">{{ publishError }}</div>
        <div v-else-if="githubTokenLocked" class="error">
          {{ t('branddozer.github_token_locked') }}
        </div>
        <div v-else-if="!githubConnected" class="error">
          {{ t('branddozer.github_not_connected') }}
        </div>
        <div class="actions">
          <q-btn
            color="primary"
            @click="runPublish"
            :loading="store.publishing"
            :disable="!publishForm.message || !githubConnected || githubTokenLocked"
          >
            {{ t('branddozer.push_now') }}
          </q-btn>
          <q-btn outline color="secondary" class="publish-cancel" @click="closePublishModal">{{ t('common.cancel') }}</q-btn>
        </div>
      </q-card>
    </q-dialog>

    <q-dialog v-model="screenshotOpen">
      <q-card class="screenshot-modal">
        <div class="screenshot-header">
          <h3>{{ selectedScreenshot?.title || t('branddozer.ui_screenshot') }}</h3>
          <q-btn flat color="secondary" @click="screenshotOpen = false">{{ t('common.close') }}</q-btn>
        </div>
        <div v-if="selectedScreenshot" class="screenshot-body">
          <img :src="artifactUrl(selectedScreenshot)" :alt="selectedScreenshot.title || t('branddozer.ui_screenshot')" />
        </div>
      </q-card>
    </q-dialog>

    <div v-if="folderModalOpen" class="modal">
      <div class="modal-card wide">
        <h3>{{ t('branddozer.select_project_folder') }}</h3>
        <p class="caption">{{ t('branddozer.browsing_under').replace('{path}', folderState.home || t('common.home')) }}</p>
        <div class="folder-controls">
          <button type="button" class="btn ghost" @click="loadFolders(folderState.home)" :disabled="folderLoading">
            {{ t('common.home') }}
          </button>
          <button type="button" class="btn ghost" @click="goToParent" :disabled="!folderState.parent || folderLoading">
            {{ t('common.up') }}
          </button>
          <span class="current-path">{{ folderState.current_path || t('common.na') }}</span>
        </div>
        <div v-if="folderError" class="error">{{ folderError }}</div>
        <div v-if="folderLoading" class="caption">{{ t('branddozer.loading_folders') }}</div>
        <div v-else class="folder-list">
          <button
            v-for="dir in folderState.directories"
            :key="dir.path"
            type="button"
            class="folder-row"
            @click="loadFolders(dir.path)"
          >
            <span>{{ dir.name }}</span>
            <span class="caption">{{ dir.path }}</span>
          </button>
          <div v-if="!folderState.directories.length" class="empty">{{ t('branddozer.no_subfolders') }}</div>
        </div>
        <div class="actions">
          <button type="button" class="btn" :disabled="!folderState.current_path" @click="chooseFolder">
            {{ t('branddozer.use_this_folder') }}
          </button>
          <button type="button" class="btn ghost" @click="folderModalOpen = false">{{ t('common.close') }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useBrandDozerStore } from '@/stores/branddozer';
import { t } from '@/i18n';

const store = useBrandDozerStore();
const selectedId = ref<string>('');
const showForm = ref(false);
const form = ref({
  id: '',
  name: '',
  root_path: '',
  default_prompt: '',
  interjections: [] as string[],
  interval_minutes: 120,
});
const folderModalOpen = ref(false);
const folderLoading = ref(false);
const folderError = ref('');
const folderState = ref<{ current_path: string; parent: string | null; home: string; directories: any[] }>({
  current_path: '',
  parent: null,
  home: '',
  directories: [],
});
const githubAccountForm = ref({
  username: '',
  token: '',
});
const githubAccountSelection = ref('new');
const githubImportForm = ref({
  repo_full_name: '',
  branch: '',
  destination: '',
  name: '',
  default_prompt: '',
});
const githubRepoSearch = ref('');
const lastAutoDestination = ref('');
const githubError = ref('');
const githubImportStatus = ref(t('branddozer.import_ready'));
const githubImportDetail = ref('');
const githubImportJobId = ref('');
const confirmOpen = ref(false);
const interjectionError = ref('');
const logBox = ref<HTMLElement | null>(null);
const deliveryForm = ref({
  project_id: '',
  mode: 'auto',
  team_mode: 'full',
  session_provider: 'codex',
  codex_model: '',
  codex_reasoning: 'medium',
  c0d3r_model: '',
  c0d3r_reasoning: 'high',
  prompt: '',
  smoke_test_cmd: '',
});
const publishOpen = ref(false);
const publishError = ref('');
const publishStatus = ref('');
const publishTarget = ref<any | null>(null);
const publishForm = ref({
  message: t('branddozer.default_commit_message'),
  repo_name: '',
  private: true,
});
const publishJobId = ref('');
let publishTimer: number | null = null;
const deliveryError = ref('');
const deliveryStatusNote = ref('');
const uiCaptureError = ref('');
const uiCaptureStatus = ref('');
let deliveryTimer: number | null = null;
let deliveryPollActive = false;
const deliveryRefreshing = ref(false);
let lastSessionLogFetch = 0;
const deliveryLogWorker = ref<Worker | null>(null);
const workerActive = ref(false);
let deliveryPollTick = 0;
const performanceMode = ref(true);
const pageVisible = ref(true);
const deliveryPollIntervalMs = computed(() => (performanceMode.value ? 12000 : 5000));
const deliveryLogIntervalMs = computed(() => (performanceMode.value ? 15000 : 6000));
const deliveryLogLimit = computed(() => (performanceMode.value ? 80 : 120));
const consoleLogMaxLines = computed(() => (performanceMode.value ? 80 : 120));
const consoleLogMaxChars = computed(() => (performanceMode.value ? 320 : 500));
const logWorkerIntervalMs = computed(() => (performanceMode.value ? 5000 : 2500));
const logWorkerMaxBytes = computed(() => (performanceMode.value ? 160000 : 320000));
const logWorkerConcurrency = computed(() => (performanceMode.value ? 2 : 4));

let logTimer: number | null = null;
let importTimer: number | null = null;
const deliveryDesktopOpen = ref(false);
const deliveryDesktopMinimized = ref(false);
const desktopReady = ref(false);
const desktopLiveLogs = ref(false);
let desktopOpenTimer: number | null = null;
const desktopLayout = ref<'free' | 'grid' | 'masonry' | 'cascade'>('free');
const focusedSessionId = ref('');
const desktopRef = ref<HTMLElement | null>(null);
const windowPositions = ref<Record<string, { x: number; y: number; z: number }>>({});
const dragState = ref<{ id: string; offsetX: number; offsetY: number } | null>(null);
const logRefs = ref<Record<string, any>>({});
const screenshotOpen = ref(false);
const selectedScreenshot = ref<any | null>(null);
let visibilityHandler: (() => void) | null = null;
const desktopSeededRunId = ref('');

onMounted(async () => {
  pageVisible.value = typeof document !== 'undefined' ? !document.hidden : true;
  visibilityHandler = () => {
    pageVisible.value = !document.hidden;
    if (!pageVisible.value) {
      stopLogTimer();
      stopDeliveryPolling();
      desktopLiveLogs.value = false;
      stopLogWorker();
      return;
    }
    startLogTimer();
    if (deliveryRunning.value) {
      startDeliveryPolling();
    }
    if (deliveryDesktopOpen.value && desktopLiveLogs.value) {
      startLogWorker();
    }
  };
  document.addEventListener('visibilitychange', visibilityHandler);
  await store.load();
  await loadFolders();
  if (store.projects.length && !deliveryForm.value.project_id) {
    deliveryForm.value.project_id = store.projects[0].id;
  }
  try {
    await store.loadGithubAccount();
    githubAccountSelection.value = store.githubActiveAccountId || (store.githubAccounts.length ? store.githubAccounts[0].id : 'new');
    githubAccountForm.value.username = activeGithubAccount.value?.username || store.githubUsername || '';
    if (activeGithubAccount.value?.has_token) {
      await refreshGithubRepos();
    }
  } catch (err) {
    // Ignore account load errors in UI init.
  }
  if (store.projects.length) {
    selectedId.value = store.projects[0].id;
    if (!form.value.root_path) {
      form.value.root_path = store.projects[0].root_path;
    }
  } else if (!form.value.root_path && folderState.value.current_path) {
    form.value.root_path = folderState.value.current_path;
  }
  startLogTimer();
  await refreshDelivery();
});

onBeforeUnmount(() => {
  stopLogTimer();
  if (importTimer) window.clearInterval(importTimer);
  if (deliveryTimer) window.clearInterval(deliveryTimer);
  if (publishTimer) window.clearInterval(publishTimer);
  if (desktopOpenTimer) window.clearTimeout(desktopOpenTimer);
  if (visibilityHandler) {
    document.removeEventListener('visibilitychange', visibilityHandler);
  }
  stopDrag();
  stopLogWorker();
  if (deliveryLogWorker.value) {
    deliveryLogWorker.value.terminate();
    deliveryLogWorker.value = null;
  }
});

watch(selectedId, () => {
  refreshLogs();
  startLogTimer();
});

watch(
  () => store.logs,
  () => {
    nextTick(() => {
      if (logBox.value) {
        logBox.value.scrollTop = logBox.value.scrollHeight;
      }
    });
  },
);

watch(
  () => deliveryForm.value.project_id,
  () => {
    refreshDelivery();
  },
);

watch(
  () => store.deliverySessions.map((session: any) => session.id).join('|'),
  () => {
    const sessionIds = store.deliverySessions.map((session: any) => session.id);
    if (focusedSessionId.value && !sessionIds.includes(focusedSessionId.value)) {
      focusedSessionId.value = sessionIds[0] || '';
    } else if (!focusedSessionId.value && sessionIds.length) {
      focusedSessionId.value = sessionIds[0];
    }
    const nextPositions: Record<string, { x: number; y: number; z: number }> = {};
    sessionIds.forEach((id) => {
      if (windowPositions.value[id]) {
        nextPositions[id] = windowPositions.value[id];
      }
    });
    windowPositions.value = nextPositions;
    const nextRefs: Record<string, HTMLElement | null> = {};
    sessionIds.forEach((id) => {
      if (logRefs.value[id]) {
        nextRefs[id] = logRefs.value[id];
      }
    });
    logRefs.value = nextRefs;
    nextTick(() => {
      ensureWindowPositions();
      if (desktopLayout.value === 'cascade') {
        cascadeWindows();
      }
    });
    if (workerActive.value) {
      updateLogWorkerSessions();
    }
  },
);

watch(
  () => store.deliverySessionLogs,
  () => {
    if (!desktopLiveLogs.value || !deliveryDesktopOpen.value) {
      return;
    }
    nextTick(() => {
      Object.entries(logRefs.value).forEach(([sessionId, list]) => {
        const lines = store.deliverySessionLogs[sessionId];
        if (list?.scrollTo && lines?.length) {
          list.scrollTo(lines.length - 1);
        }
      });
    });
  },
  { deep: true },
);
watch(
  () => githubImportForm.value.repo_full_name,
  async (fullName) => {
    if (!fullName) {
      githubImportForm.value.branch = '';
      store.githubBranches = [];
      return;
    }
    await refreshGithubBranches(fullName);
    const repoName = fullName.split('/').pop() || '';
    const selected = store.githubRepos.find((repo: any) => repo.full_name === fullName);
    if (!githubImportForm.value.branch && selected?.default_branch) {
      githubImportForm.value.branch = selected.default_branch;
    }
    setAutoDestination(repoName);
    if (!githubImportForm.value.name) {
      githubImportForm.value.name = repoName || githubImportForm.value.name;
    }
  },
);

watch(
  () => githubAccountSelection.value,
  async () => {
    await switchGithubAccount();
  },
);

watch(
  () => performanceMode.value,
  () => {
    nextTick(() => {
      ensureWindowPositions();
      if (desktopLayout.value === 'cascade') {
        cascadeWindows();
      }
    });
    if (workerActive.value) {
      updateLogWorkerSessions();
    }
  },
);

watch(
  () => desktopRunId.value,
  (nextId, prevId) => {
    if (nextId && nextId !== prevId) {
      store.resetDeliverySessionLogs();
      resetLogWorker(nextId);
      if (desktopLiveLogs.value) {
        startLogWorker();
      }
    }
    if (!nextId) {
      stopLogWorker();
    }
  },
);

const logText = computed(() => {
  if (!store.logs.length) return t('branddozer.no_output');
  const lines = store.logs.slice(-consoleLogMaxLines.value).map((line) => {
    if (line.length > consoleLogMaxChars.value) {
      return `${line.slice(0, consoleLogMaxChars.value)}...`;
    }
    return line;
  });
  return lines.join('\n');
});
const selectedRepo = computed(() =>
  store.githubRepos.find((repo: any) => repo.full_name === githubImportForm.value.repo_full_name),
);
const filteredRepos = computed(() => {
  const term = githubRepoSearch.value.trim().toLowerCase();
  if (!term) return store.githubRepos;
  return store.githubRepos.filter((repo: any) => {
    const haystack = `${repo.full_name || ''} ${repo.description || ''}`.toLowerCase();
    return haystack.includes(term);
  });
});
const activeGithubAccount = computed(() =>
  store.githubAccounts.find((account: any) => account.id === store.githubActiveAccountId) || store.githubActiveAccount,
);
const githubTokenLocked = computed(() => Boolean(activeGithubAccount.value?.token_locked));
const githubConnected = computed(() => Boolean(activeGithubAccount.value?.has_token && !githubTokenLocked.value));
const activeGithubLabel = computed(
  () => activeGithubAccount.value?.label || activeGithubAccount.value?.username || store.githubUsername || '',
);
const githubConnectionLabel = computed(() => {
  const label = activeGithubLabel.value ? ` · ${activeGithubLabel.value}` : '';
  if (githubTokenLocked.value) {
    return t('branddozer.github_token_needs_resave').replace('{label}', label);
  }
  if (githubConnected.value) {
    return t('branddozer.github_connected').replace('{label}', label);
  }
  return t('branddozer.github_not_connected');
});
const githubConnectionTone = computed(() => (githubConnected.value ? 'ok' : 'warn'));
const activeDelivery = computed(() => store.activeDeliveryRun);
const desktopRunId = computed(() => activeDelivery.value?.id || '');
const deliveryRunning = computed(() => {
  const status = activeDelivery.value?.status;
  return status === 'running' || status === 'queued';
});
const deliveryComplete = computed(() => {
  const status = activeDelivery.value?.status;
  return status === 'complete' || activeDelivery.value?.acceptance_recorded;
});
const deliveryActivity = computed(() => {
  const run = activeDelivery.value;
  const context = run?.context || {};
  const note = context.status_note;
  const jobMessage = run?.job?.message;
  return note || jobMessage || '';
});
const deliveryActivityDetail = computed(() => {
  const run = activeDelivery.value;
  const context = run?.context || {};
  return context.status_detail || run?.job?.detail || '';
});
const deliveryEta = computed(() => {
  const eta = activeDelivery.value?.context?.eta;
  if (!eta || typeof eta !== 'object') return '';
  const minutes = eta.minutes;
  if (minutes === undefined || minutes === null) return '';
  const asOf = eta.as_of ? t('branddozer.as_of').replace('{time}', String(eta.as_of)) : t('branddozer.current');
  return t('branddozer.eta_minutes').replace('{minutes}', String(minutes)).replace('{as_of}', asOf);
});
const deliveryActivityTime = computed(() => {
  const ts = activeDelivery.value?.context?.status_ts;
  if (!ts) return '';
  try {
    const parsed = Date.parse(ts);
    if (Number.isNaN(parsed)) return '';
    return new Date(parsed).toLocaleString();
  } catch (err) {
    return '';
  }
});
const deliveryPromptSnippet = computed(() => {
  const prompt = activeDelivery.value?.prompt || '';
  if (!prompt) return '';
  return prompt.length > 160 ? `${prompt.slice(0, 160)}…` : prompt;
});
const deliveryProjectName = computed(() => {
  const projectId = activeDelivery.value?.project_id || deliveryForm.value.project_id;
  return store.projects.find((project) => project.id === projectId)?.name || '';
});
const deliveryGateSummary = computed(() => {
  const seen = new Set<string>();
  const summary: Array<{ name: string; label: string; tone: string }> = [];
  for (const gate of store.deliveryGates) {
    if (!gate?.name || seen.has(gate.name)) continue;
    seen.add(gate.name);
    const status = String(gate.status || '').toLowerCase();
    const isOk = status === 'passed' || status === 'skipped';
    summary.push({
      name: gate.name,
      label: status === 'skipped' ? t('branddozer.not_relevant') : (gate.status || t('common.unknown')),
      tone: isOk ? 'ok' : 'warn',
    });
  }
  return summary;
});
const sortedDeliverySessions = computed(() => {
  const order = ["orchestrator", "pm", "integrator", "qa", "dev"];
  return [...store.deliverySessions].sort((a: any, b: any) => {
    const rankA = order.indexOf(a.role);
    const rankB = order.indexOf(b.role);
    const safeA = rankA === -1 ? order.length : rankA;
    const safeB = rankB === -1 ? order.length : rankB;
    if (safeA !== safeB) return safeA - safeB;
    return (a.created_at || "").localeCompare(b.created_at || "");
  });
});
const desktopWindowSessions = computed(() => sortedDeliverySessions.value);
const hiddenDesktopSessionCount = computed(() => 0);
const desktopEmptyMessage = computed(() => {
  if (!activeDelivery.value?.id) {
    return t('branddozer.no_delivery_selected');
  }
  if (!deliveryRunning.value) {
    return t('branddozer.run_inactive')
      .replace('{status}', String(activeDelivery.value?.status || t('branddozer.inactive_status')));
  }
  return t('branddozer.waiting_sessions');
});
const uiSnapshotKinds = ['ui_screenshot', 'ui_screenshot_mobile', 'ui_screenshot_desktop'];
const uiSnapshots = computed(() =>
  store.deliveryArtifacts.filter((artifact: any) => uiSnapshotKinds.includes(artifact.kind))
);
const uxReport = computed(() => store.deliveryArtifacts.find((artifact: any) => artifact.kind === 'ux_audit_report'));
const taskIntents = computed(() =>
  store.deliveryArtifacts
    .filter((artifact: any) => artifact.kind === 'task_intent')
    .map((artifact: any) => {
      let meta: any = {};
      try {
        meta = artifact.content ? JSON.parse(artifact.content) : artifact.data || {};
      } catch (err) {
        meta = artifact.data || {};
      }
      return { ...artifact, meta };
    })
);
const conversionArtifacts = computed(() => store.deliveryArtifacts.filter((artifact: any) => artifact.kind === 'conversion_check'));
const uxGateStatuses = computed(() => {
  const interesting = ['ui-snapshot', 'ui-review', 'ux-audit', 'conversion', 'conversion-path', 'ux-check'];
  const seen = new Set<string>();
  const rows: Array<{ name: string; label: string; tone: string }> = [];
  for (const gate of store.deliveryGates) {
    const name = gate?.name;
    if (!name || !interesting.includes(name) || seen.has(name)) continue;
    seen.add(name);
    const status = String(gate.status || '').toLowerCase();
    const isOk = status === 'passed' || status === 'skipped';
    rows.push({ name, label: gate.status || 'unknown', tone: isOk ? 'ok' : 'warn' });
  }
  return rows;
});

async function loadFolders(path?: string) {
  folderLoading.value = true;
  folderError.value = '';
  try {
    const data = await store.browseRoots(path);
    folderState.value = {
      current_path: data.current_path,
      parent: data.parent || null,
      home: data.home,
      directories: data.directories || [],
    };
    if (!form.value.root_path && data.current_path) {
      form.value.root_path = data.current_path;
    }
  } catch (err: any) {
    folderError.value = err?.message || t('branddozer.error_load_folders');
  } finally {
    folderLoading.value = false;
  }
}

function openFolderPicker() {
  folderModalOpen.value = true;
  if (!folderState.value.current_path && !folderLoading.value) {
    loadFolders();
  }
}

function goToParent() {
  if (folderState.value.parent) {
    loadFolders(folderState.value.parent);
  }
}

function chooseFolder() {
  if (folderState.value.current_path) {
    form.value.root_path = folderState.value.current_path;
  }
  folderModalOpen.value = false;
}

function setAutoDestination(repoName: string) {
  if (!repoName) return;
  const autoPath = `~/BrandDozerProjects/${repoName}`;
  if (!githubImportForm.value.destination || githubImportForm.value.destination === lastAutoDestination.value) {
    githubImportForm.value.destination = autoPath;
    lastAutoDestination.value = autoPath;
  }
}

async function refreshGithubRepos() {
  githubError.value = '';
  try {
    await store.fetchGithubRepos();
    if (!githubImportForm.value.repo_full_name && store.githubRepos.length) {
      githubImportForm.value.repo_full_name = store.githubRepos[0].full_name;
    }
  } catch (err: any) {
    githubError.value = err?.message || t('branddozer.error_load_repos');
  }
}

async function refreshGithubBranches(fullName: string) {
  githubError.value = '';
  try {
    await store.fetchGithubBranches(fullName);
  } catch (err: any) {
    githubError.value = err?.message || t('branddozer.error_load_branches');
  }
}

async function switchGithubAccount() {
  githubError.value = '';
  const selection = githubAccountSelection.value;
  if (!selection || selection === 'new') {
    githubAccountForm.value.username = '';
    githubAccountForm.value.token = '';
    store.githubRepos = [];
    store.githubBranches = [];
    githubRepoSearch.value = '';
    githubImportForm.value.repo_full_name = '';
    githubImportForm.value.branch = '';
    githubImportForm.value.destination = '';
    githubImportForm.value.name = '';
    setImportStatus(t('branddozer.import_ready'), '');
    return;
  }
  if (selection !== store.githubActiveAccountId) {
    try {
      await store.setGithubActiveAccount(selection);
    } catch (err: any) {
      githubError.value = err?.message || t('branddozer.error_switch_account');
      return;
    }
  }
  const active = store.githubAccounts.find((account: any) => account.id === selection) || store.githubActiveAccount;
  githubAccountForm.value.username = active?.username || '';
  githubAccountForm.value.token = '';
  githubRepoSearch.value = '';
  githubImportForm.value.repo_full_name = '';
  githubImportForm.value.branch = '';
  githubImportForm.value.destination = '';
  githubImportForm.value.name = '';
  await refreshGithubRepos();
}

function stopDeliveryPolling() {
  deliveryPollActive = false;
  if (deliveryTimer) {
    window.clearTimeout(deliveryTimer);
    deliveryTimer = null;
  }
}

async function refreshDelivery() {
  if (deliveryRefreshing.value) return;
  deliveryRefreshing.value = true;
  deliveryError.value = '';
  if (!deliveryForm.value.project_id) {
    deliveryRefreshing.value = false;
    return;
  }
  try {
    deliveryPollTick += 1;
    if (deliveryPollTick % 2 === 1) {
      await store.fetchDeliveryRuns(deliveryForm.value.project_id);
    }
    if (!store.activeDeliveryRun && store.deliveryRuns.length) {
      store.activeDeliveryRun = store.deliveryRuns[0];
    }
    if (store.activeDeliveryRun?.id) {
      const runId = store.activeDeliveryRun.id;
      const tasks = [store.fetchDeliveryRun(runId), store.fetchDeliverySessions(runId)];
      if (deliveryPollTick % 2 === 0) {
        tasks.push(store.fetchDeliveryBacklog(runId), store.fetchDeliveryGates(runId), store.fetchDeliveryArtifacts(runId));
      }
      if (deliveryPollTick % 3 === 0) {
        tasks.push(store.fetchDeliveryGovernance(runId), store.fetchDeliverySprints(runId));
      }
      await Promise.all(tasks);
      const now = Date.now();
      if (desktopLiveLogs.value && desktopRunId.value && now - lastSessionLogFetch >= deliveryLogIntervalMs.value) {
        lastSessionLogFetch = now;
        updateLogWorkerSessions();
      }
      if (deliveryRunning.value && !deliveryPollActive && pageVisible.value) {
        startDeliveryPolling();
      }
      if (!deliveryRunning.value && deliveryPollActive) {
        stopDeliveryPolling();
      }
    }
  } catch (err: any) {
    deliveryError.value = err?.message || t('branddozer.error_delivery_status');
  } finally {
    deliveryRefreshing.value = false;
  }
}

function startDeliveryPolling() {
  stopDeliveryPolling();
  if (!pageVisible.value) return;
  deliveryPollActive = true;
  const tick = async () => {
    if (!deliveryPollActive) return;
    if (store.activeDeliveryRun?.id) {
      await refreshDelivery();
    }
    if (!deliveryPollActive) return;
    deliveryTimer = window.setTimeout(tick, deliveryPollIntervalMs.value);
  };
  deliveryTimer = window.setTimeout(tick, deliveryPollIntervalMs.value);
}

async function refreshSessionLogs() {
  if (!store.deliverySessions.length) return;
  const sessionIds = desktopWindowSessions.value.length
    ? desktopWindowSessions.value.map((session: any) => session.id)
    : sortedDeliverySessions.value.map((session: any) => session.id);
  if (!sessionIds.length) return;
  await Promise.all(
    sessionIds.map((sessionId: string) => store.fetchDeliverySessionLogs(sessionId, deliveryLogLimit.value)),
  );
}

function ensureLogWorker() {
  if (deliveryLogWorker.value) return;
  deliveryLogWorker.value = new Worker(new URL('../workers/deliveryLogs.worker.ts', import.meta.url), { type: 'module' });
  deliveryLogWorker.value.onmessage = (event: MessageEvent<any>) => {
    const payload = event.data || {};
    if (payload.type === 'logs' && payload.sessionId) {
      store.appendDeliverySessionLogs(payload.sessionId, payload.lines || [], payload.cursor, payload.reset);
    }
  };
}

function logWorkerConfig() {
  return {
    intervalMs: logWorkerIntervalMs.value,
    maxBytes: logWorkerMaxBytes.value,
    concurrency: logWorkerConcurrency.value,
    runId: desktopRunId.value,
  };
}

function startLogWorker() {
  if (!desktopRunId.value || !desktopLiveLogs.value) return;
  ensureLogWorker();
  const sessions = store.deliverySessions.map((session: any) => ({ id: session.id }));
  deliveryLogWorker.value?.postMessage({ type: 'start', sessions, config: logWorkerConfig() });
  workerActive.value = true;
}

function updateLogWorkerSessions() {
  if (!workerActive.value || !deliveryLogWorker.value) return;
  const sessions = store.deliverySessions.map((session: any) => ({ id: session.id }));
  deliveryLogWorker.value.postMessage({ type: 'update', sessions, config: logWorkerConfig() });
}

function stopLogWorker() {
  if (!deliveryLogWorker.value) return;
  deliveryLogWorker.value.postMessage({ type: 'stop' });
  workerActive.value = false;
}

function resetLogWorker(runId: string) {
  if (!deliveryLogWorker.value) return;
  deliveryLogWorker.value.postMessage({ type: 'reset', runId });
}

async function openDeliveryDesktop(force = false) {
  if (!force && !activeDelivery.value?.id) return;
  if (desktopOpenTimer) {
    window.clearTimeout(desktopOpenTimer);
    desktopOpenTimer = null;
  }
  deliveryDesktopOpen.value = true;
  deliveryDesktopMinimized.value = false;
  desktopReady.value = false;
  desktopLiveLogs.value = true;
  await nextTick();
  startLogWorker();
  desktopOpenTimer = window.setTimeout(() => {
    desktopReady.value = true;
    if (activeDelivery.value?.id && desktopSeededRunId.value !== activeDelivery.value.id) {
      scatterWindows();
      desktopSeededRunId.value = activeDelivery.value.id;
    }
  }, 200);
}

function minimizeDeliveryDesktop() {
  if (deliveryComplete.value) {
    closeDeliveryDesktop();
    return;
  }
  deliveryDesktopOpen.value = false;
  deliveryDesktopMinimized.value = true;
  desktopLiveLogs.value = false;
  stopLogWorker();
}

function restoreDeliveryDesktop() {
  deliveryDesktopOpen.value = true;
  deliveryDesktopMinimized.value = false;
  if (!desktopReady.value) {
    desktopReady.value = true;
  }
  if (desktopLiveLogs.value) {
    startLogWorker();
  }
}

function closeDeliveryDesktop() {
  deliveryDesktopOpen.value = false;
  deliveryDesktopMinimized.value = false;
  desktopLiveLogs.value = false;
  desktopReady.value = false;
  stopLogWorker();
  if (desktopOpenTimer) {
    window.clearTimeout(desktopOpenTimer);
    desktopOpenTimer = null;
  }
}

function toggleDesktopLogs() {
  desktopLiveLogs.value = !desktopLiveLogs.value;
  if (desktopLiveLogs.value) {
    startLogWorker();
  } else {
    stopLogWorker();
  }
}

function applyDesktopLayout(layout: 'free' | 'grid' | 'masonry' | 'cascade') {
  desktopLayout.value = layout;
  if (layout === 'cascade') {
    cascadeWindows();
  }
}

function scatterWindows() {
  if (desktopLayout.value === 'grid' || desktopLayout.value === 'masonry') return;
  const desktop = desktopRef.value;
  const rect = desktop?.getBoundingClientRect();
  const padding = 24;
  const maxX = rect ? Math.max(padding, rect.width - 320) : 600;
  const maxY = rect ? Math.max(padding, rect.height - 240) : 320;
  const updated = { ...windowPositions.value };
  desktopWindowSessions.value.forEach((session: any, idx: number) => {
    updated[session.id] = {
      x: Math.round(padding + Math.random() * Math.max(0, maxX - padding)),
      y: Math.round(padding + Math.random() * Math.max(0, maxY - padding)),
      z: 10 + idx,
    };
  });
  windowPositions.value = updated;
}

function ensureWindowPositions() {
  const desktop = desktopRef.value;
  const rect = desktop?.getBoundingClientRect();
  const maxX = rect ? Math.max(0, rect.width - 340) : 600;
  const maxY = rect ? Math.max(0, rect.height - 260) : 320;
  const updated = { ...windowPositions.value };
  desktopWindowSessions.value.forEach((session: any, idx: number) => {
    if (!updated[session.id]) {
      updated[session.id] = {
        x: Math.round(Math.random() * maxX),
        y: Math.round(Math.random() * maxY),
        z: 10 + idx,
      };
    }
  });
  windowPositions.value = updated;
}

function cascadeWindows() {
  const updated = { ...windowPositions.value };
  desktopWindowSessions.value.forEach((session: any, idx: number) => {
    updated[session.id] = {
      x: 40 + idx * 40,
      y: 40 + idx * 32,
      z: 10 + idx,
    };
  });
  windowPositions.value = updated;
}

function bringToFront(sessionId: string) {
  const current = windowPositions.value[sessionId];
  if (!current) return;
  const maxZ = Math.max(0, ...Object.values(windowPositions.value).map((pos) => pos.z));
  windowPositions.value = {
    ...windowPositions.value,
    [sessionId]: { ...current, z: maxZ + 1 },
  };
  focusedSessionId.value = sessionId;
}

function startDrag(sessionId: string, event: PointerEvent) {
  if (desktopLayout.value === 'grid' || desktopLayout.value === 'masonry') return;
  const target = event.currentTarget as HTMLElement | null;
  const rect = target?.getBoundingClientRect();
  const pos = windowPositions.value[sessionId];
  if (!rect || !pos) return;
  dragState.value = {
    id: sessionId,
    offsetX: event.clientX - rect.left,
    offsetY: event.clientY - rect.top,
  };
  bringToFront(sessionId);
  window.addEventListener('pointermove', onDragMove);
  window.addEventListener('pointerup', stopDrag);
}

function onDragMove(event: PointerEvent) {
  if (!dragState.value || !desktopRef.value) return;
  const rect = desktopRef.value.getBoundingClientRect();
  const x = Math.max(0, event.clientX - rect.left - dragState.value.offsetX);
  const y = Math.max(0, event.clientY - rect.top - dragState.value.offsetY);
  windowPositions.value = {
    ...windowPositions.value,
    [dragState.value.id]: {
      ...(windowPositions.value[dragState.value.id] || { z: 1 }),
      x,
      y,
    },
  };
}

function stopDrag() {
  dragState.value = null;
  window.removeEventListener('pointermove', onDragMove);
  window.removeEventListener('pointerup', stopDrag);
}

function getWindowStyle(sessionId: string) {
  if (desktopLayout.value !== 'free' && desktopLayout.value !== 'cascade') {
    return {};
  }
  const pos = windowPositions.value[sessionId];
  if (!pos) return {};
  return {
    left: `${pos.x}px`,
    top: `${pos.y}px`,
    zIndex: pos.z,
  };
}

function setLogRef(sessionId: string) {
  return (el: any | null) => {
    logRefs.value = { ...logRefs.value, [sessionId]: el };
  };
}

function sessionLogLines(sessionId: string) {
  return store.deliverySessionLogs[sessionId] || [];
}

function focusSession(sessionId: string) {
  focusedSessionId.value = sessionId;
  if (desktopLiveLogs.value) {
    if (workerActive.value) {
      updateLogWorkerSessions();
    } else {
      refreshSessionLogs();
    }
  }
}

async function startDeliveryRun() {
  deliveryError.value = '';
  if (!deliveryForm.value.project_id) {
    deliveryError.value = t('branddozer.error_select_project');
    return;
  }
  if (!deliveryForm.value.prompt.trim()) {
    deliveryError.value = t('branddozer.error_add_prompt');
    return;
  }
  try {
    const run = await store.startDeliveryRun({
      project_id: deliveryForm.value.project_id,
      prompt: deliveryForm.value.prompt,
      mode: deliveryForm.value.mode,
      team_mode: deliveryForm.value.team_mode,
      session_provider: deliveryForm.value.session_provider,
      codex_model: deliveryForm.value.codex_model,
      codex_reasoning: deliveryForm.value.codex_reasoning,
      c0d3r_model: deliveryForm.value.c0d3r_model,
      c0d3r_reasoning: deliveryForm.value.c0d3r_reasoning,
      smoke_test_cmd: deliveryForm.value.smoke_test_cmd,
    });
    store.activeDeliveryRun = run;
    deliveryStatusNote.value = 'Delivery run started.';
    desktopSeededRunId.value = '';
    windowPositions.value = {};
    focusedSessionId.value = '';
    store.resetDeliverySessionLogs();
    resetLogWorker(run?.id || '');
    openDeliveryDesktop();
    await refreshDelivery();
  } catch (err: any) {
    deliveryError.value = err?.message || t('branddozer.error_start_delivery');
  }
}

async function stopDeliveryRun() {
  deliveryError.value = '';
  if (!activeDelivery.value?.id) {
    deliveryError.value = t('branddozer.error_no_active_delivery');
    return;
  }
  try {
    await store.stopDeliveryRun(activeDelivery.value.id);
    deliveryStatusNote.value = 'Stop requested.';
    await refreshDelivery();
  } catch (err: any) {
    deliveryError.value = err?.message || t('branddozer.error_stop_delivery');
  }
}

async function runUiCapture() {
  uiCaptureError.value = '';
  uiCaptureStatus.value = '';
  if (!activeDelivery.value?.id) {
    uiCaptureError.value = t('branddozer.error_select_delivery');
    return;
  }
  try {
    await store.triggerDeliveryUiCapture(activeDelivery.value.id);
    uiCaptureStatus.value = t('branddozer.capture_queued');
    window.setTimeout(() => {
      refreshDelivery();
    }, 3000);
  } catch (err: any) {
    uiCaptureError.value = err?.message || t('branddozer.error_capture');
  }
}

function artifactUrl(artifact: any) {
  return `/api/branddozer/delivery/artifacts/${artifact.id}/file/`;
}

function openScreenshot(artifact: any) {
  selectedScreenshot.value = artifact;
  screenshotOpen.value = true;
}

async function acceptDeliveryRun() {
  deliveryError.value = '';
  if (!store.activeDeliveryRun?.id) return;
  try {
    await store.acceptDeliveryRun(store.activeDeliveryRun.id, {
      notes: t('branddozer.accepted_via_dashboard'),
      checklist: store.activeDeliveryRun.definition_of_done || [],
    });
    deliveryStatusNote.value = t('branddozer.acceptance_recorded');
    await refreshDelivery();
  } catch (err: any) {
    deliveryError.value = err?.message || t('branddozer.error_accept_delivery');
  }
}

function setImportStatus(message: string, detail?: string) {
  githubImportStatus.value = message;
  if (detail !== undefined) {
    githubImportDetail.value = detail;
  }
}

function stopImportPolling() {
  if (importTimer) {
    window.clearInterval(importTimer);
    importTimer = null;
  }
}

async function pollImportStatus(jobId: string) {
  try {
    const data = await store.fetchGithubImportStatus(jobId);
    const message = data.message || data.step || t('common.working');
    const detail = data.detail || '';
    setImportStatus(message, detail);
    if (data.error) {
      githubError.value = data.error;
      setImportStatus(t('branddozer.import_failed'), data.error);
    }
    if (data.status === 'completed') {
      stopImportPolling();
      githubImportJobId.value = '';
      setImportStatus(t('branddozer.import_complete'));
      if (data.project?.id) {
        await store.load();
        selectedId.value = data.project.id;
        await store.refreshLogs(data.project.id, 200);
      }
    }
    if (data.status === 'error') {
      stopImportPolling();
      githubImportJobId.value = '';
      setImportStatus(t('branddozer.import_failed'), data.error || '');
    }
  } catch (err: any) {
    stopImportPolling();
    githubImportJobId.value = '';
    const message = err?.message || t('branddozer.import_status_failed');
    githubError.value = message;
    setImportStatus(t('branddozer.import_failed'), message);
  }
}

function startImportPolling(jobId: string) {
  stopImportPolling();
  importTimer = window.setInterval(() => {
    pollImportStatus(jobId);
  }, 1200);
}

async function connectGithub() {
  githubError.value = '';
  if (!githubAccountForm.value.token) {
    githubError.value = t('branddozer.error_add_pat');
    return;
  }
  try {
    const payload = {
      username: githubAccountForm.value.username || undefined,
      token: githubAccountForm.value.token,
      account_id: githubAccountSelection.value !== 'new' ? githubAccountSelection.value : undefined,
    };
    await store.saveGithubAccount(payload);
    githubAccountSelection.value = store.githubActiveAccountId || githubAccountSelection.value;
    githubAccountForm.value.username = activeGithubAccount.value?.username || payload.username || githubAccountForm.value.username;
    githubAccountForm.value.token = '';
    await refreshGithubRepos();
  } catch (err: any) {
    githubError.value = err?.message || t('branddozer.error_save_token');
  }
}

function resetGithubForm() {
  githubAccountForm.value.token = '';
  githubAccountForm.value.username = activeGithubAccount.value?.username || githubAccountForm.value.username;
  githubImportForm.value = {
    repo_full_name: '',
    branch: '',
    destination: '',
    name: '',
    default_prompt: '',
  };
  githubRepoSearch.value = '';
  lastAutoDestination.value = '';
  githubError.value = '';
  githubImportJobId.value = '';
  setImportStatus(t('branddozer.import_ready'), '');
  stopImportPolling();
  store.githubBranches = [];
}

function toggleForm() {
  showForm.value = !showForm.value;
}

function resetForm() {
  form.value = {
    id: '',
    name: '',
    root_path: folderState.value.current_path || folderState.value.home || '',
    default_prompt: '',
    interjections: [],
    interval_minutes: 120,
  };
  interjectionError.value = '';
  showForm.value = false;
}

async function runGithubImport() {
  githubError.value = '';
  if (!githubImportForm.value.repo_full_name) {
    githubError.value = 'Select a repository to import.';
    return;
  }
  const payload: Record<string, any> = {
    repo_full_name: githubImportForm.value.repo_full_name,
    branch: githubImportForm.value.branch || undefined,
    destination: githubImportForm.value.destination || undefined,
    name: githubImportForm.value.name || undefined,
    default_prompt: githubImportForm.value.default_prompt || form.value.default_prompt || undefined,
    remember_token: true,
    async: true,
  };
  const accountId = store.githubActiveAccountId || (githubAccountSelection.value !== 'new' ? githubAccountSelection.value : '');
  if (accountId) {
    payload.account_id = accountId;
  }
  try {
    setImportStatus(t('branddozer.import_starting'), '');
    const data = await store.importFromGitHub(payload);
    const jobId = data.job_id;
    if (jobId) {
      githubImportJobId.value = jobId;
      await pollImportStatus(jobId);
      startImportPolling(jobId);
    } else if (data.project?.id) {
      resetGithubForm();
      selectedId.value = data.project.id;
      await store.refreshLogs(data.project.id, 200);
      setImportStatus(t('branddozer.import_complete'));
    }
  } catch (err: any) {
    const message = err?.message || 'GitHub import failed';
    githubError.value = message;
    setImportStatus(t('branddozer.import_failed'), message);
  }
}

async function saveProject() {
  if (!form.value.root_path && folderState.value.current_path) {
    form.value.root_path = folderState.value.current_path;
  }
  if (form.value.id) {
    await store.update(form.value.id, form.value);
  } else {
    await store.create(form.value);
  }
  resetForm();
}

function editProject(project: any) {
  form.value = {
    id: project.id,
    name: project.name,
    root_path: project.root_path,
    default_prompt: project.default_prompt,
    interjections: [...(project.interjections || [])],
    interval_minutes: project.interval_minutes,
  };
  interjectionError.value = '';
  showForm.value = true;
}

async function remove(id: string) {
  await store.remove(id);
  if (selectedId.value === id) {
    selectedId.value = store.projects[0]?.id || '';
  }
}

async function start(id: string) {
  await store.start(id);
  selectedId.value = id;
  refreshLogs();
}

async function stop(id: string) {
  await store.stop(id);
}

async function refreshLogs() {
  if (!selectedId.value) return;
  await store.refreshLogs(selectedId.value, 200);
  nextTick(() => {
    if (logBox.value) {
      logBox.value.scrollTop = logBox.value.scrollHeight;
    }
  });
}

function startLogTimer() {
  stopLogTimer();
  if (!selectedId.value || !pageVisible.value) return;
  logTimer = window.setInterval(() => {
    const project = store.projects.find((p) => p.id === selectedId.value);
    if (project?.running && pageVisible.value) {
      store.refreshLogs(project.id, 200);
    }
  }, 5000);
}

function stopLogTimer() {
  if (logTimer) {
    window.clearInterval(logTimer);
    logTimer = null;
  }
}

function addInterjection() {
  form.value.interjections.push('');
}

function removeInterjection(idx: number) {
  form.value.interjections.splice(idx, 1);
}

function selectProject(id: string) {
  selectedId.value = id;
}

function openAiConfirm() {
  interjectionError.value = '';
  confirmOpen.value = true;
}

async function openPublish(project: any) {
  publishTarget.value = project;
  publishForm.value.message = `Update ${project.name}`;
  publishForm.value.repo_name = project.repo_url ? '' : (project.name || '').toLowerCase().replace(/[^a-z0-9-_]+/g, '-');
  publishForm.value.private = true;
  publishError.value = '';
  publishStatus.value = '';
  stopPublishPolling();
  if (!store.githubAccounts.length && !store.githubAccountLoading) {
    try {
      await store.loadGithubAccount();
    } catch (err) {
      // Ignore account refresh errors here; they surface on publish.
    }
  }
  publishOpen.value = true;
}

function closePublishModal() {
  publishOpen.value = false;
  stopPublishPolling();
}

function stopPublishPolling() {
  if (publishTimer) {
    window.clearInterval(publishTimer);
    publishTimer = null;
  }
  publishJobId.value = '';
}

function startPublishPolling(jobId: string) {
  stopPublishPolling();
  publishJobId.value = jobId;
  publishTimer = window.setInterval(() => {
    pollPublishStatus(jobId);
  }, 1500);
}

function formatAheadBehind(result: any) {
  const ahead = Number.isFinite(result?.ahead) ? Number(result.ahead) : 0;
  const behind = Number.isFinite(result?.behind) ? Number(result.behind) : 0;
  if (!ahead && !behind) return '';
  const parts = [];
  if (ahead) parts.push(t('branddozer.ahead').replace('{count}', String(ahead)));
  if (behind) parts.push(t('branddozer.behind').replace('{count}', String(behind)));
  return t('branddozer.local_ahead_behind').replace('{details}', parts.join(', '));
}

async function pollPublishStatus(jobId: string) {
  try {
    const data = await store.fetchGithubPublishStatus(jobId);
    if (data.status === 'queued') {
      publishStatus.value = t('branddozer.push_queued');
    } else if (data.status === 'running') {
      publishStatus.value = data.message || t('branddozer.pushing');
    } else if (data.message) {
      publishStatus.value = data.message;
    }
    if (data.status === 'completed') {
      stopPublishPolling();
      const resultStatus = data?.result?.status;
      const branch = data?.result?.branch;
      const repoUrl = data?.result?.repo_url;
      if (resultStatus === 'no_changes') {
        const detail = data?.result?.detail || t('branddozer.no_changes');
        publishStatus.value = `${detail}${formatAheadBehind(data?.result)}${branch ? t('branddozer.branch_suffix').replace('{branch}', String(branch)) : ''}`;
      } else {
        const target = repoUrl || 'GitHub';
        publishStatus.value = t('branddozer.pushed_to')
          .replace('{target}', String(target))
          .replace('{branch}', branch ? t('branddozer.branch_suffix').replace('{branch}', String(branch)) : '');
      }
    }
    if (data.status === 'error') {
      stopPublishPolling();
      publishError.value = data.error || t('branddozer.error_push');
    }
  } catch (err: any) {
    stopPublishPolling();
    publishError.value = err?.message || t('branddozer.error_publish_status');
  }
}

async function runPublish() {
  if (!publishTarget.value?.id) return;
  publishError.value = '';
  publishStatus.value = t('branddozer.preparing_push');
  if (!store.githubAccounts.length && !store.githubAccountLoading) {
    try {
      await store.loadGithubAccount();
    } catch (err) {
      // Ignore auto-load errors; we'll handle missing auth below.
    }
  }
  if (githubTokenLocked.value) {
    publishError.value = t('branddozer.github_token_locked');
    return;
  }
  if (!githubConnected.value) {
    publishError.value = t('branddozer.github_not_connected_detail');
    return;
  }
  const payload: Record<string, any> = {
    message: publishForm.value.message,
    repo_name: publishForm.value.repo_name || undefined,
    private: publishForm.value.private,
  };
  const accountId = store.githubActiveAccountId || (githubAccountSelection.value !== 'new' ? githubAccountSelection.value : '');
  if (accountId) {
    payload.account_id = accountId;
  }
  try {
    const data = await store.publishProject(publishTarget.value.id, payload);
    if (data?.job_id) {
      publishStatus.value = t('branddozer.push_queued_short');
      startPublishPolling(data.job_id);
      return;
    }
    if (data?.status === 'no_changes') {
      const detail = data?.detail || t('branddozer.no_changes');
      publishStatus.value = `${detail}${formatAheadBehind(data)}${data?.branch ? t('branddozer.branch_suffix').replace('{branch}', String(data.branch)) : ''}`;
    } else {
      const target = data?.repo_url || 'GitHub';
      publishStatus.value = t('branddozer.pushed_to')
        .replace('{target}', String(target))
        .replace('{branch}', data?.branch ? t('branddozer.branch_suffix').replace('{branch}', String(data.branch)) : '');
    }
  } catch (err: any) {
    publishError.value = resolveErrorMessage(err, t('branddozer.error_push'));
  }
}

async function generateInterjections() {
  interjectionError.value = '';
  confirmOpen.value = false;
  const defaultPrompt = form.value.default_prompt.trim();
  if (!defaultPrompt) {
    interjectionError.value = t('branddozer.error_default_prompt_required');
    return;
  }
  try {
    const id = form.value.id;
    let prompts: string[] = [];
    if (id) {
      prompts = await store.generateInterjections(id, defaultPrompt);
    } else {
      prompts = await store.generateInterjectionsPreview(defaultPrompt, form.value.name || 'Project');
    }
    if (prompts && prompts.length) {
      form.value.interjections = prompts;
    } else {
      interjectionError.value = t('branddozer.error_no_interjections');
    }
  } catch (err: any) {
    interjectionError.value = resolveErrorMessage(err, t('branddozer.error_generate_interjections'));
  }
}

function resolveErrorMessage(err: any, fallback: string) {
  return err?.response?.data?.detail || err?.response?.data?.error || err?.message || fallback;
}

function formatTime(ts?: number | string | null) {
  if (!ts) return t('common.na');
  const value = Number(ts);
  if (!Number.isFinite(value)) return t('common.na');
  const delta = Date.now() / 1000 - value;
  if (delta < 60) return t('common.just_now');
  if (delta < 3600) return t('common.minutes_ago').replace('{count}', String(Math.round(delta / 60)));
  if (delta < 86400) return t('common.hours_ago').replace('{count}', String(Math.round(delta / 3600)));
  return t('common.days_ago').replace('{count}', String(Math.round(delta / 86400)));
}
</script>

<style scoped>
.branddozer {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}

.panel {
  background: rgba(8, 16, 30, 0.92);
  border: 1px solid rgba(94, 152, 255, 0.25);
  border-radius: 20px;
  padding: 1.2rem 1.4rem;
  color: #e5edff;
}

.panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin-bottom: 0.8rem;
}

.caption {
  color: rgba(229, 237, 255, 0.7);
  font-size: 0.85rem;
}

.caption.warn {
  color: #f6b143;
}

.projects-panel {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  min-width: 0;
}

.project-form {
  background: rgba(6, 12, 22, 0.8);
  border: 1px solid rgba(126, 168, 255, 0.2);
  border-radius: 14px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.project-list-panel {
  grid-column: 1 / -1;
}

.logs-panel {
  grid-column: 1 / -1;
}

.project-form label {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.9rem;
}

.path-picker {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.4rem;
  align-items: center;
}

.project-form input,
.project-form textarea,
.logs-panel select,
.import-card input,
.import-card textarea,
.import-card select,
.delivery-card input,
.delivery-card textarea,
.delivery-card select {
  background: rgba(3, 8, 18, 0.9);
  border: 1px solid rgba(126, 168, 255, 0.3);
  color: #e5edff;
  border-radius: 10px;
  padding: 0.55rem 0.7rem;
  font-size: 0.92rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.7rem;
}

.form-grid.compact {
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.55rem;
}

.form-grid .full {
  grid-column: 1 / -1;
}

.import-card {
  background: rgba(6, 12, 22, 0.8);
  border: 1px dashed rgba(126, 168, 255, 0.4);
  border-radius: 14px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.delivery-card {
  background: rgba(6, 12, 22, 0.85);
  border: 1px solid rgba(126, 168, 255, 0.3);
  border-radius: 16px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.delivery-status {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.6rem;
  background: rgba(5, 12, 24, 0.7);
  border: 1px solid rgba(126, 168, 255, 0.2);
  border-radius: 12px;
  padding: 0.6rem 0.75rem;
}

.delivery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.7rem;
}

.delivery-block {
  background: rgba(5, 12, 24, 0.7);
  border: 1px solid rgba(126, 168, 255, 0.25);
  border-radius: 12px;
  padding: 0.6rem 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.delivery-block h4 {
  margin: 0;
  font-size: 0.95rem;
}

.delivery-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
}

.import-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}

.import-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.status-chip {
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(126, 168, 255, 0.4);
  background: rgba(5, 12, 24, 0.9);
  color: #e5edff;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.85rem;
}

.status-chip.ok {
  border-color: rgba(52, 211, 153, 0.6);
  color: #34d399;
}

.status-chip.warn {
  border-color: rgba(246, 177, 67, 0.6);
  color: #f6b143;
}

.github-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 0.8rem;
}

.github-block {
  background: rgba(3, 8, 18, 0.7);
  border: 1px solid rgba(126, 168, 255, 0.25);
  border-radius: 12px;
  padding: 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.github-block.wide {
  margin-top: 0.4rem;
}

.block-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.5rem;
}

.connection-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.repo-meta {
  background: rgba(255, 255, 255, 0.03);
  border: 1px dashed rgba(126, 168, 255, 0.3);
  border-radius: 10px;
  padding: 0.5rem 0.65rem;
}

.import-status {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  background: rgba(5, 12, 24, 0.85);
  border: 1px solid rgba(126, 168, 255, 0.25);
  border-radius: 12px;
  padding: 0.6rem 0.8rem;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: rgba(229, 237, 255, 0.6);
  font-size: 0.75rem;
  margin: 0;
}

.muted {
  color: rgba(229, 237, 255, 0.65);
}

.interjections {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.interjections-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.interjection-actions {
  display: flex;
  gap: 0.35rem;
}

.interjection-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.5rem;
  align-items: start;
}

.actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.btn.small {
  padding: 0.4rem 0.8rem;
  font-size: 0.85rem;
}

.project-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.6rem;
}

.project-card {
  background: rgba(10, 18, 32, 0.8);
  border: 1px solid rgba(122, 170, 255, 0.2);
  border-radius: 12px;
  padding: 0.8rem;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  min-width: 0;
}

.project-card.selected {
  border-color: rgba(94, 152, 255, 0.7);
  box-shadow: 0 0 12px rgba(94, 152, 255, 0.3);
}

.project-card header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem 0.75rem;
}

.path {
  color: rgba(229, 237, 255, 0.65);
  overflow-wrap: anywhere;
}

.status-pill {
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.8rem;
  text-transform: uppercase;
}

.status-pill.ok {
  background: rgba(34, 197, 94, 0.15);
  border: 1px solid rgba(34, 197, 94, 0.5);
  color: #34d399;
}

.status-pill.warn {
  background: rgba(250, 204, 21, 0.15);
  border: 1px solid rgba(250, 204, 21, 0.5);
  color: #facc15;
}

.meta {
  color: rgba(229, 237, 255, 0.7);
  font-size: 0.9rem;
}

.card-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.logs-panel .console-output {
  min-height: 420px;
  background: rgba(5, 10, 20, 0.9);
  border: 1px solid rgba(126, 168, 255, 0.2);
  border-radius: 12px;
  padding: 0.8rem;
  white-space: pre-wrap;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: #dbeafe;
  overflow-y: auto;
}

.empty {
  color: rgba(229, 237, 255, 0.6);
  font-style: italic;
}

.error {
  color: #ff9b9b;
  background: rgba(255, 90, 95, 0.1);
  border: 1px solid rgba(255, 90, 95, 0.4);
  border-radius: 10px;
  padding: 0.55rem 0.75rem;
}

.header-actions {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
}

.modal-card {
  background: rgba(8, 12, 22, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.35);
  border-radius: 14px;
  padding: 1.2rem;
  width: min(420px, 92vw);
  color: #e5edff;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.modal-card.wide {
  width: min(720px, 95vw);
}

.publish-card {
  background: rgba(8, 12, 22, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.35);
  border-radius: 0;
  padding: 1.2rem;
  width: min(520px, 92vw);
  color: #e5edff;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  position: relative;
  overflow-x: hidden;
}

.publish-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}

.publish-close {
  position: absolute;
  top: 0.6rem;
  right: 0.6rem;
  min-width: 32px;
  height: 32px;
  border: 1px solid rgba(126, 168, 255, 0.45);
  color: #dbeafe;
  font-weight: 600;
}

.publish-grid {
  display: grid;
  gap: 0.6rem;
}

.publish-cancel {
  border-color: rgba(126, 168, 255, 0.55) !important;
  color: #dbeafe !important;
}

.folder-controls {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.current-path {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: rgba(229, 237, 255, 0.85);
  word-break: break-all;
}

.folder-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.5rem;
  max-height: 320px;
  overflow-y: auto;
}

.folder-row {
  text-align: left;
  border: 1px solid rgba(126, 168, 255, 0.3);
  background: rgba(5, 12, 24, 0.8);
  border-radius: 10px;
  padding: 0.6rem 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  color: #e5edff;
  cursor: pointer;
}

.folder-row:hover {
  border-color: rgba(126, 168, 255, 0.7);
}

.delivery-desktop {
  width: 100%;
  height: 100%;
  background: radial-gradient(circle at 10% 0%, rgba(10, 28, 60, 0.6), rgba(3, 9, 18, 0.98));
  color: #e5edff;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
  padding: 0.9rem;
  position: relative;
  overflow: hidden;
}

.delivery-desktop.performance-mode {
  background: #040a16;
}

.delivery-desktop.performance-mode::before,
.delivery-desktop.performance-mode::after {
  display: none;
}

.delivery-desktop::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(0deg, rgba(125, 176, 255, 0.06), rgba(125, 176, 255, 0.06) 1px, transparent 1px, transparent 22px),
    repeating-linear-gradient(90deg, rgba(125, 176, 255, 0.04), rgba(125, 176, 255, 0.04) 1px, transparent 1px, transparent 26px);
  opacity: 0.35;
  pointer-events: none;
}

.delivery-desktop::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  height: 120px;
  top: -120px;
  background: linear-gradient(180deg, rgba(125, 176, 255, 0) 0%, rgba(125, 176, 255, 0.15) 50%, rgba(125, 176, 255, 0) 100%);
  animation: scanline 6s linear infinite;
  pointer-events: none;
}

.desktop-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  flex-wrap: wrap;
  position: relative;
  z-index: 1;
}

.desktop-brand {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.topbar-left {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.topbar-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}

.topbar-actions {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.layout-group {
  display: flex;
  gap: 0.2rem;
  border: 1px solid rgba(126, 168, 255, 0.3);
  padding: 0.2rem;
  background: rgba(5, 12, 24, 0.8);
  border-radius: 0;
}

.desktop-shell {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr minmax(240px, 360px);
  gap: 0.9rem;
  min-height: 0;
  position: relative;
  z-index: 1;
}

.desktop-canvas {
  position: relative;
  border: 1px solid rgba(126, 168, 255, 0.3);
  background: rgba(4, 10, 20, 0.85);
  overflow: hidden;
  min-height: 0;
  border-radius: 0;
}

.desktop-canvas.layout-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.7rem;
  padding: 0.7rem;
}

.desktop-canvas.layout-masonry {
  column-count: 2;
  column-gap: 0.7rem;
  padding: 0.7rem;
}

.desktop-canvas.layout-masonry .desktop-window {
  display: inline-block;
  width: 100%;
  margin: 0 0 0.7rem;
}

.desktop-window {
  width: 320px;
  background: rgba(6, 12, 22, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.45);
  position: absolute;
  display: flex;
  flex-direction: column;
  min-height: 200px;
  max-height: 420px;
  box-shadow: 0 18px 30px rgba(0, 0, 0, 0.45);
  border-radius: 0;
}

.desktop-window.active {
  border-color: rgba(140, 196, 255, 0.85);
  box-shadow: 0 0 0 1px rgba(140, 196, 255, 0.4), 0 18px 30px rgba(0, 0, 0, 0.45);
}

.delivery-desktop.performance-mode .desktop-window {
  box-shadow: none;
}

.desktop-window::after {
  content: '';
  position: absolute;
  inset: 0;
  border: 1px solid rgba(127, 176, 255, 0.2);
  pointer-events: none;
}

.desktop-canvas.layout-grid .desktop-window,
.desktop-canvas.layout-masonry .desktop-window {
  position: static;
  width: 100%;
  max-height: none;
}

.window-titlebar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.45rem 0.6rem;
  background: rgba(10, 18, 32, 0.95);
  border-bottom: 1px solid rgba(126, 168, 255, 0.25);
  cursor: grab;
  user-select: none;
}

.window-title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.9rem;
}

.window-name {
  font-weight: 600;
}

.window-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.role-dot {
  width: 10px;
  height: 10px;
  background: #7fb0ff;
  display: inline-block;
}

.desktop-window.role-orchestrator .role-dot {
  background: #7fb0ff;
}

.desktop-window.role-orchestrator {
  border-color: rgba(127, 176, 255, 0.6);
}

.desktop-window.role-pm .role-dot {
  background: #f6b143;
}

.desktop-window.role-pm {
  border-color: rgba(246, 177, 67, 0.55);
}

.desktop-window.role-integrator .role-dot {
  background: #34d399;
}

.desktop-window.role-integrator {
  border-color: rgba(52, 211, 153, 0.55);
}

.desktop-window.role-dev .role-dot {
  background: #9db9ff;
}

.desktop-window.role-dev {
  border-color: rgba(157, 185, 255, 0.55);
}

.desktop-window.role-qa .role-dot {
  background: #ff5a5f;
}

.desktop-window.role-qa {
  border-color: rgba(255, 90, 95, 0.6);
}

.window-body {
  flex: 1;
  padding: 0.5rem;
  overflow: hidden;
  position: relative;
}

.terminal-output {
  height: 100%;
  flex: 1;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.78rem;
  color: #dbeafe;
}

.terminal-line {
  white-space: pre-wrap;
  word-break: break-word;
}

.terminal-empty,
.terminal-paused {
  position: absolute;
  bottom: 0.6rem;
  left: 0.6rem;
  right: 0.6rem;
  padding: 0.35rem 0.5rem;
  background: rgba(6, 12, 22, 0.8);
  border: 1px solid rgba(126, 168, 255, 0.2);
  font-size: 0.75rem;
  color: rgba(229, 237, 255, 0.6);
}

.desktop-panels {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  min-height: 0;
}

.desktop-module {
  border: 1px solid rgba(126, 168, 255, 0.3);
  background: rgba(6, 12, 22, 0.85);
  padding: 0.6rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-height: 0;
  border-radius: 0;
}

.module-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.4rem;
}

.module-body {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.module-row {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  font-size: 0.82rem;
}

.module-row.stacked {
  flex-direction: column;
  align-items: flex-start;
}

.module-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.evidence-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  gap: 0.5rem;
}

.evidence-thumb {
  border: 1px solid rgba(126, 168, 255, 0.35);
  background: rgba(5, 10, 18, 0.75);
  padding: 0.3rem;
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  cursor: pointer;
}

.evidence-thumb img {
  width: 100%;
  height: 70px;
  object-fit: cover;
  border: 1px solid rgba(126, 168, 255, 0.2);
}

.screenshot-modal {
  min-width: min(900px, 92vw);
  background: rgba(6, 12, 22, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.35);
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.screenshot-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}

.screenshot-body img {
  width: 100%;
  height: auto;
  border: 1px solid rgba(126, 168, 255, 0.2);
}

.checklist-scroll {
  height: 100%;
  max-height: 420px;
}

.checklist-row {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.4rem;
  align-items: start;
  padding: 0.3rem 0;
}

.checklist-text {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}

.desktop-empty {
  padding: 1rem;
  color: rgba(229, 237, 255, 0.65);
}

.desktop-loading {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(229, 237, 255, 0.8);
  font-size: 1rem;
  z-index: 1;
}

.delivery-minibar {
  position: fixed;
  bottom: 0.8rem;
  right: 0.8rem;
  background: rgba(5, 12, 24, 0.92);
  border: 1px solid rgba(126, 168, 255, 0.35);
  padding: 0.6rem 0.9rem;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  cursor: pointer;
  z-index: 4000;
  border-radius: 0;
}

@keyframes scanline {
  0% {
    transform: translateY(0);
  }
  100% {
    transform: translateY(120vh);
  }
}

@media (max-width: 960px) {
  .branddozer {
    grid-template-columns: 1fr;
  }
  .desktop-shell {
    grid-template-columns: 1fr;
  }
  .desktop-canvas {
    min-height: 320px;
  }
  .desktop-canvas.layout-masonry {
    column-count: 1;
  }
}
</style>
