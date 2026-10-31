<template>
  <div class="ar-root">
    <!-- Header -->
    <header class="ar-head">
      <div class="ar-head__titleblock">
        <span class="ar-head__eyebrow">Autonomous Multi-Chain Research · Alpha Wallet Mirror · Vault Sweeper</span>
        <h2 class="ar-head__title">Agent Market Research &amp; Multi-Chain Arbitrage</h2>
        <p class="ar-head__desc">
          Researches crypto chains to discover high-performing alpha wallets with multi-month winning track records,
          confirms them on live incoming data, and graduates them straight to live mirror trading.
          Simulates dynamic buy-low/sell-high strategies against historical OHLCV data, scores itself, and manages profit sweeps
          where all accumulations under $1,000 stay put until you press the manual Send button.
        </p>
      </div>

      <div class="ar-head__actions">
        <!-- Manual Send All Profits Button -->
        <button
          class="btn btn--accent"
          :disabled="sendingProfits || (runningTotals.accumulated_pending_usd <= 0 && vault.accumulated_pending_profits_usd <= 0)"
          @click="sendAllTotalProfits"
        >
          {{ sendingProfits ? 'DISPATCHING...' : `💸 SEND ALL TOTAL PROFITS ($${(runningTotals.accumulated_pending_usd || vault.accumulated_pending_profits_usd || 0).toFixed(2)})` }}
        </button>
        <button class="btn btn--primary" :disabled="scanning" @click="runGhostResearch">
          {{ scanning ? 'RESEARCHING...' : '⚡ RUN GHOST RESEARCH' }}
        </button>
        <button class="btn" :disabled="loading" @click="refreshAll">
          {{ loading ? 'REFRESHING...' : '🔄 REFRESH' }}
        </button>
        <a class="btn btn--secondary" :href="`${API}/export/?format=xlsx`" target="_blank" rel="noopener">
          📊 EXCEL (.XLSX)
        </a>
        <a class="btn" :href="`${API}/export/?format=csv`" target="_blank" rel="noopener">
          📄 CSV
        </a>
      </div>
    </header>

    <!-- Notification Toast -->
    <div v-if="notice" class="ar-notice" :class="{ 'ar-notice--error': isNoticeError }">
      <span>{{ notice }}</span>
      <button class="ar-notice__close" @click="notice = ''">✕</button>
    </div>

    <!-- External Agent Token Limits & Quota Auto-Resume Bar -->
    <div class="ar-token-bar">
      <div class="ar-token-bar__header">
        <span class="ar-token-bar__badge">⚡ TOKEN LIMIT &amp; AUTO-RESUME MONITOR</span>
        <span class="ar-token-bar__desc">
          Unified token and rate limit awareness across external agents (AGY, Codex, Claude). When quotas exhaust, agents pause cleanly and automatically restart as tokens replenish.
        </span>
      </div>
      <div class="ar-token-bar__pills">
        <div
          v-for="(lim, key) in tokenLimits"
          :key="key"
          class="ar-token-pill"
          :class="{ 'ar-token-pill--cooling': !lim.is_available, 'ar-token-pill--ready': lim.is_available }"
        >
          <div class="ar-token-pill__title">
            <span class="indicator-dot" :class="lim.is_available ? 'dot--green' : 'dot--amber-pulse'"></span>
            <strong class="pill-name">{{ String(key).toUpperCase() }}</strong>
            <span class="pill-tag" :class="lim.is_available ? 'tag--green' : 'tag--amber'">
              {{ lim.is_available ? 'Tokens Available' : `Cooling Down (${lim.seconds_remaining}s)` }}
            </span>
          </div>
          <div class="ar-token-pill__meta">
            <span>Hits: {{ lim.total_rate_limits ?? 0 }}</span>
            <span v-if="!lim.is_available" class="text-amber">· Resumes: {{ lim.available_at }}</span>
            <span v-else class="text-muted">· Auto-restart: Armed</span>
          </div>
          <button
            v-if="!lim.is_available"
            class="btn btn--xs btn--outline"
            @click="resetAgentTokenLimit(String(key))"
            title="Clear cooldown manually"
          >
            Clear / Resume
          </button>
        </div>
      </div>
    </div>

    <!-- Summary Metrics Cards -->
    <section class="ar-metrics">
      <div class="card">
        <span class="card__label">Active System Agents</span>
        <span class="card__value text-cyan">{{ counts.active_agents ?? 0 }} / {{ counts.total_agents ?? 0 }}</span>
        <small class="card__sub">AGY &amp; User Agents Connected</small>
      </div>
      <div class="card card--glow">
        <span class="card__label">Alpha Wallets Discovered</span>
        <span class="card__value text-amber">{{ counts.alpha_wallets_count ?? 4 }} ({{ counts.alpha_mirroring_count ?? 3 }} Live)</span>
        <small class="card__sub">3+ Months Consecutive Winning</small>
      </div>
      <div class="card">
        <span class="card__label">Pending Profit (Stays Put)</span>
        <span class="card__value text-green">${{ (runningTotals.accumulated_pending_usd || vault.accumulated_pending_profits_usd || 0).toFixed(2) }}</span>
        <small class="card__sub">Under $1,000 · Retained in Pool</small>
      </div>
      <div class="card">
        <span class="card__label">Total Dispatched Profit</span>
        <span class="card__value text-emerald">${{ (runningTotals.total_dispatched_usd || vault.total_dispatched_profits_usd || 0).toFixed(2) }}</span>
        <small class="card__sub">Sent to Vault / Settlement</small>
      </div>
      <div class="card">
        <span class="card__label">Live Graduated Strategies</span>
        <span class="card__value text-purple">{{ counts.live_active_count ?? 0 }} / {{ counts.graduated_live_count ?? 0 }}</span>
        <small class="card__sub">Active Execution Registry</small>
      </div>
      <div class="card">
        <span class="card__label">Profitable Arbitrage</span>
        <span class="card__value text-yellow">{{ counts.profitable_arbitrage ?? 0 }}</span>
        <small class="card__sub">Net positive after fees &amp; gas</small>
      </div>
    </section>

    <!-- Navigation Tabs -->
    <nav class="ar-tabs">
      <button
        v-for="t in tabs"
        :key="t.id"
        class="ar-tabs__btn"
        :class="{ active: currentTab === t.id }"
        @click="currentTab = t.id"
      >
        <span class="tab-icon">{{ t.icon }}</span>
        <span class="tab-title">{{ t.title }}</span>
        <span v-if="t.count !== undefined" class="tab-count">{{ t.count }}</span>
      </button>
    </nav>

    <!-- TAB 1: SMART MONEY ALPHA WALLETS & LIVE MIRROR TRADING -->
    <section v-if="currentTab === 'alpha'" class="tab-pane">
      <div class="panel">
        <header class="panel__head">
          <div>
            <h3>🎯 Alpha Trader Wallet Tracker &amp; Live Mirroring</h3>
            <p class="panel__desc">
              Researches on-chain DEX transactions across Base, Arbitrum, Ethereum, and Solana to find wallets that have been
              consistently winning for the past several months. Confirms their profitability on incoming live data, and
              <strong>immediately graduates them straight to live mirror trading</strong> (buying what they buy, selling what they sell).
            </p>
          </div>
          <div class="filter-controls">
            <button class="btn btn--primary" :disabled="scanningWallets" @click="scanOnchainWallets">
              {{ scanningWallets ? 'SCANNING ON-CHAIN...' : '🔍 SCAN ON-CHAIN ALPHA' }}
            </button>
            <button class="btn btn--secondary" @click="showAddAlphaModal = true">+ TRACK WALLET</button>
          </div>
        </header>

        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Chain</th>
                <th>Alias &amp; Address</th>
                <th>Track Record</th>
                <th>Historical Win %</th>
                <th>Est. Net PnL</th>
                <th>Preferred Tokens</th>
                <th>Live Status</th>
                <th>Live Confirm Rate</th>
                <th>Live Confirm PnL</th>
                <th>Mirror Mode</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="alphaWallets.length === 0">
                <td colspan="11" class="empty-state">No alpha wallets loaded yet. Click 'Scan On-Chain Alpha' above.</td>
              </tr>
              <tr v-for="w in alphaWallets" :key="w.id" :class="{ 'row--graduated': w.is_mirror_active }">
                <td><span class="chain-badge" :class="`chain--${w.chain}`">{{ w.chain.toUpperCase() }}</span></td>
                <td>
                  <strong>{{ w.alias || 'Alpha Trader' }}</strong><br />
                  <small class="code-font" :title="w.wallet_address">{{ shortenAddress(w.wallet_address) }}</small>
                </td>
                <td><span class="badge badge--sys">{{ w.months_winning }} Months Winning</span></td>
                <td><strong class="good">{{ w.win_rate }}%</strong> ({{ w.total_trades }} trades)</td>
                <td class="good">${{ w.estimated_pnl_usd.toLocaleString() }}</td>
                <td>
                  <div class="token-tags">
                    <span v-for="tok in w.preferred_tokens" :key="tok" class="token-pill">{{ tok }}</span>
                  </div>
                </td>
                <td>
                  <span
                    class="badge"
                    :class="w.verification_status === 'live_mirroring' ? 'badge--live' : (w.verification_status === 'live_confirming' ? 'badge--ghost' : 'badge--sys')"
                  >
                    {{ w.verification_status.toUpperCase() }}
                  </span>
                </td>
                <td>
                  <span v-if="w.live_confirm_trades_count > 0" :class="w.live_confirm_win_rate >= 70 ? 'good' : 'bad'">
                    {{ w.live_confirm_win_rate }}% ({{ w.live_confirm_trades_count }} live)
                  </span>
                  <span v-else class="muted">Pending Stream</span>
                </td>
                <td :class="w.live_confirm_net_pnl_usd >= 0 ? 'good' : 'bad'">
                  ${{ w.live_confirm_net_pnl_usd.toFixed(4) }}
                </td>
                <td>
                  <span class="badge" :class="w.is_mirror_active ? 'badge--live' : 'badge--paused'">
                    {{ w.is_mirror_active ? 'MIRRORING LIVE' : 'MIRROR OFF' }}
                  </span>
                </td>
                <td>
                  <div class="btn-group">
                    <button class="btn btn--sm btn--primary" @click="confirmAlphaWallet(w.id)" title="Evaluate on incoming live data">
                      ⚡ CONFIRM LIVE
                    </button>
                    <button
                      class="btn btn--sm"
                      :class="w.is_mirror_active ? 'btn--secondary' : 'btn--accent'"
                      @click="toggleAlphaMirror(w.id)"
                    >
                      {{ w.is_mirror_active ? 'PAUSE' : 'MIRROR' }}
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 2: MULTI-AGENT HUB & 2-WAY COMMS -->
    <section v-if="currentTab === 'agents'" class="tab-pane">
      <div class="agent-layout">
        <!-- Agent Directory Sidebar -->
        <div class="agent-sidebar">
          <div class="agent-sidebar__head">
            <h3>Connected Agents ({{ agents.length }})</h3>
            <button class="btn btn--sm btn--primary" @click="showNewAgentModal = true">+ NEW AGENT</button>
          </div>
          <div class="agent-list">
            <div
              v-for="ag in agents"
              :key="ag.agent_id"
              class="agent-card"
              :class="{ selected: selectedAgent && selectedAgent.agent_id === ag.agent_id }"
              @click="selectAgent(ag)"
            >
              <div class="agent-card__top">
                <span class="agent-badge" :class="ag.system_role_tag === 'system' ? 'badge--sys' : 'badge--user'">
                  {{ ag.system_role_tag.toUpperCase() }}
                </span>
                <span class="agent-status" :class="`status--${ag.status}`">
                  ● {{ ag.status }}
                </span>
              </div>
              <h4 class="agent-card__name">{{ ag.name }}</h4>
              <p class="agent-card__role">{{ ag.role }}</p>
              <div class="agent-card__meta">
                <span>Model: {{ ag.model_tier }}</span>
                <span v-if="ag.live_net_pnl_usd !== 0" class="good">Live PnL: ${{ ag.live_net_pnl_usd.toFixed(2) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 2-Way Chat Console -->
        <div class="agent-console">
          <div v-if="selectedAgent" class="console-box">
            <header class="console-box__head">
              <div>
                <h3 class="console-box__title">{{ selectedAgent.name }}</h3>
                <small class="console-box__sub">{{ selectedAgent.role }} · Domain: {{ selectedAgent.domain }}</small>
              </div>
              <div class="console-box__stats">
                <span class="chip">Ghost Win Rate: {{ (selectedAgent.ghost_win_rate * 100).toFixed(1) }}%</span>
                <span class="chip" :class="selectedAgent.live_net_pnl_usd >= 0 ? 'chip--good' : 'chip--bad'">
                  Live PnL: ${{ selectedAgent.live_net_pnl_usd.toFixed(2) }}
                </span>
              </div>
            </header>

            <!-- Chat Transcript -->
            <div ref="chatContainerRef" class="chat-messages">
              <div v-if="chatMessages.length === 0" class="empty-state">
                <span>No conversation history yet. Send a directive or query below.</span>
              </div>
              <div
                v-for="msg in chatMessages"
                :key="msg.id"
                class="chat-bubble"
                :class="msg.sender === 'user' ? 'bubble--user' : 'bubble--agent'"
              >
                <div class="bubble-meta">
                  <span class="bubble-sender">{{ msg.sender === 'user' ? 'OPERATOR' : selectedAgent.name }}</span>
                  <span class="bubble-time">{{ formatTime(msg.created_at) }}</span>
                </div>
                <div class="bubble-text">{{ msg.content }}</div>
              </div>
            </div>

            <!-- Chat Input -->
            <form class="chat-form" @submit.prevent="sendAgentMessage">
              <input
                v-model="userMessageInput"
                type="text"
                class="chat-input"
                :placeholder="`Send directive to ${selectedAgent.name}...`"
                :disabled="sendingMessage"
              />
              <button type="submit" class="btn btn--primary" :disabled="sendingMessage || !userMessageInput.trim()">
                {{ sendingMessage ? 'TRANSMITTING...' : 'SEND' }}
              </button>
            </form>
          </div>
          <div v-else class="empty-selection">
            <p>Select an agent from the left directory to open its direct communication channel.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 3: PROFIT ACCRUAL MARKERS & RUNNING TOTALS -->
    <section v-if="currentTab === 'markers'" class="tab-pane">
      <div class="panel">
        <header class="panel__head">
          <div>
            <h3>📊 Profit Accrual Audit Ledger &amp; Status Markers</h3>
            <p class="panel__desc">
              Tracks every dollar of profit accrued by agents and alpha mirror trades.
              <strong>Under $1,000 Rule:</strong> All deposits stay put in the pool (status: <code>STAYS_PUT</code>) until you click
              <strong>SEND ALL TOTAL PROFITS</strong>. Click the button to dispatch all accumulated profits or leave them in the pool.
            </p>
          </div>
          <button
            class="btn btn--accent"
            :disabled="sendingProfits || (runningTotals.accumulated_pending_usd <= 0 && vault.accumulated_pending_profits_usd <= 0)"
            @click="sendAllTotalProfits"
          >
            {{ sendingProfits ? 'DISPATCHING...' : `💸 SEND ALL TOTAL PROFITS ($${(runningTotals.accumulated_pending_usd || vault.accumulated_pending_profits_usd || 0).toFixed(2)})` }}
          </button>
        </header>

        <!-- Running Totals Cards -->
        <div class="marker-totals-grid">
          <div class="total-box">
            <span class="total-box__label">ACCUMULATED PENDING (STAYS PUT)</span>
            <span class="total-box__val good">${{ (runningTotals.accumulated_pending_usd || vault.accumulated_pending_profits_usd || 0).toFixed(2) }}</span>
            <small class="muted">Status: STAYS_PUT · Awaiting manual Send All button</small>
          </div>
          <div class="total-box">
            <span class="total-box__label">TOTAL DISPATCHED PROFITS</span>
            <span class="total-box__val text-cyan">${{ (runningTotals.total_dispatched_usd || vault.total_dispatched_profits_usd || 0).toFixed(2) }}</span>
            <small class="muted">Status: DISPATCHED_TO_VAULT · Swept to Vault/Settlement</small>
          </div>
          <div class="total-box">
            <span class="total-box__label">MANUAL SWEEP THRESHOLD</span>
            <span class="total-box__val">${{ (runningTotals.manual_threshold_usd || 1000).toFixed(2) }}</span>
            <small class="muted">Deposits under $1,000 require manual button trigger</small>
          </div>
          <div class="total-box">
            <span class="total-box__label">ACTIVE MARKERS COUNT</span>
            <span class="total-box__val text-yellow">{{ profitMarkers.length }}</span>
            <small class="muted">Audit tracked markers in database</small>
          </div>
        </div>

        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Marker ID</th>
                <th>Source</th>
                <th>Chain</th>
                <th>Symbol</th>
                <th>Gross Profit</th>
                <th>Sweep %</th>
                <th>Accrued for Sweep</th>
                <th>Status Marker</th>
                <th>&lt; $1,000 Rule Applied?</th>
                <th>Dispatched At</th>
                <th>Audit Notes</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="profitMarkers.length === 0">
                <td colspan="11" class="empty-state">No profit accrual markers recorded yet. Closed profitable trades appear here.</td>
              </tr>
              <tr v-for="m in profitMarkers" :key="m.marker_id">
                <td class="code-font">{{ m.marker_id }}</td>
                <td><span class="badge">{{ m.source }}</span></td>
                <td><span class="chain-badge" :class="`chain--${m.chain}`">{{ m.chain.toUpperCase() }}</span></td>
                <td><strong>{{ m.symbol }}</strong></td>
                <td>${{ m.gross_profit_usd.toFixed(4) }}</td>
                <td>{{ m.allocated_pct }}%</td>
                <td class="good"><strong>${{ m.accrued_profit_usd.toFixed(4) }}</strong></td>
                <td>
                  <span
                    class="badge"
                    :class="m.status === 'dispatched_to_vault' ? 'badge--live' : (m.status === 'stays_put' ? 'badge--ghost' : 'badge--sys')"
                  >
                    {{ m.status.toUpperCase() }}
                  </span>
                </td>
                <td>
                  <span class="badge badge--accent">{{ m.under_1000_rule_applied ? 'YES (STAYS PUT)' : 'NO' }}</span>
                </td>
                <td>{{ m.dispatched_at ? formatTime(m.dispatched_at) : 'Awaiting Send' }}</td>
                <td><small class="muted">{{ m.dispatch_tx_or_note || 'Retained in pool' }}</small></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 4: ENCRYPTED VAULT & SETTLEMENT WALLETS -->
    <section v-if="currentTab === 'vault'" class="tab-pane">
      <div class="vault-grid">
        <!-- Encrypted Vault Card -->
        <div class="panel">
          <header class="panel__head">
            <h3>🔐 Quantum-Resistant Encrypted Vault &amp; Profit Sweeper</h3>
            <span class="badge badge--live">Kyber512 + AES-GCM Encrypted</span>
          </header>
          <p class="panel__desc">
            Your destination profit sweep address/key is encrypted in the database using Kyber Post-Quantum KEM and AES-GCM.
            It can only be decrypted when you explicitly trigger the reveal button below.
          </p>

          <div class="vault-box">
            <!-- Current Masked State -->
            <div class="vault-field">
              <label>Encrypted Vault Secret / Destination Account</label>
              <div class="vault-input-group">
                <input
                  type="text"
                  class="vault-display"
                  :value="revealedSecret || vault.masked_preview || 'No secret encrypted yet'"
                  readonly
                />
                <button
                  type="button"
                  class="btn btn--secondary"
                  :disabled="revealingSecret || !vault.has_secret"
                  @click="toggleRevealSecret"
                >
                  {{ revealedSecret ? '🙈 HIDE SECRET' : (revealingSecret ? 'DECRYPTING...' : '👁️ DECRYPT & REVEAL') }}
                </button>
              </div>
              <small class="muted">
                {{ vault.has_secret ? 'Secret is secured in Kyber ciphertext. Reveal triggers on-demand decryption.' : 'No secret currently stored in the vault.' }}
              </small>
            </div>

            <!-- Profit Sweep Percentage Dropdown -->
            <div class="vault-field">
              <label>Profit Transfer Percentage (%) to Sweep</label>
              <div class="vault-select-group">
                <select v-model="selectedProfitPct" class="vault-select" @change="saveProfitPct">
                  <option :value="0">0% (Keep 100% in trading pool)</option>
                  <option :value="10">10% Profit Sweep</option>
                  <option :value="25">25% Profit Sweep</option>
                  <option :value="50">50% Profit Sweep (Balanced)</option>
                  <option :value="75">75% Profit Sweep (Conservative)</option>
                  <option :value="100">100% Full Profit Sweep</option>
                </select>
                <span class="badge badge--accent">{{ vault.profit_transfer_pct }}% Current Setting</span>
              </div>
              <small class="muted">
                Whenever the agent closes a winning trade or harvests arbitrage edge, this percentage is swept to your designated settlement account.
              </small>
            </div>

            <!-- Update / Replace Vault Secret -->
            <div class="vault-field">
              <label>Store New / Update Vault Destination Secret</label>
              <div class="vault-input-group">
                <input
                  v-model="newVaultSecretInput"
                  type="password"
                  class="vault-input"
                  placeholder="Paste destination wallet private key, API secret, or sweep address..."
                />
                <button
                  type="button"
                  class="btn btn--primary"
                  :disabled="savingVault || !newVaultSecretInput.trim()"
                  @click="saveVaultSecret"
                >
                  {{ savingVault ? 'ENCRYPTING...' : '🔒 ENCRYPT & STORE IN VAULT' }}
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Multi-Chain Settlement Wallets -->
        <div class="panel">
          <header class="panel__head">
            <h3>🌐 Multi-Chain Stablecoin Settlement Wallets</h3>
            <button class="btn btn--sm btn--primary" @click="showAddWalletModal = true">+ ADD / CONFIGURE WALLET</button>
          </header>
          <p class="panel__desc">
            Designated public addresses where trading profits on each chain are settled in stablecoins (`USDC`, `USDT`, `EURC`).
          </p>

          <div class="wallet-table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Chain</th>
                  <th>Stablecoin</th>
                  <th>Wallet Address</th>
                  <th>Allocation</th>
                  <th>Total Settled</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="w in settlementWallets" :key="w.chain">
                  <td><span class="chain-badge" :class="`chain--${w.chain}`">{{ w.chain.toUpperCase() }}</span></td>
                  <td><strong>{{ w.stablecoin_symbol }}</strong></td>
                  <td class="code-font" :title="w.wallet_address">{{ shortenAddress(w.wallet_address) }}</td>
                  <td>{{ w.allocated_profit_pct }}%</td>
                  <td class="good">${{ w.total_settled_usd.toFixed(2) }}</td>
                  <td>
                    <span class="badge" :class="w.is_active ? 'badge--live' : 'badge--ghost'">
                      {{ w.is_active ? 'ACTIVE' : 'INACTIVE' }}
                    </span>
                  </td>
                  <td>
                    <button class="btn btn--sm btn--danger" @click="deleteSettlementWallet(w.chain)">REMOVE</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 5: MARKET SCANNER & LOWEST FEES -->
    <section v-if="currentTab === 'scanner'" class="tab-pane">
      <div class="panel">
        <header class="panel__head">
          <div class="filter-bar">
            <h3>Real-Time Multi-Chain Quotes &amp; Lowest Fee Routing</h3>
            <div class="filter-controls">
              <select v-model="filterScannerChain" class="filter-select" @change="fetchMarketSnapshots">
                <option value="">All Chains</option>
                <option value="base">Base</option>
                <option value="arbitrum">Arbitrum</option>
                <option value="ethereum">Ethereum</option>
                <option value="optimism">Optimism</option>
                <option value="polygon">Polygon</option>
              </select>
              <button class="btn btn--sm" @click="scanMarketsNow">⚡ FORCE SCAN</button>
            </div>
          </div>
        </header>

        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Chain</th>
                <th>DEX</th>
                <th>Pair</th>
                <th>Price</th>
                <th>Spread</th>
                <th>Pool Fee</th>
                <th>Gas USD</th>
                <th>Roundtrip Cost</th>
                <th>Trend</th>
                <th>Kalman Vel.</th>
                <th>RSI (14)</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="snapshots.length === 0">
                <td colspan="12" class="empty-state">No market snapshots captured yet. Click Force Scan to fetch.</td>
              </tr>
              <tr v-for="m in snapshots" :key="m.id">
                <td><span class="chain-badge" :class="`chain--${m.chain}`">{{ m.chain.toUpperCase() }}</span></td>
                <td>{{ m.dex }}</td>
                <td><strong>{{ m.pair_symbol }}</strong></td>
                <td>${{ m.price.toFixed(2) }}</td>
                <td>{{ m.spread_bps.toFixed(1) }} bps</td>
                <td>{{ m.fee_tier_bps.toFixed(1) }} bps</td>
                <td class="text-cyan">${{ m.gas_estimate_usd.toFixed(4) }}</td>
                <td class="text-yellow">{{ (m.roundtrip_fee_pct * 100).toFixed(3) }}%</td>
                <td>
                  <span class="trend-badge" :class="`trend--${m.trend_direction}`">
                    {{ m.trend_direction.toUpperCase() }} ({{ m.trend_strength.toFixed(0) }}%)
                  </span>
                </td>
                <td :class="m.kalman_velocity >= 0 ? 'good' : 'bad'">{{ m.kalman_velocity >= 0 ? '+' : '' }}{{ m.kalman_velocity.toFixed(4) }}</td>
                <td>{{ m.rsi_14.toFixed(1) }}</td>
                <td>{{ formatTime(m.created_at) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 6: ARBITRAGE RADAR -->
    <section v-if="currentTab === 'arbitrage'" class="tab-pane">
      <div class="panel">
        <header class="panel__head">
          <div class="filter-bar">
            <h3>Multi-Chain &amp; Spatial Arbitrage Edge Radar</h3>
            <div class="filter-controls">
              <label class="check-label">
                <input v-model="filterProfitableOnly" type="checkbox" @change="fetchArbitrage" />
                <span>Profitable After Fees Only</span>
              </label>
              <button class="btn btn--sm" @click="fetchArbitrage">REFRESH ARBITRAGE</button>
            </div>
          </div>
        </header>

        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Type</th>
                <th>Asset</th>
                <th>Buy Route</th>
                <th>Buy Price</th>
                <th>Sell Route</th>
                <th>Sell Price</th>
                <th>Gross Spread</th>
                <th>Fees + Gas</th>
                <th>Net Edge</th>
                <th>Profitable?</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="arbitrageOpps.length === 0">
                <td colspan="11" class="empty-state">No active arbitrage dislocations recorded.</td>
              </tr>
              <tr v-for="opp in arbitrageOpps" :key="opp.id" :class="{ 'row--highlight': opp.is_profitable_after_fees }">
                <td><span class="badge">{{ opp.opp_type }}</span></td>
                <td><strong>{{ opp.asset }}</strong></td>
                <td>
                  <span class="chain-badge" :class="`chain--${opp.buy_chain}`">{{ opp.buy_chain.toUpperCase() }}</span>
                  <small>{{ opp.buy_dex }}</small>
                </td>
                <td>${{ opp.buy_price.toFixed(2) }}</td>
                <td>
                  <span class="chain-badge" :class="`chain--${opp.sell_chain}`">{{ opp.sell_chain.toUpperCase() }}</span>
                  <small>{{ opp.sell_dex }}</small>
                </td>
                <td>${{ opp.sell_price.toFixed(2) }}</td>
                <td>{{ (opp.gross_spread_pct * 100).toFixed(3) }}%</td>
                <td class="muted">{{ ((opp.swap_fees_pct + (opp.gas_cost_usd / 0.25)) * 100).toFixed(3) }}%</td>
                <td :class="opp.net_edge_pct > 0 ? 'good' : 'bad'">
                  <strong>{{ (opp.net_edge_pct * 100).toFixed(3) }}%</strong>
                </td>
                <td>
                  <span class="badge" :class="opp.is_profitable_after_fees ? 'badge--live' : 'badge--ghost'">
                    {{ opp.is_profitable_after_fees ? 'PROFITABLE' : 'UNPROFITABLE' }}
                  </span>
                </td>
                <td>{{ formatTime(opp.created_at) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 7: GHOST TRADING & SELF-SCORING -->
    <section v-if="currentTab === 'ghost'" class="tab-pane">
      <div class="panel">
        <header class="panel__head">
          <div>
            <h3>Historical Research &amp; Ghost Trading Simulations</h3>
            <p class="panel__desc">
              Agents simulate buy-low and sell-high rules against historical OHLCV data, score their own results (0-100),
              and immediately qualify winning strategies for live trading.
            </p>
          </div>
          <button class="btn btn--primary" :disabled="scanning" @click="runGhostResearch">
            {{ scanning ? 'SIMULATING...' : '⚡ EXECUTE GHOST RESEARCH PASS' }}
          </button>
        </header>

        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Session</th>
                <th>Agent</th>
                <th>Strategy</th>
                <th>Chain / Symbol</th>
                <th>Buy Rule</th>
                <th>Sell Rule</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Net PnL</th>
                <th>P-Factor</th>
                <th>Self Score</th>
                <th>Live Graduated?</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="ghostSessions.length === 0">
                <td colspan="13" class="empty-state">No ghost trading sessions executed yet. Click button above to start.</td>
              </tr>
              <tr v-for="s in ghostSessions" :key="s.id" :class="{ 'row--graduated': s.graduated_to_live }">
                <td class="code-font">{{ s.session_id.slice(-8) }}</td>
                <td>{{ s.agent_name || s.agent_id }}</td>
                <td><strong>{{ s.strategy_name }}</strong></td>
                <td>{{ s.chain.toUpperCase() }} · {{ s.symbol }}</td>
                <td><small class="code-font">{{ s.buy_low_rule }}</small></td>
                <td><small class="code-font">{{ s.sell_high_rule }}</small></td>
                <td>{{ s.total_trades }} ({{ s.wins }}W / {{ s.losses }}L)</td>
                <td :class="s.win_rate >= 0.55 ? 'good' : 'bad'">{{ (s.win_rate * 100).toFixed(1) }}%</td>
                <td :class="s.net_pnl_usd >= 0 ? 'good' : 'bad'">${{ s.net_pnl_usd.toFixed(4) }}</td>
                <td>{{ s.profit_factor.toFixed(2) }}</td>
                <td>
                  <span class="score-badge" :class="s.self_score >= 70 ? 'score--pass' : 'score--fail'">
                    {{ s.self_score.toFixed(0) }}/100
                  </span>
                </td>
                <td>
                  <span class="badge" :class="s.graduated_to_live ? 'badge--live' : 'badge--ghost'">
                    {{ s.graduated_to_live ? 'LIVE GRADUATED' : 'GHOST ONLY' }}
                  </span>
                </td>
                <td>{{ formatTime(s.created_at) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 8: STRATEGY LINEAGE & LIVE STRATEGIES -->
    <section v-if="currentTab === 'lineage'" class="tab-pane">
      <div class="lineage-layout">
        <!-- Live Graduated Strategies Panel -->
        <div class="panel">
          <header class="panel__head">
            <h3>🚀 Active Live Graduated Strategies ({{ graduatedStrategies.length }})</h3>
            <span class="badge badge--live">Micro-Cent Dynamic Sizing ($0.05–$0.25+)</span>
          </header>
          <div class="grad-grid">
            <div v-if="graduatedStrategies.length === 0" class="empty-state">
              No strategies currently graduated. Once a ghost session achieves &ge; 70 self-score, it appears here automatically.
            </div>
            <div v-for="g in graduatedStrategies" :key="g.strategy_id" class="grad-card">
              <div class="grad-card__top">
                <span class="chain-badge" :class="`chain--${g.chain}`">{{ g.chain.toUpperCase() }}</span>
                <span class="badge" :class="g.live_status === 'active' ? 'badge--live' : 'badge--paused'">
                  {{ g.live_status.toUpperCase() }}
                </span>
              </div>
              <h4 class="grad-card__title">{{ g.name }}</h4>
              <p class="grad-card__symbol">{{ g.symbol }} · Derived by {{ g.agent_name || g.agent_id }}</p>
              <div class="grad-card__rules">
                <div><small>BUY LOW:</small> <code>{{ g.buy_low_logic }}</code></div>
                <div><small>SELL HIGH:</small> <code>{{ g.sell_high_logic }}</code></div>
              </div>
              <div class="grad-card__stats">
                <div>
                  <span class="stat-label">Ghost Win Rate</span>
                  <span class="stat-val good">{{ (g.ghost_win_rate * 100).toFixed(1) }}%</span>
                </div>
                <div>
                  <span class="stat-label">Live Trades</span>
                  <span class="stat-val">{{ g.live_trades_count }}</span>
                </div>
                <div>
                  <span class="stat-label">Live Net PnL</span>
                  <span class="stat-val" :class="g.live_net_pnl_usd >= 0 ? 'good' : 'bad'">${{ g.live_net_pnl_usd.toFixed(2) }}</span>
                </div>
                <div>
                  <span class="stat-label">Clip Sizing</span>
                  <span class="stat-val">${{ g.clip_usd.toFixed(2) }}</span>
                </div>
              </div>
              <footer class="grad-card__foot">
                <button
                  class="btn btn--sm"
                  :class="g.live_status === 'active' ? 'btn--secondary' : 'btn--primary'"
                  @click="toggleGraduatedStrategy(g.strategy_id)"
                >
                  {{ g.live_status === 'active' ? '⏸️ PAUSE STRATEGY' : '▶️ RESUME LIVE' }}
                </button>
              </footer>
            </div>
          </div>
        </div>

        <!-- Strategy Lineage Tree (Agent Root -> Child Strategies) -->
        <div class="panel">
          <header class="panel__head">
            <h3>🌳 Strategy Lineage &amp; Derivation Tree</h3>
            <span class="badge badge--accent">Agent Root Strategy &amp; Derived Children</span>
          </header>
          <p class="panel__desc">
            Every agent acts as its own root strategy. Child strategies are born from the agent's historical discovery,
            tagged with their execution mode (`PARENT_AGENT`, `GHOST`, or `LIVE`).
          </p>

          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Strategy ID</th>
                  <th>Parent Strategy</th>
                  <th>Name</th>
                  <th>Execution Mode</th>
                  <th>Derivation Archetype</th>
                  <th>Win Rate</th>
                  <th>Net PnL</th>
                  <th>Trades</th>
                  <th>Rules Summary</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="node in lineageNodes" :key="node.strategy_id">
                  <td class="code-font">{{ node.strategy_id }}</td>
                  <td class="code-font">{{ node.parent_strategy_id || 'ROOT (SELF)' }}</td>
                  <td><strong>{{ node.name }}</strong></td>
                  <td>
                    <span
                      class="badge"
                      :class="node.execution_mode === 'live' ? 'badge--live' : (node.execution_mode === 'ghost' ? 'badge--ghost' : 'badge--sys')"
                    >
                      {{ node.execution_mode.toUpperCase() }}
                    </span>
                  </td>
                  <td>{{ node.derivation_type }}</td>
                  <td class="good">{{ (node.win_rate * 100).toFixed(1) }}%</td>
                  <td :class="node.net_pnl_usd >= 0 ? 'good' : 'bad'">${{ node.net_pnl_usd.toFixed(2) }}</td>
                  <td>{{ node.trades_count }}</td>
                  <td><small class="muted">{{ node.rules_summary }}</small></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 9: AGENT TRADE LEDGER -->
    <section v-if="currentTab === 'trades'" class="tab-pane">
      <div class="panel">
        <header class="panel__head">
          <div class="filter-bar">
            <h3>Agent Trade Ledger (Database Tracked)</h3>
            <div class="filter-controls">
              <select v-model="filterTradeMode" class="filter-select" @change="fetchAgentTrades">
                <option value="">All Modes (Ghost + Live)</option>
                <option value="ghost">Ghost Only</option>
                <option value="live">Live Only</option>
              </select>
              <select v-model="filterTradeChain" class="filter-select" @change="fetchAgentTrades">
                <option value="">All Chains</option>
                <option value="base">Base</option>
                <option value="arbitrum">Arbitrum</option>
                <option value="ethereum">Ethereum</option>
                <option value="optimism">Optimism</option>
                <option value="polygon">Polygon</option>
              </select>
              <button class="btn btn--sm" @click="fetchAgentTrades">REFRESH TRADES</button>
            </div>
          </div>
        </header>

        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Trade ID</th>
                <th>Mode</th>
                <th>Agent</th>
                <th>Chain</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Notional ($)</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>Fees ($)</th>
                <th>Net PnL ($)</th>
                <th>Return %</th>
                <th>Status</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="agentTrades.length === 0">
                <td colspan="14" class="empty-state">No agent trades recorded yet.</td>
              </tr>
              <tr v-for="t in agentTrades" :key="t.id">
                <td class="code-font">#{{ t.id }}</td>
                <td>
                  <span class="badge" :class="t.trade_mode === 'live' ? 'badge--live' : 'badge--ghost'">
                    {{ t.trade_mode.toUpperCase() }}
                  </span>
                </td>
                <td>{{ t.agent_name || t.agent_id }}</td>
                <td><span class="chain-badge" :class="`chain--${t.chain}`">{{ t.chain.toUpperCase() }}</span></td>
                <td><strong>{{ t.symbol }}</strong></td>
                <td><span class="side-badge" :class="`side--${t.side}`">{{ t.side.toUpperCase() }}</span></td>
                <td>${{ t.notional_usd.toFixed(2) }}</td>
                <td>${{ t.entry_price.toFixed(2) }}</td>
                <td>${{ t.exit_price.toFixed(2) }}</td>
                <td class="muted">${{ t.fees_usd.toFixed(4) }}</td>
                <td :class="t.net_pnl_usd >= 0 ? 'good' : 'bad'">
                  <strong>{{ t.net_pnl_usd >= 0 ? '+' : '' }}${{ t.net_pnl_usd.toFixed(4) }}</strong>
                </td>
                <td :class="t.return_pct >= 0 ? 'good' : 'bad'">{{ t.return_pct >= 0 ? '+' : '' }}{{ t.return_pct.toFixed(2) }}%</td>
                <td>{{ t.status.toUpperCase() }}</td>
                <td>{{ formatTime(t.entry_time) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- Modal: Create New Agent -->
    <div v-if="showNewAgentModal" class="modal-backdrop" @click.self="showNewAgentModal = false">
      <div class="modal">
        <header class="modal__head">
          <h3>Create New Autonomous Agent</h3>
          <button class="modal__close" @click="showNewAgentModal = false">✕</button>
        </header>
        <form class="modal__body" @submit.prevent="createNewAgent">
          <div class="form-group">
            <label>Agent Name</label>
            <input v-model="newAgentForm.name" type="text" class="form-input" placeholder="e.g. Base Volatility Sniper" required />
          </div>
          <div class="form-group">
            <label>Role / Job Title</label>
            <input v-model="newAgentForm.role" type="text" class="form-input" placeholder="e.g. Micro-Volatility Scalper & Rapid Impulse Hunter" required />
          </div>
          <div class="form-group">
            <label>Site Domain / Area</label>
            <input v-model="newAgentForm.domain" type="text" class="form-input" placeholder="e.g. Base L2 DEX Trading" required />
          </div>
          <div class="form-group">
            <label>AGY Model Tier</label>
            <select v-model="newAgentForm.model_tier" class="form-select">
              <option value="gemini-2.5-flash">gemini-2.5-flash (Fast, Economical)</option>
              <option value="gemini-2.5-pro">gemini-2.5-pro (Deep Reasoning)</option>
              <option value="claude-sonnet-4-5">claude-sonnet-4-5</option>
            </select>
          </div>
          <div class="form-group">
            <label>System Directives &amp; Persona</label>
            <textarea
              v-model="newAgentForm.system_prompt"
              class="form-textarea"
              rows="4"
              placeholder="Provide directives, buy/sell rules, and goals for this agent..."
            ></textarea>
          </div>
          <footer class="modal__foot">
            <button type="button" class="btn" @click="showNewAgentModal = false">Cancel</button>
            <button type="submit" class="btn btn--primary" :disabled="creatingAgent">
              {{ creatingAgent ? 'CREATING...' : 'INITIALIZE AGENT' }}
            </button>
          </footer>
        </form>
      </div>
    </div>

    <!-- Modal: Track Alpha Wallet -->
    <div v-if="showAddAlphaModal" class="modal-backdrop" @click.self="showAddAlphaModal = false">
      <div class="modal">
        <header class="modal__head">
          <h3>Track New Alpha Trader Wallet</h3>
          <button class="modal__close" @click="showAddAlphaModal = false">✕</button>
        </header>
        <form class="modal__body" @submit.prevent="saveAlphaWallet">
          <div class="form-group">
            <label>Chain</label>
            <select v-model="alphaForm.chain" class="form-select" required>
              <option value="base">Base</option>
              <option value="arbitrum">Arbitrum</option>
              <option value="ethereum">Ethereum</option>
              <option value="optimism">Optimism</option>
              <option value="polygon">Polygon</option>
              <option value="solana">Solana</option>
            </select>
          </div>
          <div class="form-group">
            <label>Wallet Address</label>
            <input v-model="alphaForm.wallet_address" type="text" class="form-input" placeholder="0x... or Solana pubkey" required />
          </div>
          <div class="form-group">
            <label>Alias / Label</label>
            <input v-model="alphaForm.alias" type="text" class="form-input" placeholder="e.g. Base Whale Scalper" />
          </div>
          <div class="form-group">
            <label>Months Winning Track Record</label>
            <input v-model.number="alphaForm.months_winning" type="number" min="1" max="36" class="form-input" required />
          </div>
          <div class="form-group">
            <label>Historical Win Rate (0.0 - 1.0)</label>
            <input v-model.number="alphaForm.win_rate" type="number" step="0.01" min="0.5" max="1.0" class="form-input" required />
          </div>
          <div class="form-group">
            <label>Mirror Clip Sizing ($)</label>
            <input v-model.number="alphaForm.mirror_clip_usd" type="number" step="0.05" min="0.05" class="form-input" required />
          </div>
          <footer class="modal__foot">
            <button type="button" class="btn" @click="showAddAlphaModal = false">Cancel</button>
            <button type="submit" class="btn btn--primary">START TRACKING</button>
          </footer>
        </form>
      </div>
    </div>

    <!-- Modal: Add Settlement Wallet -->
    <div v-if="showAddWalletModal" class="modal-backdrop" @click.self="showAddWalletModal = false">
      <div class="modal">
        <header class="modal__head">
          <h3>Add / Update Settlement Wallet</h3>
          <button class="modal__close" @click="showAddWalletModal = false">✕</button>
        </header>
        <form class="modal__body" @submit.prevent="saveSettlementWallet">
          <div class="form-group">
            <label>Chain</label>
            <select v-model="walletForm.chain" class="form-select" required>
              <option value="base">Base</option>
              <option value="arbitrum">Arbitrum</option>
              <option value="ethereum">Ethereum</option>
              <option value="optimism">Optimism</option>
              <option value="polygon">Polygon</option>
              <option value="solana">Solana</option>
            </select>
          </div>
          <div class="form-group">
            <label>Settlement Stablecoin</label>
            <input v-model="walletForm.stablecoin_symbol" type="text" class="form-input" placeholder="USDC" required />
          </div>
          <div class="form-group">
            <label>Public Wallet Address</label>
            <input v-model="walletForm.wallet_address" type="text" class="form-input" placeholder="0x..." required />
          </div>
          <div class="form-group">
            <label>Profit Allocation (%)</label>
            <input v-model.number="walletForm.allocated_profit_pct" type="number" min="0" max="100" class="form-input" required />
          </div>
          <footer class="modal__foot">
            <button type="button" class="btn" @click="showAddWalletModal = false">Cancel</button>
            <button type="submit" class="btn btn--primary">SAVE WALLET</button>
          </footer>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue';

const API = '/api/agent-research';

function csrf(): string {
  const m = document.cookie.match(/(?:^|; )csrftoken=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : '';
}

async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const token = csrf();
  if (token && !headers.has('X-CSRFToken')) {
    headers.set('X-CSRFToken', token);
  }
  return fetch(url, {
    credentials: 'same-origin',
    ...init,
    headers,
  });
}

// State
const loading = ref(false);
const scanning = ref(false);
const scanningWallets = ref(false);
const sendingProfits = ref(false);
const notice = ref('');
const isNoticeError = ref(false);

const currentTab = ref('alpha');
const tabs = computed(() => [
  { id: 'alpha', title: 'Alpha Wallets & Live Mirror', icon: '🎯', count: alphaWallets.value.length },
  { id: 'agents', title: 'Multi-Agent Hub & Comms', icon: '🤖', count: agents.value.length },
  { id: 'markers', title: 'Profit Markers & Totals', icon: '📊', count: profitMarkers.value.length },
  { id: 'vault', title: 'Vault & Settlement', icon: '🔐' },
  { id: 'scanner', title: 'Market Scanner & Fees', icon: '📡', count: snapshots.value.length },
  { id: 'arbitrage', title: 'Arbitrage Radar', icon: '⚡', count: arbitrageOpps.value.length },
  { id: 'ghost', title: 'Ghost Research & Scoring', icon: '👻', count: ghostSessions.value.length },
  { id: 'lineage', title: 'Live & Strategy Lineage', icon: '🌳', count: graduatedStrategies.value.length },
  { id: 'trades', title: 'Agent Trade Ledger', icon: '📜', count: agentTrades.value.length },
]);

// Top level data
const cfg = ref<any>({ auto_graduate: true, clip_usd: 0.25 });
const counts = ref<any>({});
const agents = ref<any[]>([]);
const graduatedStrategies = ref<any[]>([]);
const ghostSessions = ref<any[]>([]);
const arbitrageOpps = ref<any[]>([]);
const snapshots = ref<any[]>([]);
const lineageNodes = ref<any[]>([]);
const agentTrades = ref<any[]>([]);
const settlementWallets = ref<any[]>([]);
const alphaWallets = ref<any[]>([]);
const tokenLimits = ref<Record<string, any>>({
  agy: { agent: 'agy', status: 'available', is_available: true, seconds_remaining: 0, total_rate_limits: 0 },
  codex: { agent: 'codex', status: 'available', is_available: true, seconds_remaining: 0, total_rate_limits: 0 },
  claude: { agent: 'claude', status: 'available', is_available: true, seconds_remaining: 0, total_rate_limits: 0 },
});
const profitMarkers = ref<any[]>([]);
const runningTotals = ref<any>({
  accumulated_pending_usd: 0.0,
  total_dispatched_usd: 0.0,
  manual_threshold_usd: 1000.0,
});

// Vault state
const vault = ref<any>({
  has_secret: false,
  masked_preview: '',
  profit_transfer_pct: 50.0,
  accumulated_pending_profits_usd: 0.0,
  total_dispatched_profits_usd: 0.0,
});
const selectedProfitPct = ref(50);
const revealedSecret = ref('');
const revealingSecret = ref(false);
const newVaultSecretInput = ref('');
const savingVault = ref(false);

// Chat state
const selectedAgent = ref<any>(null);
const chatMessages = ref<any[]>([]);
const userMessageInput = ref('');
const sendingMessage = ref(false);
const chatContainerRef = ref<HTMLDivElement | null>(null);

// Filter states
const filterScannerChain = ref('');
const filterProfitableOnly = ref(false);
const filterTradeMode = ref('');
const filterTradeChain = ref('');

// Modals
const showNewAgentModal = ref(false);
const creatingAgent = ref(false);
const newAgentForm = ref({
  name: '',
  role: '',
  domain: 'Market Research & Trading',
  model_tier: 'gemini-2.5-flash',
  system_prompt: '',
});

const showAddWalletModal = ref(false);
const walletForm = ref({
  chain: 'base',
  stablecoin_symbol: 'USDC',
  wallet_address: '',
  allocated_profit_pct: 100,
  is_active: true,
});

const showAddAlphaModal = ref(false);
const alphaForm = ref({
  chain: 'base',
  wallet_address: '',
  alias: '',
  months_winning: 4,
  win_rate: 0.78,
  mirror_clip_usd: 0.25,
});

let pollTimer: number | null = null;

// Helpers
function formatTime(iso: string) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return iso;
  }
}

function shortenAddress(addr: string) {
  if (!addr || addr.length < 16) return addr;
  return `${addr.slice(0, 8)}...${addr.slice(-6)}`;
}

function setNotice(msg: string, isError = false) {
  notice.value = msg;
  isNoticeError.value = isError;
  setTimeout(() => {
    if (notice.value === msg) notice.value = '';
  }, 6000);
}

// Fetchers
async function fetchStatus() {
  try {
    const r = await authFetch(`${API}/status/`);
    if (!r.ok) return;
    const data = await r.json();
    cfg.value = data.config || {};
    counts.value = data.counts || {};
    agents.value = data.agents || [];
    graduatedStrategies.value = data.graduated_strategies || [];
    ghostSessions.value = data.recent_ghost_sessions || [];
    arbitrageOpps.value = data.top_arbitrage_opportunities || [];
    vault.value = data.vault || {};
    selectedProfitPct.value = data.vault?.profit_transfer_pct ?? 50;
    settlementWallets.value = data.settlement_wallets || [];
    alphaWallets.value = data.alpha_wallets || [];
    if (data.running_totals) runningTotals.value = data.running_totals;
    if (data.token_limits) tokenLimits.value = data.token_limits;

    if (!selectedAgent.value && agents.value.length > 0) {
      selectAgent(agents.value[0]);
    }
  } catch (err: any) {
    console.error('Failed to fetch agent research status', err);
  }
}

async function fetchAlphaWallets() {
  try {
    const r = await authFetch(`${API}/alpha-wallets/`);
    if (!r.ok) return;
    const data = await r.json();
    alphaWallets.value = data.wallets || [];
  } catch (err) {
    console.error('Failed to fetch alpha wallets', err);
  }
}

async function fetchProfitMarkers() {
  try {
    const r = await authFetch(`${API}/profit-markers/`);
    if (!r.ok) return;
    const data = await r.json();
    profitMarkers.value = data.markers || [];
    if (data.running_totals) runningTotals.value = data.running_totals;
  } catch (err) {
    console.error('Failed to fetch profit markers', err);
  }
}

async function fetchMarketSnapshots() {
  try {
    const url = filterScannerChain.value
      ? `${API}/markets/?chain=${encodeURIComponent(filterScannerChain.value)}`
      : `${API}/markets/`;
    const r = await authFetch(url);
    if (!r.ok) return;
    const data = await r.json();
    snapshots.value = data.results || [];
  } catch (err) {
    console.error('Failed to fetch market snapshots', err);
  }
}

async function fetchArbitrage() {
  try {
    const url = filterProfitableOnly.value ? `${API}/arbitrage/?profitable_only=true` : `${API}/arbitrage/`;
    const r = await authFetch(url);
    if (!r.ok) return;
    const data = await r.json();
    arbitrageOpps.value = data.results || [];
  } catch (err) {
    console.error('Failed to fetch arbitrage opps', err);
  }
}

async function fetchGhostRuns() {
  try {
    const r = await authFetch(`${API}/ghost-runs/`);
    if (!r.ok) return;
    const data = await r.json();
    ghostSessions.value = data.results || [];
  } catch (err) {
    console.error('Failed to fetch ghost runs', err);
  }
}

async function fetchLineage() {
  try {
    const r = await authFetch(`${API}/lineage/`);
    if (!r.ok) return;
    const data = await r.json();
    lineageNodes.value = data.lineage || [];
  } catch (err) {
    console.error('Failed to fetch lineage', err);
  }
}

async function fetchAgentTrades() {
  try {
    const params = new URLSearchParams();
    if (filterTradeMode.value) params.append('mode', filterTradeMode.value);
    if (filterTradeChain.value) params.append('chain', filterTradeChain.value);
    const r = await authFetch(`${API}/trades/?${params.toString()}`);
    if (!r.ok) return;
    const data = await r.json();
    agentTrades.value = data.trades || [];
  } catch (err) {
    console.error('Failed to fetch agent trades', err);
  }
}

async function fetchSettlementWallets() {
  try {
    const r = await authFetch(`${API}/settlement-wallets/`);
    if (!r.ok) return;
    const data = await r.json();
    settlementWallets.value = data.wallets || [];
  } catch (err) {
    console.error('Failed to fetch settlement wallets', err);
  }
}

async function refreshAll() {
  loading.value = true;
  await Promise.all([
    fetchStatus(),
    fetchAlphaWallets(),
    fetchProfitMarkers(),
    fetchMarketSnapshots(),
    fetchArbitrage(),
    fetchGhostRuns(),
    fetchLineage(),
    fetchAgentTrades(),
    fetchSettlementWallets(),
  ]);
  loading.value = false;
}

// Alpha Wallet Actions
async function scanOnchainWallets() {
  scanningWallets.value = true;
  try {
    const r = await authFetch(`${API}/alpha-wallets/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'scan' }),
    });
    const data = await r.json();
    if (r.ok) {
      setNotice(`On-Chain Alpha Scan Complete: Found ${data.discovered_count} top-performing wallets!`);
      await fetchAlphaWallets();
    }
  } catch (err: any) {
    setNotice(`Alpha scan error: ${err.message}`, true);
  } finally {
    scanningWallets.value = false;
  }
}

async function confirmAlphaWallet(walletId: number) {
  try {
    const r = await authFetch(`${API}/alpha-wallets/${walletId}/confirm/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await r.json();
    if (r.ok) {
      if (data.graduated_straight_to_live) {
        setNotice(`🎉 Wallet ${data.alias} GRADUATED STRAIGHT TO LIVE MIRROR TRADING! Live Win Rate: ${data.live_confirm_win_rate}%`);
      } else {
        setNotice(`Confirmed live trade for ${data.alias}. Status: ${data.verification_status.toUpperCase()} (WinRate: ${data.live_confirm_win_rate}%)`);
      }
      await refreshAll();
    } else {
      setNotice(data.error || 'Confirmation failed', true);
    }
  } catch (err: any) {
    setNotice(`Confirm error: ${err.message}`, true);
  }
}

async function toggleAlphaMirror(walletId: number) {
  try {
    const r = await authFetch(`${API}/alpha-wallets/${walletId}/toggle/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await r.json();
    if (r.ok) {
      setNotice(`Mirroring for wallet #${walletId} is now ${data.is_mirror_active ? 'ACTIVE' : 'PAUSED'}.`);
      await fetchAlphaWallets();
    }
  } catch (err: any) {
    setNotice(`Toggle error: ${err.message}`, true);
  }
}

async function saveAlphaWallet() {
  if (!alphaForm.value.wallet_address.trim()) return;
  try {
    const r = await authFetch(`${API}/alpha-wallets/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alphaForm.value),
    });
    const data = await r.json();
    if (r.ok) {
      setNotice(`Alpha trader wallet ${data.wallet.alias} added to tracker.`);
      showAddAlphaModal.value = false;
      await fetchAlphaWallets();
    } else {
      setNotice(data.error || 'Failed to save wallet', true);
    }
  } catch (err: any) {
    setNotice(`Save error: ${err.message}`, true);
  }
}

// Manual Send All Profits Button Action
async function sendAllTotalProfits() {
  if (!confirm('Send all accumulated profits to your encrypted vault account and chain settlement wallets?')) return;
  sendingProfits.value = true;
  try {
    const r = await authFetch(`${API}/profit-markers/send-all/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await r.json();
    if (r.ok) {
      setNotice(`💸 SUCCESS! Dispatched $${data.total_sent_usd} across ${data.markers_dispatched_count} profit markers to your vault and settlement wallets.`);
      await refreshAll();
    } else {
      setNotice(data.error || 'Failed to send profits', true);
    }
  } catch (err: any) {
    setNotice(`Dispatch error: ${err.message}`, true);
  } finally {
    sendingProfits.value = false;
  }
}

// Agent Directory & Chat
async function selectAgent(ag: any) {
  selectedAgent.value = ag;
  chatMessages.value = [];
  try {
    const r = await authFetch(`${API}/agents/${encodeURIComponent(ag.agent_id)}/chat/`);
    if (!r.ok) return;
    const data = await r.json();
    chatMessages.value = data.messages || [];
    scrollToBottom();
  } catch (err) {
    console.error('Failed to fetch chat history', err);
  }
}

async function sendAgentMessage() {
  if (!selectedAgent.value || !userMessageInput.value.trim() || sendingMessage.value) return;
  const content = userMessageInput.value.trim();
  userMessageInput.value = '';
  sendingMessage.value = true;

  try {
    const r = await authFetch(`${API}/agents/${encodeURIComponent(selectedAgent.value.agent_id)}/chat/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: content }),
    });
    const data = await r.json();
    if (data.user_message) chatMessages.value.push(data.user_message);
    if (data.agent_message) chatMessages.value.push(data.agent_message);
    scrollToBottom();
  } catch (err: any) {
    setNotice(`Failed to send message: ${err.message}`, true);
  } finally {
    sendingMessage.value = false;
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (chatContainerRef.value) {
      chatContainerRef.value.scrollTop = chatContainerRef.value.scrollHeight;
    }
  });
}

async function createNewAgent() {
  if (!newAgentForm.value.name || !newAgentForm.value.role) return;
  creatingAgent.value = true;
  try {
    const r = await authFetch(`${API}/agents/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newAgentForm.value),
    });
    const data = await r.json();
    if (r.ok) {
      setNotice(`Agent ${data.agent.name} initialized successfully.`);
      showNewAgentModal.value = false;
      newAgentForm.value = {
        name: '',
        role: '',
        domain: 'Market Research & Trading',
        model_tier: 'gemini-2.5-flash',
        system_prompt: '',
      };
      await fetchStatus();
      if (data.agent) selectAgent(data.agent);
    } else {
      setNotice(data.error || 'Failed to create agent', true);
    }
  } catch (err: any) {
    setNotice(`Error creating agent: ${err.message}`, true);
  } finally {
    creatingAgent.value = false;
  }
}

// Vault Actions
async function toggleRevealSecret() {
  if (revealedSecret.value) {
    revealedSecret.value = '';
    return;
  }
  revealingSecret.value = true;
  try {
    const r = await authFetch(`${API}/vault/reveal/`, { method: 'POST' });
    const data = await r.json();
    if (r.ok && data.secret) {
      revealedSecret.value = data.secret;
    } else {
      setNotice(data.error || 'Failed to decrypt vault secret', true);
    }
  } catch (err: any) {
    setNotice(`Vault decryption error: ${err.message}`, true);
  } finally {
    revealingSecret.value = false;
  }
}

async function saveVaultSecret() {
  if (!newVaultSecretInput.value.trim()) return;
  savingVault.value = true;
  try {
    const r = await authFetch(`${API}/vault/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        secret: newVaultSecretInput.value.trim(),
        profit_transfer_pct: selectedProfitPct.value,
      }),
    });
    const data = await r.json();
    if (r.ok) {
      setNotice('Secret successfully encrypted with Kyber + AES-GCM and stored in Vault.');
      newVaultSecretInput.value = '';
      revealedSecret.value = '';
      await fetchStatus();
    } else {
      setNotice(data.error || 'Failed to encrypt secret', true);
    }
  } catch (err: any) {
    setNotice(`Vault error: ${err.message}`, true);
  } finally {
    savingVault.value = false;
  }
}

async function saveProfitPct() {
  try {
    const r = await authFetch(`${API}/vault/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profit_transfer_pct: selectedProfitPct.value }),
    });
    const data = await r.json();
    if (r.ok) {
      vault.value.profit_transfer_pct = data.profit_transfer_pct;
      setNotice(`Profit sweep percentage updated to ${data.profit_transfer_pct}%.`);
    }
  } catch (err: any) {
    setNotice(`Failed to update profit %: ${err.message}`, true);
  }
}

async function saveSettlementWallet() {
  try {
    const r = await authFetch(`${API}/settlement-wallets/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(walletForm.value),
    });
    const data = await r.json();
    if (r.ok) {
      setNotice(`Settlement wallet for ${walletForm.value.chain.toUpperCase()} saved.`);
      showAddWalletModal.value = false;
      await fetchSettlementWallets();
    } else {
      setNotice(data.error || 'Failed to save settlement wallet', true);
    }
  } catch (err: any) {
    setNotice(`Wallet error: ${err.message}`, true);
  }
}

async function deleteSettlementWallet(chain: string) {
  if (!confirm(`Remove settlement wallet for ${chain.toUpperCase()}?`)) return;
  try {
    const r = await authFetch(`${API}/settlement-wallets/${encodeURIComponent(chain)}/`, { method: 'DELETE' });
    if (r.ok) {
      setNotice(`Settlement wallet for ${chain.toUpperCase()} removed.`);
      await fetchSettlementWallets();
    }
  } catch (err: any) {
    setNotice(`Delete error: ${err.message}`, true);
  }
}

// Research Execution & Live Controls
async function runGhostResearch() {
  scanning.value = true;
  try {
    const r = await authFetch(`${API}/ghost-runs/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok) {
      const res = data.result || {};
      const chainInfo = res.chain ? ` [${res.chain.toUpperCase()}:${res.symbol}]` : '';
      setNotice(`Historical Ghost Research Pass Complete!${chainInfo} Self-Score: ${res.self_score ?? 'Done'}`);
      await refreshAll();
    } else {
      setNotice(data.error || data.detail || 'Research pass failed', true);
    }
  } catch (err: any) {
    setNotice(`Ghost execution error: ${err.message}`, true);
  } finally {
    scanning.value = false;
  }
}

async function scanMarketsNow() {
  try {
    const r = await authFetch(`${API}/control/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'scan_markets' }),
    });
    if (r.ok) {
      setNotice('Market scan iteration completed.');
      await fetchMarketSnapshots();
      await fetchArbitrage();
    }
  } catch (err: any) {
    setNotice(`Scan error: ${err.message}`, true);
  }
}

async function resetAgentTokenLimit(agentKey: string) {
  try {
    const r = await authFetch(`${API}/token-limits/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: agentKey, action: 'reset' }),
    });
    if (r.ok) {
      const d = await r.json();
      if (d.token_limits) tokenLimits.value = d.token_limits;
      setNotice(`Token cooldown cleared for ${agentKey.toUpperCase()}. Tokens replenished and ready.`);
    }
  } catch (err: any) {
    setNotice(`Failed to reset limit for ${agentKey}: ${err.message}`, true);
  }
}

async function toggleGraduatedStrategy(strategyId: string) {
  try {
    const r = await authFetch(`${API}/graduated/${encodeURIComponent(strategyId)}/toggle/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'toggle' }),
    });
    const data = await r.json();
    if (r.ok) {
      setNotice(`Strategy status toggled to: ${data.live_status.toUpperCase()}`);
      await fetchStatus();
    }
  } catch (err: any) {
    setNotice(`Toggle error: ${err.message}`, true);
  }
}

onMounted(() => {
  refreshAll();
  pollTimer = window.setInterval(refreshAll, 12000);
});

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer);
});
</script>

<style scoped>
.ar-root {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  padding: 1.5rem;
  color: #e5e7eb;
  background: #030712;
  min-height: 100vh;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header */
.ar-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1.5rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid #1f2937;
  flex-wrap: wrap;
}
.ar-head__eyebrow {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #06b6d4;
  font-weight: 700;
}
.ar-head__title {
  font-size: 1.75rem;
  font-weight: 800;
  margin: 0.25rem 0 0.5rem;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #f3f4f6 0%, #9ca3af 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.ar-head__desc {
  font-size: 0.85rem;
  color: #9ca3af;
  max-width: 820px;
  line-height: 1.45;
  margin: 0;
}
.ar-head__actions {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

/* Notice Toast */
.ar-notice {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #064e3b;
  color: #6ee7b7;
  padding: 0.65rem 1rem;
  border-radius: 6px;
  border: 1px solid #059669;
  font-size: 0.875rem;
}
.ar-notice--error {
  background: #7f1d1d;
  color: #fca5a5;
  border-color: #dc2626;
}
.ar-notice__close {
  background: none;
  border: none;
  color: inherit;
  font-size: 1.1rem;
  cursor: pointer;
}

/* Metrics Cards */
.ar-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.85rem;
}
.card {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.card--glow {
  border-color: #f59e0b;
  box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
}
.card__label {
  font-size: 0.72rem;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.card__value {
  font-size: 1.45rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.card__sub {
  font-size: 0.7rem;
  color: #6b7280;
}

/* Tabs */
.ar-tabs {
  display: flex;
  gap: 0.4rem;
  border-bottom: 1px solid #1f2937;
  overflow-x: auto;
  padding-bottom: 0.25rem;
}
.ar-tabs__btn {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  background: #0f172a;
  border: 1px solid #1e293b;
  color: #94a3b8;
  padding: 0.55rem 0.9rem;
  border-radius: 6px 6px 0 0;
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s ease;
}
.ar-tabs__btn:hover {
  background: #1e293b;
  color: #f8fafc;
}
.ar-tabs__btn.active {
  background: #1e293b;
  border-color: #06b6d4;
  border-bottom: 2px solid #06b6d4;
  color: #38bdf8;
}
.tab-count {
  background: #334155;
  color: #e2e8f0;
  font-size: 0.68rem;
  padding: 0.1rem 0.4rem;
  border-radius: 10px;
}

/* Panes */
.tab-pane {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}
.panel {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  padding: 1.25rem;
}
.panel__head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.65rem;
  flex-wrap: wrap;
  gap: 0.75rem;
}
.panel__head h3 {
  font-size: 1.1rem;
  font-weight: 700;
  margin: 0;
}
.panel__desc {
  font-size: 0.82rem;
  color: #9ca3af;
  margin: 0 0 1rem;
  line-height: 1.45;
}

/* Buttons & Badges */
.btn {
  background: #1f2937;
  border: 1px solid #374151;
  color: #f3f4f6;
  padding: 0.45rem 0.85rem;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  text-decoration: none;
  transition: all 0.15s ease;
}
.btn:hover:not(:disabled) {
  background: #374151;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn--primary {
  background: #0284c7;
  border-color: #0369a1;
  color: #fff;
}
.btn--primary:hover:not(:disabled) {
  background: #0369a1;
}
.btn--secondary {
  background: #334155;
  border-color: #475569;
}
.btn--accent {
  background: #059669;
  border-color: #047857;
  color: #fff;
}
.btn--danger {
  background: #991b1b;
  border-color: #b91c1c;
  color: #fecaca;
}
.btn--danger:hover:not(:disabled) {
  background: #b91c1c;
}
.btn--sm {
  padding: 0.25rem 0.55rem;
  font-size: 0.72rem;
}
.btn-group {
  display: flex;
  gap: 0.35rem;
}

.badge {
  font-size: 0.7rem;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  background: #1f2937;
  color: #9ca3af;
}
.badge--live {
  background: #064e3b;
  color: #34d399;
  border: 1px solid #059669;
}
.badge--ghost {
  background: #312e81;
  color: #a5b4fc;
  border: 1px solid #4338ca;
}
.badge--sys {
  background: #1e1b4b;
  color: #c7d2fe;
}
.badge--user {
  background: #365314;
  color: #bef264;
}
.badge--paused {
  background: #451a03;
  color: #fcd34d;
}
.badge--accent {
  background: #164e63;
  color: #67e8f9;
}

.chain-badge {
  font-size: 0.68rem;
  font-weight: 700;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
}
.chain--base { background: #1e3a8a; color: #93c5fd; }
.chain--arbitrum { background: #1e293b; color: #38bdf8; }
.chain--ethereum { background: #3730a3; color: #c7d2fe; }
.chain--optimism { background: #881337; color: #fecdd3; }
.chain--polygon { background: #581c87; color: #e9d5ff; }
.chain--solana { background: #134e4a; color: #5eead4; }

.side-badge { font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px; }
.side--buy { background: #064e3b; color: #6ee7b7; }
.side--sell { background: #7f1d1d; color: #fca5a5; }

.trend-badge { font-size: 0.68rem; font-weight: 600; padding: 0.15rem 0.4rem; border-radius: 4px; }
.trend--bullish { background: #064e3b; color: #34d399; }
.trend--bearish { background: #7f1d1d; color: #f87171; }
.trend--neutral { background: #374151; color: #d1d5db; }

.score-badge { font-size: 0.75rem; font-weight: 800; padding: 0.2rem 0.45rem; border-radius: 4px; }
.score--pass { background: #064e3b; color: #34d399; border: 1px solid #10b981; }
.score--fail { background: #450a0a; color: #f87171; }

.token-tags {
  display: flex;
  gap: 0.25rem;
  flex-wrap: wrap;
}
.token-pill {
  font-size: 0.65rem;
  background: #1f2937;
  color: #38bdf8;
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
  font-weight: 700;
}

/* Colors */
.text-cyan { color: #38bdf8; }
.text-green { color: #34d399; }
.text-yellow { color: #fbbf24; }
.text-amber { color: #f59e0b; }
.text-purple { color: #c084fc; }
.text-emerald { color: #10b981; }
.good { color: #34d399; }
.bad { color: #f87171; }
.muted { color: #9ca3af; }
.code-font { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }

/* Tables */
.table-wrap, .wallet-table-wrap {
  overflow-x: auto;
  border: 1px solid #1f2937;
  border-radius: 6px;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
  text-align: left;
}
.data-table th {
  background: #0f172a;
  color: #94a3b8;
  padding: 0.65rem 0.75rem;
  border-bottom: 1px solid #1f2937;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  white-space: nowrap;
}
.data-table td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid #1f2937;
  white-space: nowrap;
}
.data-table tr:hover {
  background: #1e293b;
}
.row--highlight {
  background: rgba(16, 185, 129, 0.07);
}
.row--graduated {
  background: rgba(139, 92, 246, 0.07);
}
.empty-state {
  text-align: center;
  color: #6b7280;
  padding: 2rem !important;
  font-style: italic;
}

/* Marker Totals Grid */
.marker-totals-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.85rem;
  margin-bottom: 1.25rem;
}
.total-box {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.total-box__label {
  font-size: 0.72rem;
  font-weight: 700;
  color: #94a3b8;
}
.total-box__val {
  font-size: 1.5rem;
  font-weight: 800;
}

/* Agent Layout */
.agent-layout {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 1.25rem;
  height: 640px;
}
.agent-sidebar {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
}
.agent-sidebar__head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #1f2937;
}
.agent-sidebar__head h3 {
  font-size: 0.95rem;
  margin: 0;
}
.agent-list {
  overflow-y: auto;
  padding: 0.6rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.agent-card {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 0.75rem;
  cursor: pointer;
  transition: all 0.15s ease;
}
.agent-card:hover {
  border-color: #38bdf8;
  background: #1e293b;
}
.agent-card.selected {
  border-color: #06b6d4;
  background: #1e293b;
  box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
}
.agent-card__top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.35rem;
}
.agent-card__name {
  font-size: 0.88rem;
  font-weight: 700;
  margin: 0 0 0.25rem;
}
.agent-card__role {
  font-size: 0.75rem;
  color: #94a3b8;
  margin: 0 0 0.4rem;
  line-height: 1.3;
}
.agent-card__meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.68rem;
  color: #64748b;
}

.agent-console {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  height: 100%;
}
.console-box {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.console-box__head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid #1f2937;
}
.console-box__title {
  font-size: 1.15rem;
  margin: 0 0 0.2rem;
}
.console-box__sub {
  color: #9ca3af;
  font-size: 0.75rem;
}
.console-box__stats {
  display: flex;
  gap: 0.5rem;
}
.chip {
  font-size: 0.72rem;
  background: #1f2937;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
}
.chip--good { color: #34d399; }
.chip--bad { color: #f87171; }

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}
.chat-bubble {
  max-width: 80%;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  line-height: 1.45;
  font-size: 0.85rem;
}
.bubble--user {
  align-self: flex-end;
  background: #0369a1;
  color: #f0f9ff;
  border-bottom-right-radius: 2px;
}
.bubble--agent {
  align-self: flex-start;
  background: #1f2937;
  border: 1px solid #374151;
  color: #f3f4f6;
  border-bottom-left-radius: 2px;
}
.bubble-meta {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: 0.68rem;
  color: #94a3b8;
  margin-bottom: 0.3rem;
  text-transform: uppercase;
}
.bubble-sender { font-weight: 700; }
.bubble-text { white-space: pre-wrap; word-break: break-word; }

.chat-form {
  display: flex;
  gap: 0.6rem;
  padding: 0.85rem 1rem;
  border-top: 1px solid #1f2937;
  background: #0f172a;
}
.chat-input {
  flex: 1;
  background: #1e293b;
  border: 1px solid #334155;
  color: #f8fafc;
  padding: 0.55rem 0.85rem;
  border-radius: 6px;
  font-size: 0.85rem;
}
.chat-input:focus {
  outline: none;
  border-color: #38bdf8;
}

/* Vault Grid */
.vault-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.25rem;
}
.vault-box {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}
.vault-field label {
  display: block;
  font-size: 0.8rem;
  font-weight: 600;
  color: #cbd5e1;
  margin-bottom: 0.4rem;
}
.vault-input-group, .vault-select-group {
  display: flex;
  gap: 0.6rem;
  align-items: center;
}
.vault-display, .vault-input, .vault-select {
  flex: 1;
  background: #030712;
  border: 1px solid #374151;
  color: #f3f4f6;
  padding: 0.55rem 0.85rem;
  border-radius: 6px;
  font-size: 0.85rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.vault-select {
  font-family: inherit;
  max-width: 320px;
}

/* Graduated Strategy Cards */
.grad-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1rem;
}
.grad-card {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}
.grad-card__top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.grad-card__title {
  font-size: 1rem;
  font-weight: 700;
  margin: 0;
}
.grad-card__symbol {
  font-size: 0.75rem;
  color: #94a3b8;
  margin: 0;
}
.grad-card__rules {
  background: #030712;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 0.6rem;
  font-size: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.grad-card__rules code {
  color: #38bdf8;
  font-family: ui-monospace, monospace;
}
.grad-card__stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  padding: 0.5rem 0;
  border-top: 1px solid #1e293b;
  border-bottom: 1px solid #1e293b;
}
.stat-label {
  display: block;
  font-size: 0.68rem;
  color: #64748b;
  text-transform: uppercase;
}
.stat-val {
  font-size: 0.95rem;
  font-weight: 700;
}
.grad-card__foot {
  display: flex;
  justify-content: flex-end;
}

/* Modals */
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(4px);
}
.modal {
  background: #111827;
  border: 1px solid #374151;
  border-radius: 10px;
  width: 90%;
  max-width: 520px;
  overflow: hidden;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
}
.modal__head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid #1f2937;
}
.modal__head h3 {
  font-size: 1.1rem;
  margin: 0;
}
.modal__close {
  background: none;
  border: none;
  color: #9ca3af;
  font-size: 1.2rem;
  cursor: pointer;
}
.modal__body {
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}
.form-group label {
  display: block;
  font-size: 0.75rem;
  font-weight: 600;
  color: #cbd5e1;
  margin-bottom: 0.3rem;
  text-transform: uppercase;
}
.form-input, .form-select, .form-textarea {
  width: 100%;
  background: #030712;
  border: 1px solid #374151;
  color: #f3f4f6;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
  box-sizing: border-box;
}
.modal__foot {
  display: flex;
  justify-content: flex-end;
  gap: 0.6rem;
  padding-top: 0.5rem;
}

/* Filter Bar */
.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  flex-wrap: wrap;
  gap: 0.75rem;
}
.filter-controls {
  display: flex;
  gap: 0.6rem;
  align-items: center;
}
.filter-select {
  background: #030712;
  border: 1px solid #374151;
  color: #f3f4f6;
  padding: 0.35rem 0.65rem;
  border-radius: 6px;
  font-size: 0.75rem;
}
.check-label {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.75rem;
  color: #cbd5e1;
  cursor: pointer;
}

/* Token Limit & Auto-Resume Monitor */
.ar-token-bar {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  padding: 0.85rem 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.ar-token-bar__header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.ar-token-bar__badge {
  font-size: 0.7rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: #1e1b4b;
  color: #a5b4fc;
  border: 1px solid #4338ca;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
}
.ar-token-bar__desc {
  font-size: 0.75rem;
  color: #94a3b8;
}
.ar-token-bar__pills {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.ar-token-pill {
  flex: 1;
  min-width: 200px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 0.6rem 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  transition: all 0.2s ease;
}
.ar-token-pill--ready {
  border-color: rgba(16, 185, 129, 0.3);
}
.ar-token-pill--cooling {
  background: #271d10;
  border-color: #f59e0b;
  box-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
}
.ar-token-pill__title {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}
.pill-name {
  font-size: 0.85rem;
  color: #f8fafc;
}
.pill-tag {
  font-size: 0.68rem;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  font-weight: 600;
  margin-left: auto;
}
.tag--green {
  background: rgba(16, 185, 129, 0.15);
  color: #34d399;
}
.tag--amber {
  background: rgba(245, 158, 11, 0.2);
  color: #fbbf24;
}
.ar-token-pill__meta {
  font-size: 0.7rem;
  color: #94a3b8;
  display: flex;
  gap: 0.35rem;
  align-items: center;
}
.dot--green {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 6px #10b981;
}
.dot--amber-pulse {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #f59e0b;
  box-shadow: 0 0 8px #f59e0b;
  animation: pulse-amber 1.5s infinite;
}
@keyframes pulse-amber {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.85); }
}
.btn--xs {
  padding: 0.2rem 0.45rem;
  font-size: 0.68rem;
  align-self: flex-start;
  margin-top: 0.2rem;
}
.btn--outline {
  background: transparent;
  border: 1px solid #f59e0b;
  color: #f59e0b;
}
.btn--outline:hover {
  background: #f59e0b;
  color: #030712;
}
</style>
