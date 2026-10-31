"""Unit and integration tests for Agent Market Research, Arbitrage,
Ghost Self-Scoring & Live Graduation, and Multi-Agent Communication.
"""
from __future__ import annotations

import json
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

from .engine import (
    RESEARCH_COORDINATOR,
    AgentMessenger,
    ArbitrageHunter,
    GhostResearchSimulator,
    GraduationCoordinator,
    MarketDataScanner,
)
from .models import (
    AgentMessage,
    AgentProfile,
    ArbitrageOpportunity,
    GhostTrade,
    GhostTradingSession,
    LiveGraduatedStrategy,
    MarketResearchConfig,
    MarketSnapshot,
)


class AgentResearchCoreEngineTests(TestCase):
    """Test engine logic: fees, arbitrage, ghost simulation, self-scoring, and graduation."""

    def setUp(self):
        self.cfg = MarketResearchConfig.load()
        RESEARCH_COORDINATOR.ensure_default_agents()

    def test_market_scanner_fee_calculation(self):
        """Verify accurate fee accounting across different chains and DEX tiers."""
        fees_base = MarketDataScanner.calculate_fees(chain="base", dex="aerodrome", notional_usd=0.25, fee_tier_bps=2.0)
        self.assertGreater(fees_base["gas_usd"], 0.0)
        self.assertGreater(fees_base["total_roundtrip_usd"], 0.0)
        self.assertGreater(fees_base["effective_roundtrip_pct"], 0.0)

        # Ethereum L1 should have significantly higher gas cost than Base L2
        fees_eth = MarketDataScanner.calculate_fees(chain="ethereum", dex="uniswap_v3", notional_usd=0.25, fee_tier_bps=30.0)
        self.assertGreater(fees_eth["gas_usd"], fees_base["gas_usd"])

    def test_trend_evaluation(self):
        """Verify multi-factor trend and momentum metrics."""
        # Upward trending closes
        closes_up = [100.0 + (i * 0.5) for i in range(30)]
        res_up = MarketDataScanner.evaluate_trend(closes_up)
        self.assertIn(res_up["direction"], ["bullish", "neutral"])
        self.assertGreater(res_up["kalman_velocity"], 0.0)
        self.assertGreaterEqual(res_up["rsi_14"], 45.0)

        # Downward trending closes
        closes_down = [100.0 - (i * 0.5) for i in range(30)]
        res_down = MarketDataScanner.evaluate_trend(closes_down)
        self.assertIn(res_down["direction"], ["bearish", "neutral"])
        self.assertLess(res_down["kalman_velocity"], 0.0)

    def test_arbitrage_detection(self):
        """Verify cross-chain and cross-DEX spread discovery with net edge calculation."""
        markets = [
            {
                "chain": "base",
                "dex": "aerodrome",
                "pair_symbol": "WETH-USDC",
                "price": 2840.0,
                "fee_tier_bps": 2.0,
                "liquidity_depth_usd": 500000.0,
            },
            {
                "chain": "arbitrum",
                "dex": "camelot",
                "pair_symbol": "WETH-USDC",
                "price": 2855.0,  # ~0.52% spread
                "fee_tier_bps": 5.0,
                "liquidity_depth_usd": 600000.0,
            },
        ]
        opps = ArbitrageHunter.detect_arbitrage(markets, notional_usd=0.25)
        self.assertGreater(len(opps), 0)
        top = opps[0]
        self.assertEqual(top["asset"], "WETH")
        self.assertGreater(top["gross_spread_pct"], 0.0)
        self.assertIn("net_edge_pct", top)
        self.assertIn("is_profitable_after_fees", top)

    def test_ghost_trading_simulation(self):
        """Verify historical ghost trading simulation executes buy-low/sell-high trades."""
        bars = GhostResearchSimulator.load_historical_bars("base", "WETH-USDC", limit=150)
        self.assertGreaterEqual(len(bars), 50)

        trades, stats = GhostResearchSimulator.simulate_buy_low_sell_high(
            chain="base",
            symbol="WETH-USDC",
            bars=bars,
            notional_usd=0.25,
            tp_fee_mult=2.2,
            sl_fee_mult=1.5,
        )
        self.assertIn("total_trades", stats)
        self.assertIn("win_rate", stats)
        self.assertIn("profit_factor", stats)
        self.assertIn("net_pnl_usd", stats)
        self.assertIn("holdout_t_stat", stats)

    def test_self_scoring_and_immediate_graduation(self):
        """Verify agent self-scoring and immediate graduation to live trading."""
        # Simulated high-performing stats that meet all graduation gates
        stats = {
            "total_trades": 18,
            "wins": 12,
            "losses": 6,
            "win_rate": 0.6667,
            "gross_pnl_usd": 0.12,
            "total_fees_usd": 0.02,
            "net_pnl_usd": 0.10,
            "profit_factor": 2.15,
            "sharpe_ratio": 2.4,
            "max_drawdown_pct": 0.06,
            "holdout_t_stat": 2.85,
        }
        score, assessment = GraduationCoordinator.calculate_self_score(stats)
        self.assertGreaterEqual(score, 70.0)
        self.assertIn("qualified for immediate live graduation", assessment.lower())

        grad_info = GraduationCoordinator.graduate_to_live(
            session_id="test_sess_1234",
            agent_id="agy_market_trader",
            strategy_name="Dip Rebound WETH",
            chain="base",
            symbol="WETH-USDC",
            buy_low_rule="RSI < 35 & Lower Band",
            sell_high_rule="TP at 2.5x Fee",
            stats=stats,
            clip_usd=0.25,
        )
        self.assertIn("strategy_id", grad_info)
        self.assertIn("discovered_rules.json", grad_info["targets_wired"])
        self.assertIn("money_button_edge_overrides.json", grad_info["targets_wired"])

    def test_agent_messenger(self):
        """Verify agent two-way messaging dispatch with role-aware replies."""
        agent = AgentProfile.objects.filter(agent_id="agy_market_trader").first()
        self.assertIsNotNone(agent)
        reply = AgentMessenger.send_to_agent(
            agent_profile=agent,
            user_message="Scan Base and Arbitrum for the highest net arbitrage opportunities.",
        )
        self.assertIsInstance(reply, str)
        self.assertGreater(len(reply), 10)


class AgentResearchAPITests(TestCase):
    """Test REST API views and control endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.cfg = MarketResearchConfig.load()
        RESEARCH_COORDINATOR.ensure_default_agents()

    def test_status_endpoint(self):
        """GET /api/agent-research/status/"""
        url = reverse("agentresearch-status")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertIn("config", data)
        self.assertIn("counts", data)
        self.assertIn("agents", data)
        self.assertGreaterEqual(data["counts"]["total_agents"], 5)

    def test_markets_endpoint(self):
        """GET /api/agent-research/markets/"""
        # Seed a snapshot
        MarketSnapshot.objects.create(
            chain="base",
            dex="aerodrome",
            pair_symbol="WETH-USDC",
            token_in="WETH",
            token_out="USDC",
            price=2850.0,
            bid_price=2849.5,
            ask_price=2850.5,
            spread_bps=3.5,
            fee_tier_bps=2.0,
            gas_estimate_usd=0.0015,
            roundtrip_fee_pct=0.0008,
            liquidity_depth_usd=250000.0,
            volume_24h_usd=1200000.0,
            volatility_atr14=0.015,
            trend_direction="bullish",
            trend_strength=68.5,
            kalman_velocity=0.004,
            ema_alignment=0.002,
            rsi_14=58.2,
        )
        url = reverse("agentresearch-markets")
        resp = self.client.get(url, {"chain": "base"})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertGreaterEqual(data["total_snapshots"], 1)
        self.assertEqual(data["results"][0]["pair_symbol"], "WETH-USDC")

    def test_arbitrage_endpoint(self):
        """GET /api/agent-research/arbitrage/"""
        ArbitrageOpportunity.objects.create(
            opp_type=ArbitrageOpportunity.OppType.CROSS_DEX,
            asset="WETH",
            buy_chain="base",
            buy_dex="aerodrome",
            buy_price=2845.0,
            sell_chain="base",
            sell_dex="uniswap_v3",
            sell_price=2855.0,
            gross_spread_pct=0.0035,
            swap_fees_pct=0.0007,
            gas_cost_usd=0.003,
            slippage_est_pct=0.0002,
            net_edge_pct=0.0026,
            is_profitable_after_fees=True,
            status=ArbitrageOpportunity.Status.ACTIVE,
        )
        url = reverse("agentresearch-arbitrage")
        resp = self.client.get(url, {"profitable_only": "true"})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertGreaterEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["asset"], "WETH")

    def test_ghost_runs_and_detail_endpoint(self):
        """POST & GET /api/agent-research/ghost-runs/"""
        url = reverse("agentresearch-ghost-runs")
        # Trigger run
        post_resp = self.client.post(url)
        self.assertEqual(post_resp.status_code, status.HTTP_201_CREATED)
        post_data = post_resp.json()
        self.assertEqual(post_data["status"], "success")

        # List runs
        get_resp = self.client.get(url)
        self.assertEqual(get_resp.status_code, status.HTTP_200_OK)
        get_data = get_resp.json()
        self.assertGreaterEqual(get_data["count"], 1)

        # Detail run
        session_id = get_data["results"][0]["id"]
        detail_url = reverse("agentresearch-ghost-detail", kwargs={"pk": session_id})
        det_resp = self.client.get(detail_url)
        self.assertEqual(det_resp.status_code, status.HTTP_200_OK)
        det_data = det_resp.json()
        self.assertIn("session", det_data)
        self.assertIn("trades", det_data)

    def test_graduated_strategies_endpoint_and_toggle(self):
        """GET /api/agent-research/graduated/ and toggle pause/active."""
        agent = AgentProfile.objects.filter(agent_id="agy_market_trader").first()
        strat = LiveGraduatedStrategy.objects.create(
            strategy_id="grad_test_001",
            agent=agent,
            name="Dip Rebound Live",
            strategy_type="buy_low_sell_high",
            chain="base",
            symbol="WETH-USDC",
            buy_low_logic="RSI < 35",
            sell_high_logic="TP 2.5x Fee",
            ghost_win_rate=0.65,
            ghost_profit_factor=2.1,
            ghost_net_pnl_usd=0.08,
            ghost_t_stat=2.4,
            live_status=LiveGraduatedStrategy.Status.ACTIVE,
            clip_usd=0.25,
        )

        url = reverse("agentresearch-graduated")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(resp.json()["count"], 1)

        toggle_url = reverse("agentresearch-graduated-toggle", kwargs={"strategy_id": "grad_test_001"})
        t_resp = self.client.post(toggle_url, {"action": "toggle"}, format="json")
        self.assertEqual(t_resp.status_code, status.HTTP_200_OK)
        self.assertEqual(t_resp.json()["live_status"], LiveGraduatedStrategy.Status.PAUSED)

    def test_agent_directory_and_custom_agent_creation(self):
        """POST /api/agent-research/agents/ creates a new agent with a custom role."""
        url = reverse("agentresearch-agents")
        create_payload = {
            "name": "Meme Momentum Scout",
            "role": "High-Velocity Token Breakout & Dip Specialist",
            "domain": "Micro-Cap Dex Screener",
            "model_tier": "gemini-2.5-flash",
            "system_prompt": "Scan new pairs for rapid volume surges and sudden liquidity injections.",
        }
        resp = self.client.post(url, create_payload, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        data = resp.json()
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["agent"]["name"], "Meme Momentum Scout")
        self.assertEqual(data["agent"]["system_role_tag"], "custom")

    def test_agent_two_way_communication(self):
        """POST & GET /api/agent-research/agents/<agent_id>/chat/"""
        agent = AgentProfile.objects.filter(agent_id="agy_market_trader").first()
        chat_url = reverse("agentresearch-agent-chat", kwargs={"agent_id": agent.agent_id})

        msg_payload = {"message": "What is our current live trading status and top arbitrage target?"}
        post_resp = self.client.post(chat_url, msg_payload, format="json")
        self.assertEqual(post_resp.status_code, status.HTTP_201_CREATED)
        data = post_resp.json()
        self.assertEqual(data["user_message"]["sender"], "user")
        self.assertEqual(data["agent_message"]["sender"], "agent")
        self.assertGreater(len(data["agent_message"]["content"]), 10)

        # Retrieve transcript
        get_resp = self.client.get(chat_url)
        self.assertEqual(get_resp.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(get_resp.json()["messages"]), 2)

    def test_control_and_export_endpoints(self):
        """POST /api/agent-research/control/ and GET /api/agent-research/export/"""
        ctrl_url = reverse("agentresearch-control")
        scan_resp = self.client.post(ctrl_url, {"action": "scan_markets"}, format="json")
        self.assertEqual(scan_resp.status_code, status.HTTP_200_OK)

        export_url = reverse("agentresearch-export")
        csv_resp = self.client.get(export_url, {"format": "csv"})
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn("text/csv", csv_resp["Content-Type"])

        excel_resp = self.client.get(export_url, {"format": "excel"})
        self.assertEqual(excel_resp.status_code, 200)
        self.assertIn("application/vnd.ms-excel", excel_resp["Content-Type"])

    def test_vault_encryption_storage_and_reveal(self):
        """Verify post-quantum Kyber/AES-GCM encryption of profit sweep secret and reveal."""
        vault_url = reverse("agentresearch-vault")
        reveal_url = reverse("agentresearch-vault-reveal")

        # 1. Initial vault state
        resp = self.client.get(vault_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        # 2. Store secret in vault with 75% profit transfer
        secret_text = "0x9876543210fedcba9876543210fedcba98765432"
        post_resp = self.client.post(vault_url, {"secret": secret_text, "profit_transfer_pct": 75.0}, format="json")
        self.assertEqual(post_resp.status_code, status.HTTP_200_OK)
        self.assertTrue(post_resp.json()["has_secret"])
        self.assertEqual(post_resp.json()["profit_transfer_pct"], 75.0)

        # 3. Decrypt and reveal secret
        reveal_resp = self.client.post(reveal_url, format="json")
        self.assertEqual(reveal_resp.status_code, status.HTTP_200_OK)
        self.assertEqual(reveal_resp.json()["secret"], secret_text)

        # 4. Update percentage alone
        pct_resp = self.client.post(vault_url, {"profit_transfer_pct": 40.0}, format="json")
        self.assertEqual(pct_resp.status_code, status.HTTP_200_OK)
        self.assertEqual(pct_resp.json()["profit_transfer_pct"], 40.0)

    def test_settlement_wallets_endpoints(self):
        """Verify listing, updating, and deleting per-chain settlement wallets."""
        wallets_url = reverse("agentresearch-settlement-wallets")

        # GET ensures defaults
        resp = self.client.get(wallets_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(resp.json()["count"], 5)

        # POST update for Base
        post_resp = self.client.post(
            wallets_url,
            {
                "chain": "base",
                "stablecoin_symbol": "USDC",
                "wallet_address": "0x1111222233334444555566667777888899990000",
                "allocated_profit_pct": 100.0,
                "is_active": True,
            },
            format="json",
        )
        self.assertEqual(post_resp.status_code, status.HTTP_200_OK)
        self.assertEqual(post_resp.json()["wallet"]["wallet_address"], "0x1111222233334444555566667777888899990000")

        # DELETE for Optimism
        del_url = reverse("agentresearch-settlement-wallet-delete", kwargs={"chain": "optimism"})
        del_resp = self.client.delete(del_url)
        self.assertEqual(del_resp.status_code, status.HTTP_200_OK)

    def test_strategy_lineage_endpoint(self):
        """Verify strategy lineage nodes: agent parent archetype and derived child strategies."""
        lineage_url = reverse("agentresearch-lineage")
        resp = self.client.get(lineage_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertGreaterEqual(data["count"], 1)
        # Check that parent agent node exists
        root_nodes = [n for n in data["lineage"] if n["execution_mode"] == "parent_agent"]
        self.assertGreaterEqual(len(root_nodes), 1)

    def test_agent_trades_endpoints(self):
        """Verify recording and filtering agent trades with dynamic sizing."""
        trades_url = reverse("agentresearch-trades")

        # Record a ghost micro-cent trade ($0.25)
        post_resp = self.client.post(
            trades_url,
            {
                "agent_id": "agy_market_trader",
                "trade_mode": "ghost",
                "chain": "base",
                "symbol": "WETH-USDC",
                "side": "buy",
                "notional_usd": 0.25,
                "entry_price": 2840.0,
                "exit_price": 2850.0,
                "fees_usd": 0.001,
            },
            format="json",
        )
        self.assertEqual(post_resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(post_resp.json()["trade"]["trade_mode"], "ghost")

        # GET trade list
        get_resp = self.client.get(trades_url, {"mode": "ghost", "chain": "base"})
        self.assertEqual(get_resp.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(get_resp.json()["count"], 1)

    def test_alpha_wallet_discovery_and_live_confirmation(self):
        """Verify on-chain alpha wallet discovery and immediate graduation straight to live mirror trading."""
        wallets_url = reverse("agentresearch-alpha-wallets")

        # GET seeded wallets
        resp = self.client.get(wallets_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(resp.json()["count"], 3)

        wallet_id = resp.json()["wallets"][0]["id"]
        confirm_url = reverse("agentresearch-alpha-wallet-confirm", kwargs={"pk": wallet_id})

        # Process live confirmations
        for _ in range(4):
            conf_resp = self.client.post(confirm_url, format="json")
            self.assertEqual(conf_resp.status_code, status.HTTP_200_OK)

        # Verify live graduation straight to live
        w_resp = self.client.get(wallets_url)
        w = [x for x in w_resp.json()["wallets"] if x["id"] == wallet_id][0]
        self.assertGreaterEqual(w["live_confirm_trades_count"], 3)
        self.assertIn("live", w["verification_status"].lower())

    def test_profit_accrual_stays_put_and_manual_send_all(self):
        """Verify under $1000 stays put rule and manual 'SEND ALL TOTAL PROFITS' button dispatch."""
        from .engine import ProfitAccrualCoordinator
        from .models import ProfitAccrualMarker, ProfitVaultConfig

        # 1. Accrue profits from strategy trade ($12.50)
        marker = ProfitAccrualCoordinator.record_profit(
            gross_profit_usd=12.50,
            source="agent_strategy",
            chain="base",
            symbol="WETH-USDC",
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker.status, ProfitAccrualMarker.MarkerStatus.STAYS_PUT)
        self.assertTrue(marker.under_1000_rule_applied)

        # 2. Check running totals via API: funds stay put
        markers_url = reverse("agentresearch-profit-markers")
        m_resp = self.client.get(markers_url)
        self.assertEqual(m_resp.status_code, status.HTTP_200_OK)
        totals = m_resp.json()["running_totals"]
        self.assertGreater(totals["accumulated_pending_usd"], 0.0)

        # 3. Press manual "SEND ALL TOTAL PROFITS" button
        send_url = reverse("agentresearch-send-all-profits")
        send_resp = self.client.post(send_url, format="json")
        self.assertEqual(send_resp.status_code, status.HTTP_200_OK)
        send_data = send_resp.json()
        self.assertEqual(send_data["status"], "dispatched")
        self.assertGreater(send_data["total_sent_usd"], 0.0)
        self.assertEqual(send_data["running_totals"]["accumulated_pending_profits_usd"], 0.0)

        # 4. Verify marker transitioned to DISPATCHED_TO_VAULT
        marker.refresh_from_db()
        self.assertEqual(marker.status, ProfitAccrualMarker.MarkerStatus.DISPATCHED_TO_VAULT)
        self.assertIsNotNone(marker.dispatched_at)

        # 5. Verify FundTracker recorded the transfer so BusScheduler is immediately aware
        from services.fund_tracker import get_fund_tracker
        tracker = get_fund_tracker()
        locked = tracker.get_locked_amount("base", "USDC")
        self.assertGreaterEqual(locked, marker.accrued_profit_usd)

    def test_fund_tracker_api_endpoints(self):
        """Verify GET and POST endpoints for fund reservation tracking."""
        from services.fund_tracker import get_fund_tracker
        tracker = get_fund_tracker()
        tracker.clear_all()

        url = reverse("agentresearch-fund-tracker")
        get_resp = self.client.get(url)
        self.assertEqual(get_resp.status_code, status.HTTP_200_OK)
        data = get_resp.json()
        self.assertIn("active_reservations_count", data)
        self.assertIn("locked_totals", data)

        # Record manual outbound transfer via API
        post_resp = self.client.post(url, {
            "action": "record_outbound",
            "chain": "base",
            "token": "USDC",
            "amount": 12.50,
            "destination": "0xexternal_wallet",
            "reason": "operator_test",
        }, format="json")
        self.assertEqual(post_resp.status_code, status.HTTP_201_CREATED)
        self.assertTrue(post_resp.json()["ok"])
        self.assertEqual(tracker.get_locked_amount("base", "USDC"), 12.50)

        # Create reservation and release via API
        res = tracker.reserve_funds(chain="base", token="USDC", amount=5.0, reason="test_res")
        self.assertEqual(tracker.get_locked_amount("base", "USDC"), 17.50)

        rel_resp = self.client.post(url, {
            "action": "release_reservation",
            "reservation_id": res["id"],
        }, format="json")
        self.assertEqual(rel_resp.status_code, status.HTTP_200_OK)
        self.assertTrue(rel_resp.json()["ok"])
        self.assertEqual(tracker.get_locked_amount("base", "USDC"), 12.50)




class AgentTokenLimiterAndAutoResumeTests(TestCase):
    """Unit and integration tests for TokenLimiter, quota awareness, and auto-restart."""

    def setUp(self):
        self.client = APIClient()
        from services.token_limiter import TokenLimiter
        # Reset limiter states for test isolation
        for ag in ("agy", "codex", "claude"):
            TokenLimiter.record_success(ag)

    def test_token_limit_marker_detection_and_delay_parsing(self):
        """Verify token limit keyword detection and retry delay parsing across formats."""
        from services.token_limiter import is_token_or_rate_limit, parse_retry_delay

        # Keyword checks
        self.assertTrue(is_token_or_rate_limit("Error 429: Too Many Requests"))
        self.assertTrue(is_token_or_rate_limit("Rate limit exceeded for model: gemini-2.5-flash"))
        self.assertTrue(is_token_or_rate_limit("insufficient_quota: upgrade your plan"))
        self.assertTrue(is_token_or_rate_limit("ResourceExhausted: TPM limit reached"))
        self.assertFalse(is_token_or_rate_limit("All calculations completed successfully."))

        # Delay parsing
        delay_sec = parse_retry_delay("Please try again in 14.5 seconds.")
        self.assertAlmostEqual(delay_sec, 14.5, places=1)

        delay_min = parse_retry_delay("Retry after 2 minutes")
        self.assertAlmostEqual(delay_min, 120.0, places=1)

        import time
        future_ts = time.time() + 45
        delay_ts = parse_retry_delay(f'{{"error": "rate_limit", "resets_at": {int(future_ts)}}}')
        self.assertGreater(delay_ts, 40.0)
        self.assertLessEqual(delay_ts, 50.0)

    def test_agent_limit_cooldown_and_availability(self):
        """Verify an agent cools down on rate limit and becomes available on success."""
        from services.token_limiter import TokenLimiter

        # Initially available
        self.assertTrue(TokenLimiter.is_available("agy"))

        # Record rate limit with 5s delay
        TokenLimiter.record_rate_limit("agy", "429 Too Many Requests", delay_seconds=5.0)
        self.assertFalse(TokenLimiter.is_available("agy"))
        state = TokenLimiter.get_state("agy")
        self.assertEqual(state.status, "cooling_down")
        self.assertGreater(state.seconds_remaining, 0.0)
        self.assertEqual(state.consecutive_hits, 1)

        # Clear/success restores availability
        TokenLimiter.record_success("agy")
        self.assertTrue(TokenLimiter.is_available("agy"))
        self.assertEqual(TokenLimiter.get_state("agy").status, "available")

    def test_auto_restart_call_retries_and_succeeds(self):
        """Verify auto_restart_call automatically waits and retries when hitting token limits."""
        from services.token_limiter import TokenLimiter

        call_count = 0

        def flaky_external_agent_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt hits rate limit with tiny 0.2s pause
                raise RuntimeError("429 rate limit exceeded; try again in 0.2 seconds")
            return {"decision": "BUY", "confidence": 0.95}

        result = TokenLimiter.auto_restart_call("codex", flaky_external_agent_call, max_retries=3)
        self.assertEqual(call_count, 2)
        self.assertEqual(result["decision"], "BUY")
        self.assertTrue(TokenLimiter.is_available("codex"))

    def test_token_limits_api_endpoints(self):
        """Verify GET and POST endpoints on /api/agent-research/token-limits/."""
        url = reverse("agentresearch-token-limits")

        # GET token limits
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        limits = resp.json()["token_limits"]
        self.assertIn("agy", limits)
        self.assertIn("codex", limits)
        self.assertIn("claude", limits)

        # POST simulate limit
        sim_resp = self.client.post(url, {"agent": "claude", "action": "simulate_limit", "delay": 2.0}, format="json")
        self.assertEqual(sim_resp.status_code, status.HTTP_200_OK)
        claude_state = sim_resp.json()["token_limits"]["claude"]
        self.assertEqual(claude_state["status"], "cooling_down")
        self.assertFalse(claude_state["is_available"])

        # POST reset
        reset_resp = self.client.post(url, {"agent": "claude", "action": "reset"}, format="json")
        self.assertEqual(reset_resp.status_code, status.HTTP_200_OK)
        self.assertTrue(reset_resp.json()["token_limits"]["claude"]["is_available"])

    def test_agent_research_status_includes_token_limits(self):
        """Verify /api/agent-research/status/ includes unified token limits across agents."""
        url = reverse("agentresearch-status")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertIn("token_limits", data)
        self.assertIn("agy", data["token_limits"])
        self.assertIn("codex", data["token_limits"])
        self.assertIn("claude", data["token_limits"])


class PipelineFeedbackAndRapidProfitabilityTests(TestCase):
    """Unit and integration tests for cross-app feedback loops, alpha mirroring, and dynamic sizing."""

    def setUp(self):
        self.client = APIClient()
        from .engine import RESEARCH_COORDINATOR, AlphaWalletResearcher
        RESEARCH_COORDINATOR.ensure_default_agents()
        AlphaWalletResearcher.ensure_default_alpha_wallets()

    def test_alpha_wallet_broadcasts_signal_to_money_button(self):
        """Verify confirmed alpha wallet trades broadcast directly to money button overrides."""
        from .engine import AlphaWalletResearcher
        from .models import AlphaTraderWallet
        from services.atomic_json import read_json
        from pathlib import Path

        wallet = AlphaTraderWallet.objects.filter(chain="base").first()
        self.assertIsNotNone(wallet)

        # Confirm trade
        res = AlphaWalletResearcher.confirm_and_graduate_wallet(wallet.pk)
        self.assertNotIn("error", res)

        # Check money_button_edge_overrides.json
        ov_path = Path("data") / "money_button_edge_overrides.json"
        self.assertTrue(ov_path.is_file())
        data, ok = read_json(ov_path, default={})
        self.assertTrue(ok)
        symbols = data.get("symbols", {})
        self.assertIn("WETH-USDC", symbols)
        sym_ov = symbols["WETH-USDC"]
        self.assertTrue(sym_ov.get("has_alpha_signal"))
        self.assertEqual(sym_ov.get("chain"), "base")
        self.assertGreater(sym_ov.get("alpha_win_rate"), 0.70)

    def test_money_button_rapid_evaluation_with_overrides(self):
        """Verify MoneyButtonStrategy incorporates alpha signal, arbitrage radar, and GA overrides."""
        import time
        import types
        from trading.strategies.base import StrategyContext
        from trading.strategies.money_button import MoneyButtonStrategy

        now = time.time()
        # Create a trending price series (15 samples, 100s apart)
        dt = 100.0
        prices = [100.0 + (i * 1.5) for i in range(16)]  # strong trend ~22.5% over 1500s
        samples = [(now - (len(prices) - 1 - i) * dt, p, 500.0) for i, p in enumerate(prices)]

        state = types.SimpleNamespace()
        state.symbol = "WETH-USDC"
        state.base_token = "WETH"
        state.quote_token = "USDC"
        state.samples = samples

        ctx = StrategyContext(
            chain="base",
            last_price=float(prices[-1]),
            last_volume=1000.0,
            fee_rate=0.0005,  # 5 bps typical Aerodrome / Uni v3
            available_quote=100.0,
            available_base=0.0,
            risk_budget=1.0,
        )

        strat = MoneyButtonStrategy()
        candidate = strat.evaluate(state, ctx)
        self.assertIsNotNone(candidate)
        meta = candidate["meta"]
        self.assertEqual(meta["lane"], "money_button")
        self.assertGreater(candidate["score"], 0.0)
        self.assertGreaterEqual(meta["confidence"], 0.62)

    def test_dynamic_win_streak_clip_scaling(self):
        """Verify Strategy._size_enter applies streak multiplier and respects minimum clip bounds."""
        from trading.strategies.base import Strategy, StrategyContext

        class DummyStrategy(Strategy):
            strategy_id = "test_strat"
            default_horizon = "5m"
            def evaluate(self, state, ctx):
                return None

        strat = DummyStrategy()
        ctx = StrategyContext(
            chain="base",
            last_price=2800.0,
            last_volume=1000.0,
            fee_rate=0.0005,
            available_quote=5.0,  # $5 balance (cents trading)
            available_base=0.0,
            risk_budget=1.0,
        )

        # Baseline entry size should clamp to min_clip ($0.25)
        size_quote = strat._size_enter(ctx, edge=0.02)
        self.assertGreaterEqual(size_quote, 0.25)
        self.assertLessEqual(size_quote, 5.0)
