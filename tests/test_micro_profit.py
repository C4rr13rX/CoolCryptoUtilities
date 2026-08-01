from trading.micro_profit import evaluate_micro_profit
from trading.bot import TradingBot


def test_small_trade_is_allowed_when_cents_remain_after_all_costs() -> None:
    result = evaluate_micro_profit(
        notional_usd=10.0,
        gross_return=0.012,
        variable_cost_rate=0.0065,
        fixed_cost_usd=0.01,
        minimum_net_profit_usd=0.02,
    )
    assert result.viable is True
    assert result.net_profit_usd == 0.045


def test_small_trade_is_rejected_when_fees_consume_the_edge() -> None:
    result = evaluate_micro_profit(
        notional_usd=3.0,
        gross_return=0.006,
        variable_cost_rate=0.0065,
        minimum_net_profit_usd=0.02,
    )
    assert result.viable is False
    assert result.reason == "edge_does_not_cover_variable_costs"


def test_rates_are_not_confused_with_the_dollar_floor() -> None:
    result = evaluate_micro_profit(
        notional_usd=100.0,
        gross_return=0.007,
        variable_cost_rate=0.0065,
        minimum_net_profit_usd=0.02,
    )
    assert result.viable is True
    assert 0.049 < result.net_profit_usd < 0.051


def test_control_only_bus_actions_do_not_require_a_swap_route() -> None:
    bot = TradingBot.__new__(TradingBot)
    result = bot._execute_bus_actions_sync(
        actions=[
            {"action": "notify_add_funds", "required_usd": 4.25},
            {"action": "scan_micro_opportunities", "dust_tokens": ["ABC"]},
        ],
        plan_snapshot={},
        dry_run=False,
    )
    assert result["ok"] is True
    assert [row["action"] for row in result["executed"]] == [
        "notify_add_funds", "scan_micro_opportunities"
    ]
