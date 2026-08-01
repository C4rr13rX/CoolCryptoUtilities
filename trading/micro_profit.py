"""Unit-safe viability checks for very small crypto trades."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MicroProfitDecision:
    viable: bool
    notional_usd: float
    gross_profit_usd: float
    estimated_cost_usd: float
    net_profit_usd: float
    minimum_net_profit_usd: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_micro_profit(
    *,
    notional_usd: float,
    gross_return: float,
    variable_cost_rate: float,
    fixed_cost_usd: float = 0.0,
    minimum_net_profit_usd: float = 0.02,
) -> MicroProfitDecision:
    """Return whether an opportunity earns the requested dollars after costs.

    Rates and dollar amounts deliberately have separate arguments.  This keeps
    a two-cent profit floor from accidentally becoming a 2% or 20% margin.
    """
    notional = max(0.0, float(notional_usd))
    gross = max(0.0, float(gross_return)) * notional
    costs = max(0.0, float(variable_cost_rate)) * notional + max(0.0, float(fixed_cost_usd))
    net = gross - costs
    floor = max(0.0, float(minimum_net_profit_usd))
    if notional <= 0.0:
        reason = "no_notional"
    elif gross_return <= variable_cost_rate:
        reason = "edge_does_not_cover_variable_costs"
    elif net + 1e-12 < floor:
        reason = "net_profit_below_dollar_floor"
    else:
        reason = "profitable_after_costs"
    return MicroProfitDecision(
        viable=reason == "profitable_after_costs",
        notional_usd=notional,
        gross_profit_usd=gross,
        estimated_cost_usd=costs,
        net_profit_usd=net,
        minimum_net_profit_usd=floor,
        reason=reason,
    )
