from __future__ import annotations

from django.apps import AppConfig


class TradingAgentConfig(AppConfig):
    """The reading agent: an LLM that studies market data and trades on it.

    It is a READING agent. It never modifies system code -- it reads market
    data, forms hypotheses, tests them cheaply in ghost, and graduates the ones
    that earn it to live trading. Everything it learns lives in this app's
    tables, not in the codebase.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "tradingagent"
    verbose_name = "Trading Agent"
