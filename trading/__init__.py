from typing import TYPE_CHECKING

from .bot import TradingBot
from .data_stream import MarketDataStream
from .pipeline import TrainingPipeline
from .optimizer import BayesianBruteForceOptimizer
from .portfolio import PortfolioState, TokenHolding
from .scheduler import BusScheduler, TradeDirective

if TYPE_CHECKING:
    from .selector import GhostTradingSupervisor, select_pairs


def __getattr__(name: str):
    if name in {"GhostTradingSupervisor", "select_pairs"}:
        from .selector import GhostTradingSupervisor, select_pairs

        return GhostTradingSupervisor if name == "GhostTradingSupervisor" else select_pairs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TradingBot",
    "MarketDataStream",
    "TrainingPipeline",
    "BayesianBruteForceOptimizer",
    "GhostTradingSupervisor",
    "select_pairs",
    "PortfolioState",
    "TokenHolding",
    "BusScheduler",
    "TradeDirective",
]
