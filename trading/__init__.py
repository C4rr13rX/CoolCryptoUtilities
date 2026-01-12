from typing import TYPE_CHECKING

# Avoid importing heavy dependencies (e.g., TensorFlow via pipeline) unless the
# attribute is actually requested.
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

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from .bot import TradingBot
    from .data_stream import MarketDataStream
    from .pipeline import TrainingPipeline
    from .optimizer import BayesianBruteForceOptimizer
    from .portfolio import PortfolioState, TokenHolding
    from .scheduler import BusScheduler, TradeDirective
    from .selector import GhostTradingSupervisor, select_pairs


def __getattr__(name: str):
    if name == "TradingBot":
        from .bot import TradingBot
        return TradingBot
    if name == "MarketDataStream":
        from .data_stream import MarketDataStream
        return MarketDataStream
    if name == "TrainingPipeline":
        from .pipeline import TrainingPipeline
        return TrainingPipeline
    if name == "BayesianBruteForceOptimizer":
        from .optimizer import BayesianBruteForceOptimizer
        return BayesianBruteForceOptimizer
    if name == "GhostTradingSupervisor":
        from .selector import GhostTradingSupervisor
        return GhostTradingSupervisor
    if name == "select_pairs":
        from .selector import select_pairs
        return select_pairs
    if name == "PortfolioState":
        from .portfolio import PortfolioState
        return PortfolioState
    if name == "TokenHolding":
        from .portfolio import TokenHolding
        return TokenHolding
    if name == "BusScheduler":
        from .scheduler import BusScheduler
        return BusScheduler
    if name == "TradeDirective":
        from .scheduler import TradeDirective
        return TradeDirective
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
