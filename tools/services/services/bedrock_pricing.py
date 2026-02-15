
    import threading
    import time
    from typing import Optional, Tuple
    from aws_cost_fetcher import AWSCostFetcher, ModelPricing
    pass
    _cost_fetcher = None
    _fetcher_lock = threading.Lock()
    pass
    def get_cost_fetcher() -> AWSCostFetcher:
        global _cost_fetcher
        with _fetcher_lock:
            if _cost_fetcher is None:
                _cost_fetcher = AWSCostFetcher()
            return _cost_fetcher
    pass
    def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> Tuple[Optional[float], Optional[float]]:
        if not model_id or input_tokens < 0 or output_tokens < 0:
            return None, None
    pass
        fetcher = get_cost_fetcher()
        pricing = fetcher.get_pricing_sync(model_id)
    pass
        if not pricing:
            # Start background fetch for next time
            fetcher.get_pricing_async(model_id)
            return None, None
    pass
        input_cost = (input_tokens / 1000.0) * pricing.input_per_1k
        output_cost = (output_tokens / 1000.0) * pricing.output_per_1k
    pass
        return input_cost, output_cost
    pass
    def lookup_pricing(model_id: str) -> Optional[ModelPricing]:
        if not model_id:
            return None
    pass
        fetcher = get_cost_fetcher()
        pricing = fetcher.get_pricing_sync(model_id)
    pass
        if not pricing:
            # Start background fetch for next time
            fetcher.get_pricing_async(model_id)
    pass
        return pricing
    