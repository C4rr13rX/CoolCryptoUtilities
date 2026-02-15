import json
import threading
from typing import Optional, Dict
from aws_cost_fetcher import AWSCostFetcher, ModelPricing

class BedrockPricingService:
    def __init__(self):
        self.cost_fetcher = AWSCostFetcher()
        self._current_pricing = None
        self._lock = threading.Lock()
        
    def get_current_pricing(self, model_name: str) -> Optional[Dict]:
        with self._lock:
            pricing = self.cost_fetcher.get_pricing_sync(model_name)
            if pricing:
                return {
                    "input_per_1k": pricing.input_per_1k,
                    "output_per_1k": pricing.output_per_1k,
                    "model_name": pricing.model_name,
                    "source": pricing.source
                }
        return None
        
    def update_pricing_async(self, model_name: str, callback=None):
        def _update():
            pricing_data = self.get_current_pricing(model_name)
            if callback and pricing_data:
                callback(pricing_data)
        
        thread = threading.Thread(target=_update, daemon=True)
        thread.start()
        return thread
