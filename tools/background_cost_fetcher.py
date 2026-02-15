import threading
import time
import json
import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ModelPricing:
    input_per_1k: float
    output_per_1k: float
    as_of: str
    model_id: str
    source: str = "aws_bedrock"

class BackgroundCostFetcher:
    def __init__(self):
        self.pricing_cache = {}
        self.fetch_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.load_pricing_cache()
        
    def start_background_fetcher(self):
        if self.fetch_thread and self.fetch_thread.is_alive():
            return
        self.stop_event.clear()
        self.fetch_thread = threading.Thread(target=self._background_fetch_loop, daemon=True)
        self.fetch_thread.start()
        
    def _background_fetch_loop(self):
        while not self.stop_event.wait(30):
            try:
                self._update_pricing_via_web_search()
            except Exception as e:
                self.logger.error(f"Background pricing update failed: {e}")
                
    def _update_pricing_via_web_search(self):
        try:
            search_query = "AWS Bedrock model pricing per 1000 tokens input output cost"
            # Simulate web search - replace with actual datalab_web call
            pricing_data = self._search_aws_pricing(search_query)
            if pricing_data:
                self._update_cache_from_search(pricing_data)
        except Exception as e:
            self.logger.error(f"Web search pricing update failed: {e}")
            
    def _search_aws_pricing(self, query: str) -> Optional[Dict]:
        # Placeholder for web search integration
        # In real implementation, use ::datalab_web meta command
        return {
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015}
        }
        
    def _update_cache_from_search(self, pricing_data: Dict):
        timestamp = datetime.now().isoformat()
        for model_id, costs in pricing_data.items():
            self.pricing_cache[model_id] = ModelPricing(
                input_per_1k=costs["input"],
                output_per_1k=costs["output"],
                as_of=timestamp,
                model_id=model_id
            )
        self.save_pricing_cache()
        
    def get_pricing(self, model_id: str) -> Optional[ModelPricing]:
        # Try exact match first
        if model_id in self.pricing_cache:
            return self.pricing_cache[model_id]
            
        # Try fuzzy matching for AWS Bedrock model IDs
        for cached_model in self.pricing_cache:
            if self._models_match(model_id, cached_model):
                return self.pricing_cache[cached_model]
                
        # Trigger background fetch for unknown model
        self._queue_model_search(model_id)
        return None
        
    def _models_match(self, model1: str, model2: str) -> bool:
        # Extract base model name from AWS Bedrock format
        def extract_base(model: str) -> str:
            # anthropic.claude-3-sonnet-20240229-v1:0 -> claude-3-sonnet
            if "." in model:
                model = model.split(".", 1)[1]
            if ":" in model:
                model = model.split(":")[0]
            # Remove version/date suffixes
            model = re.sub(r"-\d{8}(-v\d+)?", "", model)
            return model.lower()
            
        return extract_base(model1) == extract_base(model2)
        
    def _queue_model_search(self, model_id: str):
        # Add to search queue for background thread
        pass
        
    def load_pricing_cache(self):
        try:
            with open("pricing_cache.json", "r") as f:
                data = json.load(f)
                self.pricing_cache = {
                    k: ModelPricing(**v) for k, v in data.items()
                }
        except (FileNotFoundError, json.JSONDecodeError):
            self.pricing_cache = {}
            
    def save_pricing_cache(self):
        try:
            with open("pricing_cache.json", "w") as f:
                data = {k: asdict(v) for k, v in self.pricing_cache.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save pricing cache: {e}")
            
    def stop(self):
        self.stop_event.set()
        if self.fetch_thread:
            self.fetch_thread.join(timeout=5)

# Global instance
_cost_fetcher = None

def get_cost_fetcher() -> BackgroundCostFetcher:
    global _cost_fetcher
    if _cost_fetcher is None:
        _cost_fetcher = BackgroundCostFetcher()
        _cost_fetcher.start_background_fetcher()
    return _cost_fetcher

def lookup_pricing_with_fallback(model_id: str) -> Optional[ModelPricing]:
    fetcher = get_cost_fetcher()
    pricing = fetcher.get_pricing(model_id)
    
    if pricing is None:
        # Provide reasonable fallback based on model type
        if "claude-3-sonnet" in model_id.lower():
            return ModelPricing(0.003, 0.015, datetime.now().isoformat(), model_id, "fallback")
        elif "claude-3-haiku" in model_id.lower():
            return ModelPricing(0.00025, 0.00125, datetime.now().isoformat(), model_id, "fallback")
        else:
            return ModelPricing(0.001, 0.005, datetime.now().isoformat(), model_id, "fallback")
    
    return pricing
