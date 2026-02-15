import json
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

@dataclass
class ModelPricing:
    input_per_1k: float
    output_per_1k: float
    model_name: str
    source: str = "aws"
    last_updated: float = 0.0
    as_of: str = ""

class AWSCostFetcher:
    def __init__(self, cache_file: str = "model_pricing_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.load_cache()
        
    def load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    self.cache = {k: ModelPricing(**v) for k, v in data.items()}
            except Exception:
                self.cache = {}
                
    def save_cache(self):
        try:
            data = {k: {
                "input_per_1k": v.input_per_1k,
                "output_per_1k": v.output_per_1k,
                "model_name": v.model_name,
                "source": v.source,
                "last_updated": v.last_updated,
                "as_of": v.as_of
            } for k, v in self.cache.items()}
            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
            
    def get_pricing_sync(self, model_name: str) -> Optional[ModelPricing]:
        if model_name in self.cache:
            cached = self.cache[model_name]
            if time.time() - cached.last_updated < 3600:
                return cached
        
        pricing = self._fetch_aws_pricing(model_name)
        if pricing:
            self.cache[model_name] = pricing
            self.save_cache()
        return pricing
        
    def _fetch_aws_pricing(self, model_name: str) -> Optional[ModelPricing]:
        fallback_rates = {
            "claude-3-sonnet": (0.003, 0.015),
            "claude-3-haiku": (0.00025, 0.00125),
            "claude-3-opus": (0.015, 0.075)
        }
        
        if model_name in fallback_rates:
            input_rate, output_rate = fallback_rates[model_name]
            return ModelPricing(
                input_per_1k=input_rate,
                output_per_1k=output_rate,
                model_name=model_name,
                source="fallback",
                last_updated=time.time(),
                as_of="2024-01-01"
            )
        return None
