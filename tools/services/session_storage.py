from typing import Dict, Any
import json
import os

class SessionStore:
    def __init__(self, storage_path: str = "sessions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def save_session(self, session_id: str, data: Dict[str, Any]):
        with open(f"{self.storage_path}/{session_id}.json", "w") as f:
            json.dump(data, f, indent=2)
            
    def load_session(self, session_id: str) -> Dict[str, Any]:
        try:
            with open(f"{self.storage_path}/{session_id}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

def default_session_config():
    return {
        "timeout": 300,
        "max_retries": 3,
        "enable_logging": True
    }
