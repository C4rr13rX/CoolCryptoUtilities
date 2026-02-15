import os
import json
import time
import signal
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class HierarchicalFileFinder:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.session_cache = {}
        self.cache_file = self.project_root / ".file_cache.json"
        self.load_project_cache()

    def load_project_cache(self):
        try:
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    self.project_cache = json.load(f)
            else:
                self.project_cache = {"system_info": {}, "working_directory": str(self.project_root), "project_context": {}, "files": {}}
        except:
            self.project_cache = {"system_info": {}, "working_directory": str(self.project_root), "project_context": {}, "files": {}}

    def save_project_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.project_cache, f, indent=2)
        except Exception as e:
            print(f"Cache save failed: {e}")

    def session_cache_lookup(self, filename: str) -> Optional[str]:
        return self.session_cache.get(filename)

    def project_cache_lookup(self, filename: str) -> Optional[str]:
        return self.project_cache.get("files", {}).get(filename)

    def filesystem_search(self, filename: str) -> Optional[str]:
        search_paths = [self.project_root]
        for search_path in search_paths:
            for root, dirs, files in os.walk(search_path):
                if filename in files:
                    return str(Path(root) / filename)
        return None

    def find_file(self, filename: str) -> Tuple[Optional[str], str]:
        location = self.session_cache_lookup(filename)
        if location:
            return location, f"Found {filename} in session cache"
        
        location = self.project_cache_lookup(filename)
        if location and Path(location).exists():
            self.session_cache[filename] = location
            return location, f"Found {filename} in project cache"
        
        location = self.filesystem_search(filename)
        if location:
            self.session_cache[filename] = location
            self.project_cache["files"][filename] = location
            self.project_cache["system_info"] = {
                "timestamp": time.time(),
                "working_directory": str(Path.cwd()),
                "project_context": str(self.project_root)
            }
            self.save_project_cache()
            return location, f"Found {filename} via filesystem search"
        
        return None, f"File {filename} not found in session cache, project cache, or filesystem under {self.project_root}"
