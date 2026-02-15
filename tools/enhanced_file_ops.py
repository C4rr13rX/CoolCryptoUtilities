"""Enhanced file operations with smart search and signal-based commands."""
import os
import sys
import json
import signal
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any

class EnhancedFileOps:
    def __init__(self):
        self.search_cache = {}
        
    def smart_file_search(self, pattern: str, context: Dict[str, Any]) -> List[str]:
        """Intelligent file search with context awareness."""
        results = []
        search_paths = context.get("search_paths", ["."]) 
        for path in search_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if pattern.lower() in file.lower():
                            results.append(os.path.join(root, file))
        return results[:50]  # Limit results
        
    def signal_based_command(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute command with signal-based timeout control."""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, 
                                  text=True, timeout=timeout)
            return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
