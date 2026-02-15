"""Tool capability mapper for enhanced execution"""
from typing import Dict, List

class ToolCapabilityMapper:
    def __init__(self):
        self.capabilities: Dict[str, List[str]] = {}
    def map_capabilities(self, tool_name: str) -> List[str]:
        return self.capabilities.get(tool_name, [])
    def register_capability(self, tool: str, capability: str):
        if tool not in self.capabilities:
            self.capabilities[tool] = []
        self.capabilities[tool].append(capability)
