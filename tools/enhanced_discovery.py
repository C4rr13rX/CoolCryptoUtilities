import os
import json
from pathlib import Path

class AIDiscoveryEngine:
    def __init__(self, project_path):
        self.project_path = project_path
    
    def methodical_project_scan(self):
        """Perform comprehensive project analysis"""
        return {
            "project_structure": self._analyze_structure(),
            "file_types": self._categorize_files(),
            "key_patterns": self._find_patterns(),
            "cloud_indicators": self._detect_cloud_resources()
        }
    
    def _analyze_structure(self):
        """Analyze directory structure"""
        structure = {}
        if os.path.exists(self.project_path):
            for root, dirs, files in os.walk(self.project_path):
                rel_path = os.path.relpath(root, self.project_path)
                structure[rel_path] = {"dirs": dirs, "files": files}
        return structure
    
    def _categorize_files(self):
        """Categorize files by type and purpose"""
        categories = {"web": [], "config": [], "audio": [], "data": [], "code": [], "assets": []}
        if os.path.exists(self.project_path):
            for root, dirs, files in os.walk(self.project_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    full_path = os.path.join(root, file)
                    if ext in [".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".vue"]:
                        categories["web"].append(full_path)
                    elif ext in [".json", ".yml", ".yaml", ".config", ".env"]:
                        categories["config"].append(full_path)
                    elif ext in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]:
                        categories["audio"].append(full_path)
                    elif ext in [".py", ".java", ".cpp", ".c", ".go", ".rs"]:
                        categories["code"].append(full_path)
                    elif ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"]:
                        categories["assets"].append(full_path)
                    else:
                        categories["data"].append(full_path)
        return categories
    
    def _find_patterns(self):
        """Find key patterns in project"""
        patterns = {"russian": [], "audio": [], "lessons": []}
        if os.path.exists(self.project_path):
            for root, dirs, files in os.walk(self.project_path):
                for file in files:
                    if any(term in file.lower() for term in ["russian", "ru"]):
                        patterns["russian"].append(os.path.join(root, file))
                    if any(term in file.lower() for term in ["audio", "sound", "mp3"]):
                        patterns["audio"].append(os.path.join(root, file))
                    if any(term in file.lower() for term in ["lesson", "course", "learn"]):
                        patterns["lessons"].append(os.path.join(root, file))
        return patterns
    
    def _detect_cloud_resources(self):
        """Detect cloud deployment indicators"""
        indicators = {"aws": [], "deployment": [], "config": []}
        if os.path.exists(self.project_path):
            for root, dirs, files in os.walk(self.project_path):
                for file in files:
                    if any(term in file.lower() for term in ["aws", "s3", "cloudfront"]):
                        indicators["aws"].append(os.path.join(root, file))
                    if any(term in file.lower() for term in ["deploy", "build", "dist"]):
                        indicators["deployment"].append(os.path.join(root, file))
                    if file.lower() in ["package.json", "requirements.txt", "dockerfile"]:
                        indicators["config"].append(os.path.join(root, file))
        return indicators
