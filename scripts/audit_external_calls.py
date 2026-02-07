import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "free_api_dependency_matrix.md"

SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "collected_static",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".c0d3r",
    "runtime",
    "logs",
    "storage",
    "models",
    "data",
    "codex_transcripts",
}

FILE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".env"}

URL_RE = re.compile(r"https?://[A-Za-z0-9\.\-_/:%\?#&=]+")
SDK_HINTS = [
    "boto3",
    "openai",
    "anthropic",
    "cohere",
    "googleapiclient",
    "google.cloud",
    "azure",
    "stripe",
    "twilio",
    "sendgrid",
    "slack_sdk",
    "telegram",
    "pusher",
    "amplitude",
    "posthog",
    "datadog",
    "sentry",
    "newrelic",
    "algoliasearch",
    "pinecone",
    "weaviate",
    "qdrant_client",
    "redis",
    "elasticsearch",
    "opensearch",
    "praw",
    "tweepy",
    "feedparser",
    "requests",
    "httpx",
    "urllib",
    "aiohttp",
    "websockets",
]


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in FILE_EXTS:
                yield path


def scan_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return [], []
    urls = URL_RE.findall(text)
    hits = []
    for hint in SDK_HINTS:
        if hint in text:
            hits.append(hint)
    return urls, hits


def main():
    url_map = {}
    sdk_map = {}
    for path in iter_files(ROOT):
        urls, hits = scan_file(path)
        for url in urls:
            url_map.setdefault(url, set()).add(str(path.relative_to(ROOT)))
        for hit in hits:
            sdk_map.setdefault(hit, set()).add(str(path.relative_to(ROOT)))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write("# Free API Dependency Matrix (Auto-Scan)\n\n")
        f.write("This is an automated scan of outbound URLs and SDK hints. Manual review required.\n\n")

        f.write("## URL References\n\n")
        if not url_map:
            f.write("- No URLs detected.\n")
        else:
            for url in sorted(url_map.keys()):
                locations = ", ".join(sorted(url_map[url]))
                f.write(f"- `{url}`\n")
                f.write(f"  - locations: {locations}\n")

        f.write("\n## SDK / Library Hints\n\n")
        if not sdk_map:
            f.write("- No SDK hints detected.\n")
        else:
            for hint in sorted(sdk_map.keys()):
                locations = ", ".join(sorted(sdk_map[hint]))
                f.write(f"- `{hint}`\n")
                f.write(f"  - locations: {locations}\n")


if __name__ == "__main__":
    main()
