import os
import json
import sys
from datetime import datetime, timedelta, timezone

CHAIN_NAME = os.getenv("CHAIN_NAME", "base").strip().lower() or "base"
PAIR_INDEX_FILE = os.getenv("PAIR_INDEX_FILE", f"data/pair_index_{CHAIN_NAME}.json")
ASSIGNMENT_FILE = os.getenv("PAIR_ASSIGNMENT_FILE", f"data/{CHAIN_NAME}_pair_provider_assignment.json")

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# Allow overrides from Data Lab (env), with safe defaults.
GRANULARITY_SECONDS = _int_env("GRANULARITY_SECONDS", 60 * 5)
YEARS_BACK = _int_env("YEARS_BACK", 3)

def assign_ankr_only():
    if not os.path.exists(PAIR_INDEX_FILE):
        print(f"[assignments] File not found: {PAIR_INDEX_FILE}")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    with open(PAIR_INDEX_FILE, "r") as f:
        pair_data = json.load(f)

    # Compute ISO8601 start date for human clarity and replayability
    utc_now = datetime.now(timezone.utc)
    start_date = (utc_now - timedelta(days=YEARS_BACK * 365)).strftime("%Y-%m-%dT%H:%M:%SZ")

    assignment = {
        "granularity_seconds": GRANULARITY_SECONDS,
        "years_back": YEARS_BACK,
        "start_date": start_date,
        "chain": CHAIN_NAME,
        "pairs": {}
    }

    for address, metadata in pair_data.items():
        assignment["pairs"][address] = {
            "symbol": metadata["symbol"],
            "index": metadata["index"],
            "completed": False,
            # This field records the next block to fetch, or None to start from the beginning
            "next_block": None
        }

    try:
        with open(ASSIGNMENT_FILE, "w", encoding="utf-8") as f:
            json.dump(assignment, f, indent=2)
    except Exception as exc:
        print(f"[assignments] Failed to write assignment file: {exc}")
        sys.exit(1)

    print(f"[assignments] Saved to {ASSIGNMENT_FILE}")
    print(f"[assignments] Total pairs assigned: {len(assignment['pairs'])}")

if __name__ == "__main__":
    assign_ankr_only()
