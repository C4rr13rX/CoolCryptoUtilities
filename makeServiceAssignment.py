import os
import json
from datetime import datetime, timedelta

CHAIN_NAME = os.getenv("CHAIN_NAME", "base").strip().lower() or "base"
PAIR_INDEX_FILE = os.getenv("PAIR_INDEX_FILE", f"data/pair_index_{CHAIN_NAME}.json")
ASSIGNMENT_FILE = os.getenv("PAIR_ASSIGNMENT_FILE", f"data/{CHAIN_NAME}_pair_provider_assignment.json")

# Update these to match your main script
GRANULARITY_SECONDS = 60 * 5
YEARS_BACK = 3

def assign_ankr_only():
    if not os.path.exists(PAIR_INDEX_FILE):
        print(f"❌ File not found: {PAIR_INDEX_FILE}")
        return

    os.makedirs("data", exist_ok=True)

    with open(PAIR_INDEX_FILE, "r") as f:
        pair_data = json.load(f)

    # Compute ISO8601 start date for human clarity and replayability
    utc_now = datetime.utcnow()
    start_date = (utc_now - timedelta(days=YEARS_BACK * 365)).strftime('%Y-%m-%dT%H:%M:%SZ')

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

    with open(ASSIGNMENT_FILE, "w") as f:
        json.dump(assignment, f, indent=2)

    print(f"✅ Assignments saved to {ASSIGNMENT_FILE}")
    print(f"🔢 Total pairs assigned: {len(assignment['pairs'])}")

if __name__ == "__main__":
    assign_ankr_only()
