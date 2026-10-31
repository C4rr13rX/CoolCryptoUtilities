import os
import json

PAIR_INDEX_FILE = "data/pair_index_top2000.json"
ASSIGNMENT_FILE = "data/pair_provider_assignment.json"

def assign_providers():
    if not os.path.exists(PAIR_INDEX_FILE):
        print(f"❌ File not found: {PAIR_INDEX_FILE}")
        return

    os.makedirs("data", exist_ok=True)

    with open(PAIR_INDEX_FILE, "r") as f:
        pair_data = json.load(f)

    provider_assignment = {}
    providers = ["ankr", "thegraph"]
    i = 0

    for address, metadata in pair_data.items():
        provider = providers[i % len(providers)]
        provider_assignment[address] = {
            "symbol": metadata["symbol"],
            "index": metadata["index"],
            "provider": provider,
            "completed": False
        }
        i += 1

    with open(ASSIGNMENT_FILE, "w") as f:
        json.dump(provider_assignment, f, indent=2)

    print(f"✅ Provider assignments saved to {ASSIGNMENT_FILE}")
    print(f"🔢 Total pairs assigned: {len(provider_assignment)}")

if __name__ == "__main__":
    assign_providers()
