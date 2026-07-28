from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "c0d3rV2"))

from tools.c0d3rV2.outline_refiner import OutlineRefiner


PROMPTS = [
    "Build a Django inventory application with CSV import and export.",
    "Create an Android field notebook that works offline.",
    "Write a practical book teaching small businesses process automation.",
    "Build a desktop scientific instrument GUI for logging temperature experiments.",
    "Create a responsive website for selling downloadable design templates.",
    "Develop a C++ Qt radio spectrum visualization tool using recorded sample data.",
]


def main() -> None:
    rows = []
    for passes in range(1, 7):
        scores = []
        passed = []
        for prompt in PROMPTS:
            # Fixed evidence isolates refinement-depth performance from live
            # search-provider availability. Runtime commercial plans still
            # require real results from C0D3R's web_search tool.
            result = OutlineRefiner(
                passes=passes,
                market_search=lambda _query: {"results": [{
                    "title": "Benchmark professional-quality evidence fixture",
                    "url": "benchmark://planning-quality",
                    "snippet": "Representative workflow, validation, reliability, and buyer-value requirements.",
                }]},
            ).refine(prompt)
            scores.append(float(result["quality"]["score"]))
            passed.append(bool(result["quality"]["passed"]))
        rows.append({
            "configured_passes": passes,
            "average_score": round(sum(scores) / len(scores), 2),
            "minimum_score": min(scores),
            "threshold_pass_rate": round(sum(passed) / len(passed), 3),
        })
    for index, row in enumerate(rows):
        previous = rows[index - 1]["average_score"] if index else 0
        row["marginal_gain"] = round(row["average_score"] - previous, 2)
    premium = [row for row in rows if row["minimum_score"] >= 98.0 and row["threshold_pass_rate"] == 1.0]
    chosen = premium[0]["configured_passes"] if premium else 6
    payload = {
        "created_at": time.time(),
        "prompts": PROMPTS,
        "quality_threshold": 92,
        "premium_planning_target": 98,
        "results": rows,
        "chosen_default": chosen,
        "selection_reason": "lowest tested pass count reaching the 98/100 premium planning target for every benchmark before marginal gains plateau",
    }
    path = ROOT / "runtime" / "c0d3rv2" / "outline_refinement_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
