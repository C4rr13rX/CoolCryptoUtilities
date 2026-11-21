"""tools/wizard_learning/build_civilizational_discovery_course.py

Permissible-License Civilizational Foundations Course & Environmental Redundancy
Training Script (`CC0-1.0 / Public Domain Scientific Invariant`).

Compiles and saves:
  1. `data/wizard_learning/civilizational_discovery_course/LICENSE_AND_PROVENANCE.md`
  2. `data/wizard_learning/civilizational_discovery_course/civilizational_discoveries_course.jsonl`
  3. `data/wizard_learning/civilizational_discovery_course/redundant_environment_episodes.jsonl`
  4. `data/wizard_learning/civilizational_discovery_course/converged_causal_weights.json`
  5. `data/wizard_learning/civilizational_discovery_course/course_manifest.json`

And with `--train-live`, trains W1z4rD's live `:8090` neural multi-pools
(`causal_env_in -> causal_mechanics_out` and `chat_in -> chat_out`) + `AnchoredMemory`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "tools" / "c0d3rV2"), str(PROJECT_ROOT / "tools"), str(PROJECT_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from tools.wizard_learning.civilizational_causal_engine import (
    CIVILIZATIONAL_DISCOVERIES,
    COURSE_DIR,
    POOL_CAUSAL_IN,
    POOL_CAUSAL_OUT,
    RedundantCausalConvergenceModel,
)


LICENSE_DOC = """# License & Provenance — W1z4rD Civilizational Foundations & Redundant Causal Course

- **License**: `CC0 1.0 Universal (Public Domain Dedication)` / Public-Domain Natural Laws & SI Constants.
- **Provenance**:
  1. Classical mechanics, hydrostatics, thermodynamics, electromagnetism, optics, chemical stoichiometry, Boolean algebra, calculus, Nyquist-Shannon sampling/capacity, PID control, cryptographic hashing, Hebbian plasticity, and molecular biology invariants represent uncopyrightable laws of nature and public-domain historical milestones (3500 BCE – 1985 CE).
  2. All structured causal tuples `(constituents_x, conditions_zrqst, co_action_p -> emergent_effect_y -> prerogative_tie)`, counterfactual failure rules, and austere reinvention recipes are original first-principles formulations dedicated under **CC0 1.0** for unrestricted training, modification, and redistribution.
"""


def build_course_lessons_jsonl() -> List[Dict[str, Any]]:
    lessons: List[Dict[str, Any]] = []
    for idx, d in enumerate(CIVILIZATIONAL_DISCOVERIES, 1):
        q = f"What key discovery is {d.title} and what are its first-principles causal mechanics?"
        paraphrases = [
            f"Explain the historical discovery of {d.title} and how to reinvent it from scratch",
            f"Teach me {d.title} in terms of constituents, enabling conditions, and mechanics",
            f"How did {d.title} advance human civilization and how does W1z4rD use it?",
        ]
        broken_bullets = "\n".join(f"  - Missing `{k}`: {v}" for k, v in d.broken_condition_effects.items())
        ans = (
            f"**Civilizational Discovery #{idx:02d}: {d.title}** *({d.era} · `{d.pillar}` · {d.license})*\n\n"
            f"- **Historical Breakthrough**: {d.historical_milestone}.\n"
            f"- **Constituents ($x$)**: `{', '.join(d.constituents_x)}`\n"
            f"- **Enabling Conditions ($z, r, q, s, t$)**: `{', '.join(d.conditions_zrqst)}`\n"
            f"- **Co-Acting Process ($p$)**: `{d.co_action_p}`\n"
            f"- **Emergent Mechanics ($y$)**: {d.emergent_effect_y}\n"
            f"- **Governing Law**: `{d.governing_law}`\n"
            f"- **Counterfactual Necessity**:\n{broken_bullets}\n"
            f"- **Reinvent-the-Wheel Recipe**: {d.reinvent_recipe}\n"
            f"- **W1z4rD Prerogative & Goal Alignment**: {d.prerogative_tie}"
        )
        lessons.append({
            "id": f"civ_course_{idx:03d}",
            "discovery_id": d.id,
            "category": f"civilizational_{d.pillar}",
            "license": d.license,
            "question": q,
            "paraphrases": paraphrases,
            "answer": ans,
        })
    return lessons


def build_redundant_environment_episodes(rounds: int = 18) -> List[Dict[str, Any]]:
    distractors = [f"noise:context_clutter_{i:02d}" for i in range(18)]
    episodes: List[Dict[str, Any]] = []
    for d_idx, d in enumerate(CIVILIZATIONAL_DISCOVERIES):
        for r in range(rounds):
            carrier = d.surface_carriers[r % len(d.surface_carriers)]
            d1 = distractors[(d_idx + r) % len(distractors)]
            d2 = distractors[(d_idx * 3 + r * 5) % len(distractors)]
            episodes.append({
                "episode_id": f"env_{d.id}_r{r:02d}",
                "discovery_id": d.id,
                "license": d.license,
                "surface_carrier": carrier,
                "constituents_x": list(d.constituents_x),
                "conditions_zrqst": list(d.conditions_zrqst),
                "co_action_p": d.co_action_p,
                "distractor_clutter": [d1, d2],
                "emergent_effect_y": d.emergent_effect_y,
                "prerogative_tie": d.prerogative_tie,
                "dynamic": True,  # Excluded from exact Q&A index so inference uses converged weights!
            })
    return episodes


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def train_course_into_wizard(lessons: List[Dict[str, Any]], endpoint: str = "http://127.0.0.1:8090") -> Dict[str, int]:
    from services.anchored_memory import AnchoredMemoryManager
    from tools.wizard_learning.fast_conversational_engine import (
        POOL_CHAT_IN,
        POOL_CHAT_OUT,
        record_hot_chat_pair,
    )

    am = AnchoredMemoryManager.get_instance()
    anchored = 0
    causal_pool_trained = 0
    chat_pool_trained = 0

    for pid in (POOL_CAUSAL_IN, POOL_CAUSAL_OUT, POOL_CHAT_IN, POOL_CHAT_OUT):
        _post_json(f"{endpoint}/multi_pool/register", {"pool_id": pid}, timeout=1.5)

    # 1. Anchor course lessons & train `chat_in -> chat_out`
    for row in lessons:
        q = str(row["question"])
        a = str(row["answer"])
        pars = list(row.get("paraphrases") or [])
        am.record_memory(q, a, variations=pars, source="civilizational_discovery_course_cc0", category=row["category"])
        anchored += 1
        record_hot_chat_pair(q, a)
        for pv in pars:
            record_hot_chat_pair(pv, a)
        if _post_json(
            f"{endpoint}/multi_pool/train_pair",
            {"src_pool": POOL_CHAT_IN, "src": q[:320], "tgt_pool": POOL_CHAT_OUT, "tgt": a[:1800], "passes": 12, "lr": 0.45},
            timeout=1.2,
        ):
            chat_pool_trained += 1

    # 2. Train redundant environmental causal tuples into `causal_env_in -> causal_mechanics_out`
    for d in CIVILIZATIONAL_DISCOVERIES:
        tgt_mechanics = f"{d.id} | {d.title} | {d.governing_law} | {d.emergent_effect_y} | {d.prerogative_tie}"
        for carrier in d.surface_carriers:
            src_env = f"{carrier} {' '.join(d.constituents_x)} {' '.join(d.conditions_zrqst)} {d.co_action_p}"
            if _post_json(
                f"{endpoint}/multi_pool/train_pair",
                {"src_pool": POOL_CAUSAL_IN, "src": src_env[:320], "tgt_pool": POOL_CAUSAL_OUT, "tgt": tgt_mechanics[:1200], "passes": 10, "lr": 0.5},
                timeout=1.0,
            ):
                causal_pool_trained += 1

    return {
        "anchored_lessons": anchored,
        "chat_pool_pairs_trained": chat_pool_trained,
        "causal_pool_redundant_exposures_trained": causal_pool_trained,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-live", action="store_true", help="Train W1z4rD neural pools (:8090) and AnchoredMemory")
    ap.add_argument("--rounds", type=int, default=18, help="Redundant environmental exposure rounds per discovery")
    args = ap.parse_args()

    COURSE_DIR.mkdir(parents=True, exist_ok=True)
    (COURSE_DIR / "LICENSE_AND_PROVENANCE.md").write_text(LICENSE_DOC, encoding="utf-8")

    # 1. Train & save redundancy-converged causal weights
    conv_stats = RedundantCausalConvergenceModel.get_instance().train_default_redundant_environment(
        rounds_per_discovery=args.rounds,
        save=True,
    )

    # 2. Write course lessons & redundant environmental episodes JSONLs
    lessons = build_course_lessons_jsonl()
    episodes = build_redundant_environment_episodes(rounds=args.rounds)

    with (COURSE_DIR / "civilizational_discoveries_course.jsonl").open("w", encoding="utf-8") as fh:
        for r in lessons:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (COURSE_DIR / "redundant_environment_episodes.jsonl").open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")

    live_stats: Dict[str, int] = {}
    if args.train_live:
        live_stats = train_course_into_wizard(lessons)

    manifest = {
        "compiled_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "license": "CC0-1.0 / Public Domain Scientific Invariant",
        "discoveries_count": len(CIVILIZATIONAL_DISCOVERIES),
        "course_lessons_count": len(lessons),
        "redundant_episodes_count": len(episodes),
        "convergence": conv_stats,
        "live_training": live_stats,
    }
    (COURSE_DIR / "course_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
