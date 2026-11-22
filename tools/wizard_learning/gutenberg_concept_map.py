"""tools/wizard_learning/gutenberg_concept_map.py

Resource-Aware Concept-to-Concept Relational Graph built from Public-Domain
Project Gutenberg Dictionaries (#29765 Webster's Unabridged Dictionary & #22 Roget's Thesaurus).

Provides:
  1. Sub-millisecond concept lookup (`lookup_concept(word)`) returning structured
     relational edges:
       - `is_a`: hypernym / genus concepts ("bicycle" -> ["vehicle", "machine"])
       - `has_part`: constituent parts ("bicycle" -> ["wheel", "handle", "saddle", "pedal", "crank"])
       - `used_for`: functional verbs/affordances ("bicycle" -> ["propel", "steer", "ride"])
       - `roget_classes`: macro conceptual categories ("Vehicle", "Locomotion", ...)
       - `domain`: scientific/technical domain tag ("Mech.", "Phys.", "Chem.", ...)
       - `definition`: concise public-domain sense
  2. Multi-hop concept-to-concept path discovery (`explain_concept_relationship(a, b)`)
     via BFS over `is_a`, `has_part`, `used_for`, and shared `roget_classes` edges!
  3. Direct offline constituent & trait enrichment for `compositional_domain_coder.py`
     so W1z4rD can model thousands of real-world objects & concepts in code even with
     zero external web calls!
"""
from __future__ import annotations

import collections
import json
import re
import threading
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONCEPT_MAP_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "gutenberg_concept_map"
CONCEPT_GRAPH_JSONL = CONCEPT_MAP_DIR / "gutenberg_concept_graph.jsonl"


def _stem_simple(word: str) -> str:
    w = re.sub(r"[^a-z0-9\- ]+", "", (word or "").strip().lower()).strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and not w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-1]
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


class GutenbergConceptMap:
    _instance: Optional["GutenbergConceptMap"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._roget_members: Dict[str, List[str]] = collections.defaultdict(list)
        self._reverse_is_a: Dict[str, List[str]] = collections.defaultdict(list)
        self._reverse_part_of: Dict[str, List[str]] = collections.defaultdict(list)
        self._loaded = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "GutenbergConceptMap":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def ensure_loaded(self) -> bool:
        if self._loaded:
            return bool(self._nodes)
        with self._load_lock:
            if self._loaded:
                return bool(self._nodes)
            if not CONCEPT_GRAPH_JSONL.exists():
                self._loaded = True
                return False
            with CONCEPT_GRAPH_JSONL.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    hw = str(rec.get("concept") or "").lower().strip()
                    if not hw:
                        continue
                    self._nodes[hw] = rec
                    for rc in rec.get("roget_classes", ()):
                        if len(self._roget_members[rc]) < 250:
                            self._roget_members[rc].append(hw)
                    for hyp in rec.get("is_a", ()):
                        if len(self._reverse_is_a[hyp]) < 500:
                            self._reverse_is_a[hyp].append(hw)
                    for prt in rec.get("has_part", ()):
                        if len(self._reverse_part_of[prt]) < 300:
                            self._reverse_part_of[prt].append(hw)

            # Promote any concept referenced >= 2 times via `is_a` or `has_part` (whose Sense 1 in 1913 had `[Obs.]`, e.g. `lever`)
            # into a first-class bridge node so relational paths traverse seamlessly!
            ref_candidates = set(self._reverse_is_a.keys()) | set(self._reverse_part_of.keys())
            for cand in ref_candidates:
                if cand in self._nodes or len(cand) < 3:
                    continue
                subs = self._reverse_is_a.get(cand, [])
                containers = self._reverse_part_of.get(cand, [])
                if len(subs) + len(containers) >= 2:
                    self._nodes[cand] = {
                        "concept": cand,
                        "pos": "n.",
                        "domain": "Inferred Graph Bridge",
                        "definition": (
                            f"Relational concept referenced across {len(subs)} sub-concepts "
                            f"and as a constituent part of {len(containers)} composite entities "
                            f"(e.g. {', '.join((containers + subs)[:5])})."
                        ),
                        "is_a": ["mechanism"] if containers else ["concept"],
                        "has_part": [],
                        "used_for": [],
                        "roget_classes": [],
                    }
            self._loaded = True
            return bool(self._nodes)

    def reload(self) -> int:
        with self._load_lock:
            self._nodes.clear()
            self._roget_members.clear()
            self._reverse_is_a.clear()
            self._reverse_part_of.clear()
            self._loaded = False
        self.ensure_loaded()
        return len(self._nodes)

    @property
    def size(self) -> int:
        self.ensure_loaded()
        return len(self._nodes)

    def lookup(self, term: str) -> Optional[Dict[str, Any]]:
        if not self.ensure_loaded():
            return None
        clean = re.sub(r"[^a-zA-Z0-9\-\s]", "", (term or "")).strip().lower()
        if not clean:
            return None
        if clean in self._nodes:
            return self._nodes[clean]
        stemmed = _stem_simple(clean)
        if stemmed in self._nodes:
            return self._nodes[stemmed]
        # Try head noun of multi-word phrase (e.g. 'mountain bicycle' -> 'bicycle')
        toks = clean.split()
        if len(toks) >= 2:
            last_tok = _stem_simple(toks[-1])
            if last_tok in self._nodes:
                return self._nodes[last_tok]
        return None

    def find_concept_path(self, source: str, target: str, max_hops: int = 4) -> Optional[List[Tuple[str, str, str]]]:
        """BFS shortest relational path `[(u, relation, v), ...]` connecting `source` and `target`."""
        if not self.ensure_loaded():
            return None
        src_rec = self.lookup(source)
        tgt_rec = self.lookup(target)
        if not src_rec or not tgt_rec:
            return None
        src = src_rec["concept"]
        tgt = tgt_rec["concept"]
        if src == tgt:
            return [(src, "identical_concept", tgt)]

        # Check immediate shared Roget conceptual category first
        shared_roget = set(src_rec.get("roget_classes", ())) & set(tgt_rec.get("roget_classes", ()))
        queue: Deque[Tuple[str, List[Tuple[str, str, str]]]] = collections.deque([(src, [])])
        visited: Set[str] = {src}

        while queue:
            curr, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            node = self._nodes.get(curr)
            if not node:
                continue

            neighbors: List[Tuple[str, str]] = []
            for h in node.get("is_a", ()):
                neighbors.append(("is_a", h))
            for p in node.get("has_part", ()):
                neighbors.append(("has_part", p))
            for u in node.get("used_for", ()):
                neighbors.append(("acts_by", u))
            for sub in self._reverse_is_a.get(curr, ())[:120]:
                neighbors.append(("encompasses", sub))
            for whole in self._reverse_part_of.get(curr, ())[:80]:
                neighbors.append(("part_of", whole))
            for rc in node.get("roget_classes", ())[:4]:
                for peer in self._roget_members.get(rc, ())[:80]:
                    neighbors.append((f"roget:{rc}", peer))

            for rel, nxt in neighbors:
                if nxt == tgt:
                    return path + [(curr, rel, nxt)]
                if nxt in self._nodes and nxt not in visited and len(visited) < 4000:
                    visited.add(nxt)
                    queue.append((nxt, path + [(curr, rel, nxt)]))

        if shared_roget:
            rc0 = sorted(shared_roget)[0]
            return [(src, f"shares_roget_category({rc0})", tgt)]
        return None


_CONCEPT_REL_QUERY_RE = re.compile(
    r"^(?:how\s+(?:are|is|do|does)\s+([a-zA-Z\- ]+?)\s+(?:and|related\s+to|connected\s+to)\s+([a-zA-Z\- ]+?)(?:\s+related|\s+connected)?|"
    r"relate\s+([a-zA-Z\- ]+?)\s+(?:to|and)\s+([a-zA-Z\- ]+?))\??$",
    re.I,
)
_CONCEPT_MAP_INSPECT_RE = re.compile(
    r"^(?:concept\s+map\s+(?:for|of)|what\s+concepts\s+define|gutenberg\s+definition\s+of)\s+([a-zA-Z\- ]+?)\??$",
    re.I,
)
_SIMPLE_DEFN_QUERY_RE = re.compile(
    r"^(?:what\s+(?:is|are)\s+(?:a|an|the)\s+|define\s+(?:a|an|the)?\s*|meaning\s+of\s+)([a-zA-Z][a-zA-Z\-]{2,28})\??$",
    re.I,
)


def resolve_gutenberg_concept_query(text: str) -> Optional[Dict[str, Any]]:
    """Answer explicit concept-relationship or concept-anatomy queries using the Gutenberg Concept Map."""
    raw = (text or "").strip()
    if not raw:
        return None

    m_rel = _CONCEPT_REL_QUERY_RE.match(raw)
    if m_rel:
        a = (m_rel.group(1) or m_rel.group(3) or "").strip()
        b = (m_rel.group(2) or m_rel.group(4) or "").strip()
        # Strip trailing helper words
        b = re.sub(r"\s+(?:related|connected|linked)$", "", b, flags=re.I).strip()
        cmap = GutenbergConceptMap.get_instance()
        rec_a = cmap.lookup(a)
        rec_b = cmap.lookup(b)
        if rec_a and rec_b:
            path = cmap.find_concept_path(a, b)
            chain_str = (
                " -> ".join(f"**{u}** *[{rel}]* -> **{v}**" for u, rel, v in path)
                if path
                else f"**{rec_a['concept']}** (`{', '.join(rec_a.get('is_a', [])[:2])}`) and **{rec_b['concept']}** (`{', '.join(rec_b.get('is_a', [])[:2])}`) occupy distinct branches of the Gutenberg ontology."
            )
            ans = (
                f"**Concept-Map Relationship (`{rec_a['concept']}` <-> `{rec_b['concept']}`)**\n\n"
                f"- **Relational Path**: {chain_str}\n"
                f"- **`{rec_a['concept']}`**: {rec_a.get('definition', '')} *(is_a: `{', '.join(rec_a.get('is_a', []))}`, parts: `{', '.join(rec_a.get('has_part', []))}`)*\n"
                f"- **`{rec_b['concept']}`**: {rec_b.get('definition', '')} *(is_a: `{', '.join(rec_b.get('is_a', []))}`, parts: `{', '.join(rec_b.get('has_part', []))}`)*\n"
                f"*(Source: Project Gutenberg Public Domain Dictionaries #29765 & #22)*"
            )
            return {
                "question": text,
                "answer": ans,
                "decoder": "gutenberg_concept_graph",
                "confidence": 0.94,
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": [f"concept:{rec_a['concept']}", f"concept:{rec_b['concept']}", "graph:gutenberg"],
            }

    m_ins = _CONCEPT_MAP_INSPECT_RE.match(raw)
    if m_ins:
        term = m_ins.group(1).strip()
        rec = GutenbergConceptMap.get_instance().lookup(term)
        if rec:
            ans = (
                f"**Gutenberg Concept Node: `{rec['concept']}`** *({rec.get('pos', 'n.')}{' · ' + rec['domain'] if rec.get('domain') else ''})*\n\n"
                f"- **Definition**: {rec.get('definition', '')}\n"
                f"- **Genus (`is_a`)**: `{', '.join(rec.get('is_a', [])) or 'root_concept'}`\n"
                f"- **Constituents (`has_part`)**: `{', '.join(rec.get('has_part', [])) or 'atomic'}`\n"
                f"- **Affordances (`used_for` / `acts_by`)**: `{', '.join(rec.get('used_for', [])) or 'general'}`\n"
                f"- **Roget Taxonomic Classes**: `{', '.join(rec.get('roget_classes', [])) or 'unclassified'}`\n"
                f"*(Source: Project Gutenberg #29765 Webster's 1913 & #22 Roget's Thesaurus — Public Domain)*"
            )
            return {
                "question": text,
                "answer": ans,
                "decoder": "gutenberg_concept_graph",
                "confidence": 0.96,
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": [f"concept:{rec['concept']}", "graph:gutenberg"],
            }
    return None


def resolve_gutenberg_definition_fallback(text: str) -> Optional[Dict[str, Any]]:
    """Offline dictionary & concept-edge fallback for simple 'What is a <word>?' / 'Define <word>' queries."""
    m = _SIMPLE_DEFN_QUERY_RE.match((text or "").strip())
    if not m:
        return None
    term = m.group(1).strip()
    rec = GutenbergConceptMap.get_instance().lookup(term)
    if not rec:
        return None
    parts_str = f"\n- **Constituents (`has_part`)**: `{', '.join(rec['has_part'])}`" if rec.get("has_part") else ""
    isa_str = f"\n- **Genus (`is_a`)**: `{', '.join(rec['is_a'])}`" if rec.get("is_a") else ""
    roget_str = f"\n- **Conceptual Classes**: `{', '.join(rec['roget_classes'])}`" if rec.get("roget_classes") else ""
    ans = (
        f"**{rec['concept'].capitalize()}** *({rec.get('pos', 'n.')}{' · ' + rec['domain'] if rec.get('domain') else ''})*\n\n"
        f"{rec.get('definition', '')}"
        f"{isa_str}{parts_str}{roget_str}\n"
        f"*(Source: Project Gutenberg Public Domain Dictionaries #29765 & #22)*"
    )
    return {
        "question": text,
        "answer": ans,
        "decoder": "gutenberg_concept_graph",
        "confidence": 0.91,
        "confidence_tier": "high",
        "hypothesis": False,
        "activated_concepts": [f"concept:{rec['concept']}", "graph:gutenberg"],
    }
