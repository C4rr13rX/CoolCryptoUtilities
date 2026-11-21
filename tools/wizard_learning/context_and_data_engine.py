"""tools/wizard_learning/context_and_data_engine.py

Sub-millisecond (<0.5 ms) Context-Aware Time/Date/Location, Inline Data/JSON,
Output-Format Adherence, Paraphrase Inverted-Index, and Online-Lookup Engine.
"""
from __future__ import annotations

import datetime
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .runtime_context import WORLD_TZ_OFFSETS, build_runtime_context, convert_time_for_zone

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "massive_context_corpus"
ENTERPRISE_CORPUS_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "enterprise_stem_steam_corpus"
CIVILIZATIONAL_COURSE_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "civilizational_discovery_course"
ONLINE_LEARNED_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "online_self_learned_corpus"
PC_APLUS_COURSE_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "pc_aplus_and_desktop_course"
CORPUS_DIRS = (
    CORPUS_DIR,
    ENTERPRISE_CORPUS_DIR,
    CIVILIZATIONAL_COURSE_DIR,
    PC_APLUS_COURSE_DIR,
    ONLINE_LEARNED_DIR,
)

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "into",
    "about", "it", "its", "this", "that", "these", "those", "me", "my", "you",
    "your", "we", "our", "i", "do", "does", "did", "can", "could", "would",
    "should", "will", "please", "tell", "give", "show", "explain", "what",
    "whats", "how", "why", "when", "where", "who", "which", "briefly", "quickly",
})


def _want_json(low: str) -> bool:
    return bool(re.search(r"\b(?:as|in|return|output|format(?:\s+as)?|reply(?:\s+only)?\s+in)\s+(?:valid\s+|strict\s+)?json\b|\bjson\s+(?:format|object|output|response)\b", low))


def _want_csv(low: str) -> bool:
    return bool(re.search(r"\b(?:as|in|return|output|convert\s+to|format(?:\s+as)?)\s+csv\b", low))


def _want_md_table(low: str) -> bool:
    return bool(re.search(r"\bmarkdown\s+table\b|\bas\s+a\s+table\b|\btable\s+format\b", low))


def _want_bullets(low: str) -> bool:
    return bool(re.search(r"\bbullet\s*(?:points?|list)\b|\bas\s+bullets\b", low))


def _want_one_sentence(low: str) -> bool:
    return bool(re.search(r"\b(?:one|1|single)\s+sentence\b|\bin\s+brief\b|\bconcisely\b", low))


def _format_answer_if_requested(ans: str, low: str, structured_payload: Optional[Dict[str, Any]] = None) -> str:
    if _want_json(low):
        if structured_payload is not None:
            return json.dumps(structured_payload, ensure_ascii=False)
        return json.dumps({"answer": ans}, ensure_ascii=False)
    if _want_one_sentence(low):
        clean = re.sub(r"\*\*|\*|`|#+\s*", "", ans).strip()
        first_line = " ".join(ln.strip() for ln in clean.splitlines() if ln.strip())
        m = re.match(r"^(.+?[.!?])(?:\s|$)", first_line)
        return m.group(1) if m else first_line[:220]
    return ans


# ── 1. Time, Date, Timezone, Calendar & Location Resolver (<0.2 ms) ─────────

_TIME_RE = re.compile(
    r"\b(?:what(?:'s|\s+is)?\s+(?:the\s+)?(?:current\s+|local\s+)?time|current\s+time|local\s+time|"
    r"time\s+(?:is\s+it|right\s+now|now)|got\s+the\s+time|tell\s+me\s+the\s+time|clock\s+say|"
    r"give\s+me\s+the\s+time|what\s+hour\s+is\s+it|current\s+timestamp|unix\s+timestamp|iso\s+timestamp)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(?:what(?:'s|\s+is)?\s+(?:today'?s?\s+|the\s+current\s+|the\s+)?date|today'?s?\s+date|"
    r"current\s+date|what\s+day\s+(?:is\s+it|of\s+the\s+week|is\s+today)|which\s+day\s+is\s+today|"
    r"what\s+month\s+is\s+it|what\s+year\s+is\s+it|date\s+today|date\s+right\s+now|"
    r"tomorrow'?s?\s+date|what\s+day\s+is\s+tomorrow|yesterday'?s?\s+date|what\s+day\s+was\s+yesterday|"
    r"date\s+in\s+\d+\s+days?|days?\s+(?:until|till|left\s+until|to)\b)\b",
    re.I,
)
_LOC_RE = re.compile(
    r"\b(?:where\s+am\s+i|my\s+(?:current\s+)?location|what(?:'s|\s+is)?\s+my\s+location|"
    r"current\s+location|what\s+city\s+am\s+i\s+in|what\s+timezone\s+am\s+i\s+in|"
    r"my\s+timezone|current\s+timezone|local\s+timezone|what\s+time\s*zone)\b",
    re.I,
)
_ENV_RE = re.compile(
    r"\b(?:what\s+(?:os|operating\s+system)\s+(?:am\s+i|is\s+this)|which\s+os\b|"
    r"what\s+workspace\s+(?:am\s+i|is\s+this)|current\s+workspace|python\s+version)\b",
    re.I,
)


def resolve_time_date_location(text: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Answer time, date, world-clock, calendar-delta, location, and host-env queries
    strictly from the live runtime context envelope — NEVER firing on unrelated queries."""
    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.lower()

    # A. World timezone conversion ("What time is it in Tokyo?", "time in UTC", etc.)
    tz_in = re.search(r"\b(?:time|clock|date)\b.*?\b(?:in|at|for)\s+([a-z][a-z\s]{1,20}?)(?:\?|$|,|\bin\s+json\b|\bas\s+json\b)", low)
    if not tz_in:
        tz_in = re.search(r"\b(?:convert\s+time\s+to|current\s+time\s+in)\s+([a-z][a-z\s]{1,20}?)(?:\?|$|,)", low)
    if tz_in:
        zone_cand = tz_in.group(1).strip()
        for key in sorted(WORLD_TZ_OFFSETS.keys(), key=len, reverse=True):
            if zone_cand == key or zone_cand.startswith(key + " ") or key in zone_cand:
                conv = convert_time_for_zone(key)
                if conv:
                    ans = f"It is currently **{conv['time_12h']}** (`{conv['time_24h']}`) on **{conv['weekday']}, {conv['date_iso']}** in **{conv['zone']}**."
                    out = _format_answer_if_requested(ans, low, conv)
                    return {
                        "question": text,
                        "answer": out,
                        "decoder": "context_clock",
                        "confidence": 1.0,
                        "confidence_tier": "high",
                        "hypothesis": False,
                        "activated_concepts": ["tool:context_clock", f"tz:{key}"],
                    }

    # B. Calendar arithmetic: tomorrow, yesterday, in N days, days until X
    today_dt = datetime.date(int(ctx["year"]), datetime.datetime.strptime(ctx["month"], "%B").month, int(ctx["day"]))

    if "tomorrow" in low and any(w in low for w in ("date", "day", "what")):
        tmrw = today_dt + datetime.timedelta(days=1)
        payload = {"relative": "tomorrow", "date_iso": tmrw.isoformat(), "weekday": tmrw.strftime("%A"), "date_human": f"{tmrw.strftime('%A, %B')} {tmrw.day}, {tmrw.year}"}
        ans = f"Tomorrow is **{payload['date_human']}** (`{payload['date_iso']}`)."
        return {"question": text, "answer": _format_answer_if_requested(ans, low, payload), "decoder": "context_clock", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:context_clock", "calendar:tomorrow"]}

    if "yesterday" in low and any(w in low for w in ("date", "day", "what")):
        yest = today_dt - datetime.timedelta(days=1)
        payload = {"relative": "yesterday", "date_iso": yest.isoformat(), "weekday": yest.strftime("%A"), "date_human": f"{yest.strftime('%A, %B')} {yest.day}, {yest.year}"}
        ans = f"Yesterday was **{payload['date_human']}** (`{payload['date_iso']}`)."
        return {"question": text, "answer": _format_answer_if_requested(ans, low, payload), "decoder": "context_clock", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:context_clock", "calendar:yesterday"]}

    in_days = re.search(r"\b(?:in|after)\s+(\d+)\s+days?\b", low)
    if in_days and any(w in low for w in ("date", "day", "what")):
        n = int(in_days.group(1))
        fut = today_dt + datetime.timedelta(days=n)
        payload = {"offset_days": n, "date_iso": fut.isoformat(), "weekday": fut.strftime("%A"), "date_human": f"{fut.strftime('%A, %B')} {fut.day}, {fut.year}"}
        ans = f"In {n} day(s) it will be **{payload['date_human']}** (`{payload['date_iso']}`)."
        return {"question": text, "answer": _format_answer_if_requested(ans, low, payload), "decoder": "context_clock", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:context_clock", "calendar:offset"]}

    until_m = re.search(r"\bdays?\s+(?:until|till|left\s+until|to)\s+([a-z0-9\-\s]+?)(?:\?|$|,|\bin\s+json\b|\bas\s+json\b)", low)
    if until_m:
        target_s = until_m.group(1).strip()
        target_dt: Optional[datetime.date] = None
        if "christmas" in target_s:
            target_dt = datetime.date(today_dt.year, 12, 25)
            if target_dt < today_dt:
                target_dt = datetime.date(today_dt.year + 1, 12, 25)
        elif "new year" in target_s:
            target_dt = datetime.date(today_dt.year + 1, 1, 1)
        elif "halloween" in target_s:
            target_dt = datetime.date(today_dt.year, 10, 31)
            if target_dt < today_dt:
                target_dt = datetime.date(today_dt.year + 1, 10, 31)
        else:
            iso_m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", target_s)
            if iso_m:
                try:
                    target_dt = datetime.date(int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3)))
                except ValueError:
                    target_dt = None
        if target_dt is not None:
            delta = (target_dt - today_dt).days
            payload = {"today": today_dt.isoformat(), "target": target_dt.isoformat(), "days_until": delta}
            ans = f"There are **{delta} day(s)** from `{today_dt.isoformat()}` until `{target_dt.isoformat()}`."
            return {"question": text, "answer": _format_answer_if_requested(ans, low, payload), "decoder": "context_clock", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:context_clock", "calendar:delta"]}

    # C. Local time / date / location / timezone
    ask_time = bool(_TIME_RE.search(low))
    ask_date = bool(_DATE_RE.search(low))
    ask_loc = bool(_LOC_RE.search(low))
    ask_env = bool(_ENV_RE.search(low))

    if ask_time or ask_date or ask_loc or ask_env:
        payload: Dict[str, Any] = {}
        parts: List[str] = []
        if ask_time:
            payload.update({"time_12h": ctx["time_12h"], "time_24h": ctx["time_24h"], "timezone": ctx["timezone"], "utc_offset": ctx["utc_offset"], "iso_timestamp": ctx["iso_timestamp"]})
            if "unix" in low:
                payload["unix_ts"] = ctx["unix_ts"]
                parts.append(f"Current Unix timestamp: `{int(ctx['unix_ts'])}` (`{ctx['iso_timestamp']}`).")
            elif "iso" in low:
                parts.append(f"Current ISO-8601 timestamp: `{ctx['iso_timestamp']}`.")
            else:
                parts.append(f"It is currently **{ctx['time_12h']}** (`{ctx['time_24h']}` {ctx['timezone']}, `{ctx['utc_offset']}`).")
        if ask_date:
            payload.update({"date_human": ctx["date_human"], "date_iso": ctx["date_iso"], "weekday": ctx["weekday"], "month": ctx["month"], "year": ctx["year"]})
            if "what year" in low and not ("date" in low or "day" in low or "month" in low):
                parts.append(f"The current year is **{ctx['year']}**.")
            elif "what month" in low and not ("date" in low or "day" in low):
                parts.append(f"The current month is **{ctx['month']} {ctx['year']}**.")
            elif "day of the week" in low or "which day" in low:
                parts.append(f"Today is **{ctx['weekday']}** ({ctx['date_human']}).")
            else:
                parts.append(f"Today's date is **{ctx['date_human']}** (`{ctx['date_iso']}`).")
        if ask_loc:
            payload.update({"location": ctx["location"], "timezone": ctx["timezone"], "utc_offset": ctx["utc_offset"]})
            if "timezone" in low or "time zone" in low:
                parts.append(f"Your current timezone is **{ctx['timezone']}** (`{ctx['utc_offset']}`).")
            else:
                parts.append(f"Your current location context is **{ctx['location']}** (timezone `{ctx['timezone']}`, `{ctx['utc_offset']}`).")
        if ask_env:
            payload.update({"os": ctx["os"], "python_version": ctx["python_version"], "workspace": ctx["workspace"]})
            if "python" in low:
                parts.append(f"Python runtime: `{ctx['python_version']}` on `{ctx['os']}`.")
            elif "workspace" in low:
                parts.append(f"Active workspace: `{ctx['workspace']}`.")
            else:
                parts.append(f"Host environment: `{ctx['os']}` (Python `{ctx['python_version']}`, workspace `{ctx['workspace']}`).")

        ans = " ".join(parts)
        return {
            "question": text,
            "answer": _format_answer_if_requested(ans, low, payload),
            "decoder": "context_clock",
            "confidence": 1.0,
            "confidence_tier": "high",
            "hypothesis": False,
            "activated_concepts": ["tool:context_clock", "context:live_runtime"],
        }

    return None


# ── 2. Inline Data, Caller Context & Structured Output Engine (<0.5 ms) ─────

def _render_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "| empty |\n| --- |"
    cols = list(rows[0].keys())
    hdr = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows[:30]]
    return "\n".join([hdr, sep, *body])


def _render_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = [",".join(str(c) for c in cols)]
    for r in rows[:50]:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    return "\n".join(lines)


def resolve_inline_data_or_context(text: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process inline JSON/CSV payloads or caller-supplied `caller_data` context,
    honoring explicit JSON, CSV, Markdown table, filtering, aggregation, and extraction."""
    from tools.c0d3rV2.context_tools import extract_inline_csv, extract_inline_json

    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.lower()

    data = extract_inline_json(raw)
    if data is None:
        data = extract_inline_csv(raw)
    caller_data = ctx.get("caller_data") if isinstance(ctx.get("caller_data"), dict) else None

    # A. Inline JSON / CSV processing
    if data is not None:
        # Match requested keys/verbs against the instruction text OUTSIDE the JSON payload
        instr_low = re.sub(r"\{.*\}|\[.*\]", " ", low, flags=re.S)
        if isinstance(data, dict):
            # 1. Explicit key extraction: "extract <key>", "what is <key>", "get <key>"
            for k, v in data.items():
                kl = str(k).lower()
                if re.search(rf"\b(?:extract|get|return|what\s+is|value\s+of|find)\b.*?\b{re.escape(kl)}\b", instr_low):
                    payload = {k: v}
                    ans = json.dumps(payload, ensure_ascii=False) if _want_json(instr_low) else f"`{k}` = **{v}**"
                    return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", f"key:{k}"]}

            # 2. Multiply price * qty / compute total / sum numeric values
            if any(w in instr_low for w in ("total", "sum", "multiply", "average", "mean", "min", "max", "count", "keys")):
                nums = [float(v) for v in data.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if "keys" in low:
                    keys_list = list(data.keys())
                    ans = json.dumps({"keys": keys_list}, ensure_ascii=False) if _want_json(low) else f"Keys: `{', '.join(keys_list)}`"
                    return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "op:keys"]}
                if "price" in data and ("qty" in data or "quantity" in data) and ("total" in low or "multiply" in low):
                    q_val = float(data.get("qty", data.get("quantity", 1)))
                    total = round(float(data["price"]) * q_val, 6)
                    total_clean = int(total) if total.is_integer() else total
                    payload = {"total": total_clean, "price": data["price"], "qty": data.get("qty", data.get("quantity"))}
                    ans = json.dumps(payload, ensure_ascii=False) if _want_json(low) else f"Total = **{total_clean}** (`{data['price']} × {data.get('qty', data.get('quantity'))}`)"
                    return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "op:total"]}
                if nums:
                    if "average" in low or "mean" in low:
                        val = round(sum(nums) / len(nums), 6)
                        op = "average"
                    elif "max" in low:
                        val = max(nums)
                        op = "max"
                    elif "min" in low:
                        val = min(nums)
                        op = "min"
                    elif "count" in low:
                        val = len(data)
                        op = "count"
                    else:
                        val = round(sum(nums), 6)
                        op = "sum"
                    val_c = int(val) if isinstance(val, float) and val.is_integer() else val
                    payload = {op: val_c}
                    ans = json.dumps(payload, ensure_ascii=False) if _want_json(low) else f"{op.capitalize()} = **{val_c}**"
                    return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", f"op:{op}"]}

            # 3. Format conversion on dict
            if _want_json(low):
                return {"question": text, "answer": json.dumps(data, ensure_ascii=False), "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "format:json"]}
            if _want_md_table(low):
                return {"question": text, "answer": _render_table([data]), "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "format:table"]}

        elif isinstance(data, list):
            # Numeric array operations
            if data and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in data):
                nums = [float(x) for x in data]
                if "sort" in low:
                    rev = "desc" in low or "reverse" in low or "largest" in low
                    sorted_v = sorted(data, reverse=rev)
                    ans = json.dumps({"sorted": sorted_v}, ensure_ascii=False) if _want_json(low) else f"Sorted: `{sorted_v}`"
                    return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "op:sort"]}
                if "unique" in low or "dedup" in low:
                    uniq = list(dict.fromkeys(data))
                    ans = json.dumps({"unique": uniq}, ensure_ascii=False) if _want_json(low) else f"Unique: `{uniq}`"
                    return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "op:unique"]}
                if any(w in low for w in ("sum", "total", "average", "mean", "min", "max", "count", "length")):
                    if "average" in low or "mean" in low:
                        v, op = round(sum(nums) / len(nums), 6), "average"
                    elif "max" in low or "largest" in low:
                        v, op = max(nums), "max"
                    elif "min" in low or "smallest" in low:
                        v, op = min(nums), "min"
                    elif "count" in low or "length" in low:
                        v, op = len(nums), "count"
                    else:
                        v, op = round(sum(nums), 6), "sum"
                    vc = int(v) if isinstance(v, float) and v.is_integer() else v
                    ans = json.dumps({op: vc}, ensure_ascii=False) if _want_json(low) else f"{op.capitalize()} = **{vc}**"
                    return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", f"op:{op}"]}

            # List of dicts (records / CSV rows)
            if data and all(isinstance(x, dict) for x in data):
                rows: List[Dict[str, Any]] = list(data)
                # Filter `where field=value` or `status=active` (strictly from instruction outside JSON!)
                filt_m = re.search(r"\b(?:where|filter(?:\s+by)?|with)\s+([a-zA-Z_]\w*)\s*(?:==|=|is)\s*['\"]?([a-zA-Z0-9_.-]+)['\"]?", instr_low)
                if filt_m:
                    fk, fv = filt_m.group(1), filt_m.group(2).lower()
                    rows = [r for r in rows if str(r.get(fk, "")).lower() == fv]
                elif "active" in instr_low and any("active" in r or str(r.get("status", "")).lower() == "active" for r in rows):
                    rows = [r for r in rows if r.get("active") is True or str(r.get("status", "")).lower() == "active"]

                # Sort by field
                sort_m = re.search(r"\b(?:sort|order)\s+by\s+([a-zA-Z_]\w*)", instr_low)
                if sort_m:
                    sk = sort_m.group(1)
                    rev = "desc" in instr_low or "highest" in instr_low
                    def _sk_val(r: Dict[str, Any]):
                        v = r.get(sk, "")
                        try:
                            return (0, float(v))
                        except Exception:
                            return (1, str(v))
                    rows = sorted(rows, key=_sk_val, reverse=rev)

                # Aggregate numeric column if requested
                if any(w in instr_low for w in ("sum", "total", "average", "mean", "count", "how many")):
                    if "count" in instr_low or "how many" in instr_low:
                        payload = {"count": len(rows)}
                        ans = json.dumps(payload, ensure_ascii=False) if _want_json(instr_low) else f"Count = **{len(rows)}**"
                        return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "op:count"]}
                    # Prefer explicitly named column in instruction first, then fallback to non-id numeric column
                    cols_ordered = sorted(
                        list(rows[0].keys()) if rows else [],
                        key=lambda c: (0 if c.lower() in instr_low else 1, 1 if c.lower() == "id" else 0),
                    )
                    for col in cols_ordered:
                        vals = []
                        for r in rows:
                            try:
                                vals.append(float(r[col]))
                            except Exception:
                                pass
                        if len(vals) == len(rows) and vals and (col.lower() in instr_low or len(rows[0]) <= 3):
                            is_avg = "average" in instr_low or "mean" in instr_low
                            res_v = round(sum(vals) / len(vals), 6) if is_avg else round(sum(vals), 6)
                            res_c = int(res_v) if res_v.is_integer() else res_v
                            op = "average" if is_avg else "sum"
                            payload = {"field": col, op: res_c}
                            ans = json.dumps(payload, ensure_ascii=False) if _want_json(instr_low) else f"{op.capitalize()} of `{col}` = **{res_c}**"
                            return {"question": text, "answer": ans, "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", f"op:{op}"]}

                if _want_csv(low):
                    return {"question": text, "answer": _render_csv(rows), "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "format:csv"]}
                if _want_md_table(low):
                    return {"question": text, "answer": _render_table(rows), "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "format:table"]}
                if _want_json(low):
                    return {"question": text, "answer": json.dumps(rows, ensure_ascii=False), "decoder": "data_transform", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform", "format:json"]}
                return {"question": text, "answer": json.dumps(rows, ensure_ascii=False), "decoder": "data_transform", "confidence": 0.95, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["tool:data_transform"]}

    # B. Caller-supplied `caller_data` context (only surfaced when asked!)
    if caller_data:
        for k, v in caller_data.items():
            kl = str(k).lower().replace("_", " ")
            k_raw = str(k).lower()
            if (len(k_raw) >= 3 and (re.search(rf"\b{re.escape(k_raw)}\b", low) or re.search(rf"\b{re.escape(kl)}\b", low))) and any(
                cue in low for cue in ("what", "who", "which", "tell", "give", "show", "extract", "return", "current", "my", "the", "context", "json")
            ):
                payload = {k: v}
                ans = json.dumps(payload, ensure_ascii=False) if _want_json(low) else f"Based on your provided context, `{k}` is **{v}**."
                return {
                    "question": text,
                    "answer": ans,
                    "decoder": "context_data",
                    "confidence": 1.0,
                    "confidence_tier": "high",
                    "hypothesis": False,
                    "activated_concepts": ["context:caller_data", f"key:{k}"],
                }

    # C. Explicit request for a JSON status/context payload
    if _want_json(low) and any(w in low for w in ("status", "context", "health", "clock", "summary")):
        payload = {
            "status": "online",
            "date": ctx["date_iso"],
            "time": ctx["time_24h"],
            "timezone": ctx["timezone"],
            "location": ctx["location"],
        }
        return {
            "question": text,
            "answer": json.dumps(payload, ensure_ascii=False),
            "decoder": "context_json",
            "confidence": 1.0,
            "confidence_tier": "high",
            "hypothesis": False,
            "activated_concepts": ["tool:data_transform", "format:json"],
        }

    return None


# ── 3. Sub-Millisecond Paraphrase Inverted Index (<0.4 ms) ──────────────────

def _stem_token(tok: str) -> str:
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokenize_for_index(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    out = [_stem_token(w) for w in words if w not in _STOPWORDS]
    return out if out else words


class ParaphraseCorpusIndex:
    """In-RAM inverted token index over saved compiled English corpus files.
    Resolves exact and paraphrased queries in <0.4 ms with zero disk I/O after load."""

    _instance: Optional["ParaphraseCorpusIndex"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []
        self._exact: Dict[str, int] = {}
        self._inverted: Dict[str, List[int]] = {}
        self._item_tokens: List[Set[str]] = []
        self._loaded_sig: Tuple[Tuple[str, int], ...] = ()
        self._last_check_ts: float = 0.0

    @classmethod
    def get_instance(cls) -> "ParaphraseCorpusIndex":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _iter_corpus_files(self) -> List[Path]:
        files: List[Path] = []
        for cdir in CORPUS_DIRS:
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.glob("*.jsonl")):
                if p.name in ("weakness_log.jsonl", "settling_cycles.jsonl"):
                    continue
                files.append(p)
        return files

    def insert_live_record(self, rec: Dict[str, Any]) -> None:
        """Incrementally index a newly online-learned record in O(1) RAM (<0.03 ms)
        without re-reading all JSONL files from disk."""
        self.ensure_loaded()
        q = str(rec.get("question") or "").strip()
        a = str(rec.get("answer") or "").strip()
        if not q or not a:
            return
        with self._lock:
            idx = len(self._items)
            self._items.append(rec)
            all_qs = [q] + [str(v).strip() for v in (rec.get("paraphrases") or []) if str(v).strip()]
            tset: Set[str] = set()
            for qv in all_qs:
                ek = re.sub(r"\s+", " ", qv.lower()).rstrip("?!.").strip()
                if ek:
                    self._exact[ek] = idx
                for tok in _tokenize_for_index(qv):
                    tset.add(tok)
            self._item_tokens.append(tset)
            for tok in tset:
                self._inverted.setdefault(tok, []).append(idx)
            self._loaded_sig = self._dir_sig()
            self._last_check_ts = time.time()

    def _dir_sig(self) -> Tuple[Tuple[str, int], ...]:
        sig = []
        for p in self._iter_corpus_files():
            try:
                sig.append((f"{p.parent.name}/{p.name}", p.stat().st_mtime_ns))
            except Exception:
                pass
        return tuple(sig)

    def ensure_loaded(self) -> None:
        now = time.time()
        if self._items and (now - self._last_check_ts) < 10.0:
            return
        self._last_check_ts = now
        sig = self._dir_sig()
        if sig == self._loaded_sig:
            return
        with self._lock:
            if sig == self._loaded_sig:
                return
            items: List[Dict[str, Any]] = []
            exact: Dict[str, int] = {}
            inv: Dict[str, List[int]] = {}
            tok_sets: List[Set[str]] = []
            for p in self._iter_corpus_files():
                try:
                    for ln in p.read_text(encoding="utf-8").splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        rec = json.loads(ln)
                        q = str(rec.get("question") or "").strip()
                        a = str(rec.get("answer") or "").strip()
                        if not q or not a or rec.get("dynamic"):
                            continue
                        idx = len(items)
                        items.append(rec)
                        all_qs = [q] + [str(v).strip() for v in (rec.get("paraphrases") or []) if str(v).strip()]
                        tset: Set[str] = set()
                        for qv in all_qs:
                            ek = re.sub(r"\s+", " ", qv.lower()).rstrip("?!.").strip()
                            if ek and ek not in exact:
                                exact[ek] = idx
                            for tok in _tokenize_for_index(qv):
                                tset.add(tok)
                        tok_sets.append(tset)
                        for tok in tset:
                            inv.setdefault(tok, []).append(idx)
                except Exception:
                    continue
            self._items = items
            self._exact = exact
            self._inverted = inv
            self._item_tokens = tok_sets
            self._loaded_sig = sig

    def query(self, text: str) -> Optional[Dict[str, Any]]:
        self.ensure_loaded()
        if not self._items:
            return None
        low = (text or "").strip().lower()
        ek = re.sub(r"\s+", " ", low).rstrip("?!.").strip()

        # 1. O(1) exact match (<0.02 ms)
        hit_idx = self._exact.get(ek)
        if hit_idx is not None:
            rec = self._items[hit_idx]
            ans = _format_answer_if_requested(str(rec["answer"]), low)
            return {
                "question": text,
                "answer": ans,
                "decoder": "corpus_paraphrase_index",
                "confidence": 0.98,
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": [f"corpus:{rec.get('category', 'english')}", "index:exact"],
            }

        # 2. Inverted-index Jaccard/overlap paraphrase match (<0.3 ms)
        q_toks = set(_tokenize_for_index(text))
        if len(q_toks) < 2:
            return None

        generic_stems = {
            "check", "python", "rust", "typescript", "javascript", "sql",
            "write", "cod", "code", "build", "creat", "create", "implement",
            "function", "modul", "module", "number", "data", "test", "run",
            "verifi", "verify", "solver", "summar", "overview", "archit",
        }
        # Relation words distinguish WHAT property of an entity is asked (e.g. symbol vs atomic number,
        # continent vs capital vs currency), while entity tokens identify WHICH subject is asked!
        relation_frame_stems = {
            "capital", "citi", "city", "countri", "country", "currenc", "currency",
            "continent", "offici", "official", "languag", "language", "system",
            "script", "alphabet", "used", "atom", "atomic", "chemic", "chemical",
            "element", "symbol", "formula", "miner", "mineral", "compound",
            "planet", "moon", "orbit", "parent", "astronom", "astronomical", "bodi", "body",
            "constell", "constellation", "star", "locat", "located", "situat",
            "wrote", "author", "literari", "literary", "work", "novel", "book", "play",
            "compos", "composer", "opera", "music", "musical", "paint", "painted", "creator",
            "direct", "director", "film", "design", "designer", "develop", "developer",
            "program", "programming", "software", "discov", "discovered", "invent", "inventor",
            "mountain", "rang", "range", "peak", "river", "water", "flow", "mouth", "watercours",
            "heritag", "heritage", "site", "space", "agenc", "agency", "oper", "operated", "operator", "probe", "spacecraft",
            "univers", "university", "lake", "museum", "headquart", "headquarters",
            "taxon", "species", "belong", "organ", "organism", "born", "place", "birth", "architect", "building",
        }
        q_entity_toks = q_toks - generic_stems - relation_frame_stems
        q_rel_toks = q_toks & relation_frame_stems
        q_specific = q_toks - generic_stems

        # Single-letter subject (e.g. 'E (programming language)' or 'C') leaves q_entity_toks empty
        # when relation words are present -> require exact match only so 'E' never matches 'Java'!
        if q_rel_toks and not q_entity_toks:
            return None

        # Any unseen entity token means this subject is NOT in local memory yet -> return None for online lookup!
        for et in q_entity_toks:
            if et not in self._inverted:
                return None

        counts: Dict[int, int] = {}
        for tok in q_toks:
            for idx in self._inverted.get(tok, ()):
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return None

        best_idx = -1
        best_score = 0.0
        for idx, overlap in counts.items():
            if overlap < 2:
                continue
            item_t = self._item_tokens[idx]
            # 1. ALL entity tokens with df <= 25 MUST appear in the candidate item
            if q_entity_toks:
                rare_ent = {t for t in q_entity_toks if len(self._inverted.get(t, ())) <= 25}
                if rare_ent and not rare_ent.issubset(item_t):
                    continue
                if (len(q_entity_toks & item_t) / len(q_entity_toks)) < 0.85:
                    continue
            # 2. If relation words are present (e.g. 'symbol' vs 'atomic', 'continent' vs 'currency'),
            #    at least one relation token MUST also match the candidate item!
            if q_rel_toks and not (q_rel_toks & item_t):
                continue
            if q_specific and (len(q_specific & item_t) / len(q_specific)) < 0.65:
                continue
            recall_q = overlap / max(1, len(q_toks))
            best_prec = overlap / max(1, len(item_t))
            score = 0.65 * recall_q + 0.35 * best_prec
            if recall_q >= 0.72 and score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= 0.52:
            rec = self._items[best_idx]
            ans = _format_answer_if_requested(str(rec["answer"]), low)
            return {
                "question": text,
                "answer": ans,
                "decoder": "corpus_paraphrase_index",
                "confidence": round(min(0.95, best_score + 0.15), 3),
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": [f"corpus:{rec.get('category', 'english')}", "index:paraphrase"],
            }
        return None


# ── 4. Online Search + Instant Auto-Learn Resolver ──────────────────────────

_ONLINE_EXPLICIT_RE = re.compile(
    r"^(?:search\s+online\s+for|look\s+up|web\s+search(?:\s+for)?|google|find\s+online|search\s+the\s+web\s+for)\s+",
    re.I,
)


def resolve_online_lookup_if_needed(
    text: str,
    *,
    endpoint: str = "http://127.0.0.1:8090",
    allow_auto_lookup: bool = False,
) -> Optional[Dict[str, Any]]:
    """When W1z4rD does not know an answer locally (or when online search is explicitly
    requested), look it up from strictly open-licensed online sources (Wikidata CC0 /
    Wikipedia CC BY-SA 4.0), persist licensable training material, self-train, and
    track biological settling."""
    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.lower()
    if low == "[settle_online_memory]" or "settle and verify online learned memory" in low:
        from .open_licensed_online_learner import OpenLicensedOnlineLearner
        srec = OpenLicensedOnlineLearner.get_instance().settle_and_verify(reason="post_online_corpus_settle")
        return {
            "question": text,
            "answer": json.dumps(srec, ensure_ascii=False),
            "decoder": "settling_controller",
            "confidence": 1.0,
            "confidence_tier": "high",
            "hypothesis": False,
            "web_used": False,
            "activated_concepts": ["lifecycle:settling_verified"],
        }
    # Never leak or delay internal W1z4rD / cluster / workspace questions on external web APIs
    if any(k in low for k in ("our ", "w1z4rd", "wbrain", "c0d3r", "revenir", "pegasus", "39106", "39105", "8090")):
        return None
    is_explicit = bool(_ONLINE_EXPLICIT_RE.match(raw))
    if not is_explicit and not allow_auto_lookup:
        return None

    # 1. Primary Open-Licensed Coalescing Online Learner (Wikidata CC0 + Wikipedia CC BY-SA 4.0)
    try:
        from .open_licensed_online_learner import OpenLicensedOnlineLearner
        learned = OpenLicensedOnlineLearner.get_instance().lookup_train_and_remember(raw)
        if learned and learned.get("answer"):
            ans = _format_answer_if_requested(
                str(learned["answer"]),
                low,
                {
                    "query": raw,
                    "answer": learned["answer"],
                    "subject": learned.get("subject", ""),
                    "fact_values": learned.get("fact_values", []),
                    "source_url": learned.get("source_url", ""),
                    "license": learned.get("license", ""),
                },
            )
            return {
                "question": text,
                "answer": ans,
                "decoder": "online_licensed_self_trained",
                "confidence": 0.94,
                "confidence_tier": "high",
                "hypothesis": False,
                "web_used": True,
                "license": learned.get("license", ""),
                "source_url": learned.get("source_url", ""),
                "activated_concepts": ["tool:open_licensed_online_learner", "memory:self_trained", f"license:{learned.get('source_name', 'open')}"],
            }
    except Exception:
        pass

    return None
