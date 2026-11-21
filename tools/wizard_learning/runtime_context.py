"""tools/wizard_learning/runtime_context.py

Zero-latency (<0.1 ms) live runtime context provider for W1z4rD & C0d3rV2.

Every chat and agent turn receives a structured runtime context envelope:
  - local time (12h & 24h), date, day of week, ISO-8601 timestamp, timezone, UTC offset
  - cached host location (env WIZARD_LOCATION, or background-resolved IP geolocation cached to disk)
  - host OS, Python runtime, workspace path
  - caller-supplied context / inline JSON / CSV / key-value data

Strict Relevance Contract:
  The context envelope is ALWAYS provided to the reasoning/tool layer, but W1z4rD
  NEVER dumps it unprompted — it surfaces time, date, location, or system fields
  only when asked or when directly relevant to the user's request.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GEO_CACHE_FILE = _PROJECT_ROOT / "runtime" / "wizard_location_cache.json"
_GEO_LOCK = threading.Lock()
_GEO_MEM: Optional[Dict[str, Any]] = None
_GEO_FETCH_STARTED = False

# Well-known world timezone offsets (hours from UTC) for instant <0.05ms conversion
WORLD_TZ_OFFSETS: Dict[str, tuple[str, float]] = {
    "utc": ("UTC", 0.0),
    "gmt": ("GMT", 0.0),
    "london": ("Europe/London (GMT/UTC)", 0.0),
    "uk": ("Europe/London (GMT/UTC)", 0.0),
    "paris": ("Europe/Paris (CET)", 1.0),
    "berlin": ("Europe/Berlin (CET)", 1.0),
    "cet": ("CET", 1.0),
    "cairo": ("Africa/Cairo (EET)", 2.0),
    "dubai": ("Asia/Dubai (GST)", 4.0),
    "mumbai": ("Asia/Kolkata (IST)", 5.5),
    "india": ("Asia/Kolkata (IST)", 5.5),
    "ist": ("IST", 5.5),
    "singapore": ("Asia/Singapore (SGT)", 8.0),
    "beijing": ("Asia/Shanghai (CST)", 8.0),
    "shanghai": ("Asia/Shanghai (CST)", 8.0),
    "hong kong": ("Asia/Hong_Kong (HKT)", 8.0),
    "tokyo": ("Asia/Tokyo (JST)", 9.0),
    "japan": ("Asia/Tokyo (JST)", 9.0),
    "jst": ("JST", 9.0),
    "seoul": ("Asia/Seoul (KST)", 9.0),
    "sydney": ("Australia/Sydney (AEDT)", 11.0),
    "auckland": ("Pacific/Auckland (NZDT)", 13.0),
    "new york": ("America/New_York (EST)", -5.0),
    "nyc": ("America/New_York (EST)", -5.0),
    "est": ("EST", -5.0),
    "eastern": ("US Eastern (EST)", -5.0),
    "edt": ("EDT", -4.0),
    "chicago": ("America/Chicago (CST)", -6.0),
    "cst": ("CST", -6.0),
    "central": ("US Central (CST)", -6.0),
    "denver": ("America/Denver (MST)", -7.0),
    "mst": ("MST", -7.0),
    "mountain": ("US Mountain (MST)", -7.0),
    "los angeles": ("America/Los_Angeles (PST)", -8.0),
    "la": ("America/Los_Angeles (PST)", -8.0),
    "san francisco": ("America/Los_Angeles (PST)", -8.0),
    "seattle": ("America/Los_Angeles (PST)", -8.0),
    "pst": ("PST", -8.0),
    "pacific": ("US Pacific (PST)", -8.0),
    "pdt": ("PDT", -7.0),
    "hawaii": ("Pacific/Honolulu (HST)", -10.0),
    "honolulu": ("Pacific/Honolulu (HST)", -10.0),
    "toronto": ("America/Toronto (EST)", -5.0),
    "sao paulo": ("America/Sao_Paulo (BRT)", -3.0),
}


def _load_geo_cache() -> Dict[str, Any]:
    global _GEO_MEM, _GEO_FETCH_STARTED
    if _GEO_MEM is not None:
        return _GEO_MEM

    with _GEO_LOCK:
        if _GEO_MEM is not None:
            return _GEO_MEM
        env_loc = (os.getenv("WIZARD_LOCATION") or "").strip()
        if env_loc:
            _GEO_MEM = {"label": env_loc, "source": "env", "updated_at": time.time()}
            return _GEO_MEM

        if _GEO_CACHE_FILE.is_file():
            try:
                data = json.loads(_GEO_CACHE_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("label"):
                    _GEO_MEM = data
                    return _GEO_MEM
            except Exception:
                pass

        # Default immediately from local timezone without blocking; kick off one-shot background IP-geo lookup
        now_local = datetime.datetime.now().astimezone()
        tz_name = now_local.tzname() or "Local"
        _GEO_MEM = {
            "label": f"Local host ({tz_name})",
            "timezone": tz_name,
            "source": "system_tz",
            "updated_at": time.time(),
        }
        if not _GEO_FETCH_STARTED:
            _GEO_FETCH_STARTED = True
            threading.Thread(target=_bg_refresh_geo, name="wizard-geo-cache", daemon=True).start()
        return _GEO_MEM


def _bg_refresh_geo() -> None:
    global _GEO_MEM
    try:
        req = urllib.request.Request(
            "http://ip-api.com/json/?fields=status,city,regionName,country,timezone",
            headers={"User-Agent": "W1z4rD-Context/1.0"},
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("status") == "success":
                    parts = [p for p in (payload.get("city"), payload.get("regionName"), payload.get("country")) if p]
                    label = ", ".join(parts) if parts else str(payload.get("timezone") or "Local host")
                    geo = {
                        "label": label,
                        "city": payload.get("city") or "",
                        "region": payload.get("regionName") or "",
                        "country": payload.get("country") or "",
                        "timezone": payload.get("timezone") or "",
                        "source": "ip_geo_cached",
                        "updated_at": time.time(),
                    }
                    with _GEO_LOCK:
                        _GEO_MEM = geo
                    try:
                        _GEO_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                        _GEO_CACHE_FILE.write_text(json.dumps(geo, indent=2), encoding="utf-8")
                    except Exception:
                        pass
    except Exception:
        pass


def build_runtime_context(
    *,
    caller_context: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    active_topic: str = "",
) -> Dict[str, Any]:
    """Build a complete, zero-blocking (<0.1ms) live runtime context envelope."""
    now_local = datetime.datetime.now().astimezone()
    now_utc = now_local.astimezone(datetime.timezone.utc)
    offset = now_local.utcoffset() or datetime.timedelta(0)
    total_mins = int(offset.total_seconds() // 60)
    sign = "+" if total_mins >= 0 else "-"
    oh, om = divmod(abs(total_mins), 60)
    utc_offset_str = f"UTC{sign}{oh:02d}:{om:02d}"
    tz_name = now_local.tzname() or utc_offset_str

    geo = dict(_load_geo_cache())
    caller = caller_context if isinstance(caller_context, dict) else {}
    if caller.get("location"):
        geo["label"] = str(caller["location"]).strip()
        geo["source"] = "caller"
    if caller.get("timezone"):
        tz_name = str(caller["timezone"]).strip()

    hour_12 = now_local.strftime("%I").lstrip("0") or "12"
    time_12h = f"{hour_12}:{now_local.strftime('%M %p')}"
    time_24h = now_local.strftime("%H:%M:%S")
    date_human = f"{now_local.strftime('%A, %B')} {now_local.day}, {now_local.year}"
    date_iso = now_local.strftime("%Y-%m-%d")

    ctx: Dict[str, Any] = {
        "iso_timestamp": now_local.isoformat(timespec="seconds"),
        "utc_timestamp": now_utc.isoformat(timespec="seconds"),
        "unix_ts": round(time.time(), 3),
        "time_12h": time_12h,
        "time_24h": time_24h,
        "date_human": date_human,
        "date_iso": date_iso,
        "weekday": now_local.strftime("%A"),
        "month": now_local.strftime("%B"),
        "day": now_local.day,
        "year": now_local.year,
        "timezone": tz_name,
        "utc_offset": utc_offset_str,
        "location": geo.get("label") or f"Local host ({tz_name})",
        "location_detail": geo,
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "workspace": str(_PROJECT_ROOT),
        "session_id": session_id,
        "active_topic": active_topic,
    }
    if caller:
        # Preserve any caller-supplied data/metadata fields for contextual answering
        extra_data = {k: v for k, v in caller.items() if k not in ("location", "timezone")}
        if extra_data:
            ctx["caller_data"] = extra_data
    return ctx


def convert_time_for_zone(zone_key: str) -> Optional[Dict[str, str]]:
    """Convert current UTC time to a named world city or timezone in <0.05ms."""
    k = re.sub(r"\s+", " ", (zone_key or "").strip().lower())
    hit = WORLD_TZ_OFFSETS.get(k)
    if not hit:
        return None
    label, hours = hit
    tz = datetime.timezone(datetime.timedelta(hours=hours))
    dt = datetime.datetime.now(datetime.timezone.utc).astimezone(tz)
    h12 = dt.strftime("%I").lstrip("0") or "12"
    return {
        "zone": label,
        "time_12h": f"{h12}:{dt.strftime('%M %p')}",
        "time_24h": dt.strftime("%H:%M:%S"),
        "date_iso": dt.strftime("%Y-%m-%d"),
        "weekday": dt.strftime("%A"),
        "iso": dt.isoformat(timespec="seconds"),
    }
