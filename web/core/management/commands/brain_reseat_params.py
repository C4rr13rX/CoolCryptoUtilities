"""brain_reseat_params — push vault-stored brain tier params to the running brain.

Vault is the source of truth for brain tier-orchestrator parameters. The
brain itself reads no env vars on launch; instead, this command pulls
the params from SecureSetting (category="brain") and POSTs them to the
brain's /brain/tier_orchestrator/params endpoint. Idempotent — safe to
call on boot, on a timer, or after a brain restart.

Seeds the vault with sane defaults the first time it runs so the user
doesn't have to hand-author rows. After seeding, edit the SecureSetting
rows in the dashboard to retune.

Usage:
    python manage.py brain_reseat_params              # apply once
    python manage.py brain_reseat_params --watch 60   # repeat every 60s
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from django.core.management.base import BaseCommand
from django.db import transaction

from securevault.models import SecureSetting


BRAIN_CATEGORY = "brain"

# Param name → (vault_key, default_value, type). The user's RAM target
# leans on TIER_TARGET_TERMS — the smaller this is, the smaller the
# brain's working set, with the orchestrator pushing the rest to cold
# tier on SSD per the architecture.
PARAM_SCHEMA: Dict[str, Any] = {
    "target_terminals_per_pool": ("TIER_TARGET_TERMS", 100_000, int),
    "run_every_n_ticks":         ("TIER_RUN_EVERY_N",  5,       int),
    "scan_budget":               ("TIER_SCAN_BUDGET",  2048,    int),
    "max_evict_per_pass":        ("TIER_MAX_EVICT",    1024,    int),
    "evict_threshold":           ("TIER_THRESHOLD",    2.5,     float),
    "w_terminals":               ("TIER_W_TERM",       0.5,     float),
    "w_staleness":               ("TIER_W_STALE",      2.0,     float),
    "w_inverse_salience":        ("TIER_W_INVSAL",     1.0,     float),
    "w_pinned":                  ("TIER_W_PIN",        10000.0, float),
    "decay_horizon_ticks":       ("TIER_DECAY_HORIZON", 1000,   int),
    "page_in_salience_floor":    ("TIER_PAGEIN_FLOOR", 0.0,     float),
    "max_page_in_per_pass":      ("TIER_MAX_PAGEIN",   0,       int),
    "min_age_ticks":             ("TIER_MIN_AGE_TICKS", 1000,   int),
}


def _read_vault() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rows = {
        s.name: s.value_plain
        for s in SecureSetting.objects.filter(category=BRAIN_CATEGORY)
    }
    seeded = False
    with transaction.atomic():
        for param_name, (vault_key, default, caster) in PARAM_SCHEMA.items():
            raw = rows.get(vault_key)
            if raw is None or str(raw).strip() == "":
                SecureSetting.objects.update_or_create(
                    user=None,
                    name=vault_key,
                    category=BRAIN_CATEGORY,
                    defaults={
                        "is_secret": False,
                        "value_plain": str(default),
                    },
                )
                out[param_name] = default
                seeded = True
                continue
            try:
                out[param_name] = caster(str(raw).strip())
            except (TypeError, ValueError):
                out[param_name] = default
    if seeded:
        print(f"[brain_reseat] seeded vault with default brain.* params")
    return out


def _push(params: Dict[str, Any], endpoint: str) -> bool:
    url = f"{endpoint.rstrip('/')}/brain/tier_orchestrator/params"
    payload = json.dumps(params).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"[brain_reseat] applied → {body}")
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[brain_reseat] brain unreachable: {exc}")
        return False


class Command(BaseCommand):
    help = "Push vault-stored brain tier params to the running brain."

    def add_arguments(self, parser):
        parser.add_argument(
            "--endpoint", default="http://127.0.0.1:8090",
            help="Brain HTTP endpoint",
        )
        parser.add_argument(
            "--watch", type=int, default=0,
            help="If > 0, repeat every N seconds.",
        )

    def handle(self, *args, **opts):
        endpoint = opts["endpoint"]
        watch_sec = int(opts["watch"])
        while True:
            params = _read_vault()
            _push(params, endpoint)
            if watch_sec <= 0:
                break
            time.sleep(watch_sec)
