"""The ghost entry site refuses a basis the symbol's own book disputes.

Item [d763940a]. The mechanism (services.entry_price_corroboration) and the
threshold were proven in
tests/test_the_aero_contaminated_entry_basis_is_refused.py. This holds the
WIRING, which is the part the item had blocked on for three passes and where the
previous owner's first reading was wrong.

TWO THINGS THIS TEST EXISTS TO PIN, both of them mistakes that were nearly made:

1. The guard sits ABOVE ``_release_position_for_entry``. A refused entry must
   not disturb a position slot -- releasing a slot for a position that is then
   refused is the disarming-stranded-the-live-position class.
2. It is on the GHOST branch only. The live branch books from a SETTLED receipt,
   where money has already left the wallet; refusing there strands a position
   and loses the record of a swap that happened.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "trading" / "bot.py").read_text(encoding="utf-8", errors="replace")


def _ghost_branch() -> str:
    i = SRC.index("# ghost / paper entry")
    return SRC[i:i + 4000]


def test_the_guard_is_wired_at_the_ghost_entry():
    b = _ghost_branch()
    assert "book_disagreement" in b, (
        "the ghost entry books the feed price straight into entry_price; "
        "without this call a contaminated tick becomes the next cost basis")
    assert "entry-refused-implausible-basis" in b


def test_the_guard_runs_before_the_slot_is_released():
    """Order is the whole safety argument -- see the module docstring."""
    b = _ghost_branch()
    guard = b.index("book_disagreement")
    release = b.index("self._release_position_for_entry(")
    assert guard < release, (
        "a refused entry must not have already released or claimed a slot")


def test_the_guard_refuses_by_returning_a_hold_not_by_raising():
    b = _ghost_branch()
    seg = b[b.index("book_disagreement"):b.index("self._release_position_for_entry(")]
    assert '"action": "hold"' in seg or "'action': 'hold'" in seg
    assert "return decision" in seg


def test_a_failure_inside_the_guard_does_not_refuse_the_entry():
    """A gate that blocks everything is a bug, not safety.

    If the import or the database read fails, the price is unjudgeable, not
    contaminated, and the harness must keep gathering evidence.
    """
    seg = _ghost_branch()
    seg = seg[:seg.index("self._release_position_for_entry(")]
    assert "except Exception" in seg
    assert re.search(r'"disagrees":\s*False', seg), (
        "the failure path must fall through to allowed, not to refused")


def test_the_live_booking_site_is_not_guarded_this_way():
    """The live entry books a SETTLED receipt; refusing there strands money."""
    i = SRC.index("# ghost / paper entry")
    live = SRC[max(0, i - 12000):i]
    assert "book_disagreement" not in live, (
        "guarding the live booking site refuses a position whose money has "
        "already left the wallet -- the live check belongs before the swap")
