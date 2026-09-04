"""Assert what a value IS at the boundaries where money is decided.

Constraint 2 asks every input and output to be checked for type, shape, units,
range and time. Asked as an instruction it is skipped, because the code always
looks right locally -- that is precisely how each of these shipped:

    wrong units    an entry price quoted at the wrong scale reached the gate
    wrong sign     money_button was built on a signal with the sign inverted
    wrong source   an exit sized from a stored quantity rather than the chain,
                   selling 40-60% of the position and stranding the rest
    stale read     usd_amount carried forward from a previous read was used as
                   the current balance, halting a wallet that held 15.196303
    wrong identity a Uniswap v4 pool id used as a token address, spending
                   1.50 USDC on a contract with one byte of code

Every one was locally correct code disagreeing with something across a
boundary. So the check belongs AT the boundary, at runtime, where the value
actually crosses -- not in a review pass.

Design choices that matter:

  * Violations RAISE in tests and on a dry run, and LOG in production. A
    contract bug should stop a test suite, but must not halt live trading on
    an assertion the market never sees -- the downstream guards are still
    there. Set BOUNDARY_STRICT=1 to raise everywhere.
  * Every check names the value, what was expected, and what arrived, because
    "assertion failed" tells you nothing at 3am.
  * These are cheap comparisons on values already in hand. Nothing here calls
    the network.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Iterable, Optional


class BoundaryViolation(ValueError):
    """A value crossing a boundary is not what the other side expects."""


def _strict() -> bool:
    if os.getenv("BOUNDARY_STRICT", "").strip().lower() in {"1", "true", "yes"}:
        return True
    # A test run should fail loudly; production should log and keep trading.
    return "pytest" in sys.modules or "unittest" in sys.modules


def _fail(message: str) -> None:
    if _strict():
        raise BoundaryViolation(message)
    try:
        from services.logging_utils import log_message

        log_message("boundary", message, severity="error")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------- numbers --

def number(value: Any, *, name: str, allow_none: bool = False,
           minimum: Optional[float] = None, maximum: Optional[float] = None,
           allow_zero: bool = True, allow_negative: bool = True) -> Optional[float]:
    """A finite number in range, or a named violation.

    NaN and inf are refused because they propagate silently: a NaN price
    compares False against every threshold, so every guard downstream passes
    it through while nothing about it is true.
    """
    if value is None:
        if allow_none:
            return None
        _fail(f"{name}: expected a number, got None")
        return None

    if isinstance(value, bool):
        # bool is an int in Python, and a True that reaches arithmetic as 1.0
        # is a bug every time.
        _fail(f"{name}: expected a number, got bool {value!r}")
        return None

    try:
        out = float(value)
    except (TypeError, ValueError):
        _fail(f"{name}: expected a number, got {type(value).__name__} {value!r}")
        return None

    if math.isnan(out) or math.isinf(out):
        _fail(f"{name}: expected a finite number, got {out!r}")
        return None
    if not allow_negative and out < 0:
        _fail(f"{name}: must not be negative, got {out!r}")
        return out
    if not allow_zero and out == 0:
        _fail(f"{name}: must not be zero")
        return out
    if minimum is not None and out < minimum:
        _fail(f"{name}: {out!r} is below the minimum {minimum!r}")
        return out
    if maximum is not None and out > maximum:
        _fail(f"{name}: {out!r} is above the maximum {maximum!r}")
        return out
    return out


def usd(value: Any, *, name: str, allow_none: bool = False) -> Optional[float]:
    """A dollar amount, bounded to what this wallet could plausibly hold.

    The upper bound catches a units error rather than a rich wallet: reading
    raw base units as dollars turns 0.75 USDC into 750000, and that number
    passed every "is it positive" check in the system.
    """
    out = number(value, name=name, allow_none=allow_none, allow_negative=True)
    # Bounded by what this wallet could plausibly hold, not by an abstract
    # large number. A 1_000_000 ceiling missed the bug it was written for: the
    # wrong-units price arrived as 750000, which is absurd for a wallet that
    # has never held more than about 25 USDC, and sailed through. Override
    # with BOUNDARY_MAX_USD if the account ever grows past this.
    ceiling = float(os.getenv("BOUNDARY_MAX_USD", "100000"))
    if out is not None and abs(out) > ceiling:
        _fail(f"{name}: {out!r} exceeds the plausible USD ceiling {ceiling!r} "
              f"-- raw base units read as dollars?")
    return out


def fraction(value: Any, *, name: str, allow_none: bool = False) -> Optional[float]:
    """A ratio in 0..1. A percent that arrives here reads as 55.0, not 0.55."""
    out = number(value, name=name, allow_none=allow_none)
    if out is not None and not (0.0 <= out <= 1.0):
        _fail(f"{name}: expected a fraction in 0..1, got {out!r} "
              f"(a percent passed as a fraction?)")
    return out


def token_quantity(value: Any, *, name: str, decimals: Optional[int] = None
                   ) -> Optional[float]:
    """A human-scale token amount, not raw base units.

    An 18-decimal token at raw scale arrives as 1e18-ish. Nothing this wallet
    trades holds that many tokens, so a value that large is a decimals bug.
    """
    out = number(value, name=name, allow_negative=False)
    if out is not None and out > 1e12:
        _fail(f"{name}: {out!r} looks like raw base units, not a token "
              f"quantity (decimals={decimals})")
    return out


# ---------------------------------------------------------------- shapes --

def mapping(value: Any, *, name: str, required: Iterable[str] = ()) -> dict:
    """A dict with the keys the caller will actually read."""
    if not isinstance(value, dict):
        _fail(f"{name}: expected a dict, got {type(value).__name__} {value!r}")
        return {}
    missing = [k for k in required if k not in value]
    if missing:
        _fail(f"{name}: missing required key(s) {missing}; has {sorted(value)}")
    return value


def sequence(value: Any, *, name: str, allow_empty: bool = True) -> list:
    """A list, never a bare string.

    A str is iterable, so a function returning "USDC" where a list of symbols
    was expected iterates as ['U','S','D','C'] and every consumer downstream
    is quietly wrong.
    """
    if isinstance(value, (str, bytes)):
        _fail(f"{name}: expected a sequence, got a string {value!r} -- "
              f"iterating it yields characters")
        return []
    if value is None:
        _fail(f"{name}: expected a sequence, got None")
        return []
    try:
        out = list(value)
    except TypeError:
        _fail(f"{name}: expected a sequence, got {type(value).__name__}")
        return []
    if not allow_empty and not out:
        _fail(f"{name}: must not be empty")
    return out


# ------------------------------------------------------------------ time --

def epoch_seconds(value: Any, *, name: str, allow_none: bool = False
                  ) -> Optional[float]:
    """A timestamp in SECONDS. Milliseconds arrive ~1000x too large.

    A ms timestamp read as seconds lands in the year 58000, so every "is this
    recent?" test says yes and every staleness guard is disabled.
    """
    out = number(value, name=name, allow_none=allow_none, allow_negative=False)
    if out is None or out == 0:
        return out
    # Anything past ~2100 in seconds is a millisecond value.
    if out > 4_102_444_800:
        _fail(f"{name}: {out!r} is milliseconds, not epoch seconds")
    # Anything before 2001 is not a timestamp we produce.
    elif out < 1_000_000_000:
        _fail(f"{name}: {out!r} is too small to be an epoch-seconds timestamp")
    return out


# --------------------------------------------------------------- identity --

def address(value: Any, *, name: str, allow_native: bool = True) -> Optional[str]:
    """A 42-character hex address, checked for shape only.

    Whether the address is a real contract is a separate, on-chain question --
    see services/token_contract_guard.py. This catches the shape errors: a
    symbol passed where an address was expected, or a 32-byte pool id.
    """
    if value is None:
        _fail(f"{name}: expected an address, got None")
        return None
    text = str(value).strip()
    if allow_native and text.lower() in {"native", "eth", ""}:
        return text
    if not text.startswith("0x"):
        _fail(f"{name}: expected a 0x address, got {text!r} "
              f"(a symbol passed where an address belongs?)")
        return None
    if len(text) != 42:
        _fail(f"{name}: expected a 42-char address, got {len(text)} chars "
              f"{text!r} (a 66-char value is a tx hash or a v4 pool id)")
        return None
    return text
