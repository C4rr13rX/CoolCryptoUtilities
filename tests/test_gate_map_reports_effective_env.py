"""The gate map must report the flags actually in force.

Its ENV column printed the table's hardcoded default unconditionally, so it
rendered ``ENABLE_LIVE_TRADING=0`` while the production process was running
with ``1`` -- read straight from that process's environment on 2026-09-02. The
one row that says whether real money can move was reporting the opposite of
the truth, on the tool built to diagnose exactly that, and it cost a pass of
this loop concluding live trading was switched off when it was not.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from scripts.live_gate_map import _effective


class EffectiveEnvTest(unittest.TestCase):
    def test_a_set_variable_is_reported_at_its_real_value(self):
        with mock.patch.dict(os.environ, {"ENABLE_LIVE_TRADING": "1"}, clear=False):
            self.assertEqual(_effective("ENABLE_LIVE_TRADING", "0"), "ENABLE_LIVE_TRADING=1")

    def test_an_unset_variable_is_marked_as_a_default(self):
        env = dict(os.environ)
        env.pop("ENABLE_LIVE_TRADING", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                _effective("ENABLE_LIVE_TRADING", "0"),
                "ENABLE_LIVE_TRADING=0 (default)",
            )

    def test_the_default_is_never_shown_as_the_live_value(self):
        """The exact failure: default 0, real value 1, reported 0."""
        with mock.patch.dict(os.environ, {"ENABLE_LIVE_TRADING": "1"}, clear=False):
            self.assertNotIn("=0", _effective("ENABLE_LIVE_TRADING", "0"))

    def test_an_empty_value_is_not_confused_with_unset(self):
        with mock.patch.dict(os.environ, {"SOME_FLAG": ""}, clear=False):
            self.assertEqual(_effective("SOME_FLAG", "9"), "SOME_FLAG=")

    def test_placeholder_rows_are_left_alone(self):
        """Rows like "(swap_validator)" name code, not an environment variable."""
        self.assertEqual(_effective("(swap_validator)", "-"), "(swap_validator)=-")


if __name__ == "__main__":
    unittest.main()
