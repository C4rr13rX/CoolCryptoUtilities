"""A ``--endpoint 127.0.0.1:8092`` must reach :8092, not the default node.

THE FAILURE THIS PREVENTS. ``urlparse("127.0.0.1:8092")`` parses the whole
string as a PATH: ``hostname`` is None and ``port`` is None, because there is
no scheme to mark the authority. Both node clients then did
``u.hostname or "127.0.0.1"`` and ``u.port or <their own default>``, so a
caller that explicitly named port 8092 was silently pointed at 8091 (omen) or
8090 (production). Nothing failed loudly: the client connected, the fabric
census used the same resolution, and the experiment reported a
self-consistent set of numbers about the WRONG BRAIN -- including, on
:8090, production's own fabric.

Measured in pass 117 (Cove): the shape-mutation arm aimed at a fresh node on
:8092 reported ``FAIL: node has no /brain/predict/multi -- stale binary or
port`` while :8092 answered that very route with a 200. It was talking to
:8091, where nothing was listening. Had a node been up there, it would have
trained it instead and said nothing.
"""
from __future__ import annotations

import pytest

from trading.brain_bridge import BrainBridge, resolve_node_endpoint
from trading.omen_brain import OmenBrain, resolve_omen_endpoint


@pytest.mark.parametrize("target", ["127.0.0.1:8092", ":8092", "8092",
                                    "http://127.0.0.1:8092"])
def test_a_named_port_is_honoured_however_it_is_spelled(target):
    """Every spelling that NAMES 8092 must resolve to 8092."""
    assert resolve_omen_endpoint(target) == ("127.0.0.1", 8092)
    assert resolve_node_endpoint(target, 8090) == ("127.0.0.1", 8092)


def test_the_clients_themselves_connect_to_the_named_port():
    """The resolver is not enough -- the clients must actually use it."""
    omen = OmenBrain(endpoint="127.0.0.1:8092")
    assert (omen._host, omen._port) == ("127.0.0.1", 8092)
    bridge = BrainBridge(endpoint="127.0.0.1:8092")
    assert (bridge._host, bridge._port) == ("127.0.0.1", 8092)


def test_a_host_is_carried_through_without_a_scheme():
    assert resolve_node_endpoint("localhost:8094", 8090) == ("localhost", 8094)


def test_only_a_target_naming_no_port_falls_back_to_the_default():
    """The fallback still exists; it just stops eating an explicit port."""
    assert resolve_omen_endpoint(None) == ("127.0.0.1", 8091)
    assert resolve_omen_endpoint("") == ("127.0.0.1", 8091)
    assert resolve_omen_endpoint("http://127.0.0.1") == ("127.0.0.1", 8091)
    assert resolve_node_endpoint(None, 8090) == ("127.0.0.1", 8090)
    assert resolve_node_endpoint("http://127.0.0.1", 8090) == ("127.0.0.1", 8090)


def test_production_is_not_moved_by_a_well_formed_endpoint():
    """The live client's own default must be unchanged by this fix."""
    bridge = BrainBridge(endpoint="http://127.0.0.1:8090")
    assert (bridge._host, bridge._port) == ("127.0.0.1", 8090)
