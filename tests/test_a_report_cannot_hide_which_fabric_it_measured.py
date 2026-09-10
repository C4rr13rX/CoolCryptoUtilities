"""A brain-experiment report must name the fabric that produced it.

THE FAILURE THIS PREVENTS, and it cost item dcd6d654 three passes. Every
report written to data/brain_experiments/ named its corpus, its windows and
its held-out accuracy, and NOT ONE named the fabric. The pass-107 baseline --
omen-AERO-USDC-h12-20260910-133426.json, held-out exact 0.2850 against a
0.3025 majority -- was the comparison point for every later run, and there was
no way to show it came from a clean node rather than from one that had
inherited a warm checkpoint.

It could not be recovered from the disk either: no brain-data* directory under
W1z4rDV1510n was written between 13:15 and 13:36 that day, so file mtimes do
not identify the run. And it could not be recovered from the node, because
node_id is the HOST's -- production on :8090 and the experiment node on :8091
both answer node-cd4c5a9a7225.

total_neurons is the discriminator that works, because it is a property of the
fabric rather than of the directory name: it catches a freshly-named brain dir
that loaded an old checkpoint just as well as it catches a re-used one.
"""
from __future__ import annotations

import ast
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from scripts.omen_experiment import fabric_census, fabric_is_empty

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "omen_experiment.py"


def test_a_failed_census_does_not_read_as_a_clean_fabric():
    """The guard must not pass when its evidence is missing.

    A census that could not reach the node has no counters. Treating that as
    zero would be a guard that opens precisely when it cannot see -- the shape
    that let a dirty fabric through in the first place.
    """
    assert fabric_is_empty({"endpoint": "http://127.0.0.1:8091",
                            "stats": {"error": "connection refused"}}) is False


def test_a_warm_fabric_is_never_reported_clean():
    warm = {"total_neurons": 41230, "total_concepts": 900,
            "total_binding": 5, "tick": 12}
    assert fabric_is_empty(warm) is False
    # One non-zero counter is enough. A fabric with zero neurons but a
    # non-zero tick has already been driven, and is not a fresh node.
    assert fabric_is_empty({"total_neurons": 0, "total_concepts": 0,
                            "total_binding": 0, "tick": 7}) is False


def test_an_all_zero_fabric_is_clean():
    assert fabric_is_empty({"total_neurons": 0, "total_concepts": 0,
                            "total_binding": 0, "tick": 0}) is True


class _StubNode(BaseHTTPRequestHandler):
    """A node whose id is production's and whose fabric is production-sized.

    This is the exact trap: the id says nothing, so the census must read the
    counters.
    """

    STATS = {"total_neurons": 8_412_003, "total_concepts": 51_884,
             "total_binding": 3, "pool_count": 3, "tick": 990_112}

    def do_GET(self):  # noqa: N802 -- BaseHTTPRequestHandler's spelling
        body = json.dumps(
            {"status": "OK", "node_id": "node-cd4c5a9a7225", "uptime_secs": 90890}
            if self.path == "/health" else self.STATS).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):  # keep pytest output readable
        return


@pytest.fixture()
def stub_node():
    server = HTTPServer(("127.0.0.1", 0), _StubNode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_the_census_reads_the_fabric_not_the_node_id(stub_node):
    census = fabric_census(stub_node)
    assert census["health"]["node_id"] == "node-cd4c5a9a7225"
    assert census["total_neurons"] == 8_412_003
    assert census["pool_count"] == 3
    # The id is production's AND the fabric is production-sized, and it is the
    # second fact that must decide.
    assert fabric_is_empty(census) is False


def _report_keys() -> set[str]:
    """Every key the report dict literal in main() writes."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "report" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)}
    raise AssertionError("no `report = {...}` literal found in main()")


@pytest.mark.parametrize("key", ["node_endpoint", "fabric_before",
                                 "fabric_after", "fabric_was_clean",
                                 "allow_warm_fabric"])
def test_the_report_records_the_fabric_it_measured(key):
    assert key in _report_keys(), (
        f"the report omits {key!r}; a held-out number whose fabric is unknown "
        "cannot be compared to anything")


def test_training_refuses_a_warm_fabric_without_an_explicit_override():
    """Pin the refusal in the source.

    The check runs before a single sample is taught, so there is no artifact
    to assert on afterwards -- a run that trained onto a dirty fabric produces
    a report that looks exactly like a clean one, which is the whole defect.
    """
    src = SOURCE.read_text(encoding="utf-8")
    assert "if not args.skip_train and not fabric_is_empty(fabric_before):" in src
    assert "REFUSING TO TRAIN: this fabric is not clean." in src
    assert "if not args.allow_warm_fabric:" in src
    assert "--allow-warm-fabric" in src
