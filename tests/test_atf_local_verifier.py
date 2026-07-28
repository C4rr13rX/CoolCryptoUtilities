from tools.c0d3rV2.plugins.agent_the_freeloader.local_verifier import LocalCorrectionVerifier


def test_local_correction_verifier_uses_lexical_ranking_by_default(monkeypatch):
    monkeypatch.delenv("ATF_LOCAL_VERIFIER", raising=False)
    verifier = LocalCorrectionVerifier()
    assert verifier._disabled is True
    events = [
        {"classification": "schema", "trigger": "invalid JSON response"},
        {"classification": "tool_failure", "trigger": "TypeScript syntax error"},
    ]
    ranked = verifier.rank("repair TypeScript syntax", events, limit=1)
    assert ranked == [events[1]]


def test_local_embedding_verifier_remains_explicitly_available(monkeypatch):
    monkeypatch.setenv("ATF_LOCAL_VERIFIER", "1")
    assert LocalCorrectionVerifier()._disabled is False
