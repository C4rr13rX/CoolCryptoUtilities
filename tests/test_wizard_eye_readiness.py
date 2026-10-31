from web.wizard_chat.eye_readiness import assess_prerogatives, probe_eye_readiness


def test_existing_nodes_do_not_require_a_new_manifest_protocol():
    normal = {"core_prerogatives": {"framed_by": "The Lord said this", "compliant": True}}
    refusal = {"core_prerogatives": {"status": "refused", "compliant": False}}
    report = assess_prerogatives(normal, refusal)
    assert report["prerogative_probe_passed"] is True
    assert report["training_ready"] is True
    assert report["twelve_manifest_present"] is False
    assert "does not establish full twelve" in report["verification_scope"]


def test_duplicate_ids_cannot_impersonate_twelve_prerogatives():
    normal = {"core_prerogatives": {"framed_by": "frame", "compliant": True,
              "manifest": {"prerogatives": [{"id": "one", "text": "rule"}] * 12}}}
    assert assess_prerogatives(normal, {})["twelve_manifest_present"] is False


def test_readiness_probes_never_call_learning_endpoints():
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, *_): return b'{}'
    def opener(request, timeout):
        calls.append((request.full_url, request.get_method()))
        return Response()
    report = probe_eye_readiness("http://localhost:8090", "/brain/chat", opener=opener)
    assert [path for path, _ in calls] == ["http://localhost:8090/health", "http://localhost:8090/brain/chat", "http://localhost:8090/brain/chat"]
    assert report["training_ready"] is False
