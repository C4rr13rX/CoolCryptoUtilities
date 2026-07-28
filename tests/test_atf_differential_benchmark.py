from services.atf_differential_benchmark import HIDDEN, REFERENCE, evaluate
from tools.c0d3rV2.outline_refiner import OutlineRefiner


def test_all_reference_implementations_pass_hidden_contracts():
    assert set(REFERENCE) == set(HIDDEN)
    for task_id, source in REFERENCE.items():
        result = evaluate(task_id, source)
        assert result["reference_hidden_passed"], (task_id, result)
        assert result["candidate_hidden_passed"], (task_id, result)
        assert result["parity"], (task_id, result)


def test_explicit_class_contract_uses_atomic_refinement_path():
    prompt = (
        "Create the Python class described by contract.json. Write the implementation "
        "to solution.py. The public class must be named Example. Do not change "
        "test_solution.py. Run: test_solution.py."
    )
    assert OutlineRefiner._contract_ready(prompt)
    outline = OutlineRefiner(passes=4).refine(prompt)
    assert outline["quality"]["contract_ready_fast_path"]
    assert outline["quality"]["refinement_passes"] == 4
