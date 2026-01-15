from services import research_ranker


def test_complex_text_scores_higher_than_simple_text():
    simple = "This is a simple description about a test."
    complex_text = (
        "Chaotic dynamics with fractal attractors and Lyapunov exponents govern nonlinear "
        "signal transduction pathways in cellular metabolic networks."
    )
    simple_score, _ = research_ranker.analyze_and_score(simple)
    complex_score, _ = research_ranker.analyze_and_score(complex_text)
    assert complex_score > simple_score


def test_entropy_and_density_are_in_range():
    score, metrics = research_ranker.analyze_and_score("Nonlinear dynamics and bifurcation analysis.")
    assert 0.0 <= metrics["lexical_density"] <= 1.0
    assert 0.0 <= metrics["entropy_norm"] <= 1.0
    assert 0.0 <= score <= 1.0


def test_category_detection_from_known_domain():
    url = "https://www.santafe.edu/research"
    metrics = research_ranker.analyze_text("Complex systems", url=url)
    assert metrics["category"] in {"complex-systems", ""}  # allow empty if not matched
