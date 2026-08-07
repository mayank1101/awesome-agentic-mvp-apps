"""Tests for requirement-to-evidence matching, in both modes."""

import pytest

from app.core.exceptions import EmbeddingError
from app.models.schemas import JobRequirement
from app.services import matching


def test_tokenize_keeps_technical_punctuation() -> None:
    tokens = matching.tokenize("Node.js, CI/CD and scikit-learn")
    assert {"node.js", "ci/cd", "scikit-learn"} <= tokens


def test_tokenize_drops_filler_words() -> None:
    assert matching.tokenize("years of experience with strong skills") == set()


def test_lexical_similarity_is_requirement_coverage() -> None:
    score = matching.lexical_similarity(
        "Strong Python and Django",
        "Built Django services in Python for merchant onboarding",
    )
    assert score == pytest.approx(1.0)


def test_lexical_similarity_ignores_evidence_length() -> None:
    """A long bullet containing every requirement word still satisfies it."""
    short = matching.lexical_similarity("Python", "Python")
    long = matching.lexical_similarity("Python", "x " * 200 + "Python")
    assert short == long == 1.0


def test_lexical_similarity_of_unrelated_text_is_zero() -> None:
    assert matching.lexical_similarity("Kubernetes", "Wrote Django views") == 0.0


def test_cosine_bounds() -> None:
    assert matching.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert matching.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert matching.cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0
    assert matching.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_matching_falls_back_to_lexical_without_a_key() -> None:
    requirements = [JobRequirement(id="R-01", text="Strong Python")]
    matches, mode = matching.match_requirements(requirements, ["Built services in Python"])

    assert mode == "lexical"
    assert matches[0].similarity == pytest.approx(1.0)
    assert matches[0].evidence == ["Built services in Python"]


def test_matching_falls_back_when_embeddings_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """An embedding outage degrades the run; it does not end it."""

    def boom(_: list[str]) -> list[list[float]]:
        raise EmbeddingError("provider down")

    monkeypatch.setattr(matching, "embed_texts", boom)
    matches, mode = matching.match_requirements(
        [JobRequirement(id="R-01", text="Python")], ["Python developer"]
    )

    assert mode == "lexical"
    assert matches[0].similarity > 0


def test_semantic_mode_uses_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str]) -> list[list[float]]:
        # Requirement, then two evidence lines: the second is the close one.
        return [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]][: len(texts)]

    monkeypatch.setattr(matching, "embed_texts", fake_embed)
    matches, mode = matching.match_requirements(
        [JobRequirement(id="R-01", text="Go")], ["unrelated", "shipped Golang services"]
    )

    assert mode == "semantic"
    assert matches[0].evidence[0] == "shipped Golang services"
    assert matches[0].similarity > 0.9


def test_zero_similarity_evidence_is_dropped() -> None:
    matches, _ = matching.match_requirements(
        [JobRequirement(id="R-01", text="Kubernetes")], ["Wrote Django views"]
    )
    assert matches[0].evidence == []
    assert matches[0].similarity == 0.0


def test_empty_inputs_are_handled() -> None:
    assert matching.match_requirements([], ["anything"]) == ([], "lexical")

    matches, _ = matching.match_requirements([JobRequirement(id="R-01", text="Python")], [])
    assert matches[0].similarity == 0.0
