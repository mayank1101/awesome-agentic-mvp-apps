import pytest

from app.core.exceptions import EmbeddingError
from app.models.schemas import CandidateProfile, JobHit, SearchFilters
from app.services import ranking


def _hit(title: str, snippet: str = "", score: float = 0.5) -> JobHit:
    return JobHit(
        url=f"https://boards.greenhouse.io/acme/jobs/{abs(hash(title)) % 10**6}",
        title=title,
        domain="boards.greenhouse.io",
        snippet=snippet,
        provider_score=score,
    )


def _fake_embeddings(monkeypatch: pytest.MonkeyPatch, vectors: list[list[float]]) -> None:
    monkeypatch.setattr(ranking, "embed_texts", lambda texts: vectors)


def test_no_hits_ranks_nothing(profile: CandidateProfile) -> None:
    assert ranking.rank_hits(profile, []) == ([], "lexical")


def test_lexical_fallback_when_embeddings_fail(
    monkeypatch: pytest.MonkeyPatch, profile: CandidateProfile
) -> None:
    def boom(texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("no key")

    monkeypatch.setattr(ranking, "embed_texts", boom)

    rankings, mode = ranking.rank_hits(
        profile,
        [_hit("Backend Engineer", "Python Django PostgreSQL")],
    )

    assert mode == "lexical"
    assert rankings[0].score > 0


def test_lexical_ranking_prefers_the_on_topic_job(
    monkeypatch: pytest.MonkeyPatch, profile: CandidateProfile
) -> None:
    monkeypatch.setattr(
        ranking, "embed_texts", lambda texts: (_ for _ in ()).throw(EmbeddingError("offline"))
    )

    rankings, _ = ranking.rank_hits(
        profile,
        [
            _hit("Graphic Designer", "Adobe Illustrator branding print layout"),
            _hit("Backend Engineer", "Python Django PostgreSQL REST APIs payments"),
        ],
    )

    assert rankings[0].hit.title == "Backend Engineer"
    assert rankings[0].score > rankings[1].score


def test_semantic_scores_sit_in_the_shallow_band(
    monkeypatch: pytest.MonkeyPatch, profile: CandidateProfile
) -> None:
    # An unrelated job and a strong one, at cosines either side of the measured
    # floor. The unrelated one must not read as a 60% match just because hosted
    # embeddings never return a low number.
    _fake_embeddings(
        monkeypatch,
        [[1.0, 0.0], [0.58, 0.815], [0.999, 0.045]],
    )

    rankings, mode = ranking.rank_hits(profile, [_hit("Unrelated"), _hit("Backend Engineer")])

    assert mode == "semantic"
    assert rankings[0].hit.title == "Backend Engineer"
    assert rankings[0].score <= 75.0
    assert rankings[-1].score <= 20.0


def test_a_mismatched_seniority_is_nudged_down(
    monkeypatch: pytest.MonkeyPatch, profile: CandidateProfile
) -> None:
    _fake_embeddings(monkeypatch, [[1.0, 0.0], [0.8, 0.6], [0.8, 0.6]])

    rankings, _ = ranking.rank_hits(
        profile,
        [_hit("Junior Backend Engineer"), _hit("Senior Backend Engineer")],
        SearchFilters(seniority="senior"),
    )

    assert rankings[0].hit.title == "Senior Backend Engineer"
    assert rankings[0].score > rankings[1].score


def test_an_adjacent_seniority_is_not_penalised(
    monkeypatch: pytest.MonkeyPatch, profile: CandidateProfile
) -> None:
    _fake_embeddings(monkeypatch, [[1.0, 0.0], [0.8, 0.6], [0.8, 0.6]])

    rankings, _ = ranking.rank_hits(
        profile,
        [_hit("Staff Backend Engineer"), _hit("Senior Backend Engineer")],
        SearchFilters(seniority="senior"),
    )

    assert rankings[0].score == rankings[1].score


def test_an_empty_profile_falls_back_to_provider_order() -> None:
    rankings, mode = ranking.rank_hits(
        CandidateProfile(),
        [_hit("Second", score=0.2), _hit("First", score=0.9)],
    )

    assert mode == "lexical"
    assert [item.hit.title for item in rankings] == ["First", "Second"]
