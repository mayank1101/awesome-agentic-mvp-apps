"""Tests for reconciliation and the score arithmetic.

The score is the app's most load-bearing number, and it is computed here rather
than asked for, so it can be tested exactly.
"""

from app.models.schemas import JobPosting, JobRequirement, RequirementAssessment
from app.services import scoring
from app.services.matching import RequirementMatch


def _requirement(index: int, *, must: bool = True) -> JobRequirement:
    return JobRequirement(id=f"R-{index:02d}", text=f"requirement {index}", must_have=must)


def test_covered_verdict_without_supporting_text_is_downgraded() -> None:
    assessments = [RequirementAssessment(requirement_id="R-01", status="covered", evidence="x")]
    matches = [RequirementMatch(requirement_id="R-01", similarity=0.10)]

    result = scoring.reconcile(assessments, matches, "semantic")

    assert result[0].status == "missing"
    assert result[0].evidence == ""


def test_covered_verdict_with_weak_support_becomes_partial() -> None:
    """Weak in absolute terms *and* barely above the resume's own noise floor."""
    assessments = [RequirementAssessment(requirement_id="R-01", status="covered", evidence="x")]
    matches = [RequirementMatch(requirement_id="R-01", similarity=0.66, baseline=0.61)]

    result = scoring.reconcile(assessments, matches, "semantic")

    assert result[0].status == "partial"
    assert "Downgraded" in result[0].note


def test_strong_support_is_left_alone() -> None:
    assessments = [RequirementAssessment(requirement_id="R-01", status="covered", evidence="x")]
    matches = [RequirementMatch(requirement_id="R-01", similarity=0.82, baseline=0.65)]

    assert scoring.reconcile(assessments, matches, "semantic")[0].status == "covered"


def test_a_modest_score_with_a_clear_margin_survives() -> None:
    """Measured case: a real Django match scored 0.73 against a 0.65 baseline.

    Demoting that would punish a candidate for the embedding model's floor, which
    is exactly what the margin rule exists to prevent.
    """
    assessments = [RequirementAssessment(requirement_id="R-01", status="covered", evidence="x")]
    matches = [RequirementMatch(requirement_id="R-01", similarity=0.734, baseline=0.654)]

    assert scoring.reconcile(assessments, matches, "semantic")[0].status == "covered"


def test_a_gross_mismatch_is_demoted_even_with_a_plausible_score() -> None:
    """Measured case: "Kubernetes" against a frontend resume scored 0.660/0.609."""
    assessments = [RequirementAssessment(requirement_id="R-01", status="covered", evidence="React")]
    matches = [RequirementMatch(requirement_id="R-01", similarity=0.660, baseline=0.609)]

    assert scoring.reconcile(assessments, matches, "semantic")[0].status == "partial"


def test_a_skipped_requirement_counts_as_missing() -> None:
    """Dropping it instead would raise the score by shrinking the denominator."""
    result = scoring.reconcile(
        [], [RequirementMatch(requirement_id="R-01", similarity=0.9)], "semantic"
    )

    assert result[0].status == "missing"
    assert "not met" in result[0].note.lower()


def test_lexical_mode_uses_its_own_floors() -> None:
    """0.35 is solid vocabulary coverage and nowhere near a cosine match."""
    assessments = [RequirementAssessment(requirement_id="R-01", status="covered")]
    matches = [RequirementMatch(requirement_id="R-01", similarity=0.35, baseline=0.02)]

    assert scoring.reconcile(assessments, matches, "lexical")[0].status == "covered"
    assert scoring.reconcile(assessments, matches, "semantic")[0].status == "missing"


def test_weights_renormalise_when_a_dimension_is_empty() -> None:
    posting = JobPosting(requirements=[_requirement(1)], keywords=[])
    assessments = [RequirementAssessment(requirement_id="R-01", status="covered", similarity=0.9)]

    dimensions = scoring.build_dimensions(posting, assessments, "python")

    assert sum(d.weight for d in dimensions) == 1.0
    assert next(d for d in dimensions if d.name == "Keyword coverage").weight == 0.0


def test_full_coverage_scores_high_and_no_coverage_scores_zero() -> None:
    posting = JobPosting(
        requirements=[_requirement(1), _requirement(2, must=False)], keywords=["python"]
    )

    perfect = [
        RequirementAssessment(requirement_id="R-01", status="covered", similarity=1.0),
        RequirementAssessment(requirement_id="R-02", status="covered", similarity=1.0),
    ]
    none = [
        RequirementAssessment(requirement_id="R-01", status="missing", similarity=0.0),
        RequirementAssessment(requirement_id="R-02", status="missing", similarity=0.0),
    ]

    assert scoring.overall_score(scoring.build_dimensions(posting, perfect, "python")) == 100
    assert scoring.overall_score(scoring.build_dimensions(posting, none, "cobol")) == 0


def test_must_haves_outweigh_preferred() -> None:
    posting = JobPosting(requirements=[_requirement(1), _requirement(2, must=False)], keywords=[])

    must_only = [
        RequirementAssessment(requirement_id="R-01", status="covered", similarity=0.9),
        RequirementAssessment(requirement_id="R-02", status="missing"),
    ]
    nice_only = [
        RequirementAssessment(requirement_id="R-01", status="missing"),
        RequirementAssessment(requirement_id="R-02", status="covered", similarity=0.9),
    ]

    assert scoring.overall_score(
        scoring.build_dimensions(posting, must_only, "")
    ) > scoring.overall_score(scoring.build_dimensions(posting, nice_only, ""))


def test_keyword_coverage_counts_multiword_keywords() -> None:
    posting = JobPosting(requirements=[], keywords=["rest apis", "kubernetes"])
    dimensions = scoring.build_dimensions(posting, [], "Designed REST APIs for payments")

    keyword = next(d for d in dimensions if d.name == "Keyword coverage")
    assert keyword.earned == 50.0
    assert "kubernetes" in keyword.detail


def test_bands_cover_the_whole_range() -> None:
    assert scoring.band_for(95) == "Strong match"
    assert scoring.band_for(70) == "Good match"
    assert scoring.band_for(55) == "Partial match"
    assert scoring.band_for(40) == "Weak match"
    assert scoring.band_for(0) == "Poor match"


# --------------------------------------------------------------------------- #
# Keyword advice
#
# The failure mode this replaces: every keyword tool tells the applicant to paste
# the posting's terms in. These tests pin the split between "you can say this
# honestly" and "you cannot".
# --------------------------------------------------------------------------- #


def test_a_keyword_already_in_the_resume_produces_no_action() -> None:
    actions = scoring.keyword_actions(["python"], "Built services in Python", ["Built in Python"])

    assert actions == []


def test_a_keyword_with_adjacent_evidence_is_marked_supported() -> None:
    actions = scoring.keyword_actions(
        ["rest apis"],
        "Designed REST endpoints for payments",
        ["Designed REST endpoints for payments"],
    )

    assert actions[0].supported
    assert actions[0].evidence == "Designed REST endpoints for payments"


def test_a_keyword_with_nothing_behind_it_is_marked_unsupported() -> None:
    actions = scoring.keyword_actions(
        ["kubernetes"],
        "Built Django views and templates",
        ["Built Django views and templates"],
    )

    assert not actions[0].supported
    assert actions[0].evidence == ""


def test_supported_keywords_are_listed_first() -> None:
    actions = scoring.keyword_actions(
        ["kubernetes", "rest apis"],
        "Designed REST endpoints",
        ["Designed REST endpoints"],
    )

    assert [action.keyword for action in actions] == ["rest apis", "kubernetes"]


def test_a_generic_token_alone_does_not_support_a_keyword() -> None:
    """Seen live: "AI professional" was read as support for "ai agents"."""
    actions = scoring.keyword_actions(
        ["ai agents"],
        "Data, Analytics & AI professional with 4 years of experience",
        ["Data, Analytics & AI professional with 4 years of experience"],
    )

    assert not actions[0].supported


def test_a_distinctive_token_does_support_a_keyword() -> None:
    actions = scoring.keyword_actions(
        ["rag architectures"],
        "Built Retrieval-Augmented Generation (RAG) pipelines",
        ["Built Retrieval-Augmented Generation (RAG) pipelines"],
    )

    assert actions[0].supported


def test_wholly_generic_keywords_still_use_overlap() -> None:
    """A keyword of only generic tokens has nothing distinctive; overlap is all there is."""
    actions = scoring.keyword_actions(
        ["product management"],
        "Consulting Data Scientist - Product Owner",
        ["Consulting Data Scientist - Product Owner"],
    )

    assert actions[0].supported
