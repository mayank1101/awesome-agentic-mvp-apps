import pytest

from app.models.schemas import JobAssessment, JobRequirement, RequirementAssessment
from app.services import scoring


def test_resume_lines_drop_headings_and_duplicates(resume_text: str) -> None:
    lines = scoring.resume_lines(resume_text)

    assert "SKILLS" not in lines
    assert any("settlement pipeline" in line for line in lines)
    assert len(lines) == len(set(lines))


def test_score_is_arithmetic_over_the_verdicts(assessment: JobAssessment, resume_text: str) -> None:
    score, breakdown, _, _ = scoring.score_assessment(assessment, resume_text)

    # Two of three must-haves covered, one missing, and the single preferred item
    # covered: 66.7 * 0.8 + 100 * 0.2.
    assert breakdown.must_have_total == 3
    assert breakdown.must_have_covered == 2
    assert breakdown.nice_to_have_score == 100.0
    assert score == pytest.approx(73.3, abs=0.2)


def test_a_strong_and_a_weak_resume_score_far_apart(
    assessment: JobAssessment, resume_text: str
) -> None:
    strong, _, _, _ = scoring.score_assessment(assessment, resume_text)

    weak = assessment.model_copy(
        update={
            "assessments": [
                verdict.model_copy(update={"status": "missing", "evidence": ""})
                for verdict in assessment.assessments
            ]
        }
    )
    weak_score, _, _, _ = scoring.score_assessment(weak, resume_text)

    assert strong - weak_score >= 25


def test_invented_evidence_is_demoted(assessment: JobAssessment, resume_text: str) -> None:
    lying = assessment.model_copy(
        update={
            "assessments": [
                assessment.assessments[0].model_copy(
                    update={"evidence": "Ran the Kubernetes platform team at Google for four years"}
                ),
                *assessment.assessments[1:],
            ]
        }
    )

    score, breakdown, checked, _ = scoring.score_assessment(lying, resume_text)

    assert "R-01" in breakdown.demoted
    assert checked.assessments[0].status == "partial"
    assert "does not appear" in checked.assessments[0].note
    assert score < 73.3


def test_evidence_survives_pdf_extraction_artefacts(
    assessment: JobAssessment, resume_text: str
) -> None:
    # A quote that is correct but arrived with kerning splits and a glued glyph.
    spaced = assessment.model_copy(
        update={
            "assessments": [
                assessment.assessments[0].model_copy(
                    update={
                        "evidence": "Backend  engineer with 6 years build ing payment services."
                    }
                ),
                *assessment.assessments[1:],
            ]
        }
    )

    _, breakdown, _, _ = scoring.score_assessment(spaced, resume_text)

    assert "R-01" not in breakdown.demoted


def test_a_requirement_nothing_in_the_resume_touches_is_demoted(resume_text: str) -> None:
    claim = JobAssessment(
        requirements=[
            JobRequirement(
                id="R-01",
                text="FDA submissions for Class III medical devices",
                must_have=True,
                category="domain",
            )
        ],
        assessments=[
            RequirementAssessment(requirement_id="R-01", status="covered", note="Claimed.")
        ],
    )

    _, breakdown, checked, _ = scoring.score_assessment(claim, resume_text)

    assert breakdown.demoted == ["R-01"]
    assert checked.assessments[0].status == "partial"


def test_a_posting_with_no_preferred_section_is_not_penalised(resume_text: str) -> None:
    must_only = JobAssessment(
        requirements=[
            JobRequirement(id="R-01", text="Strong Python", must_have=True, category="hard_skill")
        ],
        assessments=[
            RequirementAssessment(
                requirement_id="R-01",
                status="covered",
                evidence="Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%",
            )
        ],
    )

    score, breakdown, _, _ = scoring.score_assessment(must_only, resume_text)

    assert breakdown.nice_to_have_score is None
    assert score == 100.0


def test_a_posting_with_no_requirements_scores_zero(resume_text: str) -> None:
    score, breakdown, _, _ = scoring.score_assessment(JobAssessment(), resume_text)

    assert score == 0.0
    assert breakdown.must_have_total == 0


def test_explain_names_the_missing_must_haves(assessment: JobAssessment, resume_text: str) -> None:
    _, breakdown, checked, _ = scoring.score_assessment(assessment, resume_text)

    reason = scoring.explain(checked, breakdown)

    assert "Kubernetes" in reason
    assert "2 of 3" in reason


def test_explain_says_so_when_everything_is_covered(resume_text: str) -> None:
    full = JobAssessment(
        requirements=[
            JobRequirement(id="R-01", text="Python", must_have=True, category="hard_skill")
        ],
        assessments=[
            RequirementAssessment(
                requirement_id="R-01",
                status="covered",
                evidence="Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%",
            )
        ],
    )

    _, breakdown, checked, _ = scoring.score_assessment(full, resume_text)

    assert scoring.explain(checked, breakdown) == "Covers all 1 must-haves."
