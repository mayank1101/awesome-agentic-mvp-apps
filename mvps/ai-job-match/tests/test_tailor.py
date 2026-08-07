"""Tests for the rewrite path and its refusal behaviour."""

import pytest

from app.core.config import get_settings
from app.core.exceptions import FabricationDetected
from app.models.schemas import (
    FitReport,
    JobPosting,
    JobRequirement,
    RequirementAssessment,
    ResumeProfile,
    TailoredResumeDraft,
)
from app.services import tailor
from app.services.analyzer import AnalysisResult

_FAITHFUL = (
    "# Priya Raman\n"
    "priya.raman@example.com | Bengaluru\n\n"
    "## Summary\n"
    "Backend engineer with 6 years building payment services in Python.\n\n"
    "## Experience\n"
    "### Senior Backend Engineer, Fintrail\n"
    "*Mar 2021 - Present, Bengaluru*\n"
    "- Designed REST APIs serving 12000 requests per minute\n"
)

_FABRICATED = _FAITHFUL + "- Ran production workloads on Kubernetes across 30 clusters\n"


def _analysis(resume_text: str) -> AnalysisResult:
    report = FitReport(
        overall_score=60,
        band="Partial match",
        dimensions=[],
        assessments=[
            RequirementAssessment(requirement_id="R-01", status="covered", similarity=0.8),
            RequirementAssessment(requirement_id="R-02", status="missing"),
        ],
        suggestions=["Lead with the payments work"],
    )
    posting = JobPosting(
        title="Senior Backend Engineer",
        company="Northwind Pay",
        requirements=[
            JobRequirement(id="R-01", text="Strong Python", must_have=True),
            JobRequirement(id="R-02", text="Kubernetes in production", must_have=True),
        ],
        keywords=["python", "kubernetes"],
    )
    return AnalysisResult(
        report=report,
        profile=ResumeProfile(name="Priya Raman"),
        posting=posting,
        resume_text=resume_text,
    )


def _fake_rewrites(monkeypatch: pytest.MonkeyPatch, *drafts: str) -> list[str]:
    """Serve `drafts` in order, recording each prompt the rewrite sent."""
    prompts: list[str] = []
    replies = iter(drafts)

    def fake_complete_model(*, user: str, **_: object) -> TailoredResumeDraft:
        prompts.append(user)
        return TailoredResumeDraft(markdown=next(replies), changes=[])

    monkeypatch.setattr(tailor, "complete_model", fake_complete_model)
    return prompts


def test_faithful_rewrite_is_returned(resume_text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _fake_rewrites(monkeypatch, _FAITHFUL)

    resume = tailor.tailor_resume(_analysis(resume_text))

    assert "Priya Raman" in resume.markdown
    assert resume.flagged == []
    assert len(prompts) == 1


def test_a_fabricated_rewrite_gets_one_repair_attempt(
    resume_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = _fake_rewrites(monkeypatch, _FABRICATED, _FAITHFUL)

    resume = tailor.tailor_resume(_analysis(resume_text))

    assert len(prompts) == 2
    assert "PREVIOUS ATTEMPT WAS REJECTED" in prompts[1]
    assert "Kubernetes" in prompts[1]
    assert "Kubernetes" not in resume.markdown


def test_strict_mode_refuses_after_two_fabrications(
    resume_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_rewrites(monkeypatch, _FABRICATED, _FABRICATED)

    with pytest.raises(FabricationDetected) as caught:
        tailor.tailor_resume(_analysis(resume_text))

    assert any("Kubernetes" in offender for offender in caught.value.offenders)


def test_non_strict_mode_returns_the_resume_with_flags(
    resume_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRICT_FABRICATION_GUARD", "false")
    get_settings.cache_clear()
    _fake_rewrites(monkeypatch, _FABRICATED, _FABRICATED)

    resume = tailor.tailor_resume(_analysis(resume_text))

    assert resume.flagged
    assert "Kubernetes" in resume.markdown


def test_the_prompt_forbids_the_missing_requirements(
    resume_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap list is sent as a do-not-write list, not as material to write from."""
    prompts = _fake_rewrites(monkeypatch, _FAITHFUL)

    tailor.tailor_resume(_analysis(resume_text))

    assert "must NOT" in prompts[0]
    assert "Kubernetes in production" in prompts[0].split("must NOT")[1]


def test_the_original_resume_is_sent_verbatim(
    resume_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = _fake_rewrites(monkeypatch, _FAITHFUL)

    tailor.tailor_resume(_analysis(resume_text))

    assert "settlement pipeline" in prompts[0]


def test_generated_markdown_is_sanitised(resume_text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rewrites(monkeypatch, "# Priya Raman\n![x](http://evil/p.png)\n", _FAITHFUL)

    resume = tailor.tailor_resume(_analysis(resume_text))

    assert "![" not in resume.markdown


def test_progress_is_reported(resume_text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rewrites(monkeypatch, _FAITHFUL)
    seen: list[str] = []

    tailor.tailor_resume(_analysis(resume_text), progress=seen.append)

    assert seen and "Rewriting" in seen[0]
