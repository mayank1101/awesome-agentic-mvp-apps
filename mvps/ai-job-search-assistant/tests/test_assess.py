from typing import Any

import pytest

from app.core.exceptions import ModelRequestTooLarge
from app.models.schemas import JobAssessment, JobHit, JobRequirement, RequirementAssessment
from app.services import assess


def _patch_model(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> list[dict[str, Any]]:
    """Serve canned assessments, recording each call's arguments."""
    calls: list[dict[str, Any]] = []
    queue = list(replies)

    def fake_complete_model(**kwargs: Any) -> JobAssessment:
        calls.append(kwargs)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(assess, "complete_model", fake_complete_model)
    return calls


def test_ids_are_renumbered_and_verdicts_follow(
    monkeypatch: pytest.MonkeyPatch, hit: JobHit
) -> None:
    # Models number things however they like. What matters is that the pairing
    # survives, because the scorer trusts it.
    _patch_model(
        monkeypatch,
        [
            JobAssessment(
                requirements=[
                    JobRequirement(id="1", text="Python", must_have=True),
                    JobRequirement(id="Req-2", text="Kubernetes", must_have=True),
                ],
                assessments=[
                    RequirementAssessment(requirement_id="R2", status="missing"),
                    RequirementAssessment(requirement_id="1", status="covered", evidence="Python"),
                ],
            )
        ],
    )

    result = assess.assess_job(hit, "posting text", "resume text")

    assert [requirement.id for requirement in result.requirements] == ["R-01", "R-02"]
    assert [verdict.requirement_id for verdict in result.assessments] == ["R-01", "R-02"]
    assert result.assessments[0].status == "covered"
    assert result.assessments[1].status == "missing"


def test_a_requirement_with_no_verdict_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch, hit: JobHit
) -> None:
    _patch_model(
        monkeypatch,
        [
            JobAssessment(
                requirements=[JobRequirement(id="R-01", text="Python", must_have=True)],
                assessments=[],
            )
        ],
    )

    result = assess.assess_job(hit, "posting", "resume")

    assert len(result.assessments) == 1
    assert result.assessments[0].status == "missing"


def test_verdicts_for_requirements_that_do_not_exist_are_dropped(
    monkeypatch: pytest.MonkeyPatch, hit: JobHit
) -> None:
    _patch_model(
        monkeypatch,
        [
            JobAssessment(
                requirements=[JobRequirement(id="R-01", text="Python", must_have=True)],
                assessments=[
                    RequirementAssessment(requirement_id="R-01", status="covered"),
                    RequirementAssessment(requirement_id="R-09", status="covered"),
                ],
            )
        ],
    )

    result = assess.assess_job(hit, "posting", "resume")

    assert len(result.assessments) == len(result.requirements) == 1


def test_the_cap_keeps_must_haves(monkeypatch: pytest.MonkeyPatch, hit: JobHit) -> None:
    monkeypatch.setenv("MAX_REQUIREMENTS", "2")
    _patch_model(
        monkeypatch,
        [
            JobAssessment(
                requirements=[
                    JobRequirement(id="R-01", text="Nice one", must_have=False),
                    JobRequirement(id="R-02", text="Nice two", must_have=False),
                    JobRequirement(id="R-03", text="Required one", must_have=True),
                ],
                assessments=[
                    RequirementAssessment(requirement_id=f"R-0{index}", status="missing")
                    for index in (1, 2, 3)
                ],
            )
        ],
    )

    result = assess.assess_job(hit, "posting", "resume")

    assert [requirement.text for requirement in result.requirements] == ["Required one", "Nice one"]


def test_blank_requirements_are_discarded(monkeypatch: pytest.MonkeyPatch, hit: JobHit) -> None:
    _patch_model(
        monkeypatch,
        [
            JobAssessment(
                requirements=[
                    JobRequirement(id="R-01", text="   ", must_have=True),
                    JobRequirement(id="R-02", text="Python", must_have=True),
                ],
                assessments=[
                    RequirementAssessment(requirement_id="R-02", status="covered"),
                ],
            )
        ],
    )

    result = assess.assess_job(hit, "posting", "resume")

    assert [requirement.text for requirement in result.requirements] == ["Python"]


def test_a_too_large_request_is_retried_with_less_posting(
    monkeypatch: pytest.MonkeyPatch, hit: JobHit
) -> None:
    calls = _patch_model(
        monkeypatch,
        [
            ModelRequestTooLarge("6000 token ceiling"),
            JobAssessment(
                requirements=[JobRequirement(id="R-01", text="Python", must_have=True)],
                assessments=[RequirementAssessment(requirement_id="R-01", status="covered")],
            ),
        ],
    )

    posting = "x" * 10_000
    result = assess.assess_job(hit, posting, "resume")

    assert len(calls) == 2
    assert len(calls[1]["user"]) < len(calls[0]["user"])
    assert result.requirements[0].id == "R-01"


def test_both_documents_are_fenced(monkeypatch: pytest.MonkeyPatch, hit: JobHit) -> None:
    calls = _patch_model(monkeypatch, [JobAssessment()])

    assess.assess_job(hit, "posting text", "resume text")

    assert calls[0]["user"].count("<<<UNTRUSTED_DOCUMENT") == 2
