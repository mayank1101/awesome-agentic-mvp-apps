"""Tests that actually run the Streamlit script.

`AppTest` executes `streamlit_app.py` the way the server does -- top to bottom,
once per rerun -- so these catch the class of bug that unit tests structurally
cannot: a widget built from a value that does not exist, a screen that renders
only when a branch is taken, a caption computed from a field that is `None` on
the path nobody clicked.

File upload is not drivable through `AppTest`, so the input path is covered up to
the upload, and the result screens are driven by seeding session state with a
finished run -- which is exactly the state the pipeline leaves behind.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.models.schemas import (
    JobAssessment,
    JobHit,
    JobRequirement,
    RequirementAssessment,
    RunResult,
    RunSummary,
    ScoreBreakdown,
    ScoredJob,
)

_APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _deep_job() -> ScoredJob:
    assessment = JobAssessment(
        company="Acme",
        title="Senior Backend Engineer",
        location="Bengaluru",
        requirements=[
            JobRequirement(id="R-01", text="Strong Python", must_have=True),
            JobRequirement(id="R-02", text="Kubernetes in production", must_have=True),
            JobRequirement(id="R-03", text="Payments domain", must_have=False),
        ],
        assessments=[
            RequirementAssessment(
                requirement_id="R-01",
                status="covered",
                evidence="Rebuilt the settlement pipeline in Python",
                note="Direct match.",
            ),
            RequirementAssessment(
                requirement_id="R-02", status="missing", note="Not on the resume."
            ),
            RequirementAssessment(requirement_id="R-03", status="partial", note="Adjacent."),
        ],
    )
    return ScoredJob(
        hit=JobHit(
            url="https://boards.greenhouse.io/acme/jobs/123456",
            title="Senior Backend Engineer",
            domain="boards.greenhouse.io",
            snippet="Python, PostgreSQL, payments",
        ),
        tier="deep",
        score=71.0,
        reason="Covers 1 of 2 must-haves. Missing: Kubernetes in production.",
        company="Acme",
        title="Senior Backend Engineer",
        location="Bengaluru",
        assessment=assessment,
        breakdown=ScoreBreakdown(
            must_have_total=2,
            must_have_covered=1,
            nice_to_have_total=1,
            nice_to_have_partial=1,
            must_have_score=50.0,
            nice_to_have_score=50.0,
            demoted=["R-03"],
        ),
        matching_mode="lexical",
    )


def _shallow_job() -> ScoredJob:
    return ScoredJob(
        hit=JobHit(
            url="https://jobs.lever.co/acme/2f1c9a44-1111-2222-3333-444455556666",
            title="Backend Engineer, Platform",
            domain="jobs.lever.co",
            snippet="Go, Kubernetes",
        ),
        tier="shallow",
        score=42.0,
        reason="Ranked on its title and search snippet; the posting was not read.",
        title="Backend Engineer, Platform",
        matching_mode="lexical",
    )


def _run_result(profile) -> RunResult:  # noqa: ANN001 - fixture type comes from conftest
    return RunResult(
        profile=profile,
        jobs=[_deep_job(), _shallow_job()],
        summary=RunSummary(
            queries=["senior backend engineer jobs Bengaluru"],
            sites=["boards.greenhouse.io", "jobs.lever.co"],
            results_found=18,
            results_kept=12,
            deep_scored=1,
            postings_unreadable=1,
            matching_mode="lexical",
            notices=["Ranking and matching used word overlap, not meaning."],
        ),
    )


def _app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    return AppTest.from_file(_APP, default_timeout=30)


def test_missing_keys_show_the_setup_screen() -> None:
    app = AppTest.from_file(_APP, default_timeout=30).run()

    assert app.error
    assert "TAVILY_API_KEY" in app.markdown[0].value or any(
        "TAVILY_API_KEY" in block.value for block in app.markdown
    )


def test_the_form_renders_with_the_default_site_list(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(monkeypatch).run()

    assert not app.exception
    assert app.text_area[0].value.splitlines()[0] == "boards.greenhouse.io"


def test_submitting_without_a_resume_asks_for_one(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(monkeypatch).run()
    app.button[-1].click().run()

    assert any("Upload a resume" in error.value for error in app.error)


def test_a_finished_run_renders_both_tiers(monkeypatch: pytest.MonkeyPatch, profile) -> None:  # noqa: ANN001
    app = _app(monkeypatch)
    app.session_state["result"] = _run_result(profile)
    app.session_state["error"] = None
    app.session_state["notice"] = None
    app.session_state["sites"] = "boards.greenhouse.io"
    app.session_state["runs"] = 1
    app.run()

    assert not app.exception
    rendered = " ".join(block.value for block in app.markdown)
    assert "Senior Backend Engineer" in rendered
    assert "https://boards.greenhouse.io/acme/jobs/123456" in rendered

    captions = " ".join(block.value for block in app.caption)
    assert "Snippet-ranked" in captions
    assert "could not be read" in captions


def test_the_profile_is_shown_with_the_results(monkeypatch: pytest.MonkeyPatch, profile) -> None:  # noqa: ANN001
    app = _app(monkeypatch)
    app.session_state["result"] = _run_result(profile)
    app.session_state["error"] = None
    app.session_state["notice"] = None
    app.session_state["sites"] = "boards.greenhouse.io"
    app.session_state["runs"] = 1
    app.run()

    rendered = " ".join(block.value for block in app.markdown)
    assert "Backend Engineer" in rendered


def test_an_error_state_offers_a_way_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.state import ErrorState

    app = _app(monkeypatch)
    app.session_state["result"] = None
    app.session_state["error"] = ErrorState(
        title="The search did not finish",
        detail="This Tavily key has no search credits left.",
        items=["Resume: looks like an attempt to override the assistant's instructions"],
    )
    app.session_state["notice"] = None
    app.session_state["sites"] = "boards.greenhouse.io"
    app.session_state["runs"] = 0
    app.run()

    assert any("did not finish" in error.value for error in app.error)
    assert app.button

    app.button[0].click().run()
    assert not app.error
