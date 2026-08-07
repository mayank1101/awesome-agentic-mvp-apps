"""Tests that actually run the Streamlit script.

`AppTest` executes `streamlit_app.py` the way the server does -- top to bottom,
once per rerun -- so these catch the class of bug that unit tests structurally
cannot: a widget built from a value that does not exist, a screen that renders
only when a branch is taken, a download button whose bytes are produced by code
that raises.

File upload is not drivable through `AppTest`, so the input path is covered up to
the upload and the result screens are driven by seeding session state with a
finished analysis -- which is exactly the state the pipeline leaves behind.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.models.schemas import (
    ChangeNote,
    DimensionScore,
    FitReport,
    JobPosting,
    JobRequirement,
    KeywordAction,
    RequirementAssessment,
    ResumeAction,
    ResumeProfile,
    TailoredResume,
)
from app.services.analyzer import AnalysisResult


def _analysis(resume_text: str) -> AnalysisResult:
    report = FitReport(
        overall_score=72,
        band="Good match",
        dimensions=[
            DimensionScore(name="Must-have requirements", earned=80.0, weight=0.6, detail="4 of 5"),
            DimensionScore(name="Keyword coverage", earned=50.0, weight=0.4, detail="2 of 4"),
        ],
        assessments=[
            RequirementAssessment(
                requirement_id="R-01",
                status="covered",
                evidence="Designed REST APIs serving 12000 requests per minute",
                note="Direct match.",
                similarity=0.88,
            ),
            RequirementAssessment(
                requirement_id="R-02", status="missing", note="Not shown.", similarity=0.41
            ),
        ],
        strengths=["Designed REST APIs at scale"],
        gaps=["No Kubernetes experience shown"],
        actions=[
            ResumeAction(
                priority=1,
                section="Summary",
                change="Move the REST API scale line into the first sentence.",
                rationale="R-01 is the posting's first must-have.",
                requirement_ids=["R-01"],
                category="surface",
            ),
            ResumeAction(
                priority=2,
                change="You have no Kubernetes experience; say so in the cover letter.",
                category="gap",
            ),
        ],
        keyword_actions=[
            KeywordAction(keyword="rest apis", supported=True, evidence="Designed REST APIs"),
            KeywordAction(keyword="kubernetes", supported=False),
        ],
        matching_mode="lexical",
        truncated_resume=True,
    )
    return AnalysisResult(
        report=report,
        profile=ResumeProfile(name="Priya Raman"),
        posting=JobPosting(
            title="Senior Backend Engineer",
            company="Northwind Pay",
            requirements=[
                JobRequirement(id="R-01", text="REST APIs at scale", must_have=True),
                JobRequirement(id="R-02", text="Kubernetes in production", must_have=True),
            ],
        ),
        resume_text=resume_text,
    )


#: Relative paths are resolved against *this* file, not the working directory.
_ENTRY_POINT = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _app(**state: object) -> AppTest:
    app = AppTest.from_file(_ENTRY_POINT, default_timeout=30)
    # `SessionStateProxy` forwards attribute access to keys, so `.update(...)`
    # looks for a key called "update". Item assignment is the supported path.
    for key, value in state.items():
        app.session_state[key] = value
    return app


def test_missing_key_shows_the_setup_screen() -> None:
    app = _app().run()

    assert not app.exception
    assert any("not configured" in error.value for error in app.error)


def test_input_form_renders_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    app = _app().run()

    assert not app.exception
    assert app.file_uploader
    assert app.text_area
    assert app.button[0].label == "Analyse fit"


def test_report_screen_renders(monkeypatch: pytest.MonkeyPatch, resume_text: str) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    app = _app(analysis=_analysis(resume_text)).run()

    assert not app.exception
    assert app.metric[0].value == "72/100"
    assert any("Good match" in markdown.value for markdown in app.markdown)
    # The lexical-mode caveat and the truncation notice both have to be visible:
    # they change how much the numbers mean.
    captions = " ".join(caption.value for caption in app.caption)
    assert "embedding key" in captions
    assert "truncated" in captions


def test_report_screen_offers_both_paths(monkeypatch: pytest.MonkeyPatch, resume_text: str) -> None:
    """The candidate chooses; the app must not funnel them into the paid one."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    app = _app(analysis=_analysis(resume_text)).run()

    labels = [button.label for button in app.button]
    assert "Give me the checklist" in labels
    assert "Rewrite it for me" in labels
    assert "Analyse another job" in labels


def test_action_plan_separates_fixable_work_from_real_gaps(
    monkeypatch: pytest.MonkeyPatch, resume_text: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    app = _app(analysis=_analysis(resume_text)).run()

    body = " ".join(m.value for m in app.markdown)
    assert "Move it up" in body
    assert "the evidence is already in your resume" in body
    assert "do not write around them" in body


def test_self_edit_path_offers_a_downloadable_checklist(
    monkeypatch: pytest.MonkeyPatch, resume_text: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    app = _app(analysis=_analysis(resume_text), next_step="self").run()

    assert not app.exception
    assert "Download checklist (Markdown)" in [b.label for b in app.download_button]
    body = " ".join(m.value for m in app.markdown)
    assert "Resume edit checklist" in body
    # The keyword advice has to distinguish what the resume supports from what it
    # does not -- that distinction is the whole point of the section.
    assert "rest apis" in body
    assert "kubernetes" in body


def test_the_self_edit_path_can_switch_to_a_rewrite(
    monkeypatch: pytest.MonkeyPatch, resume_text: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    app = _app(analysis=_analysis(resume_text), next_step="self").run()

    assert "Actually, let AI rewrite it instead" in [b.label for b in app.button]


def test_tailored_screen_renders_both_downloads(
    monkeypatch: pytest.MonkeyPatch, resume_text: str
) -> None:
    """The PDF button's bytes come from a real render -- an exception here is a bug."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    tailored = TailoredResume(
        markdown="# Priya Raman\n## Summary\nBackend engineer.\n",
        changes=[ChangeNote(section="Summary", change="Reordered", reason="Posting leads with it")],
    )
    app = _app(analysis=_analysis(resume_text), tailored=tailored, next_step="ai").run()

    assert not app.exception
    labels = [button.label for button in app.download_button]
    assert "Download Markdown" in labels
    assert "Download PDF" in labels
    assert any("Reordered" in markdown.value for markdown in app.markdown)


def test_error_screen_lists_its_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    from ui.state import ErrorState

    app = _app(
        error=ErrorState(
            title="The rewrite was refused",
            detail="It kept inventing details.",
            items=["“Kubernetes” — name or tool not in your resume"],
        )
    ).run()

    assert not app.exception
    assert any("refused" in error.value for error in app.error)
    assert any("Kubernetes" in markdown.value for markdown in app.markdown)
