"""Rendering a graded report to Markdown.

Kept apart from the models so other output formats can sit beside it later without
the schema growing rendering concerns.

Every free-text field goes through :func:`~app.services.guardrails.sanitize_markdown`
on the way out. That is not belt-and-braces here: the quoted `evidence` fields are
model-generated text derived from candidate input, and the exported file is the one
artifact that leaves this app and gets opened somewhere with different rendering
rules.
"""

from app.agents import presets, rubric
from app.models.schemas import FeedbackReport, InterviewConfig
from app.services.guardrails import sanitize_markdown

#: Filled and hollow blocks for the four-point scale. Four segments because the
#: scale has four points -- a percentage bar would imply a precision the rubric
#: does not have.
_FILLED = "█"
_EMPTY = "░"
_MAX_SCORE = 4


def score_meter(score: int) -> str:
    """Render one score as a four-segment meter."""
    filled = max(0, min(_MAX_SCORE, score))
    return _FILLED * filled + _EMPTY * (_MAX_SCORE - filled)


def render_markdown(report: FeedbackReport, config: InterviewConfig) -> str:
    """Assemble a report into one Markdown string, ready to download.

    Args:
        report: The validated report.
        config: The configuration the interview ran under, for the subtitle and
            for ordering scores by this interview type's primary dimensions.

    Returns:
        Markdown text with exactly one top-level heading, so it imports cleanly
        into editors that build a table of contents from heading levels.
    """
    interview = presets.INTERVIEW_TYPE_PRESETS[config.interview_type]
    seniority = presets.SENIORITY_PRESETS[config.seniority]
    archetype = presets.ARCHETYPE_PRESETS[config.archetype]
    primary = presets.primary_dimensions(config.interview_type)

    parts: list[str] = [
        "# Interview report",
        "",
        f"**{interview.label}** · {seniority.label} · {archetype.label}",
        "",
        f"> {_clean(report.headline)}",
        "",
        "## Scores",
        "",
        "| Dimension | | Score | Why |",
        "|:---|:---|:---|:---|",
    ]

    for score in report.ordered_scores(primary):
        dimension = rubric.dimension(score.dimension)
        star = " ⌾" if score.dimension in primary else ""
        parts.append(
            f"| {dimension.label}{star} | `{score_meter(score.score)}` | "
            f"{score.score}/4 | {_clean(score.justification)} |"
        )

    parts += ["", "⌾ marks the dimensions this interview format primarily tests.", ""]

    parts += ["## Evidence", ""]
    for score in report.ordered_scores(primary):
        dimension = rubric.dimension(score.dimension)
        parts.append(f"**{dimension.label} — {score.score}/4** · {_clean(score.evidence)}")
        parts.append("")

    if report.what_worked:
        parts += ["## What worked", ""]
        parts += [f"- {_clean(item)}" for item in report.what_worked]
        parts.append("")

    if report.what_cost_points:
        parts += ["## What cost you points", ""]
        for deduction in report.what_cost_points:
            parts.append(f"- {_clean(deduction.moment)}")
            parts.append(f"  - **Stronger move:** {_clean(deduction.stronger_move)}")
        parts.append("")

    rewrite = report.rewritten_answer
    parts += [
        "## One answer, rewritten",
        "",
        f"**Question:** {_clean(rewrite.question)}",
        "",
        _clean(rewrite.rewrite),
        "",
        f"**Why it is stronger:** {_clean(rewrite.why_better)}",
        "",
        "## Practise next",
        "",
        _clean(report.next_focus),
        "",
    ]

    return "\n".join(parts)


def download_filename(config: InterviewConfig) -> str:
    """A stable, filesystem-safe name for the exported report."""
    return f"interview-report-{config.interview_type}-{config.seniority}.md"


def _clean(text: str) -> str:
    """Sanitise one field, and flatten it so it cannot break a table row.

    The flattening matters for the score table: a newline inside a cell ends the
    row, which would silently mangle every score below it.
    """
    return " ".join(sanitize_markdown(text).split())
