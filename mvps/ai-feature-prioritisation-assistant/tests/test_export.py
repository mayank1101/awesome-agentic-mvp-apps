"""Tests for the Markdown and CSV exports.

The property worth testing is not formatting but *recomputability*: someone who
receives the file must be able to check the score by hand. So the tests read the
factors back out of the CSV and run the formula on them.
"""

import csv
import io

from app.services.export import render_csv, render_markdown
from app.services.scales import rice_score
from app.services.scoring import score_backlog
from tests.conftest import make_backlog, make_backlog_estimate, make_estimate


def _ranked():
    backlog = make_backlog("Bulk export", "Dark mode", "SSO")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=3000, impact=2, confidence=1.0, effort_months=2),
        make_estimate("F2", reach=4000, impact=0.25, confidence=0.5, effort_months=0.5),
        make_estimate(
            "F3",
            reach=90,
            impact=3,
            confidence=0.8,
            effort_months=6,
            assumptions=["assumed 3 deals"],
        ),
    )
    return backlog, estimate


def test_csv_rows_can_be_recomputed_by_hand():
    backlog, estimate = _ranked()
    ranked = score_backlog(backlog, estimate)

    rows = list(csv.DictReader(io.StringIO(render_csv(ranked))))

    assert len(rows) == 3
    for row in rows:
        recomputed = rice_score(
            float(row["reach_per_quarter"]),
            float(row["impact"]),
            float(row["confidence"]),
            float(row["effort_person_months"]),
        )
        assert recomputed == float(row["rice_score"])


def test_csv_is_ordered_by_rice_rank():
    backlog, estimate = _ranked()
    ranked = score_backlog(backlog, estimate)

    rows = list(csv.DictReader(io.StringIO(render_csv(ranked))))

    assert [row["rice_rank"] for row in rows] == ["1", "2", "3"]


def test_csv_marks_which_factors_the_user_edited():
    backlog, estimate = _ranked()
    ranked = score_backlog(backlog, estimate, overrides={"F1": {"effort_months": 0.5}})

    rows = {row["feature"]: row for row in csv.DictReader(io.StringIO(render_csv(ranked)))}

    assert rows["Bulk export"]["edited_by_user"] == "effort_months"
    assert rows["Dark mode"]["edited_by_user"] == ""


def test_markdown_carries_the_rationales_not_just_the_scores():
    backlog, estimate = _ranked()
    ranked = score_backlog(backlog, estimate)

    document = render_markdown(ranked, backlog)

    assert "## Ranking" in document
    assert "## Reasoning" in document
    assert "reach from the stated account base" in document
    assert "assumed 3 deals" in document


def test_markdown_states_the_formulas_it_used():
    backlog, estimate = _ranked()

    document = render_markdown(score_backlog(backlog, estimate), backlog)

    assert "Reach × Impact × Confidence ÷ Effort" in document
    assert "no Reach term" in document


def test_markdown_marks_edited_factors():
    backlog, estimate = _ranked()
    ranked = score_backlog(backlog, estimate, overrides={"F2": {"impact": 3}})

    document = render_markdown(ranked, backlog)

    assert "(edited)" in document


def test_markdown_lists_unestimated_features_rather_than_hiding_them():
    backlog, _ = _ranked()
    partial = make_backlog_estimate(make_estimate("F1"))

    document = render_markdown(score_backlog(backlog, partial), backlog)

    assert "## Not estimated" in document
    assert "Dark mode" in document
    assert "SSO" in document


def test_markdown_includes_the_divergence_section_when_they_disagree():
    backlog = make_backlog("Broad", "Narrow-but-easy", "Mid", "Filler")
    estimate = make_backlog_estimate(
        make_estimate("F1", reach=90_000, impact=1, confidence=0.8, effort_months=12),
        make_estimate("F2", reach=15, impact=3, confidence=1.0, effort_months=0.25),
        make_estimate("F3", reach=2000, impact=2, confidence=0.8, effort_months=2),
        make_estimate("F4", reach=1500, impact=1, confidence=0.8, effort_months=3),
    )

    document = render_markdown(score_backlog(backlog, estimate), backlog)

    assert "## Where RICE and ICE disagree" in document
