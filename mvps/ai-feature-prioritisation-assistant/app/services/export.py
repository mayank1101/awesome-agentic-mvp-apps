"""Exporting a ranked backlog to Markdown and CSV.

Both formats carry the **factors and the rationales**, not just the scores. A
score exported alone is unfalsifiable — the person who receives it cannot check
it, argue with it, or reproduce it, which defeats the purpose of having used a
framework at all. The columns that let someone recompute the number by hand are
the export.

Overridden factors are marked in both formats. A number the user chose and a
number the model chose are different kinds of claim, and a spreadsheet that
renders them identically launders one into the other.
"""

import csv
import io

from app.models.schemas import BacklogInput, RankedBacklog, ScoredFeature
from app.services.scales import IMPACT_SCALE

_CSV_COLUMNS = (
    "rice_rank",
    "ice_rank",
    "feature",
    "reach_per_quarter",
    "reach_unit",
    "impact",
    "confidence",
    "effort_person_months",
    "ease",
    "rice_score",
    "ice_score",
    "edited_by_user",
    "reach_rationale",
    "impact_rationale",
    "confidence_rationale",
    "effort_rationale",
    "assumptions",
)


def _factor_note(row: ScoredFeature, factor: str) -> str:
    """Mark a factor the user overrode, so the export never launders an edit."""
    return " (edited)" if factor in row.overridden else ""


def render_markdown(ranked: RankedBacklog, backlog: BacklogInput) -> str:
    """Assemble the full ranking into one Markdown document.

    Args:
        ranked: The scored and ranked backlog.
        backlog: The input it came from, for the product context header.

    Returns:
        Markdown text, ready to download.
    """
    parts: list[str] = ["# Feature prioritisation — RICE and ICE\n"]

    if backlog.product_context:
        parts.append(f"**Product context:** {backlog.product_context}\n")

    parts.append(
        "Scores are computed from the factors in the table: "
        "`RICE = Reach × Impact × Confidence ÷ Effort`, and "
        "`ICE = Impact × Confidence × Ease` on 1–10 scales derived from the same factors. "
        "ICE has no Reach term, which is where the two rankings part company.\n"
    )
    parts.append(
        f"Every Reach figure counts **{ranked.reach_unit} per quarter**. One unit across the whole "
        "list is what makes the column comparable — a row counting blocked deals next to a row "
        "counting seats will invert the ranking, so check the column against this before "
        "circulating it.\n"
    )

    parts.append("## Ranking\n")
    parts.append(
        f"| # | Feature | Reach ({ranked.reach_unit}/qtr) | Impact | Conf. | Effort (pm) | "
        "Ease | RICE | ICE | ICE rank |"
    )
    parts.append("|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for row in sorted(ranked.rows, key=lambda item: item.rice_rank):
        factors = row.factors
        parts.append(
            f"| {row.rice_rank} | {row.idea.title} | {factors.reach:,.0f} | {factors.impact:g} | "
            f"{factors.confidence:.0%} | {factors.effort_months:g} | {factors.ease} | "
            f"{row.rice:,.1f} | {row.ice} | {row.ice_rank} |"
        )
    parts.append("")

    if ranked.divergence:
        parts.append("## Where RICE and ICE disagree\n")
        parts.extend(f"* {note}" for note in ranked.divergence)
        parts.append("")

    parts.append("## Reasoning\n")
    for row in sorted(ranked.rows, key=lambda item: item.rice_rank):
        parts.append(f"### {row.rice_rank}. {row.idea.title}\n")
        if row.idea.notes:
            parts.append(f"> {row.idea.notes}\n")
        factors = row.factors
        parts.append(
            f"* **Reach** — {factors.reach:,.0f} {ranked.reach_unit}/quarter"
            f"{_factor_note(row, 'reach')}. {row.rationales.get('reach', '')}"
        )
        parts.append(
            f"* **Impact** — {factors.impact:g} ({IMPACT_SCALE[factors.impact]})"
            f"{_factor_note(row, 'impact')}. {row.rationales.get('impact', '')}"
        )
        parts.append(
            f"* **Confidence** — {factors.confidence:.0%}"
            f"{_factor_note(row, 'confidence')}. {row.rationales.get('confidence', '')}"
        )
        parts.append(
            f"* **Effort** — {factors.effort_months:g} person-months"
            f"{_factor_note(row, 'effort_months')}. {row.rationales.get('effort_months', '')}"
        )
        if row.assumptions:
            parts.append("* **Assumed:** " + "; ".join(row.assumptions))
        parts.append("")

    if ranked.unestimated:
        parts.append("## Not estimated\n")
        titles = {feature.id: feature.title for feature in backlog.features}
        parts.extend(f"* {titles.get(fid, fid)}" for fid in ranked.unestimated)
        parts.append(
            "\nThe estimator returned no usable factors for these, so they are absent from the "
            "ranking rather than ranked on invented numbers. Re-run the estimate to try again.\n"
        )

    return "\n".join(parts)


def render_csv(ranked: RankedBacklog) -> str:
    """Render the ranking as CSV, one row per feature.

    Every column needed to recompute both scores by hand is present, which is
    what makes the file worth sending to someone who did not run the tool.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    for row in sorted(ranked.rows, key=lambda item: item.rice_rank):
        factors = row.factors
        writer.writerow(
            {
                "rice_rank": row.rice_rank,
                "ice_rank": row.ice_rank,
                "feature": row.idea.title,
                "reach_per_quarter": f"{factors.reach:.0f}",
                "reach_unit": ranked.reach_unit,
                "impact": f"{factors.impact:g}",
                "confidence": f"{factors.confidence:g}",
                "effort_person_months": f"{factors.effort_months:g}",
                "ease": factors.ease,
                "rice_score": f"{row.rice:.2f}",
                "ice_score": row.ice,
                "edited_by_user": " ".join(row.overridden),
                "reach_rationale": row.rationales.get("reach", ""),
                "impact_rationale": row.rationales.get("impact", ""),
                "confidence_rationale": row.rationales.get("confidence", ""),
                "effort_rationale": row.rationales.get("effort_months", ""),
                "assumptions": "; ".join(row.assumptions),
            }
        )

    return buffer.getvalue()
