"""The results screen: the ranking, the disagreement, the editor, the reasoning.

Two things about how this module is wired are worth knowing before reading it.

**It does no arithmetic.** Every number here came out of
:func:`app.services.scoring.score_backlog`. The temptation in a Streamlit file is
to compute a percentage or a delta inline; doing that would put a second source
of scores in the codebase, and the claim this app makes is that there is exactly
one.

**Edits are read before anything is drawn.** :func:`collect_editor_overrides`
pulls the deltas out of the editor widget's own state at the top of the run, so
the ranking rendered *above* the editor already reflects the edit made in it.
The alternative -- reading the editor's return value -- would render the table
one interaction behind, which reads as a bug even though the numbers eventually
agree. This is also why the editor's rows are ordered by feature id and never by
rank: the deltas are keyed by row position, so a table that reorders itself
under an edit would apply that edit to the wrong feature.
"""

import streamlit as st

from app.models.schemas import BacklogEstimate, BacklogInput, RankedBacklog
from app.services.export import render_csv, render_markdown
from app.services.scales import CONFIDENCE_SCALE, EFFORT_LADDER, IMPACT_SCALE
from app.services.scoring import attach_levers, score_backlog
from ui import state

#: Editor labels for the two banded factors. A free-text number invites 1.7,
#: which snaps to 2 and looks to the user like the app ignored them; a dropdown
#: of the actual rungs cannot produce that surprise.
_IMPACT_LABELS = {f"{value:g} — {name}": value for value, name in IMPACT_SCALE.items()}
_CONFIDENCE_LABELS = {
    f"{value:.0%} — {name.split(' — ')[1]}": value for value, name in CONFIDENCE_SCALE.items()
}
_EFFORT_LABELS = {f"{value:g}": value for value in EFFORT_LADDER}

_FROM_LABEL = {
    "impact": _IMPACT_LABELS,
    "confidence": _CONFIDENCE_LABELS,
    "effort_months": _EFFORT_LABELS,
}


def _label_for(factor: str, value: float) -> str:
    """The editor label matching a factor value."""
    for label, candidate in _FROM_LABEL[factor].items():
        if candidate == value:
            return label
    return f"{value:g}"


def collect_editor_overrides(backlog: BacklogInput, ranked: RankedBacklog) -> bool:
    """Fold this run's editor deltas into the stored overrides.

    Must run before the ranking is drawn. Returns True when something changed,
    which tells the caller to score once more so the table above the editor
    reflects the edit in the same interaction rather than one behind. Scoring
    twice is free -- it is arithmetic over a handful of rows and touches no
    model.

    Args:
        backlog: The backlog, which fixes the editor's row order.
        ranked: The ranking as it stood before this run's edits, used only to
            know which features have editor rows at all.

    Returns:
        Whether any override was added or changed.
    """
    delta = st.session_state.get(state.editor_key(), {})
    edited_rows = delta.get("edited_rows") if isinstance(delta, dict) else None
    if not edited_rows:
        return False

    scored_ids = {row.idea.id for row in ranked.rows}
    ordered_ids = [feature.id for feature in backlog.features if feature.id in scored_ids]

    changed = False
    for index, columns in edited_rows.items():
        position = int(index)
        if position >= len(ordered_ids):
            continue
        feature_id = ordered_ids[position]
        for factor, value in columns.items():
            if factor == "reach":
                state.set_override(feature_id, "reach", float(value))
                changed = True
            elif factor in _FROM_LABEL:
                mapped = _FROM_LABEL[factor].get(value)
                if mapped is not None:
                    state.set_override(feature_id, factor, mapped)
                    changed = True
    return changed


def _render_divergence(ranked: RankedBacklog) -> None:
    """Show where the two frameworks part company, and which factor did it."""
    if not ranked.divergence:
        return

    st.markdown("##### Where RICE and ICE disagree")
    for note in ranked.divergence:
        st.markdown(f"- {note}")
    st.caption(
        "Both scores read the same factor set, so a disagreement is never about the underlying "
        "estimate — it is about what each formula can see. ICE has no Reach term."
    )


def _titled(row) -> str:  # noqa: ANN001 - ScoredFeature, kept loose for one formatter
    """The feature name with its flags appended.

    The flags used to be two separate icon columns. At nine features and eleven
    columns the table already overflowed a laptop viewport, and two columns whose
    cells are empty on most rows are the cheapest thing to fold away.
    """
    flags = " ✎" if row.overridden else ""
    flags += " ⚠︎" if row.is_low_confidence else ""
    return f"{row.idea.title}{flags}"


def _render_table(ranked: RankedBacklog) -> None:
    """The ranking itself, ordered by RICE."""
    rows = [
        {
            "#": row.rice_rank,
            "Feature": _titled(row),
            "RICE": row.rice,
            "ICE": row.ice,
            "ICE #": row.ice_rank,
            "Move": row.rank_shift,
            "Reach": row.factors.reach,
            "Impact": row.factors.impact,
            # Held as a percentage, not as the 0.8 the formula multiplies by:
            # Streamlit's number format is printf applied to the raw value, so
            # "%.0f%%" over 0.8 renders "1%" on every row -- correct printf,
            # nonsense table. The factor itself is unchanged; only the display
            # unit is.
            "Conf.": row.factors.confidence * 100,
            "Effort": row.factors.effort_months,
            "Ease": row.factors.ease,
        }
        for row in sorted(ranked.rows, key=lambda item: item.rice_rank)
    ]

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Feature": st.column_config.TextColumn("Feature", width="large"),
            "RICE": st.column_config.NumberColumn("RICE", format="%.1f"),
            "ICE": st.column_config.NumberColumn("ICE", format="%d"),
            "ICE #": st.column_config.NumberColumn("ICE #", width="small"),
            "Move": st.column_config.NumberColumn(
                "Move",
                width="small",
                help=(
                    "Places ICE ranks this above RICE. A big positive number is a narrow feature "
                    "that is cheap to build — ICE cannot see the small Reach holding it back."
                ),
            ),
            "Reach": st.column_config.NumberColumn(
                f"Reach ({ranked.reach_unit}/qtr)", format="%,d"
            ),
            "Impact": st.column_config.NumberColumn("Impact", format="%g"),
            "Conf.": st.column_config.NumberColumn("Conf.", format="%.0f%%"),
            "Effort": st.column_config.NumberColumn("Effort (pm)", format="%g"),
            "Ease": st.column_config.NumberColumn("Ease", format="%d", help="Derived from Effort"),
        },
    )
    st.caption(
        f"`RICE = Reach × Impact × Confidence ÷ Effort` · `ICE = Impact × Confidence × Ease`, "
        f"both computed in code from the factors above. Reach counts **{ranked.reach_unit} per "
        "quarter** — check that column reads consistently before you circulate this."
    )
    st.caption("✎ a factor you changed · ⚠︎ low confidence, or resting on stated assumptions")


def _render_editor(backlog: BacklogInput, ranked: RankedBacklog) -> None:
    """The factor editor: disagree with the estimate and watch the order move."""
    scored = {row.idea.id: row for row in ranked.rows}
    rows = [
        {
            "Feature": scored[feature.id].idea.title,
            "reach": scored[feature.id].factors.reach,
            "impact": _label_for("impact", scored[feature.id].factors.impact),
            "confidence": _label_for("confidence", scored[feature.id].factors.confidence),
            "effort_months": _label_for("effort_months", scored[feature.id].factors.effort_months),
        }
        for feature in backlog.features
        if feature.id in scored
    ]

    st.data_editor(
        rows,
        key=state.editor_key(),
        hide_index=True,
        width="stretch",
        column_config={
            "Feature": st.column_config.TextColumn("Feature", disabled=True, width="large"),
            "reach": st.column_config.NumberColumn(
                f"Reach ({ranked.reach_unit}/qtr)", min_value=0, step=10, format="%,d"
            ),
            "impact": st.column_config.SelectboxColumn("Impact", options=list(_IMPACT_LABELS)),
            "confidence": st.column_config.SelectboxColumn(
                "Confidence", options=list(_CONFIDENCE_LABELS)
            ),
            "effort_months": st.column_config.SelectboxColumn(
                "Effort (person-months)", options=list(_EFFORT_LABELS)
            ),
        },
    )
    st.caption(
        "Editing re-ranks instantly and calls no model, so an argument about one Effort estimate "
        "costs nothing. Changed values are attributed to you in the table and in both exports."
    )
    if state.get_overrides():
        st.button(
            "Reset to the estimator's numbers",
            on_click=state.clear_overrides,
            icon=":material/undo:",
        )


def _render_reasoning(ranked: RankedBacklog) -> None:
    """Per-feature detail: what each factor rests on, and what would move it."""
    levers = attach_levers(ranked)

    st.markdown("##### Why each feature scored what it did")
    for row in sorted(ranked.rows, key=lambda item: item.rice_rank):
        flags = " ✎" if row.overridden else ""
        flags += " ⚠︎" if row.is_low_confidence else ""
        with st.expander(f"{row.rice_rank}. {row.idea.title}{flags}"):
            if row.idea.notes:
                st.caption(f"Your note: {row.idea.notes}")

            for factor, label, shown in (
                ("reach", "Reach", f"{row.factors.reach:,.0f} {ranked.reach_unit}/quarter"),
                (
                    "impact",
                    "Impact",
                    f"{row.factors.impact:g} ({IMPACT_SCALE[row.factors.impact]})",
                ),
                ("confidence", "Confidence", f"{row.factors.confidence:.0%}"),
                ("effort_months", "Effort", f"{row.factors.effort_months:g} person-months"),
            ):
                mark = " *(your value)*" if factor in row.overridden else ""
                rationale = row.rationales.get(factor, "")
                note = "" if factor in row.overridden else f" — {rationale}"
                st.markdown(f"**{label}:** {shown}{mark}{note}")

            if row.assumptions:
                st.markdown("**Assumed, because the notes did not say:**")
                for assumption in row.assumptions:
                    st.markdown(f"- {assumption}")

            hint = levers.get(row.idea.id)
            if hint:
                st.info(hint, icon=":material/trending_up:")


def _render_gaps(ranked: RankedBacklog, backlog: BacklogInput) -> None:
    """Name the features the estimator did not cover, rather than hiding them."""
    if not ranked.unestimated:
        return

    titles = {feature.id: feature.title for feature in backlog.features}
    missing = ", ".join(titles.get(fid, fid) for fid in ranked.unestimated)
    st.warning(
        f"No usable estimate came back for: {missing}. They are absent from the ranking rather "
        "than ranked on invented numbers — run it again to try them.",
        icon=":material/report:",
    )


def render_exports(ranked: RankedBacklog, backlog: BacklogInput) -> None:
    """Download buttons, for the sidebar.

    Both formats carry the factors and rationales, not just the scores — a score
    on its own cannot be checked by whoever receives it.
    """
    st.download_button(
        "Markdown",
        data=render_markdown(ranked, backlog),
        file_name="prioritisation.md",
        mime="text/markdown",
        width="stretch",
        icon=":material/download:",
    )
    st.download_button(
        "CSV",
        data=render_csv(ranked),
        file_name="prioritisation.csv",
        mime="text/csv",
        width="stretch",
        icon=":material/download:",
    )


def build_ranking(backlog: BacklogInput, estimate: BacklogEstimate) -> RankedBacklog:
    """Score the stored estimate against the stored overrides.

    Called fresh on every rerun rather than cached: it is pure arithmetic over a
    handful of rows, and a cached ranking is a ranking that can fall out of step
    with the factors on screen.
    """
    return score_backlog(backlog, estimate, overrides=state.get_overrides())


def render_results(backlog: BacklogInput, ranked: RankedBacklog) -> None:
    """Draw the whole results screen."""
    st.markdown(f"#### Ranked backlog — {len(ranked.rows)} features")
    _render_gaps(ranked, backlog)
    _render_divergence(ranked)
    _render_table(ranked)

    with st.expander("Disagree with a factor? Edit it here", expanded=False):
        _render_editor(backlog, ranked)

    _render_reasoning(ranked)
