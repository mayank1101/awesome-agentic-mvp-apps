"""The sidebar: what produced this ranking, how to take it away, how to start over."""

import streamlit as st

from app.core.config import get_settings
from app.models.schemas import BacklogInput, RankedBacklog
from ui import state
from ui.results import render_exports


def render_sidebar(backlog: BacklogInput | None, ranked: RankedBacklog | None) -> None:
    """Draw the sidebar for this run.

    Args:
        backlog: The current backlog, or None before the first estimate.
        ranked: The current ranking, or None before the first estimate.
    """
    settings = get_settings()

    with st.sidebar:
        st.markdown("### Feature Prioritisation")
        st.caption(
            "RICE and ICE, from rough notes. The model estimates; the scores are arithmetic."
        )

        if ranked is not None and backlog is not None:
            st.divider()
            st.markdown("**Export**")
            st.caption("Both formats carry the factors and the reasoning, not just the scores.")
            render_exports(ranked, backlog)

            st.divider()
            st.button(
                "Start over",
                on_click=state.reset,
                width="stretch",
                icon=":material/restart_alt:",
            )

        st.divider()
        with st.expander("The two frameworks"):
            st.markdown(
                """
**RICE** = Reach × Impact × Confidence ÷ Effort
Reach in absolute counts per quarter, Impact on the 3/2/1/0.5/0.25 scale,
Confidence 100/80/50%, Effort in person-months.

**ICE** = Impact × Confidence × Ease, each 1–10.

Both read **one** factor set here. ICE's Impact and Confidence are mapped from
RICE's, and Ease is a fixed function of Effort — so the same feature can never
be "two months" under one framework and "very easy" under the other.

The trade: ICE is not an independent second opinion. Its value is in what it
**ignores** — there is no Reach term, so a narrow, cheap feature outranks a
broad, costly one under ICE and loses under RICE.
"""
            )

        st.caption(f"`{settings.model_provider}` · `{settings.model_name}`")
        if settings.max_estimations_per_session:
            used = st.session_state[state.ESTIMATIONS]
            st.caption(f"{used}/{settings.max_estimations_per_session} estimates this session")
