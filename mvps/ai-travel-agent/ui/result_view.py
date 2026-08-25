"""The result screen: the itinerary, the sources behind it, and error screens.

Every named place in the itinerary came from a search result the app actually
gathered, never from the model alone -- so the sources list is not a footnote,
it is the evidence the plan can be checked against. It is shown in full,
grouped by category, regardless of which items the model's prose leaned on.
"""

import streamlit as st

from app.core.exceptions import (
    DestinationBlocked,
    ModelQuotaExhausted,
    ModelRateLimited,
    RunDeadlineExceeded,
    SearchAuthError,
    SearchQuotaExhausted,
    TravelAgentError,
)
from app.models.schemas import TripPlan, TripRequest
from app.services.planner import plan_trip
from ui import state as S

_CATEGORY_LABEL = {
    "activities": "Activities",
    "accommodation": "Where to stay",
    "tips": "Practical tips",
}


# --------------------------------------------------------------------------- #
# Running the work
# --------------------------------------------------------------------------- #


def run_plan(request: TripRequest) -> None:
    """Produce one trip plan and store the outcome in session state."""
    if S.busy():
        return

    S.set_busy(True)
    status = st.status("Planning your trip…", expanded=True)
    try:
        result = plan_trip(request, progress=lambda m: status.update(label=m))
    except DestinationBlocked as exc:
        status.update(label="Blocked", state="error")
        S.store_error(
            S.ErrorState(
                title="That request tries to instruct the assistant",
                detail=(
                    "One of the fields contains text aimed at the model rather than at a "
                    "human reader, so the run was stopped. Rephrase and try again."
                ),
                items=[f"{f.field}: {f.pattern}" for f in exc.findings],
            )
        )
    except SearchAuthError as exc:
        status.update(label="Search is not configured", state="error")
        S.store_error(S.ErrorState(title="The search provider rejected the key", detail=str(exc)))
    except SearchQuotaExhausted as exc:
        status.update(label="Search credits exhausted", state="error")
        S.store_error(S.ErrorState(title="Out of search credits", detail=str(exc)))
    except (ModelRateLimited, ModelQuotaExhausted) as exc:
        status.update(label="Provider limit reached", state="error")
        S.store_error(S.ErrorState(title="The model provider is throttling us", detail=str(exc)))
    except RunDeadlineExceeded as exc:
        status.update(label="Timed out", state="error")
        S.store_error(S.ErrorState(title="That took too long", detail=f"{exc} Try a shorter trip."))
    except TravelAgentError as exc:
        status.update(label="Failed", state="error")
        S.store_error(S.ErrorState(title="The plan could not finish", detail=str(exc)))
    else:
        status.update(label="Done", state="complete")
        S.store_plan(result)
    finally:
        S.set_busy(False)


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def render_plan(plan: TripPlan) -> None:
    """Draw the summary, the day-by-day itinerary, and the sources."""
    request = plan.request
    st.subheader(f"{request.destination} — {request.days} day{'s' if request.days != 1 else ''}")

    if plan.synthesis_degraded:
        st.caption(
            "⚠️ The itinerary needed a repair pass to come back in the right shape. "
            "Double-check the day count and details below."
        )

    if plan.summary:
        st.write(plan.summary)

    if plan.accommodation_advice:
        st.write("**Where to stay**")
        st.write(plan.accommodation_advice)

    if plan.practical_tips:
        st.write("**Practical tips**")
        st.write(plan.practical_tips)

    st.divider()
    st.write("**Day by day**")
    for day in plan.itinerary:
        title = f"Day {day.day}" + (f" — {day.title}" if day.title else "")
        with st.expander(title, expanded=True):
            if day.morning:
                st.markdown(f"**Morning** — {day.morning}")
            if day.afternoon:
                st.markdown(f"**Afternoon** — {day.afternoon}")
            if day.evening:
                st.markdown(f"**Evening** — {day.evening}")
            if day.note:
                st.caption(day.note)

    _render_sources(plan)

    st.divider()
    if st.button("Plan another trip", use_container_width=True):
        S.reset()
        st.rerun()


def _render_sources(plan: TripPlan) -> None:
    """Every source the app gathered, grouped by category."""
    if not plan.sources:
        return

    with st.expander(f"Sources ({len(plan.sources)})", expanded=False):
        st.caption(
            "Every named place above is grounded in one of these — general pacing and "
            "advice come from the assistant's own judgement."
        )
        for category, label in _CATEGORY_LABEL.items():
            items = [item for item in plan.sources if item.category == category]
            if not items:
                continue
            st.write(f"**{label}**")
            for item in items:
                st.markdown(f"- [{item.title}]({item.url}) — {item.domain}")


# --------------------------------------------------------------------------- #
# Error screens
# --------------------------------------------------------------------------- #


def render_error(error: S.ErrorState) -> None:
    """Draw a stored error, with a way back to the form."""
    st.error(f"**{error.title}**\n\n{error.detail}")
    if error.items:
        for item in error.items:
            st.markdown(f"- {item}")

    if st.button("Back", use_container_width=True):
        S.clear_error()
        st.rerun()


def render_setup_error(missing: list[str]) -> None:
    """Draw the startup screen for missing credentials."""
    st.error(
        "**This app is not configured yet.**\n\n"
        "Set the following before running it:\n\n" + "\n".join(f"- `{name}`" for name in missing)
    )
    st.caption(
        "Locally, put them in a `.env` beside `streamlit_app.py`. On Streamlit "
        "Community Cloud, put them in the app's Secrets."
    )
