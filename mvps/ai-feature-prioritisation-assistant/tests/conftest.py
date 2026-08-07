"""Shared fixtures and builders.

Two things every test in this suite depends on:

**Settings are isolated from the developer's own `.env`.** `Settings` reads one
by default, so a machine with `MAX_FEATURES=5` in its `.env` would fail the
parser tests for reasons that have nothing to do with the code. The fixture
patches `Settings.model_config` rather than the module-level constant, because
pydantic-settings reads the config off the class at instantiation time and a
patched constant is simply ignored.

**Estimates are built here, not inline.** Almost every scoring test needs a
backlog plus a matching estimate, and building both by hand in each test buries
the one factor the test is actually about.
"""

import pytest

from app.core.config import Settings, get_settings
from app.models.schemas import BacklogEstimate, BacklogInput, FeatureEstimate, FeatureIdea


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test against default settings, ignoring any local `.env`."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in (
        "MODEL_PROVIDER",
        "MODEL_NAME",
        "MAX_FEATURES",
        "MAX_FEATURE_CHARS",
        "MAX_CONTEXT_CHARS",
        "GUARDRAILS_ENABLED",
        "BLOCK_FLAGGED_INPUT",
        "MAX_ESTIMATIONS_PER_SESSION",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_backlog(
    *titles: str, context: str = "B2B SaaS, 4,000 accounts, 8 engineers."
) -> BacklogInput:
    """Build a backlog of bare-titled features with sequential ids."""
    return BacklogInput(
        features=[
            FeatureIdea(id=f"F{index}", title=title, notes="")
            for index, title in enumerate(titles, start=1)
        ],
        product_context=context,
    )


def make_estimate(
    feature_id: str,
    *,
    reach: float = 1000,
    impact: float = 1.0,
    confidence: float = 0.8,
    effort_months: float = 1.0,
    assumptions: list[str] | None = None,
) -> FeatureEstimate:
    """Build one estimate with plausible rationales and overridable factors."""
    return FeatureEstimate(
        id=feature_id,
        reach=reach,
        reach_rationale=f"{feature_id}: reach from the stated account base",
        impact=impact,
        impact_rationale=f"{feature_id}: impact from the note",
        confidence=confidence,
        confidence_rationale=f"{feature_id}: evidence in the note",
        effort_months=effort_months,
        effort_rationale=f"{feature_id}: size from the team context",
        assumptions=assumptions or [],
    )


def make_backlog_estimate(*estimates: FeatureEstimate) -> BacklogEstimate:
    """Wrap estimates into the reply object the scorer consumes."""
    return BacklogEstimate(estimates=list(estimates))
