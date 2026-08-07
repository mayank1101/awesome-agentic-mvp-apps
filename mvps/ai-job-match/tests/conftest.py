"""Shared fixtures.

Two things every test in this suite depends on:

**Settings are cached per process**, so a test that changes an environment
variable has to clear that cache or it reads the previous test's configuration.
The autouse fixture below does it on both sides of every test.

**No test may reach the network.** A `.env` on a developer's machine is enough to
turn a unit test into a live API call without anyone noticing -- this repo has
hit that exact bug before. So the keys are blanked by default, and the two
outbound call sites are the ones tests patch explicitly.
"""

import pytest

from app.core.config import Settings, get_settings
from app.services import validation


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean, offline configuration."""
    for name in (
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "MODEL_NAME",
        "STRICT_FABRICATION_GUARD",
        "GUARDRAILS_ENABLED",
        "BLOCK_FLAGGED_INPUT",
        "MAX_REQUIREMENTS",
    ):
        monkeypatch.delenv(name, raising=False)

    # pydantic-settings would otherwise read the developer's own `.env`, which
    # silently turns this suite into a live-API suite the moment someone creates
    # one. Patching the module constant is not enough: `Settings.model_config`
    # captured the path when the class was defined, so the override has to land
    # on that dict.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    validation._guard.cache_clear()
    yield
    get_settings.cache_clear()
    validation._guard.cache_clear()


@pytest.fixture
def resume_text() -> str:
    """A short resume with the facts the provenance tests check against."""
    return (
        "Priya Raman\n"
        "priya.raman@example.com | +91 98765 43210 | Bengaluru\n\n"
        "SUMMARY\n"
        "Backend engineer with 6 years building payment services.\n\n"
        "SKILLS\n"
        "Python, Django, PostgreSQL, Redis, Docker, REST APIs\n\n"
        "EXPERIENCE\n"
        "Senior Backend Engineer, Fintrail\n"
        "Mar 2021 - Present, Bengaluru\n"
        "- Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%\n"
        "- Led a team of 4 engineers across two payment integrations\n"
        "- Designed REST APIs serving 12000 requests per minute\n\n"
        "Backend Engineer, Kite Systems\n"
        "Jul 2018 - Feb 2021, Pune\n"
        "- Built Django services for merchant onboarding\n"
        "- Moved batch jobs onto Docker, halving deploy time\n\n"
        "EDUCATION\n"
        "B.Tech, Computer Science, VIT Vellore, 2018\n"
    )


@pytest.fixture
def job_description() -> str:
    """A posting with a clear must-have / preferred split."""
    return (
        "Senior Backend Engineer at Northwind Pay\n\n"
        "Requirements:\n"
        "- 5+ years of backend engineering experience\n"
        "- Strong Python\n"
        "- Experience designing REST APIs at scale\n"
        "- Kubernetes in production\n\n"
        "Nice to have:\n"
        "- Payments domain experience\n"
        "- Go\n"
    )
