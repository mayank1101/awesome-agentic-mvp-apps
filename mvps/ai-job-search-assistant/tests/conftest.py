"""Shared fixtures.

Two things every test in this suite depends on:

**Settings are cached per process**, so a test that changes an environment
variable has to clear that cache or it reads the previous test's configuration.
The autouse fixture below does it on both sides of every test.

**No test may reach the network.** A `.env` on a developer's machine is enough to
turn a unit test into a live API call without anyone noticing -- this repo has
hit that exact bug before, and this app has three outbound call sites rather than
two. So every key is blanked by default, and each call site is patched explicitly
by the tests that exercise it.
"""

import pytest

from app.core.config import Settings, get_settings
from app.models.schemas import (
    CandidateProfile,
    JobAssessment,
    JobHit,
    JobRequirement,
    RequirementAssessment,
    SearchFilters,
)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean, offline configuration."""
    for name in (
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "MISTRAL_API_KEY",
        "MODEL_NAME",
        "GUARDRAILS_ENABLED",
        "BLOCK_FLAGGED_INPUT",
        "DEEP_SCORE_COUNT",
        "MAX_REQUIREMENTS",
        "MAX_RESULTS_TOTAL",
        "RUN_DEADLINE_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    # pydantic-settings would otherwise read the developer's own `.env`, which
    # silently turns this suite into a live-API suite the moment someone creates
    # one. Patching the module constant is not enough: `Settings.model_config`
    # captured the path when the class was defined, so the override has to land
    # on that dict.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def resume_text() -> str:
    """A short resume with facts the scoring tests check against."""
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
def profile() -> CandidateProfile:
    """The profile that resume would produce."""
    return CandidateProfile(
        titles=["Backend Engineer", "Payments Engineer"],
        seniority="senior",
        years_experience=6,
        skills=["Python", "Django", "PostgreSQL", "Docker", "REST APIs"],
        domains=["payments", "fintech"],
        locations=["Bengaluru"],
        highlights=["Rebuilt the settlement pipeline in Python"],
        summary="Backend engineer building payment services in Python and Django.",
    )


@pytest.fixture
def filters() -> SearchFilters:
    """Filters with a whitelist, as the UI always supplies them."""
    return SearchFilters(
        role="Backend Engineer",
        location="Bengaluru",
        remote_only=False,
        seniority="senior",
        recency_days=30,
        sites=["boards.greenhouse.io", "jobs.lever.co"],
    )


@pytest.fixture
def hit() -> JobHit:
    """One search result on a whitelisted board."""
    return JobHit(
        url="https://boards.greenhouse.io/acme/jobs/4567890",
        title="Senior Backend Engineer",
        domain="boards.greenhouse.io",
        snippet="We are hiring a backend engineer with Python and PostgreSQL experience.",
        provider_score=0.9,
        query="senior backend engineer jobs",
    )


@pytest.fixture
def assessment() -> JobAssessment:
    """A normalised assessment: three must-haves, one preferred."""
    return JobAssessment(
        company="Acme",
        title="Senior Backend Engineer",
        location="Bengaluru",
        remote=False,
        requirements=[
            JobRequirement(
                id="R-01",
                text="5+ years of backend engineering",
                must_have=True,
                category="experience",
            ),
            JobRequirement(id="R-02", text="Strong Python", must_have=True, category="hard_skill"),
            JobRequirement(
                id="R-03", text="Kubernetes in production", must_have=True, category="hard_skill"
            ),
            JobRequirement(id="R-04", text="Payments domain", must_have=False, category="domain"),
        ],
        assessments=[
            RequirementAssessment(
                requirement_id="R-01",
                status="covered",
                evidence="Backend engineer with 6 years building payment services.",
                note="Six years stated.",
            ),
            RequirementAssessment(
                requirement_id="R-02",
                status="covered",
                evidence="Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%",
                note="Python throughout.",
            ),
            RequirementAssessment(
                requirement_id="R-03", status="missing", note="No Kubernetes on the resume."
            ),
            RequirementAssessment(
                requirement_id="R-04",
                status="covered",
                evidence="Led a team of 4 engineers across two payment integrations",
                note="Payments experience.",
            ),
        ],
    )
