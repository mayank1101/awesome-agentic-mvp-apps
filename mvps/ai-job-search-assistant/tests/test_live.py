"""Opt-in tests that hit the real services: `pytest -m live`.

These exist because the offline suite proves the code does what it was told and
says nothing about whether the *services* behave the way the code assumes. Three
assumptions here are worth spending credits to check, and every one of them was
wrong at least once during the build:

* that a domain-restricted search returns single job postings rather than the
  board's index pages, at a rate that leaves a usable shortlist;
* that the extraction endpoint returns enough text from a real applicant-tracking
  page to score requirements against;
* that a strong resume and an unrelated one produce visibly different scores
  against the same posting -- the failure this whole app is built to avoid is a
  list where everything reads 80.

They are excluded from the default run (see `addopts` in `pyproject.toml`), so a
missing key is never the reason a run is red.
"""

import pytest

from app.core.config import get_settings
from app.models.schemas import SearchFilters
from app.services import fetch, pipeline, search
from app.services.profile import extract_profile
from app.services.queries import build_queries

pytestmark = pytest.mark.live

_SITES = ["boards.greenhouse.io", "jobs.lever.co", "jobs.ashbyhq.com", "linkedin.com"]

_RESUME = (
    "Priya Raman\nBengaluru\n\n"
    "SUMMARY\nBackend engineer with 6 years building payment services at scale.\n\n"
    "SKILLS\nPython, Django, FastAPI, PostgreSQL, Redis, Docker, Kafka, REST APIs, AWS\n\n"
    "EXPERIENCE\nSenior Backend Engineer, Fintrail, Mar 2021 - Present, Bengaluru\n"
    "- Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%\n"
    "- Led a team of 4 engineers across two payment gateway integrations\n"
    "- Designed REST APIs serving 12000 requests per minute on AWS\n\n"
    "Backend Engineer, Kite Systems, Jul 2018 - Feb 2021, Pune\n"
    "- Built Django services for merchant onboarding\n\n"
    "EDUCATION\nB.Tech, Computer Science, VIT Vellore, 2018\n"
)


@pytest.fixture(autouse=True)
def _require_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the offline fixture: these tests need the developer's real `.env`."""
    from pathlib import Path

    from app.core.config import Settings

    monkeypatch.setitem(
        Settings.model_config,
        "env_file",
        Path(__file__).resolve().parents[1] / ".env",
    )
    get_settings.cache_clear()

    if get_settings().missing_credentials():
        pytest.skip("live tests need TAVILY_API_KEY and GROQ_API_KEY")


def test_a_real_search_returns_real_postings() -> None:
    hits, raw = search.search_jobs(["senior backend engineer python jobs"], _SITES)

    assert raw > 0, "the provider returned nothing at all"
    assert hits, "every result was rejected as a non-posting -- check the URL patterns"
    assert all(hit.url.startswith("https://") for hit in hits)
    # The whitelist is the app's central promise.
    assert all(
        hit.domain in _SITES or any(hit.domain.endswith(f".{site}") for site in _SITES)
        for hit in hits
    )


def test_a_real_posting_can_be_read() -> None:
    hits, _ = search.search_jobs(["senior backend engineer python jobs"], _SITES)
    postings = fetch.fetch_postings([hit.url for hit in hits[:3]])

    readable = [posting for posting in postings.values() if posting.ok]
    assert readable, "no posting in the top three yielded scoreable text"
    assert len(readable[0].text) >= get_settings().min_posting_chars


def test_the_resume_parses_into_something_searchable() -> None:
    profile = extract_profile(_RESUME)

    assert profile.titles, "no title came out of the resume, so no query can be built"
    assert profile.seniority in {"mid", "senior", "lead"}
    assert build_queries(profile, SearchFilters(), limit=2)


def test_scores_discriminate_between_a_matched_and_an_unrelated_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure mode that would make this app worthless: everything scores 80."""
    monkeypatch.setenv("DEEP_SCORE_COUNT", "2")
    monkeypatch.setenv("MAX_QUERIES", "1")
    get_settings.cache_clear()

    filters = SearchFilters(role="Backend Engineer", recency_days=90, sites=_SITES)

    matched = pipeline.collect(pipeline.run_search(_RESUME, filters))
    unrelated = pipeline.collect(
        pipeline.run_search(
            "Ravi Kumar\nBengaluru\n\nSUMMARY\nRegistered nurse with 8 years in "
            "post-operative care.\n\nSKILLS\nTriage, patient assessment, wound care, IV therapy\n\n"
            "EXPERIENCE\nSenior Staff Nurse, City Hospital, 2018 - Present\n"
            "- Managed post-operative recovery for 20 beds\n"
            "- Trained six junior nurses in triage protocol\n",
            filters,
        )
    )

    matched_deep = [job.score for job in matched.jobs if job.tier == "deep"]
    unrelated_deep = [job.score for job in unrelated.jobs if job.tier == "deep"]

    if not matched_deep or not unrelated_deep:
        pytest.skip("no posting was deeply scored on either run")

    assert max(matched_deep) - max(unrelated_deep) >= 25
