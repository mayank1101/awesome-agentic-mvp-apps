"""Tests for the search transport, the query templates, and identity resolution.

Covers E-17 (empty results), E-19/E-20/E-25 (typed failures), E-26 (SDK shape
confined to one module), E-30 (budget checked before spending), and E-07/E-08/
E-11/E-13/E-14/E-16 (resolution and abstaining).
"""

from typing import Any

import pytest
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError
from tavily.errors import TimeoutError as TavilyTimeoutError

from app.core.config import Settings
from app.core.exceptions import (
    BudgetExhausted,
    CompanyNotFound,
    SearchAuthError,
    SearchError,
    SearchQuotaError,
    SearchTimeoutError,
)
from app.models.schemas import RunBudget, SectionKey
from app.search.client import SearchClient
from app.search.queries import build_query, search_params
from app.search.resolver import resolve


class FakeTransport:
    """Stands in for the Tavily SDK. Records calls, returns or raises on cue."""

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None):
        self.response = response or {"results": []}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _hit(url: str, title: str = "A title", content: str = "Some content", score: float = 0.5):
    return {"url": url, "title": title, "content": content, "score": score}


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, tavily_api_key="tvly-test", **overrides)


def _client(transport: FakeTransport, **overrides: Any) -> SearchClient:
    settings = _settings(**overrides)
    return SearchClient(
        settings=settings,
        budget=RunBudget(limit=settings.max_search_credits_per_report),
        transport=transport,
    )


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def test_every_section_has_a_query():
    for section in SectionKey:
        assert build_query(section, "Acme", "acme.com").strip()


def test_snapshot_query_uses_the_domain_when_known():
    assert "acme.com" in build_query(SectionKey.SNAPSHOT, "Acme", "acme.com")


def test_query_has_no_double_spaces_without_a_domain():
    query = build_query(SectionKey.SNAPSHOT, "Acme", None)
    assert "  " not in query


def test_recent_moves_constrains_recency_at_the_api():
    # E-23/Q-4: enforced by the provider, not requested of the model.
    params = search_params(SectionKey.RECENT_MOVES, _settings())
    assert params["topic"] == "news"
    assert params["time_range"] == "year"


def test_other_sections_use_the_general_index():
    params = search_params(SectionKey.PRICING, _settings())
    assert params["topic"] == "general"
    assert "time_range" not in params


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def test_results_become_hits():
    transport = FakeTransport({"results": [_hit("https://acme.com/pricing", "Pricing")]})
    hits = _client(transport).search("acme pricing")

    assert len(hits) == 1
    assert hits[0].url == "https://acme.com/pricing"
    assert hits[0].host == "acme.com"


def test_results_without_a_url_are_dropped():
    # An uncitable result is worse than a missing one in an app whose claim is
    # traceability.
    transport = FakeTransport({"results": [{"title": "No link", "content": "x"}]})
    assert _client(transport).search("acme") == []


def test_empty_results_are_not_an_error():
    # E-17: the section renders "Not found in public sources" and the run goes on.
    assert _client(FakeTransport({"results": []})).search("acme") == []


def test_missing_response_fields_do_not_raise():
    transport = FakeTransport({"results": [{"url": "https://acme.com"}]})
    hits = _client(transport).search("acme")

    assert hits[0].title == "https://acme.com"
    assert hits[0].content == ""


def test_defaults_are_sent_with_every_call():
    transport = FakeTransport()
    _client(transport).search("acme")
    call = transport.calls[0]

    assert call["search_depth"] == "basic"
    assert call["include_answer"] is False
    assert call["include_raw_content"] is False
    assert call["timeout"] == 20.0


def test_per_section_params_override_defaults():
    transport = FakeTransport()
    _client(transport).search("acme", topic="news", max_results=8)

    assert transport.calls[0]["topic"] == "news"
    assert transport.calls[0]["max_results"] == 8


# --------------------------------------------------------------------------- #
# Typed failures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (InvalidAPIKeyError("bad key"), SearchAuthError),
        (UsageLimitExceededError("limit"), SearchQuotaError),
        (TavilyTimeoutError(20.0), SearchTimeoutError),
        (TimeoutError(), SearchTimeoutError),
        (ValueError("something else"), SearchError),
    ],
)
def test_provider_errors_map_to_typed_exceptions(error: Exception, expected: type):
    # Tavily's TimeoutError shadows the builtin without subclassing it, so both
    # must land on SearchTimeoutError -- this is exactly where that was missed.
    with pytest.raises(expected):
        _client(FakeTransport(error=error)).search("acme")


def test_auth_error_when_no_key_is_configured():
    client = SearchClient(settings=Settings(_env_file=None), budget=RunBudget())
    with pytest.raises(SearchAuthError):
        client.search("acme")


# --------------------------------------------------------------------------- #
# Budget (E-30, SC-8)
# --------------------------------------------------------------------------- #


def test_budget_is_charged_per_call():
    transport = FakeTransport()
    client = _client(transport)
    client.search("one")
    client.search("two")

    assert client.budget.spent == 2


def test_budget_stops_the_next_call():
    transport = FakeTransport()
    client = SearchClient(settings=_settings(), budget=RunBudget(limit=1), transport=transport)
    client.search("one")

    with pytest.raises(BudgetExhausted):
        client.search("two")

    # The refused call never reached the provider.
    assert len(transport.calls) == 1


def test_a_failed_call_still_costs_a_credit():
    # A budget that only counts successes overspends exactly when things are
    # going wrong.
    client = _client(FakeTransport(error=ValueError("boom")))
    with pytest.raises(SearchError):
        client.search("acme")

    assert client.budget.spent == 1


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_supplied_domain_skips_the_search():
    transport = FakeTransport()
    identity = resolve("Acme", "acme.com", _client(transport))

    assert identity.domain == "acme.com"
    assert identity.supplied_by_user is True
    assert transport.calls == []  # no credit spent


def test_resolution_picks_the_most_frequent_own_domain():
    transport = FakeTransport(
        {
            "results": [
                _hit("https://acme.com/", "Acme — home"),
                _hit("https://acme.com/about", "About Acme"),
                _hit("https://linkedin.com/company/acme", "Acme | LinkedIn"),
            ]
        }
    )
    identity = resolve("Acme", None, _client(transport))

    assert identity.domain == "acme.com"
    assert identity.supplied_by_user is False


def test_aggregators_never_win_resolution():
    # E-16: they are fine as evidence and wrong as an answer to "their own site".
    transport = FakeTransport(
        {
            "results": [
                _hit("https://linkedin.com/company/acme", "Acme | LinkedIn"),
                _hit("https://crunchbase.com/acme", "Acme - Crunchbase"),
                _hit("https://acme.com/", "Acme"),
                _hit("https://acme.com/about", "About Acme"),
            ]
        }
    )
    assert resolve("Acme", None, _client(transport)).domain == "acme.com"


def test_only_aggregators_abstains():
    # E-13: other people listing a company is not evidence it has a findable site.
    transport = FakeTransport(
        {
            "results": [
                _hit("https://linkedin.com/company/acme", "Acme | LinkedIn"),
                _hit("https://crunchbase.com/acme", "Acme - Crunchbase"),
            ]
        }
    )
    with pytest.raises(CompanyNotFound):
        resolve("Acme", None, _client(transport))


def test_no_results_abstains_and_names_the_query():
    # SC-5: the screen says what was searched rather than shrugging.
    with pytest.raises(CompanyNotFound) as caught:
        resolve("Zzzqqx", None, _client(FakeTransport({"results": []})))

    assert caught.value.query and "Zzzqqx" in caught.value.query


def test_unrelated_results_abstain():
    # E-08/E-14: hits exist, none of them are about what was asked for.
    transport = FakeTransport(
        {"results": [_hit("https://example.com/weather", "Weather", "Forecast for Tuesday")]}
    )
    with pytest.raises(CompanyNotFound):
        resolve("Zzzqqx", None, _client(transport))


def test_search_failure_during_resolution_abstains():
    with pytest.raises(CompanyNotFound):
        resolve("Acme", None, _client(FakeTransport(error=ValueError("boom"))))


def test_site_furniture_is_not_used_as_a_description():
    # From a live run: a language picker on Notion's homepage was rendered at the
    # top of the report as though it described the company.
    furniture = "English (US) Bahasa Indonesia Dansk Deutsch Espanol Francais Italiano Nederlands"
    transport = FakeTransport(
        {
            "results": [
                _hit("https://acme.com/", "Acme", furniture),
                _hit("https://acme.com/x", "Acme", furniture),
            ]
        }
    )
    assert resolve("Acme", None, _client(transport)).description is None


def test_a_description_must_mention_the_company():
    prose = "This page explains how the platform helps teams organise their work every day."
    transport = FakeTransport(
        {
            "results": [
                _hit("https://acme.com/", "Acme", prose),
                _hit("https://acme.com/x", "Acme", prose),
            ]
        }
    )
    assert resolve("Acme", None, _client(transport)).description is None


def test_resolution_carries_a_description():
    transport = FakeTransport(
        {
            "results": [
                _hit(
                    "https://acme.com/",
                    "Acme",
                    "Acme builds project management tools for software teams everywhere.",
                ),
                _hit("https://acme.com/about", "About Acme", "More about Acme."),
            ]
        }
    )
    identity = resolve("Acme", None, _client(transport))

    assert identity.description and "project management tools" in identity.description


def test_a_single_supporting_page_abstains():
    # E-11: one hit on a domain that echoes the name is what a parked page or a
    # coincidence looks like, and profiling it yields six confident sections
    # about nothing.
    transport = FakeTransport(
        {
            "results": [
                _hit("https://acme.com/", "Acme"),
                _hit("https://unrelated.example/blog", "Something about acme"),
            ]
        }
    )
    with pytest.raises(CompanyNotFound):
        resolve("Acme", None, _client(transport))


def test_two_supporting_pages_resolve():
    transport = FakeTransport(
        {"results": [_hit("https://acme.com/", "Acme"), _hit("https://acme.com/about", "About")]}
    )
    assert resolve("Acme", None, _client(transport)).domain == "acme.com"


def test_a_supplied_domain_bypasses_the_two_page_rule():
    # The user asserting the domain is stronger evidence than two search hits.
    assert resolve("Acme", "acme.com", _client(FakeTransport())).domain == "acme.com"
