"""The search transport: the only module that knows Tavily's shape.

Everything above this file speaks in :class:`~app.models.schemas.SearchHit`, so
swapping the search provider means rewriting this module and nothing else. That
containment is also what makes E-26 survivable — an SDK that renames a response
field breaks one file with one set of tests.

Two rules the rest of the app depends on:

* **Every call is charged before it is made.** :class:`RunBudget` is checked
  first, so the cap in SC-8 is arithmetic rather than intent (E-30).
* **Failures are typed.** A bad key, an exhausted pool, a timeout, and everything
  else are four different exceptions, because they need four different messages
  on screen (E-19, E-20, E-25).
"""

from typing import Any, Protocol

import requests
from tavily import TavilyClient
from tavily.errors import (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)

# Tavily defines its own `TimeoutError` that shadows the builtin without
# subclassing it, so catching the builtin alone silently mislabels every real
# timeout as a generic failure. Aliased here so both are caught explicitly.
from tavily.errors import TimeoutError as TavilyTimeoutError

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    BudgetExhausted,
    SearchAuthError,
    SearchError,
    SearchQuotaError,
    SearchTimeoutError,
)
from app.core.logging import get_logger
from app.models.schemas import RunBudget, SearchHit

logger = get_logger(__name__)


class SearchTransport(Protocol):
    """What the search client needs from an SDK.

    Narrow on purpose: tests supply a fake with this one method rather than
    mocking the Tavily package, so a test failure means our logic broke and not
    that a mock drifted from the library.
    """

    def search(self, **kwargs: Any) -> dict[str, Any]:
        """Run one search and return the provider's raw response."""
        ...


def _to_hits(response: dict[str, Any]) -> list[SearchHit]:
    """Convert a provider response into hits, dropping anything unusable.

    A result with no URL cannot be cited, and an uncitable result is worse than a
    missing one in an app whose whole claim is traceability.
    """
    hits: list[SearchHit] = []
    for raw in response.get("results") or []:
        url = (raw.get("url") or "").strip()
        if not url:
            continue
        hits.append(
            SearchHit(
                title=(raw.get("title") or url).strip(),
                url=url,
                content=(raw.get("content") or "").strip(),
                score=float(raw.get("score") or 0.0),
                published_date=raw.get("published_date"),
            )
        )
    return hits


class SearchClient:
    """Budget-aware wrapper around the search provider."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        budget: RunBudget | None = None,
        transport: SearchTransport | None = None,
    ) -> None:
        """Build a client for one report.

        Args:
            settings: Runtime configuration; defaults to the process settings.
            budget: The run's credit cap. A fresh one is created per report, since
                the cap is per report and not per process.
            transport: Injected SDK, for tests. Built from the configured key when
                omitted.
        """
        self.settings = settings or get_settings()
        self.budget = budget or RunBudget(limit=self.settings.max_search_credits_per_report)
        self._transport = transport

    @property
    def transport(self) -> SearchTransport:
        """The SDK client, built lazily so a missing key fails at use, not import."""
        if self._transport is None:
            if not self.settings.tavily_api_key:
                raise SearchAuthError("TAVILY_API_KEY is not set")
            self._transport = TavilyClient(api_key=self.settings.tavily_api_key)
        return self._transport

    def search(self, query: str, **params: Any) -> list[SearchHit]:
        """Run one search, charging one credit.

        Args:
            query: The query string.
            **params: Per-section parameters from
                :func:`app.search.queries.search_params`, plus any override.

        Returns:
            The hits, possibly empty. An empty list is a legitimate outcome that
            renders as "Not found in public sources" (E-17), not an error.

        Raises:
            BudgetExhausted: The run's own cap would be exceeded (E-30).
            SearchAuthError: The provider rejected the credentials (E-19).
            SearchQuotaError: The provider's credit pool is spent (E-20).
            SearchTimeoutError: The call exceeded its timeout (E-25).
            SearchError: Anything else, so one failed section cannot fail a run.
        """
        if not self.budget.can_afford():
            raise BudgetExhausted(
                f"search budget of {self.budget.limit} credits is spent for this report"
            )

        call: dict[str, Any] = {
            "query": query,
            "search_depth": self.settings.search_depth,
            "include_answer": False,
            "include_raw_content": False,
            "timeout": self.settings.search_timeout_seconds,
            **params,
        }

        # Resolved before the try block, deliberately: building the transport can
        # raise SearchAuthError, and the catch-all below would otherwise re-wrap
        # our own typed error as a generic SearchError -- turning "your key is
        # missing" into "search failed".
        transport = self.transport

        # Charged before the call, not after: a call that was made and then failed
        # still cost a credit, and a budget that only counts successes will
        # overspend exactly when things are going wrong.
        self.budget.charge()

        try:
            response = transport.search(**call)
        except (InvalidAPIKeyError, MissingAPIKeyError) as exc:
            raise SearchAuthError("the search API key was rejected") from exc
        except UsageLimitExceededError as exc:
            raise SearchQuotaError("the search credit pool is exhausted") from exc
        except (TavilyTimeoutError, TimeoutError, requests.exceptions.Timeout) as exc:
            raise SearchTimeoutError(f"search timed out after {call['timeout']}s") from exc
        except Exception as exc:  # noqa: BLE001 - one bad section must not fail the run
            raise SearchError(f"search failed: {type(exc).__name__}") from exc

        hits = _to_hits(response)
        logger.info("search returned %d hit(s) [credits spent: %d]", len(hits), self.budget.spent)
        return hits
