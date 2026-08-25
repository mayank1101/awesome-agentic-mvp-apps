"""Exception hierarchy for the travel-planning app.

Everything the application raises on purpose derives from
:class:`TravelAgentError`, so a caller can catch the whole domain in one
``except`` without also swallowing bugs like `TypeError`.

The search/model split mirrors this repo's other search-and-synthesise apps:
a 429 from either provider can mean two different things -- a per-minute limit
worth a short wait, or an exhausted credit or token pool that a wait cannot
fix -- and those need different messages on screen.
"""


class TravelAgentError(Exception):
    """Base class for every error this application raises deliberately."""


# --------------------------------------------------------------------------- #
# Trip request
# --------------------------------------------------------------------------- #


class InvalidTripRequest(TravelAgentError):
    """The destination or trip length failed validation before any call was made."""


class DestinationBlocked(TravelAgentError):
    """Input scanning found a high-severity pattern and blocking is enabled."""

    def __init__(self, message: str, *, findings: list | None = None) -> None:
        """Store the findings alongside the message.

        Args:
            message: What to tell the user.
            findings: The guardrail findings that caused the block.
        """
        super().__init__(message)
        self.findings = findings or []


# --------------------------------------------------------------------------- #
# Search provider (Tavily)
# --------------------------------------------------------------------------- #


class SearchError(TravelAgentError):
    """A search call did not return a usable result."""


class SearchAuthError(SearchError):
    """The search API key was missing or rejected."""


class SearchQuotaExhausted(SearchError):
    """The search provider's credit pool is spent, or is rate-limiting hard."""


# --------------------------------------------------------------------------- #
# Model provider (Groq)
# --------------------------------------------------------------------------- #


class ModelError(TravelAgentError):
    """A model call did not return a usable answer."""


class ModelRateLimited(ModelError):
    """The provider returned 429 for a per-minute limit.

    Retryable after a short backoff, unlike :class:`ModelQuotaExhausted`.
    """


class ModelQuotaExhausted(ModelError):
    """The provider's daily token budget is spent."""


class ModelRequestTooLarge(ModelError):
    """The request exceeded the provider's per-request or per-minute token cap."""


class ModelUnavailable(ModelError):
    """The configured model id was rejected by the provider."""


class ModelResponseInvalid(ModelError):
    """The reply could not be parsed into the requested schema.

    Raised only after the repair retry has also failed.
    """


# --------------------------------------------------------------------------- #
# Run-wide
# --------------------------------------------------------------------------- #


class RunDeadlineExceeded(TravelAgentError):
    """The global run deadline expired."""
