"""Exception hierarchy for the competitor analyzer.

Everything the application raises on purpose derives from :class:`AnalyzerError`,
so a caller can catch the whole domain in one ``except`` without also swallowing
bugs like `TypeError`.

The split among the search errors is not decorative. A bad key, an exhausted
credit pool, and a slow network need three different messages on screen, because
they need three different actions from whoever is reading them (E-19, E-20,
E-25). Collapsing them into one "search failed" would make the app's most common
operational failure the least diagnosable.
"""


class AnalyzerError(Exception):
    """Base class for every error this application raises deliberately."""


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class SearchError(AnalyzerError):
    """A search call failed for a reason the run may be able to survive.

    A single failed section search degrades that section to "Not found in public
    sources" rather than failing the report (E-17).
    """


class SearchAuthError(SearchError):
    """The search provider rejected the credentials (E-19).

    Distinct from :class:`SearchQuotaError` because the fix is a different one:
    a wrong key needs replacing, an exhausted pool needs waiting or upgrading.
    """


class SearchQuotaError(SearchError):
    """The search provider's credit pool is exhausted (E-20).

    Never retried: a retry against an exhausted pool is latency with a certain
    outcome.
    """


class SearchTimeoutError(SearchError):
    """A single search call exceeded its per-call timeout (E-25, E-32)."""


class BudgetExhausted(AnalyzerError):
    """The run's own credit cap would be exceeded by the next call (E-30, SC-8).

    Raised *before* spending, never after. This is the app's cap on itself, not
    the provider's cap on the app -- see :class:`SearchQuotaError` for that.
    """


class CompanyNotFound(AnalyzerError):
    """Identity resolution could not settle on a company (E-11, E-13, E-14).

    Carries the query that was run so the screen can name what was searched
    rather than shrugging (SC-5).
    """

    def __init__(self, message: str, *, query: str | None = None) -> None:
        """Store the failing query alongside the message.

        Args:
            message: What to tell the user.
            query: The search query that failed to resolve, if there was one.
        """
        super().__init__(message)
        self.query = query


class RunDeadlineExceeded(AnalyzerError):
    """The global run deadline expired (E-31).

    Not surfaced as a failure: the pipeline catches it, treats the remaining
    sections as empty, and still synthesises what it has. A late report beats no
    report, and a guaranteed report beats both.
    """


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


class SynthesisError(AnalyzerError):
    """The model did not return a usable report body.

    Raised only after the repair retry has also failed (E-34, E-38), so it means
    the model genuinely could not produce the schema -- not that it wrapped it
    oddly on the first attempt.
    """


class ProviderRateLimited(SynthesisError):
    """The model provider returned 429 for a per-minute rate limit (E-35).

    Retryable after a short backoff, unlike :class:`ProviderQuotaExhausted`.
    """


class ProviderQuotaExhausted(SynthesisError):
    """The model provider's daily token budget is spent (E-35).

    One status code, a different situation: retrying costs a minute and fails
    identically, so the app does not.
    """


class ProviderRequestTooLarge(SynthesisError):
    """The prompt exceeded the provider's per-minute token allowance (E-28).

    Free-tier tiers cap tokens *per minute*, and the cap counts the requested
    output reservation as well as the prompt. Retryable, but only with less
    evidence -- waiting changes nothing about the size of the request.
    """


class StructuredOutputUnsupported(SynthesisError):
    """The model cannot be asked for JSON via ``response_format``.

    A *model* capability, not a provider one -- two models behind the same
    OpenRouter key disagree about it. Recoverable: the prompt asks for JSON in
    words too, so the request is simply re-sent without the structured-output
    flag. What is lost is the provider-side guarantee, not the output.
    """


class ModelUnavailable(SynthesisError):
    """The configured model id was rejected by the provider (E-36).

    Free-tier model ids are retired without notice, and this cannot be detected
    at startup without spending a call, so it is handled where it surfaces.
    """
