"""Exception hierarchy for the review tracker.

Everything the application raises on purpose derives from :class:`TrackerError`,
so a caller can catch the whole domain in one ``except`` without also
swallowing bugs like `TypeError`.
"""


class TrackerError(Exception):
    """Base class for every error this application raises deliberately."""


# --------------------------------------------------------------------------- #
# App resolution
# --------------------------------------------------------------------------- #


class AppNotFound(TrackerError):
    """The input could not be resolved to one App Store app.

    Carries the query that was tried so the screen can name what was searched
    rather than shrugging.
    """

    def __init__(self, message: str, *, query: str | None = None) -> None:
        """Store the failing query alongside the message."""
        super().__init__(message)
        self.query = query


class InvalidAppReference(TrackerError):
    """A supplied App Store URL or ID did not parse into a usable app id."""


# --------------------------------------------------------------------------- #
# Reviews
# --------------------------------------------------------------------------- #


class ReviewFetchError(TrackerError):
    """The review feed request failed for a reason the run cannot survive.

    Unlike a multi-section search pipeline, this app makes exactly one review
    request per report, so there is no "degrade this section" path -- a failed
    fetch fails the run, with a message naming what to try.
    """


class ReviewFetchTimeout(ReviewFetchError):
    """The review feed request exceeded its timeout."""


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


class InsufficientSignal(TrackerError):
    """Too few critical reviews in the sample to run a gap analysis (SC-4).

    Not a failure: the pipeline catches this and still renders the rating
    snapshot, with an explicit message in place of gaps.
    """


class AnalysisError(TrackerError):
    """The model did not return a usable gap analysis.

    Raised only after the repair retry has also failed, meaning the model
    genuinely could not produce the schema.
    """


class ProviderRateLimited(AnalysisError):
    """The model provider returned 429 for a per-minute rate limit.

    Retryable after a short backoff, unlike :class:`ProviderQuotaExhausted`.
    """


class ProviderQuotaExhausted(AnalysisError):
    """The model provider's daily token budget is spent.

    One status code, a different situation: retrying costs a minute and fails
    identically, so the app does not.
    """


class ProviderRequestTooLarge(AnalysisError):
    """The prompt exceeded the provider's per-minute token allowance.

    Retryable, but only with less evidence -- waiting changes nothing about
    the size of the request.
    """


class StructuredOutputUnsupported(AnalysisError):
    """The model cannot be asked for JSON via ``response_format``.

    A model capability, not a provider one. Recoverable: the prompt asks for
    JSON in words too, so the request is simply re-sent without the flag.
    """


class ModelUnavailable(AnalysisError):
    """The configured model id was rejected by the provider.

    Free-tier model ids are retired without notice, and this cannot be
    detected at startup without spending a call.
    """


class InputRejected(TrackerError):
    """The request was refused before any call was made (guardrails, E-41-style)."""
