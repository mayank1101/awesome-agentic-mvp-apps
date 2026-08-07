"""Exception hierarchy for the job-search app.

Everything the application raises on purpose derives from :class:`JobSearchError`,
so a caller can catch the whole domain in one ``except`` without also swallowing
bugs like `TypeError`.

The splits are not decorative. A password-protected PDF, a scanned PDF, and a
corrupt one all fail at the same call site and need three different sentences on
screen, because they need three different actions from whoever is reading them.
The same is true of a search that found nothing versus a search key that was
refused: identical empty list, opposite meanings.
"""


class JobSearchError(Exception):
    """Base class for every error this application raises deliberately."""


# --------------------------------------------------------------------------- #
# Resume intake
# --------------------------------------------------------------------------- #


class ResumeExtractionError(JobSearchError):
    """The uploaded file could not be turned into usable text."""


class EncryptedPdfError(ResumeExtractionError):
    """The PDF is password-protected.

    Distinct because the fix belongs to the user and is easy: re-export without
    a password. Nothing the app can do on its own.
    """


class ScannedPdfError(ResumeExtractionError):
    """The PDF parsed, but yielded almost no text.

    Which means it is a scan or an image export. There is no OCR in this app --
    it would need a system binary or a model that does not fit the deployment
    target -- so this is a stated limit, not a bug.
    """


class ResumeTooLargeError(ResumeExtractionError):
    """The upload exceeded the byte or page cap, checked before parsing."""


# --------------------------------------------------------------------------- #
# Web search
# --------------------------------------------------------------------------- #


class SearchError(JobSearchError):
    """A call to the search provider did not return usable results."""


class SearchAuthError(SearchError):
    """The search key was rejected.

    Separate from a generic failure because the fix is a key, not a retry, and
    because a rejected key and an empty result set are otherwise indistinguishable
    from the outside.
    """


class SearchQuotaExhausted(SearchError):
    """The search provider's credit balance or rate limit is spent.

    Retrying costs time and fails identically, so the app does not. Free-tier
    Tavily credits are a monthly pool, which means this is a state that persists
    rather than one that clears in a minute.
    """


class ExtractError(JobSearchError):
    """Fetching the full text of a posting failed.

    Never fatal. A posting whose page cannot be read is scored on its search
    snippet instead, with the shallower basis stated on the row -- the whole
    point of showing which tier produced a score.
    """


# --------------------------------------------------------------------------- #
# Model provider
# --------------------------------------------------------------------------- #


class ModelError(JobSearchError):
    """A model call did not return a usable answer."""


class ModelRateLimited(ModelError):
    """The provider returned 429 for a per-minute limit.

    Retryable after a short backoff, unlike :class:`ModelQuotaExhausted`.
    """


class ModelQuotaExhausted(ModelError):
    """The provider's daily token budget is spent.

    One status code, a different situation: retrying costs a minute and fails
    identically, so the app does not. This one matters more here than in a
    single-document app -- a run scores several jobs, so a run can be the thing
    that exhausts the budget, and the next run starts already broken.
    """


class ModelRequestTooLarge(ModelError):
    """The request exceeded the provider's per-request or per-minute token cap.

    Distinct from a rate limit because waiting does not help: the request is the
    wrong size, and the only fix is to send less. Groq's free tier caps tokens
    per *minute* and counts the requested output reservation toward it, so a long
    posting plus a two-page resume plus the output reservation can exceed the
    ceiling on the first try. Callers are expected to shrink and retry rather
    than surface this.
    """


class ModelUnavailable(ModelError):
    """The configured model id was rejected by the provider.

    Free-tier model ids are retired without notice, and this cannot be detected
    at startup without spending a call, so it is handled where it surfaces.
    """


class ModelResponseInvalid(ModelError):
    """The reply could not be parsed into the requested schema.

    Raised only after the repair retry has also failed, so it means the model
    genuinely could not produce the shape -- not that it wrapped it oddly once.
    """


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


class EmbeddingError(JobSearchError):
    """An embedding call failed.

    Never fatal: ranking and evidence matching fall back to lexical overlap and
    the UI says which mode produced the numbers.
    """


# --------------------------------------------------------------------------- #
# Guardrails and lifecycle
# --------------------------------------------------------------------------- #


class InputBlocked(JobSearchError):
    """Resume scanning found a high-severity pattern and blocking is enabled."""

    def __init__(self, message: str, *, findings: list | None = None) -> None:
        """Store the findings alongside the message.

        Args:
            message: What to tell the user.
            findings: The guardrail findings that caused the block, so the screen
                can name them rather than shrugging.
        """
        super().__init__(message)
        self.findings = findings or []


class RunDeadlineExceeded(JobSearchError):
    """The global run deadline expired.

    Raised between steps, never mid-call. A run that dies here still has partial
    results, and the UI shows them rather than throwing away work the user
    already waited for.
    """
