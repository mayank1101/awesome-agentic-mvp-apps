"""Exception hierarchy for the job-match app.

Everything the application raises on purpose derives from :class:`JobMatchError`,
so a caller can catch the whole domain in one ``except`` without also swallowing
bugs like `TypeError`.

The splits are not decorative. A password-protected PDF, a scanned PDF, and a
PDF with a broken byte stream all fail at the same call site and need three
different sentences on screen, because they need three different actions from
whoever is reading them.
"""


class JobMatchError(Exception):
    """Base class for every error this application raises deliberately."""


# --------------------------------------------------------------------------- #
# Resume intake
# --------------------------------------------------------------------------- #


class ResumeExtractionError(JobMatchError):
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
# Model provider
# --------------------------------------------------------------------------- #


class ModelError(JobMatchError):
    """A model call did not return a usable answer."""


class ModelRateLimited(ModelError):
    """The provider returned 429 for a per-minute limit.

    Retryable after a short backoff, unlike :class:`ModelQuotaExhausted`.
    """


class ModelQuotaExhausted(ModelError):
    """The provider's daily token budget is spent.

    One status code, a different situation: retrying costs a minute and fails
    identically, so the app does not.
    """


class ModelRequestTooLarge(ModelError):
    """The request exceeded the provider's per-request or per-minute token cap.

    Distinct from a rate limit because waiting does not help: the request is the
    wrong size, and the only fix is to send less. Groq's free tier caps tokens
    per *minute* and counts the requested output reservation toward it, so a
    2-page resume with a dozen requirements plus a 2200-token reservation can
    exceed a 6000 TPM ceiling on the first try. Callers are expected to shrink
    and retry rather than surface this.
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


class EmbeddingError(JobMatchError):
    """An embedding call failed.

    Never fatal: the matcher falls back to lexical overlap and the report says
    which mode produced it.
    """


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


class InputBlocked(JobMatchError):
    """Input scanning found a high-severity pattern and blocking is enabled."""

    def __init__(self, message: str, *, findings: list | None = None) -> None:
        """Store the findings alongside the message.

        Args:
            message: What to tell the user.
            findings: The guardrail findings that caused the block, so the screen
                can name them rather than shrugging.
        """
        super().__init__(message)
        self.findings = findings or []


class FabricationDetected(JobMatchError):
    """The tailored resume introduced facts that are not in the original.

    The hard guarantee this app makes. Carries the offending fragments so the
    screen can show exactly what was invented rather than asserting that
    something was.
    """

    def __init__(self, message: str, *, offenders: list[str] | None = None) -> None:
        """Store the fabricated fragments alongside the message.

        Args:
            message: What to tell the user.
            offenders: The specific strings absent from the original resume.
        """
        super().__init__(message)
        self.offenders = offenders or []


class RunDeadlineExceeded(JobMatchError):
    """The global run deadline expired."""
