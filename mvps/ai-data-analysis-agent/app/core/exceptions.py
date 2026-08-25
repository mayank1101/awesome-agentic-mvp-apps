"""Exception hierarchy for the data-analysis app.

Everything the application raises on purpose derives from
:class:`DataAnalysisError`, so a caller can catch the whole domain in one
``except`` without also swallowing bugs like `TypeError`.

The code-related split matters most here, because this app's core operation is
running language-model-generated code: a request that fails AST validation
(the model tried something disallowed), one that raises at runtime (a real
KeyError against this dataset), and one that runs too long are three different
situations, and only the first two are worth retrying with a repair prompt.
"""


class DataAnalysisError(Exception):
    """Base class for every error this application raises deliberately."""


# --------------------------------------------------------------------------- #
# CSV intake
# --------------------------------------------------------------------------- #


class CsvError(DataAnalysisError):
    """The uploaded file could not be turned into a usable dataset."""


class CsvTooLargeError(CsvError):
    """The upload exceeded the byte, row, or column cap."""


class CsvEmptyError(CsvError):
    """The file parsed but produced no rows or no columns."""


class CsvParseError(CsvError):
    """The bytes are not readable as CSV."""


# --------------------------------------------------------------------------- #
# Model provider
# --------------------------------------------------------------------------- #


class ModelError(DataAnalysisError):
    """A model call did not return a usable answer."""


class ModelRateLimited(ModelError):
    """The provider returned 429 for a per-minute limit.

    Retryable after a short backoff, unlike :class:`ModelQuotaExhausted`.
    """


class ModelQuotaExhausted(ModelError):
    """The provider's daily token budget is spent."""


class ModelRequestTooLarge(ModelError):
    """The request exceeded the provider's per-request or per-minute token cap.

    Distinct from a rate limit because waiting does not help -- the request is
    the wrong size, and the only fix is to send less.
    """


class ModelUnavailable(ModelError):
    """The configured model id was rejected by the provider."""


class ModelResponseInvalid(ModelError):
    """The reply could not be parsed into the requested schema.

    Raised only after the repair retry has also failed.
    """


# --------------------------------------------------------------------------- #
# Generated code
# --------------------------------------------------------------------------- #


class CodeError(DataAnalysisError):
    """Something about the model-generated analysis code went wrong."""


class CodeGenerationError(CodeError):
    """The generated code used a construct the sandbox does not allow.

    Never a security bypass in itself -- the sandbox's execution-time
    restrictions (a stripped builtins dict, a copy of the dataframe) hold even
    if this check somehow missed something. This exists so the failure is
    diagnosed before execution, with a message specific enough to repair from.
    """


class CodeExecutionError(CodeError):
    """The code passed validation but raised, or did not produce a result."""


class CodeTimeoutError(CodeError):
    """The code did not finish within the configured time budget."""


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


class QuestionBlocked(DataAnalysisError):
    """Input scanning found a high-severity pattern and blocking is enabled."""

    def __init__(self, message: str, *, findings: list | None = None) -> None:
        """Store the findings alongside the message.

        Args:
            message: What to tell the user.
            findings: The guardrail findings that caused the block.
        """
        super().__init__(message)
        self.findings = findings or []


class RunDeadlineExceeded(DataAnalysisError):
    """The global run deadline expired."""
