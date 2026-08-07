"""Guardrails-AI validators wrapping this app's own checks.

Two custom validators, registered locally rather than pulled from Guardrails Hub:

* ``no-fabricated-facts`` -- runs :func:`app.services.provenance.check` and fails
  when the rewrite introduces a number, name, or contact detail that is absent
  from the original resume.
* ``safe-resume-markdown`` -- fails when generated Markdown still contains active
  content after sanitising, which would mean the sanitiser regressed.

**Local, not Hub, on purpose.** Hub validators are installed with
``guardrails hub install``, which needs a token and a network call at build time.
That is a second thing to configure before the app runs and a second thing to
break in a container build, for validators that would still not know what this
app's "fabrication" means. A `@register_validator` class in the repo travels with
the code.

**What Guardrails buys here, stated plainly.** The decision logic is the same
function either way -- it lives in :mod:`app.services.provenance` and is unit
tested directly. Guardrails supplies the uniform validator interface, the on-fail
policy, and a single place to add the Hub validators (toxicity, PII) that a
production version of this would want. It is not a second opinion, and this
module does not pretend it is. If the package is missing or its API has moved,
:func:`check_tailored_resume` runs the same checks directly and reports which
path it took.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from app.core.logging import get_logger
from app.services import provenance
from app.services.guardrails import sanitize_markdown
from app.services.provenance import Violation

logger = get_logger(__name__)

Engine = Literal["guardrails", "builtin"]


@dataclass
class ValidationOutcome:
    """The result of validating one tailored resume.

    Attributes:
        passed: Whether every check passed.
        violations: Fabricated fragments found, empty when `passed`.
        unsafe_markdown: Whether active content survived sanitising.
        engine: Which path ran the checks, surfaced in the UI so a reader knows
            whether the Guardrails layer was actually exercised.
    """

    passed: bool
    violations: list[Violation] = field(default_factory=list)
    unsafe_markdown: bool = False
    engine: Engine = "builtin"


def _build_guard() -> Any | None:
    """Register the validators and return a configured Guard, or None.

    Returns:
        A ``Guard`` with both validators attached, or ``None`` when the
        package is not installed or its API differs from the one used here. A
        missing optional dependency is not an error worth failing a rewrite over.
    """
    try:
        from guardrails import Guard, OnFailAction
        from guardrails.validator_base import (
            FailResult,
            PassResult,
            ValidationResult,
            Validator,
            register_validator,
        )
    except Exception as exc:  # noqa: BLE001 - any import-time failure means fall back
        logger.info("Guardrails-AI unavailable (%s); using the built-in checks", type(exc).__name__)
        return None

    @register_validator(name="no-fabricated-facts", data_type="string")
    class NoFabricatedFacts(Validator):  # type: ignore[misc]
        """Fails when the rewrite states a fact the original resume does not."""

        def validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
            """Compare `value` against ``metadata["original_resume"]``."""
            original = (metadata or {}).get("original_resume", "")
            violations = provenance.check(original, str(value))
            if not violations:
                return PassResult()
            return FailResult(
                error_message=(
                    "The rewrite introduced facts absent from the original resume: "
                    + "; ".join(f"{v.kind}: {v.text}" for v in violations[:5])
                ),
                metadata={"violations": violations},
            )

    @register_validator(name="safe-resume-markdown", data_type="string")
    class SafeResumeMarkdown(Validator):  # type: ignore[misc]
        """Fails when sanitising would still have to change the output."""

        def validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
            """Check that the value is already inert Markdown."""
            text = str(value)
            if sanitize_markdown(text) == text:
                return PassResult()
            return FailResult(error_message="The generated Markdown contained active content.")

    # NOOP rather than EXCEPTION: this module reports an outcome and lets the
    # caller decide what to do with it, and the exception class Guardrails raises
    # has moved between versions -- catching an outcome is stabler than catching
    # a name.
    return (
        Guard()
        .use(NoFabricatedFacts, on_fail=OnFailAction.NOOP)
        .use(SafeResumeMarkdown, on_fail=OnFailAction.NOOP)
    )


@lru_cache
def _guard() -> Any | None:
    """Return the process-wide Guard, building it on first use."""
    return _build_guard()


def check_tailored_resume(markdown: str, original_resume: str) -> ValidationOutcome:
    """Validate a rewrite against the resume it was derived from.

    Args:
        markdown: The tailored resume as returned by the model.
        original_resume: The extracted text of the uploaded PDF.

    Returns:
        The outcome, including the specific fabricated fragments when there are
        any. Never raises: the caller decides whether a violation blocks the
        rewrite or is shown as a warning, based on ``STRICT_FABRICATION_GUARD``.
    """
    violations = provenance.check(original_resume, markdown)
    unsafe = sanitize_markdown(markdown) != markdown

    guard = _guard()
    if guard is None:
        return ValidationOutcome(
            passed=not violations and not unsafe,
            violations=violations,
            unsafe_markdown=unsafe,
            engine="builtin",
        )

    try:
        result = guard.validate(markdown, metadata={"original_resume": original_resume})
        passed = bool(getattr(result, "validation_passed", False))
    except Exception as exc:  # noqa: BLE001 - a guard that breaks must not block a rewrite
        logger.warning(
            "Guardrails validation raised %s; using built-in results", type(exc).__name__
        )
        return ValidationOutcome(
            passed=not violations and not unsafe,
            violations=violations,
            unsafe_markdown=unsafe,
            engine="builtin",
        )

    return ValidationOutcome(
        passed=passed and not violations and not unsafe,
        violations=violations,
        unsafe_markdown=unsafe,
        engine="guardrails",
    )
