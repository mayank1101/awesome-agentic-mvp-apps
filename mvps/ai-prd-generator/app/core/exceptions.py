"""Exception hierarchy for the PRD generator.

Everything the application raises deliberately derives from
:class:`PRDGeneratorError`, so a caller can catch the whole domain in one
`except` without also swallowing bugs like `TypeError`.
"""


class PRDGeneratorError(Exception):
    """Base class for every error this application raises on purpose."""


class OutlineParseError(PRDGeneratorError):
    """The outline agent replied with something that was not a usable outline.

    Raised only after every parsing strategy has been tried, so it means the
    model genuinely did not return JSON -- not that it returned JSON in an
    unexpected wrapper.
    """
