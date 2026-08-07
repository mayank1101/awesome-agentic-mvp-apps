"""Exception hierarchy for the prioritisation assistant.

Everything the application raises deliberately derives from
:class:`PrioritizationError`, so a caller can catch the whole domain in one
`except` without also swallowing bugs like `TypeError`.
"""


class PrioritizationError(Exception):
    """Base class for every error this application raises on purpose."""


class BacklogParseError(PrioritizationError):
    """The pasted backlog could not be read as a list of features.

    Raised for input problems the user can fix -- an empty paste, or a list that
    exceeds the configured cap -- not for anything a model did.
    """


class EstimateParseError(PrioritizationError):
    """The estimator replied with something that was not a usable factor set.

    Raised only after every parsing strategy has been tried, so it means the
    model genuinely did not return JSON -- not that it returned JSON in an
    unexpected wrapper. A reply that parses but covers only *some* of the
    backlog is not an error: the missing features are reported as unestimated,
    because a partial ranking with honest gaps beats a full one with invented
    numbers.
    """
