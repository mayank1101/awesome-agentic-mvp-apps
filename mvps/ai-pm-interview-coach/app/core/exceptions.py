"""Exception hierarchy for the interview coach.

Everything the application raises deliberately derives from
:class:`InterviewCoachError`, so a caller can catch the whole domain in one
`except` without also swallowing bugs like `TypeError`.
"""


class InterviewCoachError(Exception):
    """Base class for every error this application raises on purpose."""


class ReportParseError(InterviewCoachError):
    """The evaluator replied with something that was not a usable report.

    Raised only after every parsing strategy and the repair attempt have been
    tried, so it means the model genuinely did not return a valid report -- not
    that it returned one in an unexpected wrapper.
    """


class PreflightError(InterviewCoachError):
    """The configured provider cannot be used, checked before a session starts.

    Carries the specific missing credential or unreachable host, so the message
    a user sees names what to fix. Raised at the start button rather than on the
    opening question, which would already have minted and counted a session.
    """


class EmptyInterviewError(InterviewCoachError):
    """A session ended with nothing to grade.

    The opening question alone is not an interview. Grading an empty transcript
    would have the evaluator invent five scores from no evidence, which is worse
    than no report at all.
    """
