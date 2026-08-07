"""Pydantic models for every boundary in the app.

Two of these are parsed straight from model output (:class:`CandidateProfile`,
:class:`JobAssessment`) and are therefore the app's real validation layer: a
model that returns the wrong shape fails here, loudly, at the edge, rather than
three functions later as an `AttributeError`.

The rest describe things this app computes itself. :class:`ScoredJob`'s ``score``
in particular is **arithmetic over the requirement verdicts**, not a number the
model was asked for -- see :mod:`app.services.scoring`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: How a job's score was produced. The distinction is shown on every row and the
#: two are never blended into one number: a `deep` score read the posting and can
#: name the requirement it scored, a `shallow` one ranked a title and a
#: few-hundred-character search snippet. Presenting the second as if it were the
#: first is the quiet dishonesty this app is built to avoid.
ScoreTier = Literal["deep", "shallow"]

#: Whether the numbers behind a ranking or a match came from embeddings or from
#: word overlap. Comparable within a mode, not across, which is why it travels
#: with the results rather than being logged and forgotten.
MatchingMode = Literal["semantic", "lexical"]

CoverageStatus = Literal["covered", "partial", "missing"]

RequirementCategory = Literal[
    "hard_skill",
    "experience",
    "education",
    "domain",
    "soft_skill",
    "responsibility",
]

Seniority = Literal["junior", "mid", "senior", "lead"]

#: Words a model writes into a string field when it means "absent". Seen on a
#: live run in the sibling app: a posting with no company name came back as the
#: *string* ``"null"``, which is truthy, so the heading read "Engineer at null".
_NULLISH = frozenset({"null", "none", "nil", "n/a", "na", "unknown", "not specified", "-"})


class _Strict(BaseModel):
    """Base for parsed model output: unknown keys are dropped, not fatal.

    Every string field also passes through :meth:`_blank_nullish`, because a
    model asked for JSON will write "null" as text about as often as it omits
    the key, and only one of those is caught by the schema.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @field_validator("*", mode="after")
    @classmethod
    def _blank_nullish(cls, value: object) -> object:
        """Turn a model's textual stand-in for "absent" into an empty string."""
        if isinstance(value, str) and value.strip().lower() in _NULLISH:
            return ""
        return value


# --------------------------------------------------------------------------- #
# What the user asks for
# --------------------------------------------------------------------------- #


class SearchFilters(BaseModel):
    """The optional narrowing the user applies on top of their resume.

    All optional on purpose: the zero-effort path is upload-and-run, and every
    field here exists for the user whose resume points at their past while their
    intent points somewhere else. A career switcher's search should follow the
    intent; only the *scoring* should stay honest about the gap.

    Attributes:
        role: Target job title, free text. Empty means "use the titles the
            resume evidences".
        location: Free text, matched loosely by the search provider.
        remote_only: Whether to bias queries toward remote postings.
        seniority: Target level, or ``None`` to infer it from the resume.
        recency_days: Restrict results to postings seen in this window. A job
            board's oldest listings are its least likely to still be open.
        sites: The domain whitelist for this run. Never empty by the time it
            reaches the search layer -- an empty whitelist searches the entire
            web, which is the one thing this app promises not to do.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    role: str = ""
    location: str = ""
    remote_only: bool = False
    seniority: Seniority | None = None
    recency_days: int | None = 30
    sites: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# What the model reads out of the resume
# --------------------------------------------------------------------------- #


class CandidateProfile(_Strict):
    """The resume, reduced to the facts that drive search and scoring.

    Shown to the user *before* the search runs. If the app misread the resume,
    that is a thing to see in two seconds -- not something to infer later from a
    list of jobs that all look wrong.

    Deliberately carries no name, email, or phone number. Nothing downstream
    needs them, and the surest way not to leak contact details into a search
    query or a log line is not to put them in the object that gets passed
    around.

    Attributes:
        titles: Job titles the resume evidences, most recent first. These become
            search queries, so they are wanted as a hiring manager would write
            them, not as the candidate's employer styled them.
        seniority: Level the resume reads at.
        years_experience: Total professional years, as the model reads it.
        skills: Concrete, searchable skills -- languages, frameworks, tools.
        domains: Industries or problem spaces worked in.
        locations: Places the resume mentions as the candidate's own.
        highlights: A few achievements, used as scoring context rather than as
            search terms.
        summary: One or two sentences. The text that gets embedded for ranking,
            so it should read like a job description of the candidate.
    """

    titles: list[str] = Field(default_factory=list)
    seniority: Seniority = "mid"
    years_experience: float | None = None
    skills: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("titles", "skills", "domains", "locations", "highlights", mode="after")
    @classmethod
    def _drop_blanks(cls, values: list[str]) -> list[str]:
        """Remove empties left behind by :meth:`_Strict._blank_nullish`."""
        return [value for value in values if value and value.strip()]

    def profile_text(self) -> str:
        """Return one block of text standing in for the candidate.

        Used wherever the whole profile has to be compared against something as
        a single string -- ranking a search snippet, embedding once per run.
        """
        parts = [
            self.summary,
            " ".join(self.titles),
            " ".join(self.skills),
            " ".join(self.domains),
        ]
        return "\n".join(part for part in parts if part.strip())


# --------------------------------------------------------------------------- #
# What comes back from search
# --------------------------------------------------------------------------- #


class JobHit(BaseModel):
    """One search result that survived filtering and de-duplication.

    Attributes:
        url: The posting's canonical URL, normalised for de-duplication.
        title: The result title, as the search provider reported it.
        domain: Registrable host, used to show where a job came from and to
            prove the whitelist held.
        snippet: The provider's content extract. A few hundred characters of
            marketing copy -- enough to rank on, never enough to score on.
        published: Publication date, when the provider supplies one.
        provider_score: The search provider's own relevance number for the
            query that returned this. Kept for tie-breaking and for debugging a
            surprising result set; it says nothing about resume fit.
        query: Which of the run's queries produced this hit first.
    """

    url: str
    title: str = ""
    domain: str = ""
    snippet: str = ""
    published: str = ""
    provider_score: float = 0.0
    query: str = ""

    def ranking_text(self) -> str:
        """Return the text a shallow rank is computed from.

        Title first and snippet second, which is also the order of their
        reliability: a title is written to describe the role, a snippet is
        whatever the crawler happened to catch.
        """
        return f"{self.title}\n{self.snippet}".strip()


class PostingText(BaseModel):
    """The full text of a posting, when it could be fetched.

    Attributes:
        url: The URL that was fetched.
        text: Normalised page text, capped.
        truncated: Whether the cap removed anything.
        ok: Whether the fetch produced enough text to score against. ``False``
            covers a JavaScript shell, a login wall, an expired listing, and a
            provider error -- four causes, one consequence, which is that the
            job is scored on its snippet instead and the row says so.
        reason: Why ``ok`` is ``False``, for the row's tooltip.
    """

    url: str
    text: str = ""
    truncated: bool = False
    ok: bool = True
    reason: str = ""


# --------------------------------------------------------------------------- #
# What the model reads out of a posting
# --------------------------------------------------------------------------- #


class JobRequirement(_Strict):
    """One thing a posting asks of a candidate.

    Attributes:
        id: ``R-01``-style identifier, assigned by the app after parsing so the
            ids are dense and stable regardless of what the model emitted.
        text: The requirement in one line.
        must_have: Whether the posting states it as required rather than
            preferred. Weighted differently in the score, because a missing
            must-have is a screen-out and a missing nice-to-have is not.
        category: What kind of requirement it is.
    """

    id: str = ""
    text: str
    must_have: bool = True
    category: RequirementCategory = "hard_skill"


class RequirementAssessment(_Strict):
    """The model's verdict on one requirement, before the app checks it.

    "Before the app checks it" is the important part. The verdict is a claim,
    and :mod:`app.services.scoring` tests it against two things the model does
    not control: whether the quoted evidence actually appears in the resume, and
    how well the requirement matches any resume line at all. A verdict that
    survives both becomes score; one that does not gets demoted.

    Attributes:
        requirement_id: The ``R-nn`` this answers.
        status: Covered, partially covered, or missing.
        evidence: The resume line the model says shows it. Quoted, not
            paraphrased, so containment can be checked.
        note: One short sentence of reasoning, shown in the expanded row.
    """

    requirement_id: str
    status: CoverageStatus = "missing"
    evidence: str = ""
    note: str = ""


class JobAssessment(_Strict):
    """One model call's whole read of one posting.

    Requirements and verdicts arrive together, in a single call, on purpose:
    two calls per job would double both the latency and the token spend of the
    most expensive part of a run, and the second call would only be re-reading
    text the first one already had in context.

    Attributes:
        company: Employer name as the posting states it.
        title: Role title as the posting states it, which is often cleaner than
            the search result title.
        location: Where the job is, as stated.
        remote: Whether the posting says the role is remote.
        requirements: What the posting asks for.
        assessments: One verdict per requirement.
    """

    company: str = ""
    title: str = ""
    location: str = ""
    remote: bool = False
    requirements: list[JobRequirement] = Field(default_factory=list)
    assessments: list[RequirementAssessment] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# What the app computes
# --------------------------------------------------------------------------- #


class ScoreBreakdown(BaseModel):
    """How a deep score was arrived at, in numbers a reader can add up.

    Attributes:
        must_have_total: How many required items the posting stated.
        must_have_covered: How many of those the resume covers outright.
        must_have_partial: How many it covers partially.
        nice_to_have_total: How many preferred items the posting stated.
        nice_to_have_covered: How many of those the resume covers outright.
        nice_to_have_partial: How many it covers partially.
        must_have_score: 0-100 over the required items alone.
        nice_to_have_score: 0-100 over the preferred items alone, or ``None``
            when the posting stated none -- a posting with no preferred section
            must not cost the candidate the points allocated to it.
        demoted: Requirement ids whose claimed coverage was demoted because the
            evidence did not hold up. Surfaced rather than hidden: it is the
            difference between a score and a score you can trust.
    """

    must_have_total: int = 0
    must_have_covered: int = 0
    must_have_partial: int = 0
    nice_to_have_total: int = 0
    nice_to_have_covered: int = 0
    nice_to_have_partial: int = 0
    must_have_score: float = 0.0
    nice_to_have_score: float | None = None
    demoted: list[str] = Field(default_factory=list)


class ScoredJob(BaseModel):
    """A job as the user finally sees it: a link, a score, and the reasoning.

    Attributes:
        hit: The search result this came from.
        tier: Whether the score came from reading the posting or from ranking a
            snippet.
        score: 0-100. For a deep score this is arithmetic over the verdicts; for
            a shallow one it is a rescaled similarity between the resume profile
            and the result's title and snippet. The two are never mixed, and the
            tier is always displayed next to the number.
        reason: One line saying what drove the score.
        company: Employer, from the assessment when there is one.
        title: Role title, preferring the posting's own over the search result's.
        location: Where the role is.
        remote: Whether the posting says remote.
        assessment: The full read of the posting, for deep rows.
        breakdown: The arithmetic, for deep rows.
        matching_mode: Which mode produced the similarity numbers behind this
            row.
        posting_ok: Whether the posting text was readable. ``False`` on a deep
            row means the job fell back to snippet scoring, which is why the
            tier can be shallow while the job was selected for deep scoring.
        error: Set when this job was selected for deep scoring and the scoring
            itself failed. The row still appears -- a job the user might want,
            found and linked, is worth more than a silently dropped one.
    """

    hit: JobHit
    tier: ScoreTier = "shallow"
    score: float = 0.0
    reason: str = ""
    company: str = ""
    title: str = ""
    location: str = ""
    remote: bool = False
    assessment: JobAssessment | None = None
    breakdown: ScoreBreakdown | None = None
    matching_mode: MatchingMode = "lexical"
    posting_ok: bool = True
    error: str = ""

    @property
    def display_title(self) -> str:
        """The best title available, preferring the posting's own."""
        return self.title or self.hit.title or self.hit.url


class RunSummary(BaseModel):
    """What one run did, in the numbers the UI has to state plainly.

    A run that found four jobs is not a broken run, and a run that deep-scored
    three of thirty is not a run that scored thirty. Both facts are shown.

    Attributes:
        queries: The search queries that were issued.
        sites: The whitelist that was in force.
        results_found: Results returned before filtering.
        results_kept: Results left after non-postings and duplicates were
            dropped.
        deep_scored: How many jobs were read in full.
        postings_unreadable: How many fetches came back too thin to score on.
        matching_mode: The mode the run's similarity numbers came from.
        notices: Non-fatal things worth telling the user -- a degraded mode, a
            truncated resume, a partial run that hit the deadline.
    """

    queries: list[str] = Field(default_factory=list)
    sites: list[str] = Field(default_factory=list)
    results_found: int = 0
    results_kept: int = 0
    deep_scored: int = 0
    postings_unreadable: int = 0
    matching_mode: MatchingMode = "lexical"
    notices: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    """Everything one run produced.

    Attributes:
        profile: How the resume was read.
        jobs: Scored jobs, best first.
        summary: What the run did.
    """

    profile: CandidateProfile
    jobs: list[ScoredJob] = Field(default_factory=list)
    summary: RunSummary = Field(default_factory=RunSummary)
