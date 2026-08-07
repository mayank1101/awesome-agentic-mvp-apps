"""Pydantic models for every boundary in the app.

Three of these are parsed straight from model output (:class:`ResumeProfile`,
:class:`JobPosting`, :class:`AssessmentBatch`) and are therefore the app's real
validation layer: a model that returns the wrong shape fails here, loudly, at the
edge, rather than three functions later as an `AttributeError`.

The rest describe results this app computes itself. :class:`FitReport`'s score in
particular is **arithmetic over the assessments**, not a number the model was
asked for -- see :mod:`app.services.scoring`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RequirementCategory = Literal[
    "hard_skill",
    "experience",
    "education",
    "domain",
    "soft_skill",
    "responsibility",
]

CoverageStatus = Literal["covered", "partial", "missing"]

#: What kind of edit an action asks for. The split matters more than it looks:
#: the first four are changes that can genuinely raise the score, because the
#: evidence is already on the resume and is merely buried, vague, or worded
#: differently from the posting. ``gap`` is the opposite -- the candidate does not
#: have it, and the only honest advice is how to handle that, never how to write
#: around it.
ActionCategory = Literal["surface", "reword", "quantify", "restructure", "gap"]

MatchingMode = Literal["semantic", "lexical"]


#: Words a model writes into a string field when it means "absent". Seen on a
#: live run: a posting with no company name came back as the *string* ``"null"``,
#: which is truthy, so the report heading read "Senior AI/ML Engineer · null".
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
# The resume, as structure
# --------------------------------------------------------------------------- #


class ExperienceEntry(_Strict):
    """One job on the resume.

    Attributes:
        company: Employer name, verbatim from the resume.
        title: Role title, verbatim.
        start: Start date as written ("Jan 2021", "2021"). Never normalised --
            a normalised date is a new fact, and new facts are what this app
            refuses to produce.
        end: End date as written, or "Present".
        location: As written, when present.
        bullets: The achievement lines under this role, verbatim.
    """

    company: str = ""
    title: str = ""
    start: str = ""
    end: str = ""
    location: str = ""
    bullets: list[str] = Field(default_factory=list)


class EducationEntry(_Strict):
    """One qualification.

    Attributes:
        institution: School or university, verbatim.
        degree: Degree as written ("B.Tech", "MSc").
        field: Field of study, when stated separately.
        year: Graduation year or range, as written.
    """

    institution: str = ""
    degree: str = ""
    field: str = ""
    year: str = ""


class ProjectEntry(_Strict):
    """One project.

    Attributes:
        name: Project name.
        description: One-line description, when the resume has one.
        bullets: Detail lines, verbatim.
    """

    name: str = ""
    description: str = ""
    bullets: list[str] = Field(default_factory=list)


class ResumeProfile(_Strict):
    """The uploaded resume, parsed into sections.

    Attributes:
        name: Candidate name.
        email: Contact email as written.
        phone: Contact phone as written.
        location: Candidate location.
        links: Profile or portfolio URLs found in the resume.
        headline: The title line under the name, when there is one.
        summary: The summary or objective paragraph.
        skills: Skill tokens, as listed.
        experience: Roles, most recent first when the resume orders them that way.
        education: Qualifications.
        projects: Projects.
        certifications: Certification lines, verbatim.
    """

    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)
    headline: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    def evidence_texts(self) -> list[str]:
        """Return every line of the resume that can support a requirement.

        Bullets, skills, the summary, project lines, and role titles -- each as
        its own string, because matching a requirement against a whole resume
        returns "yes" for everything.

        Returns:
            Non-empty strings, in reading order, deduplicated.
        """
        chunks: list[str] = []
        if self.headline:
            chunks.append(self.headline)
        if self.summary:
            chunks.append(self.summary)
        chunks.extend(self.skills)
        for role in self.experience:
            role_label = " ".join(part for part in (role.title, role.company) if part)
            if role_label:
                chunks.append(role_label)
            chunks.extend(role.bullets)
        for project in self.projects:
            label = " ".join(part for part in (project.name, project.description) if part)
            if label:
                chunks.append(label)
            chunks.extend(project.bullets)
        for qualification in self.education:
            label = " ".join(
                part
                for part in (
                    qualification.degree,
                    qualification.field,
                    qualification.institution,
                )
                if part
            )
            if label:
                chunks.append(label)
        chunks.extend(self.certifications)

        seen: set[str] = set()
        unique: list[str] = []
        for chunk in chunks:
            cleaned = " ".join(chunk.split())
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                unique.append(cleaned)
        return unique

    def is_empty(self) -> bool:
        """Whether parsing produced nothing usable to match against."""
        return not (self.skills or self.experience or self.projects or self.education)


# --------------------------------------------------------------------------- #
# The job description, as structure
# --------------------------------------------------------------------------- #


class JobRequirement(_Strict):
    """One thing the posting asks for.

    Attributes:
        id: Stable id (`R-01`) so assessments, the report, and the tests can all
            refer to the same requirement without matching on prose.
        text: The requirement in one line, as close to the posting's wording as
            the extraction can keep it.
        category: What kind of requirement it is, which decides its weight.
        must_have: Whether the posting frames it as required rather than
            preferred. Drives the largest term in the score.
    """

    id: str = ""
    text: str
    category: RequirementCategory = "hard_skill"
    must_have: bool = False

    @field_validator("text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        """Reject blank requirement text, which would match everything."""
        if not value.strip():
            raise ValueError("requirement text must not be empty")
        return value


class JobPosting(_Strict):
    """The pasted job description, parsed.

    Attributes:
        title: Role title.
        company: Hiring company, when stated.
        seniority: Level as stated or implied ("Senior", "Entry-level").
        min_years_experience: Years demanded, when the posting gives a number.
            ``None`` when it does not -- which is not the same as zero, and the
            score treats it that way.
        requirements: The extracted requirement list, capped by config.
        keywords: Terms an applicant-tracking filter would likely key on.
    """

    title: str = ""
    company: str = ""
    seniority: str = ""
    min_years_experience: float | None = None
    requirements: list[JobRequirement] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Assessment and score
# --------------------------------------------------------------------------- #


class RequirementAssessment(_Strict):
    """The verdict on one requirement.

    Attributes:
        requirement_id: The `R-nn` this refers to.
        status: Whether the resume covers it, partly covers it, or misses it.
        evidence: The resume line that supports the verdict, quoted. Empty for a
            miss. This is what makes the score auditable rather than a vibe.
        note: One line of reasoning, shown in the gap list.
        similarity: Best cosine similarity (or lexical overlap) between the
            requirement and any resume line. Computed by the app, never by the
            model, and used to catch the case where the model claims coverage
            that no line supports.
    """

    requirement_id: str
    status: CoverageStatus = "missing"
    evidence: str = ""
    note: str = ""
    similarity: float = 0.0


class ResumeAction(_Strict):
    """One concrete edit the candidate should make, before applying.

    The app's other half. A score tells someone where they stand; this tells them
    what to do about it, which is the thing they actually came for.

    Attributes:
        priority: 1 is the most important. Ordered by how much this posting cares
            about the requirement behind the edit, not by how easy it is.
        section: Where the change goes ("Summary", "Experience - Lamipak").
        change: The edit, phrased as an instruction, specific enough to act on
            without re-reading the posting.
        rationale: Which requirement this serves and why it matters here.
        requirement_ids: The `R-nn`s this addresses, so the advice is traceable
            back to the posting rather than being generic resume coaching.
        category: See :data:`ActionCategory`. ``gap`` means the resume genuinely
            lacks it and the advice is about handling that honestly.
    """

    priority: int = 5
    section: str = ""
    change: str
    rationale: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    category: ActionCategory = "reword"

    @field_validator("change")
    @classmethod
    def _non_empty_change(cls, value: str) -> str:
        """An action with no instruction in it is not an action."""
        if not value.strip():
            raise ValueError("an action must say what to change")
        return value

    @property
    def is_gap(self) -> bool:
        """Whether this describes something the resume does not support."""
        return self.category == "gap"


class KeywordAction(BaseModel):
    """One posting keyword the resume does not contain, and what to do about it.

    Computed in code, not asked of the model. Each missing keyword is matched
    against the resume's own lines: when something close is already there, the
    advice is to use the posting's word for work the candidate has genuinely
    done. When nothing is close, it is named as a real gap -- because the failure
    mode of every keyword-optimisation tool is telling someone to paste in a
    skill they do not have.

    Attributes:
        keyword: The posting's term.
        supported: Whether the resume shows adjacent evidence for it.
        evidence: The closest resume line, when there is one.
        similarity: How close that line was.
    """

    keyword: str
    supported: bool = False
    evidence: str = ""
    similarity: float = 0.0


class AssessmentBatch(_Strict):
    """The model's verdicts and advice for one run, as returned.

    Verdicts and advice come back from a single call. They could be two, and two
    would read more cleanly -- but the free-tier ceiling this app is designed
    against is tokens *per minute*, and the advice call would re-send the same
    requirements and the same evidence to say something about them.

    Attributes:
        assessments: One entry per requirement. Missing entries are filled in as
            ``missing`` by the pipeline rather than dropped, so the count of
            requirements assessed always equals the count extracted.
        strengths: The matches worth leading with, in the candidate's favour.
        gaps: What the posting wants that the resume does not show.
        actions: Prioritised, specific edits to make before applying.
    """

    assessments: list[RequirementAssessment] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    actions: list[ResumeAction] = Field(default_factory=list)


class DimensionScore(BaseModel):
    """One weighted component of the overall score.

    Attributes:
        name: Display label.
        earned: Points earned, 0-100 within this dimension.
        weight: Share of the overall score, 0-1.
        detail: One line explaining what produced the number.
    """

    name: str
    earned: float
    weight: float
    detail: str = ""

    @property
    def contribution(self) -> float:
        """Points this dimension adds to the overall score."""
        return self.earned * self.weight


class FitReport(BaseModel):
    """The finished analysis.

    Attributes:
        overall_score: 0-100, computed in code from the dimensions below.
        band: Human label for the score, so a number is never shown alone.
        dimensions: The weighted components, in display order.
        assessments: Per-requirement verdicts, ordered as the requirements were.
        strengths: The strongest genuine matches, for the top of the report.
        gaps: What the posting wants and the resume does not show.
        actions: Prioritised edits, most important first.
        keyword_actions: Per missing keyword, whether the resume has adjacent
            evidence for it. Computed in code.
        matching_mode: Whether requirement matching used embeddings or the
            lexical fallback. Shown on screen, because it changes how much the
            similarity numbers mean.
        truncated_resume: Whether the resume text was cut to fit the cap.
        truncated_jd: Whether the job description was cut to fit the cap.
    """

    overall_score: int
    band: str
    dimensions: list[DimensionScore]
    assessments: list[RequirementAssessment]
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    actions: list[ResumeAction] = Field(default_factory=list)
    keyword_actions: list[KeywordAction] = Field(default_factory=list)
    matching_mode: MatchingMode = "lexical"
    truncated_resume: bool = False
    truncated_jd: bool = False


# --------------------------------------------------------------------------- #
# The rewrite
# --------------------------------------------------------------------------- #


class ChangeNote(_Strict):
    """One edit the rewrite made, for the change log shown beside the resume.

    Attributes:
        section: Where the change landed ("Summary", "Experience - Acme").
        change: What changed, in one line.
        reason: Which requirement or gap motivated it.
    """

    section: str = ""
    change: str = ""
    reason: str = ""


class TailoredResumeDraft(_Strict):
    """What the rewrite call returns, before the fabrication guard runs.

    Attributes:
        markdown: The full tailored resume as Markdown.
        changes: The rewrite's own account of what it changed.
    """

    markdown: str = ""
    changes: list[ChangeNote] = Field(default_factory=list)


class TailoredResume(BaseModel):
    """A tailored resume that has passed the fabrication guard.

    Attributes:
        markdown: The resume, sanitised, ready to render and to download.
        changes: The change log.
        flagged: Fragments the guard found absent from the original and did not
            block on, because strict mode was off. Empty in strict mode -- there,
            a violation raises instead.
    """

    markdown: str
    changes: list[ChangeNote] = Field(default_factory=list)
    flagged: list[str] = Field(default_factory=list)
