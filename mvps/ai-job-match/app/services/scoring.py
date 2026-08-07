"""Turning per-requirement verdicts into a number.

The model is never asked for the score. It is asked, per requirement, whether the
resume covers it and which line shows that; the number is arithmetic over those
answers, here, in code. That ordering is the whole design:

* the same resume and posting produce the same score twice, because the only
  non-deterministic step is a per-requirement verdict with three possible values;
* the score can be explained line by line, which is what a candidate actually
  needs -- "68" helps nobody, "68, and here are the four must-haves you are
  missing" does;
* a model in a generous mood cannot inflate a total, only individual verdicts,
  and :func:`reconcile` checks those against the text similarity the app measured
  itself.

Four dimensions, non-overlapping by construction, each 0-100 internally and
weighted into the total. Weights renormalise when a dimension has nothing to
measure -- a posting with no "preferred" section should not cost the candidate
20 points for it.
"""

from app.core.logging import get_logger
from app.models.schemas import (
    DimensionScore,
    JobPosting,
    KeywordAction,
    MatchingMode,
    RequirementAssessment,
    ResumeAction,
)
from app.services.matching import RequirementMatch, lexical_similarity, tokenize

logger = get_logger(__name__)

#: Credit per coverage status.
_STATUS_CREDIT: dict[str, float] = {"covered": 1.0, "partial": 0.5, "missing": 0.0}

# The thresholds below are calibrated against measured `mistral-embed` output,
# not chosen by feel. Two resumes -- one covering nine requirements, one covering
# none of them -- produced:
#
#   genuinely covered:  best 0.706-0.899, margin over baseline 0.080-0.183
#   genuinely missing:  best 0.618-0.755, margin over baseline 0.046-0.148
#
# Those ranges *overlap*, which is the finding that matters: with hosted
# embeddings, absolute cosine cannot separate "the resume covers this" from "the
# resume is also professional English". Any single floor that catches the misses
# also demotes real matches, and demoting a real match is the failure that costs
# a candidate something.
#
# So the similarity check is not the judge here -- the model's per-requirement
# verdict is. This is a **backstop for gross mismatch**: it fires only when the
# best line is both weak in absolute terms *and* barely above the resume's own
# noise floor for that requirement. On the calibration set that demoted three of
# nine genuine misses and none of the nine genuine matches, which is the trade
# this app wants.
#
# Lexical mode is a different situation and keeps a plain floor: vocabulary
# coverage of an unrelated requirement really is 0.0, so there is no noise floor
# to subtract.

#: A "covered" verdict below this absolute similarity is a downgrade candidate.
_SUPPORT_FLOOR: dict[MatchingMode, float] = {"semantic": 0.70, "lexical": 0.20}

#: ...but in semantic mode only when the margin over the baseline is also thin.
#: Ignored in lexical mode, where the baseline is already ~0.
_MARGIN_FLOOR: dict[MatchingMode, float] = {"semantic": 0.08, "lexical": 0.0}

#: Below this absolute similarity, nothing in the resume is even adjacent, and
#: any verdict other than "missing" is discarded.
_MISS_FLOOR: dict[MatchingMode, float] = {"semantic": 0.50, "lexical": 0.05}

_BANDS: tuple[tuple[int, str], ...] = (
    (80, "Strong match"),
    (65, "Good match"),
    (50, "Partial match"),
    (35, "Weak match"),
    (0, "Poor match"),
)


def reconcile(
    assessments: list[RequirementAssessment],
    matches: list[RequirementMatch],
    mode: MatchingMode,
) -> list[RequirementAssessment]:
    """Attach measured similarity to each verdict, downgrading unsupported ones.

    A model asked "does this resume cover Kubernetes?" over a resume that never
    mentions it will sometimes answer yes, and will produce an evidence quote by
    paraphrasing something adjacent. The similarity number is the app's own
    measurement of whether *any* resume line is close to the requirement.

    It is a backstop, not the judge -- see the calibration note above the
    thresholds for why a stronger claim would not survive contact with real
    embedding output.

    Args:
        assessments: The model's verdicts, one per requirement.
        matches: The app's own text matches, one per requirement.
        mode: Which matching mode produced the similarities.

    Returns:
        New assessments, in the order of `matches`, each carrying its similarity
        and a note when a downgrade happened.
    """
    by_id = {assessment.requirement_id: assessment for assessment in assessments}
    support_floor = _SUPPORT_FLOOR[mode]
    margin_floor = _MARGIN_FLOOR[mode]
    miss_floor = _MISS_FLOOR[mode]

    reconciled: list[RequirementAssessment] = []
    for match in matches:
        assessment = by_id.get(match.requirement_id)
        if assessment is None:
            # A requirement the model skipped is a miss, not a silent omission:
            # dropping it would quietly raise the score by shrinking the
            # denominator.
            reconciled.append(
                RequirementAssessment(
                    requirement_id=match.requirement_id,
                    status="missing",
                    note="Not assessed; treated as not met.",
                    similarity=match.similarity,
                )
            )
            continue

        status = assessment.status
        note = assessment.note
        evidence = assessment.evidence

        weakly_supported = match.similarity < support_floor and match.margin <= margin_floor
        if status == "covered" and weakly_supported:
            status = "partial"
            note = (note + " " if note else "") + (
                "Downgraded: no resume line stands out as matching this requirement."
            )
            logger.info(
                "Downgraded %s to partial (similarity %.2f, margin %.2f)",
                match.requirement_id,
                match.similarity,
                match.margin,
            )

        if status != "missing" and match.similarity < miss_floor:
            status = "missing"
            evidence = ""
            note = "No supporting line found in the resume."
            logger.info(
                "Downgraded %s to missing (similarity %.2f)", match.requirement_id, match.similarity
            )

        reconciled.append(
            RequirementAssessment(
                requirement_id=match.requirement_id,
                status=status,
                evidence=evidence,
                note=note,
                similarity=match.similarity,
            )
        )
    return reconciled


def _coverage(
    assessments: list[RequirementAssessment],
    ids: set[str],
) -> tuple[float, int, int]:
    """Return percent coverage over `ids`, plus the covered and total counts."""
    relevant = [a for a in assessments if a.requirement_id in ids]
    if not relevant:
        return 0.0, 0, 0
    earned = sum(_STATUS_CREDIT[a.status] for a in relevant)
    fully = sum(1 for a in relevant if a.status == "covered")
    return 100.0 * earned / len(relevant), fully, len(relevant)


def _keyword_coverage(keywords: list[str], resume_text: str) -> tuple[float, list[str]]:
    """Fraction of posting keywords that appear in the resume, and which are absent.

    A crude proxy for what a keyword-filtering applicant-tracking system does,
    and labelled as such on screen. It is the one dimension here that rewards
    literal wording, which is exactly why it is only 10% of the total: writing
    for the filter at the cost of writing for the human is a bad trade at any
    higher weight.
    """
    if not keywords:
        return 0.0, []
    present = tokenize(resume_text)
    missing = [
        keyword for keyword in keywords if not (tokenize(keyword) and tokenize(keyword) <= present)
    ]
    return 100.0 * (len(keywords) - len(missing)) / len(keywords), missing


#: How much of a missing keyword's vocabulary must appear in a resume line
#: before the app will say the candidate has adjacent evidence for it. Lexical
#: rather than semantic on purpose: this decides whether to tell someone "you can
#: honestly use this word", and word overlap is the conservative test.
_KEYWORD_SUPPORT_FLOOR = 0.34

#: Tokens that carry no evidence on their own. Overlap alone said the resume
#: supported "ai agents" because it contained the word "AI" -- which would have
#: told a candidate they could honestly claim agent work on the strength of the
#: phrase "AI professional". A keyword with any distinctive token now needs that
#: token, not the filler around it.
_GENERIC_KEYWORD_TOKENS = frozenset(
    # One string, split at import time -- see the note in `provenance._ALLOWED`.
    [
        "ai",
        "artificial",
        "intelligence",
        "data",
        "digital",
        "tech",
        "technology",
        "technologies",
        "product",
        "products",
        "project",
        "projects",
        "program",
        "management",
        "manager",
        "system",
        "systems",
        "platform",
        "platforms",
        "tool",
        "tools",
        "framework",
        "frameworks",
        "architecture",
        "architectures",
        "concept",
        "concepts",
        "solution",
        "solutions",
        "service",
        "services",
        "application",
        "applications",
        "software",
        "development",
        "experience",
        "enabled",
        "based",
        "driven",
        "modern",
        "advanced",
        "strong",
        "hands-on",
        "end-to-end",
        "cross-functional",
    ]
)


def keyword_actions(
    keywords: list[str],
    resume_text: str,
    evidence_texts: list[str],
) -> list[KeywordAction]:
    """Work out which missing keywords the resume could honestly carry.

    Every keyword tool tells the applicant to paste the posting's terms into
    their resume. That advice is how people end up defending a skill they have
    never used. This splits the list: keywords with a close resume line behind
    them are ones the candidate can legitimately reword *into*, and keywords with
    nothing behind them are named as real gaps instead.

    Args:
        keywords: The posting's keywords.
        resume_text: Extracted resume text, for the "already present" check.
        evidence_texts: Resume lines, for the adjacency check.

    Returns:
        One entry per keyword absent from the resume, supported ones first.
    """
    present = tokenize(resume_text)
    actions: list[KeywordAction] = []

    for keyword in keywords:
        wanted = tokenize(keyword)
        if not wanted or wanted <= present:
            continue

        distinctive = wanted - _GENERIC_KEYWORD_TOKENS

        best_score, best_line = 0.0, ""
        for line in evidence_texts:
            score = lexical_similarity(keyword, line)
            if score <= best_score:
                continue
            # A keyword with a distinctive token needs *that* token present.
            # "ai agents" is not supported by "AI professional".
            if distinctive and not (distinctive & tokenize(line)):
                continue
            best_score, best_line = score, line

        supported = best_score >= _KEYWORD_SUPPORT_FLOOR
        actions.append(
            KeywordAction(
                keyword=keyword,
                supported=supported,
                evidence=best_line if supported else "",
                similarity=round(best_score, 3),
            )
        )

    actions.sort(key=lambda action: (not action.supported, -action.similarity))
    return actions


def build_dimensions(
    posting: JobPosting,
    assessments: list[RequirementAssessment],
    resume_text: str,
) -> list[DimensionScore]:
    """Compute the four weighted dimensions of the score.

    Args:
        posting: The parsed posting, for the must-have split and the keywords.
        assessments: Reconciled verdicts.
        resume_text: The extracted resume text, for keyword coverage.

    Returns:
        Dimensions in display order, with weights already renormalised so they
        sum to 1 across the dimensions that had something to measure.
    """
    must_ids = {r.id for r in posting.requirements if r.must_have}
    nice_ids = {r.id for r in posting.requirements if not r.must_have}

    must_pct, must_hit, must_total = _coverage(assessments, must_ids)
    nice_pct, nice_hit, nice_total = _coverage(assessments, nice_ids)

    supported = [a.similarity for a in assessments if a.status != "missing"]
    evidence_pct = 100.0 * (sum(supported) / len(supported)) if supported else 0.0

    keyword_pct, missing_keywords = _keyword_coverage(posting.keywords, resume_text)

    dimensions = [
        DimensionScore(
            name="Must-have requirements",
            earned=must_pct,
            weight=0.55 if must_total else 0.0,
            detail=(
                f"{must_hit} of {must_total} fully met"
                if must_total
                else "The posting states no explicit must-haves."
            ),
        ),
        DimensionScore(
            name="Preferred requirements",
            earned=nice_pct,
            weight=0.20 if nice_total else 0.0,
            detail=(
                f"{nice_hit} of {nice_total} fully met"
                if nice_total
                else "The posting states no preferred extras."
            ),
        ),
        DimensionScore(
            name="Evidence strength",
            earned=evidence_pct,
            weight=0.15 if supported else 0.0,
            detail=(
                f"Mean similarity {evidence_pct / 100:.2f} across {len(supported)} matched "
                "requirements"
                if supported
                else "Nothing in the resume matched a requirement."
            ),
        ),
        DimensionScore(
            name="Keyword coverage",
            earned=keyword_pct,
            weight=0.10 if posting.keywords else 0.0,
            detail=(
                f"{len(posting.keywords) - len(missing_keywords)} of {len(posting.keywords)} "
                f"posting keywords appear in the resume"
                + (f"; missing: {', '.join(missing_keywords[:6])}" if missing_keywords else "")
                if posting.keywords
                else "No keywords extracted from the posting."
            ),
        ),
    ]

    total_weight = sum(d.weight for d in dimensions)
    if total_weight <= 0:
        return dimensions
    return [
        DimensionScore(
            name=d.name,
            earned=round(d.earned, 1),
            weight=round(d.weight / total_weight, 4),
            detail=d.detail,
        )
        for d in dimensions
    ]


def overall_score(dimensions: list[DimensionScore]) -> int:
    """Sum the weighted dimensions into a 0-100 integer."""
    return max(0, min(100, round(sum(d.contribution for d in dimensions))))


def band_for(score: int) -> str:
    """Return the human label for a score, so a bare number is never shown."""
    return next(label for floor, label in _BANDS if score >= floor)


def fallback_actions(
    posting: JobPosting,
    assessments: list[RequirementAssessment],
    matches: list[RequirementMatch],
    keywords: list[KeywordAction],
) -> list[ResumeAction]:
    """Build actions from the report itself, with no model involved.

    The advice is the half of this app a candidate actually acts on, and a weaker
    free-tier model can return two vague lines for an eighteen-requirement
    posting. Everything needed to do better is already computed: which
    requirements are unmet, which resume line is closest to each, and which of
    the posting's words the resume could honestly carry.

    These are merged *after* the model's own actions, which are more specific
    when they are good. They are the floor, not the ceiling.

    Args:
        posting: The parsed posting.
        assessments: Reconciled verdicts.
        matches: The app's own requirement-to-evidence matches.
        keywords: Keyword advice from :func:`keyword_actions`.

    Returns:
        Actions in priority order.
    """
    requirements = {r.id: r for r in posting.requirements}
    evidence_by_id = {match.requirement_id: match.evidence for match in matches}
    actions: list[ResumeAction] = []

    for assessment in assessments:
        requirement = requirements.get(assessment.requirement_id)
        if requirement is None or assessment.status == "covered":
            continue

        evidence = evidence_by_id.get(assessment.requirement_id) or []
        must = requirement.must_have

        if assessment.status == "partial" and evidence:
            actions.append(
                ResumeAction(
                    priority=1 if must else 3,
                    section="Experience / Summary",
                    change=(
                        f"Strengthen how you show “{requirement.text}”. Your closest line is "
                        f"“{evidence[0][:160]}” — make the connection explicit and move it "
                        "earlier."
                    ),
                    rationale=(
                        f"{'Must-have' if must else 'Preferred'} requirement, currently read as "
                        "partial."
                    ),
                    requirement_ids=[assessment.requirement_id],
                    category="surface",
                )
            )
        elif assessment.status == "missing":
            actions.append(
                ResumeAction(
                    priority=2 if must else 4,
                    section="",
                    change=(
                        f"“{requirement.text}” is not shown anywhere in your resume. Do not add "
                        "it. If you have done this work and simply left it out, add the real "
                        "example; otherwise say so in the cover letter or apply as you are."
                    ),
                    rationale=f"{'Must-have' if must else 'Preferred'} requirement with no support.",
                    requirement_ids=[assessment.requirement_id],
                    category="gap",
                )
            )

    for keyword in keywords:
        if not keyword.supported:
            continue
        actions.append(
            ResumeAction(
                priority=3,
                section="Skills / Experience",
                change=(
                    f"Use the posting's term “{keyword.keyword}” where you already describe this "
                    f"work: “{keyword.evidence[:160]}”."
                ),
                rationale="The posting uses this wording and an automated filter will look for it.",
                category="reword",
            )
        )

    actions.sort(key=lambda action: action.priority)
    return actions


def merge_actions(
    model_actions: list[ResumeAction],
    computed: list[ResumeAction],
    *,
    limit: int = 10,
) -> list[ResumeAction]:
    """Combine the model's actions with the computed ones, without repeating.

    A computed action is dropped when the model already produced one for the same
    requirement -- the model's version names the section and the bullet, which is
    more use than a generic instruction about the same requirement.
    """
    covered = {rid for action in model_actions for rid in action.requirement_ids}
    merged = list(model_actions)

    for action in computed:
        if action.requirement_ids and set(action.requirement_ids) <= covered:
            continue
        merged.append(action)
        covered.update(action.requirement_ids)

    merged.sort(key=lambda action: (action.priority, action.category == "gap"))
    return merged[:limit]
