"""Turning a candidate profile plus filters into search queries.

Pure functions over data. No network, no model, no settings beyond a cap -- which
means the thing most likely to make a run return nothing useful is also the
thing that is completely testable offline.

Three rules shape what comes out.

**A query is a role, not a resume.** Search engines index job postings, and a
posting is written around a title, a few technologies, and a place. Pasting a
candidate's whole skill list into one query matches nothing, because no posting
contains all of it. So each query stays short and carries one angle.

**Nothing that identifies the candidate goes into a query.** A search query
leaves this app and is logged by a third party. :class:`CandidateProfile` has no
name, email, or phone field to begin with, and the query builder only reads
titles, skills, domains, and the user's own filters -- never the summary, never a
highlight, both of which can quote a resume line verbatim.

**The user's stated intent wins over the resume's history.** When a target role
is typed in, it leads every query. That is the career-switcher case, and getting
it backwards is how a search for "product manager" returns eight backend jobs.
"""

import re

from app.models.schemas import CandidateProfile, SearchFilters

#: Words a search engine will match against half the postings on the internet.
#: Dropped from titles before they become queries, because a query of "senior
#: engineer" plus these adds length without adding a constraint.
_NOISE = frozenset(
    """
    experienced hands on hands-on strong solid excellent proven demonstrated
    passionate motivated dynamic rockstar ninja guru wizard
    ii iii iv i v
    """.split()
)

#: Characters that mean something to a search engine's parser and nothing to a
#: job title.
_PUNCTUATION = re.compile(r"[^\w+#/&. -]+")

#: Skills whose names are too generic to narrow a search. A query with "agile"
#: in it is not a narrower query.
_WEAK_SKILLS = frozenset(
    """
    agile scrum kanban jira confluence git github gitlab excel powerpoint word
    communication leadership teamwork collaboration problem-solving
    """.split()
)

#: Tavily rejects very long queries, and a long query is a worse query anyway.
_MAX_QUERY_CHARS = 220


def clean_term(term: str) -> str:
    """Reduce one title or skill to something worth putting in a query."""
    cleaned = _PUNCTUATION.sub(" ", term or "")
    words = [word for word in cleaned.split() if word.lower() not in _NOISE]
    return " ".join(words).strip()


def _seniority_word(profile: CandidateProfile, filters: SearchFilters) -> str:
    """The level to put in a query, or an empty string to leave it out.

    ``mid`` deliberately produces nothing: postings for mid-level roles are not
    titled "mid", so adding the word narrows the search to the handful of
    postings that happen to use it.
    """
    level = filters.seniority or profile.seniority
    return {"junior": "junior", "senior": "senior", "lead": "lead", "mid": ""}.get(level, "")


def _place_words(filters: SearchFilters) -> list[str]:
    """Location terms, in the order a posting would carry them."""
    words: list[str] = []
    if filters.remote_only:
        words.append("remote")
    if filters.location.strip():
        words.append(clean_term(filters.location))
    return [word for word in words if word]


def build_queries(
    profile: CandidateProfile,
    filters: SearchFilters,
    *,
    limit: int,
) -> list[str]:
    """Build the run's search queries, most important first.

    Each query is one angle on the same search, because issuing four variations
    of a single phrasing costs four credits and returns one result set. The
    angles are: the plain role, the role plus the candidate's strongest
    technologies, the second role the resume evidences, and the role plus a
    domain.

    Args:
        profile: How the resume was read.
        filters: What the user asked to narrow to.
        limit: Maximum queries to return. Every query is a paid credit, so this
            is a budget, not a hint.

    Returns:
        Between one and `limit` distinct queries. Never empty: if the resume
        yielded no titles and the user typed nothing, the skills alone still
        make a usable query, and a generic "jobs" query is the last resort --
        an empty list here would mean a run that searched for nothing and
        reported "no results", which reads as a broken app rather than an
        under-specified one.
    """
    if limit <= 0:
        return []

    titles = [clean_term(title) for title in ([filters.role] if filters.role else [])]
    titles += [clean_term(title) for title in profile.titles]
    titles = _dedupe([title for title in titles if title])

    skills = _dedupe(
        [
            clean_term(skill)
            for skill in profile.skills
            if clean_term(skill) and skill.lower() not in _WEAK_SKILLS
        ]
    )
    domains = _dedupe([clean_term(domain) for domain in profile.domains if clean_term(domain)])

    level = _seniority_word(profile, filters)
    place = _place_words(filters)
    lead_title = titles[0] if titles else " ".join(skills[:2])

    candidates: list[str] = []

    if lead_title:
        candidates.append(_assemble([level, lead_title, "jobs"], place))
        if skills:
            candidates.append(_assemble([level, lead_title] + skills[:3], place))
    if len(titles) > 1:
        candidates.append(_assemble([titles[1], "jobs"], place))
    if lead_title and domains:
        candidates.append(_assemble([lead_title, domains[0], "jobs"], place))
    if skills and not lead_title:
        candidates.append(_assemble(skills[:4] + ["jobs"], place))

    queries = _dedupe([query for query in candidates if query])
    if not queries:
        queries = [_assemble(["jobs hiring"], place) or "jobs hiring"]

    return queries[:limit]


def _assemble(terms: list[str], place: list[str]) -> str:
    """Join terms and location words into one capped query string."""
    words = [term.strip() for term in terms + place if term and term.strip()]
    query = " ".join(words)
    query = re.sub(r"\s+", " ", query).strip()
    return query[:_MAX_QUERY_CHARS].strip()


def _dedupe(values: list[str]) -> list[str]:
    """De-duplicate case-insensitively, preserving first-seen order."""
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            kept.append(value)
    return kept
