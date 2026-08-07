from app.models.schemas import CandidateProfile, SearchFilters
from app.services import queries


def test_user_role_leads_every_query(profile: CandidateProfile) -> None:
    filters = SearchFilters(role="Product Manager", sites=["linkedin.com"])
    built = queries.build_queries(profile, filters, limit=4)

    assert built, "a run with a typed role must produce queries"
    assert built[0].lower().startswith("senior product manager")


def test_resume_titles_are_used_when_no_role_is_typed(profile: CandidateProfile) -> None:
    built = queries.build_queries(profile, SearchFilters(), limit=4)
    assert "backend engineer" in built[0].lower()


def test_queries_are_distinct_and_capped(profile: CandidateProfile) -> None:
    built = queries.build_queries(profile, SearchFilters(), limit=3)
    assert len(built) <= 3
    assert len(set(built)) == len(built)


def test_location_and_remote_reach_the_query(profile: CandidateProfile) -> None:
    filters = SearchFilters(location="Berlin", remote_only=True)
    built = queries.build_queries(profile, filters, limit=1)
    assert "remote" in built[0].lower()
    assert "berlin" in built[0].lower()


def test_mid_level_adds_no_seniority_word(profile: CandidateProfile) -> None:
    mid = profile.model_copy(update={"seniority": "mid"})
    built = queries.build_queries(mid, SearchFilters(), limit=1)
    assert "mid" not in built[0].lower()


def test_an_empty_profile_still_produces_a_query() -> None:
    built = queries.build_queries(CandidateProfile(), SearchFilters(), limit=2)
    assert built and built[0].strip()


def test_queries_never_carry_resume_prose(profile: CandidateProfile) -> None:
    # The summary and highlights can quote a resume line verbatim, and a query
    # leaves this app. Neither may appear in one.
    loud = profile.model_copy(
        update={
            "summary": "Priya Raman, priya.raman@example.com, backend engineer",
            "highlights": ["Led a team of 4 engineers across two payment integrations"],
        }
    )
    built = " ".join(queries.build_queries(loud, SearchFilters(), limit=4)).lower()
    assert "priya" not in built
    assert "@" not in built
    assert "led a team" not in built


def test_weak_skills_are_left_out(profile: CandidateProfile) -> None:
    noisy = profile.model_copy(update={"skills": ["agile", "jira", "Kubernetes"]})
    built = " ".join(queries.build_queries(noisy, SearchFilters(), limit=4)).lower()
    assert "jira" not in built
    assert "kubernetes" in built


def test_limit_of_zero_returns_nothing(profile: CandidateProfile) -> None:
    assert queries.build_queries(profile, SearchFilters(), limit=0) == []
