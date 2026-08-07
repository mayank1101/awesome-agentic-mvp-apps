from typing import Any

import pytest

from app.models.schemas import CandidateProfile
from app.services import profile as profile_service


def _patch_model(monkeypatch: pytest.MonkeyPatch, reply: CandidateProfile) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_complete_model(**kwargs: Any) -> CandidateProfile:
        calls.append(kwargs)
        return reply

    monkeypatch.setattr(profile_service, "complete_model", fake_complete_model)
    return calls


def test_the_resume_is_fenced(monkeypatch: pytest.MonkeyPatch, resume_text: str) -> None:
    calls = _patch_model(monkeypatch, CandidateProfile(titles=["Backend Engineer"]))

    profile_service.extract_profile(resume_text)

    assert "<<<UNTRUSTED_DOCUMENT" in calls[0]["user"]


def test_list_fields_are_capped(monkeypatch: pytest.MonkeyPatch, resume_text: str) -> None:
    # The prompt asks for at most 15 skills. This is what makes it a guarantee:
    # forty skills become a 400-character search query.
    _patch_model(
        monkeypatch,
        CandidateProfile(skills=[f"skill{index}" for index in range(40)]),
    )

    parsed = profile_service.extract_profile(resume_text)

    assert len(parsed.skills) == 15


def test_duplicate_and_blank_entries_are_removed(
    monkeypatch: pytest.MonkeyPatch, resume_text: str
) -> None:
    _patch_model(
        monkeypatch,
        CandidateProfile(
            titles=["Backend Engineer", "backend engineer", "  ", "Payments Engineer"]
        ),
    )

    parsed = profile_service.extract_profile(resume_text)

    assert parsed.titles == ["Backend Engineer", "Payments Engineer"]


def test_a_sentence_in_a_skill_field_is_dropped(
    monkeypatch: pytest.MonkeyPatch, resume_text: str
) -> None:
    _patch_model(
        monkeypatch,
        CandidateProfile(
            skills=[
                "Python",
                "Has extensive experience designing and operating distributed systems at scale",
            ]
        ),
    )

    parsed = profile_service.extract_profile(resume_text)

    assert parsed.skills == ["Python"]


def test_a_graduation_year_read_as_experience_is_discarded(
    monkeypatch: pytest.MonkeyPatch, resume_text: str
) -> None:
    _patch_model(monkeypatch, CandidateProfile(years_experience=2018))

    parsed = profile_service.extract_profile(resume_text)

    assert parsed.years_experience is None


def test_profile_text_carries_the_searchable_fields() -> None:
    text = CandidateProfile(
        titles=["Backend Engineer"],
        skills=["Python"],
        domains=["payments"],
        summary="Builds payment services.",
    ).profile_text()

    assert "Backend Engineer" in text
    assert "Python" in text
    assert "payments" in text


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Backend Engineer", "senior"),
        ("Staff Engineer", "lead"),
        ("Principal Data Scientist", "lead"),
        ("Junior Developer", "junior"),
        ("Backend Engineer", None),
        ("Engineering Manager", None),
    ],
)
def test_seniority_is_read_from_a_title(title: str, expected: str | None) -> None:
    assert profile_service.infer_seniority(title) == expected
