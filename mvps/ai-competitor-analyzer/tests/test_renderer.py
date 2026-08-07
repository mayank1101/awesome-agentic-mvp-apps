"""Tests for the renderer.

Covers E-23 (undated items dropped), E-40 (evidence ids in prose), E-43
(retrieved titles), E-45/SC-1 (URL allowlist), E-55 (download filename),
E-58 (a section with content always has sources), and SC-2 (six sections).
"""

from datetime import date

import pytest

from app.models.schemas import (
    NOT_FOUND,
    SECTION_TITLES,
    CompanyIdentity,
    EvidenceItem,
    SectionEvidence,
    SectionKey,
    SynthesisResult,
)
from app.services.renderer import download_filename, drop_undated_items, render_report

TODAY = date(2026, 8, 4)


def _identity(**overrides) -> CompanyIdentity:
    return CompanyIdentity(**{"name": "Acme", "domain": "acme.com", **overrides})


def _result(**overrides: str) -> SynthesisResult:
    payload = {key.value: f"{key.value} body" for key in SectionKey}
    # Recent moves needs a dated bullet or the renderer correctly empties it.
    payload["recent_moves"] = "- 2026-03-01: raised a Series B"
    payload.update(overrides)
    return SynthesisResult(**payload)


def _evidence(*, sections=None, title="Acme pricing", url="https://acme.com/pricing"):
    return tuple(
        SectionEvidence(
            section=section,
            query=f"{section.value} query",
            items=(
                EvidenceItem(id=f"{section.value}-1", title=title, url=url, content="Some text."),
            ),
        )
        for section in (sections or list(SectionKey))
    )


def _render(**overrides) -> str:
    kwargs = {
        "identity": _identity(),
        "result": _result(),
        "evidence": _evidence(),
        "generated_on": TODAY,
        **overrides,
    }
    return render_report(**kwargs)


# --------------------------------------------------------------------------- #
# Structure (SC-2)
# --------------------------------------------------------------------------- #


def test_all_six_headings_are_present():
    document = _render()
    for title in SECTION_TITLES.values():
        assert f"## {title}" in document


def test_sections_render_in_order():
    document = _render()
    positions = [document.index(f"## {SECTION_TITLES[section]}") for section in SectionKey]
    assert positions == sorted(positions)


def test_identity_is_at_the_top():
    # A wrong-company brief is the worst output; the banner is what makes it
    # visible in two seconds.
    document = _render()
    assert document.startswith("# Competitor brief: Acme")
    assert "acme.com" in document.split("##")[0]


def test_resolved_identity_carries_a_check_prompt():
    document = _render(identity=_identity(supplied_by_user=False))
    assert "check the domain" in document.lower()


def test_user_supplied_domain_has_no_check_prompt():
    document = _render(identity=_identity(supplied_by_user=True))
    assert "check the domain" not in document.lower()


def test_generated_date_is_stamped():
    assert "2026-08-04" in _render()


# --------------------------------------------------------------------------- #
# Empty sections (E-17, E-58)
# --------------------------------------------------------------------------- #


def test_section_without_evidence_says_not_found_and_names_the_query():
    evidence = (
        SectionEvidence(section=SectionKey.PRICING, query="acme pricing plans"),
        *_evidence(sections=[SectionKey.SNAPSHOT]),
    )
    document = render_report(_identity(), _result(), evidence, generated_on=TODAY)

    pricing = document.split(f"## {SECTION_TITLES[SectionKey.PRICING]}")[1].split("##")[0]
    assert NOT_FOUND in pricing
    assert "acme pricing plans" in pricing


def test_a_section_with_content_always_has_sources():
    # E-58: the inverse would be an unsupported claim presented as sourced.
    document = _render()
    for section in SectionKey:
        block = document.split(f"## {SECTION_TITLES[section]}")[1].split("\n## ")[0]
        assert "**Sources**" in block


def test_synthesis_failure_is_stated_plainly():
    document = _render(result=SynthesisResult.all_not_found(), synthesis_failed=True)

    assert "Synthesis failed" in document
    assert "sources listed are real" in document


# --------------------------------------------------------------------------- #
# Undated recent moves (E-23)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bullet",
    [
        "- 2026-03-01: raised a Series B",
        "- March 2026 — launched an API",
        "- 12 March 2026: acquired a rival",
        "- Q1 2026: opened an EU region",
        "- 2025: hired a new CTO",
    ],
)
def test_dated_bullets_survive(bullet: str):
    assert bullet.strip("- ") in drop_undated_items(bullet)


@pytest.mark.parametrize(
    "bullet",
    [
        "- raised a Series B recently",
        "- launched an API",
        "- has been growing fast",
    ],
)
def test_undated_bullets_are_dropped(bullet: str):
    assert drop_undated_items(bullet) == NOT_FOUND


def test_mixed_bullets_keep_only_the_dated_ones():
    text = "- 2026-03-01: raised a Series B\n- launched something\n- Q2 2026: opened an office"
    out = drop_undated_items(text)

    assert "Series B" in out
    assert "launched something" not in out
    assert "opened an office" in out


def test_not_found_passes_through_unchanged():
    assert drop_undated_items(NOT_FOUND) == NOT_FOUND


def test_undated_moves_render_as_not_found():
    document = _render(result=_result(recent_moves="- they grew a lot\n- no dates here"))
    block = document.split(f"## {SECTION_TITLES[SectionKey.RECENT_MOVES]}")[1].split("\n## ")[0]

    assert NOT_FOUND in block
    assert "grew a lot" not in block


# --------------------------------------------------------------------------- #
# Model output hygiene (E-40, SC-1)
# --------------------------------------------------------------------------- #


def test_evidence_ids_do_not_reach_the_page():
    document = _render(result=_result(pricing="Team is $10 per seat [pricing-1]."))

    assert "[pricing-1]" not in document
    assert "$10 per seat" in document


def test_invented_urls_are_stripped():
    # SC-1, the hard guarantee. The model never sees a URL, so anything
    # URL-shaped in its output was invented.
    document = _render(result=_result(pricing="See https://acme-pricing.example/fake for tiers."))

    assert "acme-pricing.example" not in document
    assert "[link removed]" in document


def test_retrieved_urls_survive():
    document = _render()
    assert "https://acme.com/pricing" in document


def test_active_html_from_the_model_is_escaped():
    document = _render(result=_result(product="<script>alert(1)</script>"))
    assert "<script" not in document


def test_hostile_retrieved_title_cannot_inject_a_link():
    # E-43: the sources list is the one place third-party text is rendered as-is.
    document = _render(evidence=_evidence(title="Pricing](https://evil.example) x"))

    assert "evil.example" not in document


# --------------------------------------------------------------------------- #
# Download (E-55, SC-9)
# --------------------------------------------------------------------------- #


def test_download_filename_is_slugged_and_dated():
    assert download_filename(_identity(name="Acme Corp / Ltd."), TODAY) == (
        "acme-corp-ltd-brief-2026-08-04.md"
    )


def test_download_filename_survives_a_non_latin_name():
    assert download_filename(_identity(name="日本電気"), TODAY) == "competitor-brief-2026-08-04.md"


def test_a_not_found_section_carries_no_sources():
    # E-58 inverted, and seen in a live run: four unrelated news links sat under
    # a "Not found in public sources" heading, reading as though they supported it.
    document = _render(result=_result(recent_moves="- undated thing\n- another undated thing"))
    block = document.split(f"## {SECTION_TITLES[SectionKey.RECENT_MOVES]}")[1].split("\n## ")[0]

    assert NOT_FOUND in block
    assert "**Sources**" not in block


def test_populated_sections_still_carry_sources():
    document = _render()
    block = document.split(f"## {SECTION_TITLES[SectionKey.PRICING]}")[1].split("\n## ")[0]
    assert "**Sources**" in block
