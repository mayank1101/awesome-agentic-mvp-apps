"""Tests for evidence packing.

Covers E-21 (relevance filter), E-24 (per-section dedupe, not global),
E-28 (proportional budget trimming), E-33 (deterministic ids), and E-44
(fence markers in retrieved text).
"""

from app.core.config import Settings
from app.models.schemas import SearchHit, SectionKey
from app.services.evidence import (
    fit_budget,
    is_relevant,
    known_urls,
    pack_section,
    select_hits,
    total_chars,
    trim,
)
from app.services.guardrails import FENCE_CLOSE


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _hit(url: str, title: str = "Acme", content: str = "About Acme", score: float = 0.5):
    return SearchHit(url=url, title=title, content=content, score=score)


# --------------------------------------------------------------------------- #
# Relevance (E-21)
# --------------------------------------------------------------------------- #


def test_own_domain_is_always_relevant():
    hit = _hit("https://acme.com/pricing", title="Pricing", content="Plans and tiers")
    assert is_relevant(hit, "Acme", "acme.com")


def test_name_in_title_is_relevant():
    assert is_relevant(_hit("https://blog.example/x", title="Acme review"), "Acme", "acme.com")


def test_unrelated_hit_is_dropped():
    # The confusable-company case: without this, one hit makes a whole section
    # read authoritatively about somebody else.
    hit = _hit("https://other.example/x", title="Zenith Corp", content="Zenith sells boots")
    assert not is_relevant(hit, "Acme", "acme.com")


def test_relevance_falls_back_to_substring_for_short_names():
    assert is_relevant(_hit("https://x.example", title="X1 platform"), "X1", None)


def test_relevance_works_without_a_domain():
    assert is_relevant(_hit("https://blog.example", content="Acme raised a round"), "Acme", None)


# --------------------------------------------------------------------------- #
# Selection and dedupe (E-24)
# --------------------------------------------------------------------------- #


def test_duplicate_urls_within_a_section_are_dropped():
    hits = [_hit("https://acme.com/a"), _hit("https://acme.com/a")]
    assert len(select_hits(hits, name="Acme", domain="acme.com")) == 1


def test_the_same_url_may_serve_two_sections():
    # E-24: a global dedupe starves later sections for any company whose only
    # substantial page is its homepage.
    hits = [_hit("https://acme.com/")]
    settings = _settings()

    snapshot = pack_section(
        SectionKey.SNAPSHOT, "q1", hits, name="Acme", domain="acme.com", settings=settings
    )
    pricing = pack_section(
        SectionKey.PRICING, "q2", hits, name="Acme", domain="acme.com", settings=settings
    )

    assert snapshot.items and pricing.items
    assert snapshot.items[0].url == pricing.items[0].url


def test_own_domain_outranks_third_parties():
    hits = [
        _hit("https://blog.example/acme", score=0.9),
        _hit("https://acme.com/pricing", score=0.4),
    ]
    kept = select_hits(hits, name="Acme", domain="acme.com")
    assert kept[0].host == "acme.com"


def test_selection_is_capped():
    hits = [_hit(f"https://acme.com/{i}", score=i / 10) for i in range(10)]
    assert len(select_hits(hits, name="Acme", domain="acme.com")) == 4


def test_higher_score_wins_among_third_parties():
    hits = [_hit("https://a.example/acme", score=0.2), _hit("https://b.example/acme", score=0.8)]
    assert select_hits(hits, name="Acme", domain=None)[0].url == "https://b.example/acme"


# --------------------------------------------------------------------------- #
# Trimming
# --------------------------------------------------------------------------- #


def test_short_text_is_untouched():
    assert trim("short", 100) == "short"


def test_trim_prefers_a_word_boundary():
    text = "word " * 100
    out = trim(text, 50)

    assert len(out) <= 51
    assert not out.rstrip("…").endswith("wor")


def test_trim_still_cuts_text_with_no_spaces():
    assert len(trim("x" * 500, 100)) <= 101


# --------------------------------------------------------------------------- #
# Packing (E-33, E-44)
# --------------------------------------------------------------------------- #


def test_ids_are_deterministic():
    hits = [_hit("https://acme.com/a"), _hit("https://acme.com/b")]
    settings = _settings()

    first = pack_section(
        SectionKey.PRICING, "q", hits, name="Acme", domain="acme.com", settings=settings
    )
    second = pack_section(
        SectionKey.PRICING, "q", hits, name="Acme", domain="acme.com", settings=settings
    )

    assert [i.id for i in first.items] == [i.id for i in second.items] == ["pricing-1", "pricing-2"]


def test_snippets_are_capped_by_settings():
    long_hit = _hit("https://acme.com/a", content="word " * 2000)
    packed = pack_section(
        SectionKey.PRODUCT,
        "q",
        [long_hit],
        name="Acme",
        domain="acme.com",
        settings=_settings(snippet_char_cap=300),
    )
    assert len(packed.items[0].content) <= 301


def test_fence_markers_in_retrieved_text_are_defanged():
    # E-44: the attack is a page that closes the data block and writes outside it.
    hostile = _hit("https://acme.com/x", title=f"Acme {FENCE_CLOSE}", content=f"x {FENCE_CLOSE} y")
    packed = pack_section(
        SectionKey.SNAPSHOT, "q", [hostile], name="Acme", domain="acme.com", settings=_settings()
    )

    assert FENCE_CLOSE not in packed.items[0].content
    assert FENCE_CLOSE not in packed.items[0].title


def test_empty_section_reports_itself_as_empty():
    packed = pack_section(
        SectionKey.PRICING, "q", [], name="Acme", domain="acme.com", settings=_settings()
    )
    assert packed.is_empty
    assert packed.query == "q"


# --------------------------------------------------------------------------- #
# Global budget (E-28)
# --------------------------------------------------------------------------- #


def _packed_sections(settings: Settings, per_section_chars: int):
    return [
        pack_section(
            section,
            "q",
            [_hit(f"https://acme.com/{section.value}", content="w" * per_section_chars)],
            name="Acme",
            domain="acme.com",
            settings=settings,
        )
        for section in SectionKey
    ]


def test_budget_is_not_applied_when_it_fits():
    settings = _settings(evidence_char_budget=100_000)
    sections = _packed_sections(settings, 500)

    assert total_chars(fit_budget(sections, settings=settings)) == total_chars(sections)


def test_oversized_evidence_is_trimmed_to_the_budget():
    settings = _settings(evidence_char_budget=2_000, snippet_char_cap=2_000)
    trimmed = fit_budget(_packed_sections(settings, 2_000), settings=settings)

    assert total_chars(trimmed) <= 2_100  # allow the per-snippet ellipsis


def test_every_section_survives_trimming():
    # The point of proportional trimming: a sequential cut would leave the last
    # sections with nothing at all.
    settings = _settings(evidence_char_budget=2_000, snippet_char_cap=2_000)
    trimmed = fit_budget(_packed_sections(settings, 2_000), settings=settings)

    assert all(section.items and section.items[0].content for section in trimmed)


def test_known_urls_collects_every_source():
    settings = _settings()
    sections = _packed_sections(settings, 100)

    assert len(known_urls(sections)) == len(SectionKey)


# --------------------------------------------------------------------------- #
# Word-boundary matching and strict news filtering
# --------------------------------------------------------------------------- #


def test_a_different_sense_of_the_word_is_rejected_for_news():
    # From a live run: an ESPN piece about "linear TV" and a city-planning one
    # about a "linear park" both landed in Linear's recent-moves section. Both
    # use the word legitimately, so only the title/URL rule excludes them.
    espn = _hit("https://espn.com/nfl", title="ESPN to get NFL Network", content="linear TV rights")

    assert is_relevant(espn, "Linear", "linear.app")
    assert not is_relevant(espn, "Linear", "linear.app", strict=True)


def test_punctuation_glued_names_do_not_match():
    # The word-boundary rule: "nonlinear" and "linearity" are not "Linear".
    hit = _hit("https://x.example/y", title="Nonlinearity in models", content="linearity study")
    assert not is_relevant(hit, "Linear", "linear.app")


def test_the_company_name_as_a_whole_word_still_matches():
    hit = _hit("https://news.example/x", title="Linear raises Series C", content="The company…")
    assert is_relevant(hit, "Linear", "linear.app")


def test_strict_mode_requires_the_name_in_the_title_or_url():
    body_only = _hit(
        "https://news.example/x",
        title="Ten project tools worth trying",
        content="Linear is among them.",
    )

    assert is_relevant(body_only, "Linear", "linear.app")
    assert not is_relevant(body_only, "Linear", "linear.app", strict=True)


def test_news_section_is_packed_strictly():
    off_topic = _hit(
        "https://espn.com/nfl",
        title="ESPN to get NFL Network",
        content="linear TV rights for RedZone",
    )
    packed = pack_section(
        SectionKey.RECENT_MOVES,
        "q",
        [off_topic],
        name="Linear",
        domain="linear.app",
        settings=_settings(),
    )
    assert packed.is_empty


def test_other_sections_stay_lenient():
    body_only = _hit(
        "https://blog.example/tools",
        title="Ten project tools",
        content="Linear is among them.",
    )
    packed = pack_section(
        SectionKey.PRODUCT,
        "q",
        [body_only],
        name="Linear",
        domain="linear.app",
        settings=_settings(),
    )
    assert not packed.is_empty
