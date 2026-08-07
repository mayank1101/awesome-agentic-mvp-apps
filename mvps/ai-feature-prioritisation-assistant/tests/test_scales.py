"""Tests for the scales and the two formulas.

This is the arithmetic the whole app rests on, so the tests are about properties
rather than samples: monotonicity, the divisor floor, conservative tie-breaking.
"""

import math

import pytest

from app.services.scales import (
    CONFIDENCE_SCALE,
    EFFORT_LADDER,
    IMPACT_SCALE,
    MAX_REACH,
    clamp_reach,
    ease_from_effort,
    ice_confidence,
    ice_impact,
    ice_score,
    rice_score,
    snap_confidence,
    snap_effort,
    snap_impact,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3.0), (2.9, 3.0), (1.5, 1.0), (0.7, 0.5), (0.1, 0.25), (99, 3.0), (-4, 0.25)],
)
def test_snap_impact_lands_on_a_rung(value, expected):
    assert snap_impact(value) == expected


def test_snap_impact_breaks_ties_conservatively():
    # Exactly between 1 (medium) and 2 (high): a hedging model does not get to
    # round its way up.
    assert snap_impact(1.5) == 1.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, 1.0), (0.95, 1.0), (0.8, 0.8), (0.65, 0.5), (0.2, 0.5), (80, 0.8), (100, 1.0)],
)
def test_snap_confidence_handles_fractions_and_percentages(value, expected):
    assert snap_confidence(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(0, 0.25), (0.01, 0.25), (2.3, 2.0), (1000, 24.0)])
def test_snap_effort_respects_the_ladder_and_its_floor(value, expected):
    assert snap_effort(value) == expected


def test_every_snapped_value_is_a_declared_rung():
    for raw in (0, 0.13, 0.4, 1.1, 2.7, 5, 13, 99):
        assert snap_effort(raw) in EFFORT_LADDER
        assert snap_impact(raw) in IMPACT_SCALE
        assert snap_confidence(raw) in CONFIDENCE_SCALE


def test_clamp_reach_rounds_and_bounds():
    assert clamp_reach(1200.4) == 1200
    assert clamp_reach(-5) == 0
    assert clamp_reach(MAX_REACH * 10) == MAX_REACH


def test_ease_is_monotone_decreasing_in_effort():
    eases = [ease_from_effort(rung) for rung in EFFORT_LADDER]
    assert eases == sorted(eases, reverse=True)
    assert eases[0] == 10
    assert eases[-1] == 1


def test_ice_component_maps_are_monotone():
    impacts = sorted(IMPACT_SCALE)
    assert [ice_impact(value) for value in impacts] == sorted(ice_impact(v) for v in impacts)
    confidences = sorted(CONFIDENCE_SCALE)
    assert [ice_confidence(v) for v in confidences] == sorted(
        ice_confidence(v) for v in confidences
    )


def test_rice_is_the_stated_formula():
    assert rice_score(1000, 2, 0.8, 2) == pytest.approx(800.0)


def test_rice_snaps_before_it_divides():
    # 2.3 person-months is not a rung; the score must match the snapped 2.0,
    # so the number shown next to the factor is the number that was used.
    assert rice_score(1000, 2, 0.8, 2.3) == rice_score(1000, 2, 0.8, 2.0)


def test_rice_never_divides_by_zero():
    assert math.isfinite(rice_score(1000, 2, 0.8, 0))


def test_effort_floor_caps_how_much_a_trivial_item_can_win_by():
    # Without the floor, "it's basically free" would produce an unbounded score.
    trivial = rice_score(1000, 1, 1.0, 0.0001)
    one_week = rice_score(1000, 1, 1.0, 0.25)
    assert trivial == one_week


def test_ice_is_a_product_not_an_average():
    # A 10 on Impact must not hide a weak Ease. Product: 10*10*1 = 100.
    # An average would score this 7, ahead of a balanced 6/6/6 feature.
    lopsided = ice_score(impact=3, confidence=1.0, effort_months=24)
    balanced = ice_score(impact=1, confidence=0.8, effort_months=1.5)
    assert lopsided == 100
    assert balanced > lopsided


def test_ice_stays_in_range():
    for impact in IMPACT_SCALE:
        for confidence in CONFIDENCE_SCALE:
            for effort in EFFORT_LADDER:
                assert 1 <= ice_score(impact, confidence, effort) <= 1000


def test_ice_ignores_reach_by_construction():
    """The structural fact the whole divergence feature rests on."""
    assert ice_score(1, 0.8, 2) == ice_score(1, 0.8, 2)  # no reach parameter exists
    assert rice_score(10, 1, 0.8, 2) != rice_score(10_000, 1, 0.8, 2)
