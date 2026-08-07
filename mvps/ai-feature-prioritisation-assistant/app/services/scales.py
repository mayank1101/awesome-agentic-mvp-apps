"""The scales both frameworks are defined on, and the arithmetic over them.

This module is the whole reason the app exists. Everything here is pure: given a
factor set it returns numbers, with no model, no I/O, and no configuration. That
is what lets an edited factor re-rank the table in microseconds, and it is what
makes the central claim testable -- *no score on screen ever came from a model*.

Two frameworks, one factor set
------------------------------

RICE is ``Reach x Impact x Confidence / Effort``. ICE is ``Impact x Confidence x
Ease``, each on 1-10. The obvious implementation asks the model for both factor
sets. That was rejected: two independent estimates let the same feature be
"2 person-months" under RICE and "Ease 9" under ICE, and there is no honest way
to explain that contradiction to a stakeholder.

So one factor set is estimated, and ICE's three inputs are *derived* from it by
the published mappings below. The consequence is worth stating plainly, because
the UI does: agreement between the two scores is not corroboration. What the
comparison shows is **what each formula ignores** -- ICE structurally cannot see
Reach, so a niche-but-easy win outranks a broad-but-costly one under ICE and
loses under RICE. That divergence is the interesting output.

Why the values are snapped
--------------------------

A model asked for Impact will happily answer ``1.5``, and a model asked for
Effort will answer ``2.3 person-months``. Both are false precision: the Intercom
Impact scale has five rungs, and nobody can tell 2.3 person-months from 2.
Snapping every incoming value onto a fixed ladder means two features described
with the same confidence land on the same rung, which is a precondition for
comparing their scores at all. Ties break *conservatively* -- a value exactly
between two rungs takes the less flattering one.
"""

from bisect import bisect_left

#: Impact, on the standard Intercom RICE scale, with the label each rung carries
#: in the UI and in the estimator's instructions.
IMPACT_SCALE: dict[float, str] = {
    3.0: "massive",
    2.0: "high",
    1.0: "medium",
    0.5: "low",
    0.25: "minimal",
}

#: Confidence, as the three RICE rungs. Percentages rather than a 1-10 scale,
#: because the factor multiplies the score and has to keep that meaning.
CONFIDENCE_SCALE: dict[float, str] = {
    1.0: "high — backed by evidence in the notes",
    0.8: "medium — reasoned, some evidence",
    0.5: "low — largely a guess",
}

#: Effort in person-months, snapped to a ladder that gets coarser as it grows --
#: the difference between 0.25 and 0.5 is a real planning distinction; the
#: difference between 18 and 19 is noise.
#:
#: The floor is deliberate and load-bearing. Effort is a *divisor*: an
#: unconstrained model that answers "0" for a trivial tweak produces a division
#: by zero, and one that answers "0.05" produces a RICE score twenty times the
#: rest of the backlog off the back of a rounding opinion. One week is the
#: smallest unit this framework can honestly express.
EFFORT_LADDER: tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    9.0,
    12.0,
    18.0,
    24.0,
)

#: Sanity ceiling on Reach. Not a business rule -- a guard against a model
#: answering "5000000000" for a B2B product with 4,000 accounts and swamping
#: every other row.
MAX_REACH = 100_000_000.0

#: RICE Impact -> ICE Impact (1-10). Monotone, and spread across the range so
#: the ICE product still discriminates once multiplied out.
_ICE_IMPACT: dict[float, int] = {0.25: 2, 0.5: 4, 1.0: 6, 2.0: 8, 3.0: 10}

#: RICE Confidence -> ICE Confidence (1-10).
_ICE_CONFIDENCE: dict[float, int] = {0.5: 5, 0.8: 8, 1.0: 10}

#: Effort (person-months) -> ICE Ease (1-10), as ``(inclusive upper bound, ease)``
#: pairs in ascending order. Monotone decreasing, by definition: more effort is
#: never more ease. Deliberately non-linear -- the gap between one week and one
#: month matters far more to a roadmap than the gap between six months and nine.
_EASE_BANDS: tuple[tuple[float, int], ...] = (
    (0.25, 10),
    (0.5, 9),
    (1.0, 8),
    (1.5, 7),
    (2.0, 6),
    (3.0, 5),
    (4.5, 4),
    (6.0, 3),
    (9.0, 2),
)
_EASE_FLOOR = 1


def _snap_to(value: float, allowed: tuple[float, ...]) -> float:
    """Return the rung of `allowed` nearest to `value`, ties going to the lower.

    Args:
        value: The incoming number, from a model or a user edit.
        allowed: Ascending rungs to snap onto.

    Returns:
        One of `allowed`, clamped to its endpoints.
    """
    if value <= allowed[0]:
        return allowed[0]
    if value >= allowed[-1]:
        return allowed[-1]

    index = bisect_left(allowed, value)
    lower, upper = allowed[index - 1], allowed[index]
    # `<=` rather than `<`: an exact midpoint takes the lower, less flattering
    # rung, so a model hedging at "1.5 impact" cannot round its way upward.
    return lower if (value - lower) <= (upper - value) else upper


def snap_impact(value: float) -> float:
    """Snap an Impact estimate onto the five-rung Intercom scale."""
    return _snap_to(float(value), tuple(sorted(IMPACT_SCALE)))


def snap_confidence(value: float) -> float:
    """Snap a Confidence estimate onto the three RICE rungs.

    Values above 1 are read as percentages, since "80" and "0.8" both turn up in
    model replies and mean the same thing.
    """
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return _snap_to(confidence, tuple(sorted(CONFIDENCE_SCALE)))


def snap_effort(months: float) -> float:
    """Snap an Effort estimate onto :data:`EFFORT_LADDER`, floored at 0.25."""
    return _snap_to(float(months), EFFORT_LADDER)


def clamp_reach(value: float) -> float:
    """Clamp a Reach estimate into ``[0, MAX_REACH]`` and drop the decimals.

    Reach counts people or accounts, so a fractional one is a category error;
    rounding it here means the number shown to the user is the number scored.
    """
    return float(min(max(round(float(value)), 0), int(MAX_REACH)))


def ice_impact(impact: float) -> int:
    """ICE Impact (1-10), derived from the RICE Impact rung."""
    return _ICE_IMPACT[snap_impact(impact)]


def ice_confidence(confidence: float) -> int:
    """ICE Confidence (1-10), derived from the RICE Confidence rung."""
    return _ICE_CONFIDENCE[snap_confidence(confidence)]


def ease_from_effort(effort_months: float) -> int:
    """ICE Ease (1-10), derived from Effort in person-months.

    ICE has no Effort term and RICE has no Ease term, so one of them has to be a
    function of the other for the two frameworks to describe the same feature.
    Effort is the estimated quantity because it is the one a team can actually
    argue about in units it uses.
    """
    effort = snap_effort(effort_months)
    for upper, ease in _EASE_BANDS:
        if effort <= upper:
            return ease
    return _EASE_FLOOR


def rice_score(reach: float, impact: float, confidence: float, effort_months: float) -> float:
    """Compute ``Reach x Impact x Confidence / Effort``.

    Args:
        reach: Users or accounts affected per quarter.
        impact: One of :data:`IMPACT_SCALE`.
        confidence: One of :data:`CONFIDENCE_SCALE`.
        effort_months: Person-months, at or above the ladder floor.

    Returns:
        The RICE score, rounded to two decimals so that two features whose
        factors are identical compare equal despite float arithmetic.
    """
    effort = snap_effort(effort_months)
    raw = clamp_reach(reach) * snap_impact(impact) * snap_confidence(confidence) / effort
    return round(raw, 2)


def ice_score(impact: float, confidence: float, effort_months: float) -> int:
    """Compute ``Impact x Confidence x Ease`` on the derived 1-10 scales.

    The three are multiplied rather than averaged. An average lets a 10 on
    Impact hide a 2 on Ease -- which is precisely the feature that quietly eats
    a quarter. A product punishes a weakness anywhere, and keeps ICE
    structurally comparable to RICE, which is multiplicative too.

    Returns:
        An integer in ``[1, 1000]``.
    """
    return ice_impact(impact) * ice_confidence(confidence) * ease_from_effort(effort_months)
