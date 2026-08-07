"""Turning estimated factors into a ranked backlog.

Everything in this module is pure arithmetic over :class:`FactorSet` values. It
imports no client and makes no call, which is what lets the UI re-rank on every
keystroke of an override without spending a token -- and what lets the central
guarantee be tested directly: the score in the table is the formula applied to
the factors in the table, for every row, always.

Three jobs live here:

* :func:`score_backlog` -- reconcile the estimator's reply against what was
  actually sent, apply user overrides, score, rank.
* :func:`describe_divergence` -- say where RICE and ICE disagree *and why*, which
  is the part a single-framework tool cannot show.
* :func:`lever_hint` -- what would have to be true for a row to overtake the one
  above it. Deterministic, computed by inverting the RICE formula, so it is
  exactly as trustworthy as the score itself.
"""

from collections.abc import Iterable, Mapping
from statistics import median

from app.core.logging import get_logger
from app.models.schemas import (
    BacklogEstimate,
    BacklogInput,
    FactorSet,
    FeatureEstimate,
    RankedBacklog,
    ScoredFeature,
)
from app.services.scales import EFFORT_LADDER, MAX_REACH, ice_score, rice_score

logger = get_logger(__name__)

#: How many rows each framework's "top" list holds when reporting divergence.
#: Three is what fits in a planning conversation.
TOP_N = 3

#: How many divergence notes to actually show. The symmetric difference of two
#: top-3 lists can hold six features, and six paragraphs explaining a twelve-item
#: backlog is a wall of text rather than an insight -- observed on the first live
#: run. The largest rank shifts are the ones worth the reader's attention.
MAX_DIVERGENCE_NOTES = 3

_FACTOR_FIELDS = ("reach", "impact", "confidence", "effort_months")


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------
def _apply_overrides(
    factors: FactorSet, override: Mapping[str, float] | None
) -> tuple[FactorSet, list[str]]:
    """Replace estimated factors with user-supplied ones.

    Args:
        factors: What the estimator produced.
        override: User values, keyed by factor name. Unknown keys are ignored;
            values equal to the estimate are not counted as overrides, so a user
            who opens the editor and changes nothing is not credited with the
            model's opinion.

    Returns:
        The effective factors, and the names of the ones the user changed.
    """
    if not override:
        return factors, []

    values = factors.model_dump()
    changed: list[str] = []
    for field in _FACTOR_FIELDS:
        if field not in override or override[field] is None:
            continue
        # Round-trip through FactorSet so the comparison happens *after*
        # snapping: nudging Effort from 2.0 to 2.1 lands back on 2.0 and is not
        # an override, because it changed no number the scorer will use.
        candidate = FactorSet.model_validate({**values, field: override[field]})
        if getattr(candidate, field) != getattr(factors, field):
            values[field] = getattr(candidate, field)
            changed.append(field)

    return FactorSet.model_validate(values), changed


def _sort_key(row: ScoredFeature, score: float, order: Mapping[str, int]) -> tuple:
    """Ordering for one framework's ranking.

    Score descending, then the tie-breaks: higher Confidence first (a tie
    resolved toward the better-evidenced feature), then lower Effort (the
    cheaper of two equals ships first), then the order the user typed them.
    Fully deterministic by construction -- input order is unique, so no two rows
    can compare equal.
    """
    return (
        -score,
        -row.factors.confidence,
        row.factors.effort_months,
        order[row.idea.id],
    )


def score_backlog(
    backlog: BacklogInput,
    estimate: BacklogEstimate,
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> RankedBacklog:
    """Score and rank a backlog from its estimated factors.

    Reconciliation is deliberate rather than trusting: the reply is matched back
    against the ids that were sent. Entries for ids that were never sent are
    dropped, duplicates keep the first, and features the reply skipped are
    reported in :attr:`RankedBacklog.unestimated` instead of being filled in with
    a default. A visible gap is honest; an invented factor set is not.

    Args:
        backlog: The parsed backlog, which defines the id set and the input order.
        estimate: The estimator's reply.
        overrides: Per-feature user edits, keyed by feature id then factor name.

    Returns:
        The ranked backlog, ordered by RICE rank.
    """
    order = {feature.id: index for index, feature in enumerate(backlog.features)}
    by_id: dict[str, FeatureEstimate] = {}
    for item in estimate.estimates:
        if item.id not in order:
            logger.warning("Estimator returned unknown feature id %r; dropped", item.id)
            continue
        if item.id in by_id:
            logger.warning("Estimator returned feature id %r twice; keeping the first", item.id)
            continue
        by_id[item.id] = item

    rows: list[ScoredFeature] = []
    unestimated: list[str] = []
    for feature in backlog.features:
        item = by_id.get(feature.id)
        if item is None:
            unestimated.append(feature.id)
            continue

        factors, changed = _apply_overrides(item.factors(), (overrides or {}).get(feature.id))
        rows.append(
            ScoredFeature(
                idea=feature,
                factors=factors,
                rationales=item.rationales(),
                assumptions=list(item.assumptions),
                overridden=changed,
                rice=rice_score(
                    factors.reach, factors.impact, factors.confidence, factors.effort_months
                ),
                ice=ice_score(factors.impact, factors.confidence, factors.effort_months),
                # Placeholders: ranks need the whole set, assigned just below.
                rice_rank=0,
                ice_rank=0,
            )
        )

    if unestimated:
        logger.warning(
            "Estimator skipped %d feature(s): %s", len(unestimated), ", ".join(unestimated)
        )

    for rank, row in enumerate(sorted(rows, key=lambda r: _sort_key(r, r.rice, order)), start=1):
        row.rice_rank = rank
    for rank, row in enumerate(sorted(rows, key=lambda r: _sort_key(r, r.ice, order)), start=1):
        row.ice_rank = rank

    rows.sort(key=lambda row: row.rice_rank)
    return RankedBacklog(
        rows=rows,
        unestimated=unestimated,
        divergence=describe_divergence(rows, estimate.reach_unit),
        reach_unit=estimate.reach_unit,
    )


# ---------------------------------------------------------------------------
# Where the two frameworks disagree
# ---------------------------------------------------------------------------
def _reach_is_the_story(row: ScoredFeature, median_reach: float) -> bool:
    """Whether this row's rank shift is explained by Reach.

    Reach is the only factor RICE reads and ICE cannot see, so it is the usual
    suspect -- but not the only one: ICE's Ease bands compress Effort
    differences that RICE divides by directly. Attributing every shift to Reach
    would be a tidier sentence and an occasionally false one.
    """
    if row.rank_shift > 0:  # ICE ranks it higher
        return row.factors.reach < median_reach
    return row.factors.reach > median_reach


def describe_divergence(rows: Iterable[ScoredFeature], reach_unit: str = "users") -> list[str]:
    """Explain where the RICE and ICE top lists differ, and which factor did it.

    This is the most useful thing the app produces. Two frameworks agreeing tells
    a reader almost nothing; two frameworks disagreeing tells them exactly where
    the *choice of framework* is deciding the roadmap.

    Args:
        rows: Scored features, already ranked.
        reach_unit: What the Reach numbers count, so the note says "accounts"
            when the estimate said accounts. A note that silently renames the
            unit is the same defect the unit exists to prevent.

    Returns:
        Up to :data:`MAX_DIVERGENCE_NOTES` lines, biggest rank shift first, or a
        single line noting agreement. Empty when there are too few rows for a top
        list to mean anything.
    """
    rows = list(rows)
    if len(rows) < 2:
        return []

    rice_top = {row.idea.id for row in rows if row.rice_rank <= TOP_N}
    ice_top = {row.idea.id for row in rows if row.ice_rank <= TOP_N}

    if rice_top == ice_top:
        return [
            f"RICE and ICE pick the same top {min(TOP_N, len(rows))}. That is agreement, not "
            "corroboration — ICE's factors are derived from the same estimates RICE uses, so the "
            "two can only disagree about weighting, never about the underlying read."
        ]

    median_reach = median(row.factors.reach for row in rows)
    divergent = [row for row in rows if row.idea.id in rice_top ^ ice_top]
    # Biggest movers first: a feature that swings eight places is the one the
    # choice of framework is actually deciding.
    divergent.sort(key=lambda row: (-abs(row.rank_shift), min(row.rice_rank, row.ice_rank)))
    return [
        _divergence_note(row, median_reach, reach_unit) for row in divergent[:MAX_DIVERGENCE_NOTES]
    ]


def _divergence_note(row: ScoredFeature, median_reach: float, reach_unit: str) -> str:
    """One sentence on why a single feature sits in one top list and not the other."""
    title = row.idea.title
    ranks = f"RICE #{row.rice_rank}, ICE #{row.ice_rank}"

    if _reach_is_the_story(row, median_reach):
        if row.rank_shift > 0:
            return (
                f"**{title}** — {ranks}. It reaches {row.factors.reach:,.0f} {reach_unit}/quarter against a "
                f"backlog median of {median_reach:,.0f}, and ICE has no Reach term to notice that. "
                "ICE is ranking it on impact and ease alone."
            )
        return (
            f"**{title}** — {ranks}. Its reach of {row.factors.reach:,.0f} {reach_unit}/quarter is what "
            f"carries it (backlog median {median_reach:,.0f}); ICE is blind to Reach, so it drops."
        )

    direction = "up" if row.rank_shift > 0 else "down"
    return (
        f"**{title}** — {ranks}. Reach does not explain this one: it moves {direction} because ICE's "
        f"Ease band ({row.factors.ease}/10) flattens an Effort of {row.factors.effort_months:g} "
        "person-months that RICE divides by directly."
    )


# ---------------------------------------------------------------------------
# What would move a feature up
# ---------------------------------------------------------------------------
def lever_hint(
    row: ScoredFeature,
    target: ScoredFeature,
    reach_ceiling: float = MAX_REACH,
    reach_unit: str = "users",
) -> str | None:
    """State what would have to change for `row` to reach `target`'s RICE score.

    Computed by inverting the RICE formula rather than asked of a model, for the
    same reason the score itself is: a number in a planning conversation has to
    be reproducible. Levers that cannot be pulled are omitted -- an Effort below
    the ladder floor is not a plan, and a Reach nobody in this backlog achieves
    is not a market.

    That ceiling is not cosmetic. The first live run suggested "Dark mode
    overtakes Keyboard shortcuts if Reach reaches 24,000/quarter" for a product
    with 12,000 seats. A lever that cannot be pulled is worse than no lever: it
    reads like advice.

    Args:
        row: The feature to move up.
        target: The feature immediately above it in the RICE ranking.
        reach_ceiling: The largest Reach worth suggesting. Callers pass the
            backlog's own maximum, which is the best available proxy for the
            addressable base -- the app is never told what that base is.
        reach_unit: What the Reach numbers count.

    Returns:
        A sentence naming the achievable levers, or ``None`` when neither factor
        can close the gap on its own.
    """
    factors = row.factors
    quality = factors.impact * factors.confidence
    if target.rice <= row.rice or quality <= 0:
        return None

    levers: list[str] = []

    needed_reach = target.rice * factors.effort_months / quality
    if needed_reach <= min(reach_ceiling, MAX_REACH):
        levers.append(
            f"Reach reaches {needed_reach:,.0f} {reach_unit}/quarter (now {factors.reach:,.0f})"
        )

    if factors.reach > 0:
        needed_effort = factors.reach * quality / target.rice
        if needed_effort >= EFFORT_LADDER[0]:
            levers.append(
                f"Effort drops to {needed_effort:.2g} person-months (now {factors.effort_months:g})"
            )

    if not levers:
        return None
    return f"Overtakes **{target.idea.title}** if " + ", or if ".join(levers) + "."


def attach_levers(ranked: RankedBacklog) -> dict[str, str]:
    """Build the lever hint for every row that has one above it.

    The reach ceiling is the backlog's own largest Reach. Nothing tells this app
    how many users the product has, but the widest-reaching feature on the list
    is a reasonable stand-in, and it is the number the user can sanity-check.

    Returns:
        Feature id -> hint, omitting the top row and any row whose gap no single
        factor can close.
    """
    ordered = sorted(ranked.rows, key=lambda row: row.rice_rank)
    if not ordered:
        return {}

    ceiling = max(row.factors.reach for row in ordered)
    hints: dict[str, str] = {}
    for above, row in zip(ordered, ordered[1:], strict=False):
        hint = lever_hint(row, above, reach_ceiling=ceiling, reach_unit=ranked.reach_unit)
        if hint is not None:
            hints[row.idea.id] = hint
    return hints
