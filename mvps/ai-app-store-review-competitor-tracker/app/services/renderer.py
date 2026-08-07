"""Turning stats, reviews, and a gap analysis into the finished document.

One Markdown string feeds both the screen and the download, so they cannot
drift. Everything the model is not trusted with happens here:

* **Every gap's review excerpts are the app's own fetched text**, looked up by
  the ids the model cited -- never the model's own words for what a review
  said (SC-1).
* **A gap whose ids do not resolve to any fetched review is dropped
  entirely**, not rendered with zero evidence (SC-2).
* **Fence lookalikes and stray citation-bracket tokens are stripped** from
  generated prose.
* **Review excerpts and gap descriptions are sanitised** before they reach the
  page -- the excerpt under each gap is the one place third-party (review)
  text lands verbatim; there is no separate raw-review dump to sanitise a
  second time (see :func:`_render_summary`).
"""

import re
from datetime import date

from app.models.schemas import AppIdentity, FeatureGap, Platform, Review, ReviewStats
from app.services.guardrails import defang_fence_markers, sanitize_inline, sanitize_markdown

_STORE_LABEL = {Platform.IOS: "App Store", Platform.ANDROID: "Google Play"}

#: A bracketed review id the model may have echoed into prose despite being
#: asked not to (`[14393250680]`). Stripped defensively -- belt, not braces.
_ID_TOKEN = re.compile(r"\[\d{5,}\]")

_STARS = {5: "★★★★★", 4: "★★★★☆", 3: "★★★☆☆", 2: "★★☆☆☆", 1: "★☆☆☆☆"}

#: How many resolved excerpts to show under one gap. More than this is a wall
#: of quotes, not evidence.
_MAX_EXCERPTS_PER_GAP = 4


def _resolve_gap(gap: FeatureGap, by_id: dict[str, Review]) -> tuple[FeatureGap, list[Review]] | None:
    """Resolve one gap's citations against fetched reviews.

    Returns:
        The gap and its resolved reviews, or `None` if nothing resolved
        (SC-2) -- a gap with no real evidence behind it is not rendered.
    """
    resolved = [by_id[rid] for rid in gap.review_ids if rid in by_id]
    if not resolved:
        return None
    return gap, resolved


_SEVERITY_LABEL = {"high": "High impact", "medium": "Recurring", "low": "Narrower"}


def _render_gap(gap: FeatureGap, evidence: list[Review]) -> str:
    """Render one gap: heading, description, then its resolved excerpts."""
    description = _ID_TOKEN.sub("", defang_fence_markers(gap.description)).strip()
    description = sanitize_markdown(description)

    lines = [
        f"### {sanitize_inline(gap.title)}",
        "",
        f"**{_SEVERITY_LABEL.get(gap.severity, gap.severity.title())}** "
        f"· {len(evidence)} supporting review(s)",
        "",
        description,
        "",
        "**In their words:**",
        "",
    ]
    for review in evidence[:_MAX_EXCERPTS_PER_GAP]:
        stars = _STARS.get(review.rating, f"{review.rating}★")
        dated = review.updated.date().isoformat() if review.updated else "undated"
        excerpt = sanitize_inline(review.content)
        lines.append(f"> {excerpt}\n>\n> — {stars}, {dated}, v{review.version or '?'}")
        lines.append("")

    return "\n".join(lines)


def _render_snapshot(identity: AppIdentity, stats: ReviewStats, generated_on: date) -> str:
    """Render the app identity and rating-sample stats."""
    store = _STORE_LABEL[identity.platform]
    storefront = f"{store} ({identity.country.upper()})"

    lines = [
        f"# Review gap analysis: {identity.track_name}",
        "",
        f"**Developer:** {sanitize_inline(identity.artist_name)}"
        + (f" · **Category:** {identity.primary_genre_name}" if identity.primary_genre_name else ""),
        f"**Generated:** {generated_on.isoformat()} · **Storefront:** {storefront}",
        "",
    ]

    if identity.published_average_rating is not None:
        lines.append(
            f"**{store} rating (all-time):** {identity.published_average_rating:.1f}★ "
            f"across {identity.published_rating_count:,} ratings"
        )

    lines += [
        f"**This sample:** {stats.fetched_count} most recent reviews "
        f"(average {stats.average:.1f}★, {stats.critical_count} critical / ≤3★, "
        f"{stats.critical_share:.0%} of sample)",
        "",
        "| Stars | Count |",
        "|:---|---:|",
    ]
    for stars in (5, 4, 3, 2, 1):
        lines.append(f"| {stars}★ | {stats.distribution.get(stars, 0)} |")

    if stats.oldest and stats.newest:
        lines += ["", f"*Sample spans {stats.oldest.date().isoformat()} to {stats.newest.date().isoformat()}.*"]

    if identity.platform is Platform.IOS:
        lines += [
            "",
            "> Apple's public review feed currently serves only the ~50 most recent reviews "
            "for the US storefront, and is unreliable even within that (pagination and other "
            "storefronts are not reliably available at all) — see the project README for details.",
        ]
    else:
        lines += [
            "",
            f"> Showing the most recent reviews Google Play returned for the "
            f"{identity.country.upper()} storefront.",
        ]
    return "\n".join(lines)


def _render_summary(
    stats: ReviewStats,
    resolved: list[tuple[FeatureGap, list[Review]]],
    *,
    insufficient_signal: bool,
    analysis_failed: bool,
) -> str:
    """Closing recap: sample composition and what was found, in one place.

    Replaces a dump of every fetched review. The model already cites the
    specific reviews that matter under each gap (SC-1); a second copy of all
    fifty reviews below that added length without adding information. This is
    computed from `stats` and the already-resolved gap titles, the same way
    the snapshot table is -- nothing here for a model to get wrong.
    """
    stars_desc = ", ".join(
        f"{s}★ ×{stats.distribution.get(s, 0)}" for s in (5, 4, 3, 2, 1) if stats.distribution.get(s, 0)
    )
    span = (
        f" from {stats.oldest.date().isoformat()} to {stats.newest.date().isoformat()}"
        if stats.oldest and stats.newest
        else ""
    )

    lines = [
        "## Summary",
        "",
        f"{stats.fetched_count} reviews sampled{span} ({stars_desc or 'none in this sample'}), "
        f"{stats.critical_count} critical (≤3★, {stats.critical_share:.0%} of the sample).",
    ]

    if insufficient_signal:
        lines.append("Not enough critical reviews in the sample for a gap analysis to run.")
    elif analysis_failed:
        lines.append("The gap-analysis model call failed to return a usable result.")
    elif resolved:
        titles = "; ".join(sanitize_inline(gap.title) for gap, _ in resolved)
        lines.append(f"{len(resolved)} gap(s) found: {titles}.")
    else:
        lines.append("No gap could be backed by a real review citation, so none are shown.")

    return "\n".join(lines)


def render_report(
    identity: AppIdentity,
    stats: ReviewStats,
    reviews: tuple[Review, ...],
    gaps: tuple[FeatureGap, ...],
    *,
    generated_on: date,
    insufficient_signal: bool = False,
    analysis_failed: bool = False,
) -> tuple[str, tuple[FeatureGap, ...]]:
    """Assemble the finished Markdown document.

    Args:
        identity: Which app was analyzed.
        stats: Rating arithmetic over the fetched sample.
        reviews: Every fetched review, for resolving gap citations against (SC-1)
            and for the closing summary's counts -- not rendered individually.
        gaps: The model's gaps (or an empty tuple), before resolution.
        generated_on: The date to stamp on the report.
        insufficient_signal: Too few critical reviews for analysis to run.
        analysis_failed: An analysis was attempted but produced nothing usable.

    Returns:
        The complete document, and the subset of gaps that survived citation
        resolution (SC-2) -- returned alongside the markdown so the caller can
        store the same grounded gaps on :class:`~app.models.schemas.Report`
        without re-deriving them.
    """
    by_id = {review.id: review for review in reviews}
    resolved = [r for gap in gaps if (r := _resolve_gap(gap, by_id)) is not None]

    header = _render_snapshot(identity, stats, generated_on)

    if insufficient_signal:
        gap_section = (
            "## Feature gaps\n\n"
            f"Not enough critical reviews in this sample ({stats.critical_count} found) to run a "
            "gap analysis. The rating snapshot above is still real and current -- there just isn't "
            "enough negative signal yet for a reliable pattern."
        )
    elif analysis_failed:
        gap_section = (
            "## Feature gaps\n\n"
            "**Analysis failed.** The model did not return a usable gap analysis for this batch of "
            "reviews. The rating snapshot above is real; the gaps that would have gone here are not."
        )
    elif not resolved:
        gap_section = (
            "## Feature gaps\n\n"
            "The model did not identify any gap it could support with a real review citation, so "
            "none are shown."
        )
    else:
        gap_section = "## Feature gaps\n\n" + "\n\n".join(
            _render_gap(gap, evidence) for gap, evidence in resolved
        )

    summary_section = _render_summary(
        stats, resolved, insufficient_signal=insufficient_signal, analysis_failed=analysis_failed
    )

    document = "\n\n".join([header, gap_section, summary_section])
    return sanitize_markdown(document), tuple(gap for gap, _ in resolved)


def download_filename(identity: AppIdentity, generated_on: date) -> str:
    """Build the download filename for a report."""
    slug = re.sub(r"[^a-z0-9]+", "-", identity.track_name.lower()).strip("-") or "app"
    return f"{slug}-review-gaps-{generated_on.isoformat()}.md"
