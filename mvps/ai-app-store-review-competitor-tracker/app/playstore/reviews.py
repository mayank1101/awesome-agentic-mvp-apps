"""The Play Store review fetch, mirroring `app/appstore/reviews.py`.

Confirmed by direct testing (docs/01-prd.md §7-addendum) to be materially more
reliable than the iOS review feed: both the US and India storefronts returned
real, current review data on every attempt, with no sign of the rate-limiting
that made repeated iOS testing unreliable. Country selection is a real,
working parameter here -- unlike `app/appstore/reviews.py`, this module does
not need to pin itself to one storefront.

`google-play-scraper` is unofficial (it scrapes Play's own review-list
endpoint, the one the Play Store web page itself calls) and has no published
SLA, so this module still treats an empty result as "no data available," not
as proof the app has zero reviews -- the same posture as the iOS client, just
without needing the retry that client's unreliability requires.
"""

from google_play_scraper import Sort
from google_play_scraper import reviews as gp_reviews
from google_play_scraper.exceptions import GooglePlayScraperException

from app.core.config import Settings, get_settings
from app.core.exceptions import ReviewFetchError
from app.core.logging import get_logger
from app.models.schemas import Review

logger = get_logger(__name__)


def _to_review(raw: dict, *, char_cap: int) -> Review | None:
    """Convert one Play review dict into a :class:`Review`, or `None` if unusable."""
    review_id = raw.get("reviewId")
    score = raw.get("score")
    if not review_id or score is None:
        return None

    content = (raw.get("content") or "")[:char_cap].strip()
    return Review(
        id=str(review_id),
        rating=int(score),
        title=None,  # Play reviews have no title field, unlike iOS.
        content=content,
        author=raw.get("userName") or "Anonymous",
        version=raw.get("reviewCreatedVersion") or raw.get("appVersion"),
        updated=raw.get("at"),
    )


def fetch_reviews(
    package: str,
    *,
    country: str | None = None,
    settings: Settings | None = None,
) -> list[Review]:
    """Fetch the most recent reviews for one app.

    Args:
        package: The Play Store package name.
        country: Storefront to fetch from, e.g. "us", "in". Defaults to
            `settings.playstore_country`.
        settings: Runtime configuration; defaults to the process settings.

    Returns:
        Reviews, most recent first, capped at `settings.max_reviews`.
        Possibly empty -- an app with no reviews in the sample is a
        legitimate outcome, not an error.

    Raises:
        ReviewFetchError: The request failed (network, parse error).
    """
    settings = settings or get_settings()
    resolved_country = country or settings.playstore_country

    try:
        raw_reviews, _continuation_token = gp_reviews(
            package,
            lang="en",
            country=resolved_country,
            sort=Sort.NEWEST,
            count=settings.max_reviews,
        )
    except GooglePlayScraperException as exc:
        raise ReviewFetchError(f"the Play Store review request failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - the scraper raises bare urllib/network errors
        raise ReviewFetchError(f"the Play Store review request failed: {exc}") from exc

    reviews = [_to_review(raw, char_cap=settings.review_char_cap) for raw in raw_reviews]
    kept = [r for r in reviews if r is not None][: settings.max_reviews]

    logger.info("fetched %d review(s) for package %s (%s)", len(kept), package, resolved_country)
    return kept
