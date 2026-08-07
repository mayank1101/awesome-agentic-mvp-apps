"""Live smoke tests against the real, unofficial App Store and Play Store endpoints.

Opt-in only (`pytest -m live`): these hit the network and assert the specific
behaviour documented in docs/01-prd.md §7, so a change in either endpoint
shows up here first rather than as a silent production failure.
"""

import pytest

from app.appstore.reviews import fetch_reviews
from app.appstore.search import lookup_app, resolve, search_apps
from app.core.config import Settings
from app.models.schemas import AppIdentity
from app.playstore.reviews import fetch_reviews as fetch_play_reviews
from app.playstore.search import resolve as resolve_play

#: A high-volume app almost certain to have a healthy, current review stream.
_INSTAGRAM_ID = 389801252

#: Spotify. No app id was found to be reliably immune to the review feed's
#: rate-limiting-shaped unreliability (see `app/appstore/reviews.py`) -- this
#: one, and the `page=`/`cc=` findings, were what a *sparsely-spaced* session
#: reproduced consistently before heavier testing degraded everything,
#: Spotify included. It remains the best default id to try first.
_SPOTIFY_ID = 324684580


def _settings() -> Settings:
    return Settings(_env_file=None, groq_api_key="gsk_test")


@pytest.mark.live
def test_search_finds_a_well_known_app():
    candidates = search_apps("instagram", settings=_settings())
    assert any(c.track_id == _INSTAGRAM_ID for c in candidates)


@pytest.mark.live
def test_lookup_resolves_a_known_id():
    identity = lookup_app(_INSTAGRAM_ID, settings=_settings())
    assert identity.track_id == _INSTAGRAM_ID
    assert identity.track_name


@pytest.mark.live
def test_resolve_handles_all_three_input_shapes():
    by_id = resolve(str(_INSTAGRAM_ID), settings=_settings())
    by_url = resolve(f"https://apps.apple.com/us/app/instagram/id{_INSTAGRAM_ID}", settings=_settings())

    assert isinstance(by_id, AppIdentity) and by_id.track_id == _INSTAGRAM_ID
    assert isinstance(by_url, AppIdentity) and by_url.track_id == _INSTAGRAM_ID


@pytest.mark.live
def test_the_review_feed_returns_recent_reviews_for_a_high_volume_app():
    # Confirms the bare-URL shape documented in §7 can still return real data.
    # An empty result here is expected under the rate-limiting-shaped
    # unreliability documented in app/appstore/reviews.py and is *not* proof
    # of a regression by itself -- so this skips rather than fails when
    # empty, and only asserts on the shape of whatever it did get back.
    reviews = fetch_reviews(_SPOTIFY_ID, settings=_settings())
    if not reviews:
        pytest.skip(
            "review feed returned zero entries -- consistent with the documented "
            "external rate-limiting/reliability issue, not necessarily a regression"
        )
    assert all(1 <= r.rating <= 5 for r in reviews)


# --------------------------------------------------------------------------- #
# Google Play
# --------------------------------------------------------------------------- #

_SPOTIFY_PACKAGE = "com.spotify.music"


@pytest.mark.live
def test_play_resolve_handles_package_and_url():
    by_package = resolve_play(_SPOTIFY_PACKAGE, settings=_settings())
    by_url = resolve_play(
        f"https://play.google.com/store/apps/details?id={_SPOTIFY_PACKAGE}", settings=_settings()
    )
    assert by_package.package_name == _SPOTIFY_PACKAGE
    assert by_url.package_name == _SPOTIFY_PACKAGE


@pytest.mark.live
@pytest.mark.parametrize("country", ["us", "gb", "in", "ca", "au", "de", "fr", "br", "jp", "mx", "id", "ng"])
def test_play_reviews_work_for_every_offered_country(country: str):
    # Every code offered in ui/input_form.py's _PLAYSTORE_COUNTRIES, confirmed
    # by direct testing -- materially more reliable than the iOS feed in this
    # investigation (docs/01-prd.md §7-addendum): not skipped on empty, unlike
    # the iOS review-feed test above.
    reviews = fetch_play_reviews(_SPOTIFY_PACKAGE, country=country, settings=_settings())
    assert len(reviews) > 0
    assert all(1 <= r.rating <= 5 for r in reviews)
    assert all(r.title is None for r in reviews)
