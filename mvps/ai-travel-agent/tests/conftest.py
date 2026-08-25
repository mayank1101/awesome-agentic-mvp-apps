"""Shared fixtures.

Settings are cached per process, so a test that changes an environment
variable has to clear that cache or it reads the previous test's
configuration -- the autouse fixture below does it on both sides of every
test. No test may reach the network: keys are blanked by default, and the
`.env` a developer might have locally is disabled the same way.
"""

import pytest

from app.core.config import Settings, get_settings
from app.models.schemas import TripRequest


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean, offline configuration."""
    for name in (
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "MODEL_NAME",
        "GUARDRAILS_ENABLED",
        "BLOCK_FLAGGED_INPUT",
        "MAX_DAYS",
        "MAX_QUERIES",
    ):
        monkeypatch.delenv(name, raising=False)

    # pydantic-settings would otherwise read the developer's own `.env`.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def trip_request() -> TripRequest:
    """A validated, ordinary trip request."""
    return TripRequest(destination="Lisbon, Portugal", days=3, interests="food, history")


@pytest.fixture
def tavily_response() -> dict:
    """A Tavily `/search` response shape, as `post_json` would return it."""
    return {
        "results": [
            {
                "title": "Belem Tower",
                "url": "https://example-guide.com/lisbon/belem-tower",
                "content": "A 16th-century fortification on the Tagus river.",
                "score": 0.9,
            },
            {
                "title": "Alfama District Walking Guide",
                "url": "https://example-guide.com/lisbon/alfama",
                "content": "Lisbon's oldest neighbourhood, known for fado and narrow streets.",
                "score": 0.8,
            },
        ]
    }
