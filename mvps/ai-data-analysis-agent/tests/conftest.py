"""Shared fixtures.

Settings are cached per process, so a test that changes an environment
variable has to clear that cache or it reads the previous test's
configuration -- the autouse fixture below does it on both sides of every
test. No test may reach the network: keys are blanked by default, and the
`.env` a developer might have locally is disabled the same way.
"""

import pandas as pd
import pytest

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean, offline configuration."""
    for name in (
        "GROQ_API_KEY",
        "MODEL_NAME",
        "GUARDRAILS_ENABLED",
        "BLOCK_FLAGGED_INPUT",
        "MAX_ROWS",
        "MAX_UPLOAD_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)

    # pydantic-settings would otherwise read the developer's own `.env`.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, clean dataset with a numeric and a categorical column."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "C", "B", "A"],
            "revenue": [100, 250, 150, 300, 200, 50],
        }
    )


@pytest.fixture
def sample_csv_bytes() -> bytes:
    """The same dataset, as CSV bytes."""
    return b"category,revenue\nA,100\nB,250\nA,150\nC,300\nB,200\nA,50\n"
