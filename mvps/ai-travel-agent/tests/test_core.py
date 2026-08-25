from app.core.config import Settings, get_settings
from app.core.logging import redact


def test_missing_both_keys_are_reported() -> None:
    assert Settings(groq_api_key=None, tavily_api_key=None).missing_credentials() == [
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
    ]


def test_missing_one_key_is_reported() -> None:
    assert Settings(groq_api_key="gsk_x", tavily_api_key=None).missing_credentials() == [
        "TAVILY_API_KEY"
    ]


def test_present_keys_need_nothing() -> None:
    assert Settings(groq_api_key="gsk_x", tavily_api_key="tvly-x").missing_credentials() == []


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_redact_hides_groq_keys() -> None:
    text = redact("failed with key gsk_abcdefgh12345678")
    assert "gsk_abcdefgh12345678" not in text
    assert "[redacted]" in text


def test_redact_hides_tavily_keys() -> None:
    text = redact("failed with key tvly-abcdefgh12345678")
    assert "tvly-abcdefgh12345678" not in text


def test_redact_leaves_ordinary_text_alone() -> None:
    assert redact("three days in Lisbon") == "three days in Lisbon"
