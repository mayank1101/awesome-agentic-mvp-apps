from app.core.config import Settings, get_settings
from app.core.logging import redact


def test_missing_key_is_reported() -> None:
    assert Settings(groq_api_key=None).missing_credentials() == ["GROQ_API_KEY"]


def test_present_key_needs_nothing() -> None:
    assert Settings(groq_api_key="gsk_x").missing_credentials() == []


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_redact_hides_groq_keys() -> None:
    text = redact("failed with key gsk_abcdefgh12345678")
    assert "gsk_abcdefgh12345678" not in text
    assert "[redacted]" in text


def test_redact_hides_generic_api_key_assignment() -> None:
    text = redact("api_key=supersecretvalue in request")
    assert "supersecretvalue" not in text


def test_redact_leaves_ordinary_text_alone() -> None:
    assert redact("the average revenue was 210.5") == "the average revenue was 210.5"
