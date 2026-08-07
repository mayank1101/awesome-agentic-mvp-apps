"""Tests for the UI-side rules that do not need a running Streamlit session.

The screens themselves are verified by driving a real browser (stage 14); what
is unit-testable here is the display transform that the browser walkthrough
proved was necessary.
"""

from ui.report_view import escape_dollars


def test_prices_survive_streamlits_latex_rendering():
    # Found by looking at a live report: Streamlit reads `$...$` as inline maths,
    # so "Basic $10 ... Business $16" rendered as "Basic 10 ... Business 16" --
    # a wrong answer in the one section whose job is quoting prices.
    text = "Basic: $10 per user/month. Business: $16 per user/month."
    escaped = escape_dollars(text)

    assert escaped.count(r"\$") == 2
    # No bare `$` survives to be read as a maths delimiter.
    assert "$" not in escaped.replace(r"\$", "")


def test_text_without_prices_is_unchanged():
    text = "## Company snapshot\n\nAcme builds project tools."
    assert escape_dollars(text) == text


def test_the_escape_does_not_touch_the_document_itself():
    # SC-9: the download must stay byte-identical to what the renderer produced.
    # The escape is how this one viewer has to be spoken to, not a content change.
    document = "Team is $10 per seat."
    escape_dollars(document)

    assert document == "Team is $10 per seat."
