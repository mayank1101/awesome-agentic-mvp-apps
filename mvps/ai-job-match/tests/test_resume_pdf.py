"""Tests for PDF rendering of the tailored resume.

The important test here is the round trip: render the resume, then read it back
with the same extractor the app uses on uploads. That is the closest offline
proxy for "an applicant-tracking system can parse this", and it is the property
the whole single-column, no-tables layout exists to protect.
"""

from app.services import pdf_extract
from app.services.resume_pdf import render_pdf

_RESUME = """# Priya Raman
priya.raman@example.com | Bengaluru

## Summary
Backend engineer with **6 years** building payment services.

## Skills
Python, Django, PostgreSQL, Docker

## Experience
### Senior Backend Engineer, Fintrail
*Mar 2021 - Present, Bengaluru*
- Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%
- Designed REST APIs serving 12000 requests per minute

## Education
B.Tech, Computer Science, VIT Vellore, 2018
"""


def test_output_is_a_pdf() -> None:
    assert render_pdf(_RESUME).startswith(b"%PDF")


def test_rendered_text_survives_re_extraction() -> None:
    """What goes into the PDF has to come back out, or no filter will read it."""
    extracted = pdf_extract.extract_resume(render_pdf(_RESUME))

    for fragment in (
        "Priya Raman",
        "settlement pipeline",
        "REST APIs",
        "VIT Vellore",
        "Fintrail",
    ):
        assert fragment in extracted.text


def test_markdown_syntax_does_not_reach_the_page() -> None:
    extracted = pdf_extract.extract_resume(render_pdf(_RESUME))

    assert "##" not in extracted.text
    assert "**" not in extracted.text


def test_unicode_is_transliterated_rather_than_crashing() -> None:
    """Core PDF fonts are Latin-1: an em dash must not take the whole render down."""
    rendered = render_pdf(
        "# José Álvarez — Engineer\n- Delivered “results”…\n" + _RESUME.split("## Summary", 1)[1]
    )
    text = pdf_extract.extract_resume(rendered).text

    assert "Jose" in text or "José" in text
    assert "Engineer" in text


def test_links_keep_their_target() -> None:
    # Padded with the fixture body: the extractor refuses documents with almost
    # no text, which is the scanned-PDF guard doing its job.
    text = pdf_extract.extract_resume(
        render_pdf("# Name\n[portfolio](https://example.com/p)\n" + _RESUME)
    ).text

    assert "example.com/p" in text


def test_long_resumes_paginate() -> None:
    long_resume = "# Name\n" + "\n".join(f"- Bullet number {i} about work done" for i in range(120))

    assert pdf_extract.extract_resume(render_pdf(long_resume)).page_count > 1


def test_empty_input_still_produces_a_file() -> None:
    assert render_pdf("").startswith(b"%PDF")
