"""Tests for resume intake.

The fixtures are real PDFs, built with the same library the app renders with.
A mocked extractor would prove nothing here: every bug this module has had was in
what a real PDF actually yields.
"""

import pytest
from fpdf import FPDF

from app.core.config import get_settings
from app.core.exceptions import ResumeTooLargeError, ScannedPdfError
from app.services import pdf_extract


def _make_pdf(text: str, *, pages: int = 1) -> bytes:
    pdf = FPDF()
    for _ in range(pages):
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


def test_text_survives_a_round_trip(resume_text: str) -> None:
    extracted = pdf_extract.extract_resume(_make_pdf(resume_text))

    assert "Priya Raman" in extracted.text
    assert "settlement pipeline" in extracted.text
    assert extracted.page_count == 1
    assert not extracted.truncated


def test_an_image_only_pdf_is_named_as_a_scan() -> None:
    pdf = FPDF()
    pdf.add_page()

    with pytest.raises(ScannedPdfError, match="OCR"):
        pdf_extract.extract_resume(bytes(pdf.output()))


def test_oversized_upload_is_refused_before_parsing() -> None:
    oversized = b"%PDF-1.4" + b"0" * (get_settings().max_upload_bytes + 1)

    with pytest.raises(ResumeTooLargeError, match="MB"):
        pdf_extract.extract_resume(oversized)


def test_page_cap_is_enforced(resume_text: str) -> None:
    too_many = get_settings().max_resume_pages + 1

    with pytest.raises(ResumeTooLargeError, match="pages"):
        pdf_extract.extract_resume(_make_pdf(resume_text, pages=too_many))


def test_a_non_pdf_is_reported_as_unreadable() -> None:
    from app.core.exceptions import ResumeExtractionError

    with pytest.raises(ResumeExtractionError):
        pdf_extract.extract_resume(b"this is a text file, not a PDF")


def test_normalisation_flattens_ligatures_and_smart_quotes() -> None:
    normalized = pdf_extract.normalize_text("workﬂow — the team’s “best” result")

    assert "workflow" in normalized
    assert "'" in normalized and "’" not in normalized
    assert "—" not in normalized


def test_normalisation_keeps_line_structure() -> None:
    normalized = pdf_extract.normalize_text("Title\n\n\n\n- bullet one\n-  bullet two")

    assert normalized == "Title\n\n- bullet one\n- bullet two"


def test_job_description_is_capped_on_a_line_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_JD_CHARS", "80")
    get_settings.cache_clear()

    text, truncated = pdf_extract.prepare_job_description(
        "\n".join(f"line {i}" * 3 for i in range(20))
    )

    assert truncated
    assert len(text) <= 80
    assert not text.endswith("lin")


def test_short_job_description_is_untouched(job_description: str) -> None:
    text, truncated = pdf_extract.prepare_job_description(job_description)

    assert not truncated
    assert "Kubernetes" in text


def test_latex_icon_names_are_stripped_from_the_contact_line() -> None:
    """Awesome-CV and moderncv leave glyph names glued to the values they decorate."""
    normalized = pdf_extract.normalize_text(
        "/phone(+91) - 9691314634 /envelopename.surname@example.com /linkedinLinkedIn /globePortfolio"
    )

    assert "/envelope" not in normalized
    assert "name.surname@example.com" in normalized
    assert "LinkedIn" in normalized and "Portfolio" in normalized


def test_a_real_path_is_not_mistaken_for_an_icon_name() -> None:
    assert pdf_extract.normalize_text("Deployed via /home/deploy scripts") == (
        "Deployed via /home/deploy scripts"
    )
