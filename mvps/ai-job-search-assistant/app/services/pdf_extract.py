"""Turning an uploaded PDF into resume text.

Deliberately small and deliberately strict. Every failure here is a *stated*
failure with its own exception and its own sentence on screen, because the three
ways a resume PDF fails to parse -- encrypted, scanned, corrupt -- look identical
to the user ("nothing happened") and need three different actions.

There is no OCR. Adding it means either a system binary (`tesseract`) or a vision
model, and the deployment target is a ~1GB Streamlit Community Cloud container.
A scanned resume is therefore a stated limit of this app, surfaced immediately
rather than silently producing an empty analysis.

Text extraction order is whatever `pypdf` reports, which for a two-column resume
template can interleave the columns. That is a real fidelity limit; it is
mitigated by the fact that everything downstream works on *lines*, and by showing
the extracted text to the user before the analysis runs, so a mangled parse is
visible rather than mysterious.
"""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import get_settings
from app.core.exceptions import (
    EncryptedPdfError,
    ResumeExtractionError,
    ResumeTooLargeError,
    ScannedPdfError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Ligatures and typographic characters that PDF text extraction preserves and
#: that then break naive substring matching between the original resume and a
#: rewrite ("workflow" vs "workﬂow"). Normalised once, at the edge.
_TRANSLATIONS = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "•": "*",
        " ": " ",
        "…": "...",
    }
)

#: Runs of spaces or tabs. Newlines are kept: line structure is the only layout
#: signal that survives extraction, and the resume parser depends on it.
_HORIZONTAL_SPACE = re.compile(r"[ \t]+")

#: Icon glyph names that LaTeX resume templates (moderncv, Awesome-CV and the
#: rest) leave in the text layer, glued to the value they decorate:
#: ``/envelopename@example.com``, ``/phone(+91) 12345``, ``/globePortfolio``.
#:
#: Found on a real resume, not imagined: the glued prefix corrupts the email
#: address in the extracted text, which then makes the fabrication guard reject
#: the candidate's *own* address as invented. Stripped at the edge so every layer
#: downstream sees the address the candidate actually wrote.
_ICON_GLYPH = re.compile(
    r"(?<![A-Za-z0-9])/(?:phone|telephone|mobile|envelope|email|mail|at|linkedin|github|"
    r"gitlab|globe|website|home|link|twitter|x|map|mapmarker|marker|location|pin|"
    r"calendar|user|graduationcap|mortarboard|briefcase|building|code|star|award)"
    r"(?=[A-Za-z0-9(+])",
    re.IGNORECASE,
)

#: Three or more blank lines, collapsed to one blank line.
_BLANK_RUN = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ExtractedResume:
    """Text recovered from an uploaded PDF.

    Attributes:
        text: Normalised text, capped at the configured length.
        page_count: How many pages the document had.
        truncated: Whether the cap removed anything. Surfaced on screen -- an
            analysis of two thirds of a resume is a different thing from an
            analysis of a resume, and the user should know which they got.
    """

    text: str
    page_count: int
    truncated: bool


def normalize_text(raw: str) -> str:
    """Flatten extraction artefacts without discarding line structure.

    Args:
        raw: Text exactly as `pypdf` produced it.

    Returns:
        Text with ligatures and smart punctuation replaced, horizontal
        whitespace collapsed, and blank-line runs reduced.
    """
    text = raw.translate(_TRANSLATIONS)
    text = _ICON_GLYPH.sub(" ", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_RUN.sub("\n\n", text).strip()


def extract_resume(data: bytes) -> ExtractedResume:
    """Extract resume text from PDF bytes.

    Args:
        data: The uploaded file's bytes.

    Returns:
        The extracted text with its page count and truncation flag.

    Raises:
        ResumeTooLargeError: The upload exceeds the byte or page cap.
        EncryptedPdfError: The document is password-protected.
        ScannedPdfError: The document parsed but yielded too little text to be a
            resume, which in practice means it is a scan.
        ResumeExtractionError: The bytes are not a readable PDF.
    """
    settings = get_settings()

    if len(data) > settings.max_upload_bytes:
        raise ResumeTooLargeError(
            f"That file is {len(data) / 1_048_576:.1f} MB. "
            f"The limit is {settings.max_upload_bytes / 1_048_576:.0f} MB."
        )

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ResumeExtractionError(
            "That file could not be read as a PDF. If it was exported from a word "
            "processor, try exporting it again."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - pypdf raises assorted parse errors
        raise ResumeExtractionError("That file could not be read as a PDF.") from exc

    if reader.is_encrypted:
        # An empty owner password is common on "protected" resumes and costs one
        # call to rule out; anything else is the user's to fix.
        try:
            opened = reader.decrypt("")
        except Exception:  # noqa: BLE001 - unsupported ciphers raise, not return
            opened = 0
        if not opened:
            raise EncryptedPdfError(
                "That PDF is password-protected. Save an unprotected copy and upload that."
            )

    page_count = len(reader.pages)
    if page_count > settings.max_resume_pages:
        raise ResumeTooLargeError(
            f"That PDF has {page_count} pages. The limit is {settings.max_resume_pages}. "
            "This app expects a resume, not a portfolio."
        )

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - one bad page should not lose the rest
            logger.warning("Page %d of %d failed to extract: %s", index + 1, page_count, exc)
            pages.append("")

    text = normalize_text("\n".join(pages))

    if len(text) < settings.min_resume_chars:
        raise ScannedPdfError(
            "Almost no text came out of that PDF, which usually means it is a scan or "
            "an image export. This app has no OCR. Upload a PDF exported directly from "
            "a word processor."
        )

    truncated = len(text) > settings.max_resume_chars
    if truncated:
        text = _truncate_on_boundary(text, settings.max_resume_chars)
        logger.info("Resume text truncated to %d characters", settings.max_resume_chars)

    return ExtractedResume(text=text, page_count=page_count, truncated=truncated)


def prepare_posting_text(raw: str) -> tuple[str, bool]:
    """Normalise and cap the text of a fetched job posting.

    The cap matters more here than for a pasted document, because nobody chose
    this text: a careers page arrives with navigation, benefits copy, an EEO
    statement, and sometimes the company's founding story. The requirements are
    almost always near the top, so cutting the tail costs little and keeps the
    per-job model call inside a free tier's per-minute token ceiling.

    Args:
        raw: Page text as the extraction endpoint returned it.

    Returns:
        The normalised text, and whether it was truncated.
    """
    text = normalize_text(raw)
    cap = get_settings().max_posting_chars
    if len(text) <= cap:
        return text, False
    return _truncate_on_boundary(text, cap), True


def _truncate_on_boundary(text: str, cap: int) -> str:
    """Cut `text` to `cap` characters at the last line break before the cap.

    Cutting mid-line would hand the model half a bullet, which it will then
    complete from imagination -- exactly the failure this app is built to avoid.
    """
    clipped = text[:cap]
    boundary = clipped.rfind("\n")
    return clipped[:boundary].rstrip() if boundary > cap // 2 else clipped.rstrip()
