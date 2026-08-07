"""Rendering the tailored resume to a PDF a candidate can actually submit.

`fpdf2`, with the built-in Helvetica face. No system fonts, no HTML engine, no
headless browser -- the deployment target is a ~1GB container, and WeasyPrint or
wkhtmltopdf would each drag in native libraries that do not belong in it.

The layout is single-column, black on white, with real text and no tables, no
columns, no text boxes, and no images. That is not minimalism for its own sake:
it is what an applicant-tracking system can parse. A two-column template with the
skills in a sidebar is the single most common reason a resume arrives at a
recruiter as scrambled text, and this app exists to get someone through that
filter rather than to look good on screen.

The Markdown subset understood here is exactly the subset the rewrite prompt is
told to emit -- headings, an italic meta line, bullets, bold spans, paragraphs.
Anything else degrades to a plain paragraph rather than appearing as syntax.

The core PDF fonts are Latin-1 only, so text is transliterated before it is
written: a smart quote or an em dash would otherwise raise mid-render, on a
document that had already rendered fine in tests with ASCII fixtures.
"""

import re
import unicodedata

from fpdf import FPDF

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Page geometry, in millimetres.
_MARGIN = 15.0
_LINE_HEIGHT = 4.8

#: Characters the core fonts cannot encode, mapped to what a resume means by
#: them. Anything else non-Latin-1 is decomposed and then dropped.
_TRANSLITERATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "•": "-",
        "…": "...",
        " ": " ",
        "‑": "-",
        "→": "->",
        "₹": "INR ",
        "€": "EUR ",
        "×": "x",
    }
)

_BOLD_SPAN = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_LINE = re.compile(r"^\s*[*_](?P<text>[^*_].*?)[*_]\s*$")
_BULLET = re.compile(r"^\s*[-*+]\s+(?P<text>.+)$")
_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+)$")
_RULE = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _encodable(text: str) -> str:
    """Reduce text to something the core PDF fonts can write.

    Args:
        text: Arbitrary text from a rewrite.

    Returns:
        Latin-1-safe text. Accented characters keep their base letter rather than
        vanishing, because a name is the last thing that should silently lose
        characters.
    """
    translated = text.translate(_TRANSLITERATE)
    try:
        translated.encode("latin-1")
        return translated
    except UnicodeEncodeError:
        decomposed = unicodedata.normalize("NFKD", translated)
        return decomposed.encode("latin-1", "ignore").decode("latin-1")


class _ResumePdf(FPDF):
    """A single-column resume document."""

    def __init__(self) -> None:
        """Set up an A4 page with even margins and automatic page breaks."""
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=_MARGIN)
        self.set_margins(_MARGIN, _MARGIN, _MARGIN)
        self.set_title("Resume")
        self.add_page()

    def write_rich(self, text: str, *, size: float, style: str = "", leading: float = 1.0) -> None:
        """Write a line, honouring inline `**bold**` spans.

        Args:
            text: One line of Markdown, already stripped of block syntax.
            size: Font size in points.
            style: Base style: "" or "I".
            leading: Extra vertical space after the line, as a multiple of the
                line height.
        """
        parts = _BOLD_SPAN.split(_LINK.sub(r"\1 (\2)", text))
        # `re.split` on one capturing group alternates literal, captured, literal...
        for index, part in enumerate(parts):
            if not part:
                continue
            self.set_font("Helvetica", style + ("B" if index % 2 else ""), size)
            self.write(_LINE_HEIGHT, _encodable(part))
        self.ln(_LINE_HEIGHT * leading)


def render_pdf(markdown: str, *, name_hint: str = "Resume") -> bytes:
    """Render a tailored resume to PDF bytes.

    Args:
        markdown: The resume as Markdown, in the subset the rewrite emits.
        name_hint: Used as the PDF's title metadata when the document has no
            `# Heading` of its own.

    Returns:
        The PDF file's bytes, ready for `st.download_button`.
    """
    pdf = _ResumePdf()
    pdf.set_title(_encodable(name_hint))
    seen_title = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            pdf.ln(_LINE_HEIGHT * 0.4)
            continue

        if _RULE.match(line):
            _horizontal_rule(pdf)
            continue

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group("hashes"))
            text = heading.group("text").strip()
            if level == 1:
                pdf.set_title(_encodable(text))
                seen_title = True
                pdf.write_rich(text, size=19, leading=1.2)
            elif level == 2:
                pdf.ln(_LINE_HEIGHT * 0.5)
                pdf.write_rich(text.upper(), size=11.5, leading=0.5)
                _horizontal_rule(pdf)
            else:
                pdf.ln(_LINE_HEIGHT * 0.3)
                pdf.write_rich(f"**{text}**", size=10.5, leading=1.0)
            continue

        bullet = _BULLET.match(line)
        if bullet:
            _write_bullet(pdf, bullet.group("text").strip())
            continue

        italic = _ITALIC_LINE.match(line)
        if italic:
            pdf.write_rich(italic.group("text").strip(), size=9, style="I", leading=1.0)
            continue

        pdf.write_rich(line.strip(), size=10, leading=1.0)

    if not seen_title:
        logger.info("Rendered a resume with no top-level heading")

    return bytes(pdf.output())


def _write_bullet(pdf: _ResumePdf, text: str) -> None:
    """Write one hanging-indent bullet.

    A hanging indent rather than a wrapped one: when a bullet wraps, the second
    line lining up under the first is what makes a dense resume readable, and
    `write()` alone would return it to the left margin.
    """
    left = pdf.l_margin
    pdf.set_x(left + 3)
    pdf.set_font("Helvetica", "", 10)
    pdf.write(_LINE_HEIGHT, "- ")
    pdf.set_left_margin(left + 7)
    pdf.set_x(left + 7)
    pdf.write_rich(text, size=10, leading=1.0)
    pdf.set_left_margin(left)
    pdf.set_x(left)


def _horizontal_rule(pdf: _ResumePdf) -> None:
    """Draw the thin rule that separates sections."""
    pdf.set_draw_color(150, 150, 150)
    y = pdf.get_y() + 1
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(2.5)
