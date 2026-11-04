"""PDF text and metadata extraction with PyMuPDF.

Text is extracted page by page and kept separate per page, because every
citation the system produces later points to a page number. This module does
not clean or chunk text; that happens in the chunking stage.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Literal

import pymupdf

from app.exceptions import InvalidInputError
from app.utils.logging import get_logger

logger = get_logger(__name__)

# MuPDF prints recoverable parser warnings to stderr; malformed files are
# handled through exceptions below instead.
pymupdf.TOOLS.mupdf_display_errors(False)

# A page with fewer visible characters than this is treated as having no
# extractable text (typically a scanned image with a stray page number).
MIN_CHARS_FOR_TEXT_PAGE = 25
MAX_TITLE_LENGTH = 300
# A first-page title must be at least this much larger than the body font.
TITLE_TO_BODY_SIZE_RATIO = 1.15

TitleSource = Literal["metadata", "first_page", "filename"]


class PdfRejectedError(InvalidInputError):
    """The file cannot be ingested; the message is shown to the user as-is."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int  # 1-based, as printed in citations
    text: str
    char_count: int

    @property
    def has_text(self) -> bool:
        return self.char_count >= MIN_CHARS_FOR_TEXT_PAGE


@dataclass(frozen=True)
class ExtractedPdf:
    page_count: int
    pages: list[ExtractedPage]
    title: str
    title_source: TitleSource
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pages_without_text(self) -> list[int]:
        return [page.page_number for page in self.pages if not page.has_text]


def extract_pdf(path: Path, *, original_filename: str, max_pages: int) -> ExtractedPdf:
    """Extract page text, metadata and a title from a PDF on disk.

    Raises ``PdfRejectedError`` with a specific ``code`` for files that cannot
    be ingested: corrupted, password-protected, too long, or without text.
    """
    try:
        document = pymupdf.open(path, filetype="pdf")
    except Exception as exc:  # MuPDF raises several unrelated types for bad input
        raise PdfRejectedError(
            "This file could not be read as a PDF. It may be corrupted or incomplete.",
            code="invalid_pdf",
        ) from exc

    with document:
        if document.needs_pass:
            raise PdfRejectedError(
                "This PDF is password-protected. Remove the password and upload it again.",
                code="encrypted_pdf",
            )
        if document.page_count == 0:
            raise PdfRejectedError("This PDF has no pages.", code="invalid_pdf")
        if document.page_count > max_pages:
            raise PdfRejectedError(
                f"This PDF has {document.page_count} pages; the limit is {max_pages}.",
                code="too_many_pages",
                details={"page_count": document.page_count, "max_pages": max_pages},
            )

        try:
            pages = [_extract_page(page) for page in document]
            metadata = _clean_metadata(document.metadata or {})
            title, title_source = _detect_title(document, metadata, original_filename)
        except Exception as exc:
            logger.warning("Failed to parse PDF %s", path.name, exc_info=True)
            raise PdfRejectedError(
                "This PDF is malformed and its contents could not be extracted.",
                code="invalid_pdf",
            ) from exc

    if not any(page.has_text for page in pages):
        raise PdfRejectedError(
            "No extractable text was found. This looks like a scanned or image-only PDF. "
            "OCR is not supported yet, so please upload a PDF with selectable text.",
            code="no_extractable_text",
            details={"page_count": len(pages)},
        )

    return ExtractedPdf(
        page_count=len(pages),
        pages=pages,
        title=title,
        title_source=title_source,
        metadata=metadata,
    )


def _extract_page(page: pymupdf.Page) -> ExtractedPage:
    # The default "text" mode follows the content stream, which keeps
    # two-column papers in reading order. sort=True would interleave columns.
    text = page.get_text("text")
    visible_chars = len(re.sub(r"\s+", "", text))
    return ExtractedPage(page_number=page.number + 1, text=text, char_count=visible_chars)


# --- Metadata --------------------------------------------------------------

_METADATA_FIELDS = {
    "author": "author",
    "subject": "subject",
    "keywords": "keywords",
    "creator": "creator",
    "producer": "producer",
    "format": "pdf_version",
    "title": "pdf_title",
}
_PDF_DATE = re.compile(r"^D:(\d{4})(\d{2})?(\d{2})?")


def _parse_pdf_date(value: str) -> str | None:
    """Convert a PDF date like ``D:20170228120000Z`` to ``2017-02-28``."""
    match = _PDF_DATE.match(value.strip())
    if not match:
        return None
    year, month, day = match.group(1), match.group(2) or "01", match.group(3) or "01"
    return f"{year}-{month}-{day}"


def _clean_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for source_key, target_key in _METADATA_FIELDS.items():
        value = raw.get(source_key)
        if isinstance(value, str) and value.strip():
            cleaned[target_key] = " ".join(value.split())
    for source_key, target_key in (("creationDate", "created"), ("modDate", "modified")):
        value = raw.get(source_key)
        if isinstance(value, str) and (parsed := _parse_pdf_date(value)):
            cleaned[target_key] = parsed
    return cleaned


# --- Title detection -------------------------------------------------------

# Metadata titles that are placeholders or file names rather than real titles.
_JUNK_TITLE = re.compile(
    r"^(untitled|microsoft word|title|slide \d+|document\d*|doi:|pii[: ])"
    r"|\.(pdf|docx?|tex|dvi|ps|indd)$",
    re.IGNORECASE,
)


def _is_plausible_title(text: str) -> bool:
    letters = sum(character.isalpha() for character in text)
    return 8 <= len(text) <= MAX_TITLE_LENGTH and letters >= 5 and not _JUNK_TITLE.search(text)


def _normalise_title(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    return " ".join(text.split())[:MAX_TITLE_LENGTH]


def _detect_title(
    document: pymupdf.Document, metadata: dict[str, Any], original_filename: str
) -> tuple[str, TitleSource]:
    """Pick the best title: PDF metadata, then the largest text on page 1, then the filename."""
    metadata_title = _normalise_title(metadata.get("pdf_title", ""))
    if _is_plausible_title(metadata_title):
        return metadata_title, "metadata"

    first_page_title = _title_from_first_page(document[0])
    if first_page_title and _is_plausible_title(first_page_title):
        return first_page_title, "first_page"

    stem = PurePath(original_filename).stem
    fallback = " ".join(re.sub(r"[_\-]+", " ", stem).split()) or "Untitled document"
    return fallback, "filename"


def _title_from_first_page(page: pymupdf.Page) -> str | None:
    """Return the largest-font horizontal text in the top half of the page.

    The text must be clearly larger than the page's body font, so a page set
    in one size yields no title rather than a sentence of body text. Rotated
    lines are skipped, which removes margin stamps such as the vertical arXiv
    identifier printed on preprints.
    """
    top_half = page.rect.height * 0.5
    lines: list[tuple[float, float, str]] = []  # (font size, y position, text)
    chars_by_size: dict[float, int] = {}

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 = text block; images are skipped
            continue
        for line in block["lines"]:
            direction_x, direction_y = line["dir"]
            if abs(direction_x - 1.0) > 0.01 or abs(direction_y) > 0.01:
                continue
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            for span in spans:
                size = round(span["size"], 1)
                chars_by_size[size] = chars_by_size.get(size, 0) + len(span["text"])
            if line["bbox"][1] > top_half:
                continue
            text = "".join(span["text"] for span in spans).strip()
            lines.append((max(span["size"] for span in spans), line["bbox"][1], text))

    if not lines:
        return None

    body_size = max(chars_by_size, key=chars_by_size.__getitem__)
    largest = max(size for size, _, _ in lines)
    if largest < body_size * TITLE_TO_BODY_SIZE_RATIO:
        return None
    candidates = sorted(
        ((y, size, text) for size, y, text in lines if size >= largest - 0.5),
        key=lambda item: item[0],
    )
    # Titles often wrap over two or three lines; take consecutive lines and
    # stop at the first large vertical gap.
    title_lines = [candidates[0][2]]
    previous_y = candidates[0][0]
    for y, size, text in candidates[1:]:
        if y - previous_y > size * 2.5:
            break
        title_lines.append(text)
        previous_y = y
    return _normalise_title(" ".join(title_lines))
