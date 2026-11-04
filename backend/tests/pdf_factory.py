"""Builds small, deterministic PDFs for tests, so no binary fixtures are checked in."""

import pymupdf

PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4 in points
BODY_FONT_SIZE = 10
TITLE_FONT_SIZE = 20


def _add_text_page(document: pymupdf.Document, body: str, title: str | None = None) -> None:
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    top = 72
    if title:
        # A text box wraps long titles over several lines, as real papers do.
        # insert_textbox writes nothing if the text does not fit, hence the check.
        title_box = pymupdf.Rect(72, 56, PAGE_WIDTH - 72, 56 + 5 * TITLE_FONT_SIZE * 1.3)
        if page.insert_textbox(title_box, title, fontsize=TITLE_FONT_SIZE) < 0:
            raise ValueError("Title too long for the test page")
        top = title_box.y1 + 20
    page.insert_textbox(
        pymupdf.Rect(72, top, PAGE_WIDTH - 72, PAGE_HEIGHT - 72), body, fontsize=BODY_FONT_SIZE
    )


def _add_image_page(document: pymupdf.Document) -> None:
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
    pixmap.clear_with(180)
    page.insert_image(pymupdf.Rect(72, 72, 520, 720), pixmap=pixmap)


def make_pdf(
    pages: list[str | None],
    *,
    title: str | None = None,
    metadata: dict[str, str] | None = None,
    user_password: str | None = None,
    owner_password: str | None = None,
) -> bytes:
    """Create a PDF. Each entry in ``pages`` is body text, or None for an image-only page.

    ``title`` is drawn in a large font at the top of the first page.
    """
    document = pymupdf.open()
    for index, body in enumerate(pages):
        if body is None:
            _add_image_page(document)
        else:
            _add_text_page(document, body, title if index == 0 else None)
    if metadata:
        document.set_metadata(metadata)

    if user_password or owner_password:
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=user_password or "",
            owner_pw=owner_password or user_password or "",
        )
    else:
        data = document.tobytes()
    document.close()
    return data


def paper_text(topic: str, page: int) -> str:
    return (
        f"This page {page} discusses {topic}. Federated learning trains a shared model "
        "across many clients without collecting their raw data. "
        "We evaluate the method on non-IID partitions of the dataset."
    )


def sample_paper(title: str = "Communication-Efficient Learning of Deep Networks", pages: int = 3):
    return make_pdf([paper_text(title, number) for number in range(1, pages + 1)], title=title)
