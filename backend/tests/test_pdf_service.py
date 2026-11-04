from pathlib import Path

import pymupdf
import pytest

from app.services.pdf_service import PdfRejectedError, extract_pdf
from tests.pdf_factory import make_pdf, paper_text, sample_paper


def write(tmp_path: Path, data: bytes, name: str = "paper.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def extract(path: Path, filename: str = "paper.pdf", max_pages: int = 500):
    return extract_pdf(path, original_filename=filename, max_pages=max_pages)


def test_extracts_text_page_by_page(tmp_path: Path) -> None:
    result = extract(write(tmp_path, sample_paper(pages=3)))

    assert result.page_count == 3
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    for page in result.pages:
        # Each page's text stays on that page, which is what citations rely on.
        assert f"This page {page.page_number} discusses" in page.text
        assert page.has_text
    assert result.pages_without_text == []


def test_metadata_title_is_preferred_when_plausible(tmp_path: Path) -> None:
    data = make_pdf(
        [paper_text("x", 1)],
        title="Large Heading On The Page",
        metadata={
            "title": "Federated Optimization in Heterogeneous Networks",
            "author": "Tian Li, Anit Kumar Sahu",
            "creationDate": "D:20200421093000Z",
        },
    )
    result = extract(write(tmp_path, data))

    assert result.title == "Federated Optimization in Heterogeneous Networks"
    assert result.title_source == "metadata"
    assert result.metadata["author"] == "Tian Li, Anit Kumar Sahu"
    assert result.metadata["created"] == "2020-04-21"


@pytest.mark.parametrize("junk", ["Microsoft Word - draft_v3.docx", "main.tex", "untitled"])
def test_placeholder_metadata_title_falls_back_to_first_page(tmp_path: Path, junk: str) -> None:
    data = make_pdf(
        [paper_text("x", 1)],
        title="Communication-Efficient Learning of Deep Networks",
        metadata={"title": junk},
    )
    result = extract(write(tmp_path, data))
    assert result.title == "Communication-Efficient Learning of Deep Networks"
    assert result.title_source == "first_page"


def test_title_wrapped_over_several_lines_is_joined(tmp_path: Path) -> None:
    long_title = (
        "Communication-Efficient Learning of Deep Networks from Decentralized Data "
        "with Heterogeneous Clients"
    )
    result = extract(write(tmp_path, make_pdf([paper_text("x", 1)], title=long_title)))
    assert result.title == long_title


def test_rotated_margin_stamp_is_not_taken_as_title(tmp_path: Path) -> None:
    document = pymupdf.open(stream=sample_paper("Scaffold for Federated Learning"), filetype="pdf")
    # Mimic the vertical arXiv identifier printed on preprints, in a huge font.
    document[0].insert_text(
        (30, 600), "arXiv:1910.06378v4 [cs.LG] 9 Apr 2021", fontsize=30, rotate=90
    )
    result = extract(write(tmp_path, document.tobytes()))
    assert result.title == "Scaffold for Federated Learning"


def test_uniform_font_page_falls_back_to_filename(tmp_path: Path) -> None:
    # No text is larger than the body font, so no sentence is mistaken for a title.
    result = extract(write(tmp_path, make_pdf([paper_text("x", 1)])), filename="fed_prox-paper.pdf")
    assert result.title == "fed prox paper"
    assert result.title_source == "filename"


def test_image_first_page_falls_back_to_filename(tmp_path: Path) -> None:
    data = make_pdf([None, paper_text("x", 2)])
    result = extract(write(tmp_path, data), filename="scaffold_2020.pdf")
    assert result.title == "scaffold 2020"
    assert result.title_source == "filename"


def test_image_only_pdf_is_rejected_with_ocr_message(tmp_path: Path) -> None:
    with pytest.raises(PdfRejectedError) as error:
        extract(write(tmp_path, make_pdf([None, None])))
    assert error.value.code == "no_extractable_text"
    assert "OCR is not supported" in error.value.message


def test_partially_scanned_pdf_reports_pages_without_text(tmp_path: Path) -> None:
    result = extract(write(tmp_path, make_pdf([paper_text("x", 1), None, paper_text("x", 3)])))
    assert result.page_count == 3
    assert result.pages_without_text == [2]


def test_password_protected_pdf_is_rejected(tmp_path: Path) -> None:
    data = make_pdf([paper_text("x", 1)], user_password="secret")
    with pytest.raises(PdfRejectedError) as error:
        extract(write(tmp_path, data))
    assert error.value.code == "encrypted_pdf"


def test_owner_password_only_pdf_is_readable(tmp_path: Path) -> None:
    # Permissions-only encryption opens without a password, so it is accepted.
    data = make_pdf([paper_text("x", 1)], owner_password="owner-only")
    result = extract(write(tmp_path, data))
    assert result.pages[0].has_text


def test_corrupted_pdf_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PdfRejectedError) as error:
        extract(write(tmp_path, b"%PDF-1.7\nthis is not really a pdf"))
    assert error.value.code == "invalid_pdf"


def test_truncated_pdf_is_rejected_or_recovered(tmp_path: Path) -> None:
    data = sample_paper(pages=2)
    path = write(tmp_path, data[: len(data) // 3])
    # MuPDF can sometimes repair truncated files; either outcome is acceptable
    # as long as it never crashes with an unexpected exception.
    try:
        extract(path)
    except PdfRejectedError as error:
        assert error.code in {"invalid_pdf", "no_extractable_text"}


def test_page_limit_is_enforced(tmp_path: Path) -> None:
    with pytest.raises(PdfRejectedError) as error:
        extract(write(tmp_path, sample_paper(pages=3)), max_pages=2)
    assert error.value.code == "too_many_pages"
