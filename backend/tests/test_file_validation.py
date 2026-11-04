import hashlib
import io
from pathlib import Path

import pytest

from app.exceptions import PayloadTooLargeError
from app.utils.file_validation import (
    has_pdf_extension,
    looks_like_pdf,
    sanitize_filename,
    stream_to_file,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("paper.pdf", "paper.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\Desktop\\fedavg.pdf", "fedavg.pdf"),
        ("bad\x00name\x1f.pdf", "badname.pdf"),
        ("  .hidden.pdf ", "hidden.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_truncates_but_keeps_extension() -> None:
    result = sanitize_filename("a" * 400 + ".pdf")
    assert len(result) == 255
    assert result.endswith(".pdf")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("paper.pdf", True), ("PAPER.PDF", True), ("paper.pdf.exe", False), ("notes.txt", False)],
)
def test_has_pdf_extension(filename: str, expected: bool) -> None:
    assert has_pdf_extension(filename) is expected


def test_looks_like_pdf_checks_signature() -> None:
    assert looks_like_pdf(b"%PDF-1.7\n...")
    assert looks_like_pdf(b"\xef\xbb\xbf junk %PDF-1.4")  # tolerated leading bytes
    assert not looks_like_pdf(b"PK\x03\x04 a zip file")
    assert not looks_like_pdf(b"")


def test_stream_to_file_hashes_and_measures(tmp_path: Path) -> None:
    data = b"%PDF-1.7\n" + b"x" * 3_000_000
    stored = stream_to_file(io.BytesIO(data), tmp_path / "upload.part", max_bytes=10_000_000)
    assert stored.size_bytes == len(data)
    assert stored.sha256 == hashlib.sha256(data).hexdigest()
    assert stored.header.startswith(b"%PDF-1.7")
    assert stored.path.read_bytes() == data


def test_stream_to_file_enforces_limit_and_cleans_up(tmp_path: Path) -> None:
    destination = tmp_path / "upload.part"
    with pytest.raises(PayloadTooLargeError) as error:
        stream_to_file(io.BytesIO(b"x" * 2_000), destination, max_bytes=1_000)
    assert error.value.code == "file_too_large"
    assert not destination.exists()
