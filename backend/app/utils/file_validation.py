"""Checks applied to uploaded files before any parsing happens."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import BinaryIO

from app.exceptions import PayloadTooLargeError

PDF_MAGIC = b"%PDF-"
# The PDF specification tolerates up to 1 KB of leading bytes before the header.
PDF_HEADER_SEARCH_BYTES = 1024
READ_CHUNK_BYTES = 1024 * 1024
MAX_FILENAME_LENGTH = 255

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    sha256: str
    size_bytes: int
    header: bytes


def sanitize_filename(raw_name: str | None, default: str = "document.pdf") -> str:
    """Return a display-safe filename.

    The result is only ever shown to users; files are stored under generated
    names, so this is about readable output, not filesystem safety.
    """
    name = PurePath((raw_name or "").replace("\\", "/")).name
    name = unicodedata.normalize("NFC", name)
    name = _CONTROL_CHARACTERS.sub("", name).strip().lstrip(".")
    if not name:
        return default
    if len(name) > MAX_FILENAME_LENGTH:
        suffix = PurePath(name).suffix[:16]
        name = name[: MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name


def has_pdf_extension(filename: str) -> bool:
    return PurePath(filename).suffix.lower() == ".pdf"


def looks_like_pdf(header: bytes) -> bool:
    """Check the file signature, so a renamed .txt or .exe is caught early."""
    return PDF_MAGIC in header[:PDF_HEADER_SEARCH_BYTES]


def stream_to_file(source: BinaryIO, destination: Path, max_bytes: int) -> StoredUpload:
    """Copy an upload to disk while hashing it and enforcing the size limit.

    Reading in chunks keeps memory flat for large files and stops as soon as
    the limit is crossed. The partial file is removed on any failure.
    """
    digest = hashlib.sha256()
    size = 0
    header = b""
    try:
        with destination.open("wb") as output:
            while chunk := source.read(READ_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    limit_mb = max_bytes // (1024 * 1024)
                    raise PayloadTooLargeError(
                        f"The file exceeds the {limit_mb} MB upload limit.",
                        code="file_too_large",
                        details={"max_bytes": max_bytes},
                    )
                if len(header) < PDF_HEADER_SEARCH_BYTES:
                    header += chunk[: PDF_HEADER_SEARCH_BYTES - len(header)]
                digest.update(chunk)
                output.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return StoredUpload(path=destination, sha256=digest.hexdigest(), size_bytes=size, header=header)
