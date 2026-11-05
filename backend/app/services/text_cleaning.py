"""Turns raw per-page PDF text into clean blocks of prose and headings.

PDF extraction leaves artefacts that hurt retrieval: ligature characters,
words hyphenated across line breaks, running headers and footers repeated on
every page, page numbers, and hard line breaks inside paragraphs. This module
removes them while keeping each block tied to the page it came from.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# Characters that are invisible in the PDF but split words when extracted.
_INVISIBLE_CHARACTERS = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Lines at the top or bottom of a page that are page numbers, e.g. "7", "Page 7 of 12".
_PAGE_NUMBER_LINE = re.compile(r"^(page\s+)?\d{1,4}(\s*(/|of)\s*\d{1,4})?$", re.IGNORECASE)
# How many non-empty lines at each end of a page are checked for headers/footers.
EDGE_LINES_CHECKED = 2
# A line must repeat at a page edge on at least this share of pages to be dropped.
REPEATED_EDGE_LINE_SHARE = 0.5
MIN_PAGES_FOR_REPEAT_DETECTION = 3

_NAMED_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "related works",
    "preliminaries",
    "problem formulation",
    "problem statement",
    "method",
    "methods",
    "methodology",
    "approach",
    "proposed method",
    "proposed approach",
    "experiments",
    "experimental setup",
    "experimental results",
    "experimental evaluation",
    "results",
    "evaluation",
    "discussion",
    "analysis",
    "limitations",
    "future work",
    "conclusion",
    "conclusions",
    "conclusion and future work",
    "conclusions and future work",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "supplementary material",
}
# Section numbers: "3", "3.2", "3.2.1.", "IV.". Appendix letters ("A") are only
# accepted before a known section name, since "A Survey of ..." is not a heading.
_SECTION_NUMBER = r"(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,5}\.)"
_NUMBERED_HEADING = re.compile(rf"^{_SECTION_NUMBER}\s+([A-Z][^\n]*)$")
_NAMED_HEADING = re.compile(rf"^(?:(?:{_SECTION_NUMBER}|[A-H]\.?)\s+)?([A-Za-z][A-Za-z ]+?)\.?$")
# Table rows such as "1 FedAvg 85.3 90.1" start like numbered headings.
_NUMERIC_TOKEN = re.compile(r"^[\d.,%±+\-]+$")
MAX_HEADING_WORDS = 8
MAX_HEADING_CHARS = 60


@dataclass(frozen=True)
class Block:
    text: str
    is_heading: bool


@dataclass(frozen=True)
class CleanPage:
    page_number: int
    blocks: list[Block]


def normalize_text(text: str) -> str:
    """Unicode NFKC (turns "ﬁ" into "fi", full-width digits into ASCII) and
    removal of invisible and control characters."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE_CHARACTERS)
    return _CONTROL_CHARACTERS.sub("", text)


def is_heading(line: str) -> bool:
    """Recognise section headings such as "3.2 Experimental Setup" or "References"."""
    line = line.strip()
    if not line or len(line) > MAX_HEADING_CHARS or len(line.split()) > MAX_HEADING_WORDS:
        return False
    named = _NAMED_HEADING.match(line)
    if named and named.group(1).strip().lower() in _NAMED_SECTIONS:
        return True
    numbered = _NUMBERED_HEADING.match(line)
    if not numbered or line.endswith((".", ",", ";", ":")):
        return False  # numbered headings never end like a sentence or clause
    return not any(_NUMERIC_TOKEN.match(word) for word in numbered.group(1).split())


def _edge_key(line: str) -> str:
    # Digits are masked so "Page 3" and "Page 4" count as the same footer.
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def _edge_indices(lines: list[str]) -> list[int]:
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    return non_empty[:EDGE_LINES_CHECKED] + non_empty[-EDGE_LINES_CHECKED:]


def remove_headers_and_footers(pages_lines: list[list[str]]) -> list[list[str]]:
    """Drop running headers/footers and page numbers from the edges of each page."""
    repeated: set[str] = set()
    if len(pages_lines) >= MIN_PAGES_FOR_REPEAT_DETECTION:
        counts: Counter[str] = Counter()
        for lines in pages_lines:
            counts.update({_edge_key(lines[index]) for index in _edge_indices(lines)})
        threshold = max(
            MIN_PAGES_FOR_REPEAT_DETECTION,
            math.ceil(REPEATED_EDGE_LINE_SHARE * len(pages_lines)),
        )
        repeated = {key for key, count in counts.items() if count >= threshold}

    cleaned = []
    for lines in pages_lines:
        drop = {
            index
            for index in _edge_indices(lines)
            if _edge_key(lines[index]) in repeated or _PAGE_NUMBER_LINE.match(lines[index].strip())
        }
        cleaned.append([line for index, line in enumerate(lines) if index not in drop])
    return cleaned


def join_lines(lines: list[str]) -> str:
    """Join hard-wrapped lines into prose, re-joining words hyphenated at a line end.

    "effi-" + "cient" becomes "efficient"; "non-" + "IID" keeps its hyphen
    because a capital letter signals a real compound.
    """
    text = ""
    for raw_line in lines:
        line = " ".join(raw_line.split())
        if not line:
            continue
        broken_word = len(text) >= 2 and text.endswith("-") and text[-2].isalpha()
        if broken_word and line[0].islower():
            text = text[:-1] + line  # hyphen only marked the line break
        elif broken_word:
            text += line  # real compound: keep the hyphen, no space
        elif text:
            text = f"{text} {line}"
        else:
            text = line
    return text


def _split_into_blocks(lines: list[str]) -> list[Block]:
    """Group a page's lines into headings and the prose between them."""
    blocks: list[Block] = []
    prose: list[str] = []
    for line in lines:
        if not is_heading(line):
            prose.append(line)
            continue
        if text := join_lines(prose):
            blocks.append(Block(text=text, is_heading=False))
        prose = []
        blocks.append(Block(text=" ".join(line.split()), is_heading=True))
    if text := join_lines(prose):
        blocks.append(Block(text=text, is_heading=False))
    return blocks


def clean_pages(pages: list[tuple[int, str]]) -> list[CleanPage]:
    """Clean raw page texts into heading and prose blocks, page by page."""
    pages_lines = [normalize_text(text).splitlines() for _, text in pages]
    pages_lines = remove_headers_and_footers(pages_lines)
    return [
        CleanPage(page_number=page_number, blocks=_split_into_blocks(lines))
        for (page_number, _), lines in zip(pages, pages_lines, strict=True)
    ]
