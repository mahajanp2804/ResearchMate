"""Sentence- and section-aware chunking measured in embedding-model tokens.

Pipeline: cleaned page blocks -> sentences (each tagged with its page range and
section) -> chunks packed up to ``chunk_size`` tokens.

- Chunks end on sentence boundaries; only a single sentence longer than the
  limit is cut mid-sentence.
- Consecutive chunks share up to ``chunk_overlap`` tokens of whole trailing
  sentences, so a fact split across a boundary is still retrievable.
- A new section heading starts a new chunk, and overlap never crosses a
  section boundary.
- Every chunk records the first and last page its text came from, so
  citations stay exact even when a sentence runs across a page break.
"""

import re
from dataclasses import dataclass

from app.services.text_cleaning import CleanPage, clean_pages, join_lines
from app.services.tokenization import Tokenizer
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Bump when the algorithm changes so stored chunks are known to be outdated.
CHUNKER_VERSION = 2
# A heading closes the current chunk only if it already holds this share of
# the chunk size; otherwise a short section is merged into the next one.
SECTION_BREAK_MIN_FILL = 0.2
MAX_SECTION_LENGTH = 255

# Split after . ! ? (optionally followed by a closing quote or bracket) when the
# next sentence starts with a capital, digit, quote or bracket.
_SENTENCE_BOUNDARY = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+(?=[\"'“‘(\[]?[A-Z0-9])")
# A period after these does not end a sentence ("et al. show", "Fig. 3").
_ABBREVIATIONS = {
    "al", "e.g", "i.e", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "ref",
    "refs", "no", "vs", "cf", "approx", "resp", "thm", "def", "prop", "dr", "prof",
}  # fmt: skip
_TERMINAL_PUNCTUATION = (".", "!", "?", ":")


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int
    char_count: int


@dataclass
class _Unit:
    """A sentence or heading with its location; the atom chunks are built from."""

    text: str
    page_start: int
    page_end: int
    section: str | None
    is_heading: bool
    tokens: int = 0


def chunking_signature(tokenizer_name: str, chunk_size: int, chunk_overlap: int) -> str:
    """Identifies how a document was chunked, to detect when re-chunking is needed."""
    return f"v{CHUNKER_VERSION}:{tokenizer_name}:{chunk_size}/{chunk_overlap}"


def _ends_with_abbreviation(text: str) -> bool:
    # Leading brackets/quotes are ignored so "(Tab." still counts as "tab".
    last_word = text.rsplit(None, 1)[-1].lower().rstrip(".").lstrip("([{\"'“‘")
    # Single letters are initials ("J. Smith") rather than sentence ends.
    return last_word in _ABBREVIATIONS or bool(re.fullmatch(r"[a-z]", last_word))


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_BOUNDARY.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _continues_previous_page(previous: _Unit, page_number: int, section: str | None) -> bool:
    """True if the previous page ended mid-sentence and this page carries it on."""
    return (
        not previous.is_heading
        and previous.page_end == page_number - 1
        and previous.section == section
        and not previous.text.rstrip("\"'”’)]").endswith(_TERMINAL_PUNCTUATION)
    )


def build_units(pages: list[CleanPage]) -> list[_Unit]:
    units: list[_Unit] = []
    section: str | None = None
    for page in pages:
        for block in page.blocks:
            if block.is_heading:
                section = block.text[:MAX_SECTION_LENGTH]
                units.append(_Unit(block.text, page.page_number, page.page_number, section, True))
                continue
            for index, sentence in enumerate(split_sentences(block.text)):
                if (
                    index == 0
                    and units
                    and _continues_previous_page(units[-1], page.page_number, section)
                ):
                    previous = units[-1]
                    previous.text = join_lines([previous.text, sentence])
                    previous.page_end = page.page_number
                    continue
                units.append(_Unit(sentence, page.page_number, page.page_number, section, False))
    return units


def _split_to_fit(unit: _Unit, tokenizer: Tokenizer, chunk_size: int) -> list[_Unit]:
    """Cut a unit longer than the chunk size (e.g. a table flattened into one
    "sentence") into pieces that each fit."""
    pieces = tokenizer.split(unit.text, chunk_size)
    counts = tokenizer.count_many(pieces)
    result = []
    for piece, count in zip(pieces, counts, strict=True):
        part = _Unit(piece, unit.page_start, unit.page_end, unit.section, unit.is_heading, count)
        if count > chunk_size:
            # Re-tokenising a fragment can add a token at the cut; split again tighter.
            tighter = max(1, chunk_size - (count - chunk_size))
            result.extend(_split_to_fit(part, tokenizer, tighter))
        else:
            result.append(part)
    return result


def _total_tokens(units: list[_Unit]) -> int:
    return sum(unit.tokens for unit in units)


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """Trailing whole sentences totalling at most ``overlap`` tokens.

    Never returns every unit, or the next chunk would repeat this one.
    """
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units[1:]):
        if total + unit.tokens > overlap:
            break
        tail.insert(0, unit)
        total += unit.tokens
    return tail


def _join_units(units: list[_Unit]) -> str:
    parts: list[str] = []
    for index, unit in enumerate(units):
        if index > 0:
            boundary_is_heading = unit.is_heading or units[index - 1].is_heading
            parts.append("\n" if boundary_is_heading else " ")
        parts.append(unit.text)
    return "".join(parts)


def chunk_units(
    units: list[_Unit], tokenizer: Tokenizer, *, chunk_size: int, chunk_overlap: int
) -> list[ChunkDraft]:
    counts = tokenizer.count_many([unit.text for unit in units])
    sized: list[_Unit] = []
    for unit, count in zip(units, counts, strict=True):
        unit.tokens = count
        sized.extend(_split_to_fit(unit, tokenizer, chunk_size) if count > chunk_size else [unit])

    min_fill = max(1, int(chunk_size * SECTION_BREAK_MIN_FILL))
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    carried = 0  # leading units of `current` repeated from the previous chunk

    for unit in sized:
        fresh = current[carried:]
        if unit.is_heading and (not fresh or _total_tokens(fresh) >= min_fill):
            # A new section starts a new chunk; overlap never crosses sections.
            if fresh:
                groups.append(current)
            current, carried = [], 0
        elif fresh and _total_tokens(current) + unit.tokens > chunk_size:
            groups.append(current)
            current = _overlap_tail(current, chunk_overlap)
            carried = len(current)
        # Drop overlap from the front if the next unit would not fit with it.
        while carried and _total_tokens(current) + unit.tokens > chunk_size:
            current.pop(0)
            carried -= 1
        current.append(unit)
    if current[carried:]:
        groups.append(current)

    texts = [_join_units(group) for group in groups]
    exact_counts = tokenizer.count_many(texts)
    drafts = []
    for index, (group, text, token_count) in enumerate(
        zip(groups, texts, exact_counts, strict=True)
    ):
        if token_count > chunk_size:
            # Possible only with tokenizers that merge across spaces; logged, not fatal.
            logger.warning("Chunk %d has %d tokens (limit %d)", index, token_count, chunk_size)
        drafts.append(
            ChunkDraft(
                chunk_index=index,
                text=text,
                page_start=min(unit.page_start for unit in group),
                page_end=max(unit.page_end for unit in group),
                # First labelled unit, so title lines before "Abstract" still
                # yield a chunk labelled "Abstract".
                section=next((unit.section for unit in group if unit.section), None),
                token_count=token_count,
                char_count=len(text),
            )
        )
    return drafts


def chunk_pages(
    pages: list[tuple[int, str]], tokenizer: Tokenizer, *, chunk_size: int, chunk_overlap: int
) -> list[ChunkDraft]:
    """Clean raw ``(page_number, text)`` pages and split them into chunks."""
    units = build_units(clean_pages(pages))
    return chunk_units(units, tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
