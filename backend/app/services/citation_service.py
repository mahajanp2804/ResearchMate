"""Validating the citations in a generated answer.

The LLM cites the numbered sources it was given as [S1], [S2], ... This
module checks those markers in code rather than trusting them:

- Every marker is parsed, including groups ([S1, S3]), semicolons, ranges
  ([S1-S3]) and lowercase ids.
- Ids that do not match a source actually sent are removed from the answer
  and reported, so an invented citation can never reach the user.
- Each valid id is resolved to its document, pages and section from the
  retrieved passage, never from anything the model wrote.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

# A bracketed list of source ids: [S1], [S1, S3], [S2; S4], [S1-S3], [s2].
CITATION_MARKER = re.compile(
    r"\[\s*S\d+(?:\s*[-–]\s*S\d+)?(?:\s*[,;]\s*S\d+(?:\s*[-–]\s*S\d+)?)*\s*\]",
    re.IGNORECASE,
)
_ID = re.compile(r"S(\d+)(?:\s*[-–]\s*S(\d+))?", re.IGNORECASE)
# Ranges longer than this are treated as a model error rather than expanded.
MAX_RANGE = 20


@dataclass(frozen=True)
class CitationCheck:
    # The answer with invalid ids removed and markers written as [S1][S3].
    text: str
    # Valid source ids in order of first citation, without duplicates.
    cited_ids: list[str] = field(default_factory=list)
    # Ids the model cited that were not among the sources sent.
    invalid_ids: list[str] = field(default_factory=list)


def parse_marker(marker: str) -> list[str]:
    """Ids in one bracketed marker, in order, with ranges expanded: "[S1-S3, s5]" ->
    ["S1", "S2", "S3", "S5"]."""
    ids: list[str] = []
    for match in _ID.finditer(marker):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start or end - start > MAX_RANGE:
            end = start
        ids.extend(f"S{number}" for number in range(start, end + 1))
    return ids


def _tidy(text: str) -> str:
    """Remove the gaps left where markers were deleted."""
    text = re.sub(r"[ \t]+([.,;:!?)])", r"\1", text)  # "claim ." -> "claim."
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.rstrip() for line in text.split("\n"))


def validate_citations(answer: str, source_ids: Sequence[str]) -> CitationCheck:
    """Keep only citations to ``source_ids`` and report the rest."""
    allowed = set(source_ids)
    cited: list[str] = []
    invalid: list[str] = []

    def rewrite(match: re.Match[str]) -> str:
        kept = []
        for source_id in parse_marker(match.group(0)):
            if source_id in allowed:
                if source_id not in kept:
                    kept.append(source_id)
                if source_id not in cited:
                    cited.append(source_id)
            elif source_id not in invalid:
                invalid.append(source_id)
        return "".join(f"[{source_id}]" for source_id in kept)

    text = CITATION_MARKER.sub(rewrite, answer)
    return CitationCheck(text=_tidy(text).strip(), cited_ids=cited, invalid_ids=invalid)


# --- Supporting excerpts and exact pages -----------------------------------

MAX_EXCERPT_CHARS = 400
# A very short best sentence ("See Table 2.") is extended with the next one.
MIN_EXCERPT_CHARS = 60
# Characters of a normalised excerpt used to find it in a page's raw text.
PAGE_MATCH_CHARS = 80
SHORT_PAGE_MATCH_CHARS = 30


@dataclass(frozen=True)
class SourcePassage:
    """What the citation checker needs to know about a source passage."""

    source_id: str
    text: str
    page_start: int
    page_end: int
    document_id: str = ""
    # The paper's title, which can run into the first sentence of page 1.
    title: str = ""


@dataclass(frozen=True)
class Citation:
    source_id: str
    # The sentence(s) of the source that best match the statements citing it.
    excerpt: str
    # The page the excerpt is on (exact even when the passage spans pages).
    page: int
    # Share of the citing statements' key terms found in the source (0-1).
    # Low values suggest a weak citation; kept for evaluation, not enforced.
    support: float


_LIST_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _strip_markers(text: str) -> str:
    """A statement without citation markers or a leading list bullet."""
    text = _LIST_BULLET.sub("", CITATION_MARKER.sub("", text))
    return " ".join(_tidy(text).split())


def statements_citing(answer: str) -> dict[str, list[str]]:
    """For each cited id, the answer sentences that cite it (markers removed)."""
    from app.services.chunking_service import split_sentences

    statements: dict[str, list[str]] = {}
    for line in answer.split("\n"):
        for sentence in split_sentences(line.strip()):
            ids = [i for marker in CITATION_MARKER.findall(sentence) for i in parse_marker(marker)]
            for source_id in dict.fromkeys(ids):
                statements.setdefault(source_id, []).append(_strip_markers(sentence))
    return statements


def _shorten(text: str) -> str:
    if len(text) <= MAX_EXCERPT_CHARS:
        return text
    return text[:MAX_EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"


def best_excerpt(source_text: str, claims: Sequence[str], title: str = "") -> tuple[str, float]:
    """The source sentence sharing most key terms with ``claims``, and that share."""
    from app.services.bm25_service import analyze
    from app.services.chunking_service import split_sentences
    from app.services.text_cleaning import is_heading

    # Passages keep section headings on their own line; they are not evidence.
    prose = " ".join(line for line in source_text.split("\n") if not is_heading(line))
    prose = " ".join(prose.split())
    # A title line has no full stop, so it would merge into the first sentence.
    if title and prose.startswith(title):
        prose = prose[len(title) :].lstrip()
    sentences = split_sentences(prose) or [source_text.strip()]
    claim_terms = set(analyze(" ".join(claims)))
    if not claim_terms:
        return _shorten(sentences[0]), 0.0

    scores = [len(claim_terms & set(analyze(sentence))) for sentence in sentences]
    best = max(range(len(sentences)), key=lambda index: (scores[index], -index))
    excerpt = sentences[best]
    if len(excerpt) < MIN_EXCERPT_CHARS and best + 1 < len(sentences):
        excerpt = f"{excerpt} {sentences[best + 1]}"
    support = len(claim_terms & set(analyze(source_text))) / len(claim_terms)
    return _shorten(excerpt), round(support, 3)


def _letters_and_digits(text: str) -> str:
    # Ignores whitespace, line-break hyphens and ligatures (NFKC), which differ
    # between the cleaned passage and the raw page text.
    import unicodedata

    return re.sub(r"[^0-9a-z]", "", unicodedata.normalize("NFKC", text).lower())


def locate_page(page_texts: dict[int, str], excerpt: str, page_start: int, page_end: int) -> int:
    """The page of a multi-page passage on which ``excerpt`` appears."""
    if page_start == page_end:
        return page_start
    key = _letters_and_digits(excerpt.rstrip("…"))
    for length in (PAGE_MATCH_CHARS, SHORT_PAGE_MATCH_CHARS):
        needle = key[:length]
        if not needle:
            break
        for page in range(page_start, page_end + 1):
            if needle in _letters_and_digits(page_texts.get(page, "")):
                return page
    return page_start


def build_citations(
    answer: str,
    sources: Sequence[SourcePassage],
    page_texts: dict[str, dict[int, str]] | None = None,
) -> list[Citation]:
    """A citation with excerpt, page and support for each source the answer cites.

    ``answer`` must already be validated. ``page_texts`` maps a document id to
    its raw page texts; it is only needed for passages that span pages.
    """
    by_id = {source.source_id: source for source in sources}
    citations = []
    for source_id, claims in statements_citing(answer).items():
        source = by_id.get(source_id)
        if source is None:
            continue
        excerpt, support = best_excerpt(source.text, claims, source.title)
        pages = (page_texts or {}).get(source.document_id, {})
        page = locate_page(pages, excerpt, source.page_start, source.page_end)
        citations.append(Citation(source_id, excerpt, page, support))
    return citations


# --- Coverage: statements without citations ---------------------------------

CoverageStatus = Literal["cited", "partially_cited", "uncited"]

# Statements that report missing information are not claims needing a source,
# e.g. "The papers do not report the hardware used."
_ABSENCE = re.compile(
    r"\b(?:do|does|did)\s+not\s+(?:mention|report|discuss|cover|provide|specify|describe|"
    r"state|include|address|compare|evaluate|give|say)"
    r"|\bnot\s+(?:mentioned|reported|discussed|covered|specified|described|stated|provided|"
    r"addressed|available)"
    r"|\bno\s+(?:information|details?|mention|data)\b"
    r"|\b(?:unclear|unknown)\s+from\s+the\s+(?:sources|papers)",
    re.IGNORECASE,
)
# Shorter fragments ("**Key findings**", "In short:") are not treated as claims.
MIN_CLAIM_WORDS = 5


@dataclass(frozen=True)
class Coverage:
    status: CoverageStatus
    cited_statements: int
    # Statements that make a claim without citing any source.
    uncited_statements: list[str] = field(default_factory=list)


def citation_coverage(answer: str) -> Coverage:
    """How much of a (validated) answer is backed by citations.

    A heuristic for warning the reader, not a proof: a sentence is counted as
    cited when it contains a marker, and as an uncited claim when it has none
    and is not a heading, a list introduction, a short fragment or a statement
    that information is missing.
    """
    from app.services.chunking_service import split_sentences

    cited = 0
    uncited: list[str] = []
    for line in answer.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sentence in split_sentences(line):
            if CITATION_MARKER.search(sentence):
                cited += 1
                continue
            statement = _strip_markers(sentence)
            if (
                statement.endswith(":")
                or len(re.findall(r"[A-Za-z0-9]+", statement)) < MIN_CLAIM_WORDS
                or _ABSENCE.search(statement)
            ):
                continue
            uncited.append(statement)

    if cited == 0:
        status: CoverageStatus = "uncited"
    elif uncited:
        status = "partially_cited"
    else:
        status = "cited"
    return Coverage(status=status, cited_statements=cited, uncited_statements=uncited)
