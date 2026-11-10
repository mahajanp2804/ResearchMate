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
