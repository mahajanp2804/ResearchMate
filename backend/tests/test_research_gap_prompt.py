"""The research gap prompt: prompt text, JSON schema, reply parsing and scope checks."""

import json

import pytest

from app.prompts.research_gap_prompt import (
    MAX_LIMITATIONS_PER_PAPER,
    RawFinding,
    RawLimitation,
    RawOpportunity,
    build_gap_turns,
    claims_beyond_scope,
    gap_json_schema,
    parse_gap_reply,
)
from app.services.retrieval_service import RetrievedChunk


def chunk(text: str, page: int = 1, section: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        score=1.0,
        chunk_id=page,
        document_id="doc",
        document_title="ignored",
        chunk_index=page,
        page_start=page,
        page_end=page,
        section=section,
        text=text,
        token_count=len(text.split()),
    )


def test_prompt_labels_papers_and_numbers_sources_across_them() -> None:
    (turn,) = build_gap_turns(
        ['FedProx "Heterogeneous"', "SCAFFOLD"],
        [(0, chunk("Assumes <bounded> dissimilarity.", 3, "Limitations")), (1, chunk("Drift.", 2))],
    )
    content = turn.content
    assert '- P1: "FedProx &quot;Heterogeneous&quot;"' in content
    assert '- P2: "SCAFFOLD"' in content
    assert '<source id="S1" paper="P1" pages="3" section="Limitations">' in content
    assert "Assumes &lt;bounded&gt; dissimilarity." in content
    assert '<source id="S2" paper="P2" pages="2">' in content
    assert "stated_limitations" in content and "opportunities" in content


def test_schema_limits_papers_and_list_lengths() -> None:
    schema = gap_json_schema(3)
    limitation = schema["properties"]["stated_limitations"]
    assert limitation["items"]["properties"]["paper"]["enum"] == ["P1", "P2", "P3"]
    assert limitation["maxItems"] == 3 * MAX_LIMITATIONS_PER_PAPER
    assert schema["properties"]["opportunities"]["items"]["required"] == [
        "title",
        "evidence",
        "suggestion",
    ]
    assert set(schema["required"]) == {
        "stated_limitations",
        "recurring_challenges",
        "differences",
        "limited_discussion",
        "opportunities",
    }


def test_reply_is_parsed_into_its_three_kinds() -> None:
    reply = json.dumps(
        {
            "stated_limitations": [{"paper": "P2", "text": " Doubles communication [S4]. "}],
            "recurring_challenges": [{"title": "Drift", "text": "Both note drift [S1][S4]."}],
            "differences": [],
            "limited_discussion": [{"title": "Privacy", "text": "Only P1 mentions it [S2]."}],
            "opportunities": [
                {"title": "Cheaper", "evidence": "Costs double [S4].", "suggestion": "Compress."}
            ],
        }
    )
    parsed = parse_gap_reply(reply, paper_count=2)
    assert parsed.stated_limitations == [RawLimitation(1, "Doubles communication [S4].")]
    assert parsed.patterns["recurring_challenges"] == [
        RawFinding("Drift", "Both note drift [S1][S4].")
    ]
    assert parsed.patterns["differences"] == []
    assert parsed.opportunities == [RawOpportunity("Cheaper", "Costs double [S4].", "Compress.")]
    assert parsed.malformed == 0


def test_malformed_items_are_counted_and_left_out() -> None:
    reply = json.dumps(
        {
            "stated_limitations": [
                {"paper": "P9", "text": "Unknown paper [S1]."},
                {"paper": "P1", "text": ""},
                "not an object",
            ],
            "recurring_challenges": {"title": "not a list"},
            "opportunities": [{"title": "No evidence", "evidence": "", "suggestion": "Do X."}],
        }
    )
    parsed = parse_gap_reply(reply, paper_count=2)
    assert parsed.stated_limitations == [] and parsed.opportunities == []
    assert parsed.patterns["recurring_challenges"] == []
    assert parsed.malformed == 5  # missing lists count as empty, not malformed


def test_fenced_json_is_accepted_and_garbage_is_not() -> None:
    assert parse_gap_reply('```json\n{"differences": []}\n```', 2).malformed == 0
    with pytest.raises(ValueError):
        parse_gap_reply("Here are the gaps:", 2)
    with pytest.raises(ValueError):
        parse_gap_reply("[]", 2)


@pytest.mark.parametrize(
    "text",
    [
        "No prior work studies privacy in this setting.",
        "This has never been explored.",
        "Communication cost has not yet been addressed.",
        "An unexplored direction is client sampling.",
        "This is a novel research gap.",
        "There is a clear gap in the literature.",
        "Few studies in the wider literature examine it.",
        "We would be the first to evaluate it at scale.",
    ],
)
def test_global_claims_are_detected(text: str) -> None:
    assert claims_beyond_scope(text)


@pytest.mark.parametrize(
    "text",
    [
        "None of the selected papers evaluates on more than 500 clients.",
        "Among the selected papers, only P1 discusses stragglers.",
        "Future work could test the method with real network traces.",
        "The papers do not report wall-clock time.",
    ],
)
def test_scoped_statements_are_not_flagged(text: str) -> None:
    assert not claims_beyond_scope(text)
