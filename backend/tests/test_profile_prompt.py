"""The paper-profile prompt: aspects, prompt text, JSON schema and reply parsing."""

import json

import pytest

from app.prompts.profile_prompt import (
    ASPECTS,
    ASPECTS_BY_KEY,
    COMPARISON_ASPECTS,
    SUMMARY_SECTIONS,
    ExtractedField,
    build_profile_turns,
    parse_profile_reply,
    profile_json_schema,
)
from app.services.retrieval_service import RetrievedChunk


def chunk(text: str, number: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=number,
        score=1.0,
        chunk_id=number,
        document_id="doc",
        document_title='FedProx "Heterogeneous" Networks',
        chunk_index=number,
        page_start=number,
        page_end=number,
        section="Experiments",
        text=text,
        token_count=len(text.split()),
    )


def test_summary_and_comparison_use_known_aspects() -> None:
    # The nine summary sections and ten comparison rows from the specification.
    assert len(SUMMARY_SECTIONS) == 9 and len(COMPARISON_ASPECTS) == 10
    assert set(SUMMARY_SECTIONS) | set(COMPARISON_ASPECTS) == set(ASPECTS_BY_KEY)
    assert len(ASPECTS_BY_KEY) == len(ASPECTS)  # keys are unique


def test_prompt_lists_every_field_and_escapes_the_paper() -> None:
    sources = [chunk("We use <b>MNIST</b> & FEMNIST.", 1), chunk("Results improve.", 2)]
    (turn,) = build_profile_turns('FedProx "Heterogeneous" Networks', sources)

    assert turn.role == "user"
    assert 'Paper: "FedProx &quot;Heterogeneous&quot; Networks"' in turn.content
    assert '<source id="S1"' in turn.content and '<source id="S2"' in turn.content
    assert "&lt;b&gt;MNIST&lt;/b&gt; &amp; FEMNIST" in turn.content
    for aspect in ASPECTS:
        assert f"- {aspect.key} ({aspect.label}):" in turn.content


def test_schema_requires_every_aspect_with_a_found_flag() -> None:
    schema = profile_json_schema()
    assert schema["required"] == [aspect.key for aspect in ASPECTS]
    datasets = schema["properties"]["datasets"]
    assert datasets["required"] == ["found", "text"]
    assert datasets["properties"]["found"] == {"type": "boolean"}


def test_reply_is_parsed_by_aspect() -> None:
    reply = json.dumps(
        {
            "datasets": {"found": True, "text": "  MNIST and FEMNIST [S1]. "},
            "future_work": {"found": False, "text": ""},
        }
    )
    fields = parse_profile_reply(reply)
    assert fields["datasets"] == ExtractedField(True, "MNIST and FEMNIST [S1].")
    assert fields["future_work"] == ExtractedField(False, "")
    assert "key_results" not in fields  # missing aspects are left out


def test_reply_in_a_code_fence_is_accepted() -> None:
    reply = '```json\n{"datasets": {"found": true, "text": "MNIST [S1]."}}\n```'
    assert parse_profile_reply(reply)["datasets"].text == "MNIST [S1]."


@pytest.mark.parametrize(
    "value",
    [
        {"found": "yes", "text": "MNIST [S1]."},  # not a boolean
        {"found": True},  # no text
        "MNIST [S1].",  # not an object
    ],
)
def test_malformed_fields_are_dropped(value) -> None:
    assert parse_profile_reply(json.dumps({"datasets": value})) == {}


@pytest.mark.parametrize("reply", ["Here is the profile: {", "[1, 2]", ""])
def test_unreadable_replies_raise(reply: str) -> None:
    with pytest.raises(ValueError):
        parse_profile_reply(reply)
