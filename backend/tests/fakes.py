"""Deterministic stand-ins for model-backed components, so most tests run offline."""

import re
import zlib
from collections.abc import Sequence

import numpy as np

from app.exceptions import ServiceUnavailableError

_WORD = re.compile(r"\S+")
_TERM = re.compile(r"[a-z0-9]+")


class WhitespaceTokenizer:
    """Counts whitespace-separated words as tokens. Same protocol as the real tokenizer."""

    name = "test-whitespace"

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(_WORD.findall(text)) for text in texts]

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = _WORD.findall(text)
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def unavailable_tokenizer():
    raise ServiceUnavailableError(
        "Could not load the tokenizer (simulated outage).", code="tokenizer_unavailable"
    )


class HashingEmbedder:
    """Bag-of-words vectors hashed into a few dimensions, then L2-normalised.

    Texts sharing words get similar vectors, which is enough to test indexing
    and search without a neural model. Records every batch it embeds.
    """

    max_seq_length = 512

    def __init__(self, name: str = "test-hashing", dimension: int = 64) -> None:
        self.name = name
        self.dimension = dimension
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in _TERM.findall(text.lower()):
                vectors[row, zlib.crc32(term.encode()) % self.dimension] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.batches)


def unavailable_embedder():
    raise ServiceUnavailableError(
        "Could not load the embedding model (simulated outage).",
        code="embedding_model_unavailable",
    )


class OverlapReranker:
    """Scores a passage by the share of query terms it contains.

    Mapped onto roughly the cross-encoder's logit range: about -11 for an
    unrelated passage up to +9 when every query term appears, so the default
    evidence threshold (-10) behaves realistically in tests.
    """

    name = "test-overlap"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, len(passages)))
        query_terms = set(_TERM.findall(query.lower()))
        scores = []
        for passage in passages:
            shared = query_terms & set(_TERM.findall(passage.lower()))
            fraction = len(shared) / len(query_terms) if query_terms else 0.0
            scores.append(-11.0 + 20.0 * fraction)
        return scores


def unavailable_reranker():
    raise ServiceUnavailableError(
        "Could not load the reranking model (simulated outage).", code="reranker_unavailable"
    )


class ScriptedLLM:
    """Stand-in LLM that records every call and answers deterministically.

    Rewrite calls return the latest question unchanged unless ``rewrite`` is
    given. Answer calls cite the first source with its first sentence unless
    ``answer`` is given. Either can be a string or a function of the prompt.
    """

    name = "scripted"
    model = "scripted-model"

    def __init__(self, answer=None, rewrite=None) -> None:
        self.answer = answer
        self.rewrite = rewrite
        self.calls: list[dict] = []

    def generate(
        self, *, system_prompt, turns, temperature=None, max_output_tokens=None, json_schema=None
    ):
        from app.prompts.rewrite_prompt import REWRITE_SYSTEM_PROMPT
        from app.services.llm_service import LLMResponse

        prompt = turns[-1].content
        kind = "rewrite" if system_prompt == REWRITE_SYSTEM_PROMPT else "answer"
        self.calls.append(
            {
                "kind": kind,
                "system": system_prompt,
                "turns": list(turns),
                "temperature": temperature,
            }
        )
        script = self.rewrite if kind == "rewrite" else self.answer
        if callable(script):
            text = script(prompt)
        elif script is not None:
            text = script
        elif kind == "rewrite":
            text = prompt.split("Latest question: ", 1)[1].split("\n", 1)[0]
        else:
            first_source = prompt.split(">\n", 1)[1].split("\n</source>", 1)[0]
            text = f"{first_source.split('. ')[0].rstrip('.')} [S1]."
        return LLMResponse(text=text, model=self.model, prompt_tokens=len(prompt.split()))

    def calls_of(self, kind: str) -> list[dict]:
        return [call for call in self.calls if call["kind"] == kind]
