from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.main import create_app
from tests.fakes import HashingEmbedder, OverlapReranker, ScriptedLLM, WhitespaceTokenizer


def make_settings(data_dir: Path, **overrides) -> Settings:
    # _env_file=None keeps a developer's real backend/.env (and API key) out of tests.
    values = {"environment": "test", "data_dir": data_dir, "gemini_api_key": None}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_app(settings: Settings, **overrides) -> FastAPI:
    """App wired with offline fakes unless a test overrides them.

    The fake embedder takes the configured model's name, as the real one would.
    """
    embedder = HashingEmbedder(name=settings.embedding_model_name)
    reranker = OverlapReranker()
    llm = ScriptedLLM()
    values = {
        "tokenizer_provider": WhitespaceTokenizer,
        "embedder_provider": lambda: embedder,
        "reranker_provider": lambda: reranker,
        "llm_provider": lambda: llm,
    }
    values.update(overrides)
    app = create_app(settings, **values)
    app.state.test_embedder = embedder
    app.state.test_reranker = reranker
    app.state.test_llm = llm
    return app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path / "data")


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return make_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # Entering the context runs the lifespan, which creates directories and tables.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session(client: TestClient, app: FastAPI) -> Iterator[Session]:
    with app.state.database.session() as session:
        yield session
