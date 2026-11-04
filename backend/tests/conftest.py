from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.main import create_app


def make_settings(data_dir: Path, **overrides) -> Settings:
    # _env_file=None keeps a developer's real backend/.env (and API key) out of tests.
    values = {"environment": "test", "data_dir": data_dir, "gemini_api_key": None}
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path / "data")


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # Entering the context runs the lifespan, which creates directories and tables.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session(client: TestClient, app: FastAPI) -> Iterator[Session]:
    with app.state.database.session() as session:
        yield session
