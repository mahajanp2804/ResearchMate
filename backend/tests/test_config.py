from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import BACKEND_DIR
from tests.conftest import make_settings


def test_overlap_must_be_smaller_than_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        make_settings(tmp_path, chunk_size_tokens=200, chunk_overlap_tokens=200)


def test_relative_data_dir_resolves_against_backend_dir() -> None:
    settings = make_settings(Path("some-data"))
    assert settings.data_dir == (BACKEND_DIR / "some-data").resolve()


def test_default_database_lives_in_data_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.database_url == f"sqlite:///{tmp_path / 'researchmate.db'}"
    assert settings.upload_dir == tmp_path / "uploads"
    assert settings.index_dir == tmp_path / "indexes"


def test_settings_read_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "512")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    settings = make_settings(tmp_path)
    assert settings.chunk_size_tokens == 512
    assert settings.reranker_enabled is False


def test_api_key_is_masked(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, gemini_api_key="super-secret-key")
    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in str(settings.model_dump())


def test_llm_configured_requires_non_empty_key(tmp_path: Path) -> None:
    assert make_settings(tmp_path).llm_configured is False
    assert make_settings(tmp_path, gemini_api_key="").llm_configured is False
    assert make_settings(tmp_path, gemini_api_key="abc").llm_configured is True
