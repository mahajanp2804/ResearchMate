"""Application configuration.

Every tunable value (paths, model names, chunking and retrieval parameters, LLM
settings) lives here and can be overridden through environment variables or
``backend/.env``. Services receive a ``Settings`` instance instead of reading
the environment themselves, which keeps them testable.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "ResearchMate"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Storage -----------------------------------------------------------
    # Relative paths are resolved against the backend directory so the app
    # behaves the same regardless of the working directory it is started from.
    data_dir: Path = Path("data")
    database_url: str | None = None  # defaults to SQLite inside data_dir

    # --- Uploads -----------------------------------------------------------
    max_upload_size_mb: int = Field(default=50, gt=0)
    max_files_per_upload: int = Field(default=10, gt=0)
    # Guards against pathological or decompression-bomb style PDFs.
    max_pages_per_document: int = Field(default=500, gt=0)

    # --- Chunking ----------------------------------------------------------
    chunk_size_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int = Field(default=150, ge=0)

    # --- Embeddings --------------------------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_device: str = "cpu"

    # --- Retrieval ---------------------------------------------------------
    dense_top_k: int = Field(default=30, gt=0)
    bm25_top_k: int = Field(default=30, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, gt=0)
    final_context_chunks: int = Field(default=6, gt=0)
    max_context_tokens: int = Field(default=6000, gt=0)

    # --- LLM ---------------------------------------------------------------
    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=2048, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if not self.data_dir.is_absolute():
            self.data_dir = (BACKEND_DIR / self.data_dir).resolve()
        if self.database_url is None:
            self.database_url = f"sqlite:///{self.data_dir / 'researchmate.db'}"
        return self

    # Derived locations. Kept as properties so they always follow data_dir.
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        return self.gemini_api_key is not None and bool(self.gemini_api_key.get_secret_value())

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.index_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
