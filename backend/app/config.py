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
    # Measured with the embedding model's tokenizer. all-MiniLM-L6-v2 reads at
    # most 256 tokens (254 plus [CLS]/[SEP]), so larger chunks would be cut off
    # when embedded. Raise this together with a longer-context embedding model.
    chunk_size_tokens: int = Field(default=250, ge=32)
    chunk_overlap_tokens: int = Field(default=50, ge=0)

    # --- Embeddings --------------------------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, gt=0)
    # "mps" uses the Apple Silicon GPU; "cpu" is the portable default.
    embedding_device: Literal["cpu", "mps", "cuda"] = "cpu"
    # Some models (e.g. BGE, E5) expect an instruction before search queries.
    # all-MiniLM-L6-v2 does not, so it is empty by default.
    embedding_query_prefix: str = ""

    # --- Retrieval ---------------------------------------------------------
    dense_top_k: int = Field(default=30, gt=0)
    bm25_top_k: int = Field(default=30, gt=0)
    # BM25 term-frequency saturation (k1) and document-length normalisation (b);
    # 1.5 and 0.75 are the standard Okapi defaults.
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    # Reciprocal Rank Fusion constant: larger values flatten the advantage of top ranks.
    rrf_k: int = Field(default=60, gt=0)
    # Whether the answer pipeline reranks; the search API can always compare both.
    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many fused candidates the cross-encoder scores (cost grows linearly).
    rerank_candidates: int = Field(default=20, gt=0)
    # Optional evidence gate on the cross-encoder's raw score (logit): passages
    # below it are dropped, and if none remain the question is refused without
    # calling the LLM. Off by default: with ms-marco-MiniLM-L-6-v2, answerable
    # paraphrased research questions scored as low as -11.1 while unanswerable
    # ones scored up to -9.9, so no threshold separates them (it still ranked
    # the right passage first or second in every case). Enable it with a
    # better-calibrated reranker; the Phase 11 evaluation measures the effect.
    min_rerank_score: float | None = None
    final_context_chunks: int = Field(default=8, gt=0)
    max_context_tokens: int = Field(default=6000, gt=0)

    # --- LLM ---------------------------------------------------------------
    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr | None = None
    # Current stable Flash model (Gemini API docs, November 2026);
    # gemini-2.5-flash is deprecated.
    gemini_model: str = "gemini-3.6-flash"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    # Gemini 3 models "think" before answering and the thinking tokens can
    # count towards this limit, so it leaves room beyond the answer itself.
    llm_max_output_tokens: int = Field(default=8192, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)
    # Earlier messages of a conversation used to resolve follow-up questions.
    llm_history_messages: int = Field(default=6, ge=0, le=20)

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
