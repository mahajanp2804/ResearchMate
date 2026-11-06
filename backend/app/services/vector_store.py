"""Persistent FAISS index of chunk vectors, keyed by chunk id.

Design:
- ``IndexIDMap2(IndexFlatIP)``: exact inner-product search over normalised
  vectors (= cosine similarity), with our own int64 ids. The id of every
  vector is the ``chunks.id`` primary key, so no separate mapping table is
  needed and deleting a document removes exactly its vectors.
- A JSON manifest beside the index records which embedding model produced
  the vectors. Vectors from different models are not comparable, so a model
  change discards the index instead of mixing them.
- Writes go to a temporary file that is then atomically renamed, so a crash
  mid-save never leaves a half-written index.
"""

import json
import os
import sys
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# macOS wheels of PyTorch and FAISS each bundle their own OpenMP runtime.
# Three rules keep them working in one process (verified against NumPy's
# exact results in tests/test_vector_store.py):
# 1. Allow a second runtime to start. scikit-learn, which sentence-transformers
#    imports, already sets this flag; it is set here so it never depends on
#    import side effects.
# 2. Load torch before faiss: the reverse order segfaults.
# 3. Keep FAISS single-threaded (below) so only PyTorch runs an OpenMP thread pool.
# This module is the only place that imports faiss.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402, F401, I001
import faiss  # noqa: E402
import numpy as np  # noqa: E402

from app.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

# Exact search over tens of thousands of vectors is still fast on one thread.
faiss.omp_set_num_threads(1)

INDEX_FILENAME = "chunks.faiss"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self._dir = index_dir
        self._lock = threading.RLock()
        self._index: faiss.IndexIDMap2 | None = None
        self.model_name: str | None = None
        self.dimension: int | None = None
        self._load()

    @property
    def index_path(self) -> Path:
        return self._dir / INDEX_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self._dir / MANIFEST_FILENAME

    # --- Loading and saving ------------------------------------------------

    def _load(self) -> None:
        if not (self.index_path.is_file() and self.manifest_path.is_file()):
            return
        try:
            manifest = json.loads(self.manifest_path.read_text())
            index = faiss.read_index(str(self.index_path))
        except Exception:
            logger.warning("Vector index at %s is unreadable; starting empty", self._dir)
            return
        if index.d != manifest.get("dimension"):
            logger.warning("Vector index dimension does not match its manifest; starting empty")
            return
        self._index = index
        self.model_name = manifest.get("model_name")
        self.dimension = int(index.d)
        logger.info("Loaded %d vectors built with %s", index.ntotal, self.model_name)

    def _save(self) -> None:
        if self._index is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        temporary_index = self.index_path.with_suffix(".tmp")
        faiss.write_index(self._index, str(temporary_index))
        os.replace(temporary_index, self.index_path)

        manifest = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "vector_count": int(self._index.ntotal),
            "faiss_version": faiss.__version__,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        temporary_manifest = self.manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2))
        os.replace(temporary_manifest, self.manifest_path)

    # --- Inspection --------------------------------------------------------

    @property
    def vector_count(self) -> int:
        with self._lock:
            return int(self._index.ntotal) if self._index is not None else 0

    def ids(self) -> set[int]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return set()
            return {int(value) for value in faiss.vector_to_array(self._index.id_map)}

    # --- Mutation ----------------------------------------------------------

    def ensure_model(self, model_name: str, dimension: int) -> bool:
        """Make the index ready for vectors from ``model_name``.

        Returns True if existing vectors were discarded because a different
        model (or dimension) produced them.
        """
        with self._lock:
            if (
                self._index is not None
                and self.model_name == model_name
                and self.dimension == dimension
            ):
                return False
            discarded = self.vector_count > 0
            if discarded:
                logger.warning(
                    "Embedding model changed from %s to %s; discarding %d vectors",
                    self.model_name,
                    model_name,
                    self.vector_count,
                )
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
            self.model_name = model_name
            self.dimension = dimension
            self._save()
            return discarded

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        """Insert vectors, replacing any already stored under the same ids."""
        with self._lock:
            if self._index is None or self.dimension is None:
                raise RuntimeError("Call ensure_model() before adding vectors")
            if len(ids) == 0:
                return
            if vectors.shape != (len(ids), self.dimension):
                raise ValueError(
                    f"Expected vectors of shape ({len(ids)}, {self.dimension}), got {vectors.shape}"
                )
            id_array = np.asarray(ids, dtype=np.int64)
            self._index.remove_ids(id_array)  # makes add an idempotent upsert
            self._index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), id_array)
            self._save()

    def remove(self, ids: Collection[int]) -> int:
        """Delete vectors by chunk id; unknown ids are ignored. Returns how many were removed."""
        with self._lock:
            if self._index is None or not ids:
                return 0
            removed = int(self._index.remove_ids(np.asarray(list(ids), dtype=np.int64)))
            if removed:
                self._save()
            return removed

    # --- Search ------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int, allowed_ids: Collection[int] | None = None
    ) -> list[SearchHit]:
        """Top-``k`` chunks by cosine similarity.

        ``allowed_ids`` restricts the search inside FAISS itself, so chunks from
        papers outside the user's selection are never candidates at all.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0 or k <= 0:
                return []
            params = None
            if allowed_ids is not None:
                if not allowed_ids:
                    return []
                selector = faiss.IDSelectorBatch(np.asarray(sorted(allowed_ids), dtype=np.int64))
                params = faiss.SearchParameters(sel=selector)
            query_matrix = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
            scores, ids = self._index.search(
                query_matrix, min(k, int(self._index.ntotal)), params=params
            )
        return [
            SearchHit(chunk_id=int(chunk_id), score=float(score))
            for chunk_id, score in zip(ids[0], scores[0], strict=True)
            if chunk_id != -1  # FAISS pads with -1 when fewer results match
        ]
