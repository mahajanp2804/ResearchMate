# ResearchMate

AI-powered research paper analysis. Upload academic papers, ask questions across
them, compare methods and results, and surface research gaps. Every answer comes
from your papers and cites the document, page and passage it relies on.

> **Status:** under active development. Phases 1–4 (setup, PDF ingestion, chunking,
> embeddings and the FAISS index) are complete. See [docs/ROADMAP.md](docs/ROADMAP.md) for the architecture and the
> phase plan. Full documentation will be added in Phase 13.

## Planned pipeline

```
PDF → extraction → cleaning → chunking → embeddings → FAISS + SQLite
question → dense (FAISS) + keyword (BM25) → RRF fusion → cross-encoder rerank
        → context selection → grounded LLM answer → citation validation
```

## Tech stack

- **Backend:** Python 3.12, FastAPI, Pydantic, SQLAlchemy, SQLite
- **Retrieval:** Sentence Transformers, FAISS (CPU), rank-bm25, cross-encoder reranking
- **LLM:** Gemini through a provider interface
- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Lucide, Recharts

## Running locally

### Backend

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env          # then add GEMINI_API_KEY
.venv/bin/uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

The first paper you upload downloads the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, about 90 MB) from Hugging Face; it is cached
for later runs. Uploads respond immediately and papers become searchable once the
background indexing finishes.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173. In development, Vite proxies `/api` requests to the backend.

### Tests and checks

```bash
cd backend && .venv/bin/pytest && .venv/bin/ruff check .
cd frontend && npm run build && npm run lint
```

Tests marked `models` download the embedding model's tokenizer from Hugging Face on
first run (it is cached afterwards). To run offline, use `.venv/bin/pytest -m "not models"`.
