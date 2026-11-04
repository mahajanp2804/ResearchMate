# ResearchMate

AI-powered research paper analysis. Upload academic papers, ask questions across
them, compare methods and results, and surface research gaps. Every answer comes
from your papers and cites the document, page and passage it relies on.

> **Status:** under active development. Phase 1 (project setup and architecture)
> is complete. See [docs/ROADMAP.md](docs/ROADMAP.md) for the architecture and the
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
