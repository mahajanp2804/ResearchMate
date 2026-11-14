# ResearchMate

AI-powered research paper analysis. Upload academic papers, ask questions across
them, compare methods and results, and surface research gaps. Every answer comes
from your papers and cites the document, page and passage it relies on.

> **Status:** under active development. Phases 1–12 (setup, PDF ingestion, chunking,
> embeddings, the FAISS index, dense + BM25 retrieval, hybrid fusion, reranking, context
> selection, grounded question answering with Gemini, citation validation with
> page-level excerpts, cited paper summaries and multi-paper comparison, evidence-based
> research gap analysis, the RAG evaluation dashboard, and the connected React frontend) are complete. See [docs/ROADMAP.md](docs/ROADMAP.md) for the architecture and the
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

### Gemini API key

Answers are generated with Google Gemini. Get a free key at
https://aistudio.google.com/apikey and put it in `backend/.env`:

```bash
GEMINI_API_KEY=your-key-here
```

Restart the backend afterwards. Without a key, everything except answer generation
works, and questions your papers cannot answer still get the refusal message.

### Tests and checks

```bash
cd backend && .venv/bin/pytest && .venv/bin/ruff check .
cd frontend && npm run build && npm run lint
```

Tests marked `models` download Hugging Face models on first run (cached afterwards); to
run offline, use `.venv/bin/pytest -m "not models"`. Tests marked `llm` call the real
Gemini API and are skipped unless `GEMINI_API_KEY` is set: `.venv/bin/pytest -m llm`.
