# ResearchMate — Architecture & Implementation Roadmap

This document is the working plan for ResearchMate. It records the architecture, the
directory layout, the dependencies, and the phase-by-phase build order with the
acceptance criteria each phase must meet before the next one starts.

---

## 1. Architecture

```
┌──────────────────────────── React + TypeScript (Vite) ────────────────────────────┐
│  Dashboard · Papers · Assistant · Comparison · Insights · Evaluation · Settings  │
│                       services/api.ts  (typed REST client)                        │
└───────────────────────────────────────┬───────────────────────────────────────────┘
                                        │  JSON over HTTP (/api/*)
┌───────────────────────────────────────▼───────────────────────────────────────────┐
│ FastAPI   routers/  →  thin HTTP layer: validation, status codes, error mapping   │
│           schemas/  →  Pydantic request/response contracts                        │
├───────────────────────────────────────────────────────────────────────────────────┤
│ services/ (business logic, no HTTP knowledge)                                     │
│                                                                                   │
│  Ingestion:  file_validation → pdf_service → text_cleaning → chunking_service     │
│              → embedding_service → vector_store (FAISS) + SQLite metadata         │
│                                                                                   │
│  Retrieval:  query processing → dense (FAISS) ┐                                   │
│                                BM25 ──────────┴→ RRF fusion → cross-encoder       │
│              rerank → context selection (dedupe + token budget + evidence gate)  │
│                                                                                   │
│  Generation: prompts/ → llm_service (provider interface; Gemini first)            │
│              → citation_service (validate [S#] markers against given sources)     │
│                                                                                   │
│  Analysis:   paper profiles (summary + comparison views) · research_gap           │
│  Evaluation: dataset → per-config retrieval runs → P@K / R@K / MRR                │
│              + optional LLM-judged faithfulness / relevance (kept separate)       │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Persistence: SQLite via SQLAlchemy (documents, chunks, conversations, messages,  │
│              evaluation datasets/runs, activity)  ·  FAISS index on disk         │
│              ·  uploaded PDFs on disk                                             │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| FAISS ↔ SQLite mapping | `IndexIDMap2(IndexFlatIP)` where **FAISS id = `chunks.id`** (SQLite integer PK) | No separate mapping table to drift out of sync; deleting a document removes exactly its chunk ids with `remove_ids`. |
| Similarity | Inner product on L2-normalised vectors | Equivalent to cosine similarity, exact search, fast enough for tens of thousands of chunks on a laptop. |
| Document filtering | Applied **inside** each retriever (FAISS `IDSelector`, BM25 score mask) | Out-of-scope chunks can never reach fusion, reranking or the LLM prompt. |
| Fusion | Reciprocal Rank Fusion, `k = 60` | Rank-based, so dense cosine scores and BM25 scores never need to be put on the same scale. |
| Grounding gate *(revised in Phase 6)* | Refuse **without calling the LLM** whenever retrieval yields no context. A reranker-score threshold (`MIN_RERANK_SCORE`) exists but is **off by default**; otherwise the strict grounding prompt (Phase 7) and citation validation (Phase 8) decide | Measured with `ms-marco-MiniLM-L-6-v2`: answerable paraphrased research questions scored as low as −11.1 and unanswerable ones as high as −9.9, so no threshold separates them, although the right passage ranked 1st in 9/12 and 2nd in 3/12. Phase 11 measures refusal accuracy and can re-enable the gate with a better-calibrated reranker. |
| Reranking value | Reranking is on (spec requirement); Phase 11 measured its benefit | On the sample collection it gives the best ranking (MRR 0.786 vs 0.699 for hybrid without it), mostly on paraphrased and two-paper questions, at about 160 ms per query on a laptop CPU. See *Phase 11 results* below. |
| Context de-duplication | Drop a passage whose words overlap a kept one by Jaccard ≥ 0.8 or are ≥ 90% contained in it | Repeated sections and near-identical uploads waste context slots; neighbouring chunks share only their ~20% overlap. |
| Citations | Sources numbered `[S1]…[Sn]` in the prompt; markers in the answer are parsed and validated against the sources actually sent, and markers for sources that weren't sent are removed before saving | The LLM never writes page numbers or file names itself, so it cannot invent them. |
| Citation excerpts *(Phase 8)* | Code, not the model, picks the excerpt: the sentence of the cited passage that shares the most terms with the statements citing it, located on its exact page when a passage spans two pages | A model-written quote can be paraphrased or invented; a sentence copied from the stored passage cannot. The support score is lexical overlap, not entailment; Phase 11 measures faithfulness properly. |
| Prompt injection | Document text is wrapped in delimited, labelled source blocks and the system prompt states it is untrusted data | Instructions inside a PDF are treated as content to quote, not orders to follow. |
| LLM layer | `LLMProvider` abstract base class; `GeminiProvider` first; provider chosen by config | A local model (e.g. Ollama) or another API is a new class, not a pipeline rewrite. |
| Gemini model | `gemini-3.6-flash` (configurable via `GEMINI_MODEL`) | The current stable Flash model per the Gemini API docs (Nov 2026); `gemini-2.5-flash` is deprecated. The output limit (8,192) leaves room for Gemini 3 "thinking" tokens. |
| Follow-ups | Rewrite the follow-up into a standalone question with the LLM; send **no history** to the answer step | Retrieval needs one self-contained query, and keeping earlier answers out of the answer prompt stops unverified claims (and stale [S#] ids) from leaking into new answers. |
| Saving answers | Store each answer with the full text of its sources | An answer stays traceable after its paper is deleted or re-chunked. |
| Re-embedding | Re-index compares the chunking signature, embedding model and stored vectors and returns `unchanged`, `reembedding` (chunks kept) or `rechunked` | Re-indexing an unchanged paper is free; a model change only re-embeds. |
| Persistence | SQLAlchemy 2.0 ORM, portable column types, string enums (no native DB enums) | Swapping `DATABASE_URL` to PostgreSQL needs no model changes. |
| Background work | FastAPI background tasks + status columns polled by the UI | Uploads return immediately; no Celery/Redis needed for a single-user local app. |
| Startup consistency | On start, interrupted work is reset, papers whose vectors are missing or from another model are marked not indexed, and orphaned vectors are removed | Statuses always describe what is really in the index, even after a crash. |
| PyTorch + FAISS on macOS | Load torch before faiss, allow the second OpenMP runtime (`KMP_DUPLICATE_LIB_OK`), and set FAISS to one thread **in every calling thread** | Both macOS wheels bundle their own OpenMP; any other combination segfaults or deadlocks. `omp_set_num_threads` is per-thread and FastAPI uses worker threads, so it is applied before every FAISS call. A subprocess test runs embedding and search on worker threads and checks results against NumPy. |
| BM25 IDF | Lucene's `log(1 + (N − n + 0.5)/(n + 0.5))` instead of Okapi's | Okapi IDF is zero or negative for terms in half the chunks, which in a small library drops relevant passages. |
| BM25 cache | In-memory index rebuilt only when `(count, sum, max)` of searchable chunk ids changes | Stateless invalidation with no hooks on every write; reliable because chunk ids are never reused. |
| What is searchable | Only papers with `indexing_status = indexed`, for both retrievers | "Searchable" in the UI means exactly what search uses; a half-indexed paper never appears in one method but not the other. |
| Paper profiles *(Phase 9)* | Summaries and comparisons are two views of one saved **profile** per paper: 11 aspects (the 9 summary sections and 10 comparison rows overlap), each with its own retrieval question searched within that paper, then **one** LLM call that fills every aspect as JSON or marks it not found | One model call per paper instead of one per section or cell; a paper's summary and its comparison column can never contradict each other; comparing papers that were already summarised costs nothing; and every comparison cell cites its own paper by construction, because each call only sees that paper's passages. |
| Profile evidence | Every aspect's best passage is kept before any aspect's second (3 per aspect, 8,000-token budget), then the passages are put back in reading order | One aspect with many matching passages (e.g. results) cannot crowd out a rare one (e.g. limitations). A long paper's details beyond these passages are reported as not found, which the UI words as "not found in the passages retrieved". |
| Structured output | The provider takes an optional JSON Schema (Gemini `response_json_schema`); the reply is still parsed and validated in code | Constrained output makes the shape reliable, but a missing or malformed aspect is treated as not found and an unreadable reply is a clear `llm_bad_output` error, never a partly invented profile. |
| Profile cache | One row per paper, keyed by a signature of the profile prompt version, chunking signature and embedding model | Re-processing a paper or changing the prompt makes old profiles stale automatically; `refresh` rebuilds on request. |
| Comparisons | Retrieval for each paper runs in the request thread; only the LLM calls run in parallel (up to 4) | The database session and FAISS thread settings stay in one thread. If one paper fails, the others' profiles are still saved, so a retry only redoes the failed one. |
| Questions about selected papers *(Phase 9)* | When a question is limited to 2+ selected papers, each paper is searched separately and gets `MIN_PASSAGES_PER_PAPER` (2) guaranteed context slots; the rest go to the best passages overall | Measured in a test: without it, a paper whose wording matches the question well took every slot and the other paper was never seen, so "compare the results of these papers" could not be answered. The evidence threshold still applies to guaranteed passages. |
| Gap analysis *(Phase 10)* | One LLM call over every selected paper's gap-related passages returns three separate kinds of finding: limitations a paper **states** about itself, patterns **observed** across the papers (recurring challenges, differences, limited discussion), and **inferred** research directions, whose cited `evidence` is kept apart from the uncited `suggestion` | The specification asks for these to be distinguished; putting the inference in its own field means it can never be shown as something a paper said. |
| Gap evidence | Each paper is searched separately with six gap questions (limitations, assumptions, future work, problem, evaluation scope, failure cases), 2 passages each, at most 3,000 tokens per paper | Every selected paper is represented equally, and one call sees everything needed to spot patterns across them. |
| Checks on findings | In code: a stated limitation may only cite its own paper (other citations are removed); a recurring challenge or a difference must cite at least two papers; every finding needs a valid citation. Failing findings are left out and listed with the reason | Overstating a pattern ("both papers…" resting on one) or attributing a limitation to the wrong paper is how a model most easily misleads here; checking it in code makes the claim verifiable, and listing what was left out keeps the filter transparent. |
| Scope of gap claims | A fixed scope note written by the code ("based only on the N selected papers…"), the prompt forbids novelty claims, and wording such as "unexplored", "no prior work" or "gap in the literature" is flagged on the finding | A handful of uploaded papers cannot show that a gap is new; the flag catches the model saying so anyway. |
| Saved gap analyses | Keyed by the set of papers (in any order) and their chunking signatures and embedding model; deleting a paper deletes the analyses that include it | A reload or a repeat request costs no model call. Unlike chat answers, which keep their sources as a record, a cache should not keep a deleted paper's text. |
| Evaluation collection *(Phase 11)* | Five short **synthetic** papers (original text, fictional methods and results) generated as PDFs on demand, and a hand-written dataset of 35 questions in five categories (factual, paraphrase, exact term, two papers, unanswerable) with exact evidence sentences and reference answers | Reproducible on any machine with no copyrighted PDFs in the repo, and the categories show *where* each retriever wins. Papers are referred to by title, so a dataset can also target a user's own papers. |
| Relevance judgement | A retrieved passage is relevant if it belongs to the evidence's paper and contains the evidence text (compared on letters and digits only) | Chunk ids change whenever chunking settings change; evidence text does not. A test checks every evidence sentence lands whole in a chunk, which the sentence-aware chunker guarantees. |
| Answer metrics | Optional and LLM-judged: the pipeline answers each question, the answer is split into statements in code, and one judge call rates faithfulness (per statement), context relevance (per passage), answer relevance and correctness (1-5). Refusals are checked in code against the question's category | Stored and shown apart from the retrieval metrics, which involve no model. The judge is usually the model that answered, so its scores are labelled indicative; the run records which models were used. |
| Schema changes | At startup, nullable columns that a newer version added are added to existing tables; anything else is refused with a clear error | `create_all` never alters existing tables. This keeps upgrades working for a local SQLite app; Alembic migrations come with PostgreSQL support. |

### Decision (Phase 3): chunk size vs. embedding window

`all-MiniLM-L6-v2` truncates input at **256 word-piece tokens**. With the originally
requested 800-token chunks, the dense vector would only represent roughly the first
third of each chunk, while BM25 and the LLM see the whole chunk. Options considered:

1. Keep 800/150 and accept truncation (dense retrieval quality drops).
2. **Chosen default:** 250-token chunks with 50 overlap, measured with MiniLM's own
   tokenizer, so every chunk fits the model's window (250 + [CLS] + [SEP] ≤ 256).
3. Keep 800-token chunks but switch to a 512-token model such as `BAAI/bge-small-en-v1.5`.

Chunk size, overlap and model are all configurable (`CHUNK_SIZE_TOKENS=800` restores
option 1). The Phase 11 evaluation will compare these settings on the same dataset,
and from Phase 4 the app logs a warning whenever `CHUNK_SIZE_TOKENS` exceeds the
embedding model's maximum sequence length. Phase 6 can widen the context sent to the
LLM by adding neighbouring chunks around each retrieved one.

---

## 2. Directory structure

```
ResearchMate/
├── backend/
│   ├── app/
│   │   ├── main.py               # app factory, CORS, exception handlers, routers
│   │   ├── config.py             # pydantic-settings; all tunables live here
│   │   ├── database.py           # engine, session factory, Base, get_db
│   │   ├── exceptions.py         # domain errors mapped to HTTP responses
│   │   ├── models/               # SQLAlchemy ORM
│   │   │   ├── document.py  page.py  chunk.py  conversation.py  profile.py  gap_analysis.py
│   │   │   ├── evaluation.py  activity.py
│   │   ├── schemas/              # Pydantic API contracts
│   │   │   ├── common.py  system.py  document.py  retrieval.py  chat.py  research.py  evaluation.py
│   │   ├── routers/
│   │   │   ├── system.py  documents.py  retrieval.py  chat.py  research.py  evaluation.py
│   │   ├── services/
│   │   │   ├── document_service.py    # upload, de-duplication, listing, deletion
│   │   │   ├── ingestion_service.py   # chunk → embed → index for stored documents
│   │   │   ├── pdf_service.py  text_cleaning.py  chunking_service.py  tokenization.py
│   │   │   ├── embedding_service.py  vector_store.py  bm25_service.py  retrieval_service.py
│   │   │   ├── hybrid_retriever.py  reranker.py  context_selection.py  rag_pipeline.py
│   │   │   ├── llm_service.py  citation_service.py
│   │   │   ├── paper_profile_service.py  # per-paper cited aspects behind summaries + comparisons
│   │   │   ├── research_service.py       # summaries and comparisons on top of profiles
│   │   │   ├── research_gap_service.py   # stated / observed / inferred findings, checked in code
│   │   │   ├── evaluation_service.py  evaluation_data.py  retrieval_metrics.py  answer_evaluation.py
│   │   │   └── activity_service.py  stats_service.py  chat_service.py
│   │   ├── prompts/
│   │   │   ├── answer_prompt.py  rewrite_prompt.py  profile_prompt.py  research_gap_prompt.py  judge_prompt.py
│   │   └── utils/
│   │       ├── logging.py  file_validation.py
│   ├── evaluation/               # sample_papers.json (synthetic collection), datasets/*.json
│   ├── scripts/                  # run_evaluation.py: command-line evaluation runner
│   ├── tests/                    # pytest suite + fixture PDFs
│   ├── requirements.txt  requirements-dev.txt  pyproject.toml
│   ├── .env.example  Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/  layout/ (Sidebar, Header)  ui/ (Card, Button, Badge, …)
│   │   │                DocumentCard, UploadZone, ChatMessage, CitationCard,
│   │   │                ComparisonTable, MetricCard
│   │   ├── pages/       Dashboard, Documents, Assistant, Comparison, Insights,
│   │   │                Evaluation, Settings
│   │   ├── services/api.ts   hooks/   types/   context/   utils/
│   │   ├── App.tsx  main.tsx  index.css
│   ├── package.json  vite.config.ts  tsconfig*.json  Dockerfile  nginx.conf
├── docs/ROADMAP.md
├── docker-compose.yml
├── .gitignore
└── README.md
```

Changes from the suggested layout, and why:

- `models/evaluation.py`, `models/activity.py` — evaluation datasets/runs and the dashboard's activity feed need their own tables.
- `models/page.py` — raw text per page is stored once, so chunking settings can change and papers can be re-chunked without re-parsing PDFs.
- `services/document_service.py` — upload validation, duplicate detection, extraction and deletion, keeping the documents router thin.
- `services/ingestion_service.py` — one place that runs chunking → embedding → indexing for stored documents and updates their status.
- `routers/system.py` — health check, non-secret configuration and dashboard stats.
- `exceptions.py` — a small domain-error hierarchy so services raise meaningful errors and `main.py` maps them to HTTP codes in one place.

---

## 3. Dependencies

**Backend (runtime)**

| Package | Purpose | Added in |
|---|---|---|
| fastapi, uvicorn[standard] | API server | 1 |
| pydantic, pydantic-settings (uses python-dotenv) | validation, typed config from `.env` | 1 |
| sqlalchemy | ORM over SQLite (PostgreSQL-ready) | 1 |
| python-multipart | multipart file uploads | 2 |
| pymupdf | PDF text + metadata extraction | 2 |
| sentence-transformers (brings torch, transformers) | embeddings, cross-encoder, tokenizer | 3–4 |
| faiss-cpu, numpy | vector index | 4 |
| rank-bm25 | keyword retrieval | 5 |
| google-genai | Gemini provider | 7 |

**Backend (dev):** pytest, httpx, ruff.

**Frontend:** react, react-dom, react-router-dom, lucide-react, recharts (Phase 11);
dev: vite, typescript, @vitejs/plugin-react, tailwindcss + @tailwindcss/vite, eslint.

Dependencies are added in the phase that first uses them, so every phase installs
and runs on its own. Versions are pinned in `requirements.txt` / `package-lock.json`.

---

## 4. Implementation roadmap

Each phase ends with: what was built, files touched, how it works, tests run, and
commands to verify it. A phase is not "done" until its tests pass.

### Phase 1 — Project setup & architecture
- Backend skeleton: app factory, typed config, SQLAlchemy engine/session, all ORM models, consistent JSON error format, logging.
- Endpoints: `GET /api/health`, `GET /api/system/info` (non-secret config only), `GET /api/dashboard/stats` (real counts from SQLite).
- Frontend skeleton: Vite + React + TS + Tailwind v4, design tokens with light/dark themes, sidebar layout, routing for all 7 pages, typed API client, Dashboard and Settings wired to the real backend, honest empty states elsewhere.
- **Done when:** pytest passes (health, config validation, secret redaction, cascade deletes, stats), `npm run build` passes, the UI shows live backend status.

### Phase 2 — PDF ingestion & extraction
- `POST /api/documents/upload` (multi-file), `GET /api/documents`, `GET /api/documents/{id}`, `DELETE /api/documents/{id}`, `GET /api/documents/{id}/file`.
- Validation: extension, `%PDF-` magic bytes, size limit, encrypted/malformed PDF handling, SHA-256 duplicate detection.
- PyMuPDF page-by-page extraction, PDF metadata, title detection (metadata → largest-font text on page 1 → filename), scanned-PDF detection with an "OCR not supported" message.
- **Done when:** tests cover valid, duplicate, oversize, non-PDF, corrupted, encrypted and image-only PDFs.

### Phase 3 — Text cleaning, chunking & metadata
- Normalisation: Unicode NFKC, de-hyphenation across line breaks, header/footer and page-number removal, whitespace collapse.
- Token-aware, sentence-boundary chunker (embedding model's tokenizer), configurable size/overlap; each chunk keeps `document_id`, title, `page_start`/`page_end`, `chunk_index`, text, token and character counts.
- **Done when:** tests prove overlap, size limits, and that every chunk's page numbers match where its text actually came from.

### Phase 4 — Embeddings & FAISS
- Batched, normalised Sentence-Transformers embeddings (model cached once per process).
- Persistent FAISS `IndexIDMap2`, incremental add, `remove_ids` on delete, index manifest (model name + dimension) to detect model changes.
- `POST /api/documents/{id}/reindex`; skip re-embedding when content and config are unchanged.
- Embedding runs as a background task after upload/re-index; the UI polls `indexing_status`.
- **Done when:** tests cover add/search/persist/reload/delete and the skip-if-unchanged path.
- **Status: done.** Docker (Phase 13) must install the CPU-only PyTorch wheel; the default Linux wheel pulls in CUDA.

### Phase 5 — Dense retrieval & BM25
- Dense top-K with document filtering via FAISS ID selectors.
- BM25 (rank-bm25) built from SQLite chunks, cached and invalidated when the corpus changes, same filters.
- Debug endpoint `POST /api/retrieval/search` returning ranked chunks per method.
- **Done when:** tests show keyword-only and paraphrase-only queries each retrieve the right chunk, and filters exclude other documents.
- **Status: done.** With the real model, a paraphrase sharing no keywords is found only by dense retrieval (BM25 returns nothing) and a rare identifier is matched exactly by BM25 (`tests/test_retrieval_models.py`). The RAG Evaluation page has a retrieval playground showing both side by side.

### Phase 6 — Hybrid fusion & reranking
- RRF fusion, cross-encoder rerank of the top-N fused candidates, context selection (overlap de-duplication, token budget, minimum-evidence threshold).
- **Done when:** tests verify fusion arithmetic, rerank ordering and filter preservation end-to-end.
- **Status: done.** `POST /api/retrieval/search` compares all four stages with per-stage source ranks; `POST /api/retrieval/context` shows exactly what the LLM would receive. The playground shows both. The score-based evidence gate was measured and turned off by default (see *Grounding gate* above).

### Phase 7 — LLM integration & grounded QA
- Provider interface + Gemini provider; clear errors for missing key, quota and timeouts.
- Strict grounding prompt with delimited untrusted sources; refusal sentence enforced in code when evidence is insufficient.
- `POST /api/chat/query`, conversations and history, follow-up questions rewritten into standalone queries.
- **Done when:** tests (with a fake provider) prove the no-context path returns the refusal without calling the LLM.
- **Status: done.** Refusals without an LLM call cover an empty or unsearchable library and the optional evidence gate (so they work with no API key); a model refusal is normalised to the exact sentence. Opt-in live tests (`pytest -m llm`) check grounding, refusal and resistance to instructions planted in a paper once `GEMINI_API_KEY` is set. The Research Assistant page has saved conversations, paper filters, citation chips and sources.

### Phase 8 — Citation generation & validation
- Parse `[S#]` markers, discard markers that don't match a sent source, attach document title, page and excerpt; flag uncited answers.
- **Done when:** tests cover valid, invented and missing citations.
- **Status: done.** Markers are parsed in any common form (`[S1]`, `[S1, S3]`, `[S1-S3]`); invented ids are removed from the saved answer and reported; a refusal with a stray marker is still a refusal. Each cited source gets an excerpt, its exact page and a support score. Statements of five or more words without a citation are listed, and each answer is marked `cited`, `partially_cited` or `uncited`. The Research Assistant lists cited sources first with their excerpts and warns about uncited statements and removed citations. Limitation: a source cited by several statements shows one excerpt.

### Phase 9 — Summarisation & multi-paper comparison
- Nine-section structured summary per paper, each section from targeted retrieval with citations or an explicit "not found"; cached per document.
- `POST /api/research/compare`: aspect × paper matrix with per-cell citations, "not reported" instead of guesses, follow-up questions scoped to the compared papers.
- **Status: done.** `POST /api/research/summarize/{id}` (and `GET /api/research/summaries/{id}` for the saved one) returns nine cited sections; `POST /api/research/compare` returns ten aspects for 2–5 papers. Both come from saved paper profiles (see *Paper profiles* above), are validated per aspect like answers (invented citations removed, uncited claims flagged, excerpt and exact page per citation) and say "not found" / "Not reported" instead of guessing. Follow-up questions use the normal grounded pipeline limited to the compared papers, which now gives each selected paper a share of the context. The UI has a summary page per paper and a Paper Comparison page with a table, cited passages and follow-up questions. Verified offline with a scripted model and in the browser with real retrieval and a stand-in model; the opt-in live test (`pytest -m llm`) covers Gemini's JSON output once a key is set.

### Phase 10 — Research gap analysis
- `POST /api/research/gaps`: explicit limitations → cross-paper patterns → inferred opportunities, each with citations and a scope disclaimer.
- **Status: done.** `POST /api/research/gaps` analyses 2–5 papers and returns stated limitations (grouped by paper), recurring challenges, differences, topics with limited discussion and possible research directions, every one cited with an excerpt and exact page, plus the findings that failed the checks (see *Checks on findings* above) and a scope note. The Research Insights page shows the three kinds under separate labels (stated / observed / inferred), with the inferred suggestion visually apart from its evidence. Verified offline with a scripted model (including a misattributed citation, a one-paper "pattern", an uncited opportunity and an over-claiming suggestion, all caught) and in the browser with real retrieval and a stand-in model; the opt-in live test (`pytest -m llm`) runs it against Gemini once a key is set.

### Phase 11 — Evaluation
- Curated dataset (questions, expected documents/pages/evidence, reference answers) over a reproducible sample collection.
- Precision@K, Recall@K, MRR for dense / BM25 / hybrid / hybrid+rerank; optional LLM-judged faithfulness, answer relevance and context relevance shown separately.
- `POST /api/evaluation/run`, `GET /api/evaluation/results`; runs stored in SQLite.
- **Status: done.** `GET /api/evaluation/datasets`, `POST /api/evaluation/sample-papers`, `POST /api/evaluation/run` (runs in the background, with progress), `GET /api/evaluation/results[/{id}]` and `DELETE`. The RAG Evaluation page installs the sample papers, runs experiments, and charts P@K, R@K, hit rate and MRR per method and per question type (Recharts), with a table of every number, per-question ranks, LLM-judged answer scores in a separate section, and the run history. `python -m scripts.run_evaluation` runs the same evaluation from the command line in a temporary data directory.

#### Phase 11 results (retrieval, measured)

Sample collection: 5 papers, 35 chunks; 31 answerable questions. Measured 13 Nov 2026 on an Apple Silicon laptop (CPU) with `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`, 250-token chunks with 50 overlap and RRF k = 60. Two runs in the app gave identical numbers, and the Docker image (Linux, ARM64) reproduced every quality metric exactly. Latency is measured after one untimed warm-up search per method, so it excludes loading or downloading the models (an earlier run that counted the first load reported 269 ms for reranking). Inside Docker on the same laptop, reranking took about 297 ms per query (the Docker VM is slower than native).

| Method | P@1 | P@5 | R@1 | R@3 | R@5 | R@10 | MRR | ms / query |
|---|---|---|---|---|---|---|---|---|
| Dense | 0.452 | 0.194 | 0.452 | 0.677 | 0.823 | 0.903 | 0.617 | 6.7 |
| BM25 | 0.581 | 0.206 | 0.548 | 0.774 | 0.871 | 0.919 | 0.713 | 0.5 |
| Hybrid (RRF) | 0.548 | 0.200 | 0.532 | 0.742 | 0.871 | **0.968** | 0.699 | 7.0 |
| Hybrid + rerank | **0.677** | **0.213** | **0.629** | **0.806** | **0.903** | 0.952 | **0.786** | 158.8 |

MRR by question type (number of questions in brackets): factual (12) 0.688 / 0.762 / 0.786 / 0.777; exact term (6) 0.875 / 1.000 / 1.000 / 1.000; two papers (3) 0.344 / 0.556 / 0.400 / 0.733; paraphrase (10) 0.458 / 0.528 / 0.503 / 0.683 (dense / BM25 / hybrid / hybrid + rerank).

What this shows, and what it doesn't:
- Reranking gives the best ranking overall and helps most where wording differs (paraphrases +0.18 MRR over hybrid) and on two-paper questions, at ~160 ms per query on CPU.
- Fusion finds the most evidence within 10 passages (R@10 0.968), but its MRR is slightly below BM25 here: a weaker dense ranking can push BM25's top hit down.
- BM25 beats dense even on the paraphrase questions. Five short papers with distinctive vocabulary favour lexical matching; this is not evidence that dense retrieval is weak in general.
- The collection is tiny (35 chunks, so top-10 covers almost a third of it): R@10 is inflated, and with 3-12 questions per category, differences of a few hundredths are noise. MRR, P@1 and R@1-3 are the informative columns.
- P@5 is capped near 0.2 because most questions have one relevant passage.
- Answer quality was implemented and tested with a scripted model but **not measured with Gemini** (no API key during development). Run it with "Judge answers" in the app or `--answers` on the command line.

### Phase 12 — Frontend integration
- Papers (upload with progress, status polling, delete, reindex), Assistant (scoped chat, citations, source viewer), Comparison table, Insights, Evaluation charts (Recharts), Settings.
- **Status: done.** Each screen was built with its phase; this phase connected them and checked the whole app against the specification:
  - Features hand papers to each other through a `?papers=` link: *Ask* on a paper card, its summary and the dashboard opens the assistant scoped to that paper, and a comparison and a gap analysis of the same papers link to each other.
  - The dashboard's recent papers open their summaries, every activity links to where it happened, and the quick actions cover every feature.
  - Paper details show the document ID and the processing and indexing statuses separately, as the specification lists them.
  - A page that fails to render shows a recoverable error inside the layout instead of a blank app; a skip link and named checkboxes help keyboard and screen-reader users; the evaluation page (and its charting library) loads only when visited, keeping the main bundle at about 425 kB.
  - Every page was checked at 375 px for content running off-screen (one real bug fixed: dashboard cards were pushed off-screen by long titles), in light and dark mode, and with the keyboard.

### Phase 13 — Testing, Docker & documentation
- Coverage pass, Dockerfiles + docker-compose (persistent volume for data and model cache), full README with architecture, RAG workflow, evaluation method, API reference and "How to Explain This Project in an Interview".
- **Status: done.**
  - *Tests*: 506 backend tests (6 more run against live Gemini when a key is set) cover 97% of statements and branches (`pytest --cov`); 17 frontend unit tests (Vitest) cover the answer parser, the cross-feature links and formatting. New failure-path tests found a real bug: two uploads of the same file at once returned a server error instead of "duplicate", because the duplicate surfaced at a flush outside the guarded commit; fixed.
  - *Docker*: a backend image (Python 3.12 slim, CPU-only PyTorch, non-root user, health check; 2.37 GB, of which PyTorch is 0.7 GB) and a frontend image (nginx serving the build and proxying `/api` and `/docs`; 93 MB), run by `docker compose up --build` with volumes for data and models. Verified on Apple Silicon (linux/arm64): health through nginx, client-side routes and API docs served, the sample papers indexed in about 33 s including the first model download, a real PDF uploaded and a fake one rejected through nginx, and an evaluation run that reproduced the native metrics exactly. Not verified: an x86-64 build (the CPU wheels exist for it).
  - *Measurement fix*: the evaluation now runs one untimed search per method before timing, so model loading no longer counts as query latency.
  - *Docs*: the README covers the overview, problem, features, architecture, stack, RAG workflow, setup (Docker and local), environment, API reference, evaluation method and results, the technical reasoning behind chunking, embeddings, FAISS, hybrid retrieval, reranking, citations and evaluation, example questions, testing, limitations and an interview guide, with screenshots of the real app on the sample papers. Screens with generated text are not shown, because only a scripted stand-in model was available during development.
