# PageSense

Natural-language book recommendations over the **CMU Book Summaries** corpus.

Ask for *"sad but hopeful sci-fi about friendship"*, filter by genre and year, hit
**More like this**, or blend a few favorites into a taste profile — all backed by
vector search over Weaviate, with an evaluation harness to prove it actually works.

> **Status:** early scaffolding. Phase 0 (data prep) has not landed yet — see
> the [roadmap](#roadmap). Setup steps below are the target once Phase 6 lands.

## Stack

| Layer          | Choice                                             |
| -------------- | -------------------------------------------------- |
| Data           | CMU Book Summaries (16,559 books)                  |
| Embeddings     | `all-MiniLM-L6-v2` via `sentence-transformers`     |
| Vector store   | Weaviate (self-provided vectors)                   |
| Backend        | FastAPI                                            |
| Frontend       | Next.js                                            |
| RAG            | local Ollama model, cited answers                  |
| Evaluation     | recall@10, MRR                                     |

## Architecture

```
CMU corpus ──▶ data prep ──▶ books.parquet
                                   │
                                   ▼
                         all-MiniLM-L6-v2 embeddings
                                   │
                                   ▼
                            Weaviate (vectors + metadata)
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                              ▼
             retrieval + filters          RAG (summaries / full text)
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                          FastAPI ──▶ Next.js UI
```

Retrieval is written behind a `VectorStore` interface so the engine can start as
a numpy brute-force search in a notebook and later move to Weaviate unchanged.

## Dataset

`data/raw/cmu_books/booksummaries.txt` — the **CMU Book Summaries** dataset
(Bamman & Smith, 2013). Tab-separated, one book per row:

| # | Field            | Notes                                   |
| - | ---------------- | --------------------------------------- |
| 1 | Wikipedia ID     | unique per row                          |
| 2 | Freebase ID      |                                         |
| 3 | Title            |                                         |
| 4 | Author           |                                         |
| 5 | Publication date |                                         |
| 6 | Genres           | JSON map of Freebase genre IDs → labels |
| 7 | Plot summary     |                                         |

**License:** Creative Commons **Attribution-ShareAlike 3.0**. Summaries originate
from Wikipedia; metadata is aligned from Freebase. Attribution is required and
derived data must be shared under a compatible license. See
[Attribution](#attribution).

## Repo layout

```
.
├── backend/          # Python: data prep, embeddings, retrieval, FastAPI API
├── frontend/         # Next.js UI
├── data/
│   ├── raw/          # CMU Book Summaries (source, git-ignored)
│   └── processed/    # books.parquet, chunks (generated, git-ignored)
├── eval/             # labeled queries + recall@10 / MRR harness
├── docker-compose.yml
├── backlog.md        # actionables: build order, UI, parked ideas
└── README.md
```

## Roadmap

Work is tracked as actionables in [backlog.md](./backlog.md): the must-have build
order (Phases 0–9) and the core UI features under **Now**, with everything else
parked under **Later**. Retrieval logic is handcoded; boilerplate glue may be
AI-assisted.

## Evaluation

The harness in `eval/` scores retrieval with **recall@10** and **MRR** against
labeled queries, and is reused to compare vector vs. keyword vs. hybrid search and
to catch regressions.

_Results table lands in Phase 9._

## Getting started

Target workflow once the backend and UI phases land:

```bash
# Weaviate
docker compose up -d

# Backend (data prep + API)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
pnpm install
pnpm dev
```

- Frontend: http://localhost:3000 · Backend: http://localhost:8000 · Weaviate: http://localhost:8080

## Attribution

> Contains data from the CMU Book Summary Dataset (Bamman & Smith, 2013),
> https://www.cs.cmu.edu/~dbamman/booksummaries.html, licensed under
> CC BY-SA 3.0. Plot summaries are derived from Wikipedia; metadata is aligned
> from Freebase.
