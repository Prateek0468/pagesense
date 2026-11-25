# PageSense

Natural-language book recommendations over the **CMU Book Summaries** corpus.

Ask for *"sad but hopeful sci-fi about friendship"*, filter by genre and year, hit
**More like this**, or blend a few favorites into a taste profile — all backed by
vector search over Weaviate, with an evaluation harness to prove it actually works.


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

`eval/labeled_queries.py` holds 20 hand-labeled queries. Each query lists the
wiki ids that count as relevant. `eval/run_eval.py` embeds the corpus, searches
one `VectorStore`, and reports **recall@k** (fraction of relevant books found in
the top k) and **MRR** (mean reciprocal rank of the first relevant hit).

Run the brute-force baseline from the repo root:

```bash
source .venv/bin/activate
python eval/run_eval.py --store brute \
  --embedding-cache eval/results/embedding_cache.npz
```

The `--embedding-cache` flag is optional but saves ~50 seconds on repeat runs;
delete the cache after regenerating `books.parquet`. The Weaviate row below was
produced with Docker running (`docker compose up -d`); `WeaviateStore` connects
to `localhost:8080` and `localhost:50051`:

```bash
python eval/run_eval.py --store weaviate \
  --embedding-cache eval/results/embedding_cache.npz
```

Each run writes `eval/results/<store>.json` with the aggregate scores and
per-query retrieved ids. The same harness will be reused in Phase 5 to compare
vector, keyword, and hybrid search.

| Store       | recall@10 | MRR    |
| ----------- | --------- | ------ |
| Brute force | 0.7000    | 0.5210 |
| Weaviate    | 0.7000    | 0.5210 |

Both stores score identically on the Phase 2 labels. Their top-10 lists differ
only at rank 10 for `wardrobe-winter-world`, so the named-vector Weaviate path
is effectively equivalent for these queries. The six zero-recall queries point
to corpus/label issues (truncated summaries or alternate editions), not store
divergence; revisit them only if later phases need a stronger baseline.

## Getting started

Target workflow once the backend and UI phases land:

```bash
# Weaviate (local Docker)
docker compose up -d

# Backend (data prep + API)
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
pnpm install
pnpm dev
```

- Frontend: http://localhost:3000 · Backend: http://localhost:8000 · Weaviate: http://localhost:8080

## Testing Phase 1

`backend/scripts/compare_stores.py` compares the numpy brute-force store with
Weaviate using the same books and the same query vector, so differences come from
the store rather than the embeddings.

The notebook (`notebooks/01_explore.ipynb`) exercises **brute force only**.
Weaviate results are printed by the terminal script, so Docker has to be running
whenever you want to see them. Brute-force checks still work with Docker stopped;
only the Weaviate half fails to connect.

Run these from the repo root.

1. **Start Docker and Weaviate**:
   ```bash
   docker info >/dev/null && echo "docker ok"
   docker compose config --quiet && echo "compose ok"
   docker compose up -d
   docker compose ps
   ```
   Wait until the status is `healthy`.

2. **Check Weaviate health**:
   ```bash
   curl -fsS http://localhost:8080/v1/.well-known/ready
   curl -fsS http://localhost:8080/v1/meta | python -m json.tool
   ```

3. **Check Python dependencies**:
   ```bash
   source .venv/bin/activate
   python -c "import weaviate, sentence_transformers, pandas; print('deps ok', weaviate.__version__)"
   ```
   If imports fail: `pip install -r backend/requirements.txt`.

4. **Brute-force precheck** — no Docker needed:
   ```bash
   python backend/scripts/compare_stores.py --skip-weaviate
   ```
   This validates the parquet data, embeddings, and brute-force store.

5. **Full comparison**:
   ```bash
   python backend/scripts/compare_stores.py
   ```
   The collection uses HNSW with cosine distance. The first run inserts 16,551
   objects and takes ~1–2 minutes; the script then waits for vector indexing
   before searching. Expected:
   - `weaviate count: 16551`
   - `overlap@10: 10`
   - `same order: True`
   - `max score delta: < 0.001`
   - `self present: False` in both "more like this" lists

   If `weaviate count: 0` or the results are empty, check
   `docker compose logs weaviate` and use step 7. For a fast check without
   re-embedding, run `python backend/scripts/compare_stores.py --diagnose`.

6. **Idempotency check** — run step 5 again. Output should be the same, just
   faster, with no duplicate errors. Confirm the stored count:
   ```bash
   curl -s http://localhost:8080/v1/graphql -H 'Content-Type: application/json' \
     -d '{"query":"{ Aggregate { Book { meta { count } } } }"}'
   ```
   Expect `16551`.

7. **Reset if results look stale**:
   ```bash
   docker compose down -v
   docker compose up -d
   python backend/scripts/compare_stores.py
   ```
   This deletes only local Weaviate data; the script re-ingests it from
   `books.parquet`.

8. **Stop when done**:
   ```bash
   docker compose down      # keep data
   docker compose down -v   # delete data
   ```

Phase 1 parity, the Phase 2 baseline, and the brute-vs-Weaviate comparison are
recorded above. Remaining Phase 5 work is keyword/hybrid evaluation.

## Attribution

> Contains data from the CMU Book Summary Dataset (Bamman & Smith, 2013),
> https://www.cs.cmu.edu/~dbamman/booksummaries.html, licensed under
> CC BY-SA 3.0. Plot summaries are derived from Wikipedia; metadata is aligned
> from Freebase.
