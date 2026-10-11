# Backlog

All actionables live here. The [README](./README.md) describes the project; this
file tracks the work — must-haves under **Now**, everything else under **Later**.

## Now — build order (must-haves)

In order. Retrieval logic is handcoded; boilerplate glue may be AI-assisted.

- [ ] **Phase 0 — Data prep.** Parse CMU, handle missing genres and summaries,
      dedupe by title+author, build the embedding text, save `books.parquet`. Read
      20 rows by hand.
- [ ] **Phase 1 — Embed and search.** Start with numpy brute force in a notebook,
      then move to Weaviate behind a `VectorStore` interface. Sanity check:
      *"a cozy mystery set in a small village"* returns sensible books.
- [ ] **Phase 2 — Evaluation harness.** 10–20 labeled queries, recall@10 and MRR.
      Record the baseline now.
- [ ] **Phase 3 — Filters.** Genre, year range, and "has full text" (stub until
      Gutenberg matching exists in Phase 8).
- [ ] **Phase 4 — More like this + shelf blending.** Near-vector search excluding
      the source book; blend 3–5 favorites into one query vector.
- [ ] **Phase 5 — Hybrid search.** Compare vector / keyword / hybrid with the eval
      harness, so the comparison is a number.
- [ ] **Phase 6 — FastAPI + Next.js UI.** Backend endpoints first, then the UI.
- [ ] **Phase 7 — Corpus-level RAG.** Retrieve summaries, answer with cited books.
      Add the first RAG eval questions.
- [ ] **Phase 8 — Gutenberg matching + book-level RAG.** `match_sources.py`,
      `build_chunks.py`, the `Chunk` collection, and answerable vs. unanswerable
      test questions.
- [ ] **Phase 9 — Polish.** Rerank if time allows, README results table, demo GIF,
      clean-clone test.

## Now — UI features

- [ ] Natural-language search box.
- [ ] Result cards: title, author, genres, year, short summary snippet, cover or
      placeholder.
- [ ] Sidebar filters: genre, year range, "has full text".
- [ ] "More like this" button.
- [ ] "My shelf" — pick 3–5 favorites and get a blended recommendation.
- [ ] "Why this result?" showing the similarity score.

## Later

Ideas parked here until the must-haves ship. Nothing here is scheduled; promote an
item to **Now** when it becomes a must-have.

### UI

- Vector / keyword / hybrid search toggle.
- "Ask about this book" panel — show retrieved sources, and label whether an
  answer came from a summary or the full text.
- 2D embedding map (seed of the separate playground project).

### Retrieval & ML

- Reranking (the Phase 9 stretch goal): cross-encoder or LLM reranker over the
  top-N retrieved candidates.
- Tune hybrid weighting / try reciprocal rank fusion (RRF).
- Query expansion / HyDE for weak natural-language queries.
- Embedding model bake-off: `all-MiniLM-L6-v2` vs. `bge-small` vs. `e5-small`.
- Fine-tune or adapt embeddings using genres as weak labels.

### Data enrichment

- Ratings and page counts from an external source (CMU lacks both) — re-add
  rating/pages filters and card fields if adopted.
- Cover art via Open Library or Google Books, with a placeholder fallback.
- Broaden Gutenberg full-text coverage beyond public-domain matches.

### Backend

- Persisted shelves and user accounts (auth).
- Response caching and pagination for large result sets.
- Rate limiting.
- Additional embedding providers (e.g. OpenAI) behind the existing interface.

### Frontend

- Shareable search URLs / deep links to a query + filters.
- Infinite scroll or pagination on the results grid.
- Responsive/mobile polish and empty/error states.

### Ops & quality

- Deployment (e.g. Fly.io / Render for backend, Vercel for frontend).
- CI running the eval harness on every change; regression gate on recall@10.
- Structured logging and basic monitoring.
