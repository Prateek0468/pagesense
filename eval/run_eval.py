"""Evaluation harness for Phase 2.

Data flow:

1. Read `data/processed/books.parquet`, the same corpus used by Phase 1.
2. Validate that every hand-labeled wiki id still exists in that corpus.
3. Embed `embedding_text` once, optionally reusing an .npy/.npz cache.
4. Instantiate one `VectorStore` and search it for every labeled query.
5. Compute recall@k and MRR, print a compact report, and save per-query
   evidence plus aggregate scores to `eval/results/<store>.json`.

The harness only depends on the `VectorStore` contract (`add_documents`,
`search`, `close`), so brute force, Weaviate, and future filters/hybrid stores
can be scored with the same labels and metrics.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The backend is not installed as a package, so expose it for imports. This
# keeps scripts runnable as `python eval/run_eval.py` from the repo root and
# avoids introducing packaging work that is out of scope for these phases.
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# Imports stay below the sys.path edit so `app.*` resolves correctly.
from app.core.embeddings import embed_documents, embed_query
from app.store import BruteForceStore, WeaviateStore
from labeled_queries import QUERIES

# all-MiniLM-L6-v2 emits 384-dimensional vectors; validate cached arrays against
# this so a stale cache cannot silently corrupt a comparison.
EXPECTED_DIMENSION = 384


def load_books() -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, str]]:
    """Return the corpus, store-facing documents, and an id-to-title map."""
    books = pd.read_parquet(ROOT / "data" / "processed" / "books.parquet")

    # Stores only need metadata here; `embedding_text` is consumed separately as
    # the vector input. Author and genres are passed through so future filter
    # phases can reuse the same document payload without changing the harness.
    documents = books[["wiki_id", "title", "author", "genres"]].to_dict("records")

    # Stores use "id"; the parquet uses "wiki_id". Normalize to strings because
    # Weaviate results return string ids and labels are stored as strings too.
    for document in documents:
        document["id"] = str(document.pop("wiki_id"))

    # Titles are never used for scoring. They only make the report and JSON
    # artifact readable when diagnosing why a query missed its labels.
    titles = dict(zip(books["wiki_id"].astype(str), books["title"]))
    return books, documents, titles


def validate_labels(books: pd.DataFrame) -> None:
    """Fail early if any labeled book id is missing from the processed corpus."""
    known_ids = {str(book_id) for book_id in books["wiki_id"]}
    missing: dict[str, list[str]] = {}

    # A stale label would cap recall below 1.0 and look like a retrieval bug.
    # Checking up front keeps metric changes attributable to the store itself.
    for item in QUERIES:
        absent = sorted(set(item["relevant_ids"]) - known_ids)
        if absent:
            missing[item["id"]] = absent
    if missing:
        raise ValueError(f"labeled ids missing from books.parquet: {missing}")


def load_or_compute_vectors(
    books: pd.DataFrame, cache_path: Path | None
) -> np.ndarray:
    """Load cached corpus vectors if valid; otherwise embed and optionally save."""
    texts = books["embedding_text"].tolist()

    if cache_path and cache_path.exists():
        vectors = np.load(cache_path)
        expected_shape = (len(books), EXPECTED_DIMENSION)

        # The cache is keyed by corpus size and vector width only. Delete it
        # whenever books.parquet is regenerated, even if the row count is equal.
        if vectors.shape != expected_shape:
            raise ValueError(
                f"cached vectors have shape {vectors.shape}, expected {expected_shape}"
            )
        print(f"loaded corpus vectors from {cache_path}")
        return vectors

    vectors = embed_documents(texts)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # np.save appends ".npy" when given a Path, which would miss a requested
        # ".npz" filename on the next run. A file handle preserves the exact path.
        with cache_path.open("wb") as handle:
            np.save(handle, vectors)
        print(f"saved corpus vectors to {cache_path}")
    return vectors


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant books found in the top k results."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    # Sets avoid double-counting if a store ever returns a duplicate id.
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Score the first relevant hit: 1 for rank 1, 1/2 for rank 2, etc."""
    relevant = set(relevant_ids)
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    store: Any, k: int, titles: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run every labeled query through one store and compute aggregate metrics."""
    rows: list[dict[str, Any]] = []
    for item in QUERIES:
        # Use the same embedding helper as the app, so the score measures
        # serving behavior rather than a special evaluation-only path.
        query_vector = embed_query(item["query"])
        results = store.search(query_vector, k=k)
        retrieved_ids = [str(result["id"]) for result in results]
        relevant_ids = item["relevant_ids"]

        # MRR needs the rank of the first relevant hit. Recall@k only needs
        # membership anywhere in the top k, so compute both from one pass.
        first_hit_rank = None
        for rank, document_id in enumerate(retrieved_ids, start=1):
            if document_id in set(relevant_ids):
                first_hit_rank = rank
                break

        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        rank_score = reciprocal_rank(retrieved_ids, relevant_ids)

        # Keep full per-query evidence in the JSON artifact. Aggregate numbers
        # tell us *that* retrieval changed; retrieved ids tell us *where*.
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_titles": [titles.get(doc_id, "") for doc_id in retrieved_ids],
                "first_hit_rank": first_hit_rank,
                "recall": recall,
                "reciprocal_rank": rank_score,
            }
        )

    aggregate = {
        "recall_at_k": sum(row["recall"] for row in rows) / len(rows),
        "mrr": sum(row["reciprocal_rank"] for row in rows) / len(rows),
    }
    return rows, aggregate


def print_report(
    store_name: str, k: int, rows: list[dict[str, Any]], metrics: dict[str, float]
) -> None:
    """Print a compact per-query table followed by the two aggregate scores."""
    print(f"store: {store_name} | k={k} | queries={len(rows)}")
    for row in rows:
        first_hit = row["first_hit_rank"] or "-"
        print(
            f"{row['id']:<28} recall={row['recall']:.2f}  "
            f"rr={row['reciprocal_rank']:.2f}  first_hit={str(first_hit):>3}  "
            f"{row['query']}"
        )
    print(f"\nrecall@{k}: {metrics['recall_at_k']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")


def main() -> None:
    """Parse arguments, build one store, evaluate it, and save the results."""
    parser = argparse.ArgumentParser(description="Evaluate recall@k and MRR.")
    parser.add_argument("--store", choices=["brute", "weaviate"], default="brute")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Optional .npy/.npz path to reuse corpus vectors between runs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    books, documents, titles = load_books()
    validate_labels(books)
    vectors = load_or_compute_vectors(books, args.embedding_cache)

    # WeaviateStore connects to the local Docker service via WEAVIATE_* env
    # vars, so Docker must be running for `--store weaviate`. Brute force stays
    # fully in-process and needs no services.
    store = BruteForceStore() if args.store == "brute" else WeaviateStore()
    try:
        store.add_documents(documents, vectors)
        rows, metrics = evaluate(store, args.k, titles)
    finally:
        # Always release resources, including when a search or assertion fails.
        store.close()

    # Default filename is derived from the store name so repeated runs overwrite
    # the current result instead of accumulating stale comparisons.
    output = args.output or ROOT / "eval" / "results" / f"{args.store}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "store": args.store,
        "k": args.k,
        "query_count": len(QUERIES),
        "aggregate": metrics,
        "per_query": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")

    print_report(args.store, args.k, rows, metrics)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
