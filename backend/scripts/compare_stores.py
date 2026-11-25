"""Compare brute-force and local Weaviate retrieval on ports 8080/50051."""

import argparse
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.core.embeddings import embed_documents, embed_query
from app.store import BruteForceStore, WeaviateStore


def load_books():
    books = pd.read_parquet(ROOT / "data" / "processed" / "books.parquet")
    documents = books[["wiki_id", "title", "author", "genres"]].to_dict("records")
    for document in documents:
        document["id"] = str(document.pop("wiki_id"))
    return books, documents


def summarize(name, results):
    print(f"\n{name}")
    for rank, result in enumerate(results, 1):
        document = result["document"]
        print(
            f"{rank:>2}. {result['id']:>8}  {result['score']:.4f}  "
            f"{document['title']} — {document['author']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare brute-force and Weaviate retrieval.")
    parser.add_argument("--query", default="a cozy mystery set in a small village")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--source-id", default="620")
    parser.add_argument("--skip-weaviate", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    if args.diagnose:
        store = WeaviateStore()
        try:
            print("weaviate count:", store.count())
            print("weaviate diagnostics:", store.diagnostics())
        finally:
            store.close()
        return

    print(f"query: {args.query}")
    books, documents = load_books()
    vectors = embed_documents(books["embedding_text"].tolist())

    brute = BruteForceStore()
    brute.add_documents(documents, vectors)
    query_vector = embed_query(args.query)
    brute_results = brute.search(query_vector, k=args.k)
    summarize("brute force", brute_results)

    if args.skip_weaviate:
        return

    weaviate = WeaviateStore()
    try:
        weaviate.add_documents(documents, vectors)
        print("weaviate count:", weaviate.count())
        weaviate_results = weaviate.search(query_vector, k=args.k)
        if not weaviate_results:
            print("weaviate diagnostics:", weaviate.diagnostics())
            print(
                "weaviate returned no results; check the count above and "
                "`docker compose logs weaviate`"
            )
            return
        summarize("weaviate", weaviate_results)

        brute_ids = [result["id"] for result in brute_results]
        weaviate_ids = [result["id"] for result in weaviate_results]
        print(f"\noverlap@{args.k}:", len(set(brute_ids) & set(weaviate_ids)))
        print("same order:", brute_ids == weaviate_ids)
        max_delta = max(
            (
                abs(a["score"] - b["score"])
                for a, b in zip(brute_results, weaviate_results)
            ),
            default=0.0,
        )
        print("max score delta:", round(max_delta, 6))

        source_vector = brute.get_vector(args.source_id)
        if source_vector is None:
            print(f"source id {args.source_id} not found; skipping more-like-this")
            return

        for name, store in (("brute force", brute), ("weaviate", weaviate)):
            results = store.search(source_vector, k=5, exclude_id=args.source_id)
            print(f"\nmore like this ({name}):", [result["id"] for result in results])
            print("self present:", any(result["id"] == args.source_id for result in results))
    finally:
        weaviate.close()


if __name__ == "__main__":
    main()
