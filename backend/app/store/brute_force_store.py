from typing import Any

import numpy as np

from app.store.base import VectorStore


class BruteForceStore(VectorStore):
    """A vector store that compares a query against every stored vector.

    There is no index: search is a single matrix-vector multiply over all rows.
    At ~16k books × 384 dimensions this is fast enough, and it is the baseline
    that the Weaviate implementation will be measured against.
    """

    def __init__(self) -> None:
        # Names match the VectorStore interface: documents hold the metadata,
        # embeddings is one row per document. Ids live inside each document so
        # there is only one place to update when documents are added or removed.
        self.documents: list[dict[str, Any]] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)

        # Set from the first batch of vectors; used to validate later inputs.
        self.dimension: int | None = None

    def add_documents(
        self, documents: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> None:
        """Load (or replace) documents and their precomputed embeddings."""

        # Convert the list of vectors into one 2D matrix: shape (n_docs, dim).
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {vectors.ndim}D")
        if len(vectors) != len(documents):
            raise ValueError(
                f"embeddings/documents mismatch: {len(vectors)} vectors "
                f"for {len(documents)} documents"
            )
        if self.dimension is not None and vectors.shape[1] != self.dimension:
            raise ValueError(
                f"dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        # The store addresses documents by their "id" key; the parquet uses
        # "wiki_id", so the caller maps that to "id" before calling this.
        ids = []
        for document in documents:
            if "id" not in document:
                raise ValueError("every document must contain an 'id' key")
            ids.append(str(document["id"]))
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in documents")

        # Re-adding an id replaces its old row instead of creating a duplicate.
        new_ids = set(ids)
        keep = [
            i
            for i, document in enumerate(self.documents)
            if str(document["id"]) not in new_ids
        ]
        self.documents = [self.documents[i] for i in keep] + documents

        # Stack the retained rows and the new rows into one matrix again.
        # Skip indexing an empty (0, 0) array on the very first add.
        pieces = ([self.embeddings[keep]] if keep else []) + [vectors]
        self.embeddings = np.vstack(pieces)
        self.dimension = self.embeddings.shape[1]

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        exclude_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the k nearest documents as {"id", "document", "score"}."""
        if not self.documents:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError(f"query_vector must be 1D, got {query.ndim}D")
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"dimension mismatch: expected {self.dimension}, "
                f"got {query.shape[0]}"
            )

        # Since documents were embedded with normalize_embeddings=True, this
        # dot product equals cosine similarity: higher is more similar.
        scores = self.embeddings @ query

        # For "more like this", hide the source book so it cannot rank itself.
        ids = [str(document["id"]) for document in self.documents]
        if exclude_id in ids:
            scores[ids.index(exclude_id)] = -np.inf

        # Descending order, then take the top k row positions.
        idx = np.argsort(scores)[::-1][:k]

        # Return the same dict shape the VectorStore interface promises, so a
        # Weaviate store can be swapped in without changing calling code.
        return [
            {
                "id": ids[i],
                "document": self.documents[i],
                "score": float(scores[i]),
            }
            for i in idx
        ]

    def get_vector(self, document_id: str) -> np.ndarray | None:
        """Fetch one stored vector, e.g. to find books similar to a known one."""
        ids = [str(document["id"]) for document in self.documents]
        if document_id not in ids:
            return None

        # Copy so callers cannot mutate the vector held by the store.
        return self.embeddings[ids.index(document_id)].copy()

    def delete(self, document_id: str) -> None:
        """Remove a document; deleting an unknown id is a no-op."""
        keep = [
            i
            for i, document in enumerate(self.documents)
            if str(document["id"]) != document_id
        ]
        if len(keep) == len(self.documents):
            return

        # Drop the same rows from documents and embeddings.
        self.documents = [self.documents[i] for i in keep]
        self.embeddings = self.embeddings[keep]
        if not self.documents:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.dimension = None

    def close(self) -> None:
        """Nothing to release: everything lives in local Python memory."""
        return None

    def __len__(self) -> int:
        return len(self.documents)
