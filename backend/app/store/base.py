from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Abstract interface for all vector store implementations."""

    @abstractmethod
    def add_documents(
        self, documents: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> None:
        """Store documents and their corresponding embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        exclude_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar documents.

        Each result follows this structure:

        {
            "id": str,
            "document": dict,
            "score": float
        }
        """
        pass

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Delete a document using its ID."""
        pass

    def close(self) -> None:
        """Release resources if required; in-memory stores may not need this."""
        pass
