"""Weaviate implementation of VectorStore.

Vectors are supplied by us (no server-side vectorizer) so ranking matches the
brute-force store. Connection details can be overridden with WEAVIATE_* env vars.
"""

import os
import uuid
from typing import Any

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery

from app.store.base import VectorStore

DEFAULT_COLLECTION = "Book"
VECTOR_NAME = "default"
PROPERTY_NAMES = [
    "source_id",
    "title",
    "author",
    "genres",
    "publication_date",
]


class WeaviateStore(VectorStore):
    """VectorStore backed by a local Weaviate collection with self-provided vectors."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        grpc_port: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._client = weaviate.connect_to_local(
            host=host or os.getenv("WEAVIATE_HOST", "localhost"),
            port=port or int(os.getenv("WEAVIATE_PORT", "8080")),
            grpc_port=grpc_port or int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        )
        self._collection_name = collection_name or os.getenv(
            "WEAVIATE_COLLECTION", DEFAULT_COLLECTION
        )

        if not self._client.collections.exists(self._collection_name):
            self._client.collections.create(
                name=self._collection_name,
                # "id" is reserved by Weaviate, so the original id is stored
                # under source_id.
                properties=[
                    Property(name="source_id", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="author", data_type=DataType.TEXT),
                    Property(name="genres", data_type=DataType.TEXT_ARRAY),
                    Property(name="publication_date", data_type=DataType.TEXT),
                ],
                # No vectorizer: vectors come from embed_documents(). Cosine
                # distance matches the normalized dot product of the other store.
                vector_config=Configure.Vectors.self_provided(
                    name=VECTOR_NAME,
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    ),
                ),
            )
        self._collection = self._client.collections.use(self._collection_name)
        self.dimension: int | None = None

    def add_documents(
        self, documents: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> None:
        """Insert or replace documents using their original ids."""
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

        ids = []
        for document in documents:
            if "id" not in document:
                raise ValueError("every document must contain an 'id' key")
            ids.append(str(document["id"]))
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in documents")

        # Delete first so re-adding an id replaces it rather than failing.
        uuids = [self._uuid_for(document_id) for document_id in ids]
        if uuids:
            self._collection.data.delete_many(where=Filter.by_id().contains_any(uuids))

        with self._collection.batch.dynamic() as batch:
            for document, vector, object_uuid in zip(documents, vectors, uuids):
                batch.add_object(
                    properties=self._properties(document),
                    uuid=object_uuid,
                    vector={VECTOR_NAME: vector.tolist()},
                )

        # The batch flushes on context exit, so errors are only complete here.
        if batch.number_errors > 0:
            messages = "; ".join(
                failure.message for failure in batch.failed_objects[:3]
            )
            raise RuntimeError(
                f"Weaviate batch reported {batch.number_errors} errors: {messages}"
            )

        self._collection.batch.wait_for_vector_indexing()

        self.dimension = vectors.shape[1]

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        exclude_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the k nearest documents as {"id", "document", "score"}."""
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError(f"query_vector must be 1D, got {query.ndim}D")
        if self.dimension is not None and query.shape[0] != self.dimension:
            raise ValueError(
                f"dimension mismatch: expected {self.dimension}, "
                f"got {query.shape[0]}"
            )

        # "More like this": hide the source book at query time.
        filters = None
        if exclude_id is not None:
            filters = Filter.by_id().not_equal(self._uuid_for(exclude_id))

        response = self._collection.query.near_vector(
            near_vector=query.tolist(),
            limit=k,
            filters=filters,
            target_vector=VECTOR_NAME,
            return_metadata=MetadataQuery(distance=True),
            return_properties=PROPERTY_NAMES,
        )

        results = []
        for obj in response.objects:
            properties = obj.properties
            distance = obj.metadata.distance if obj.metadata else None
            results.append(
                {
                    "id": properties.get("source_id", str(obj.uuid)),
                    "document": {
                        "id": properties.get("source_id", str(obj.uuid)),
                        "title": properties.get("title", ""),
                        "author": properties.get("author", ""),
                        "genres": properties.get("genres", []),
                        "publication_date": properties.get("publication_date", ""),
                    },
                    # Cosine distance = 1 - similarity; invert so higher wins,
                    # matching BruteForceStore's score direction.
                    "score": 1.0 - distance if distance is not None else 0.0,
                }
            )
        return results

    def count(self) -> int:
        return self._collection.aggregate.over_all(total_count=True).total_count or 0

    def diagnostics(self) -> dict[str, Any]:
        config = self._collection.config.get()
        named_configs = getattr(config, "vector_config", None)
        named_config = (
            named_configs.get(VECTOR_NAME)
            if isinstance(named_configs, dict)
            else None
        )
        index_config = getattr(named_config, "vector_index_config", None) or getattr(
            config, "vector_index_config", None
        )
        distance = getattr(index_config, "distance_metric", None) or getattr(
            index_config, "distance", None
        )

        sample = self._collection.query.fetch_objects(limit=1, include_vector=True)
        raw_vector = sample.objects[0].vector if sample.objects else None
        vector_keys = sorted(raw_vector.keys()) if isinstance(raw_vector, dict) else []
        vector = raw_vector
        if isinstance(vector, dict):
            vector = vector.get(VECTOR_NAME) or next(iter(vector.values()), None)

        self_query_objects = 0
        if vector is not None:
            probe = self._collection.query.near_vector(
                near_vector=vector,
                limit=1,
                target_vector=VECTOR_NAME,
                return_metadata=MetadataQuery(distance=True),
            )
            self_query_objects = len(probe.objects)

        return {
            "vector_names": vector_keys,
            "vector_index_type": str(
                getattr(named_config, "vector_index_type", None)
                or getattr(config, "vector_index_type", None)
            ),
            "distance_metric": str(distance),
            "vectorizer": str(
                getattr(named_config, "vectorizer", None)
                or getattr(config, "vectorizer", None)
            ),
            "sample_vector_length": len(vector) if vector is not None else 0,
            "self_query_objects": self_query_objects,
        }

    def delete(self, document_id: str) -> None:
        self._collection.data.delete_by_id(uuid=self._uuid_for(document_id))

    def close(self) -> None:
        self._client.close()

    def _properties(self, document: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_id": str(document["id"]),
            "title": str(document.get("title", "")),
            "author": str(document.get("author", "")),
            "genres": self._genres(document.get("genres")),
            "publication_date": str(document.get("publication_date", "")),
        }

    @staticmethod
    def _genres(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, str):
            value = [value]
        elif not isinstance(value, (list, tuple, set)):
            value = [value]
        return [str(genre) for genre in value]

    @staticmethod
    def _uuid_for(document_id: str) -> uuid.UUID:
        # Wiki ids like "620" are not UUIDs; derive a stable one so delete and
        # search refer to the same object across restarts.
        return uuid.uuid5(uuid.NAMESPACE_URL, f"pagesense:book:{document_id}")
