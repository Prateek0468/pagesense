"""Shared embedding helpers for ingestion and search.

Ingestion uses embed_documents(); queries use embed_query(). One cached model
and normalized vectors keep scores comparable across both stores.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256


@lru_cache
def _model() -> SentenceTransformer:
    # Load the ~90 MB of weights once per process, not once per call.
    return SentenceTransformer(MODEL_NAME)


def _cap(text: str, limit: int) -> str:
    # The model adds [CLS]/[SEP], so callers budget two fewer tokens.
    # Slice the original text by tokenizer offsets; decoding token IDs would
    # rewrite the user's casing and punctuation.
    tokenizer = _model().tokenizer
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=limit,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    return text[: offsets[-1][1]] if offsets else ""


def embed_documents(texts: list[str]) -> np.ndarray:
    # Texts are the pre-capped embedding_text column from prepare_data.py.
    return _model().encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(text: str) -> np.ndarray:
    # Queries are unbounded, so truncate here. With normalized vectors,
    # a dot product equals cosine similarity.
    return _model().encode(
        [_cap(text, MAX_SEQ_LENGTH - 2)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
