from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def _cap(text: str, limit: int) -> str:
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
    return _model().encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(text: str) -> np.ndarray:
    return _model().encode(
        [_cap(text, MAX_SEQ_LENGTH - 2)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
