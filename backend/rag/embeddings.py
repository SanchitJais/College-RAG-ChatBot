"""Free, local embeddings: ChromaDB's bundled all-MiniLM-L6-v2 model (ONNX).

No API key and no cost. The model file (~80 MB) is downloaded once on first use
and cached in ~/.cache/chroma.
"""
from functools import lru_cache

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class EmbeddingError(Exception):
    """Embedding failed. A service error, not missing knowledge."""


@lru_cache
def _model() -> DefaultEmbeddingFunction:
    return DefaultEmbeddingFunction()


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        return [[float(x) for x in vec] for vec in _model()(texts)]
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed ({type(exc).__name__}).") from exc


def embed_query(question: str) -> list[float]:
    return embed_texts([question])[0]
