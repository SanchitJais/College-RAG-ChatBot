from functools import lru_cache

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class EmbeddingError(Exception):
    pass


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
