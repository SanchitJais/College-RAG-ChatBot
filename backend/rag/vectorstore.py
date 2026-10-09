"""ChromaDB collection holding one searchable snippet per policy/notice/etc. row."""
from functools import lru_cache

import chromadb

from backend.config import settings

COLLECTION_NAME = "college_snippets"


@lru_cache
def get_collection():
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    # We supply our own embeddings; cosine distance: 0 = identical.
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
