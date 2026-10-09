from backend.config import settings
from backend.rag import embeddings, vectorstore
from backend.rag.evidence import Evidence


def search(question: str, top_k: int | None = None, max_distance: float | None = None) -> list[Evidence]:
    top_k = top_k or settings.top_k
    max_distance = settings.max_distance if max_distance is None else max_distance

    collection = vectorstore.get_collection()
    total = collection.count()
    if total == 0:
        return []

    result = collection.query(
        query_embeddings=[embeddings.embed_query(question)],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        if dist > max_distance:
            continue
        hits.append(Evidence(
            text=text, filename=meta["filename"], page=meta.get("page") or None,
            row=meta.get("row_id") or None, distance=dist,
        ))
    return hits
