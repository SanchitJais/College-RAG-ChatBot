from backend import database
from backend.rag.embeddings import EmbeddingError
from backend.rag.knowledge_base import load_knowledge_base


def main() -> int:
    database.init_db()
    try:
        counts = load_knowledge_base()
    except (EmbeddingError, ValueError) as exc:
        print(f"Load failed: {exc}")
        return 1
    for table, n in counts.items():
        print(f"{table}: {n} rows")
    print(f"Done. {sum(counts.values())} rows loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
