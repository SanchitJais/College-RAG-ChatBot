"""Load the CSV knowledge base.

* Every CSV becomes a SQL table (exact data, queried by fixed backend functions).
* Small descriptive CSVs (policies, notices, hostel, ...) are ALSO embedded, one
  snippet per row, so they can be found by meaning.
* The big students table is SQL only: 1,500 records are looked up exactly, not by
  similarity.

Re-running is safe: tables are rebuilt and the vector index is replaced.
"""
import csv
import re
from pathlib import Path

from sqlalchemy import Column, MetaData, Table, Text

from backend import database
from backend.rag import embeddings, vectorstore

KB_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base"
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
SQL_ONLY_TABLES = {"students"}
# Columns that describe the citation, not the content.
_META_COLS = {"id", "source", "page"}


def row_to_text(table: str, row: dict) -> str:
    parts = [f"{col.replace('_', ' ')}: {val}" for col, val in row.items()
             if col not in _META_COLS and val]
    return f"{table.replace('_', ' ').title()} record - " + "; ".join(parts)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        if not r.get("id"):
            raise ValueError(f"{path.name}: every row needs a non-empty 'id' column")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path.name}: duplicate ids")
    return rows


def _typed_rows(name: str, rows: list[dict]) -> list[dict]:
    casts = database.NUMERIC_COLUMNS.get(name, {})
    return [{k: (casts[k](v) if k in casts and v != "" else (v if v != "" else None))
             for k, v in r.items()} for r in rows]


def load_knowledge_base(directory: Path = KB_DIR) -> dict[str, int]:
    """Returns {table_name: row_count}."""
    loaded: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("*.csv")):
        if not _NAME.match(path.stem):
            raise ValueError(f"Bad CSV file name '{path.name}' (use lowercase letters/underscores)")
        loaded[path.stem] = _read_csv(path)
    if not loaded:
        raise ValueError(f"No CSV files found in {directory}")

    if "students" in loaded and "departments" in loaded:
        known = {d["code"] for d in loaded["departments"]}
        bad = {s["dept_code"] for s in loaded["students"]} - known
        if bad:
            raise ValueError(f"students.csv uses unknown department codes: {sorted(bad)}")

    # Embed first: if this fails, nothing has been changed yet.
    ids, texts, metas = [], [], []
    for name, rows in loaded.items():
        if name in SQL_ONLY_TABLES:
            continue
        for r in rows:
            page = int(r["page"]) if r.get("page", "").isdigit() else 0
            ids.append(f"csv:{name}:{r['id']}")
            texts.append(row_to_text(name, r))
            metas.append({
                "source_type": "csv", "document_id": f"csv:{name}",
                "filename": r.get("source") or f"{name}.csv",
                "page": page, "row_id": r["id"], "category": name,
            })
    vectors = embeddings.embed_texts(texts)

    # SQL tables (identifiers come from validated file/column names; values are bound params)
    engine = database.get_engine()
    for name, rows in loaded.items():
        cols = list(rows[0].keys()) if rows else ["id"]
        if not all(_NAME.match(c) for c in cols):
            raise ValueError(f"{name}.csv: column names must be lowercase letters/underscores")
        table = database.TYPED_TABLES.get(name)
        if table is None:
            table = Table(name, MetaData(), *[Column(c, Text) for c in cols])
        table.drop(engine, checkfirst=True)
        table.create(engine)
        if rows:
            with engine.begin() as conn:
                conn.execute(table.insert(), _typed_rows(name, rows))

    collection = vectorstore.get_collection()
    collection.delete(where={"source_type": "csv"})
    collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    return {name: len(rows) for name, rows in loaded.items()}
