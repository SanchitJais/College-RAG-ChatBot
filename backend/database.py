from functools import lru_cache

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine

from backend.config import settings

metadata = MetaData()

students = Table(
    "students",
    metadata,
    Column("id", String(10), primary_key=True),
    Column("roll_no", String(20), nullable=False, unique=True),
    Column("name", String(100), nullable=False, index=True),
    Column("gender", String(10)),
    Column("dept_code", String(10), nullable=False, index=True),
    Column("department", String(100)),
    Column("year", Integer, nullable=False),
    Column("cgpa", Float),
    Column("attendance_percent", Float),
    Column("backlogs", Integer, default=0),
    Column("hostel_resident", String(3)),
    Column("city", String(50)),
    Column("email", String(120)),
)

TYPED_TABLES = {"students": students}
NUMERIC_COLUMNS = {
    "students": {"year": int, "cgpa": float, "attendance_percent": float, "backlogs": int},
}


@lru_cache
def get_engine() -> Engine:
    url = settings.db_url
    if url.startswith("sqlite"):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True)


def init_db() -> None:
    metadata.create_all(get_engine())


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(select(1))
        return True
    except Exception:
        return False
