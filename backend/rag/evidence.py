from dataclasses import dataclass


@dataclass
class Evidence:
    text: str
    filename: str
    page: int | None = None
    row: str | None = None
    distance: float | None = None
