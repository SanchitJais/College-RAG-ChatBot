from dataclasses import dataclass


@dataclass
class Evidence:
    """One piece of supporting material shown to the LLM: a database result or a
    retrieved snippet. The citation fields come from our own data, never from the LLM."""
    text: str
    filename: str
    page: int | None = None  # source page, when the source has one
    row: str | None = None  # record id / query description, when there is no page
    distance: float | None = None  # retrieval distance (snippets only), for tuning
