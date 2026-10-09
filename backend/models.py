from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    filename: str
    page: int | None = None
    row: str | None = None  # record id / query description when the source has no page


class ChatResponse(BaseModel):
    answered: bool
    answer: str
    sources: list[Source] = []


class HealthResponse(BaseModel):
    status: str
