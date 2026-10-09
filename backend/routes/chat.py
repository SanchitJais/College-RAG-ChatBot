import logging

from fastapi import APIRouter, HTTPException

from backend.models import ChatRequest, ChatResponse
from backend.rag import generation, retrieval, router as db_router
from backend.rag.embeddings import EmbeddingError
from backend.rag.llm import GenerationError, RateLimitedError

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "Question must not be empty.")
    try:
        # 1. Does this need the student database? (validated tool calls only)
        db_evidence = db_router.run(db_router.plan(question))
        # 2. Policy/notice snippets close enough to the question
        snippets = retrieval.search(question)

        evidence = db_evidence + snippets
        if not evidence:
            return generation.refusal()  # nothing relevant found: no answer call at all
        # 3. Grounded answer; the backend verifies it before it is returned
        return generation.answer_question(question, evidence)
    except RateLimitedError:
        raise HTTPException(429, "The assistant is busy right now (free-tier rate limit). "
                                 "Please try again in a few seconds.")
    except (EmbeddingError, GenerationError) as exc:
        # Service problem: say so, never fall back to an invented answer.
        log.error("Chat dependency failure: %s", type(exc).__name__)
        raise HTTPException(503, "The assistant is temporarily unavailable. Please try again shortly.")
    except Exception:
        log.exception("Unexpected chat failure")
        raise HTTPException(503, "The knowledge base is temporarily unavailable. Please try again shortly.")
