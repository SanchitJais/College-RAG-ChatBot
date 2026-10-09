import re

from backend.config import REFUSAL_MESSAGE
from backend.models import ChatResponse, Source
from backend.rag import llm
from backend.rag.evidence import Evidence
from backend.rag.llm import GenerationError
from backend.rag.prompts import SYSTEM_PROMPT, build_user_prompt

__all__ = ["GenerationError", "refusal", "answer_question"]


def refusal() -> ChatResponse:
    return ChatResponse(answered=False, answer=REFUSAL_MESSAGE, sources=[])


_TYPOGRAPHY = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u2011": "-", "\u2009": " "})
_LEAKED_LABEL = re.compile(r"\s*[\(\[]\s*C\d+(?:\s*[,;]\s*C\d+)*\s*[\)\]]")


def _clean_answer(text: str) -> str:
    return _LEAKED_LABEL.sub("", text.translate(_TYPOGRAPHY)).strip()


def _normalize_ids(used):
    if isinstance(used, str):
        used = [part for part in re.split(r"[,;\s]+", used.strip()) if part]
    if not isinstance(used, list):
        return used
    out = []
    for item in used:
        if isinstance(item, str):
            m = re.fullmatch(r"\[?\s*c(\d+)\s*\]?", item.strip(), re.I)
            item = f"C{m.group(1)}" if m else item
        out.append(item)
    return out


def answer_question(question: str, evidence: list[Evidence]) -> ChatResponse:
    if not evidence:
        return refusal()

    user_prompt, labelled = build_user_prompt(question, evidence)
    raw = llm.call_llm(SYSTEM_PROMPT, user_prompt)

    try:
        data = llm.parse_json_object(raw)
        supported = data["supported"]
        answer = data["answer"]
        used = _normalize_ids(data["used_context_ids"])
    except (ValueError, KeyError, TypeError):
        raise GenerationError("The language model returned an unreadable reply.")

    # Backend policy: trust the answer only if the model named evidence that was really supplied.
    if supported is not True:
        return refusal()
    if not isinstance(answer, str) or not answer.strip():
        return refusal()
    if not isinstance(used, list) or not used or not all(isinstance(u, str) for u in used):
        return refusal()
    if any(u not in labelled for u in used):
        return refusal()

    sources: list[Source] = []
    for label in dict.fromkeys(used):
        item = labelled[label]
        source = Source(filename=item.filename, page=item.page, row=None if item.page else item.row)
        if source not in sources:
            sources.append(source)
    return ChatResponse(answered=True, answer=_clean_answer(answer), sources=sources)
