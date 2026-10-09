import logging

from pydantic import ValidationError

from backend.rag import llm, student_tools
from backend.rag.prompts import ROUTER_PROMPT

log = logging.getLogger(__name__)
MAX_CALLS = 3


def plan(question: str) -> list[tuple[str, object]]:
    raw = llm.call_llm(ROUTER_PROMPT, f"Question: {question}")
    try:
        calls = llm.parse_json_object(raw).get("calls", [])
    except ValueError:
        log.warning("Router reply was not valid JSON (%r); continuing without database lookup", raw[:120])
        return []
    if not isinstance(calls, list):
        return []

    planned = []
    for call in calls[:MAX_CALLS]:
        if not isinstance(call, dict):
            continue
        entry = student_tools.TOOLS.get(call.get("tool"))
        if entry is None:
            continue
        model, _ = entry
        try:
            planned.append((call["tool"], model.model_validate(call.get("args") or {})))
        except ValidationError:
            log.warning("Router proposed invalid arguments for %s; dropped", call.get("tool"))
    return planned


def run(planned) -> list:
    evidence = []
    for tool, args in planned:
        result = student_tools.TOOLS[tool][1](args)
        if result is not None:
            evidence.append(result)
    return evidence
