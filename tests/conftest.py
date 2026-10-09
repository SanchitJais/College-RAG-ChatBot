import json

import pytest
from fastapi.testclient import TestClient

from backend import database
from backend.config import settings
from backend.main import app
from backend.rag import llm, vectorstore
from backend.rag.knowledge_base import load_knowledge_base
from backend.rag.prompts import ROUTER_PROMPT


@pytest.fixture(scope="session", autouse=True)
def kb_env(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "data_dir", tmp_path_factory.mktemp("kb"))
    mp.setattr(settings, "database_url", "")
    database.get_engine.cache_clear()
    vectorstore.get_collection.cache_clear()
    database.init_db()
    load_knowledge_base()
    yield
    database.get_engine().dispose()
    database.get_engine.cache_clear()
    vectorstore.get_collection.cache_clear()
    mp.undo()


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    def blocked(system, user):
        raise AssertionError("Test tried to call the real LLM; script it with scripted_llm().")
    monkeypatch.setattr(llm, "call_llm", blocked)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def scripted_llm(monkeypatch):
    import re

    def install(router_calls=None, answer=None):
        record = {"router": [], "answer": []}

        def fake(system, user):
            if system == ROUTER_PROMPT:
                record["router"].append(user)
                return json.dumps({"calls": router_calls or []})
            blocks = dict(re.findall(r'<context id="(C\d+)">\n(.*?)\n</context>', user, re.S))
            record["answer"].append((system, user, blocks))
            supported, text, labels = answer(blocks)
            return json.dumps({"supported": supported, "answer": text, "used_context_ids": labels})

        monkeypatch.setattr(llm, "call_llm", fake)
        return record

    return install
