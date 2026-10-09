import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from backend.config import settings
from backend.rag import llm
from backend.rag.llm import GenerationError
from backend.rag.prompts import ROUTER_PROMPT

REAL_CALL_LLM = llm.call_llm


class FakeLLMServer:
    def __init__(self):
        self.requests: list[dict] = []
        self.handler = lambda body: (200, "{}")
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                      "body": body})
                status, content = outer.handler(body)
                payload = (json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]})
                           if status == 200 else json.dumps({"error": {"message": "nope"}}))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload.encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def fake(monkeypatch):
    server = FakeLLMServer()
    monkeypatch.setattr(settings, "llm_base_url", server.url)
    monkeypatch.setattr(settings, "llm_api_key", "test-key-123")
    monkeypatch.setattr(settings, "llm_model", "test-model")
    monkeypatch.setattr(llm, "call_llm", REAL_CALL_LLM)
    llm._client.cache_clear()
    yield server
    server.server.shutdown()
    llm._client.cache_clear()


def test_request_goes_to_configured_endpoint_with_expected_shape(fake):
    fake.handler = lambda body: (200, "hello")
    assert llm.call_llm("SYS", "USER") == "hello"
    req = fake.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer test-key-123"
    assert req["body"]["model"] == "test-model"
    assert req["body"]["temperature"] == 0
    assert req["body"]["messages"] == [{"role": "system", "content": "SYS"},
                                       {"role": "user", "content": "USER"}]


def test_auth_failure_becomes_generation_error_without_leaking_the_key(fake):
    fake.handler = lambda body: (401, "")
    with pytest.raises(GenerationError) as exc:
        llm.call_llm("s", "u")
    assert "test-key-123" not in str(exc.value)


def test_server_error_becomes_generation_error(fake):
    fake.handler = lambda body: (500, "")
    with pytest.raises(GenerationError):
        llm.call_llm("s", "u")


def test_full_chat_through_the_real_client(fake, client):
    def respond(body):
        system = body["messages"][0]["content"]
        user = body["messages"][1]["content"]
        if system == ROUTER_PROMPT:
            return 200, '```json\n{"calls": [{"tool": "student_summary", "args": {"department": "MECH", "year": 2}}]}\n```'
        assert "matching students: 75" in user
        return 200, json.dumps({"supported": True, "answer": "There are 75 students.",
                                "used_context_ids": ["C1"]})

    fake.handler = respond
    r = client.post("/chat", json={"question": "How many second year mechanical students are there?"})
    body = r.json()
    assert r.status_code == 200 and body["answered"] is True
    assert body["sources"][0]["filename"] == "students table (SQL database)"
    assert len(fake.requests) == 2


def test_chat_returns_503_when_the_model_server_is_down(fake, client):
    fake.handler = lambda body: (500, "")
    r = client.post("/chat", json={"question": "What is the attendance policy?"})
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
    assert "nope" not in r.text and "test-key-123" not in r.text


def test_unrelated_question_makes_only_the_router_call(fake, client):
    fake.handler = lambda body: (200, '{"calls": []}')
    r = client.post("/chat", json={"question": "What is the capital of France?"})
    assert r.json()["answered"] is False
    assert len(fake.requests) == 1


def test_rate_limit_gets_a_friendly_429_not_a_generic_error(fake, client, monkeypatch):
    monkeypatch.setattr(settings, "llm_max_retries", 0)
    llm._client.cache_clear()
    fake.handler = lambda body: (429, "")
    r = client.post("/chat", json={"question": "What is the attendance policy?"})
    assert r.status_code == 429 and "try again" in r.json()["detail"]


def test_bad_key_is_logged_with_a_hint_but_never_the_key(fake, caplog):
    fake.handler = lambda body: (401, "")
    with caplog.at_level("ERROR"), pytest.raises(GenerationError):
        llm.call_llm("s", "u")
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "HTTP 401" in logged and "LLM_API_KEY" in logged
    assert "test-key-123" not in logged
