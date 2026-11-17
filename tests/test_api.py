"""Minimal FastAPI smoke tests (no LLM required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "campusask-api"


def test_samples():
    res = client.get("/v1/samples")
    assert res.status_code == 200
    samples = res.json()["samples"]
    assert len(samples) >= 3
    assert samples[0]["id"]


def test_identity_validate_ok():
    res = client.post(
        "/v1/identity/validate",
        json={"full_name": "Alex Rivera", "ucm_id": "700012345"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["first_name"] == "Alex"
    assert body["masked_id"].endswith("2345")
    assert "Hi Alex" in (body["greeting"] or "")


def test_identity_validate_bad():
    res = client.post(
        "/v1/identity/validate",
        json={"full_name": "A", "ucm_id": "12"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert "full_name" in body["errors"]
    assert "ucm_id" in body["errors"]


def test_ask_greeting_never_abstains():
    res = client.post(
        "/v1/ask",
        json={"question": "HI", "academic_level": "all", "first_name": "Alex"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "conversational"
    assert body["intent"] == "greeting"
    assert "Hi Alex" in body["answer"]
    assert body["sources"] == []


def test_ask_hi_how_are_you_short_circuits():
    res = client.post(
        "/v1/ask",
        json={
            "question": "Hi how are you",
            "academic_level": "all",
            "first_name": "Alex",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "conversational"
    assert body["intent"] == "how_are_you"
    assert "Alex" in body["answer"]
    assert body["sources"] == []


def test_ask_masters_student_context_never_abstains():
    res = client.post(
        "/v1/ask",
        json={
            "question": "i am a masters student",
            "academic_level": "all",
            "first_name": "Alex",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "conversational"
    assert body["intent"] == "student_context"
    assert "master" in body["answer"].lower()
    assert body["sources"] == []
    assert body.get("suggested_academic_level") == "graduate"
    assert body.get("academic_level") == "graduate"


def test_ask_masters_cs_with_undergraduate_level_never_abstains():
    """Undergraduate UI + masters CS identity must not return abstained."""
    from unittest.mock import patch

    def fake_llm(prompt: str, _settings):
        blocks = prompt.split("CHUNK_ID: ")[1:]
        for block in blocks:
            cid_line, _, rest = block.partition("\n")
            if "Computer Science, MS" in rest or "53-976" in rest:
                return (
                    "The Computer Science MS (53-976) is a 30-hour graduate program. "
                    f"[cite:{cid_line.strip()}]"
                )
        return "ABSTAIN"

    with patch("src.generation._call_llm", side_effect=fake_llm):
        res = client.post(
            "/v1/ask",
            json={
                "question": "I am a Masters student in computer science",
                "academic_level": "undergraduate",
                "first_name": "Alex",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] != "abstained"
    assert body.get("suggested_academic_level") == "graduate"
    assert body.get("academic_level") == "graduate"
    assert "computer science" in body["answer"].lower() or "master" in body["answer"].lower()


def test_identity_rejects_non_700_id():
    res = client.post(
        "/v1/identity/validate",
        json={"full_name": "Alex Rivera", "ucm_id": "123456789"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert "ucm_id" in body["errors"]
    assert "700" in body["errors"]["ucm_id"]


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    import json

    events = []
    for block in text.strip().split("\n\n"):
        name, data = "message", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data += line[len("data: "):]
        events.append((name, json.loads(data) if data else {}))
    return events


def test_ask_stream_matches_blocking_answer():
    """Streamed deltas carry [n] markers; the final 'done' equals /v1/ask."""
    from unittest.mock import patch

    def cited_answer(prompt: str) -> str:
        first_id = prompt.split("CHUNK_ID: ")[1].partition("\n")[0].strip()
        return f"Students below a 2.00 GPA are placed on probation [cite:{first_id}]."

    def fake_stream(prompt: str, _settings):
        text = cited_answer(prompt)
        # Split mid-marker to exercise the hold-back of partial "[cite:" text.
        cut = text.index("[cite:") + 4
        yield text[:cut]
        yield text[cut:]

    question = "What GPA places an undergraduate student on academic probation?"
    payload = {"question": question, "academic_level": "undergraduate"}
    with patch("src.generation._stream_llm", side_effect=fake_stream):
        res = client.post("/v1/ask/stream", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(res.text)
    names = [name for name, _ in events]
    assert names[0] == "status" and names[-1] == "done"
    assert "delta" in names

    streamed = "".join(data["text"] for name, data in events if name == "delta")
    assert "[cite:" not in streamed and "[1]" in streamed

    done = events[-1][1]
    assert done["status"] == "answered"
    assert done["sources"][0]["ref"] == 1

    with patch("src.generation._call_llm", side_effect=lambda p, _s: cited_answer(p)):
        blocking = client.post("/v1/ask", json=payload).json()
    assert done["answer"] == blocking["answer"]
    assert done["sources"] == blocking["sources"]


def test_ask_stream_abstain_is_never_streamed():
    from unittest.mock import patch

    with patch("src.generation._stream_llm", side_effect=lambda _p, _s: iter(["ABS", "TAIN"])):
        res = client.post(
            "/v1/ask/stream",
            json={"question": "What GPA places an undergraduate student on academic probation?"},
        )
    events = _parse_sse(res.text)
    assert not [d for n, d in events if n == "delta"]
    assert events[-1][0] == "done" and events[-1][1]["status"] == "abstained"


def test_ask_stream_chitchat_skips_llm():
    res = client.post("/v1/ask/stream", json={"question": "hello"})
    events = _parse_sse(res.text)
    assert events[-1][0] == "done"
    assert events[-1][1]["status"] == "conversational"
