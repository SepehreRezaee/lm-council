import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lm_council.api.server import app, runner


class FakeCouncil:
    def __init__(self):
        self.executed = False

    async def execute(self, prompts):
        self.executed = True
        completions = pd.DataFrame(
            [
                {"user_prompt": prompts[0], "model": "m1", "completion_text": "hi"},
                {"user_prompt": prompts[0], "model": "m2", "completion_text": "hi2"},
            ]
        )
        judgments = pd.DataFrame(
            [
                {"user_prompt": prompts[0], "judge_model": "m1", "model_being_judged": "m2", "Overall": 5}
            ]
        )
        return completions, judgments


class InMemoryAsyncCollection:
    def __init__(self):
        self.data = {}

    async def update_one(self, filter, update, upsert=False):
        doc = update.get("$set", {})
        self.data[filter["thread_id"]] = doc

    async def find_one(self, filter, projection=None):
        doc = self.data.get(filter["thread_id"])
        if not doc:
            return None
        # simulate projection by dropping _id if requested
        if projection and projection.get("_id") == 0:
            return {k: v for k, v in doc.items() if k != "_id"}
        return doc


@pytest.fixture
def client(monkeypatch):
    fake = FakeCouncil()
    fake_collection = InMemoryAsyncCollection()

    monkeypatch.setattr(runner, "_build_council", lambda payload: fake)
    async def fake_get_collection():
        return fake_collection

    monkeypatch.setattr("lm_council.api.server.get_collection", fake_get_collection)

    client = TestClient(app)
    return client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_configs(client):
    resp = client.get("/configs")
    assert resp.status_code == 200
    assert "default_rubric" in resp.json()["preset_keys"]


def test_run_council_success(client):
    payload = {
        "models": ["m1", "m2"],
        "prompts": "hello",
        "eval_config_key": "default_rubric",
        "completion_max_tokens": 32,
        "judge_max_tokens": 32,
    }
    resp = client.post("/council/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["completions"]) == 2
    assert len(body["judgments"]) == 1


def test_thread_crud(client):
    payload = {"thread_id": "t1", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/threads", json=payload)
    assert resp.status_code == 200

    get_resp = client.get("/threads/t1")
    assert get_resp.status_code == 200
    assert get_resp.json()["thread_id"] == "t1"
