import asyncio
import shutil

import dotenv
import pandas as pd
import pytest
import requests

from lm_council import LanguageModelCouncil


def _fake_completion(prompt: str, model: str):
    return {
        "user_prompt": prompt,
        "model": model,
        "completion_text": f"{model} response to {prompt}",
        "completion_tokens": 1,
        "prompt_tokens": 1,
        "total_tokens": 2,
    }


@pytest.mark.asyncio
async def test_language_model_council(monkeypatch, tmp_path):
    dotenv.load_dotenv()

    class DummyResponse:
        def raise_for_status(self):
            raise requests.RequestException("fail")

    monkeypatch.setattr("lm_council.council.requests.get", lambda *a, **k: DummyResponse())

    lmc = LanguageModelCouncil(
        models=[
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.1-8b-instruct",
            "mistralai/mixtral-8x7b-instruct",
        ],
        openrouter_api_key="test-key",
    )

    async def fake_get_text_completions(user_prompt: str, temperature: float | None):
        return [
            asyncio.sleep(0, result=_fake_completion(user_prompt, model))
            for model in lmc.models
        ]

    async def fake_judge(completions_df: pd.DataFrame):
        # Return a simple rubric-style judgment dataframe without API calls.
        rows = []
        for _, row in completions_df.iterrows():
            rows.append(
                {
                    "user_prompt": row["user_prompt"],
                    "judge_model": lmc.judge_models[0],
                    "model_being_judged": row["model"],
                    "Coherence": 5,
                    "Relevance": 5,
                    "Overall": 5.0,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(lmc, "get_text_completions", fake_get_text_completions)
    monkeypatch.setattr(lmc, "judge", fake_judge)

    completions_df, judgments_df = await lmc.execute("Say hello.")

    outdir = tmp_path / "sample_session"
    lmc.save(outdir)
    LanguageModelCouncil.load(outdir, openrouter_api_key="test-key")
    shutil.rmtree(outdir)

    assert completions_df.shape[0] == 3
    assert judgments_df.shape[0] == 3
