import asyncio
import copy

import pandas as pd
import pytest
import requests

from lm_council.council import LanguageModelCouncil
from lm_council.judging import PRESET_EVAL_CONFIGS
from lm_council.judging.config import (
    EvaluationConfig,
    PairwiseComparisonConfig,
    PairwiseComparisonRandomMatchesConfig,
)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        LanguageModelCouncil(models=["model-a"], openrouter_api_key=None)


def test_rate_limit_fallback(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            raise requests.RequestException("fail")

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("lm_council.council.requests.get", lambda *a, **k: DummyResponse())

    council = LanguageModelCouncil(models=["model-a"], openrouter_api_key="dummy")

    assert council._limiter_max_calls == 5
    assert council._limiter_interval_seconds == 10


def test_rate_limit_clamps_to_minimum(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"rate_limit": {"requests": 0, "interval": "0s"}}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("lm_council.council.requests.get", lambda *a, **k: DummyResponse())

    council = LanguageModelCouncil(models=["model-a"], openrouter_api_key="dummy")

    assert council._limiter_max_calls == 1
    assert council._limiter_interval_seconds == 1


@pytest.mark.asyncio
async def test_random_pairwise_config_respected(monkeypatch):
    # Build a random pairwise config with a single pair sample.
    base = copy.deepcopy(PRESET_EVAL_CONFIGS["default_pairwise"])
    pairwise_config = PairwiseComparisonConfig(
        prompt_template=base.config.prompt_template,
        granularity=base.config.granularity,
        skip_equal_pairs=base.config.skip_equal_pairs,
        algorithm_type="random",
        position_flipping=False,
        algorithm_config=PairwiseComparisonRandomMatchesConfig(n_random_pairs=1),
    )
    eval_config = EvaluationConfig(
        type="pairwise_comparison",
        exclude_self_grading=False,
        cot_enabled=False,
        temperature=0.0,
        config=pairwise_config,
        reps=1,
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    class DummyResponse:
        def raise_for_status(self):
            raise requests.RequestException("fail")

    monkeypatch.setattr("lm_council.council.requests.get", lambda *a, **k: DummyResponse())
    council = LanguageModelCouncil(
        models=["openai/gpt-4o-mini", "meta-llama/llama-3.1-8b-instruct", "mistralai/mixtral-8x7b-instruct"],
        eval_config=eval_config,
        openrouter_api_key="dummy",
    )

    completions_df = pd.DataFrame(
        [
            {"user_prompt": "Hi", "model": m, "completion_text": f"{m} says hi"}
            for m in council.models
        ]
    )

    tasks = await council.get_judge_pairwise_rating_tasks_for_single_prompt(
        user_prompt="Hi",
        completions_df=completions_df,
        temperature=0.0,
    )

    # One sampled pair * three judge models (self-grading allowed).
    assert len(tasks) == len(council.judge_models) * 1
