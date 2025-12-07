import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone

from lm_council import LanguageModelCouncil
from lm_council.api.db import MongoConfig, ensure_indexes, get_collection
from lm_council.judging import PRESET_EVAL_CONFIGS
from lm_council.judging.config import EvaluationConfig


logger = logging.getLogger("lm_council.api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class RunRequest(BaseModel):
    models: List[str] = Field(..., min_items=1, description="Models to generate completions.")
    prompts: List[str] | str = Field(..., description="Single prompt or list of prompts.")
    judge_models: Optional[List[str]] = Field(
        None, description="Optional override for judge models. Defaults to models."
    )
    eval_config_key: str = Field(
        "default_rubric", description="Key of preset evaluation config to use."
    )
    openrouter_api_key: Optional[str] = Field(
        None, description="Optional OpenRouter API key. Falls back to environment."
    )
    completion_max_tokens: int = Field(
        256, gt=0, description="Token cap for completion calls."
    )
    judge_max_tokens: int = Field(256, gt=0, description="Token cap for judge calls.")

    @validator("prompts")
    def _normalize_prompts(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or not v:
            raise ValueError("prompts must be a non-empty string or list of strings.")
        if not all(isinstance(p, str) and p.strip() for p in v):
            raise ValueError("Each prompt must be a non-empty string.")
        return v

    @validator("models")
    def _validate_models(cls, v: List[str]) -> List[str]:
        if not all(isinstance(m, str) and m.strip() for m in v):
            raise ValueError("Each model must be a non-empty string.")
        return v

    @validator("judge_models")
    def _validate_judges(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if not v:
            raise ValueError("judge_models must be non-empty when provided.")
        if not all(isinstance(m, str) and m.strip() for m in v):
            raise ValueError("Each judge model must be a non-empty string.")
        return v


class RunResponse(BaseModel):
    completions: List[Dict[str, Any]]
    judgments: List[Dict[str, Any]]


class ConfigListResponse(BaseModel):
    preset_keys: List[str]


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker: user|assistant|system|judge")
    content: str = Field(..., description="Message content")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the message (UTC)",
    )

    @validator("role")
    def _role_allowed(cls, v: str) -> str:
        allowed = {"user", "assistant", "system", "judge"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}")
        return v

    @validator("content")
    def _content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content must be non-empty")
        return v


class ChatThread(BaseModel):
    thread_id: str = Field(..., description="Unique thread identifier")
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Stored conversation messages"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for the thread"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Thread creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Thread last update timestamp (UTC)",
    )

    @validator("thread_id")
    def _thread_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("thread_id must be non-empty")
        return v


class CouncilRunner:
    """Encapsulates council construction and execution to keep FastAPI layer thin."""

    def __init__(self, preset_configs: Dict[str, EvaluationConfig]):
        self._preset_configs = preset_configs

    def _get_eval_config(self, key: str) -> EvaluationConfig:
        if key not in self._preset_configs:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown eval_config_key '{key}'. Available: {list(self._preset_configs)}",
            )
        return self._preset_configs[key]

    def _build_council(self, payload: RunRequest) -> LanguageModelCouncil:
        eval_config = self._get_eval_config(payload.eval_config_key)
        return LanguageModelCouncil(
            models=payload.models,
            judge_models=payload.judge_models,
            eval_config=eval_config,
            openrouter_api_key=payload.openrouter_api_key,
            completion_max_tokens=payload.completion_max_tokens,
            judge_max_tokens=payload.judge_max_tokens,
        )

    async def run(self, payload: RunRequest) -> RunResponse:
        council = self._build_council(payload)
        completions_df, judgments_df = await council.execute(payload.prompts)
        return RunResponse(
            completions=completions_df.to_dict(orient="records"),
            judgments=judgments_df.to_dict(orient="records"),
        )


runner = CouncilRunner(PRESET_EVAL_CONFIGS)
council_router = APIRouter()
threads_router = APIRouter()
app = FastAPI(title="Language Model Council API", version="0.1.0")


@app.on_event("startup")
async def _startup():
    try:
        await ensure_indexes()
        logger.info("Mongo indexes ensured")
    except Exception as exc:  # pragma: no cover - log and keep serving
        logger.warning("Failed to ensure Mongo indexes: %s", exc)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@council_router.get("/configs", response_model=ConfigListResponse)
async def list_configs() -> ConfigListResponse:
    return ConfigListResponse(preset_keys=list(PRESET_EVAL_CONFIGS.keys()))


@council_router.post("/council/run", response_model=RunResponse)
async def run_council(payload: RunRequest) -> RunResponse:
    logger.info(
        "Starting council run with %d models, %d prompt(s), eval_config=%s",
        len(payload.models),
        len(payload.prompts),
        payload.eval_config_key,
    )
    try:
        response = await runner.run(payload)
        logger.info("Council run completed")
        return response
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - logged as server error
        logger.exception("Council run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@threads_router.post("/threads", response_model=ChatThread)
async def upsert_thread(thread: ChatThread) -> ChatThread:
    coll = await get_collection()
    now = datetime.now(timezone.utc)
    existing = await coll.find_one({"thread_id": thread.thread_id})
    created_at = existing.get("created_at") if existing else thread.created_at
    thread = thread.model_copy(update={"created_at": created_at, "updated_at": now})
    await coll.update_one(
        {"thread_id": thread.thread_id},
        {"$set": thread.model_dump()},
        upsert=True,
    )
    logger.info("Thread upserted: %s", thread.thread_id)
    return thread


@threads_router.get("/threads/{thread_id}", response_model=ChatThread)
async def get_thread(thread_id: str) -> ChatThread:
    coll = await get_collection()
    doc = await coll.find_one({"thread_id": thread_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ChatThread(**doc)


app.include_router(council_router, prefix="")
app.include_router(threads_router, prefix="")
