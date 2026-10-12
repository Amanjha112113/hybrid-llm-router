"""
server.py

AI Gateway SaaS — FastAPI application entry point.

Bug Fixes applied here:
  - FIX 1: dotenv is loaded FIRST, before any env-var-dependent imports.
    The original crash (ValueError: FIREWORKS_API_KEY is required) happened
    because FireworksClient was instantiated at module level before the shell
    had exported the vars — and uvicorn doesn't source .env automatically.
  - FIX 2: FireworksClient and TierPolicy are constructed inside lifespan()
    (deferred init), not at module-top-level, so startup errors are caught
    cleanly with a meaningful log message instead of a bare ImportError.

Phase 1 additions:
  - Async SQLAlchemy database initialised via lifespan context
  - Every /api/route call fires a BackgroundTask to persist telemetry
  - New /api/analytics/summary endpoint with all dashboard KPIs
  - /api/stats preserved for backward compatibility
  - All schemas upgraded to strict Pydantic V2 models
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

# ── CRITICAL: load .env BEFORE any env-var-dependent imports ──────────────────
# uvicorn does not source .env automatically; without this, FIREWORKS_API_KEY
# is empty at module-load time and FireworksClient raises ValueError on init.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.escalation_controller import process_task, TaskResult
from src.fireworks_client import FireworksClient
from src.tier_policy import TierPolicy
from src.auth_middleware import AuthRateLimitMiddleware
from src.category_classifier import classify_prompt
from src.task_loader import Task
from src.route_engine import RouteDecisionEngine, get_route_engine

# ── DB / Telemetry imports (Phase 1) ──────────────────────────────────────────
from db.engine import init_db, close_db, get_db_session
from db.telemetry import log_request_to_db
from db.analytics import get_summary_stats, DashboardStats

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("server")


# ── Module-level client placeholders (populated in lifespan) ─────────────────
# FIX 2: Do NOT construct FireworksClient/TierPolicy here.
# At import time the env vars may not yet be in os.environ (dotenv loads async,
# shells don't always export).  We defer to lifespan() so errors are caught and
# logged instead of causing a bare module-level crash.
client: FireworksClient | None = None
tier_policy: TierPolicy | None = None
allowed_models: list[str] = []


# ── Lifespan: DB init + client construction + teardown ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
      1. Initialise the async SQLAlchemy database + create tables if missing.
      2. Construct FireworksClient and TierPolicy from env vars (after dotenv
         has been sourced).
    Shutdown:
      1. Dispose the DB engine.
      2. Close the Fireworks HTTP client connection pool.
    """
    global client, tier_policy, allowed_models

    # ── Database ──────────────────────────────────────────────────────────────
    logger.info("Gateway starting up — initialising database …")
    await init_db()
    logger.info("Database tables ready.")

    # ── Fireworks client (FIX 2: deferred init) ───────────────────────────────
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models_raw = os.environ.get(
        "ALLOWED_MODELS",
        "accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/deepseek-v4-pro",
    )
    allowed_models = [m.strip() for m in allowed_models_raw.split(",") if m.strip()]

    if not api_key or not base_url:
        logger.warning(
            "FIREWORKS_API_KEY or FIREWORKS_BASE_URL not set — "
            "remote API calls will fail.  Local-only routing will still work."
        )

    try:
        client = FireworksClient(api_key=api_key, base_url=base_url)
        tier_policy = TierPolicy(
            allowed_models=allowed_models, config_path="config/tier_mapping.yaml"
        )
        logger.info("Fireworks client ready. Models: %s", allowed_models)
    except Exception as exc:
        logger.error("Failed to initialise Fireworks client: %s", exc)
        # Don't crash — let the server start; /api/route will return 503
        # for remote-bound requests while local routing still works.

    # ── Unified Route Engine (Phase 2: single routing source of truth) ────────
    try:
        engine = get_route_engine(
            tier_policy=tier_policy,
            allowed_models=allowed_models,
        )
        logger.info("RouteDecisionEngine initialized")
    except Exception as exc:
        logger.error("Failed to initialise RouteDecisionEngine: %s", exc)
        # Non-fatal — endpoints will fall back to inline routing
        # for remote-bound requests while local routing still works.

    logger.info("Gateway startup complete.")
    yield

    # ── Teardown ──────────────────────────────────────────────────────────────
    logger.info("Gateway shutting down …")
    await close_db()
    if client is not None:
        await client.close()
    logger.info("Shutdown complete.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Gateway SaaS",
    description="Hybrid Token-Efficient Routing Agent — production-grade API Gateway",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Authentication & Rate Limiting (P0 Security) ──────────────────────────────
# Add authentication middleware to protect endpoints from unauthorized access
# Configure with VALID_API_KEYS and RATE_LIMIT_PER_MINUTE environment variables
# Set ALLOW_ANONYMOUS=false to require authentication
app.add_middleware(AuthRateLimitMiddleware)



# ── In-memory fallback stats (preserved for /api/stats backward compat) ───────
_mem_stats: dict[str, Any] = {
    "total_requests": 0,
    "local_count": 0,
    "remote_count": 0,
    "tokens_saved": 0,
    "tokens_used": 0,
    "history": [],
}


# ── Pydantic V2 schemas ───────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    """Inbound request payload for the /api/route endpoint."""

    model_config = {"str_strip_whitespace": True}

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="The user prompt to route through the hybrid inference pipeline.",
    )


class RouteResponse(BaseModel):
    """
    Outbound response from the /api/route endpoint.

    Phase 1 additions vs original:
      - ``served_from_cache`` flag (always False in Phase 1; populated in Phase 2)
      - ``log_id``            UUID of the persisted telemetry row
      - ``cost_usd``          precise USD cost for this request
      - ``savings_usd``       USD saved by routing locally
      - ``elapsed_ms``        latency in milliseconds (not just seconds)
    """

    task_id: str
    answer: str
    category: str
    backend: str
    model_used: str
    tokens_used: int
    succeeded_validation: bool
    elapsed_seconds: float
    elapsed_ms: int
    served_from_cache: bool = False
    log_id: str | None = None
    cost_usd: float = 0.0
    savings_usd: float = 0.0
    tokens_saved: int = 0


# ── Route: primary inference endpoint ─────────────────────────────────────────

@app.post("/api/route", response_model=RouteResponse, tags=["Inference"])
async def route_query(req: RouteRequest, background_tasks: BackgroundTasks) -> RouteResponse:
    """
    Route an incoming prompt through the hybrid inference pipeline.

    Flow:
      1. Classify prompt category
      2. Attempt local inference (Qwen)
      3. Escalate to Fireworks API if local fails or lacks confidence
      4. Schedule a background DB write (telemetry) — zero latency impact
      5. Return the response immediately

    The ``BackgroundTasks`` dependency is injected by FastAPI; the telemetry
    write runs *after* the HTTP response has been sent to the client.
    """
    global client, tier_policy, allowed_models
    start_time = time.monotonic()

    # ── Unified routing decision ──────────────────────────────────────────────
    engine = get_route_engine(tier_policy=tier_policy, allowed_models=allowed_models)
    route = engine.decide(prompt=req.prompt, preference="auto")
    category = route.category

    task_id = f"web_{int(time.time() * 1000)}"
    task = Task(task_id=task_id, prompt=req.prompt)

    # Backend label from unified engine (may be corrected after process_task returns)
    local_capable = {"sentiment_classification", "named_entity_recognition", "text_summarisation"}
    backend = "local" if category in local_capable else "fireworks"

    # Ensure client is initialized if possible
    if client is None or tier_policy is None:
        api_key = os.environ.get("FIREWORKS_API_KEY", "")
        base_url = os.environ.get("FIREWORKS_BASE_URL", "")
        allowed_models_raw = os.environ.get(
            "ALLOWED_MODELS",
            "accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/deepseek-v4-pro",
        )
        allowed_models = [m.strip() for m in allowed_models_raw.split(",") if m.strip()]
        if api_key and base_url:
            try:
                client = FireworksClient(api_key=api_key, base_url=base_url)
                tier_policy = TierPolicy(
                    allowed_models=allowed_models, config_path="config/tier_mapping.yaml"
                )
            except Exception as e:
                logger.warning(f"Lazy FireworksClient init failed: {e}")

    # If category is local-capable, we can attempt local inference even without remote client
    if category in local_capable:
        from src.local_model import answer_local
        from src.answer_validator import validate_tier_a
        from src.fireworks_client import CompletionResult

        local_ans = await answer_local(req.prompt, category)
        if local_ans:
            mock_res = CompletionResult(text=local_ans, total_tokens=0, model="local_model", finish_reason="stop")
            is_valid, _ = validate_tier_a(mock_res, category, prompt=req.prompt)
            if is_valid:
                elapsed = time.monotonic() - start_time
                elapsed_ms = int(elapsed * 1000)
                backend = "local"
                model_used = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b") if os.environ.get("USE_OLLAMA", "true").lower() in ("true", "1") else "local_model"
                est_tokens_saved = int(len(req.prompt) / 4) + 150
                from db.models import _compute_savings_usd
                savings_usd = _compute_savings_usd(backend, est_tokens_saved)

                background_tasks.add_task(
                    log_request_to_db,
                    task_id=task_id,
                    prompt=req.prompt,
                    category=category,
                    backend=backend,
                    model_used=model_used,
                    tokens_used=0,
                    elapsed_ms=elapsed_ms,
                    succeeded_validation=True,
                    attempts=1,
                    served_from_cache=False,
                )


                _mem_stats["total_requests"] += 1
                _mem_stats["local_count"] += 1
                _mem_stats["tokens_saved"] += est_tokens_saved
                _mem_stats["history"].insert(0, {
                    "prompt": req.prompt[:80],
                    "category": category,
                    "backend": backend,
                    "elapsed_seconds": round(elapsed, 3),
                })
                if len(_mem_stats["history"]) > 20:
                    _mem_stats["history"].pop()

                return RouteResponse(
                    task_id=task_id,
                    prompt=req.prompt,
                    category=category,
                    backend=backend,
                    model_used=model_used,
                    tokens_used=0,
                    tokens_saved=est_tokens_saved,
                    cost_usd=0.0,
                    savings_usd=savings_usd,
                    elapsed_seconds=round(elapsed, 4),
                    elapsed_ms=elapsed_ms,
                    succeeded_validation=True,
                    answer=local_ans,
                )


    # For remote-bound or escalated requests, FireworksClient must be active
    if client is None or tier_policy is None:
        raise HTTPException(
            status_code=503,
            detail="Remote inference unavailable — FIREWORKS_API_KEY / FIREWORKS_BASE_URL not set and query could not be resolved locally."
        )


    try:
        res: TaskResult = await process_task(
            task=task,
            client=client,
            tier_policy=tier_policy,
            deadline=start_time + 120.0,
        )
    except Exception as exc:
        logger.exception("Inference pipeline failed for task %s", task_id)
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {exc}") from exc

    elapsed = time.monotonic() - start_time
    elapsed_ms = int(elapsed * 1000)

    # Correct backend label from actual token usage
    if res.tokens_used == 0 and res.attempts <= 1 and category in local_capable:
        backend = "local"
        model_used = "local_model"
    else:
        backend = "fireworks"
        # Infer which Fireworks model was used (first in the allowed list if single attempt,
        # second if escalated)
        model_used = (
            allowed_models[0]
            if res.attempts <= 1 and len(allowed_models) > 0
            else (allowed_models[1] if len(allowed_models) > 1 else allowed_models[0])
        )

    # Compute derived economics for the response payload
    from db.models import _compute_cost_usd, _compute_savings_usd
    cost_usd = _compute_cost_usd(model_used, res.tokens_used)
    est_tokens_saved = int(len(req.prompt) / 4) + 150 if backend == "local" else 0
    savings_usd = _compute_savings_usd(backend, est_tokens_saved)

    # ── Background telemetry write — fires after response is sent ────────────
    background_tasks.add_task(
        log_request_to_db,
        task_id=task_id,
        prompt=req.prompt,
        category=category,
        backend=backend,
        model_used=model_used,
        tokens_used=res.tokens_used,
        elapsed_ms=elapsed_ms,
        succeeded_validation=res.succeeded_validation,
        attempts=res.attempts,
        served_from_cache=False,  # Phase 2 will set this
    )

    # ── Update in-memory fallback stats ──────────────────────────────────────
    _mem_stats["total_requests"] += 1
    if backend == "local":
        _mem_stats["local_count"] += 1
        _mem_stats["tokens_saved"] += est_tokens_saved
    else:
        _mem_stats["remote_count"] += 1
        _mem_stats["tokens_used"] += res.tokens_used

    log_item = {
        "timestamp": time.time(),
        "prompt": req.prompt,
        "category": category,
        "backend": backend,
        "tokens_used": res.tokens_used,
        "succeeded_validation": res.succeeded_validation,
        "elapsed_seconds": round(elapsed, 2),
    }
    _mem_stats["history"].insert(0, log_item)
    if len(_mem_stats["history"]) > 20:
        _mem_stats["history"].pop()

    return RouteResponse(
        task_id=task_id,
        answer=res.answer,
        category=category,
        backend=backend,
        model_used=model_used,
        tokens_used=res.tokens_used,
        succeeded_validation=res.succeeded_validation,
        elapsed_seconds=round(elapsed, 4),
        elapsed_ms=elapsed_ms,
        served_from_cache=False,
        cost_usd=cost_usd,
        savings_usd=savings_usd,
        tokens_saved=est_tokens_saved,
    )



# ── Route: rich analytics (Phase 1) ──────────────────────────────────────────

@app.get(
    "/api/analytics/summary",
    response_model=DashboardStats,
    tags=["Analytics"],
    summary="Full dashboard analytics — all KPIs in one call",
)
async def get_analytics_summary() -> DashboardStats:
    """
    Returns all dashboard KPIs computed directly from the persisted
    ``request_logs`` table:

    - **route_split**:        Local vs Remote counts + percentages + cache hits
    - **cost_summary**:       Total spent / saved, theoretical all-remote cost,
                              savings rate %, avg cost per request
    - **latency**:            p50/p95 latency split by backend
    - **hourly_buckets**:     Last 24 h histogram for the time-series chart
    - **category_breakdown**: Cost + count + success rate per prompt category
    - **recent_logs**:        Last 50 request log entries for the activity feed
    """
    try:
        async with get_db_session() as session:
            return await get_summary_stats(session)
    except Exception as exc:
        logger.exception("Analytics query failed")
        raise HTTPException(status_code=500, detail=f"Analytics query failed: {exc}") from exc


# ── Route: backward-compat stats ─────────────────────────────────────────────

@app.get("/api/stats", tags=["Analytics"], summary="Legacy in-memory stats (backward compat)")
async def get_stats() -> dict:
    """
    Preserved for backward compatibility with the original React frontend.
    Returns the same in-memory stats shape as the v1 server.

    Prefer /api/analytics/summary for new dashboard work.
    """
    return _mem_stats


# ── Conversation Management API (Phase 1) ─────────────────────────────────────

from sse_starlette.sse import EventSourceResponse
from src.streaming import stream_chat_hybrid, stream_ollama_chat, stream_fireworks_chat
from src.dual_inference import stream_dual_comparison
from src.chat_router import _assess_turn_complexity
from db.conversations import (
    create_conversation,
    get_conversation,
    list_conversations,
    update_conversation,
    delete_conversation,
    search_conversations,
    get_conversation_messages,
    get_conversation_stats,
    add_message,
    get_conversation_history_for_llm,
)
from db.models import Conversation as ConversationModel, ConversationMessage as MessageModel


# ── SSE Streaming Chat API (Phase 1) ──────────────────────────────────────────


class StreamChatRequest(BaseModel):
    """Request to stream a chat message."""
    conversation_id: str = Field(..., description="Conversation ID to add message to")
    prompt: str = Field(..., min_length=1, max_length=32000, description="User message")
    comparison_mode: bool = Field(default=False, description="Enable dual local/cloud comparison")


@app.post("/api/chat/stream", tags=["Chat Streaming"])
async def stream_chat_endpoint(req: StreamChatRequest):
    """
    Stream a chat completion with Server-Sent Events (SSE).
    
    This endpoint:
    1. Adds the user message to the conversation
    2. Retrieves conversation history
    3. Determines routing complexity
    4. Streams tokens in real-time (local or remote)
    5. Saves the assistant response when complete
    
    SSE Event Types:
    - "token": Individual token delta
    - "metadata": Backend info, model selection
    - "done": Final stats (tokens, cost, latency)
    - "error": Error message
    
    Returns:
        EventSource stream
    """
    global client, tier_policy, allowed_models
    
    async def event_generator():
        conversation_id = req.conversation_id
        user_message = req.prompt
        
        try:
            # Verify conversation exists
            async with get_db_session() as session:
                conv = await get_conversation(session, conversation_id)
                if not conv:
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": f"Conversation {conversation_id} not found"})
                    }
                    return
                
                # Add user message to database
                await add_message(
                    session=session,
                    conversation_id=conversation_id,
                    role="user",
                    content=user_message,
                )
                
                # Get conversation history for LLM
                messages = await get_conversation_history_for_llm(
                    session=session,
                    conversation_id=conversation_id,
                    max_messages=20,  # Keep last 20 messages for context
                )
            
            # Assess complexity
            complexity, reason = _assess_turn_complexity(user_message, len(messages))
            
            yield {
                "event": "metadata",
                "data": json.dumps({
                    "complexity": complexity,
                    "reason": reason,
                    "total_messages": len(messages),
                    "comparison_mode": req.comparison_mode,
                })
            }
            
            # ── COMPARISON MODE: Dual local/cloud inference ────────────────────
            if req.comparison_mode and client and allowed_models:
                logger.info("Starting dual comparison mode")
                
                local_response = ""
                cloud_response = ""
                local_meta = {}
                cloud_meta = {}
                
                async for event in stream_dual_comparison(
                    messages=messages,
                    local_stream_func=stream_ollama_chat,
                    cloud_stream_func=stream_fireworks_chat,
                    fireworks_client=client,
                    allowed_models=allowed_models,
                ):
                    event_type = event.get("type")
                    
                    if event_type == "dual_token":
                        source = event.get("source")
                        local_data = event.get("local", {})
                        cloud_data = event.get("cloud", {})
                        
                        if source == "local" and local_data.get("token"):
                            local_response += local_data["token"]
                        if source == "cloud" and cloud_data.get("token"):
                            cloud_response += cloud_data["token"]
                        
                        yield {
                            "event": "dual_token",
                            "data": json.dumps({
                                "source": source,
                                "local": local_data,
                                "cloud": cloud_data,
                            })
                        }
                    
                    elif event_type == "dual_metadata":
                        yield {
                            "event": "dual_metadata",
                            "data": json.dumps(event)
                        }
                    
                    elif event_type == "dual_done":
                        local_meta = event.get("local", {})
                        cloud_meta = event.get("cloud", {})
                        comparison = event.get("comparison", {})
                        
                        yield {
                            "event": "dual_done",
                            "data": json.dumps({
                                "local": local_meta,
                                "cloud": cloud_meta,
                                "comparison": comparison,
                            })
                        }
                
                # Save BOTH responses to database
                async with get_db_session() as session:
                    # Save local response
                    await add_message(
                        session=session,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=f"[LOCAL MODEL]\n{local_response}",
                        backend="local",
                        model_used=local_meta.get("model", "qwen2.5:3b"),
                        complexity=complexity,
                        route_reason="Comparison mode",
                        tokens_used=0,
                        tokens_saved=local_meta.get("tokens_saved", 0),
                        cost_usd=0.0,
                        savings_usd=local_meta.get("savings_usd", 0.0),
                        latency_ms=local_meta.get("latency_ms", 0),
                        quality_score=local_meta.get("quality_score"),
                        quality_passed=local_meta.get("quality_passed"),
                    )
                    
                    # Save cloud response
                    await add_message(
                        session=session,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=f"[CLOUD MODEL]\n{cloud_response}",
                        backend="fireworks",
                        model_used=cloud_meta.get("model", "unknown"),
                        complexity=complexity,
                        route_reason="Comparison mode",
                        tokens_used=cloud_meta.get("tokens_used", 0),
                        tokens_saved=0,
                        cost_usd=cloud_meta.get("cost_usd", 0.0),
                        savings_usd=0.0,
                        latency_ms=cloud_meta.get("latency_ms", 0),
                        quality_score=cloud_meta.get("quality_score"),
                        quality_passed=cloud_meta.get("quality_passed"),
                    )
                
                return  # Exit after comparison mode
            
            # ── NORMAL MODE: Single hybrid inference ───────────────────────────
            # Stream tokens
            full_response = ""
            backend_used = None
            model_used = None
            metadata = {}
            
            async for event in stream_chat_hybrid(
                messages=messages,
                complexity=complexity,
                fireworks_client=client,
                allowed_models=allowed_models,
                preference=conv.model_preference,
            ):
                event_type = event.get("type")
                content = event.get("content")
                
                if event_type == "token":
                    full_response += content
                    backend_used = event.get("backend")
                    model_used = event.get("model")
                    
                    yield {
                        "event": "token",
                        "data": json.dumps({
                            "token": content,
                            "backend": backend_used,
                        })
                    }
                    
                elif event_type == "metadata":
                    yield {
                        "event": "metadata",
                        "data": json.dumps(content)
                    }
                    if "backend" in content:
                        backend_used = content["backend"]
                    if "model" in content:
                        model_used = content["model"]
                
                elif event_type == "router":
                    # Forward router decision to frontend
                    yield {
                        "event": "router",
                        "data": json.dumps(content)
                    }
                
                elif event_type == "quality":
                    # Forward quality validation scores to frontend (LEGACY)
                    yield {
                        "event": "quality",
                        "data": json.dumps(content)
                    }
                    
                elif event_type == "done":
                    metadata = content
                    yield {
                        "event": "done",
                        "data": json.dumps(content)
                    }
                    
                elif event_type == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps(content)
                    }
                    return
            
            # Save assistant message to database
            async with get_db_session() as session:
                await add_message(
                    session=session,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response,
                    backend=backend_used or "unknown",
                    model_used=model_used or "unknown",
                    complexity=complexity,
                    route_reason=reason,
                    tokens_used=metadata.get("tokens_used", 0),
                    tokens_saved=metadata.get("tokens_saved", 0),
                    cost_usd=metadata.get("cost_usd", 0.0),
                    savings_usd=metadata.get("savings_usd", 0.0),
                    latency_ms=metadata.get("latency_ms", 0),
                    quality_score=metadata.get("quality_score"),
                    quality_passed=metadata.get("quality_passed"),
                    router_score=metadata.get("router_score"),
                    router_confidence=metadata.get("router_confidence"),
                    router_reason=metadata.get("router_reason"),
                )
            
        except Exception as e:
            logger.exception("Stream error")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }
    
    return EventSourceResponse(event_generator())


# ── Pydantic Models for Conversation API ──────────────────────────────────────


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    title: str = Field(default="New Conversation", max_length=255)
    model_preference: str = Field(default="auto", pattern="^(auto|local|remote-efficient|remote-power)$")


class UpdateConversationRequest(BaseModel):
    """Request to update conversation metadata."""
    title: str | None = Field(default=None, max_length=255)
    model_preference: str | None = Field(default=None, pattern="^(auto|local|remote-efficient|remote-power)$")
    status: str | None = Field(default=None, pattern="^(active|archived|deleted)$")


class ConversationResponse(BaseModel):
    """Conversation response model."""
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    total_messages: int
    total_tokens_used: int
    total_cost_usd: float
    model_preference: str
    status: str


class MessageResponse(BaseModel):
    """Message response model."""
    id: str
    conversation_id: str
    role: str
    content: str
    backend: str | None
    model_used: str | None
    complexity: str | None
    route_reason: str | None
    tokens_used: int
    tokens_saved: int
    cost_usd: float
    savings_usd: float
    latency_ms: int
    created_at: str


@app.post("/api/conversations", response_model=ConversationResponse, tags=["Conversations"])
async def create_new_conversation(req: CreateConversationRequest) -> ConversationResponse:
    """
    Create a new conversation session.
    
    Returns the newly created conversation with unique ID.
    """
    async with get_db_session() as session:
        conv = await create_conversation(
            session=session,
            user_id="anonymous",  # TODO: Replace with actual user_id from auth
            title=req.title,
            model_preference=req.model_preference,
        )
        
        return ConversationResponse(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            total_messages=conv.total_messages,
            total_tokens_used=conv.total_tokens_used,
            total_cost_usd=conv.total_cost_usd,
            model_preference=conv.model_preference,
            status=conv.status,
        )


@app.get("/api/conversations", response_model=list[ConversationResponse], tags=["Conversations"])
async def list_user_conversations(
    status: str = "active",
    limit: int = 100,
    offset: int = 0,
) -> list[ConversationResponse]:
    """
    List all conversations for the current user.
    
    Query params:
    - status: Filter by status (active | archived | deleted)
    - limit: Max results (default 100)
    - offset: Pagination offset
    """
    async with get_db_session() as session:
        convs = await list_conversations(
            session=session,
            user_id="anonymous",  # TODO: Replace with actual user_id from auth
            status=status,
            limit=limit,
            offset=offset,
        )
        
        return [
            ConversationResponse(
                id=c.id,
                user_id=c.user_id,
                title=c.title,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
                total_messages=c.total_messages,
                total_tokens_used=c.total_tokens_used,
                total_cost_usd=c.total_cost_usd,
                model_preference=c.model_preference,
                status=c.status,
            )
            for c in convs
        ]


@app.get("/api/conversations/search", response_model=list[ConversationResponse], tags=["Conversations"])
async def search_user_conversations(
    q: str,
    limit: int = 20,
) -> list[ConversationResponse]:
    """
    Search conversations by title (case-insensitive).
    
    Query params:
    - q: Search query string
    - limit: Max results (default 20)
    """
    async with get_db_session() as session:
        convs = await search_conversations(
            session=session,
            user_id="anonymous",  # TODO: Replace with actual user_id from auth
            query=q,
            limit=limit,
        )
        
        return [
            ConversationResponse(
                id=c.id,
                user_id=c.user_id,
                title=c.title,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
                total_messages=c.total_messages,
                total_tokens_used=c.total_tokens_used,
                total_cost_usd=c.total_cost_usd,
                model_preference=c.model_preference,
                status=c.status,
            )
            for c in convs
        ]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse, tags=["Conversations"])
async def get_conversation_by_id(conversation_id: str) -> ConversationResponse:
    """
    Get a specific conversation by ID.
    """
    async with get_db_session() as session:
        conv = await get_conversation(session=session, conversation_id=conversation_id)
        
        if not conv:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        return ConversationResponse(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            total_messages=conv.total_messages,
            total_tokens_used=conv.total_tokens_used,
            total_cost_usd=conv.total_cost_usd,
            model_preference=conv.model_preference,
            status=conv.status,
        )


@app.patch("/api/conversations/{conversation_id}", response_model=ConversationResponse, tags=["Conversations"])
async def update_conversation_by_id(
    conversation_id: str,
    req: UpdateConversationRequest,
) -> ConversationResponse:
    """
    Update conversation metadata (title, preference, status).
    """
    async with get_db_session() as session:
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        
        conv = await update_conversation(
            session=session,
            conversation_id=conversation_id,
            **updates,
        )
        
        if not conv:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        return ConversationResponse(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            total_messages=conv.total_messages,
            total_tokens_used=conv.total_tokens_used,
            total_cost_usd=conv.total_cost_usd,
            model_preference=conv.model_preference,
            status=conv.status,
        )


@app.delete("/api/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conversation_by_id(
    conversation_id: str,
    hard_delete: bool = False,
) -> dict[str, str]:
    """
    Delete a conversation (soft delete by default).
    
    Query params:
    - hard_delete: If true, permanently remove from database
    """
    async with get_db_session() as session:
        success = await delete_conversation(
            session=session,
            conversation_id=conversation_id,
            soft_delete=not hard_delete,
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageResponse], tags=["Conversations"])
async def get_conversation_messages_endpoint(
    conversation_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[MessageResponse]:
    """
    Get all messages in a conversation, ordered chronologically.
    
    Query params:
    - limit: Max messages to return (None = all)
    - offset: Pagination offset
    """
    async with get_db_session() as session:
        messages = await get_conversation_messages(
            session=session,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )
        
        return [
            MessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                backend=m.backend,
                model_used=m.model_used,
                complexity=m.complexity,
                route_reason=m.route_reason,
                tokens_used=m.tokens_used,
                tokens_saved=m.tokens_saved,
                cost_usd=m.cost_usd,
                savings_usd=m.savings_usd,
                latency_ms=m.latency_ms,
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ]


@app.get("/api/conversations/stats/summary", tags=["Conversations"])
async def get_conversation_statistics() -> dict:
    """
    Get aggregate statistics for user's conversations.
    
    Returns:
    - total_conversations
    - total_messages
    - avg_messages_per_conversation
    - total_cost_usd
    - avg_cost_per_conversation
    """
    async with get_db_session() as session:
        stats = await get_conversation_stats(
            session=session,
            user_id="anonymous",  # TODO: Replace with actual user_id from auth
        )
        return stats


# ── Route: health check ───────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Liveness probe — returns 200 OK and model status."""
    from src.circuit_breaker import get_all_breakers_status
    from src.local_model import get_local_model_status
    return {
        "status": "ok",
        "version": "2.0.0",
        "local_model": get_local_model_status(),
        "remote_api": "ready" if client is not None else "offline",
        "circuit_breakers": get_all_breakers_status(),
    }

