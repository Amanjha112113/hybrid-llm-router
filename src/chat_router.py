"""
src/chat_router.py

Conversational AI Agent Router.
Dynamically routes each turn in a multi-turn chat conversation between
local on-device models (Ollama Qwen 2.5 3B) and remote cloud models (Fireworks AI)
to minimize token costs while maintaining ChatGPT-level response quality.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.local_model import chat_local
from src.fireworks_client import FireworksClient, CompletionResult
from src.token_utils import count_message_tokens, count_tokens
from db.models import _compute_cost_usd, _compute_savings_usd

logger = logging.getLogger(__name__)


class ChatTurnResult(BaseModel):
    content: str
    role: str = "assistant"
    backend: str = "local"           # "local" | "fireworks"
    model_used: str = "qwen2.5:3b"
    tokens_used: int = 0
    tokens_saved: int = 0
    cost_usd: float = 0.0
    savings_usd: float = 0.0
    latency_ms: int = 0
    route_reason: str = ""
    complexity: str = "standard"     # "low" | "standard" | "high"


def _assess_turn_complexity(last_msg: str, messages_count: int) -> tuple[str, str]:
    """
    Evaluate the complexity of the current conversation turn.
    Returns (complexity_level, reasoning).
    """
    text = last_msg.strip().lower()

    # 1. Very simple greetings / conversational fillers -> Local
    greetings = {"hi", "hello", "hey", "hhi", "how are you", "who are you", "good morning", "good evening", "thanks", "thank you", "bye", "ok", "okay"}
    clean_text = re.sub(r"[^\w\s]", "", text)
    if clean_text in greetings or len(text.split()) <= 3:
        return "low", "Simple conversational turn or greeting"

    # 2. High-complexity coding / deep technical reasoning -> Remote
    code_keywords = [
        "write a python", "write code", "implement", "debug this", "refactor",
        "algorithm", "dijkstra", "binary search", "regex", "sql query", "react component",
        "dockerfile", "kubernetes", "memory leak", "stack trace", "def ", "class ", "return "
    ]
    if any(k in text for k in code_keywords):
        return "high", "Technical code synthesis / debugging detected"

    math_keywords = [
        "solve for", "calculate the probability", "integral", "derivative",
        "mathematical proof", "combinatorics", "theorem"
    ]
    if any(k in text for k in math_keywords):
        return "high", "Multi-step mathematical reasoning detected"

    if len(text) > 800:
        return "high", "High token length prompt requiring extensive context"

    # 3. Standard queries (summaries, explanations, questions, writing, general chit-chat)
    return "standard", "Standard conversational / general knowledge query"


async def route_chat_turn(
    messages: list[dict[str, str]],
    client: Optional[FireworksClient],
    allowed_models: list[str],
    mode: str = "auto",  # "auto" | "local_only" | "remote_only"
) -> ChatTurnResult:
    """
    Route a multi-turn conversation turn dynamically.
    """
    start_time = time.monotonic()
    last_msg = messages[-1].get("content", "") if messages else ""
    
    # Use accurate token counting (tiktoken with fallback)
    history_tokens = count_message_tokens(messages)

    complexity, reason = _assess_turn_complexity(last_msg, len(messages))
    local_model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

    # Tier 0 (cheap remote) and Tier 1 (escalated remote)
    tier0_model = allowed_models[0] if allowed_models else "accounts/fireworks/models/ember-1"
    tier1_model = (
        allowed_models[1]
        if len(allowed_models) > 1
        else (allowed_models[0] if allowed_models else "accounts/fireworks/models/kimi-k3")
    )

    # Decide target route
    should_try_local = mode == "local_only" or (mode == "auto" and complexity in ("low", "standard"))

    # ── PATH 1: LOCAL MODEL (Ollama Qwen 2.5 3B) ──────────────────────────────
    if should_try_local:
        try:
            logger.info("Chat Router: Attempting local resolution with %s for turn: %s", local_model_name, reason)
            local_ans = await chat_local(messages, max_tokens=700)
            if local_ans and len(local_ans.strip()) > 1:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                output_tokens = count_tokens(local_ans)
                tokens_saved = history_tokens + output_tokens
                savings_usd = _compute_savings_usd("local", tokens_saved)

                return ChatTurnResult(
                    content=local_ans,
                    role="assistant",
                    backend="local",
                    model_used=local_model_name,
                    tokens_used=0,
                    tokens_saved=tokens_saved,
                    cost_usd=0.0,
                    savings_usd=savings_usd,
                    latency_ms=elapsed_ms,
                    route_reason=f"Resolved on-device ({reason}) — 0 API tokens spent",
                    complexity=complexity,
                )
            else:
                logger.warning("Chat Router: Local model produced empty response, escalating to Fireworks.")
        except Exception as exc:
            logger.warning("Chat Router: Local execution failed (%s), escalating to Fireworks.", exc)

    # ── PATH 2: REMOTE CLOUD (Fireworks AI) ───────────────────────────────────
    if client is None:
        return ChatTurnResult(
            content="I am unable to connect to the cloud model and local model is unavailable.",
            role="assistant",
            backend="error",
            model_used="none",
            tokens_used=0,
            tokens_saved=0,
            latency_ms=int((time.monotonic() - start_time) * 1000),
            route_reason="No inference backend available",
            complexity=complexity,
        )

    # Choose model based on complexity
    chosen_model = tier1_model if complexity == "high" else tier0_model
    max_tokens = 1500 if complexity == "high" else 800

    try:
        logger.info("Chat Router: Calling Fireworks model %s (complexity: %s)", chosen_model, complexity)
        res: CompletionResult = await client.complete_chat(
            model=chosen_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            deadline=time.monotonic() + 45.0,
        )
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        cost_usd = _compute_cost_usd(chosen_model, res.total_tokens)

        return ChatTurnResult(
            content=res.text,
            role="assistant",
            backend="fireworks",
            model_used=chosen_model,
            tokens_used=res.total_tokens,
            tokens_saved=0,
            cost_usd=cost_usd,
            savings_usd=0.0,
            latency_ms=elapsed_ms,
            route_reason=f"Handled by {chosen_model} ({reason})",
            complexity=complexity,
        )
    except Exception as exc:
        logger.error("Chat Router: Remote call failed: %s", exc)
        return ChatTurnResult(
            content="Sorry, I encountered a temporary connection issue reaching the AI backend.",
            role="assistant",
            backend="error",
            model_used=chosen_model,
            tokens_used=0,
            tokens_saved=0,
            latency_ms=int((time.monotonic() - start_time) * 1000),
            route_reason=f"Remote call failed: {exc}",
            complexity=complexity,
        )
