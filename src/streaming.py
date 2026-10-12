"""
src/streaming.py

Streaming token generators for local (Ollama) and remote (Fireworks) models.
Yields tokens as they're generated for real-time SSE delivery.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncIterator, Optional

import httpx

from src.token_utils import count_tokens
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
USE_OLLAMA = os.environ.get("USE_OLLAMA", "true").lower() in ("true", "1", "yes")


# ── Ollama Streaming ──────────────────────────────────────────────────────────


async def stream_ollama_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 800,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """
    Stream tokens from Ollama chat completion.
    
    Yields individual tokens as they're generated.
    
    Args:
        messages: OpenAI-format message list
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Yields:
        Token strings (deltas)
    
    Raises:
        Exception: If Ollama is unavailable or request fails
    """
    if not USE_OLLAMA:
        raise Exception("Ollama is disabled (USE_OLLAMA=false)")
    
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,  # Enable streaming
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    
    logger.info(f"Starting Ollama stream with {OLLAMA_MODEL}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"Ollama HTTP {response.status_code}: {error_text.decode()[:200]}")
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Ollama streaming format:
                        # {"model":"qwen2.5:3b","created_at":"...","message":{"role":"assistant","content":"token"},"done":false}
                        
                        if data.get("done"):
                            # Final message with stats
                            logger.info(
                                f"Ollama stream complete. "
                                f"eval_count={data.get('eval_count', 0)}, "
                                f"eval_duration={data.get('eval_duration', 0) / 1e9:.2f}s"
                            )
                            break
                        
                        message = data.get("message", {})
                        token = message.get("content", "")
                        
                        if token:
                            yield token
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama stream line: {line[:100]}")
                        continue
                        
    except httpx.RequestError as e:
        logger.error(f"Ollama stream request failed: {e}")
        raise Exception(f"Failed to connect to Ollama: {e}")
    except Exception as e:
        logger.error(f"Ollama stream error: {e}")
        raise


# ── Fireworks Streaming ───────────────────────────────────────────────────────


async def stream_fireworks_chat(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 800,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """
    Stream tokens from Fireworks AI chat completion.
    
    Yields individual tokens as they're generated.
    
    Args:
        messages: OpenAI-format message list
        model: Fireworks model ID
        api_key: Fireworks API key
        base_url: Fireworks base URL
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Yields:
        Token strings (deltas)
    
    Raises:
        Exception: If request fails
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Disable thinking for DeepSeek/Kimi models
    thinking_disabled = None
    if any(x in model.lower() for x in ["deepseek", "kimi"]):
        thinking_disabled = {"type": "disabled"}
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,  # Enable streaming
    }
    
    if thinking_disabled:
        payload["thinking"] = thinking_disabled
    
    logger.info(f"Starting Fireworks stream with {model}")
    
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"Fireworks HTTP {response.status_code}: {error_text.decode()[:200]}")
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # SSE format: "data: {json}"
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    
                    if line == "[DONE]":
                        logger.info("Fireworks stream complete")
                        break
                    
                    try:
                        data = json.loads(line)
                        
                        # OpenAI streaming format:
                        # {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"token"},"finish_reason":null}]}
                        
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        
                        if token:
                            yield token
                        
                        # Check for finish
                        if choices[0].get("finish_reason"):
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Fireworks stream line: {line[:100]}")
                        continue
                        
    except httpx.RequestError as e:
        logger.error(f"Fireworks stream request failed: {e}")
        raise Exception(f"Failed to connect to Fireworks API: {e}")
    except Exception as e:
        logger.error(f"Fireworks stream error: {e}")
        raise


# ── Hybrid Streaming Router ──────────────────────────────────────────────────


async def stream_chat_hybrid(
    messages: list[dict[str, str]],
    complexity: str,
    fireworks_client: Optional[object] = None,
    allowed_models: Optional[list[str]] = None,
    preference: str = "auto",
    enable_quality_check: bool = True,  # DEPRECATED - keeping for backward compat
) -> AsyncIterator[dict]:
    """
    Hybrid streaming router — now delegates to the unified RouteDecisionEngine.
    
    KEY CHANGE (v2): Uses src.route_engine.RouteDecisionEngine for routing
    decisions instead of inline routing logic.  This ensures /api/chat/stream
    and /api/route make identical decisions for the same prompt.
    
    Yields dictionaries with token data and metadata:
    {
        "type": "token" | "metadata" | "error" | "done" | "router",
        "content": "token text" or metadata dict,
        "backend": "local" | "fireworks",
        "model": "model name"
    }
    
    Args:
        messages: Conversation history
        complexity: Message complexity ("low" | "standard" | "high") [DEPRECATED]
        fireworks_client: Optional FireworksClient instance
        allowed_models: List of allowed Fireworks models
        preference: User preference ("auto" | "local" | "remote-efficient" | "remote-power")
        enable_quality_check: DEPRECATED - router predicts upfront
    
    Yields:
        Event dictionaries
    """
    from src.route_engine import get_route_engine
    
    start_time = time.monotonic()
    
    # Get user's prompt (last message)
    user_prompt = messages[-1]["content"] if messages else ""
    context_length = len(messages) - 1  # Number of previous messages
    
    # ── UNIFIED ROUTING via RouteDecisionEngine ───────────────────────────────
    engine = get_route_engine(allowed_models=allowed_models)
    decision = engine.decide(
        prompt=user_prompt,
        preference=preference,
        context_length=context_length,
    )
    
    # Yield router decision to frontend
    yield {
        "type": "router",
        "content": {
            "backend": decision.backend,
            "score": decision.router_score,
            "confidence": decision.router_confidence,
            "reason": decision.router_reason,
            "complexity": decision.complexity,
            "category": decision.category,
            "chosen_model": decision.chosen_model,
            "router_available": decision.router_available,
            "preference_overrode_router": decision.preference_overrode_router,
            "decision_latency_ms": decision.decision_latency_ms,
        }
    }
    
    should_use_local = (decision.backend == "local")
    
    # ── Path 1: Local (Ollama) ────────────────────────────────────────────────
    if should_use_local and USE_OLLAMA:
        logger.info(
            "Streaming via local %s (category=%s, complexity=%s, router=%s)",
            OLLAMA_MODEL, decision.category, decision.complexity,
            "ML" if decision.router_available else "heuristic",
        )
        
        try:
            yield {
                "type": "metadata",
                "content": {
                    "backend": "local",
                    "model": OLLAMA_MODEL,
                    "status": "starting",
                    "router_routed": decision.router_available,
                    "router_decision_honored": True,
                    "category": decision.category,
                }
            }
            
            full_response = ""
            async for token in stream_ollama_chat(messages):
                full_response += token
                yield {
                    "type": "token",
                    "content": token,
                    "backend": "local",
                    "model": OLLAMA_MODEL,
                }
            
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            est_tokens_saved = count_tokens(full_response)
            
            # Record success feedback for online learning
            engine.record_outcome(
                prompt=user_prompt,
                routed_to="local",
                outcome="success",
            )
            
            yield {
                "type": "done",
                "content": {
                    "backend": "local",
                    "model": OLLAMA_MODEL,
                    "tokens_used": 0,
                    "tokens_saved": est_tokens_saved,
                    "cost_usd": 0.0,
                    "savings_usd": round(est_tokens_saved * 0.0002 / 1000, 8),
                    "latency_ms": elapsed_ms,
                    "full_response": full_response,
                    "router_score": decision.router_score,
                    "router_confidence": decision.router_confidence,
                    "router_reason": decision.router_reason,
                }
            }
            return
            
        except Exception as e:
            logger.warning(f"Local streaming failed: {e}, escalating to Fireworks")
            
            # Record escalation feedback for online learning
            engine.record_outcome(
                prompt=user_prompt,
                routed_to="local",
                outcome="fail_escalated",
            )
            
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                error_reason = f"Ollama server not running on {OLLAMA_BASE_URL}. Start with 'ollama serve'"
                escalation_reason = "ollama_unavailable"
            else:
                error_reason = f"Local model error: {str(e)[:100]}"
                escalation_reason = "local_model_error"
            
            yield {
                "type": "metadata",
                "content": {
                    "backend": "local",
                    "status": "failed", 
                    "reason": error_reason,
                    "escalating": True,
                    "escalation_reason": escalation_reason,
                    "router_decision_overridden": decision.backend == "local",
                }
            }
    
    # ── Path 2: Remote (Fireworks) ────────────────────────────────────────────
    logger.info(
        "Streaming via Fireworks %s (category=%s, complexity=%s)",
        decision.chosen_model, decision.category, decision.complexity,
    )
    
    if fireworks_client is None or not allowed_models:
        yield {
            "type": "error", 
            "content": {
                "message": "Remote inference unavailable - no Fireworks client configured",
                "router_decision_failed": decision.backend,
            }
        }
        return
    
    chosen_model = decision.chosen_model
    max_tokens = decision.max_tokens
    
    try:
        yield {
            "type": "metadata",
            "content": {
                "backend": "fireworks",
                "model": chosen_model,
                "status": "starting",
                "router_routed": decision.router_available,
                "category": decision.category,
            }
        }
        
        # Get API credentials from client
        api_key = fireworks_client.api_key
        base_url = fireworks_client.base_url
        
        full_response = ""
        async for token in stream_fireworks_chat(
            messages=messages,
            model=chosen_model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=decision.temperature,
        ):
            full_response += token
            yield {
                "type": "token",
                "content": token,
                "backend": "fireworks",
                "model": chosen_model,
            }
        
        # Calculate costs
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        total_tokens = count_tokens(full_response)
        
        # Cost calculation (simplified - use actual token counts in production)
        cost_per_token = 0.0002 / 1000  # $0.0002 per 1K tokens
        cost_usd = round(total_tokens * cost_per_token, 8)
        
        # Record cloud success (no routing feedback — see record_outcome docs)
        engine.record_outcome(
            prompt=user_prompt,
            routed_to="fireworks",
            outcome="cloud_success",
        )
        
        yield {
            "type": "done",
            "content": {
                "backend": "fireworks",
                "model": chosen_model,
                "tokens_used": total_tokens,
                "tokens_saved": 0,
                "cost_usd": cost_usd,
                "savings_usd": 0.0,
                "latency_ms": elapsed_ms,
                "full_response": full_response,
                "router_score": decision.router_score,
                "router_confidence": decision.router_confidence,
                "router_reason": decision.router_reason,
            }
        }
        
    except Exception as e:
        logger.error(f"Remote streaming failed: {e}")
        yield {
            "type": "error",
            "content": {
                "message": f"Remote inference failed: {str(e)}"
            }
        }
