"""
src/dual_inference.py

Dual inference mode for demo/comparison purposes.
Generates answers from BOTH local and cloud models simultaneously
to prove quality equivalence.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Dict, Any

from src.token_utils import count_tokens
from src.pricing import compute_cost_usd, compute_savings_usd

logger = logging.getLogger(__name__)


async def stream_dual_comparison(
    messages: list[dict[str, str]],
    local_stream_func,
    cloud_stream_func,
    fireworks_client,
    allowed_models: list[str],
) -> AsyncIterator[dict]:
    """
    Generate answers from BOTH local and cloud models simultaneously.
    
    This is for demo/comparison purposes to prove that local model
    quality matches cloud model quality.
    
    Yields events with both responses:
    {
        "type": "dual_token" | "dual_metadata" | "dual_done",
        "local": {...},
        "cloud": {...}
    }
    
    Args:
        messages: Conversation history
        local_stream_func: Function to stream from local model
        cloud_stream_func: Function to stream from cloud model
        fireworks_client: Fireworks client instance
        allowed_models: List of allowed cloud models
    
    Yields:
        Dual comparison events
    """
    start_time = time.monotonic()
    
    # Choose efficient cloud model for comparison
    cloud_model = allowed_models[0] if allowed_models else None
    
    if not cloud_model:
        yield {
            "type": "error",
            "content": {"message": "No cloud model available for comparison"}
        }
        return
    
    # Storage for responses
    local_response = ""
    cloud_response = ""
    local_done = False
    cloud_done = False
    local_metadata = {}
    cloud_metadata = {}
    
    # Yield start metadata
    yield {
        "type": "dual_metadata",
        "local": {
            "status": "starting",
            "backend": "local",
        },
        "cloud": {
            "status": "starting",
            "backend": "cloud",
            "model": cloud_model,
        }
    }
    
    # Create both streams
    local_stream = local_stream_func(messages)
    cloud_stream = cloud_stream_func(
        messages=messages,
        model=cloud_model,
        api_key=fireworks_client.api_key,
        base_url=fireworks_client.base_url,
        max_tokens=800,
        temperature=0.3,
    )
    
    # Stream from both simultaneously using asyncio.gather
    async def consume_local():
        nonlocal local_response, local_done, local_metadata
        try:
            async for token in local_stream:
                local_response += token
                yield {
                    "type": "dual_token",
                    "source": "local",
                    "local": {"token": token},
                    "cloud": {}
                }
            
            # Calculate local stats
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            tokens_saved = count_tokens(local_response)
            
            local_metadata = {
                "backend": "local",
                "tokens_saved": tokens_saved,
                "cost_usd": 0.0,
                "savings_usd": compute_savings_usd("local", tokens_saved),
                "latency_ms": elapsed_ms,
                "full_response": local_response,
            }
            local_done = True
            
        except Exception as e:
            logger.error(f"Local stream error in dual mode: {e}")
            local_metadata = {"error": str(e)}
            local_done = True
    
    async def consume_cloud():
        nonlocal cloud_response, cloud_done, cloud_metadata
        try:
            async for token in cloud_stream:
                cloud_response += token
                yield {
                    "type": "dual_token",
                    "source": "cloud",
                    "local": {},
                    "cloud": {"token": token}
                }
            
            # Calculate cloud stats
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            tokens_used = count_tokens(cloud_response)
            cost_usd = compute_cost_usd(cloud_model, tokens_used, is_input=False)
            
            cloud_metadata = {
                "backend": "cloud",
                "model": cloud_model,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "latency_ms": elapsed_ms,
                "full_response": cloud_response,
            }
            cloud_done = True
            
        except Exception as e:
            logger.error(f"Cloud stream error in dual mode: {e}")
            cloud_metadata = {"error": str(e)}
            cloud_done = True
    
    # Interleave both streams
    local_gen = consume_local()
    cloud_gen = consume_cloud()
    
    pending = {
        asyncio.create_task(local_gen.__anext__()): "local",
        asyncio.create_task(cloud_gen.__anext__()): "cloud"
    }
    
    while pending:
        done, pending_tasks = await asyncio.wait(
            pending.keys(),
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in done:
            source = pending.pop(task)
            try:
                event = task.result()
                yield event
                
                # Create new task for this generator
                if source == "local" and not local_done:
                    new_task = asyncio.create_task(local_gen.__anext__())
                    pending[new_task] = "local"
                elif source == "cloud" and not cloud_done:
                    new_task = asyncio.create_task(cloud_gen.__anext__())
                    pending[new_task] = "cloud"
                    
            except StopAsyncIteration:
                # Generator finished
                pass
            except Exception as e:
                logger.error(f"Error in {source} dual stream: {e}")
    
    # Calculate final metrics
    local_elapsed = int((time.monotonic() - start_time) * 1000)
    cloud_elapsed = int((time.monotonic() - start_time) * 1000)
    
    local_tokens_saved = count_tokens(local_response)
    cloud_tokens_used = count_tokens(cloud_response)
    
    # Yield final comparison
    yield {
        "type": "dual_done",
        "local": {
            **local_metadata,
            "latency_ms": local_elapsed,
            "tokens_saved": local_tokens_saved,
            "savings_usd": compute_savings_usd("local", local_tokens_saved),
        },
        "cloud": {
            **cloud_metadata,
            "latency_ms": cloud_elapsed,
            "tokens_used": cloud_tokens_used,
            "cost_usd": compute_cost_usd(cloud_model, cloud_tokens_used, is_input=False),
        },
        "comparison": {
            "local_length": len(local_response),
            "cloud_length": len(cloud_response),
            "latency_diff_ms": abs(local_elapsed - cloud_elapsed),
            "cost_savings": compute_savings_usd("local", local_tokens_saved),
        },
    }
