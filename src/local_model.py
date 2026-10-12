import os
import logging
import asyncio
import functools
import threading
import queue
from typing import Optional

logger = logging.getLogger(__name__)

# Robust model path resolution: check workspace local models folder first,
# then environment variable, then fallback.
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_local_workspace_path = os.path.join(_base_dir, "models", "local_model.gguf")

if os.path.exists(_local_workspace_path):
    MODEL_PATH = _local_workspace_path
else:
    MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/models/local_model.gguf")

_llm = None
_local_sem = None
_local_queue = None
_local_thread = None

def _local_worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        func, loop, fut = item
        try:
            res = func()
            if not fut.cancelled():
                loop.call_soon_threadsafe(fut.set_result, res)
        except Exception as e:
            if not fut.cancelled():
                loop.call_soon_threadsafe(fut.set_exception, e)
        finally:
            q.task_done()

def _enqueue_local_task(func):
    global _local_queue, _local_thread
    if _local_queue is None:
        _local_queue = queue.Queue()
    if _local_thread is None or not _local_thread.is_alive():
        _local_thread = threading.Thread(
            target=_local_worker,
            args=(_local_queue,),
            name="local_llm",
            daemon=True
        )
        _local_thread.start()
        
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _local_queue.put((func, loop, fut))
    return fut

def _get_local_sem():
    global _local_sem
    if _local_sem is None:
        _local_sem = asyncio.Semaphore(1)
    return _local_sem

def _get_llm():
    global _llm
    if _llm is None:
        try:
            from llama_cpp import Llama
            logger.info(f"Loading local model from {MODEL_PATH} ...")
            
            # Use small n_ctx to prevent OOM.
            _llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=int(os.environ.get("LOCAL_LLM_N_CTX", "2048")),
                n_threads=min(os.cpu_count() or 2, 4),
                verbose=False,
            )
            logger.info("Local model loaded successfully.")
        except ImportError:
            logger.error("llama_cpp not installed, local model unavailable.")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            
    return _llm

SYSTEM_PROMPTS = {
    "factual_knowledge": "Answer concisely in one sentence or phrase. No commentary.",
    "sentiment_classification": (
        "Output ONLY the exact sentiment label: 'Positive', 'Negative', or 'Neutral'. "
        "Do not add any justification, explanation, or other text.\n\n"
        "Example:\n"
        "Review: 'Oh, fantastic! The package arrived three weeks late and the item inside was shattered. I absolutely love paying for broken garbage.'\n"
        "Output: Negative"
    ),
    "text_summarisation": (
        "Summarize the text in exactly one sentence. Do not add any introductory text, commentary, or explanation.\n\n"
        "Example:\n"
        "Text: 'The human brain contains roughly 86 billion neurons. These neurons communicate through trillions of synaptic connections. Functional MRI has shown that these pathways remain plastic even into adulthood.'\n"
        "Output: The human brain's 86 billion neurons communicate via trillions of synaptic connections that remain plastic into adulthood."
    ),
    "named_entity_recognition": (
        "Extract all named entities. For each entity, output exactly one line in the format: '- [entity name]: [type]' "
        "where type is one of: Person, Organization, Location, Date, Product, Event. Output ONLY the list. No explanations.\n\n"
        "Example:\n"
        "Text: 'Google was founded by Larry Page in September 1998 in California.'\n"
        "Output:\n"
        "- Google: Organization\n"
        "- Larry Page: Person\n"
        "- September 1998: Date\n"
        "- California: Location"
    ),
}

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
USE_OLLAMA = os.environ.get("USE_OLLAMA", "true").lower() in ("true", "1", "yes")

async def _answer_ollama(prompt: str, system: str, max_tokens: int) -> Optional[str]:
    """Call local Ollama service asynchronously."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
        }
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                eval_count = data.get("eval_count", 0)
                logger.info(f"Local Ollama ({OLLAMA_MODEL}) answered successfully using {eval_count} local tokens.")
                return content
            else:
                logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text[:100]}")
                return None
    except Exception as exc:
        logger.warning(f"Ollama call failed ({exc}), will attempt GGUF fallback if available.")
        return None

async def answer_local(prompt: str, category: str) -> Optional[str]:
    """
    Generate an answer using the local LLM.
    Tries native Ollama (qwen2.5:3b) first for hardware acceleration,
    then falls back to llama_cpp with local_model.gguf.
    """
    system = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual_knowledge"])
    
    # Optimize generation length
    if category == "sentiment_classification":
        max_tokens = 15
    elif category == "text_summarisation":
        max_tokens = 120
    elif category == "named_entity_recognition":
        max_tokens = 150
    else:
        max_tokens = 100

    # 1. Primary: Try Ollama (qwen2.5:3b) with native Metal GPU acceleration
    if USE_OLLAMA:
        ollama_ans = await _answer_ollama(prompt, system, max_tokens)
        if ollama_ans:
            return ollama_ans

    # 2. Secondary: Fall back to in-process llama_cpp / GGUF model
    sem = _get_local_sem()
    async with sem:
        llm = await asyncio.to_thread(_get_llm)
        if llm is None:
            return None

        try:
            func = functools.partial(
                llm.create_chat_completion,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            
            # Enqueue and await with a timeout of 20 seconds
            out = await asyncio.wait_for(_enqueue_local_task(func), timeout=20.0)
            text = out["choices"][0]["message"]["content"].strip()
            total_tokens = out.get("usage", {}).get("total_tokens", 0)
            logger.debug(f"Local GGUF model answered. Used {total_tokens} local tokens.")
            return text
        except asyncio.TimeoutError:
            logger.warning("Local GGUF model inference timed out (exceeded 20s), falling back to API.")
            return None
        except Exception as e:
            logger.warning(f"Local GGUF model generation failed: {e}")
            return None


async def chat_local(messages: list[dict[str, str]], max_tokens: int = 800) -> Optional[str]:
    """
    Multi-turn conversational completion using the local LLM.
    Uses Ollama (qwen2.5:3b) with native GPU acceleration,
    falling back to llama_cpp with local_model.gguf if unavailable.
    """
    # 1. Primary: Try Ollama
    if USE_OLLAMA:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens,
            }
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    eval_count = data.get("eval_count", 0)
                    logger.info(f"Local Ollama chat ({OLLAMA_MODEL}) answered using {eval_count} local tokens.")
                    return content
                else:
                    logger.warning(f"Ollama chat HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as exc:
            logger.warning(f"Ollama chat failed ({exc}), attempting GGUF fallback.")

    # 2. Secondary: Fall back to llama_cpp
    sem = _get_local_sem()
    async with sem:
        llm = await asyncio.to_thread(_get_llm)
        if llm is None:
            return None

        try:
            func = functools.partial(
                llm.create_chat_completion,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            out = await asyncio.wait_for(_enqueue_local_task(func), timeout=30.0)
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Local GGUF chat failed: {e}")
            return None


def get_local_model_status() -> dict:
    """Return runtime metadata about the local model configuration."""
    return {
        "primary_backend": "ollama",
        "primary_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
        "use_ollama": USE_OLLAMA,
        "fallback_gguf_path": MODEL_PATH,
        "fallback_gguf_exists": os.path.exists(MODEL_PATH),
    }


async def check_local_model_health() -> dict:
    """Active health check probe for the local inference backend."""
    status_info = get_local_model_status()
    ollama_reachable = False
    
    if USE_OLLAMA:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")
                ollama_reachable = res.status_code == 200
        except Exception:
            ollama_reachable = False

    status_info["ollama_reachable"] = ollama_reachable
    status_info["status"] = (
        "ready" if (ollama_reachable or status_info["fallback_gguf_exists"]) else "unavailable"
    )
    return status_info


