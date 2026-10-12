"""
Accurate token counting utilities using tiktoken.
Fallback to character-based estimation if tiktoken unavailable.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed - falling back to character-based token estimation")


# Default to cl100k_base (GPT-4, GPT-3.5-turbo) encoding
_DEFAULT_ENCODING = "cl100k_base"
_encoding_cache = {}


def get_encoding(model: str = "gpt-4") -> Any:
    """
    Get tiktoken encoding for a model. Uses caching to avoid repeated initialization.
    
    Args:
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
    
    Returns:
        tiktoken.Encoding instance or None if unavailable
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    
    if model not in _encoding_cache:
        try:
            _encoding_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Model not recognized, fall back to default encoding
            logger.debug(f"Model {model} not recognized by tiktoken, using {_DEFAULT_ENCODING}")
            _encoding_cache[model] = tiktoken.get_encoding(_DEFAULT_ENCODING)
    
    return _encoding_cache[model]


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken for accuracy.
    Falls back to character-based estimation if tiktoken unavailable.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection
    
    Returns:
        Token count (integer)
    """
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        encoding = get_encoding(model)
        return len(encoding.encode(text))
    else:
        # Fallback: 1 token ≈ 4 characters (OpenAI's rule of thumb)
        return max(1, len(text) // 4)


def count_message_tokens(messages: list[dict[str, str]], model: str = "gpt-4") -> int:
    """
    Count tokens for a list of chat messages.
    Includes overhead for message formatting.
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        model: Model name for tokenizer selection
    
    Returns:
        Total token count including message formatting overhead
    """
    if not messages:
        return 0
    
    if not TIKTOKEN_AVAILABLE:
        # Fallback: sum content + small overhead per message
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4 + len(messages) * 3)
    
    encoding = get_encoding(model)
    
    # Token counting logic based on OpenAI's cookbook
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    
    tokens_per_message = 3  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = 1     # if there's a name, the role is omitted
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value:
                num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    
    return num_tokens


def estimate_cost_usd(tokens: int, model: str, is_input: bool = True) -> float:
    """
    Estimate USD cost for token usage.
    Delegates to unified src.pricing config.
    
    Args:
        tokens: Number of tokens
        model: Model name
        is_input: True for input tokens, False for output tokens
    
    Returns:
        Estimated cost in USD
    """
    from src.pricing import compute_cost_usd
    return compute_cost_usd(model=model, tokens=tokens, is_input=is_input)
