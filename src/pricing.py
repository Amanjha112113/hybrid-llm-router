"""
src/pricing.py

Unified pricing configuration and calculation engine for AI Gateway.
Externalizes pricing from code to config/pricing.yaml.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Fallback defaults if config file is not found
DEFAULT_LOCAL_COST_PER_1K = 0.0
DEFAULT_REMOTE_COST_PER_1K = 0.0002

_FALLBACK_RATES: Dict[str, Dict[str, float]] = {
    "local_model": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "qwen2.5:3b": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "qwen2.5:0.5b": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "accounts/fireworks/models/llama-v3p1-8b-instruct": {
        "input_cost_per_1k": 0.0002,
        "output_cost_per_1k": 0.0002,
    },
    "accounts/fireworks/models/deepseek-v4-pro": {
        "input_cost_per_1k": 0.0009,
        "output_cost_per_1k": 0.0009,
    },
    "gpt-4": {"input_cost_per_1k": 0.030, "output_cost_per_1k": 0.060},
    "gpt-4-turbo": {"input_cost_per_1k": 0.010, "output_cost_per_1k": 0.030},
    "gpt-3.5-turbo": {"input_cost_per_1k": 0.0005, "output_cost_per_1k": 0.0015},
    "gpt-3.5-turbo-instruct": {"input_cost_per_1k": 0.0015, "output_cost_per_1k": 0.0020},
}

_pricing_config: Optional[Dict[str, Any]] = None


def load_pricing_config(config_path: str = "config/pricing.yaml") -> Dict[str, Any]:
    """Load pricing rules from YAML config file."""
    global _pricing_config
    if _pricing_config is not None:
        return _pricing_config

    path = Path(config_path)
    if not path.is_absolute():
        # Try relative to repo root if possible
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / config_path
        if candidate.exists():
            path = candidate

    if path.exists():
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                _pricing_config = yaml.safe_load(f) or {}
            logger.info("Loaded pricing configuration from %s", path)
            return _pricing_config
        except Exception as exc:
            logger.warning("Failed to parse %s: %s. Using internal fallback pricing.", path, exc)
    else:
        logger.debug("Pricing config %s not found, using default rates.", path)

    _pricing_config = {
        "models": _FALLBACK_RATES,
        "defaults": {
            "local_cost_per_1k": DEFAULT_LOCAL_COST_PER_1K,
            "remote_cost_per_1k": DEFAULT_REMOTE_COST_PER_1K,
        },
    }
    return _pricing_config


def get_model_rate(model: str, is_input: bool = True) -> float:
    """Return rate per token (in USD) for a given model."""
    cfg = load_pricing_config()
    models = cfg.get("models", _FALLBACK_RATES)
    defaults = cfg.get("defaults", {})
    fallback_remote = defaults.get("remote_cost_per_1k", DEFAULT_REMOTE_COST_PER_1K)
    fallback_local = defaults.get("local_cost_per_1k", DEFAULT_LOCAL_COST_PER_1K)

    # Normalize model key
    model_key = model.strip()
    rate_info = models.get(model_key)

    if not rate_info:
        # Check case-insensitive / partial match
        lower_key = model_key.lower()
        if "local" in lower_key or "qwen" in lower_key or "gguf" in lower_key:
            return fallback_local / 1000.0
        for m_name, m_rates in models.items():
            if m_name.lower() in lower_key or lower_key in m_name.lower():
                rate_info = m_rates
                break

    if rate_info:
        cost_per_1k = rate_info.get("input_cost_per_1k" if is_input else "output_cost_per_1k", fallback_remote)
        return cost_per_1k / 1000.0

    return fallback_remote / 1000.0


def compute_cost_usd(model: str, tokens: int, is_input: bool = True) -> float:
    """Return total USD cost for a given model and token count."""
    if tokens <= 0:
        return 0.0
    rate_per_token = get_model_rate(model, is_input=is_input)
    return round(rate_per_token * tokens, 8)


def compute_savings_usd(backend: str, tokens_saved: int) -> float:
    """
    Compute USD saved by resolving queries locally instead of cloud.
    Savings = what the default remote model would have cost.
    """
    if backend != "local" or tokens_saved <= 0:
        return 0.0
    cfg = load_pricing_config()
    defaults = cfg.get("defaults", {})
    remote_rate_per_1k = defaults.get("remote_cost_per_1k", DEFAULT_REMOTE_COST_PER_1K)
    return round((remote_rate_per_1k / 1000.0) * tokens_saved, 8)


def get_all_pricing() -> Dict[str, Any]:
    """Return raw pricing mapping."""
    return load_pricing_config()
