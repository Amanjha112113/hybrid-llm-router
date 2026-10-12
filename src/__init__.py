"""
src package — Hybrid Token-Efficient Routing Agent

Public component exports for AI Gateway SaaS.
"""

from src.circuit_breaker import CircuitBreaker, get_circuit_breaker
from src.escalation_controller import process_task, TaskResult
from src.fireworks_client import FireworksClient
from src.pricing import compute_cost_usd, compute_savings_usd, get_all_pricing
from src.request_id_middleware import RequestIDMiddleware
from src.route_engine import RouteDecisionEngine, get_route_engine
from src.router import RouteDecision, RouteLLMRouter
from src.semantic_cache import SemanticCache, get_semantic_cache
from src.tier_policy import TierPolicy
from src.token_utils import count_tokens, count_message_tokens, estimate_cost_usd

__all__ = [
    "CircuitBreaker",
    "FireworksClient",
    "RequestIDMiddleware",
    "RouteDecision",
    "RouteDecisionEngine",
    "RouteLLMRouter",
    "SemanticCache",
    "TaskResult",
    "TierPolicy",
    "compute_cost_usd",
    "compute_savings_usd",
    "count_message_tokens",
    "count_tokens",
    "estimate_cost_usd",
    "get_all_pricing",
    "get_circuit_breaker",
    "get_route_engine",
    "get_semantic_cache",
    "process_task",
]
