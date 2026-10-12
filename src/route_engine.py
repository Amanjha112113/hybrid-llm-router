"""
src/route_engine.py

Unified Route Decision Engine — single source of truth for ALL routing.

Both /api/route and /api/chat/stream go through this engine, eliminating
the inconsistency of having two independent routing systems.

Decision flow:
    1. Classify prompt category (regex-based)
    2. RouteLLM predict optimal backend (ML-based)
    3. Apply tier policy (category → model sequence)
    4. Apply user preference override
    5. Return unified RouteDecision
    6. Record feedback for online learning

Based on analysis:
- router.py: RouteLLM ML routing (sentence-transformers + matrix factorization)
- chat_router.py: Keyword heuristic routing (now DEPRECATED, use this instead)
- tier_policy.py: Category → model tier mapping
- escalation_controller.py: Escalation logic (remains for multi-attempt flow)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.category_classifier import classify_prompt
from src.router import RouteDecision, get_router
from src.tier_policy import TierPolicy

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRouteDecision:
    """Complete routing decision with all metadata for any endpoint."""

    # Core decision
    backend: Literal["local", "fireworks"]
    model_sequence: list[str]  # Ordered list of models to try (cheapest first)
    chosen_model: str  # First model in sequence (for direct calls)

    # Classification
    category: str  # From category_classifier
    complexity: str  # "low" | "standard" | "high"

    # Router intelligence
    router_score: Optional[float] = None  # Raw RouteLLM score
    router_confidence: Optional[float] = None  # 0.0 to 1.0
    router_reason: str = ""  # Human-readable explanation
    router_available: bool = False  # Whether ML router was used

    # User preference
    preference: str = "auto"  # "auto" | "local" | "remote-efficient" | "remote-power"
    preference_overrode_router: bool = False

    # Generation params from tier policy
    max_tokens: int = 800
    temperature: float = 0.2

    # Timing
    decision_latency_ms: int = 0

    # Educational intent
    educational_intent: bool = False
    educational_confidence: float = 0.0


class RouteDecisionEngine:
    """
    Unified routing engine combining all routing signals.

    Usage:
        engine = RouteDecisionEngine(tier_policy=tier_policy)

        # For any endpoint:
        decision = engine.decide(prompt="What is Python?", preference="auto")

        if decision.backend == "local":
            result = await local_model.chat(...)
        else:
            result = await fireworks_client.complete(model=decision.chosen_model, ...)

        # After inference completes, record outcome for online learning:
        engine.record_outcome(
            prompt="What is Python?",
            routed_to="local",
            outcome="success",  # or "fail_escalated"
        )
    """

    def __init__(
        self,
        tier_policy: Optional[TierPolicy] = None,
        allowed_models: Optional[list[str]] = None,
    ):
        self.tier_policy = tier_policy
        self.allowed_models = allowed_models or []

    def decide(
        self,
        prompt: str,
        preference: str = "auto",
        context_length: int = 0,
        conversation_history: Optional[list] = None,
    ) -> UnifiedRouteDecision:
        """
        Make a unified routing decision for any prompt.

        This is THE single entry point for all routing decisions in the system.

        Args:
            prompt: User query text
            preference: User routing preference
            context_length: Number of previous messages in conversation
            conversation_history: Optional conversation history

        Returns:
            UnifiedRouteDecision with complete routing metadata
        """
        start_time = time.monotonic()

        # ── Step 1: Classify prompt category ──────────────────────────────
        category = classify_prompt(prompt)

        # ── Step 2: Get RouteLLM prediction ───────────────────────────────
        route_decision: Optional[RouteDecision] = None
        router_available = False

        try:
            router = get_router()
            route_decision = router.predict_route(
                prompt=prompt,
                context_length=context_length,
                conversation_history=conversation_history,
            )
            router_available = True

            logger.info(
                "RouteEngine: RouteLLM prediction — backend=%s score=%.3f confidence=%.2f",
                route_decision.backend,
                route_decision.score,
                route_decision.confidence,
            )
        except Exception as e:
            logger.warning("RouteEngine: RouteLLM prediction failed: %s", e)

        # ── Step 3: Determine model sequence from tier policy ─────────────
        model_sequence = self.allowed_models[:] if self.allowed_models else []
        gen_params = {"max_tokens": 800, "temperature": 0.2}

        if self.tier_policy:
            try:
                model_sequence = self.tier_policy.sequence_for_category(category)
                gen_params = self.tier_policy.generation_params(category)
            except Exception as e:
                logger.warning("RouteEngine: Tier policy failed: %s", e)

        # ── Step 4: Combine router + category + preference ────────────────
        backend, chosen_model, preference_overrode = self._resolve_backend(
            route_decision=route_decision,
            category=category,
            model_sequence=model_sequence,
            preference=preference,
        )

        # ── Step 5: Determine complexity ──────────────────────────────────
        if route_decision:
            complexity = route_decision.complexity
        else:
            complexity = self._estimate_complexity_simple(prompt, context_length)

        # Adjust max_tokens for high complexity
        max_tokens = gen_params.get("max_tokens", 800)
        if complexity == "high" and max_tokens < 1500:
            max_tokens = 1500

        # ── Build unified decision ────────────────────────────────────────
        decision_latency_ms = int((time.monotonic() - start_time) * 1000)

        decision = UnifiedRouteDecision(
            backend=backend,
            model_sequence=model_sequence,
            chosen_model=chosen_model,
            category=category,
            complexity=complexity,
            router_score=route_decision.score if route_decision else None,
            router_confidence=route_decision.confidence if route_decision else None,
            router_reason=route_decision.reason if route_decision else "",
            router_available=router_available,
            preference=preference,
            preference_overrode_router=preference_overrode,
            max_tokens=max_tokens,
            temperature=gen_params.get("temperature", 0.2),
            decision_latency_ms=decision_latency_ms,
            educational_intent=(
                route_decision is not None
                and hasattr(route_decision, "embedding")
                and route_decision.embedding is not None
            ),
        )

        logger.info(
            "RouteEngine: decision — backend=%s model=%s category=%s "
            "complexity=%s router=%s preference_override=%s latency=%dms",
            decision.backend,
            decision.chosen_model,
            decision.category,
            decision.complexity,
            "ML" if router_available else "heuristic",
            preference_overrode,
            decision_latency_ms,
        )

        return decision

    def _resolve_backend(
        self,
        route_decision: Optional[RouteDecision],
        category: str,
        model_sequence: list[str],
        preference: str,
    ) -> tuple[Literal["local", "fireworks"], str, bool]:
        """
        Resolve the final backend + model, combining all signals.

        Priority (highest to lowest):
            1. Explicit user preference (local / remote-*)
            2. RouteLLM ML prediction (if available and trained)
            3. Category-based heuristic (fallback)

        Returns:
            (backend, chosen_model, preference_overrode_router)
        """
        # Categories that are known to work well on local
        local_capable = {
            "sentiment_classification",
            "named_entity_recognition",
            "text_summarisation",
        }

        default_model = model_sequence[0] if model_sequence else "unknown"
        power_model = model_sequence[-1] if model_sequence else default_model

        # ── Priority 1: User preference override ─────────────────────────
        if preference == "local":
            overrode = route_decision is not None and route_decision.backend != "local"
            if overrode:
                logger.info("RouteEngine: User preference overriding router → local")
            return "local", default_model, overrode

        if preference == "remote-efficient":
            overrode = route_decision is not None and route_decision.backend != "fireworks"
            return "fireworks", default_model, overrode

        if preference == "remote-power":
            overrode = route_decision is not None and route_decision.backend != "fireworks"
            return "fireworks", power_model, overrode

        # ── Priority 2: RouteLLM prediction (auto mode) ──────────────────
        if route_decision is not None:
            if route_decision.backend == "local":
                return "local", default_model, False
            else:
                # Use complexity to pick model within cloud tier
                if route_decision.complexity == "high":
                    return "fireworks", power_model, False
                else:
                    return "fireworks", default_model, False

        # ── Priority 3: Category-based fallback ──────────────────────────
        if category in local_capable:
            return "local", default_model, False
        else:
            return "fireworks", default_model, False

    def _estimate_complexity_simple(
        self, prompt: str, context_length: int
    ) -> str:
        """Simple complexity estimation when ML router is unavailable."""
        word_count = len(prompt.split())
        prompt_length = len(prompt)

        if word_count <= 10 and prompt_length < 100:
            return "low"
        elif word_count > 100 or prompt_length > 500 or context_length > 10:
            return "high"
        else:
            return "standard"

    def record_outcome(
        self,
        prompt: str,
        routed_to: str,
        outcome: str,
    ):
        """
        Record routing outcome for online learning feedback loop.

        Args:
            prompt: The query that was routed
            routed_to: Where it was actually sent ("local" | "fireworks")
            outcome: What happened:
                - "success": Model produced a valid answer
                - "fail_escalated": Local failed, had to escalate to cloud
                - "cloud_success": Cloud handled it successfully
        """
        try:
            router = get_router()

            if outcome == "fail_escalated":
                # Local failed → cloud was needed → label=1
                router.record_feedback(prompt=prompt, label=1)
                logger.debug("RouteEngine: Recorded escalation feedback for prompt")
            elif outcome == "success" and routed_to == "local":
                # Local succeeded → label=0
                router.record_feedback(prompt=prompt, label=0)
                logger.debug("RouteEngine: Recorded local success feedback")
            # Cloud success doesn't contribute to routing feedback
            # (we don't know if local would have also succeeded)

            # Check if we should retrain
            stats = router.get_stats()
            if stats.get("pending_feedback", 0) >= 100:
                logger.info("RouteEngine: Triggering incremental retrain (%d samples)", stats["pending_feedback"])
                retrained = router.retrain_incremental()
                if retrained:
                    router.save("models/router")
                    logger.info("RouteEngine: Router retrained and saved")

        except Exception as e:
            logger.warning("RouteEngine: Failed to record outcome: %s", e)


# ── Singleton engine ──────────────────────────────────────────────────────────

_engine: Optional[RouteDecisionEngine] = None


def get_route_engine(
    tier_policy: Optional[TierPolicy] = None,
    allowed_models: Optional[list[str]] = None,
) -> RouteDecisionEngine:
    """
    Get or create the global RouteDecisionEngine singleton.

    On first call, provide tier_policy and allowed_models.
    Subsequent calls return the existing instance.
    """
    global _engine
    if _engine is None:
        _engine = RouteDecisionEngine(
            tier_policy=tier_policy,
            allowed_models=allowed_models,
        )
        logger.info("RouteDecisionEngine initialized")
    return _engine
""", "toolAction": "Creating route engine", "toolSummary": "Unified route engine"}
