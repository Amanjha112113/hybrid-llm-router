"""
db/models.py

SQLAlchemy ORM models for the AI Gateway SaaS telemetry layer.

All models use Pydantic V2 for input validation / serialisation alongside
the SQLAlchemy ORM layer for persistence.  The two are deliberately kept
separate to avoid tight coupling.

Table: request_logs
  Stores every routed request with full cost + latency metadata so the
  analytics layer can compute precise savings without any in-memory state.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.pricing import compute_cost_usd, compute_savings_usd, DEFAULT_REMOTE_COST_PER_1K

# ── Cost constants (USD per 1 000 tokens) ─────────────────────────────────────
# Retained for backwards compatibility; calculations delegate to src.pricing
COST_LLAMA_8B_PER_1K = DEFAULT_REMOTE_COST_PER_1K
COST_DEEPSEEK_PER_1K = 0.0009
COST_LOCAL_PER_TOKEN = 0.0


def _compute_cost_usd(model: str, tokens: int) -> float:
    """Return the USD cost for a given model + token count using unified pricing config."""
    return compute_cost_usd(model, tokens, is_input=False)


def _compute_savings_usd(backend: str, tokens_saved: int) -> float:
    """Compute what those tokens *would* have cost on the remote model using unified pricing config."""
    return compute_savings_usd(backend, tokens_saved)


# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Model ─────────────────────────────────────────────────────────────────

class Conversation(Base):
    """
    Represents a multi-turn chat conversation session.
    
    Each conversation tracks cumulative token usage, cost, and preferences
    for intelligent routing across the entire session lifecycle.
    """
    
    __tablename__ = "conversations"
    
    # ── Primary key ────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # ── User association ───────────────────────────────────────────────────
    user_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="anonymous", index=True
    )
    
    # ── Conversation metadata ──────────────────────────────────────────────
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, default="New Conversation"
    )
    
    # ── Timestamps ─────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    
    # ── Conversation statistics ────────────────────────────────────────────
    total_messages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    
    # ── User preferences ───────────────────────────────────────────────────
    model_preference: Mapped[str] = mapped_column(
        String(32), nullable=False, default="auto"
    )  # "auto" | "local" | "remote-efficient" | "remote-power"
    
    # ── Status ─────────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active", index=True
    )  # "active" | "archived" | "deleted"
    
    # ── Composite indexes ──────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_conv_user_updated", "user_id", "updated_at"),
        Index("ix_conv_status_updated", "status", "updated_at"),
    )
    
    def __repr__(self) -> str:
        return (
            f"<Conversation id={self.id[:8]} title='{self.title[:30]}' "
            f"messages={self.total_messages} cost=${self.total_cost_usd:.4f}>"
        )


class ConversationMessage(Base):
    """
    Individual message within a conversation.
    
    Stores full routing metadata per message for analytics and debugging.
    """
    
    __tablename__ = "conversation_messages"
    
    # ── Primary key ────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # ── Foreign key ────────────────────────────────────────────────────────
    conversation_id: Mapped[str] = mapped_column(
        String(36), 
        nullable=False,
        index=True,
    )
    
    # ── Message content ────────────────────────────────────────────────────
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "user" | "assistant" | "system"
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # ── Routing metadata (for assistant messages) ──────────────────────────
    backend: Mapped[str] = mapped_column(
        String(16), nullable=True
    )  # "local" | "fireworks" | None for user messages
    
    model_used: Mapped[str] = mapped_column(
        String(128), nullable=True, default=None
    )
    
    complexity: Mapped[str] = mapped_column(
        String(20), nullable=True
    )  # "low" | "standard" | "high"
    
    route_reason: Mapped[str] = mapped_column(
        String(255), nullable=True
    )
    
    # ── Token economics ────────────────────────────────────────────────────
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_saved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    savings_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    
    # ── Performance ────────────────────────────────────────────────────────
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # ── Quality validation (legacy - will be replaced by router) ─────────────
    quality_score: Mapped[float] = mapped_column(Float, nullable=True, default=None)  # 0.0 to 1.0
    quality_passed: Mapped[bool] = mapped_column(Boolean, nullable=True, default=None)
    
    # ── Router metrics (RouteLLM - new) ────────────────────────────────────
    router_score: Mapped[float] = mapped_column(Float, nullable=True, default=None)  # Raw routing score
    router_confidence: Mapped[float] = mapped_column(Float, nullable=True, default=None)  # 0.0 to 1.0
    router_reason: Mapped[str] = mapped_column(String(512), nullable=True, default=None)  # Human-readable explanation
    
    # ── Timestamp ──────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    
    # ── Composite indexes ──────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_msg_conv_created", "conversation_id", "created_at"),
        Index("ix_msg_role", "role"),
    )
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"<ConversationMessage id={self.id[:8]} role={self.role} "
            f"backend={self.backend} preview='{preview}'>"
        )


class RequestLog(Base):
    """
    Persists every inbound routing decision with rich telemetry metadata.

    Designed for append-only write patterns; queries are handled via the
    analytics module using efficient indexed reads.
    """

    __tablename__ = "request_logs"

    # ── Primary key ────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # ── Timing ─────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    # ── Request metadata ───────────────────────────────────────────────────
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_length_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # ── Routing decision ───────────────────────────────────────────────────
    backend: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )  # "local" | "fireworks"
    model_used: Mapped[str] = mapped_column(String(128), nullable=False, default="unknown")
    escalation_path: Mapped[str] = mapped_column(
        String(32), nullable=False, default="direct"
    )  # "direct" | "escalated" | "fallback"

    # ── Token economics ─────────────────────────────────────────────────────
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_saved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    savings_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ── Latency ─────────────────────────────────────────────────────────────
    elapsed_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── Quality signals ─────────────────────────────────────────────────────
    succeeded_validation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    served_from_cache: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ── Composite indexes for analytics queries ──────────────────────────────
    __table_args__ = (
        Index("ix_rl_created_backend", "created_at", "backend"),
        Index("ix_rl_created_category", "created_at", "category"),
    )

    def __repr__(self) -> str:
        return (
            f"<RequestLog id={self.id[:8]} backend={self.backend} "
            f"tokens={self.tokens_used} cost=${self.cost_usd:.6f}>"
        )

    @classmethod
    def from_route_result(
        cls,
        *,
        task_id: str,
        prompt: str,
        category: str,
        backend: str,
        model_used: str,
        tokens_used: int,
        elapsed_ms: int,
        succeeded_validation: bool,
        attempts: int,
        served_from_cache: bool = False,
    ) -> "RequestLog":
        """
        Factory: build a ``RequestLog`` from the values returned by the
        routing endpoint.  Computes derived fields (cost, savings, tokens_saved)
        automatically so the calling code stays clean.
        """
        # Estimated tokens saved = accurate prompt token count + avg output tokens
        from src.token_utils import count_tokens
        est_tokens_saved = (
            count_tokens(prompt) + 150 if backend == "local" else 0
        )

        # Escalation path classification
        if attempts == 1:
            escalation = "direct"
        elif succeeded_validation:
            escalation = "escalated"
        else:
            escalation = "fallback"

        return cls(
            task_id=task_id,
            prompt=prompt,
            prompt_length_chars=len(prompt),
            category=category,
            backend=backend,
            model_used=model_used,
            escalation_path=escalation,
            tokens_used=tokens_used,
            tokens_saved=est_tokens_saved,
            cost_usd=_compute_cost_usd(model_used, tokens_used),
            savings_usd=_compute_savings_usd(backend, est_tokens_saved),
            elapsed_ms=elapsed_ms,
            succeeded_validation=succeeded_validation,
            attempts=attempts,
            served_from_cache=served_from_cache,
        )
