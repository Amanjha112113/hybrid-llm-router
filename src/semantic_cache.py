"""
src/semantic_cache.py

Semantic caching layer for AI Gateway router.
Caches query responses based on semantic embedding similarity rather than
exact string matching, saving tokens and latency on near-duplicate queries.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    prompt: str
    embedding: np.ndarray
    answer: str
    metadata: Dict[str, Any]
    created_at: float
    hits: int = 0


class SemanticCache:
    """
    In-memory semantic cache using cosine similarity over prompt embeddings.
    
    Attributes:
        similarity_threshold: Minimum cosine similarity to count as a cache hit (e.g. 0.94)
        max_entries: Maximum number of entries kept in memory
        ttl_seconds: Time-to-live for cache entries in seconds
    """

    def __init__(
        self,
        similarity_threshold: float = 0.94,
        max_entries: int = 1000,
        ttl_seconds: float = 3600.0 * 24,  # 24 hours
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: List[CacheEntry] = []
        self._lock = threading.Lock()
        
        # Statistics
        self.total_lookups = 0
        self.hits = 0
        self.misses = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(
        self,
        prompt_embedding: np.ndarray,
    ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """
        Check if semantically similar prompt exists in cache.

        Returns:
            Tuple of (cached_answer, metadata, similarity_score) if found, else None
        """
        now = time.time()
        best_match: Optional[CacheEntry] = None
        best_sim = -1.0

        with self._lock:
            self.total_lookups += 1
            # Evict expired entries
            self._entries = [
                e for e in self._entries
                if (now - e.created_at) < self.ttl_seconds
            ]

            for entry in self._entries:
                sim = self._cosine_similarity(prompt_embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_match = entry

            if best_match is not None and best_sim >= self.similarity_threshold:
                best_match.hits += 1
                self.hits += 1
                logger.info(
                    "Semantic cache HIT (similarity=%.3f, original='%s')",
                    best_sim,
                    best_match.prompt[:60],
                )
                return best_match.answer, dict(best_match.metadata), best_sim

            self.misses += 1
            return None

    def set(
        self,
        prompt: str,
        embedding: np.ndarray,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a prompt and its response in the semantic cache."""
        with self._lock:
            # Enforce max entries with FIFO eviction
            if len(self._entries) >= self.max_entries:
                self._entries.pop(0)

            entry = CacheEntry(
                prompt=prompt,
                embedding=np.array(embedding, dtype=np.float32),
                answer=answer,
                metadata=metadata or {},
                created_at=time.time(),
            )
            self._entries.append(entry)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and hit statistics."""
        with self._lock:
            hit_rate = (self.hits / self.total_lookups) if self.total_lookups > 0 else 0.0
            return {
                "total_lookups": self.total_lookups,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
                "cached_entries": len(self._entries),
                "similarity_threshold": self.similarity_threshold,
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()
            self.total_lookups = 0
            self.hits = 0
            self.misses = 0


_global_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Singleton getter for semantic cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SemanticCache()
    return _global_cache
