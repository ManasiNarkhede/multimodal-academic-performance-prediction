"""In-memory embedding cache with TTL and LRU eviction."""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np


class EmbeddingCache:
    """Thread-safe LRU cache for student embedding vectors with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries before LRU eviction.
            ttl_seconds: Time-to-live for each cache entry in seconds.
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[int, tuple[np.ndarray, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, student_id: int) -> Optional[np.ndarray]:
        """Retrieve an embedding by student_id.

        Args:
            student_id: The student identifier.

        Returns:
            The cached numpy array, or None if not found or expired.
        """
        with self._lock:
            if student_id not in self._cache:
                self._misses += 1
                return None

            embedding, timestamp = self._cache[student_id]
            if time.time() - timestamp > self._ttl_seconds:
                del self._cache[student_id]
                self._misses += 1
                return None

            self._cache.move_to_end(student_id)
            self._hits += 1
            return embedding.copy()

    def set(self, student_id: int, embedding: np.ndarray) -> None:
        """Store an embedding in the cache.

        Args:
            student_id: The student identifier.
            embedding: The embedding vector to cache.
        """
        with self._lock:
            if student_id in self._cache:
                self._cache.move_to_end(student_id)
                self._cache[student_id] = (embedding.copy(), time.time())
                return

            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[student_id] = (embedding.copy(), time.time())

    def invalidate(self, student_id: int) -> None:
        """Remove a specific entry from the cache.

        Args:
            student_id: The student identifier to remove.
        """
        with self._lock:
            self._cache.pop(student_id, None)

    def clear(self) -> None:
        """Clear all entries from the cache and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, size, max_size, and hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": round(hit_rate, 4),
                "ttl_seconds": self._ttl_seconds,
            }

    def __len__(self) -> int:
        """Return the current number of entries in the cache."""
        with self._lock:
            return len(self._cache)


_student_embedding_cache: Optional[EmbeddingCache] = None
_cache_lock = threading.Lock()


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache instance."""
    global _student_embedding_cache
    if _student_embedding_cache is None:
        with _cache_lock:
            if _student_embedding_cache is None:
                _student_embedding_cache = EmbeddingCache()
    return _student_embedding_cache


def reset_embedding_cache() -> None:
    """Reset the global embedding cache (useful for testing)."""
    global _student_embedding_cache
    with _cache_lock:
        _student_embedding_cache = None
