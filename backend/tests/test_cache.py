import threading
import time

import numpy as np
import pytest

from app.core.cache import EmbeddingCache, get_embedding_cache, reset_embedding_cache


class TestEmbeddingCache:
    def test_cache_hit_returns_correct_data(self):
        cache = EmbeddingCache()
        embedding = np.array([1.0, 2.0, 3.0])
        cache.set(1, embedding)

        result = cache.get(1)

        assert result is not None
        assert np.array_equal(result, embedding)
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 0

    def test_cache_miss_returns_none(self):
        cache = EmbeddingCache()

        result = cache.get(999)

        assert result is None
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 1

    def test_ttl_expiration(self):
        cache = EmbeddingCache(ttl_seconds=0.1)
        embedding = np.array([1.0, 2.0, 3.0])
        cache.set(1, embedding)

        assert cache.get(1) is not None

        time.sleep(0.15)

        result = cache.get(1)
        assert result is None
        assert cache.stats()["misses"] == 1

    def test_lru_eviction_at_size_limit(self):
        cache = EmbeddingCache(max_size=3)
        cache.set(1, np.array([1.0]))
        cache.set(2, np.array([2.0]))
        cache.set(3, np.array([3.0]))

        cache.set(4, np.array([4.0]))

        assert cache.get(1) is None
        assert cache.get(2) is not None
        assert cache.get(3) is not None
        assert cache.get(4) is not None
        assert len(cache) == 3

    def test_lru_updates_on_access(self):
        cache = EmbeddingCache(max_size=3)
        cache.set(1, np.array([1.0]))
        cache.set(2, np.array([2.0]))
        cache.set(3, np.array([3.0]))

        cache.get(1)
        cache.set(4, np.array([4.0]))

        assert cache.get(1) is not None
        assert cache.get(2) is None
        assert cache.get(3) is not None
        assert cache.get(4) is not None

    def test_thread_safety(self):
        cache = EmbeddingCache(max_size=100)
        errors = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(100):
                    student_id = worker_id * 1000 + i
                    embedding = np.array([float(i)])
                    cache.set(student_id, embedding)
                    result = cache.get(student_id)
                    if result is None or not np.array_equal(result, embedding):
                        errors.append(f"Mismatch for {student_id}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors[:5]}"
        assert len(cache) <= 100

    def test_stats_tracking(self):
        cache = EmbeddingCache()
        cache.set(1, np.array([1.0]))
        cache.get(1)
        cache.get(1)
        cache.get(999)

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 1000
        assert stats["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)
        assert stats["ttl_seconds"] == 3600.0

    def test_invalidate(self):
        cache = EmbeddingCache()
        cache.set(1, np.array([1.0]))
        assert cache.get(1) is not None

        cache.invalidate(1)
        assert cache.get(1) is None
        assert len(cache) == 0

    def test_clear(self):
        cache = EmbeddingCache()
        cache.set(1, np.array([1.0]))
        cache.set(2, np.array([2.0]))
        cache.get(1)

        cache.clear()

        assert len(cache) == 0
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_set_returns_copy(self):
        cache = EmbeddingCache()
        original = np.array([1.0, 2.0, 3.0])
        cache.set(1, original)
        original[0] = 99.0

        result = cache.get(1)
        assert result is not None
        assert result[0] == 1.0

    def test_get_returns_copy(self):
        cache = EmbeddingCache()
        cache.set(1, np.array([1.0, 2.0, 3.0]))

        result1 = cache.get(1)
        result1[0] = 99.0
        result2 = cache.get(1)

        assert result2 is not None
        assert result2[0] == 1.0


class TestGlobalCache:
    def test_get_embedding_cache_singleton(self):
        reset_embedding_cache()
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        assert cache1 is cache2

    def test_reset_embedding_cache(self):
        reset_embedding_cache()
        cache = get_embedding_cache()
        cache.set(1, np.array([1.0]))
        assert len(cache) == 1

        reset_embedding_cache()
        cache2 = get_embedding_cache()
        assert len(cache2) == 0
        assert cache2 is not cache
