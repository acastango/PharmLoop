"""Tests for API caching layer."""

import time
import pytest

from api.cache import InteractionCache, CacheStats


class FakeResult:
    """Minimal result for cache testing."""
    def __init__(self, severity="moderate"):
        self.severity = severity


class TestInteractionCache:
    """Test LRU caching for interaction results."""

    def test_put_and_get(self):
        cache = InteractionCache(max_size=100)
        result = FakeResult("severe")
        cache.put("fluoxetine", "tramadol", result)

        cached = cache.get("fluoxetine", "tramadol")
        assert cached is not None
        assert cached.severity == "severe"

    def test_miss_returns_none(self):
        cache = InteractionCache(max_size=100)
        assert cache.get("fluoxetine", "tramadol") is None

    def test_order_invariance(self):
        """Cache key should be order-independent for drug pairs."""
        cache = InteractionCache(max_size=100)
        result = FakeResult("severe")
        cache.put("fluoxetine", "tramadol", result)

        # Query in reverse order
        cached = cache.get("tramadol", "fluoxetine")
        assert cached is not None
        assert cached.severity == "severe"

    def test_case_insensitive(self):
        """Cache should be case-insensitive for drug names."""
        cache = InteractionCache(max_size=100)
        result = FakeResult("moderate")
        cache.put("Fluoxetine", "Tramadol", result)

        cached = cache.get("fluoxetine", "tramadol")
        assert cached is not None

    def test_context_differentiation(self):
        """Different contexts should have different cache entries."""
        cache = InteractionCache(max_size=100)
        result_no_ctx = FakeResult("moderate")
        result_with_ctx = FakeResult("severe")

        cache.put("warfarin", "acetaminophen", result_no_ctx, context=None)
        cache.put("warfarin", "acetaminophen", result_with_ctx,
                  context={"dose_b_normalized": 0.9})

        assert cache.get("warfarin", "acetaminophen", None).severity == "moderate"
        assert cache.get("warfarin", "acetaminophen",
                        {"dose_b_normalized": 0.9}).severity == "severe"

    def test_lru_eviction(self):
        """Oldest entry should be evicted when cache is full."""
        cache = InteractionCache(max_size=2)

        cache.put("a", "b", FakeResult("mild"))
        cache.put("c", "d", FakeResult("moderate"))
        cache.put("e", "f", FakeResult("severe"))  # should evict a+b

        assert cache.get("a", "b") is None  # evicted
        assert cache.get("c", "d") is not None
        assert cache.get("e", "f") is not None

    def test_ttl_expiration(self):
        """Entries should expire after TTL."""
        cache = InteractionCache(max_size=100, ttl_seconds=1)

        cache.put("a", "b", FakeResult("mild"))
        assert cache.get("a", "b") is not None

        time.sleep(1.1)
        assert cache.get("a", "b") is None  # expired

    def test_stats_tracking(self):
        cache = InteractionCache(max_size=100)

        cache.get("a", "b")  # miss
        cache.put("a", "b", FakeResult("mild"))
        cache.get("a", "b")  # hit
        cache.get("c", "d")  # miss

        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 2
        assert stats.size == 1

    def test_clear(self):
        cache = InteractionCache(max_size=100)
        cache.put("a", "b", FakeResult("mild"))
        cache.clear()
        assert cache.get("a", "b") is None

    def test_multi_cache(self):
        """Test polypharmacy caching."""
        cache = InteractionCache(max_size=100)
        result = FakeResult("severe")

        cache.put_multi(["a", "b", "c"], result)
        cached = cache.get_multi(["a", "b", "c"])
        assert cached is not None

        # Order should not matter
        cached2 = cache.get_multi(["c", "a", "b"])
        assert cached2 is not None

    def test_multi_cache_different_lists(self):
        cache = InteractionCache(max_size=100)
        cache.put_multi(["a", "b"], FakeResult("mild"))
        assert cache.get_multi(["a", "b", "c"]) is None
