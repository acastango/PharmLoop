"""
LRU result caching for PharmLoop API.

Most interaction checks are repeated (common drug pairs). Caching avoids
redundant model inference. Cache keys include drug pair (ordered) and
a hash of the context dict.

Thread-safe via Python's built-in LRU cache + a dedicated lock for
the polypharmacy cache.

Usage:
    cache = InteractionCache(max_size=10000)
    result = cache.get("fluoxetine", "tramadol", context=None)
    if result is None:
        result = engine.check("fluoxetine", "tramadol")
        cache.put("fluoxetine", "tramadol", result, context=None)
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger("pharmloop.cache")


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as fraction."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class InteractionCache:
    """
    LRU cache for drug interaction results.

    Keys are (drug_a, drug_b, context_hash) where drug names are
    alphabetically ordered for symmetry. Context is hashed to a
    fixed-length string.

    Thread-safe: all operations protected by a lock.

    Args:
        max_size: Maximum number of cached results (default 10000).
        ttl_seconds: Time-to-live per entry in seconds (default 3600 = 1hr).
            Set to 0 to disable TTL.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Current cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
        return self._stats

    def _make_key(
        self,
        drug_a: str,
        drug_b: str,
        context: dict | None = None,
    ) -> str:
        """Create a cache key from drug pair + context."""
        # Order drugs alphabetically for symmetry
        d1, d2 = sorted([drug_a.lower(), drug_b.lower()])
        ctx_hash = self._hash_context(context) if context else "none"
        return f"{d1}:{d2}:{ctx_hash}"

    def _hash_context(self, context: dict) -> str:
        """Deterministic hash of context dict."""
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]

    def get(
        self,
        drug_a: str,
        drug_b: str,
        context: dict | None = None,
    ) -> object | None:
        """
        Look up a cached interaction result.

        Returns None on cache miss or expired entry.
        """
        key = self._make_key(drug_a, drug_b, context)
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            result, timestamp = self._cache[key]

            # Check TTL
            if self._ttl > 0 and (time.time() - timestamp) > self._ttl:
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return result

    def put(
        self,
        drug_a: str,
        drug_b: str,
        result: object,
        context: dict | None = None,
    ) -> None:
        """Store an interaction result in the cache."""
        key = self._make_key(drug_a, drug_b, context)
        with self._lock:
            if key in self._cache:
                # Update existing entry
                self._cache.move_to_end(key)
                self._cache[key] = (result, time.time())
            else:
                # Evict oldest if at capacity
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1
                self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def _make_multi_key(self, drug_names: list[str], context: dict | None) -> str:
        """Create cache key for polypharmacy check."""
        sorted_drugs = sorted(d.lower() for d in drug_names)
        drugs_str = ":".join(sorted_drugs)
        ctx_hash = self._hash_context(context) if context else "none"
        return f"multi:{drugs_str}:{ctx_hash}"

    def get_multi(
        self,
        drug_names: list[str],
        context: dict | None = None,
    ) -> object | None:
        """Look up a cached polypharmacy result."""
        key = self._make_multi_key(drug_names, context)
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            result, timestamp = self._cache[key]
            if self._ttl > 0 and (time.time() - timestamp) > self._ttl:
                del self._cache[key]
                self._stats.misses += 1
                return None

            self._cache.move_to_end(key)
            self._stats.hits += 1
            return result

    def put_multi(
        self,
        drug_names: list[str],
        result: object,
        context: dict | None = None,
    ) -> None:
        """Store a polypharmacy result in the cache."""
        key = self._make_multi_key(drug_names, context)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (result, time.time())
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1
                self._cache[key] = (result, time.time())
