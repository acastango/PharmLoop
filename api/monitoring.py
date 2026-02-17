"""
Request logging and metrics for PharmLoop API.

Tracks:
  - Request latency (p50, p95, p99)
  - Severity distribution of results (detect drift)
  - Unknown drug rate
  - Confidence distribution
  - Polypharmacy alert rate
  - Most queried drug pairs

All metrics are in-memory. For production, pipe structured logs to
an external monitoring system.
"""

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger("pharmloop.monitoring")


@dataclass
class RequestLog:
    """Single request log entry."""
    timestamp: float
    endpoint: str  # "check" or "check_multiple"
    drug_a: str | None = None
    drug_b: str | None = None
    drugs: list[str] | None = None
    severity: str | None = None
    confidence: float | None = None
    converged: bool | None = None
    latency_ms: float = 0.0
    unknown_drugs: list[str] | None = None
    alert_count: int = 0
    cache_hit: bool = False


class MonitoringService:
    """
    In-memory metrics tracking for the PharmLoop API.

    Maintains a rolling window of recent requests for latency percentile
    computation and counters for severity distribution, unknown drug rate,
    and top-queried pairs.

    Thread-safe: all mutations protected by a lock.

    Args:
        window_size: Number of recent requests to keep for percentile
            computation (default 10000).
    """

    def __init__(self, window_size: int = 10000) -> None:
        self._lock = Lock()
        self._window_size = window_size
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._severity_counts: Counter = Counter()
        self._unknown_drug_requests: int = 0
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._pair_counts: Counter = Counter()
        self._alert_count: int = 0
        self._confidences: deque[float] = deque(maxlen=window_size)
        self._logs: deque[RequestLog] = deque(maxlen=1000)

    def log_check(
        self,
        drug_a: str,
        drug_b: str,
        severity: str,
        confidence: float,
        converged: bool,
        latency_ms: float,
        unknown_drugs: list[str] | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Log a single pair check request."""
        with self._lock:
            self._total_requests += 1
            self._latencies.append(latency_ms)
            self._severity_counts[severity] += 1
            self._confidences.append(confidence)

            pair_key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
            self._pair_counts[pair_key] += 1

            if unknown_drugs:
                self._unknown_drug_requests += 1

            if cache_hit:
                self._cache_hits += 1

            self._logs.append(RequestLog(
                timestamp=time.time(),
                endpoint="check",
                drug_a=drug_a,
                drug_b=drug_b,
                severity=severity,
                confidence=confidence,
                converged=converged,
                latency_ms=latency_ms,
                unknown_drugs=unknown_drugs,
                cache_hit=cache_hit,
            ))

        logger.info(
            f"CHECK {drug_a}+{drug_b} → {severity} "
            f"(conf={confidence:.2f}, {latency_ms:.1f}ms"
            f"{', CACHED' if cache_hit else ''})"
        )

    def log_check_multiple(
        self,
        drugs: list[str],
        highest_severity: str,
        pairs_checked: int,
        alert_count: int,
        latency_ms: float,
        unknown_drugs: list[str] | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Log a polypharmacy check request."""
        with self._lock:
            self._total_requests += 1
            self._latencies.append(latency_ms)
            self._alert_count += alert_count

            if unknown_drugs:
                self._unknown_drug_requests += 1

            if cache_hit:
                self._cache_hits += 1

            self._logs.append(RequestLog(
                timestamp=time.time(),
                endpoint="check_multiple",
                drugs=drugs,
                severity=highest_severity,
                latency_ms=latency_ms,
                unknown_drugs=unknown_drugs,
                alert_count=alert_count,
                cache_hit=cache_hit,
            ))

        logger.info(
            f"MULTI [{len(drugs)} drugs] → {highest_severity} "
            f"({pairs_checked} pairs, {alert_count} alerts, "
            f"{latency_ms:.1f}ms{', CACHED' if cache_hit else ''})"
        )

    def get_metrics(self) -> dict:
        """
        Get current monitoring metrics.

        Returns dict with latency percentiles, severity distribution,
        unknown drug rate, confidence stats, and top-queried pairs.
        """
        with self._lock:
            latencies = sorted(self._latencies)
            confidences = sorted(self._confidences)

            metrics: dict = {
                "total_requests": self._total_requests,
                "cache_hit_rate": (
                    self._cache_hits / self._total_requests
                    if self._total_requests > 0 else 0.0
                ),
            }

            # Latency percentiles
            if latencies:
                metrics["latency_p50_ms"] = self._percentile(latencies, 50)
                metrics["latency_p95_ms"] = self._percentile(latencies, 95)
                metrics["latency_p99_ms"] = self._percentile(latencies, 99)
            else:
                metrics["latency_p50_ms"] = 0.0
                metrics["latency_p95_ms"] = 0.0
                metrics["latency_p99_ms"] = 0.0

            # Severity distribution
            total_sev = sum(self._severity_counts.values())
            metrics["severity_distribution"] = {
                sev: count / total_sev if total_sev > 0 else 0.0
                for sev, count in self._severity_counts.most_common()
            }

            # Unknown drug rate
            metrics["unknown_drug_rate"] = (
                self._unknown_drug_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            )

            # Confidence stats
            if confidences:
                metrics["confidence_mean"] = sum(confidences) / len(confidences)
                metrics["confidence_p50"] = self._percentile(confidences, 50)
            else:
                metrics["confidence_mean"] = 0.0
                metrics["confidence_p50"] = 0.0

            # Top queried pairs
            metrics["top_pairs"] = [
                {"pair": list(pair), "count": count}
                for pair, count in self._pair_counts.most_common(20)
            ]

            # Alert rate
            metrics["total_alerts"] = self._alert_count

            return metrics

    def _percentile(self, sorted_data: list[float], p: int) -> float:
        """Compute percentile from sorted data."""
        if not sorted_data:
            return 0.0
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]
