"""
PharmLoop REST API — FastAPI server for drug interaction checking.

Phase 4b adds:
  - Result caching (LRU, 10K entries)
  - Request monitoring (latency, severity distribution, unknown drug rate)
  - Brand name resolution via DrugResolver
  - skipped_drugs field in polypharmacy responses
  - /metrics endpoint for monitoring dashboard
  - /resolve endpoint for drug name resolution

Endpoints:
  POST /check           → single pair interaction check
  POST /check-multiple  → polypharmacy check (2-20 drugs)
  GET  /drugs           → list available drugs
  GET  /resolve/{name}  → resolve a drug name (brand/fuzzy)
  GET  /health          → service health + model info
  GET  /metrics         → monitoring metrics

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Environment variables:
    PHARMLOOP_CHECKPOINT   → path to model checkpoint
    PHARMLOOP_DATA_DIR     → path to data directory
    PHARMLOOP_BRAND_NAMES  → path to brand_names.json (optional)
    PHARMLOOP_CACHE_SIZE   → max cached results (default 10000)
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.cache import InteractionCache
from api.monitoring import MonitoringService
from pharmloop.inference import PharmLoopInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmloop.api")

engine: Optional[PharmLoopInference] = None
cache: Optional[InteractionCache] = None
monitor: Optional[MonitoringService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global engine, cache, monitor

    checkpoint = os.environ.get(
        "PHARMLOOP_CHECKPOINT", "checkpoints/best_model_phase4a.pt"
    )
    data_dir = os.environ.get("PHARMLOOP_DATA_DIR", "data/processed")
    brand_names = os.environ.get("PHARMLOOP_BRAND_NAMES", None)
    cache_size = int(os.environ.get("PHARMLOOP_CACHE_SIZE", "10000"))

    logger.info(f"Loading model from {checkpoint}, data from {data_dir}")
    engine = PharmLoopInference.load(
        checkpoint_path=checkpoint,
        data_dir=data_dir,
        brand_names_path=brand_names,
    )

    cache = InteractionCache(max_size=cache_size)
    monitor = MonitoringService()

    resolver_info = ""
    if engine.drug_resolver is not None:
        resolver_info = ", drug resolver active"

    logger.info(
        f"Model loaded: {len(engine.drug_registry)} drugs, "
        f"{engine.model.cell.hopfield.count} Hopfield patterns"
        f"{resolver_info}"
    )
    yield
    engine = None
    cache = None
    monitor = None


app = FastAPI(
    title="PharmLoop",
    version="0.4b.0",
    description="Drug interaction checking with oscillatory reasoning",
    lifespan=lifespan,
)


# ── Request / Response models ──


class PairCheckRequest(BaseModel):
    """Request body for single pair interaction check."""
    drug_a: str
    drug_b: str
    context: dict | None = None


class PairCheckResponse(BaseModel):
    """Response for single pair interaction check."""
    drug_a: str
    drug_b: str
    severity: str
    mechanisms: list[str]
    flags: list[str]
    confidence: float
    converged: bool
    steps: int
    narrative: str
    partial_convergence: dict | None = None
    unknown_drugs: list[str] | None = None


class MultiCheckRequest(BaseModel):
    """Request body for polypharmacy check."""
    drugs: list[str]
    context: dict | None = None


class MultiDrugAlertResponse(BaseModel):
    """Multi-drug alert in polypharmacy response."""
    pattern: str
    alert_text: str
    involved_drugs: list[str]
    trigger_mechanism: str
    pair_count: int


class MultiCheckResponse(BaseModel):
    """Response for polypharmacy check."""
    drugs: list[str]
    total_pairs_checked: int
    highest_severity: str
    pairwise_results: list[PairCheckResponse]
    multi_drug_alerts: list[MultiDrugAlertResponse]
    skipped_drugs: list[str] | None = None


class DrugResolveResponse(BaseModel):
    """Response for drug name resolution."""
    original: str
    resolved: str | None
    method: str
    confidence: float


# ── Endpoints ──


@app.post("/check", response_model=PairCheckResponse)
def check_pair(req: PairCheckRequest) -> PairCheckResponse:
    """Check interaction between two drugs. Supports brand names."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")

    start = time.time()

    # Check cache first
    cached = cache.get(req.drug_a, req.drug_b, req.context) if cache else None
    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        if monitor:
            monitor.log_check(
                req.drug_a, req.drug_b, cached.severity,
                cached.confidence, cached.converged, latency_ms,
                unknown_drugs=cached.unknown_drugs or None,
                cache_hit=True,
            )
        return _result_to_response(cached)

    result = engine.check(req.drug_a, req.drug_b, context=req.context)

    # Cache the result
    if cache:
        cache.put(req.drug_a, req.drug_b, result, req.context)

    latency_ms = (time.time() - start) * 1000
    if monitor:
        monitor.log_check(
            req.drug_a, req.drug_b, result.severity,
            result.confidence, result.converged, latency_ms,
            unknown_drugs=result.unknown_drugs or None,
        )

    return _result_to_response(result)


@app.post("/check-multiple", response_model=MultiCheckResponse)
def check_multiple(req: MultiCheckRequest) -> MultiCheckResponse:
    """Check all pairwise interactions for a medication list."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    if len(req.drugs) < 2:
        raise HTTPException(400, "Need at least 2 drugs")
    if len(req.drugs) > 20:
        raise HTTPException(400, "Maximum 20 drugs per request")

    start = time.time()

    # Check cache for polypharmacy
    cached = cache.get_multi(req.drugs, req.context) if cache else None
    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        if monitor:
            monitor.log_check_multiple(
                req.drugs, cached.highest_severity,
                cached.total_pairs_checked,
                len(cached.multi_drug_alerts), latency_ms,
                unknown_drugs=cached.skipped_drugs or None,
                cache_hit=True,
            )
        return _report_to_response(cached)

    report = engine.check_multiple(req.drugs, context=req.context)

    # Cache the result
    if cache:
        cache.put_multi(req.drugs, report, req.context)

    latency_ms = (time.time() - start) * 1000
    if monitor:
        monitor.log_check_multiple(
            req.drugs, report.highest_severity,
            report.total_pairs_checked,
            len(report.multi_drug_alerts), latency_ms,
            unknown_drugs=report.skipped_drugs or None,
        )

    return _report_to_response(report)


@app.get("/resolve/{drug_name}", response_model=DrugResolveResponse)
def resolve_drug(drug_name: str) -> DrugResolveResponse:
    """Resolve a drug name (brand → generic, fuzzy matching)."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")

    if engine.drug_resolver is None:
        # Fallback: exact match only
        lower = drug_name.lower()
        found = lower in engine.drug_registry
        return DrugResolveResponse(
            original=drug_name,
            resolved=lower if found else None,
            method="exact" if found else "unknown",
            confidence=1.0 if found else 0.0,
        )

    result = engine.drug_resolver.resolve(drug_name)
    return DrugResolveResponse(
        original=result.original,
        resolved=result.resolved,
        method=result.method,
        confidence=result.confidence,
    )


@app.get("/drugs")
def list_drugs() -> dict:
    """List all available drugs in the registry."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "drugs": sorted(engine.drug_registry.keys()),
        "count": len(engine.drug_registry),
    }


@app.get("/health")
def health() -> dict:
    """Service health and model information."""
    if engine is None:
        return {"status": "loading", "model_loaded": False}

    info = {
        "status": "ok",
        "model_loaded": True,
        "num_drugs": len(engine.drug_registry),
        "hopfield_patterns": engine.model.cell.hopfield.count,
        "version": "0.4b.0",
        "has_drug_resolver": engine.drug_resolver is not None,
        "has_context_encoder": engine.model.context_encoder is not None,
    }

    if cache:
        stats = cache.stats
        info["cache_size"] = stats.size
        info["cache_hit_rate"] = stats.hit_rate

    return info


@app.get("/metrics")
def metrics() -> dict:
    """Monitoring metrics: latency, severity distribution, etc."""
    if monitor is None:
        return {"error": "Monitoring not initialized"}
    return monitor.get_metrics()


# ── Helpers ──


def _result_to_response(result: object) -> PairCheckResponse:
    """Convert InteractionResult to API response."""
    return PairCheckResponse(
        drug_a=result.drug_a,
        drug_b=result.drug_b,
        severity=result.severity,
        mechanisms=result.mechanisms,
        flags=result.flags,
        confidence=result.confidence,
        converged=result.converged,
        steps=result.steps,
        narrative=result.narrative,
        partial_convergence=result.partial_convergence,
        unknown_drugs=result.unknown_drugs if result.unknown_drugs else None,
    )


def _report_to_response(report: object) -> MultiCheckResponse:
    """Convert PolypharmacyReport to API response."""
    pair_responses = []
    for (pair_key, result) in report.pairwise_results:
        pair_responses.append(_result_to_response(result))

    alert_responses = [
        MultiDrugAlertResponse(
            pattern=alert.pattern,
            alert_text=alert.alert_text,
            involved_drugs=alert.involved_drugs,
            trigger_mechanism=alert.trigger_mechanism,
            pair_count=alert.pair_count,
        )
        for alert in report.multi_drug_alerts
    ]

    return MultiCheckResponse(
        drugs=report.drugs,
        total_pairs_checked=report.total_pairs_checked,
        highest_severity=report.highest_severity,
        pairwise_results=pair_responses,
        multi_drug_alerts=alert_responses,
        skipped_drugs=report.skipped_drugs if report.skipped_drugs else None,
    )
