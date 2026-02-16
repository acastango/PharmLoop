"""
PharmLoop REST API — FastAPI server for drug interaction checking.

Endpoints:
  POST /check           → single pair interaction check
  POST /check-multiple  → polypharmacy check (2-20 drugs)
  GET  /drugs           → list available drugs
  GET  /health          → service health + model info

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Environment variables:
    PHARMLOOP_CHECKPOINT  → path to model checkpoint
    PHARMLOOP_DATA_DIR    → path to data directory
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pharmloop.inference import PharmLoopInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmloop.api")

engine: Optional[PharmLoopInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global engine
    checkpoint = os.environ.get(
        "PHARMLOOP_CHECKPOINT", "checkpoints/best_model_phase4a.pt"
    )
    data_dir = os.environ.get("PHARMLOOP_DATA_DIR", "data/processed")
    logger.info(f"Loading model from {checkpoint}, data from {data_dir}")
    engine = PharmLoopInference.load(
        checkpoint_path=checkpoint,
        data_dir=data_dir,
    )
    logger.info(
        f"Model loaded: {len(engine.drug_registry)} drugs, "
        f"{engine.model.cell.hopfield.count} Hopfield patterns"
    )
    yield
    engine = None


app = FastAPI(
    title="PharmLoop",
    version="0.4a.0",
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


# ── Endpoints ──


@app.post("/check", response_model=PairCheckResponse)
def check_pair(req: PairCheckRequest) -> PairCheckResponse:
    """Check interaction between two drugs."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")

    result = engine.check(req.drug_a, req.drug_b, context=req.context)

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


@app.post("/check-multiple", response_model=MultiCheckResponse)
def check_multiple(req: MultiCheckRequest) -> MultiCheckResponse:
    """Check all pairwise interactions for a medication list."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    if len(req.drugs) < 2:
        raise HTTPException(400, "Need at least 2 drugs")
    if len(req.drugs) > 20:
        raise HTTPException(400, "Maximum 20 drugs per request")

    report = engine.check_multiple(req.drugs, context=req.context)

    # Convert pairwise results to response format
    pair_responses = []
    for (pair_key, result) in report.pairwise_results:
        pair_responses.append(PairCheckResponse(
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
        ))

    # Convert alerts to response format
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
    return {
        "status": "ok",
        "model_loaded": True,
        "num_drugs": len(engine.drug_registry),
        "hopfield_patterns": engine.model.cell.hopfield.count,
        "version": "0.4a.0",
    }
