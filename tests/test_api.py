"""
Tests for the PharmLoop REST API.

Uses FastAPI TestClient for in-process testing without a running server.
Requires a trained model checkpoint.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("PHARMLOOP_CHECKPOINT", "checkpoints/best_model_phase4a.pt")
os.environ.setdefault("PHARMLOOP_DATA_DIR", "data/processed")

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


def _checkpoint_exists() -> bool:
    for path in ["checkpoints/best_model_phase4a.pt",
                 "checkpoints/final_model_phase4a.pt"]:
        if Path(path).exists():
            return True
    return False


@pytest.fixture(scope="module")
def client():
    if not _checkpoint_exists():
        pytest.skip("No model checkpoint available")

    # Manually load the engine before creating the test client
    from pharmloop.inference import PharmLoopInference
    import api.server as server_mod

    ckpt = os.environ.get("PHARMLOOP_CHECKPOINT", "checkpoints/best_model_phase4a.pt")
    data_dir = os.environ.get("PHARMLOOP_DATA_DIR", "data/processed")
    server_mod.engine = PharmLoopInference.load(ckpt, data_dir=data_dir)

    return TestClient(server_mod.app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["num_drugs"] > 0

    def test_health_has_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


class TestDrugsEndpoint:
    def test_list_drugs(self, client):
        resp = client.get("/drugs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 280
        assert "fluoxetine" in data["drugs"]

    def test_drugs_sorted(self, client):
        resp = client.get("/drugs")
        data = resp.json()
        assert data["drugs"] == sorted(data["drugs"])


class TestCheckEndpoint:
    def test_check_known_pair(self, client):
        resp = client.post("/check", json={
            "drug_a": "fluoxetine",
            "drug_b": "tramadol",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["severity"] in (
            "none", "mild", "moderate", "severe", "contraindicated", "unknown"
        )
        assert isinstance(data["mechanisms"], list)
        assert isinstance(data["narrative"], str)
        assert len(data["narrative"]) > 0

    def test_check_unknown_drug(self, client):
        resp = client.post("/check", json={
            "drug_a": "notarealdrugxyz",
            "drug_b": "aspirin",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["severity"] == "unknown"
        assert "notarealdrugxyz" in data["unknown_drugs"]


class TestCheckMultipleEndpoint:
    def test_check_multiple(self, client):
        resp = client.post("/check-multiple", json={
            "drugs": ["fluoxetine", "tramadol", "warfarin"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pairs_checked"] == 3
        assert len(data["pairwise_results"]) == 3
        assert isinstance(data["multi_drug_alerts"], list)

    def test_check_multiple_too_few(self, client):
        resp = client.post("/check-multiple", json={
            "drugs": ["fluoxetine"],
        })
        assert resp.status_code == 400

    def test_check_multiple_too_many(self, client):
        resp = client.post("/check-multiple", json={
            "drugs": [f"drug_{i}" for i in range(21)],
        })
        assert resp.status_code == 400

    def test_check_multiple_ten_drugs(self, client):
        drugs = [
            "fluoxetine", "tramadol", "warfarin", "metformin",
            "lisinopril", "omeprazole", "amlodipine", "simvastatin",
            "metoprolol", "acetaminophen",
        ]
        resp = client.post("/check-multiple", json={"drugs": drugs})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pairs_checked"] == 45
