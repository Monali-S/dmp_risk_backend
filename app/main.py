"""FastAPI application. Run:  uvicorn app.main:app --reload"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import risk_model
from .evaluation import compare_methods, run_benchmark, sample_air_defence_mission
from .schemas import BenchmarkRequest, Mission, WhatIfRequest
from .store import MissionStore

app = FastAPI(
    title="Defence Mission Planning - Correlation-based Risk Prediction",
    version="1.0.0",
    description="Decision-support backend. Humans remain in control of all decisions.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
store = MissionStore(os.getenv("MISSION_DB", "data/missions.json"))


def _get_or_404(mission_id: str) -> Mission:
    m = store.get(mission_id)
    if m is None:
        raise HTTPException(404, f"Mission '{mission_id}' not found")
    return m


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/samples/air-defence", response_model=Mission)
def sample_mission():
    return sample_air_defence_mission()


# ---- mission CRUD
@app.post("/missions", response_model=Mission, status_code=201)
def create_mission(mission: Mission):
    return store.create(mission)


@app.get("/missions", response_model=list[Mission])
def list_missions():
    return store.list()


@app.get("/missions/{mission_id}", response_model=Mission)
def get_mission(mission_id: str):
    return _get_or_404(mission_id)


@app.put("/missions/{mission_id}", response_model=Mission)
def update_mission(mission_id: str, mission: Mission):
    updated = store.update(mission_id, mission)
    if updated is None:
        raise HTTPException(404, f"Mission '{mission_id}' not found")
    return updated


@app.delete("/missions/{mission_id}", status_code=204)
def delete_mission(mission_id: str):
    if not store.delete(mission_id):
        raise HTTPException(404, f"Mission '{mission_id}' not found")


# ---- risk analytics
@app.post("/predict")
def predict_adhoc(mission: Mission, uncertainty: bool = False):
    out = risk_model.predict(mission)
    if uncertainty:
        out["mission_risk_interval_90"] = risk_model.predict_with_uncertainty(mission)
    return out


@app.post("/missions/{mission_id}/predict")
def predict_stored(mission_id: str, uncertainty: bool = False):
    return predict_adhoc(_get_or_404(mission_id), uncertainty)


@app.post("/missions/{mission_id}/whatif")
def what_if(mission_id: str, req: WhatIfRequest):
    try:
        return risk_model.what_if(_get_or_404(mission_id), req.mitigations)
    except KeyError as e:
        raise HTTPException(400, str(e))


@app.post("/missions/{mission_id}/compare")
def compare(mission_id: str, n_sim: int = 20000):
    n_sim = max(1000, min(n_sim, 100000))
    return compare_methods(_get_or_404(mission_id), n_sim=n_sim)


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    out = run_benchmark(req.n_scenarios, req.n_tasks, req.n_sim, req.seed)
    out.pop("rows")  # keep response small; use CLI/notebook for per-scenario rows
    return out
