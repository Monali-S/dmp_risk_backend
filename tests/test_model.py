import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.evaluation import run_benchmark, sample_air_defence_mission
from app.risk_model import base_risks, predict, propagate
from app.schemas import Mission, Relation, RelationType, Task
from app.simulation import monte_carlo

H, S = RelationType.hierarchical, RelationType.synergistic


def _t(i, r=0.3, crit=0.8):
    return Task(id=i, base_risk=r, criticality=crit)


def test_independent_tasks_unchanged():
    m = Mission(name="m", tasks=[_t("A"), _t("B", 0.5)])
    assert propagate(m, base_risks(m)) == pytest.approx(base_risks(m))


def test_hierarchical_increases_target_risk():
    m = Mission(name="m", tasks=[_t("A", 0.5), _t("B", 0.2)],
                relations=[Relation(type=H, sources=["A"], target="B", strength=0.6)])
    adj = propagate(m, base_risks(m))
    assert adj["A"] == pytest.approx(0.5)
    assert adj["B"] == pytest.approx(1 - 0.8 * (1 - 0.6 * 0.5))


def test_synergy_exceeds_independent_causes():
    tasks = [_t("A", 0.4), _t("B", 0.4), _t("C", 0.1)]
    syn = Mission(name="s", tasks=tasks, relations=[Relation(type=S, sources=["A", "B"], target="C", strength=0.5)])
    hier = Mission(name="h", tasks=tasks, relations=[
        Relation(type=H, sources=["A"], target="C", strength=0.5),
        Relation(type=H, sources=["B"], target="C", strength=0.5)])
    assert propagate(syn, base_risks(syn))["C"] > propagate(hier, base_risks(hier))["C"]


def test_cycle_rejected():
    with pytest.raises(ValueError):
        Mission(name="m", tasks=[_t("A"), _t("B")], relations=[
            Relation(type=H, sources=["A"], target="B"), Relation(type=H, sources=["B"], target="A")])


def test_model_matches_monte_carlo_on_chain():
    m = Mission(name="chain", tasks=[_t("A", 0.4), _t("B", 0.2), _t("C", 0.1)], relations=[
        Relation(type=H, sources=["A"], target="B", strength=0.7),
        Relation(type=H, sources=["B"], target="C", strength=0.7)])
    adj = propagate(m, base_risks(m))
    mc = monte_carlo(m, n_sim=100000, seed=1)["task_risk"]
    for k in adj:
        assert adj[k] == pytest.approx(mc[k], abs=0.01)


def test_benchmark_model_beats_independence():
    res = run_benchmark(n_scenarios=12, n_tasks=12, n_sim=5000, seed=7)["summary"]
    assert res["mission_MAE"]["correlation_model"] < res["mission_MAE"]["independence_baseline"]


def test_api_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSION_DB", str(tmp_path / "db.json"))
    from app.main import app
    c = TestClient(app)
    mission = sample_air_defence_mission().model_dump(mode="json")
    r = c.post("/missions", json=mission)
    assert r.status_code == 201
    mid = r.json()["id"]
    p = c.post(f"/missions/{mid}/predict", params={"uncertainty": True}).json()
    assert 0 <= p["mission_risk"] <= 1 and "mission_risk_interval_90" in p
    w = c.post(f"/missions/{mid}/whatif", json={"mitigations": {"T04": 0.05}}).json()
    assert w["mission_risk_change"] < 0
    assert c.post(f"/missions/{mid}/compare", params={"n_sim": 2000}).status_code == 200
    assert c.get("/missions/nope").status_code == 404
