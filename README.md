# Defence Mission Planning - Correlation-based Risk Prediction (Backend)

Implements **Objective 2** of the Phase-I report (after Du & Chen, 2024): a risk model that
accounts for **hierarchical** and **synergistic** relations between mission tasks instead of
assuming task risks are independent.  Decision-support only: a human commander decides.

## Architecture

```
 Client (Swagger UI / Postman / future web UI / Colab)
                    |  JSON over HTTP
                    v
 +--------------------------- FastAPI  (app/main.py) ---------------------------+
 |  /missions CRUD   /predict   /missions/{id}/predict|whatif|compare   /benchmark |
 +-------+-----------------------+----------------------+----------------------+
         |                       |                      |
         v                       v                      v
  app/schemas.py          app/risk_model.py       app/evaluation.py
  validation (unique      1. base risk from        sample mission, synthetic
  ids, DAG check,         factors (threat, ...)    generator, benchmark
  relation rules)         2. propagate along              |
         |                   dependency DAG               +--> app/simulation.py
         v                3. mission aggregation          |    Monte Carlo "truth"
  app/store.py            4. sensitivity, what-if,        +--> app/baselines.py
  JSON-file store            uncertainty (Beta resample)       independence, entropy
```

Pipeline for one request: **validate -> base risk -> propagate (topological order) ->
aggregate (mission risk, critical-failure probability) -> rank risk drivers**.

## Model

For task j, processed in topological order (r = standalone risk, r' = adjusted risk):

    survive_j = (1 - r_j) * prod_hier (1 - s * r'_i) * prod_syn (1 - min(1, s * sum_k r'_k))
    r'_j      = 1 - survive_j

* independent task = no incoming relation, so r' = r
* hierarchical = one source directly impacts the target
* synergistic = several sources jointly impact the target (effects add, so stronger than independent causes)
* `mission_risk` = criticality-weighted mean of adjusted task risks
* `critical_failure_prob` = P(at least one task with criticality >= 0.9 fails)

## Run in VS Code

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --reload     # open http://127.0.0.1:8000/docs
pytest                            # 7 tests
python run_experiment.py          # benchmark -> outputs/benchmark.csv
```
Or press F5 and choose "FastAPI (uvicorn)" (config in `.vscode/launch.json`).

Quick try in the Swagger UI: `GET /samples/air-defence`, copy the JSON into `POST /predict`.

## Honest limitations (put these in your report)

* The benchmark compares the analytic model to a Monte Carlo simulator that uses the **same
  dependency mechanism**, so near-zero error is a *self-consistency check*, not proof of
  real-world accuracy. Validate against real/public historical data in Phase II.
* The entropy baseline uses different factor weights than the model, so the
  *independence baseline* is the cleaner comparison (same inputs, relations ignored).
* All data is synthetic. Relation strengths and factor weights are assumptions to be elicited
  from domain experts or learned from data.
* `critical_failure_prob` multiplies marginals, which ignores correlation between failures and
  slightly overestimates (see the sample output).
