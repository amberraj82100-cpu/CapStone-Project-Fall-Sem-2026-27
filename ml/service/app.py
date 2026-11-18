"""NeuroQuest ML service.

* GET  /api/dashboard        everything the React model dashboard shows (rebuilt from results on each call)
* POST /api/score            score one game session with the active model per disorder
* POST /api/retrain          retrain game-native models from labelled sessions (champion / challenger)
* POST /api/retrain/maybe    retrain only if enough new labels arrived since the last run
* GET  /api/models           model registry + retraining history

Run:  python -m uvicorn app:app --app-dir ml/service --port 8000 --reload
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from common import RESULTS_DIR  # noqa: E402
from export_dashboard import build_tables, experiments  # noqa: E402

import engine as E  # noqa: E402
import retrain as R  # noqa: E402

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")
RETRAIN_EVERY = int(os.environ.get("RETRAIN_EVERY", "20"))  # new labels per disorder that trigger /maybe
_retrain_lock = threading.Lock()

app = FastAPI(title="NeuroQuest ML service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",  # the React app on any local port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Figures the dashboard may show, per experiment (anything else under ml/results is not served)
FIGURES = {
    "xai_shap_beeswarm.png", "xai_shap_geometry_beeswarm.png", "xai_occlusion_maps.png",
    "xai_question_importance.png", "xai_shap_local_example.png", "roc_best_per_family.png",
    "confusion_best.png", "fusion_weights.png",
}


def _records(df):
    """DataFrame -> JSON-safe list of dicts (NaN -> None)."""
    out = []
    for r in df.to_dict("records"):
        out.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/dashboard")
def dashboard():
    """Everything the React dashboard draws, rebuilt from ml/results on every request."""
    tables = build_tables()
    figures = {}
    for folder, _l, _k, _d, base in experiments():
        d = base / folder
        figures[folder] = sorted(f.name for f in d.glob("*.png") if f.name in FIGURES) if d.exists() else []
    stamp = max((p.stat().st_mtime for p in RESULTS_DIR.glob("*/summary.json")), default=0)
    return {
        "generated_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat() if stamp else None,
        "tables": {name: _records(df) for name, df in tables.items()},
        "figures": figures,
    }


@app.get("/api/figures/{experiment}/{name}")
def figure(experiment: str, name: str):
    known = {folder: base for folder, _l, _k, _d, base in experiments()}
    if experiment not in known or name not in FIGURES:
        raise HTTPException(404, "Unknown figure")
    path = known[experiment] / experiment / name
    if not path.exists():
        raise HTTPException(404, "Figure not generated yet")
    return FileResponse(path, media_type="image/png")


@app.post("/api/score")
def score(session: dict = Body(...)):
    """Score one finished game session (the JSON the Java backend stores)."""
    return E.score_session(session)


def _fetch_training_sessions() -> list[dict]:
    with urllib.request.urlopen(f"{BACKEND_URL}/api/assessments/training-data", timeout=30) as r:
        return json.loads(r.read())


def _labelled_counts(sessions: list[dict]) -> dict[str, int]:
    from retrain import label_of
    return {d: sum(1 for s in sessions if label_of(s, d) is not None) for d in E.DISORDERS}


@app.post("/api/retrain")
def retrain_now(payload: dict | None = Body(None)):
    """Retrain from the sessions in the body, or fetch labelled sessions from the Java backend."""
    from retrain import retrain
    sessions = (payload or {}).get("sessions")
    if sessions is None:
        try:
            sessions = _fetch_training_sessions()
        except Exception as e:  # backend down
            raise HTTPException(502, f"Could not fetch training data from {BACKEND_URL}: {e}")
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "A retraining run is already in progress")
    try:
        return retrain(sessions)
    finally:
        _retrain_lock.release()


@app.post("/api/retrain/maybe")
def retrain_maybe(payload: dict | None = Body(None)):
    """Called by the backend after a label is saved: retrain when RETRAIN_EVERY new labels have arrived."""
    sessions = (payload or {}).get("sessions")
    if sessions is None:
        try:
            sessions = _fetch_training_sessions()
        except Exception as e:
            raise HTTPException(502, f"Could not fetch training data from {BACKEND_URL}: {e}")
    counts = _labelled_counts(sessions)
    hist = E.STATE_DIR / "models" / "retrain_history.json"
    last = json.loads(hist.read_text())[-1] if hist.exists() else None
    last_counts = {d: (last or {}).get("results", {}).get(d, {}).get("labelled_sessions", 0) for d in E.DISORDERS}
    due = [d for d in E.DISORDERS if counts[d] - last_counts[d] >= RETRAIN_EVERY]
    if not due:
        return {"status": "not_due", "labelled": counts, "at_last_run": last_counts, "retrain_every": RETRAIN_EVERY}
    return retrain_now({"sessions": sessions})


@app.get("/api/models")
def models():
    hist = E.STATE_DIR / "models" / "retrain_history.json"
    return {"registry": E.load_registry(),
            "history": json.loads(hist.read_text()) if hist.exists() else [],
            "policy": {"retrain_every_new_labels": RETRAIN_EVERY, "min_labelled": R.MIN_LABELLED,
                       "min_per_class": R.MIN_PER_CLASS, "promote_margin": R.PROMOTE_MARGIN,
                       "min_auc_to_deploy": R.MIN_AUC_TO_DEPLOY}}
