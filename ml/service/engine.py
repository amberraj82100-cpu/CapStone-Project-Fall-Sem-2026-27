"""Scoring engine: model registry + game-session adapters.

A game session (as stored by the Java backend) is turned into the inputs each model needs:

* handwriting image (new "copy the sentence" task)  -> pretrained dysgraphia fusion model
* number-sense round trial logs (new round task)    -> pretrained (synthetic) dyscalculia model
* the game's own answers to its 15+ questions       -> game-native models, trained by retrain.py

The registry (``<state>/registry.json``) records every model version per disorder and which one is active.
State lives in ML_STATE_DIR (default: ml/), so tests can run against a throwaway copy.
"""
from __future__ import annotations

import base64
import json
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from common import ML_DIR
import dyscalculia_features as DCF
import dysgraphia_features as DGF
from main_model import DisorderModel

STATE_DIR = Path(os.environ.get("ML_STATE_DIR", ML_DIR))
REGISTRY = STATE_DIR / "models" / "registry.json"
DISORDERS = ["DYSLEXIA", "DYSGRAPHIA", "DYSCALCULIA"]
DOMAIN_QUESTIONS = {"DYSLEXIA": range(1, 6), "DYSGRAPHIA": range(6, 11), "DYSCALCULIA": range(11, 16)}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------- registry
def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return init_registry()


def save_registry(reg: dict):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))
    _load_bundle.cache_clear()


def init_registry() -> dict:
    """First run: register the research models the game can feed (dysgraphia image, dyscalculia trials)."""
    card = json.loads((ML_DIR / "models" / "screening_system_card.json").read_text())
    reg = {d: {"active": None, "versions": []} for d in DISORDERS}
    pre = {
        "DYSGRAPHIA": ("dysgraphia_fusion.joblib", "handwriting_image", False,
                       "Trained on 249 scanned handwriting samples (research dataset)."),
        "DYSCALCULIA": ("dyscalculia_synthetic_fusion.joblib", "number_round_trials", True,
                        "Trained on 3,000 simulated children; demo until real labelled game data exists."),
    }
    for d, (bundle, kind, synthetic, note) in pre.items():
        c = card[d]
        v = {"id": f"{d.lower()}-pretrained-v1", "kind": "pretrained", "input": kind, "created_at": now(),
             "bundle": str(ML_DIR / "models" / bundle), "model": c["model"],
             "moderate_threshold": c["moderate_threshold"], "high_threshold": c["high_threshold"],
             "synthetic": synthetic, "note": note, "trained_until": None,
             "metrics": {"roc_auc": c["cv_roc_auc"], "balanced_accuracy": c["cv_balanced_accuracy"],
                         "sensitivity": c["cv_sensitivity"], "specificity": c["cv_specificity"],
                         "evaluated_on": "nested CV on the research dataset"}}
        reg[d]["versions"].append(v)
        reg[d]["active"] = v["id"]
    save_registry(reg)
    return reg


def active_version(reg: dict, disorder: str) -> dict | None:
    a = reg[disorder]["active"]
    return next((v for v in reg[disorder]["versions"] if v["id"] == a), None)


@lru_cache(maxsize=16)
def _load_bundle(path: str):
    return joblib.load(path)["fusion"]


def model_for(version: dict) -> DisorderModel:
    return DisorderModel(version["id"], _load_bundle(version["bundle"]), version["model"],
                         version["moderate_threshold"], version["high_threshold"], version["synthetic"],
                         version.get("metrics", {}))


# ---------------------------------------------------------------------------- adapters
def _responses(session: dict) -> dict[int, dict]:
    return {int(r.get("questionId", -1)): r for r in session.get("responses") or []}


def _raw(r: dict) -> dict:
    return r.get("rawData") or {}


def decode_image(data_url: str) -> np.ndarray | None:
    """data:image/png;base64,... -> grayscale uint8 with dark ink on white (transparent pixels = paper)."""
    try:
        b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
        buf = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)
        img = rgb.astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def handwriting_features(session: dict, with_cnn: bool = True) -> pd.DataFrame | None:
    for r in _responses(session).values():
        url = _raw(r).get("handwritingImage")
        if not url:
            continue
        gray = decode_image(url)
        if gray is None or (gray < 128).mean() < 0.002:  # blank canvas
            return None
        bw = DGF.preprocess(gray)
        n = cv2.connectedComponentsWithStats(bw, 8)[0] - 1
        if n < 3:
            return None
        row = DGF.geometry_features(bw)
        if with_cnn:
            emb = DGF.embed_strip(DGF.to_rgb_strip(bw))
            row = {**row, **{f"{DGF.DEEP_PREFIX}{j}": float(v) for j, v in enumerate(emb)}}
        return pd.DataFrame([row])
    return None


def number_round_features(session: dict) -> pd.DataFrame | None:
    for r in _responses(session).values():
        trials = _raw(r).get("trials")
        if not trials:
            continue
        t = pd.DataFrame(trials)
        if set(DCF.TASKS) - set(t.get("task", pd.Series(dtype=str)).unique()):
            return None  # incomplete round
        t["child_id"] = 0
        t["correct"] = t["correct"].astype(int)
        X = DCF.build_feature_table(t)
        X["age"] = float(session.get("childAge") or 8)
        X["gender_male"] = 0.5  # the game does not ask for gender
        return X.reset_index(drop=True)
    return None


def game_features(session: dict, disorder: str) -> pd.DataFrame:
    """Features from the game's own answers, for game-native models (one row)."""
    rs = _responses(session)
    row = {"age": float(session.get("childAge") or 8)}
    for q in DOMAIN_QUESTIONS[disorder]:
        r = rs.get(q, {})
        row[f"q{q}_correct"] = float(bool(r.get("isCorrect"))) if r else np.nan
        t = r.get("timeTakenMs")
        row[f"q{q}_log_time"] = math.log1p(max(0, float(t)) / 1000) if t is not None else np.nan
        if disorder == "DYSGRAPHIA":
            row[f"q{q}_jitter"] = float(r.get("motorJitterScore", np.nan)) if r else np.nan
            row[f"q{q}_hesitation"] = float(r.get("motorHesitationCount", np.nan)) if r else np.nan
    if disorder == "DYSLEXIA":
        row["q1_reversal"] = float(bool(rs.get(1, {}).get("reversalDetected"))) if 1 in rs else np.nan
    if disorder == "DYSCALCULIA":
        row["q13_magnitude_distance"] = float(rs.get(13, {}).get("magnitudeDistance", np.nan)) if 13 in rs else np.nan
    X = pd.DataFrame([row])
    extra = (handwriting_features(session, with_cnn=False) if disorder == "DYSGRAPHIA" else
             number_round_features(session) if disorder == "DYSCALCULIA" else None)
    if disorder == "DYSGRAPHIA":
        geo = extra[DGF.GEOMETRY_FEATURES] if extra is not None else pd.DataFrame([{c: np.nan for c in DGF.GEOMETRY_FEATURES}])
        X = pd.concat([X, geo.reset_index(drop=True)], axis=1)
    if disorder == "DYSCALCULIA":
        cols = DCF.ACCURACY + DCF.RESPONSE_TIME + DCF.NUMBER_SENSE
        rnd = extra[cols] if extra is not None else pd.DataFrame([{c: np.nan for c in cols}])
        X = pd.concat([X, rnd.reset_index(drop=True)], axis=1)
    return X


def game_views(disorder: str) -> dict[str, list[str]]:
    qs = list(DOMAIN_QUESTIONS[disorder])
    views = {"Answers": [f"q{q}_correct" for q in qs], "Response time": [f"q{q}_log_time" for q in qs]}
    if disorder == "DYSLEXIA":
        views["Profile"] = ["age", "q1_reversal"]
    if disorder == "DYSGRAPHIA":
        views["Motor control"] = [f"q{q}_{k}" for q in qs for k in ("jitter", "hesitation")] + ["age"]
        views["Handwriting geometry"] = list(DGF.GEOMETRY_FEATURES)
    if disorder == "DYSCALCULIA":
        views["Profile"] = ["age", "q13_magnitude_distance"]
        views["Number-sense round"] = DCF.ACCURACY + DCF.RESPONSE_TIME + DCF.NUMBER_SENSE
    return views


def inputs_for(version: dict, session: dict) -> pd.DataFrame | None:
    kind = version["input"]
    if kind == "handwriting_image":
        return handwriting_features(session)
    if kind == "number_round_trials":
        return number_round_features(session)
    if kind == "game_features":
        return game_features(session, version["disorder"])
    return None


# ---------------------------------------------------------------------------- scoring
def score_session(session: dict) -> dict:
    reg = load_registry()
    out = {"scored_at": now(), "domains": {}}
    for d in DISORDERS:
        v = active_version(reg, d)
        if v is None:
            out["domains"][d] = {"status": "no_model", "message": "No trained model yet; rule-based score applies."}
            continue
        X = inputs_for(v, session)
        if X is None:
            out["domains"][d] = {"status": "no_input", "model_version": v["id"],
                                 "message": "This session lacks the input this model needs; rule-based score applies."}
            continue
        res = model_for(v).score(X)[0]
        out["domains"][d] = {"status": "ok", "model_version": v["id"], "model_kind": v["kind"],
                             "probability": res["probability"], "risk_level": res["risk_level"],
                             "model": res["model"], "synthetic_training_data": v["synthetic"],
                             "modality_contributions_logodds": res["modality_contributions_logodds"],
                             "top_risk_modality": res["top_risk_modality"]}
    return out
