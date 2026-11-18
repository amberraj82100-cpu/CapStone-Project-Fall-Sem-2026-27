"""Retrain game-native models from labelled game sessions (champion / challenger).

For each disorder with enough labelled sessions:
  1. build game features (answers, response times, motor data, handwriting geometry, number round)
  2. run the same nested-CV multimodal fusion experiment used for the research datasets
  3. pick the simplest model within 0.005 ROC-AUC of the best (model_selection.choose_summary)
  4. evaluate the live model (champion) on game sessions it never saw
  5. promote the challenger only if it beats the champion by PROMOTE_MARGIN and clears MIN_AUC_TO_DEPLOY
Every run is recorded in the registry and in <state>/results/game_<disorder>/ for the dashboard.

Labels: session["labels"][DISORDER] = {"value": 0|1, "source": "specialist"|"parent", "at": iso}.
Specialist-confirmed labels are the target; parent-reported ones are used when no specialist label exists.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import engine as E
from common import SEED, save_json
from fusion import MultimodalFusion, nested_evaluate
from model_selection import choose_summary
from reporting import standard_report

MIN_LABELLED = 40          # sessions with a label for the disorder
MIN_PER_CLASS = 10         # of each class (at risk / not at risk)
PROMOTE_MARGIN = 0.01      # challenger must beat the champion's ROC-AUC by this much
MIN_AUC_TO_DEPLOY = 0.65   # never deploy a model weaker than this


def _factories():
    def logreg(_v):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))

    def rf(_v):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                    class_weight="balanced_subsample", n_jobs=4, random_state=SEED))
    return {"Logistic Regression": logreg, "Random Forest": rf}, {"Logistic Regression": logreg, "Random Forest": rf}


def label_of(session: dict, disorder: str):
    lab = (session.get("labels") or {}).get(disorder)
    if not lab or lab.get("value") not in (0, 1, True, False):
        return None
    return int(lab["value"]), lab.get("source", "unknown")


def _completed(s):
    return s.get("completedAt") or s.get("createdAt") or ""


def evaluate_champion(version: dict, sessions: list[dict], y: np.ndarray):
    """ROC-AUC of the live model on labelled sessions it was not trained on (None if too few)."""
    if version is None:
        return None, 0
    cutoff = version.get("trained_until") or ""
    ps, ys = [], []
    model = E.model_for(version)
    for s, yi in zip(sessions, y):
        if cutoff and _completed(s) <= cutoff:
            continue  # the champion saw this session during training
        X = E.inputs_for(version, s)
        if X is None:
            continue
        ps.append(model.score(X)[0]["probability"])
        ys.append(yi)
    ys = np.array(ys)
    if len(ys) == 0 or min((ys == 1).sum(), (ys == 0).sum()) < MIN_PER_CLASS:
        return None, len(ys)
    return float(roc_auc_score(ys, ps)), len(ys)


def retrain_disorder(disorder: str, sessions: list[dict], reg: dict) -> dict:
    rows = [(s, label_of(s, disorder)) for s in sessions]
    rows = [(s, lab) for s, lab in rows if lab is not None]
    n = len(rows)
    y = np.array([lab[0] for _, lab in rows])
    sources = pd.Series([lab[1] for _, lab in rows]).value_counts().to_dict() if rows else {}
    base = {"disorder": disorder, "labelled_sessions": n, "at_risk": int(y.sum()) if n else 0,
            "label_sources": sources, "run_at": E.now()}
    if n < MIN_LABELLED or min(int(y.sum()), n - int(y.sum())) < MIN_PER_CLASS:
        return {**base, "status": "waiting_for_data",
                "message": f"Need at least {MIN_LABELLED} labelled sessions with {MIN_PER_CLASS}+ in each class."}

    sess = [s for s, _ in rows]
    X = pd.concat([E.game_features(s, disorder) for s in sess], ignore_index=True)
    views = {v: c for v, c in E.game_views(disorder).items() if X[c].notna().any().any()}
    uni, early = _factories()
    make = lambda: MultimodalFusion(views, uni, early, inner_splits=3)
    outer = (RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED) if n < 300
             else StratifiedKFold(5, shuffle=True, random_state=SEED))
    per_fold, preds, info = nested_evaluate(make, X, y, outer, verbose=False)

    exp = f"game_{disorder.lower()}"
    out = E.STATE_DIR / "results" / exp
    out.mkdir(parents=True, exist_ok=True)
    summary, best, pooled, rep = standard_report(per_fold, preds, info, views, out, f"{disorder.title()} (game data)",
                                                 labels=("Not at risk", "At risk"), n_outer_desc="outer CV folds")
    sel = choose_summary(summary)
    challenger_auc = sel["chosen_roc_auc"]

    champion = E.active_version(reg, disorder)
    champ_auc, champ_n = evaluate_champion(champion, sess, y)
    champ_basis = "unseen game sessions" if champ_auc is not None else None
    if champion is not None and champ_auc is None and champion["kind"] == "game":
        champ_auc = champion["metrics"]["roc_auc"]  # no fresh holdout yet: compare with its own CV estimate
        champ_basis = "its own cross-validation (too few unseen sessions)"
    promote = challenger_auc >= MIN_AUC_TO_DEPLOY and (champ_auc is None or challenger_auc >= champ_auc + PROMOTE_MARGIN)

    final = make().fit(X, y)
    final.unimodal_models, final.early_models = {}, {}
    chosen = sel["chosen_model"]
    neg = preds[(preds.model == chosen) & (preds.y == 0)].p
    moderate = float(final.thresholds_[chosen])
    high = max(float(np.quantile(neg, 0.95)), moderate + 1e-6) if len(neg) else moderate + 0.2

    k = sum(1 for v in reg[disorder]["versions"] if v["kind"] == "game") + 1
    vid = f"{disorder.lower()}-game-v{k}"
    bundle = E.STATE_DIR / "models" / "game" / f"{vid}.joblib"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"fusion": final, "views": views}, bundle)
    version = {"id": vid, "kind": "game", "input": "game_features", "disorder": disorder, "created_at": E.now(),
               "bundle": str(bundle), "model": chosen, "moderate_threshold": moderate, "high_threshold": high,
               "synthetic": False, "trained_until": max(_completed(s) for s in sess), "n_train": n,
               "label_sources": sources, "selection_reason": sel["reason"],
               "metrics": {"roc_auc": challenger_auc, "balanced_accuracy": sel["chosen_balanced_accuracy"],
                           "sensitivity": sel["chosen_sensitivity"], "specificity": sel["chosen_specificity"],
                           "evaluated_on": "nested CV on labelled game sessions"},
               "promoted": promote}
    reg[disorder]["versions"].append(version)
    if promote:
        reg[disorder]["active"] = vid

    save_json({"task": exp, "n_samples": n, "n_positive": int(y.sum()), "views": {v: len(c) for v, c in views.items()},
               "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())), **rep,
               "label_sources": sources, "trained_at": version["created_at"], "xai": {}}, out / "summary.json")
    return {**base, "status": "promoted" if promote else "kept_champion", "challenger": vid,
            "challenger_model": chosen, "challenger_roc_auc": challenger_auc,
            "champion": champion["id"] if champion else None, "champion_roc_auc": champ_auc,
            "champion_score_basis": champ_basis, "champion_unseen_sessions": champ_n,
            "reason": ("Challenger clears the bar and beats the live model." if promote else
                       f"Challenger ROC-AUC {challenger_auc:.3f} did not beat the live model by {PROMOTE_MARGIN}"
                       f" or is below {MIN_AUC_TO_DEPLOY}.")}


def retrain(sessions: list[dict], disorders=E.DISORDERS) -> dict:
    reg = E.load_registry()
    report = {"run_at": E.now(), "n_sessions": len(sessions), "results": {}}
    for d in disorders:
        report["results"][d] = retrain_disorder(d, sessions, reg)
    E.save_registry(reg)
    hist = E.STATE_DIR / "models" / "retrain_history.json"
    past = json.loads(hist.read_text()) if hist.exists() else []
    hist.write_text(json.dumps((past + [report])[-50:], indent=2, default=str))
    return report
