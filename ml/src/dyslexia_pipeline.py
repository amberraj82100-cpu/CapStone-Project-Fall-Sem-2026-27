"""Dyslexia risk prediction from gamified-test interaction data (Dyt-desktop + Dyt-tablet).

Views (modalities):
  * Demographics          - gender, native-language flags, age, device
  * Task accuracy         - Hits / Score / Accuracy per game question (what the child got right)
  * Interaction behaviour - Clicks / Misses / Missrate per game question (how the child interacted)

Run:  python ml/src/dyslexia_pipeline.py
"""
from __future__ import annotations

import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import viz
from common import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED, ensure_dir, save_json, set_seed, setup_mpl
from fusion import MultimodalFusion, nested_evaluate
from reporting import standard_report

OUT = ensure_dir(RESULTS_DIR / "dyslexia")
N_QUESTIONS = 32
DROPPED_QUESTIONS = [29]  # never administered on the tablet version -> 100 % missing there


# ---------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    frames = []
    for fname, device in [("Dyt-desktop.csv", 0), ("Dyt-tablet.csv", 1)]:
        d = pd.read_csv(DATA_DIR / fname, sep=";")
        d["Device_tablet"] = device
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    for c in ["Gender"]:
        df[c] = (df[c] == "Male").astype(int)
    for c in ["Nativelang", "Otherlang", "Dyslexia"]:
        df[c] = (df[c] == "Yes").astype(int)

    # Repair ratio columns: in the raw CSV some decimals lost their point (0.875 -> 875).
    # Accuracy = Hits/Clicks and Missrate = Misses/Clicks hold for ~99 % of clean cells, so recompute.
    for q in range(1, N_QUESTIONS + 1):
        clicks = df[f"Clicks{q}"]
        safe = clicks.where(clicks > 0)
        df[f"Accuracy{q}"] = (df[f"Hits{q}"] / safe).fillna(0).where(clicks.notna())
        df[f"Missrate{q}"] = (df[f"Misses{q}"] / safe).fillna(0).where(clicks.notna())

    drop = [f"{k}{q}" for q in DROPPED_QUESTIONS for k in ["Clicks", "Hits", "Misses", "Score", "Accuracy", "Missrate"]]
    return df.drop(columns=drop)


def build_views(df: pd.DataFrame) -> dict[str, list[str]]:
    qs = [q for q in range(1, N_QUESTIONS + 1) if q not in DROPPED_QUESTIONS]
    return {
        "Demographics": ["Gender", "Nativelang", "Otherlang", "Age", "Device_tablet"],
        "Task accuracy": [f"{k}{q}" for q in qs for k in ["Hits", "Score", "Accuracy"]],
        "Interaction behaviour": [f"{k}{q}" for q in qs for k in ["Clicks", "Misses", "Missrate"]],
    }


# ---------------------------------------------------------------------------- models
def model_factories(pos_weight: float):
    def logreg(_views):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))

    def xgb(_views):
        return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.6, min_child_weight=3, reg_lambda=2.0,
                             scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=8,
                             random_state=SEED, tree_method="hist")

    def rf(_views):
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                                    class_weight="balanced_subsample", n_jobs=8,
                                                    random_state=SEED))

    def mlp(_views):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                           max_iter=300, early_stopping=True, random_state=SEED))

    unimodal = {"Logistic Regression": logreg, "XGBoost": xgb}
    early = {"Logistic Regression": logreg, "Random Forest": rf, "XGBoost": xgb, "MLP": mlp}
    return unimodal, early


# ---------------------------------------------------------------------------- XAI
def explain(fus: MultimodalFusion, X: pd.DataFrame, y: np.ndarray, views: dict[str, list[str]]):
    import shap

    plt = setup_mpl()
    xgb_name = "Early fusion | XGBoost"
    model = fus.models_[xgb_name]
    cols = [c for cs in views.values() for c in cs]
    # Exact TreeSHAP computed by XGBoost itself (shap's TreeExplainer cannot parse XGBoost>=3 base_score)
    import xgboost as xgb
    contrib = model.get_booster().predict(xgb.DMatrix(X[cols], feature_names=cols), pred_contribs=True)
    sv = shap.Explanation(values=contrib[:, :-1], base_values=contrib[:, -1], data=X[cols].values,
                          feature_names=cols)
    np.save(OUT / "shap_values_early_xgb.npy", sv.values)

    # global beeswarm
    plt.figure()
    shap.plots.beeswarm(sv, max_display=20, show=False)
    plt.title("SHAP - early-fusion XGBoost (top 20 features)", loc="left", fontweight="bold")
    plt.savefig(OUT / "xai_shap_beeswarm.png")
    plt.close("all")

    imp = pd.Series(np.abs(sv.values).mean(0), index=cols)
    imp.sort_values(ascending=False).to_csv(OUT / "xai_feature_importance.csv", header=["mean_abs_shap"])

    # modality importance (sum of |SHAP| per view)
    by_view = pd.Series({v: imp[c].sum() for v, c in views.items()})
    viz.plot_hbar(by_view / by_view.sum(), OUT / "xai_modality_importance.png",
                  "Share of total |SHAP| by modality (early fusion)", "share of mean |SHAP|")

    # importance per game question -> tells the game team which tasks are most diagnostic
    q_imp = {}
    for c, val in imp.items():
        digits = "".join(ch for ch in c if ch.isdigit())
        if digits:
            q_imp[f"Q{int(digits):02d}"] = q_imp.get(f"Q{int(digits):02d}", 0) + val
    q_imp = pd.Series(q_imp)
    q_imp.sort_values(ascending=False).to_csv(OUT / "xai_question_importance.csv", header=["sum_mean_abs_shap"])
    viz.plot_hbar(q_imp, OUT / "xai_question_importance.png",
                  "Which game questions drive the dyslexia prediction?", "sum of mean |SHAP| over the question's features")

    # local explanation for one confidently & correctly detected positive case
    p = model.predict_proba(X[cols])[:, 1]
    pos_idx = np.where((y == 1) & (p >= fus.thresholds_[xgb_name]))[0]
    i = int(pos_idx[np.argsort(p[pos_idx])[len(pos_idx) // 2]])  # median-confidence true positive
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    plt.title(f"Local explanation - child #{X.index[i]} (p={p[i]:.2f})", loc="left", fontweight="bold")
    plt.savefig(OUT / "xai_shap_local_example.png")
    plt.close("all")

    # late-fusion local explanation: per-view log-odds contribution to the stacking decision
    contrib = fus.stacking_view_contributions(X.iloc[[i]]).iloc[0]
    viz.plot_hbar(contrib, OUT / "xai_late_fusion_local_example.png",
                  f"Late fusion (stacking): view contributions, child #{X.index[i]}",
                  "contribution to log-odds of dyslexia risk")
    return {"modality_share": (by_view / by_view.sum()).to_dict(),
            "top_features": imp.sort_values(ascending=False).head(15).to_dict(),
            "top_questions": q_imp.sort_values(ascending=False).head(10).to_dict(),
            "local_example_index": int(X.index[i])}


# ---------------------------------------------------------------------------- main
def xai_only():
    """Recompute explanations from the saved final model without re-running nested CV."""
    import json
    df = load_data()
    views = build_views(df)
    y = df["Dyslexia"].values
    X = df.drop(columns=["Dyslexia"])
    final = joblib.load(MODELS_DIR / "dyslexia_fusion.joblib")["fusion"]
    summ = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    summ["xai"] = explain(final, X, y, views)
    save_json(summ, OUT / "summary.json")
    print("XAI updated ->", OUT)


def main():
    if "--xai-only" in sys.argv:
        return xai_only()
    set_seed()
    t0 = time.time()
    df = load_data()
    views = build_views(df)
    y = df["Dyslexia"].values
    X = df.drop(columns=["Dyslexia"])
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Dyslexia data: {X.shape}, positives={y.sum()} ({y.mean():.1%}), pos_weight={pos_weight:.2f}")
    unimodal, early = model_factories(pos_weight)
    make = lambda: MultimodalFusion(views, unimodal, early, inner_splits=5)

    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    per_fold, preds, info = nested_evaluate(make, X, y, outer)
    summary, best, pooled, rep = standard_report(per_fold, preds, info, views, OUT, "Dyslexia",
                                                 labels=("No risk", "Dyslexia risk"))

    # per-device breakdown of the best model
    dev = X["Device_tablet"]
    by_device = {}
    for d, name in [(0, "desktop"), (1, "tablet")]:
        sel = pooled[pooled["index"].map(dev) == d]
        yh = (sel.p >= sel.threshold).astype(int)
        by_device[name] = {"n": len(sel), "roc_auc": roc_auc_score(sel.y, sel.p),
                           "balanced_accuracy": balanced_accuracy_score(sel.y, yh)}

    # final model on all data (deployment + global explanations)
    print("Fitting final model on all data ...")
    final = make().fit(X, y, verbose=True)
    final.unimodal_models, final.early_models = {}, {}  # factories are closures; not needed for inference
    joblib.dump({"fusion": final, "views": views, "dropped_questions": DROPPED_QUESTIONS}, ensure_dir(MODELS_DIR) / "dyslexia_fusion.joblib")
    xai = explain(final, X, y, views)

    save_json({
        "task": "dyslexia",
        "n_samples": int(len(y)), "n_positive": int(y.sum()), "n_features": int(sum(map(len, views.values()))),
        "views": {k: len(v) for k, v in views.items()},
        "majority_baseline_accuracy": float(1 - y.mean()),
        **rep,
        "by_device": by_device,
        "xai": xai,
        "runtime_sec": time.time() - t0,
    }, OUT / "summary.json")
    print(f"Done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
