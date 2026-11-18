"""Dyscalculia: two experiments with the same fusion framework.

A) SYNTHETIC game telemetry (datasets/dyscalculia_synthetic) - 4 views built from trial-level logs of the
   game's Q11-Q15 tasks. Purpose: develop/test the pipeline and verify XAI against the planted ground truth.
   NOT a real-world accuracy estimate.
B) REAL small study (datasets/dyscalculia_castaldi2024, Castaldi et al. 2024, Zenodo 10966052, CC BY 4.0) -
   37 children (15 DD). Views: time perception, cognitive profile. `math` (the diagnostic criterion) and
   `reading` (46 % missing, missingness tied to group) are excluded to avoid label leakage.

Run:  python ml/src/dyscalculia_synthetic.py && python ml/src/dyscalculia_pipeline.py
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import dyscalculia_features as F
import viz
from common import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED, ensure_dir, save_json, set_seed, setup_mpl
from fusion import MultimodalFusion, nested_evaluate
from reporting import standard_report


# ---------------------------------------------------------------------------- shared helpers
def factories(pos_weight=1.0, small=False):
    def logreg(_v):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3 if small else 0.5, class_weight="balanced", max_iter=3000))

    def rf(_v):
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300 if small else 500, min_samples_leaf=2 if small else 3,
                                                    class_weight="balanced_subsample", n_jobs=8, random_state=SEED))

    def svm(_v):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             SVC(C=1.0, probability=True, class_weight="balanced", random_state=SEED))

    def xgb(_v):
        return make_pipeline(SimpleImputer(strategy="median"),
                             XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                                           n_jobs=8, random_state=SEED))

    if small:
        return {"Logistic Regression": logreg, "Random Forest": rf}, {"Logistic Regression": logreg,
                                                                      "Random Forest": rf, "SVM (RBF)": svm}
    return {"Logistic Regression": logreg, "XGBoost": xgb}, {"Logistic Regression": logreg, "Random Forest": rf,
                                                            "XGBoost": xgb}


def shap_early(final, X, views, out, title):
    """Model-agnostic SHAP on the best early-fusion model; returns per-feature and per-view importance."""
    import shap
    plt = setup_mpl()
    name = final.best_early_
    model = final.models_[name]
    cols = [c for cs in views.values() for c in cs]
    Xs = X[cols]
    sample = Xs.sample(min(len(Xs), 400), random_state=SEED)
    f = lambda a: model.predict_proba(pd.DataFrame(a, columns=cols))[:, 1]
    sv = shap.Explainer(f, shap.maskers.Independent(Xs, max_samples=100), seed=SEED)(sample)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title(f"SHAP - {title} ({name.split(' | ')[-1]})", loc="left", fontweight="bold")
    plt.savefig(out / "xai_shap_beeswarm.png")
    plt.close("all")
    imp = pd.Series(np.abs(sv.values).mean(0), index=cols).sort_values(ascending=False)
    imp.to_csv(out / "xai_feature_importance.csv", header=["mean_abs_shap"])
    by_view = pd.Series({v: imp[c].sum() for v, c in views.items()})
    by_view = by_view / by_view.sum()
    viz.plot_hbar(by_view, out / "xai_modality_importance.png", f"Share of |SHAP| by modality - {title}",
                  "share of mean |SHAP|")
    viz.plot_hbar(imp, out / "xai_feature_importance.png", f"Top features - {title}", "mean |SHAP|", top=15)
    direction = {c: float(np.corrcoef(sample[c], sv.values[:, i])[0, 1]) if sample[c].std() > 0 else 0.0
                 for i, c in enumerate(cols)}
    return imp, by_view, direction


# ---------------------------------------------------------------------------- A) synthetic
def run_synthetic():
    out = ensure_dir(RESULTS_DIR / "dyscalculia_synthetic")
    d = DATA_DIR / "dyscalculia_synthetic"
    children = pd.read_csv(d / "children.csv").set_index("child_id")
    trials = pd.read_csv(d / "trials.csv")
    print("A) synthetic: building features from", len(trials), "trials ...")
    X = F.build_feature_table(trials)
    X["age"] = children.loc[X.index, "age"]
    X["gender_male"] = (children.loc[X.index, "gender"] == "M").astype(int)
    y = children.loc[X.index, "dyscalculia"].values
    views = {"Demographics": ["age", "gender_male"], "Task accuracy": F.ACCURACY,
             "Response time": F.RESPONSE_TIME, "Number-sense profile": F.NUMBER_SENSE}
    uni, early = factories(pos_weight=(y == 0).sum() / (y == 1).sum())
    make = lambda: MultimodalFusion(views, uni, early, inner_splits=5)
    per_fold, preds, info = nested_evaluate(make, X, y, StratifiedKFold(5, shuffle=True, random_state=SEED))
    summary, best, pooled, rep = standard_report(per_fold, preds, info, views, out, "Dyscalculia (SYNTHETIC)",
                                                 labels=("No risk", "Dyscalculia risk"))
    final = make().fit(X, y)
    final.unimodal_models, final.early_models = {}, {}
    joblib.dump({"fusion": final, "views": views, "synthetic": True}, ensure_dir(MODELS_DIR) / "dyscalculia_synthetic_fusion.joblib")
    imp, by_view, direction = shap_early(final, X, views, out, "synthetic dyscalculia")

    # XAI sanity check against the planted ground truth: features of the impaired abilities should rank high
    truth = pd.read_csv(d / "ground_truth_DO_NOT_TRAIN.csv").set_index("child_id").loc[X.index]
    oof_best = pooled.set_index("index").p
    ceiling = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, truth.true_dd)
    save_json({"task": "dyscalculia_synthetic", "SYNTHETIC": True, "n_samples": int(len(y)), "n_positive": int(y.sum()),
               "views": {k: len(v) for k, v in views.items()},
               "majority_baseline_accuracy": float(1 - y.mean()), **rep,
               "label_noise_auc_ceiling": float(ceiling),
               "xai": {"modality_share": by_view.to_dict(), "top_features": imp.head(12).to_dict(),
                       "direction_corr": {k: direction[k] for k in imp.head(12).index},
                       "planted_effects_sd": {"symb (arithmetic/sequence)": -1.4, "ans (magnitude/numberline)": -1.0,
                                              "speed": -0.3, "subit": -0.3, "attn": -0.2}}},
              out / "summary.json")
    print("A) done; AUC ceiling given 5% label noise =", round(ceiling, 3))


# ---------------------------------------------------------------------------- B) real (Castaldi 2024)
def run_castaldi():
    out = ensure_dir(RESULTS_DIR / "dyscalculia_castaldi2024")
    df = pd.read_csv(DATA_DIR / "dyscalculia_castaldi2024" / "dataset.csv", sep=";", decimal=",")
    df.columns = ["group", "gender", "pse_500ms", "wf_500ms", "pse_1500ms", "wf_1500ms", "age", "nonverbal_iq",
                  "math", "reading"]
    y = (df.group == 2).astype(int).values
    X = df.drop(columns=["group", "math", "reading"])  # math = diagnostic criterion; reading = leaky missingness
    X["gender_male"] = (X.pop("gender") == 2).astype(int)
    views = {"Time perception": ["pse_500ms", "wf_500ms", "pse_1500ms", "wf_1500ms"],
             "Cognitive profile": ["age", "nonverbal_iq", "gender_male"]}
    print(f"B) Castaldi 2024: n={len(y)}, DD={y.sum()}")
    uni, early = factories(small=True)
    make = lambda: MultimodalFusion(views, uni, early, inner_splits=3)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    per_fold, preds, info = nested_evaluate(make, X, y, outer, verbose=False)
    summary, best, pooled, rep = standard_report(per_fold, preds, info, views, out, "Dyscalculia (Castaldi 2024, real)",
                                                 labels=("Control", "Dyscalculia"), n_outer_desc="5-fold CV x 10 repeats")
    final = make().fit(X, y)
    final.unimodal_models, final.early_models = {}, {}
    joblib.dump({"fusion": final, "views": views}, ensure_dir(MODELS_DIR) / "dyscalculia_castaldi2024_fusion.joblib")
    imp, by_view, direction = shap_early(final, X, views, out, "Castaldi 2024")
    save_json({"task": "dyscalculia_castaldi2024", "n_samples": int(len(y)), "n_positive": int(y.sum()),
               "views": {k: len(v) for k, v in views.items()},
               "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())), **rep,
               "excluded_columns": {"math": "diagnostic criterion (label leakage)",
                                    "reading": "46% missing; values/missingness separate the groups perfectly"},
               "xai": {"modality_share": by_view.to_dict(), "top_features": imp.to_dict(),
                       "direction_corr": direction}}, out / "summary.json")


def main():
    set_seed()
    t0 = time.time()
    run_castaldi()
    run_synthetic()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
