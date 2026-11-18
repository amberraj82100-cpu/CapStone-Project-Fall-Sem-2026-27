"""Which model goes into the game, and why.

Rule: take every model whose mean held-out ROC-AUC is within TOLERANCE of the top score, and deploy the
simplest of them. Differences that small are far inside the fold-to-fold spread, so the extra complexity of
the top scorer buys nothing measurable but costs speed, robustness and explainability.

Simplicity order: fewer stages first (unimodal < early fusion < late fusion < hybrid), then the simpler
algorithm (logistic regression < SVM < random forest < XGBoost < MLP; soft voting < weighted voting < stacking).
"""
from __future__ import annotations

import json

import pandas as pd

from common import RESULTS_DIR

TOLERANCE = 0.005
FAMILY_RANK = {"Unimodal": 0, "Early fusion": 1, "Late fusion": 2, "Hybrid fusion": 3}
ALGO_RANK = {"Logistic Regression": 0, "SVM (RBF)": 1, "Random Forest": 2, "XGBoost": 3, "MLP": 4,
             "soft voting": 0, "AUC-weighted voting": 1, "stacking (LR meta-learner)": 2,
             "views + early (LR meta-learner)": 0}


def complexity(model_name: str, family: str) -> tuple[int, int]:
    algo = model_name.split(" | ")[-1]
    return FAMILY_RANK.get(family, 9), ALGO_RANK.get(algo, 9)


def choose(experiment: str, results_dir=RESULTS_DIR) -> dict:
    return {"experiment": experiment, **choose_summary(pd.read_csv(results_dir / experiment / "metrics_summary.csv"))}


def choose_summary(summ: pd.DataFrame) -> dict:
    top = summ.sort_values("roc_auc", ascending=False).iloc[0]
    near = summ[summ.roc_auc >= top.roc_auc - TOLERANCE].copy()
    near["cx"] = [complexity(m, f) for m, f in zip(near.model, near.family)]
    chosen = near.sort_values(["cx", "roc_auc"], ascending=[True, False]).iloc[0]
    if chosen.model == top.model:
        reason = (f"Highest held-out ROC-AUC, and no simpler model came within {TOLERANCE} of it.")
    else:
        gap = top.roc_auc - chosen.roc_auc
        gap_txt = f"{gap:.3f}" if gap >= 0.0005 else "less than 0.001"
        reason = (f"Within {gap_txt} ROC-AUC of the top scorer ({top.model}), well inside the "
                  f"fold-to-fold spread of ±{top.roc_auc_std:.3f}, and simpler, so it is preferred.")
    return {"chosen_model": chosen.model, "chosen_family": chosen.family,
            "chosen_roc_auc": float(chosen.roc_auc), "chosen_roc_auc_std": float(chosen.roc_auc_std),
            "chosen_balanced_accuracy": float(chosen.balanced_accuracy),
            "chosen_sensitivity": float(chosen.sensitivity), "chosen_specificity": float(chosen.specificity),
            "top_model": top.model, "top_roc_auc": float(top.roc_auc), "n_within_tolerance": int(len(near)),
            "reason": reason}


if __name__ == "__main__":
    for e in ["dyslexia", "dysgraphia", "dyscalculia_castaldi2024", "dyscalculia_synthetic"]:
        print(json.dumps(choose(e), indent=1))
