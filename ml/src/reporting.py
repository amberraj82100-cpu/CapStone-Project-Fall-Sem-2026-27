"""Standard result tables + figures for a nested-CV fusion experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd

import viz
from common import bootstrap_ci
from fusion import summarise


def standard_report(per_fold: pd.DataFrame, preds: pd.DataFrame, info: list, views: dict, out, task: str,
                    labels=("Negative", "Positive"), n_outer_desc="5 outer test folds"):
    per_fold.to_csv(out / "metrics_per_fold.csv", index=False)
    preds.to_csv(out / "oof_predictions.csv", index=False)
    summary = summarise(per_fold)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    print(summary[["model", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "f1"]].round(3).to_string())

    viz.plot_model_comparison(summary, out / "model_comparison_auc.png", f"{task} - ROC-AUC by model and fusion strategy")
    viz.plot_model_comparison(summary, out / "model_comparison_balacc.png",
                              f"{task} - balanced accuracy by model and fusion strategy",
                              metric="balanced_accuracy", label="Balanced accuracy")
    best_per_family = summary.groupby("family", sort=False).head(1)
    best = summary.iloc[0]
    viz.plot_roc(preds, list(best_per_family["model"]), out / "roc_best_per_family.png",
                 f"{task} - best model of each family")

    pooled = preds[preds.model == best["model"]]
    yhat = (pooled.p >= pooled.threshold).astype(int)
    cm = {"tp": int(((yhat == 1) & (pooled.y == 1)).sum()), "fp": int(((yhat == 1) & (pooled.y == 0)).sum()),
          "tn": int(((yhat == 0) & (pooled.y == 0)).sum()), "fn": int(((yhat == 0) & (pooled.y == 1)).sum())}
    viz.plot_confusion(cm["tp"], cm["fp"], cm["tn"], cm["fn"], out / "confusion_best.png",
                       f"Best: {best['model']}\n(pooled over {n_outer_desc})", labels=labels)
    viz.plot_fusion_weights(info, list(views), out / "fusion_weights.png", f"{task} - how late fusion weights each modality")

    ci = None
    if pooled["index"].is_unique:  # bootstrap only valid when each sample is predicted once
        ci = bootstrap_ci(pooled.y.values, pooled.p.values, 0.5, "roc_auc")

    cols = ["family", "model", "roc_auc", "roc_auc_std", "balanced_accuracy", "balanced_accuracy_std",
            "sensitivity", "specificity", "f1", "accuracy"]
    return summary, best, pooled, {
        "best_model": best["model"],
        "best_model_metrics": best.drop(["model", "family"]).to_dict(),
        "best_model_pooled_auc_95ci": ci,
        "best_model_pooled_confusion": cm,
        "best_per_family": best_per_family[cols].to_dict("records"),
        "fusion_gain_auc_vs_best_unimodal": float(
            best["roc_auc"] - summary[summary.family == "Unimodal"]["roc_auc"].max()),
        "fold_info": info,
    }
