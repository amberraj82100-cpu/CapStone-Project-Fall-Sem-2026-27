"""Flatten every experiment's saved results into four CSVs for the accuracy dashboard.

Run after the pipelines:  python ml/src/export_dashboard.py   -> ml/results/dashboard/*.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from common import ML_DIR, MODELS_DIR, RESULTS_DIR, ensure_dir
from model_selection import TOLERANCE, choose

STATE_DIR = Path(os.environ.get("ML_STATE_DIR", ML_DIR))  # where game retraining writes (see ml/service)
INPUT_TEXT = {
    "handwriting_image": "The handwriting image from the copy-the-sentence task",
    "number_round_trials": "Trial logs from the number-sense round",
    "game_features": "This game's own answers, response times and motor data (plus handwriting and number round when present)",
    None: "Rule-based scoring of the game's questions (no trained model can use this game's data yet)",
}

EXPERIMENTS = [  # (results folder, label, data kind, disorder)
    ("dyslexia", "Dyslexia", "real", "Dyslexia"),
    ("dysgraphia", "Dysgraphia", "real", "Dysgraphia"),
    ("dyscalculia_castaldi2024", "Dyscalculia · Castaldi 2024", "real", "Dyscalculia"),
    ("dyscalculia_synthetic", "Dyscalculia · synthetic", "synthetic", "Dyscalculia"),
]
OUT = ensure_dir(RESULTS_DIR / "dashboard")


def experiments() -> list[tuple[str, str, str, str, Path]]:
    """Research experiments plus any game-data experiments written by retraining."""
    out = [(f, l, k, d, RESULTS_DIR) for f, l, k, d in EXPERIMENTS]
    for dis in ["dyslexia", "dysgraphia", "dyscalculia"]:
        if (STATE_DIR / "results" / f"game_{dis}" / "summary.json").exists():
            out.append((f"game_{dis}", f"{dis.title()} · game data", "game", dis.title(), STATE_DIR / "results"))
    return out


def build_tables() -> dict[str, pd.DataFrame]:
    """All dashboard tables, keyed by name (also served live by ml/service/app.py)."""
    exp_rows, model_rows, mod_rows, feat_rows = [], [], [], []
    for folder, label, kind, disorder, base in experiments():
        d = base / folder
        if not (d / "summary.json").exists():
            print("skip", folder)
            continue
        s = json.loads((d / "summary.json").read_text())
        summ = pd.read_csv(d / "metrics_summary.csv")
        b, cm = s["best_model_metrics"], s["best_model_pooled_confusion"]
        uni = summ[summ.family == "Unimodal"].iloc[0]
        ci = s.get("best_model_pooled_auc_95ci") or [None, None]
        exp_rows.append({
            "experiment": folder, "label": label, "disorder": disorder, "data_kind": kind,
            "n_samples": s["n_samples"], "n_positive": s["n_positive"],
            "majority_baseline_accuracy": round(s["majority_baseline_accuracy"], 4),
            "best_model": s["best_model"],
            "best_roc_auc": round(b["roc_auc"], 4), "best_roc_auc_std": round(b["roc_auc_std"], 4),
            "best_balanced_accuracy": round(b["balanced_accuracy"], 4),
            "best_balanced_accuracy_std": round(b["balanced_accuracy_std"], 4),
            "best_sensitivity": round(b["sensitivity"], 4), "best_specificity": round(b["specificity"], 4),
            "best_accuracy": round(b["accuracy"], 4), "best_f1": round(b["f1"], 4),
            "best_unimodal_model": uni.model, "best_unimodal_roc_auc": round(uni.roc_auc, 4),
            "fusion_gain_auc": round(s["fusion_gain_auc_vs_best_unimodal"], 4),
            "tp": cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
            "auc_ci_low": None if ci[0] is None else round(ci[0], 4),
            "auc_ci_high": None if ci[1] is None else round(ci[1], 4),
            "label_noise_auc_ceiling": round(s["label_noise_auc_ceiling"], 4) if s.get("label_noise_auc_ceiling") else None,
        })
        for _, r in summ.iterrows():
            parts = r.model.split(" | ")
            model_rows.append({
                "experiment": folder, "family": r.family,
                "model": r.model, "short_name": " · ".join(parts[1:]),
                "roc_auc": round(r.roc_auc, 4), "roc_auc_std": round(r.roc_auc_std, 4),
                "balanced_accuracy": round(r.balanced_accuracy, 4), "balanced_accuracy_std": round(r.balanced_accuracy_std, 4),
                "sensitivity": round(r.sensitivity, 4), "specificity": round(r.specificity, 4),
                "f1": round(r.f1, 4), "accuracy": round(r.accuracy, 4), "is_best": int(r.model == s["best_model"]),
            })
        xai = s.get("xai", {})
        if "modality_share" in xai:
            imp, kind_imp = xai["modality_share"], "share of |SHAP|"
        elif "grouped_permutation_auc_drop" not in xai:
            imp, kind_imp = {}, "not computed for game-data runs yet"
        else:
            imp, kind_imp = {k: v[0] for k, v in xai["grouped_permutation_auc_drop"].items()}, "AUC drop when shuffled"
        weights = pd.DataFrame([f["late_weights"] for f in s["fold_info"]]).mean()
        for m in s["views"]:
            mod_rows.append({"experiment": folder, "modality": m, "n_features": s["views"][m],
                             "importance": round(float(imp.get(m, 0)), 4), "importance_kind": kind_imp,
                             "late_fusion_weight": round(float(weights.get(m, 0)), 4)})
        top = xai.get("top_features") or xai.get("geometry_top_features") or {}
        for rank, (f, v) in enumerate(list(top.items())[:10], 1):
            feat_rows.append({"experiment": folder, "rank": rank, "feature": f, "mean_abs_shap": round(float(v), 4)})

    # best model of each family, and how it compares with the best single modality
    fam_rows, sel_rows = [], []
    for folder, label, kind, disorder, base in experiments():
        f = base / folder / "metrics_summary.csv"
        if not f.exists():
            continue
        summ = pd.read_csv(f)
        uni = summ[summ.family == "Unimodal"].roc_auc.max()
        for fam, g in summ.groupby("family", sort=False):
            b = g.sort_values("roc_auc", ascending=False).iloc[0]
            fam_rows.append({"experiment": folder, "family": fam, "n_models": len(g), "best_model": b.model,
                             "best_short": " · ".join(b.model.split(" | ")[1:]),
                             "roc_auc": round(b.roc_auc, 4), "roc_auc_std": round(b.roc_auc_std, 4),
                             "balanced_accuracy": round(b.balanced_accuracy, 4),
                             "sensitivity": round(b.sensitivity, 4), "specificity": round(b.specificity, 4),
                             "gain_vs_best_unimodal": round(b.roc_auc - uni, 4)})
        c = choose(folder, base)
        sel_rows.append({"experiment": folder, "chosen_model": c["chosen_model"], "chosen_family": c["chosen_family"],
                         "chosen_roc_auc": round(c["chosen_roc_auc"], 4),
                         "chosen_balanced_accuracy": round(c["chosen_balanced_accuracy"], 4),
                         "top_model": c["top_model"], "top_roc_auc": round(c["top_roc_auc"], 4),
                         "n_within_tolerance": c["n_within_tolerance"], "tolerance": TOLERANCE, "reason": c["reason"]})
    dep_rows = deployment_rows()
    return {"experiments": pd.DataFrame(exp_rows), "model_metrics": pd.DataFrame(model_rows),
            "modality_importance": pd.DataFrame(mod_rows), "top_features": pd.DataFrame(feat_rows),
            "family_summary": pd.DataFrame(fam_rows), "selection": pd.DataFrame(sel_rows),
            "deployment": pd.DataFrame(dep_rows)}


def deployment_rows() -> list[dict]:
    """What the game uses right now: the registry's active version per disorder, or the rules."""
    card = json.loads((MODELS_DIR / "screening_system_card.json").read_text())
    reg_path = STATE_DIR / "models" / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    rows = []
    for d in ["DYSLEXIA", "DYSGRAPHIA", "DYSCALCULIA"]:
        entry = reg.get(d, {})
        v = next((x for x in entry.get("versions", []) if x["id"] == entry.get("active")), None)
        if v is None:
            rows.append({"disorder": d.title(), "experiment": None, "model": "Rule-based scoring", "version": None,
                         "trained_on": "rules", "roc_auc": None, "balanced_accuracy": None, "sensitivity": None,
                         "specificity": None, "moderate_from": None, "high_from": None, "game_inputs": INPUT_TEXT[None],
                         "reason": "No trained model can read this game's data yet. A game-native model is trained "
                                   "automatically once enough labelled sessions exist.",
                         "versions": len(entry.get("versions", []))})
            continue
        m = v.get("metrics", {})
        reason = v.get("selection_reason") or card.get(d, {}).get("selection_reason", "")
        rows.append({"disorder": d.title(), "experiment": f"game_{d.lower()}" if v["kind"] == "game" else None,
                     "model": v["model"], "version": v["id"],
                     "trained_on": "synthetic" if v.get("synthetic") else ("game" if v["kind"] == "game" else "real"),
                     "roc_auc": round(m.get("roc_auc", 0), 4), "balanced_accuracy": round(m.get("balanced_accuracy", 0), 4),
                     "sensitivity": round(m.get("sensitivity", 0), 4), "specificity": round(m.get("specificity", 0), 4),
                     "moderate_from": round(v["moderate_threshold"], 3), "high_from": round(v["high_threshold"], 3),
                     "game_inputs": INPUT_TEXT.get(v["input"], v["input"]), "reason": reason,
                     "versions": len(entry.get("versions", []))})
    return rows


def main():
    tables = build_tables()
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    print("wrote", OUT, {k: len(v) for k, v in tables.items()})


if __name__ == "__main__":
    main()
